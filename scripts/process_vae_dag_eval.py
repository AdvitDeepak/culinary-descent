#!/usr/bin/env python3
"""
Pairwise LLM-as-judge evaluation: learned VQ-VAE DAGs vs grammar-constrained 14B DAGs.

Two evaluation modes:

  1. JUDGE (--mode judge, default)
     For each recipe: show original text + both DAGs to Qwen2.5-3B-Instruct as judge.
     Randomly swap A/B order for half the recipes to control position bias.
     Reports win/tie/loss rates and example judgements.

  2. ROUGE (--mode rouge)
     Decode learned DAG through Stage 2 Qwen decoder → measure ROUGE-L vs original.
     Baseline: same Qwen but with no DAG prefix (ingredient list only).
     Reports ROUGE-L delta to quantify how much the learned DAG helps generation.

Usage:
  python3 scripts/process_vae_dag_eval.py                    # judge, n=200
  python3 scripts/process_vae_dag_eval.py --n 100            # fewer recipes
  python3 scripts/process_vae_dag_eval.py --mode rouge       # roundtrip ROUGE
  python3 scripts/process_vae_dag_eval.py --mode both        # both evals
"""

import argparse, json, os, re, sys, time
from pathlib import Path
from collections import defaultdict

import anthropic
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE       = Path(__file__).resolve().parent.parent
LAYER1     = Path(os.environ.get("RECIPE1M", str(BASE.parent / "layer1.json")))
STAGE1_DIR = BASE / "data/models/process_vae"
S2_DIR     = BASE / "data/models/process_vae_stage2"
DAG_LEARNED  = BASE / "data/dags/learned_dags.jsonl"
DAG_GRAMMAR  = BASE / "data/dags/constrained_dags_14b.jsonl"
DAG_GRAMMAR_DEFAULT_NAME = "grammar"
OUT_DIR    = BASE / "data/eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QWEN_MODEL   = "Qwen/Qwen2.5-3B-Instruct"
JUDGE_MODEL  = "claude-sonnet-4-6"   # used for judging only; Qwen used for ROUGE gen

sys.path.insert(0, str(BASE / "scripts"))

# Labels that indicate non-cooking narrative steps — filter these out entirely
NOISE_LABELS = {"serve", "instruction-note"}

# Labels we trust enough to show to the judge; everything else is suppressed when
# --clean_labels is active (wrong labels actively hurt the learned DAG's score).
# Expanding to 21 labels dropped the win rate from 51.9% → 22.5%:
# saute/pour/cool-rest/stir etc. are assigned with imprecise code boundaries;
# visible wrong labels are penalised harder than invisible ones. Reverted.
# These 11 are the only codes where VQ label assignment is reliably accurate.
CLEAR_LABELS = {
    "preheat-oven", "bake-timed", "knead", "whisk", "fold",
    "season", "simmer", "mix-general", "prepare-pan", "roll-dough", "marinate",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(n_sample=200, seed=42, learned_path=None, grammar_path=None):
    """Load n_sample recipes present in both DAG sets + layer1.json."""
    print("Loading DAGs and recipe text...", flush=True)
    dag_path = Path(learned_path) if learned_path else DAG_LEARNED
    g_path   = Path(grammar_path) if grammar_path else DAG_GRAMMAR

    learned = {}
    with open(dag_path) as f:
        for line in f:
            d = json.loads(line)
            learned[d["id"]] = d

    grammar = {}
    with open(g_path) as f:
        for line in f:
            d = json.loads(line)
            rid = d.get("id") or d.get("recipe_id")
            if rid in learned:
                grammar[rid] = d

    # Load original text
    print(f"  Loading layer1.json for recipe text...", flush=True)
    raw_by_id = {}
    with open(LAYER1) as f:
        for r in json.load(f):
            if r["id"] in grammar:
                raw_by_id[r["id"]] = r

    overlap = [r for r in grammar if r in raw_by_id and len(grammar[r]["steps"]) >= 3
               and learned[r]["n_steps"] >= 3]
    print(f"  {len(overlap):,} usable overlap recipes (≥3 steps each)", flush=True)

    rng = np.random.default_rng(seed)
    sample = [overlap[i] for i in rng.choice(len(overlap), min(n_sample, len(overlap)),
                                               replace=False)]
    return [(rid, learned[rid], grammar[rid], raw_by_id[rid]) for rid in sample]


# ── DAG formatting ─────────────────────────────────────────────────────────────

def load_label_map(label_dir=None):
    p = Path(label_dir) / "codebook_analysis.json" if label_dir else STAGE1_DIR / "codebook_analysis.json"
    if not p.exists():
        return {}
    return {int(k): v["semantic_label"] for k, v in json.load(open(p)).items()}


def fmt_learned_dag(ldag, label_map, grammar_predict_fn=None,
                    filter_noise=False, clean_labels=False):
    """
    Format learned DAG as readable text for the judge.

    filter_noise=True  — drop nodes whose VQ label is in NOISE_LABELS (serve,
                         instruction-note), which are non-cooking narrative steps.
    clean_labels=True  — only show VQ label when it's in CLEAR_LABELS; for all
                         other codes the label is omitted so a wrong label can't
                         override what the step text already says.
    grammar_predict_fn — Option A: augment each node with predicted grammar type.
    """
    # Identify noise nodes to skip (by their original step index t).
    # Guard: never filter if it would leave fewer than 3 visible steps — short recipes
    # like "mix, refrigerate, serve" legitimately end with a serve step.
    noise_t = set()
    if filter_noise:
        candidates = {node["t"] for node in ldag["nodes"]
                      if label_map.get(node["code"], "") in NOISE_LABELS}
        if len(ldag["nodes"]) - len(candidates) >= 3:
            noise_t = candidates

    lines = []
    step_num = 0
    for node in ldag["nodes"]:
        if node["t"] in noise_t:
            continue
        vq_lbl = label_map.get(node["code"], f"c{node['code']}")
        if grammar_predict_fn is not None:
            gram_lbl = grammar_predict_fn(node["code"])
            lbl = f"{gram_lbl} / {vq_lbl}" if gram_lbl else vq_lbl
        else:
            lbl = vq_lbl
        text = node["step_text"][:80].strip()
        ings = ", ".join(node["ingredients"][:4]) or "—"
        if clean_labels and lbl not in CLEAR_LABELS:
            lines.append(f"  Step {step_num}: \"{text}\"  [ings: {ings}]")
        else:
            lines.append(f"  Step {step_num} [{lbl}]: \"{text}\"  [ings: {ings}]")
        step_num += 1

    # Only show edges between non-filtered steps
    valid_edges = [e for e in ldag["edges"]
                   if e["from"] not in noise_t and e["to"] not in noise_t]
    if valid_edges:
        edge_str = ", ".join(f"{e['from']}→{e['to']}" for e in valid_edges[:8])
        lines.append(f"  Dependencies: {edge_str}")
    else:
        lines.append("  Dependencies: none (linear)")
    return "\n".join(lines), step_num  # return visible step count too


def fmt_grammar_dag(gdag):
    """Format grammar DAG as readable text for the judge.

    Handles three formats:
      standard/robotics  — canonical + ingredients (original 14B grammar, robotics 6-type)
      papadopoulos       — canonical + ingredients + tool (CVPR 2022 DSL)
      kyoto              — action_type + action_verb + inputs + output (LREC 2020 flow graph)
    """
    lines = []
    for i, step in enumerate(gdag["steps"]):
        extras = []
        if step.get("temp_f"):       extras.append(f"{step['temp_f']}°F")
        if step.get("duration_min"): extras.append(f"{step['duration_min']}min")

        if "action_type" in step:
            # Kyoto Flow Graph format
            typ    = f"{step['action_type']}: {step.get('action_verb', '?')}"
            inputs = ", ".join((step.get("inputs") or [])[:5]) or "—"
            output = step.get("output") or "—"
            extra_str = f"  [{', '.join(extras)}]" if extras else ""
            lines.append(f"  Step {i} [{typ}]{extra_str}  [inputs: {inputs}]  → {output}")
        else:
            # Standard (original / robotics) or Papadopoulos DSL
            typ  = step.get("canonical", "?")
            ings = ", ".join((step.get("ingredients") or [])[:4]) or "—"
            if step.get("tool"):
                extras = [f"tool={step['tool']}"] + extras
            extra_str = f"  [{', '.join(extras)}]" if extras else ""
            lines.append(f"  Step {i} [{typ}]{extra_str}  [ings: {ings}]")

    lines.append("  Dependencies: none (linear sequence)")
    return "\n".join(lines)


def fmt_recipe_text(raw):
    """Format original recipe instructions."""
    instrs = [s.get("text", "").strip() for s in raw.get("instructions", [])[:12]
              if s.get("text", "").strip()]
    return "\n".join(f"  {i+1}. {s}" for i, s in enumerate(instrs))


# ── LLM-as-judge evaluation ────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an expert chef evaluating recipe process representations. "
    "You give concise, accurate judgements."
)

JUDGE_PROMPT = """\
I will show you a recipe and two DAG (directed acyclic graph) representations of its \
cooking process. Each DAG shows operation type labels, ingredient assignments, and \
(if available) step dependencies.

RECIPE: {title}
Original instructions:
{instructions}

---
DAG A ({a_n} steps):
{a_dag}

---
DAG B ({b_n} steps):
{b_dag}

---
Which DAG better represents this recipe's cooking process?
Consider: (1) Are the operation types accurate? (2) Are ingredients at the right steps? \
(3) Is the structure plausible?

Start your response with "A" or "B", then give one sentence of explanation."""


def parse_judge_response(text):
    """Extract A or B from the start of the judge response."""
    text = text.strip()
    m = re.match(r"^\s*([AB])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: first standalone A or B
    m = re.search(r"\b([AB])\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else "?"


def run_judge_eval(records, n_recipes, label_map, grammar_predict_fn=None,
                   filter_noise=False, clean_labels=False):
    """
    Pairwise LLM-as-judge using Claude (Anthropic API).
    Half the recipes have A=learned/B=grammar, other half A=grammar/B=learned,
    to cancel out position bias.
    """
    client = anthropic.Anthropic()

    flags = []
    if filter_noise:   flags.append("filter_noise")
    if clean_labels:   flags.append("clean_labels")
    flag_str = f"  [{', '.join(flags)}]" if flags else ""
    print(f"\n{'='*60}")
    print(f"LLM-as-judge ({JUDGE_MODEL}): {n_recipes} recipes  (half swapped){flag_str}")
    print(f"{'='*60}\n")

    rng     = np.random.default_rng(0)
    swapped = set(rng.choice(n_recipes, n_recipes // 2, replace=False).tolist())

    results = []
    learned_wins = grammar_wins = parse_fails = 0

    for i, (rid, ldag, gdag, raw) in enumerate(records[:n_recipes]):
        title  = gdag.get("title", rid)
        instrs = fmt_recipe_text(raw)
        l_fmt, l_n = fmt_learned_dag(ldag, label_map, grammar_predict_fn,
                                     filter_noise=filter_noise,
                                     clean_labels=clean_labels)
        g_fmt  = fmt_grammar_dag(gdag)

        # Always label first position A, second B; swap content for half
        if i in swapped:
            a_fmt, b_fmt = g_fmt, l_fmt          # grammar=A, learned=B
            a_n,   b_n   = len(gdag["steps"]), l_n
            a_is_learned = False
        else:
            a_fmt, b_fmt = l_fmt, g_fmt          # learned=A, grammar=B
            a_n,   b_n   = l_n, len(gdag["steps"])
            a_is_learned = True

        prompt = JUDGE_PROMPT.format(
            title=title, instructions=instrs,
            a_dag=a_fmt, b_dag=b_fmt,
            a_n=a_n, b_n=b_n)

        resp = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=150,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        response = resp.content[0].text.strip()

        choice = parse_judge_response(response)
        if choice == "?":
            parse_fails += 1
            winner = "parse_fail"
        elif (choice == "A") == a_is_learned:
            learned_wins += 1
            winner = "learned"
        else:
            grammar_wins += 1
            winner = "grammar"

        results.append({
            "rid": rid, "title": title,
            "swapped": i in swapped,
            "a_is_learned": a_is_learned,
            "choice": choice, "winner": winner, "response": response[:300],
            "n_learned": l_n, "n_grammar": len(gdag["steps"]),
        })

        if (i + 1) % 20 == 0 or i < 5:
            print(f"  [{i+1:3d}/{n_recipes}]  {title[:38]:38s}  → {winner}  |  {response[:70]}")

    n_valid = learned_wins + grammar_wins
    print(f"\n{'─'*60}")
    print(f"RESULTS (n={n_recipes}, {parse_fails} parse failures excluded)")
    print(f"  Learned  wins: {learned_wins:3d}  ({learned_wins/max(n_valid,1):.1%})")
    print(f"  Grammar  wins: {grammar_wins:3d}  ({grammar_wins/max(n_valid,1):.1%})")
    print(f"  Parse failures: {parse_fails}")

    return results, {"learned_wins": learned_wins, "grammar_wins": grammar_wins,
                     "parse_fails": parse_fails, "n": n_recipes,
                     "learned_win_rate": round(learned_wins / max(n_valid, 1), 4)}


# ── ROUGE roundtrip evaluation ─────────────────────────────────────────────────

def rouge_l(hyp_tokens, ref_tokens):
    """Sentence-level ROUGE-L (LCS-based)."""
    if not hyp_tokens or not ref_tokens:
        return 0.0
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if ref_tokens[i-1] == hyp_tokens[j-1] \
                       else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs / n if n else 0
    r = lcs / m if m else 0
    return 2 * p * r / (p + r) if (p + r) else 0.0


SYSTEM_PROMPT = (
    "You are a culinary writer. Given a cooking process and ingredient list, "
    "write a complete recipe with title, ingredients, and step-by-step instructions."
)

def run_rouge_eval(records, n_recipes, tokenizer, qwen, code_emb, device, label_map):
    """Decode learned DAG through Stage 2 and measure ROUGE-L vs original."""
    from process_vae_stage2 import TOK_DIR
    import torch.nn as nn

    tok_ids     = json.load(open(TOK_DIR / "recipe_ids.json"))
    code_ids_np = np.load(TOK_DIR / "code_ids.npy",  mmap_mode="r")
    n_steps_np  = np.load(TOK_DIR / "n_steps.npy",   mmap_mode="r")
    tok_set     = {r: i for i, r in enumerate(tok_ids)}

    print(f"\n{'='*60}")
    print(f"Roundtrip ROUGE-L: {n_recipes} recipes")
    print(f"{'='*60}\n")

    scores_dag = []; scores_base = []
    skipped = 0

    for rid, ldag, gdag, raw in records[:n_recipes]:
        if rid not in tok_set:
            skipped += 1; continue

        idx   = tok_set[rid]
        T     = int(n_steps_np[idx])
        cids  = code_ids_np[idx, :T].astype(np.int64)
        ings  = [x.get("text","").split(",")[0].strip()
                 for x in raw.get("ingredients",[])[:12] if x.get("text","")]
        ing_text = ", ".join(ings) if ings else "various ingredients"
        ref_text = " ".join(s.get("text","") for s in raw.get("instructions",[]))
        ref_tok  = ref_text.lower().split()
        if not ref_tok:
            skipped += 1; continue

        def generate(prefix_emb, pfx_mask):
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Ingredients: {ing_text}"}]
            prompt   = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                      add_generation_prompt=True)
            enc      = tokenizer(prompt, return_tensors="pt").to(device)
            tok_emb  = qwen.get_input_embeddings()(enc["input_ids"])
            full_emb = torch.cat([prefix_emb, tok_emb], dim=1)
            full_msk = torch.cat([pfx_mask, enc["attention_mask"]], dim=1)
            with torch.no_grad():
                out = qwen.generate(inputs_embeds=full_emb, attention_mask=full_msk,
                                    max_new_tokens=256, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(out[0], skip_special_tokens=True)

        # DAG-conditioned generation
        cids_t   = torch.tensor(cids, dtype=torch.long, device=device).unsqueeze(0)
        pfx      = code_emb(cids_t).to(torch.bfloat16)
        pfx_mask = torch.ones(1, T, dtype=torch.long, device=device)
        gen_dag  = generate(pfx, pfx_mask)

        # Baseline: empty 1-token prefix (no process info)
        null_pfx  = torch.zeros(1, 1, code_emb.embedding_dim,
                                dtype=torch.bfloat16, device=device)
        null_mask = torch.ones(1, 1, dtype=torch.long, device=device)
        gen_base  = generate(null_pfx, null_mask)

        r_dag  = rouge_l(gen_dag.lower().split(),  ref_tok)
        r_base = rouge_l(gen_base.lower().split(), ref_tok)
        scores_dag.append(r_dag)
        scores_base.append(r_base)

    n_valid = len(scores_dag)
    print(f"\n{'─'*60}")
    print(f"ROUGE-L  (n={n_valid}, {skipped} skipped — not in tokenised set)")
    print(f"  DAG-conditioned:  {np.mean(scores_dag):.4f}  ± {np.std(scores_dag):.4f}")
    print(f"  Baseline (no DAG): {np.mean(scores_base):.4f}  ± {np.std(scores_base):.4f}")
    delta = np.mean(scores_dag) - np.mean(scores_base)
    print(f"  DAG lift:         {delta:+.4f}  ({'better' if delta > 0 else 'worse'})")

    return {"rouge_dag": round(float(np.mean(scores_dag)), 4),
            "rouge_base": round(float(np.mean(scores_base)), 4),
            "rouge_delta": round(float(delta), 4),
            "n": n_valid}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   choices=["judge", "rouge", "both"], default="judge")
    parser.add_argument("--n",      type=int, default=200,
                        help="Number of recipes to evaluate")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--learned_dags", type=str, default=None,
                        help="Path to learned DAGs jsonl (default: learned_dags.jsonl). "
                             "Use learned_dags_hybrid.jsonl for Option B.")
    parser.add_argument("--grammar_dags", type=str, default=None,
                        help="Path to alternative grammar DAGs jsonl "
                             "(default: constrained_dags_14b.jsonl). "
                             "Use alt_grammar_robotics.jsonl etc. for ablation.")
    parser.add_argument("--grammar_name", type=str, default=DAG_GRAMMAR_DEFAULT_NAME,
                        help="Display label for the grammar system (default: grammar)")
    parser.add_argument("--use_grammar_classifier", action="store_true",
                        help="Option A: augment VQ labels with predicted grammar types")
    parser.add_argument("--filter_noise", action="store_true",
                        help="Drop serve/instruction-note nodes before showing to judge")
    parser.add_argument("--clean_labels", action="store_true",
                        help="Suppress VQ labels not in CLEAR_LABELS; show step text only")
    parser.add_argument("--clean", action="store_true",
                        help="Shorthand for --filter_noise --clean_labels")
    parser.add_argument("--out_suffix", type=str, default="",
                        help="Suffix appended to output filename, e.g. '_clean'")
    parser.add_argument("--label_dir", type=str, default=None,
                        help="Dir containing codebook_analysis.json (default: process_vae/)")
    args = parser.parse_args()
    if args.clean:
        args.filter_noise = True
        args.clean_labels = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    records   = load_data(n_sample=args.n, seed=args.seed, learned_path=args.learned_dags,
                          grammar_path=args.grammar_dags)
    label_map = load_label_map(label_dir=args.label_dir)

    # Option A: grammar classifier
    grammar_predict_fn = None
    if args.use_grammar_classifier:
        print("Loading grammar classifier (Option A)...", flush=True)
        from process_vae_grammar_classifier import load_classifier
        _, _, grammar_predict_fn, _ = load_classifier()
        print("  Classifier loaded.", flush=True)

    all_results = {}

    if args.mode in ("judge", "both"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY not set. Run:  export ANTHROPIC_API_KEY=sk-ant-...")
            sys.exit(1)
        print(f"Grammar label: {args.grammar_name}", flush=True)
        results, stats = run_judge_eval(records, args.n, label_map, grammar_predict_fn,
                                        filter_noise=args.filter_noise,
                                        clean_labels=args.clean_labels)
        stats["grammar_name"] = args.grammar_name
        all_results["judge"] = stats
        out_name = f"dag_judge_results{args.out_suffix}.json"
        json.dump(results, open(OUT_DIR / out_name, "w"), indent=2)
        print(f"\nDetailed results → {OUT_DIR}/{out_name}")

    if args.mode in ("rouge", "both"):
        print(f"\nLoading {QWEN_MODEL} for ROUGE eval...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
        model     = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL, dtype=torch.bfloat16, trust_remote_code=True).to(device)
        model.eval()
        import torch.nn as nn
        from peft import PeftModel

        print(f"\nLoading Stage 2 LoRA for ROUGE eval...", flush=True)
        from peft import PeftModel as _PeftModel
        qwen = _PeftModel.from_pretrained(model, str(S2_DIR / "lora_best")).to(device)
        qwen.eval()

        ckpt     = torch.load(S2_DIR / "code_emb_best.pt", map_location=device,
                              weights_only=False)
        meta1    = ckpt["meta"]
        code_emb = nn.Embedding(meta1["n_codes"], 2048).to(device)
        code_emb.load_state_dict(ckpt["code_emb"])

        rouge_stats = run_rouge_eval(records, args.n, tokenizer, qwen, code_emb,
                                     device, label_map)
        all_results["rouge"] = rouge_stats

    json.dump(all_results, open(OUT_DIR / "dag_eval_summary.json", "w"), indent=2)
    print(f"\nSummary → {OUT_DIR}/dag_eval_summary.json")
    print("\n=== EVAL COMPLETE ===")
    for k, v in all_results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
