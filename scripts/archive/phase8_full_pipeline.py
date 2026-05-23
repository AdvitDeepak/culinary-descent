#!/usr/bin/env python3
"""
Phase 8: Full constrained-encoder pipeline.

  NL recipe
    → constrained LLM encoder  (guided JSON, vocab-locked to 43 types)
    → guided DAG steps          (canonical, ingredients, temp_f, duration_min)
    → LLM decoder               (Qwen2.5-3B, free text)
    → reconstructed NL
    → SentBERT / ROUGE-L / LLM judge

Compares against BGE-encoder + LLM-decoder baseline (phase6 numbers).

The key hypothesis: because the constrained encoder assigns ingredients to steps
at 97% (vs BGE's 34%), the decoder prompt is richer and reconstruction quality
should be higher.

Outputs:
  data/eval/phase8_full_pipeline.json    — all metrics + baseline comparison
  data/eval/phase8_examples.txt          — 20 side-by-side examples
"""

import gc
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

RECIPE1M  = Path("/home/addeepak/AdvitResearch/cs348k/recipes.json")
DAGS_DIR  = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/dags")
OUT       = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/eval")
OUT.mkdir(parents=True, exist_ok=True)

EVAL_N  = 300   # recipes for SentBERT / ROUGE-L
JUDGE_N = 50    # pairs per judge condition
SEED    = 42
random.seed(SEED); np.random.seed(SEED)

# ── Vocabulary ────────────────────────────────────────────────────────────────

CANONICAL_43 = [
    "bake","roast","broil","grill","toast",
    "saute","fry","sear","brown",
    "boil","simmer","steam","poach","braise","blanch",
    "chop","dice","slice","mince","grate","peel","crush",
    "mix","stir","whisk","fold","blend","beat","knead","toss",
    "season","coat","brush","drizzle",
    "cool","chill","freeze",
    "rest","marinate",
    "reduce","dissolve","melt","drain",
]

DAG_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "canonical":    {"type": "string", "enum": CANONICAL_43},
                    "ingredients":  {"type": "array", "items": {"type": "string"}, "maxItems": 6},
                    "temp_f":       {"anyOf": [{"type": "number"}, {"type": "null"}]},
                    "duration_min": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                },
                "required": ["canonical", "ingredients", "temp_f", "duration_min"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
})


# ── Prompts ───────────────────────────────────────────────────────────────────

def recipe_to_encode_prompt(recipe: dict) -> str:
    steps = [s["text"].strip() for s in recipe.get("instructions", []) if s["text"].strip()]
    ings  = [i.get("text", "").split(",")[0].strip() for i in recipe.get("ingredients", [])]
    inst  = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    return (
        f"Parse these cooking instructions into structured steps.\n\n"
        f"Ingredients available: {', '.join(ings[:12])}\n\n"
        f"Instructions:\n{inst}\n\n"
        f"For each cooking action: pick the canonical verb, list the ingredients used "
        f"in that step, and extract temperature (°F) and duration (minutes) if mentioned."
    )


def guided_steps_to_decode_prompt(steps: list[dict], all_ings: list[str]) -> str:
    """Build decoder prompt from constrained-encoder output."""
    lines = []
    for i, s in enumerate(steps, 1):
        verb = s["canonical"]
        ings = s.get("ingredients") or []
        temp = s.get("temp_f")
        dur  = s.get("duration_min")
        line = f"{i}. {verb}"
        if ings:
            line += f" [{', '.join(ings[:4])}]"
        if temp is not None:
            line += f" at {int(temp)}°F"
        if dur is not None:
            if dur >= 60:
                h = int(dur // 60); rem = int(dur % 60)
                line += f" for {h}h {rem}m" if rem else f" for {h}hr"
            else:
                line += f" for {int(dur)} min"
        lines.append(line)

    step_str = "\n".join(lines)
    ing_str  = ", ".join(all_ings[:12])

    return (
        f"Convert this recipe outline into plain cooking instructions. "
        f"No markdown, no headers, no preamble. Start immediately with step 1.\n\n"
        f"Ingredients: {ing_str}\n\n"
        f"Steps:\n{step_str}\n\n"
        f"Instructions ({len(steps)} steps, plain prose, no lists or headers):\n1."
    )


JUDGE_PROMPT = (
    "Are these two recipe descriptions semantically equivalent — same ingredients, "
    "same cooking methods, same rough order?\n\n"
    "Recipe A:\n{a}\n\nRecipe B:\n{b}\n\n"
    "Answer YES or NO only."
)


# ── Metrics ───────────────────────────────────────────────────────────────────

def rouge_l(hyp: str, ref: str) -> float:
    h = re.findall(r'\w+', hyp.lower())
    r = re.findall(r'\w+', ref.lower())
    if not h or not r: return 0.0
    m, n = len(h), len(r)
    if m < n: h, r, m, n = r, h, n, m
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j-1]+1 if h[i-1]==r[j-1] else max(curr[j-1], prev[j])
        prev = curr
    lcs = prev[n]
    p, r_ = lcs/m, lcs/n
    return 2*p*r_/(p+r_) if p+r_ else 0.0


def process_lcs_f1(a: list, b: list) -> float:
    m, n = len(a), len(b)
    if not m or not n: return 0.0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p, r = lcs/m, lcs/n
    return 2*p*r/(p+r) if p+r else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading Recipe1M + BGE DAGs...", flush=True)
    with open(RECIPE1M) as f:
        all_recipes = json.load(f)
    recipe_by_id = {r["id"]: r for r in all_recipes if r.get("partition") == "train"}

    bge_dags = {}
    with open(DAGS_DIR / "train_dags.jsonl") as f:
        for line in f:
            d = json.loads(line)
            bge_dags[d["id"]] = d

    candidates = [r for r in all_recipes
                  if r.get("partition") == "train" and r["id"] in bge_dags]
    sample = random.sample(candidates, EVAL_N + JUDGE_N * 4)
    eval_recipes  = sample[:EVAL_N]
    judge_pool    = sample[EVAL_N:]
    print(f"  {len(candidates):,} candidates; eval={EVAL_N}, judge pool={len(judge_pool)}")

    # ── Load vLLM ─────────────────────────────────────────────────────────────
    print("\nLoading Qwen2.5-3B via vLLM...", flush=True)
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(
        model="Qwen/Qwen2.5-3B-Instruct",
        max_model_len=2048,
        gpu_memory_utilization=0.80,
        dtype="bfloat16",
        enforce_eager=True,
    )
    tok = llm.get_tokenizer()

    encode_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        structured_outputs=StructuredOutputsParams(json=DAG_SCHEMA),
    )
    decode_params = SamplingParams(temperature=0.3, max_tokens=300)
    judge_params  = SamplingParams(temperature=0.0, max_tokens=4)

    def batch_generate(messages_list, params, desc="generate"):
        prompts = [
            tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]
        outputs = []
        batch_size = 30
        for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
            outs = llm.generate(prompts[i:i+batch_size], params)
            outputs.extend(o.outputs[0].text.strip() for o in outs)
        return outputs

    # ── Step 1: Constrained encode ─────────────────────────────────────────────
    print(f"\n[1/4] Constrained encoding {EVAL_N} recipes...", flush=True)

    encode_msgs = [
        [
            {"role": "system", "content":
             "You parse recipe instructions into structured JSON. "
             "Only use the cooking verbs from the enum in the schema."},
            {"role": "user", "content": recipe_to_encode_prompt(r)},
        ]
        for r in eval_recipes
    ]
    raw_encodes = batch_generate(encode_msgs, encode_params, "encode")

    encoded = []   # (recipe, steps | None)
    n_fail  = 0
    for recipe, raw in zip(eval_recipes, raw_encodes):
        try:
            parsed = json.loads(raw)
            encoded.append((recipe, parsed["steps"]))
        except Exception:
            encoded.append((recipe, None))
            n_fail += 1

    parse_rate = 1.0 - n_fail / EVAL_N
    print(f"  Encode parse rate: {parse_rate:.1%}  ({EVAL_N - n_fail}/{EVAL_N})")

    # ── Step 2: Decode encoded steps ───────────────────────────────────────────
    print(f"\n[2/4] Decoding constrained DAGs...", flush=True)

    valid_pairs = [(r, s) for r, s in encoded if s is not None]
    decode_msgs = [
        [
            {"role": "system", "content":
             "You write recipe instructions as plain numbered prose. "
             "No markdown, no headers, no preamble. Start directly with the first step."},
            {"role": "user", "content": guided_steps_to_decode_prompt(
                steps,
                [i.get("text","").split(",")[0].strip() for i in r.get("ingredients",[])],
            )},
        ]
        for r, steps in valid_pairs
    ]
    raw_decoded = batch_generate(decode_msgs, decode_params, "decode")
    decoded_texts = ["1. " + t for t in raw_decoded]

    # ── Step 3: Metrics ────────────────────────────────────────────────────────
    print(f"\n[3/4] Computing ROUGE-L + SentBERT...", flush=True)

    orig_texts   = []
    rouge_scores = []
    lcs_scores   = []

    for (recipe, steps), decoded in zip(valid_pairs, decoded_texts):
        instr = [s["text"].strip() for s in recipe.get("instructions",[]) if s["text"].strip()]
        orig  = " ".join(instr)
        orig_texts.append(orig)
        rouge_scores.append(rouge_l(decoded, orig))

        # process LCS vs BGE encoder
        bge_dag = bge_dags.get(recipe["id"])
        if bge_dag:
            bge_seq = [n["canonical"] for n in sorted(
                [n for n in bge_dag["nodes"] if n["type"] == "process"],
                key=lambda n: n["step_idx"]
            )]
            llm_seq = [s["canonical"] for s in steps]
            lcs_scores.append(process_lcs_f1(llm_seq, bge_seq))

    rouge_arr = np.array(rouge_scores)
    lcs_arr   = np.array(lcs_scores) if lcs_scores else np.array([0.0])

    print(f"  ROUGE-L: {rouge_arr.mean():.4f}  (BGE+LLM decoder baseline: 0.189)")

    # SentBERT on CPU — keeps vLLM alive (del llm kills Python ref but not EngineCore subprocess)
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    ea = sbert.encode(orig_texts, batch_size=256, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)
    eb = sbert.encode(decoded_texts, batch_size=256, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)
    sims = (ea * eb).sum(axis=1)

    dec_shuf = decoded_texts.copy(); random.shuffle(dec_shuf)
    eb2 = sbert.encode(dec_shuf, batch_size=256, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
    null_sims = (ea * eb2).sum(axis=1)

    print(f"  SentBERT: {sims.mean():.3f}  null={null_sims.mean():.3f}  "
          f"gap={sims.mean()-null_sims.mean():+.3f}  (BGE+LLM: 0.619, gap +0.222)")

    del sbert

    # ── Step 4: LLM judge ─────────────────────────────────────────────────────
    print(f"\n[4/4] LLM judge ({JUDGE_N} pairs/condition)...", flush=True)

    # encode judge pool with constrained encoder
    judge_encode_msgs = [
        [
            {"role": "system", "content":
             "You parse recipe instructions into structured JSON. "
             "Only use the cooking verbs from the enum in the schema."},
            {"role": "user", "content": recipe_to_encode_prompt(r)},
        ]
        for r in judge_pool[:JUDGE_N]
    ]
    raw_judge_enc = batch_generate(judge_encode_msgs, encode_params, "judge-encode")

    tp_decoded_texts = []
    tp_recipes       = []
    for recipe, raw in zip(judge_pool[:JUDGE_N], raw_judge_enc):
        try:
            steps = json.loads(raw)["steps"]
        except Exception:
            continue
        all_ings = [i.get("text","").split(",")[0].strip() for i in recipe.get("ingredients",[])]
        tp_recipes.append(recipe)
        decode_msg = [
            [{"role":"system","content":
              "You write recipe instructions as plain numbered prose. "
              "No markdown, no headers, no preamble. Start directly with the first step."},
             {"role":"user","content":guided_steps_to_decode_prompt(steps, all_ings)}]
        ]
        raw_dec = batch_generate(decode_msg, decode_params, "")
        tp_decoded_texts.append("1. " + raw_dec[0])

    # negative pool: template-decoded BGE DAGs for different recipes
    neg_pool_texts = []
    for r in judge_pool[JUDGE_N:JUDGE_N*2]:
        bge_dag = bge_dags.get(r["id"])
        if not bge_dag: continue
        proc_nodes = sorted(
            [n for n in bge_dag["nodes"] if n["type"] == "process"],
            key=lambda n: n["step_idx"]
        )
        parts = [f"- {p['canonical']}" for p in proc_nodes[:8]]
        neg_pool_texts.append("\n".join(parts))

    judge_pairs = []
    for i, (recipe, dec) in enumerate(zip(tp_recipes, tp_decoded_texts)):
        instr = [s["text"].strip() for s in recipe.get("instructions",[]) if s["text"].strip()]
        orig  = " ".join(instr[:10])
        judge_pairs.append(("true_pos",     orig, dec))
        judge_pairs.append(("easy_neg",     orig, neg_pool_texts[i % max(len(neg_pool_texts),1)]))
        # shuffled: same recipe, reverse step order
        rev_dec = ". ".join(reversed(dec.split(". ")))
        judge_pairs.append(("shuffled_neg", orig, rev_dec))

    judge_msgs = [
        [{"role":"system","content":"Answer with a single word: YES or NO."},
         {"role":"user","content":JUDGE_PROMPT.format(a=a[:600], b=b[:600])}]
        for _, a, b in judge_pairs
    ]
    judge_answers = batch_generate(judge_msgs, judge_params, "judge")

    by_label = defaultdict(list)
    for (label, _, _), ans in zip(judge_pairs, judge_answers):
        by_label[label].append("YES" if ans.upper().startswith("Y") else "NO")

    tp_yes  = by_label["true_pos"].count("YES")    / max(len(by_label["true_pos"]), 1)
    easy_no = by_label["easy_neg"].count("NO")     / max(len(by_label["easy_neg"]), 1)
    shuf_no = by_label["shuffled_neg"].count("NO") / max(len(by_label["shuffled_neg"]), 1)

    # ── Print comparison table ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("FULL PIPELINE COMPARISON")
    print(f"{'='*65}")
    print(f"{'Metric':<35}  {'BGE+LLM':>9}  {'Constrained':>12}")
    print(f"{'-'*65}")
    print(f"{'Encoder parse rate':<35}  {'97.5%':>9}  {parse_rate:>11.1%}")
    print(f"{'ROUGE-L mean':<35}  {'0.189':>9}  {rouge_arr.mean():>12.4f}")
    print(f"{'SentBERT mean':<35}  {'0.619':>9}  {sims.mean():>12.3f}")
    print(f"{'SentBERT null baseline':<35}  {'0.398':>9}  {null_sims.mean():>12.3f}")
    print(f"{'SentBERT gap':<35}  {'+0.222':>9}  {sims.mean()-null_sims.mean():>+12.3f}")
    print(f"{'Process LCS (encoder agreement)':<35}  {'0.436':>9}  {lcs_arr.mean():>12.3f}")
    print(f"{'LLM judge TP YES rate':<35}  {'1.3%':>9}  {tp_yes:>11.1%}")
    print(f"{'LLM judge easy-neg NO rate':<35}  {'100%':>9}  {easy_no:>11.1%}")
    print(f"{'LLM judge shuffled NO rate':<35}  {'100%':>9}  {shuf_no:>11.1%}")
    print(f"{'='*65}")

    # ── Save ──────────────────────────────────────────────────────────────────
    results = {
        "encoder": "constrained-LLM (Qwen2.5-3B guided JSON)",
        "decoder": "Qwen2.5-3B-Instruct",
        "n_eval": len(valid_pairs),
        "encode_parse_rate": parse_rate,
        "rouge_l":  {"mean": float(rouge_arr.mean()), "median": float(np.median(rouge_arr))},
        "sbert":    {"mean": float(sims.mean()), "null_mean": float(null_sims.mean()),
                     "gap":  float(sims.mean() - null_sims.mean())},
        "process_lcs_vs_bge": float(lcs_arr.mean()),
        "judge":    {"true_pos_yes": float(tp_yes), "easy_neg_no": float(easy_no),
                     "shuffled_neg_no": float(shuf_no)},
        "bge_llm_baseline": {
            "encode_parse_rate": 0.975,
            "rouge_l": 0.189,
            "sbert": 0.619,
            "sbert_null": 0.398,
            "sbert_gap": 0.222,
            "judge_tp_yes": 0.013,
        },
    }

    with open(OUT / "phase8_full_pipeline.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(OUT / "phase8_examples.txt", "w") as f:
        for i, ((recipe, steps), decoded) in enumerate(zip(valid_pairs[:20], decoded_texts[:20])):
            instr = [s["text"].strip() for s in recipe.get("instructions",[]) if s["text"].strip()]
            orig  = " ".join(instr)
            rl    = rouge_l(decoded, orig)
            sb    = float((ea[i] * eb[i]).sum()) if i < len(ea) else 0.0
            f.write(f"{'='*70}\nExample {i+1}: {recipe.get('title','')}\n"
                    f"ROUGE-L={rl:.3f}  SentBERT={sb:.3f}\n\n")
            f.write(f"ORIGINAL:\n{orig[:500]}\n\n")
            f.write(f"ENCODED STEPS:\n")
            for s in (steps or []):
                ings = ", ".join(s.get("ingredients",[])[:3]) or "—"
                t    = f"{s['temp_f']}°F" if s.get("temp_f") else "—"
                d    = f"{s['duration_min']}min" if s.get("duration_min") else "—"
                f.write(f"  {s['canonical']:12s}  [{ings}]  temp={t}  dur={d}\n")
            f.write(f"\nDECODED:\n{decoded[:500]}\n\n")

    print(f"\n  Saved → {OUT}/phase8_full_pipeline.json")
    print(f"  Saved → {OUT}/phase8_examples.txt")


if __name__ == "__main__":
    main()
