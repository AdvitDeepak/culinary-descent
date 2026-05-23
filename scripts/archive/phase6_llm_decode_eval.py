#!/usr/bin/env python3
"""
Phase 6: LLM-based decoder + re-evaluation.

Instead of generic templates, use Qwen2.5-3B to generate recipe steps
from the DAG's structured content (process types, args, ingredients).

This is the minimal fix: one function change (decode_dag_llm vs decode_dag_template).
Everything else stays the same.

Outputs:
  data/eval/llm_decode_results.json   — ROUGE-L, SentBERT, LLM judge
  data/eval/llm_decode_examples.txt   — 20 side-by-side examples
"""

import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

DAGS_DIR  = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/dags")
RECIPE1M  = Path("/home/addeepak/AdvitResearch/cs348k/recipes.json")
OUT       = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/eval")
OUT.mkdir(parents=True, exist_ok=True)

EVAL_N    = 500    # recipes to eval (full ROUGE-L + SentBERT)
JUDGE_N   = 75     # per condition for LLM judge
SEED      = 42
random.seed(SEED); np.random.seed(SEED)

# ── DAG → structured prompt ───────────────────────────────────────────────────

def dag_to_prompt(dag: dict) -> str:
    """Build a compact structured prompt from a DAG for the LLM decoder."""
    proc_nodes = sorted(
        [n for n in dag["nodes"] if n["type"] == "process"],
        key=lambda n: n["step_idx"]
    )
    ings = [
        n["name"].split(",")[0].strip()
        for n in dag["nodes"] if n["type"] == "ingredient"
    ]

    steps = []
    for i, proc in enumerate(proc_nodes, 1):
        args = proc.get("args", {})
        step = proc["canonical"]
        if "temp_c" in args:
            f = round(args["temp_c"] * 9/5 + 32)
            step += f" at {f}°F"
        if "duration_min" in args:
            m = args["duration_min"]
            if m >= 60:
                h = int(m // 60); rem = int(m % 60)
                step += f" for {h}h {rem}m" if rem else f" for {h}hr"
            else:
                step += f" for {int(m)} min"
        steps.append(f"{i}. {step}")

    ing_str  = ", ".join(ings[:12])
    step_str = "\n".join(steps)

    return (
        f"Convert this recipe outline into plain cooking instructions. "
        f"No markdown, no headers, no preamble. Start immediately with step 1.\n\n"
        f"Ingredients: {ing_str}\n\n"
        f"Steps:\n{step_str}\n\n"
        f"Instructions ({len(proc_nodes)} steps, plain prose, no lists or headers):\n1."
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


JUDGE_PROMPT = (
    "Are these two recipe descriptions semantically equivalent — same ingredients, "
    "same cooking methods, same rough order?\n\n"
    "Recipe A:\n{a}\n\nRecipe B:\n{b}\n\n"
    "Answer YES or NO only."
)


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading DAGs + recipes...", flush=True)
    dags = []
    with open(DAGS_DIR / "train_dags.jsonl") as f:
        for line in f:
            dags.append(json.loads(line))

    with open(RECIPE1M) as f:
        all_recipes = json.load(f)
    recipe_by_id = {r["id"]: r for r in all_recipes if r.get("partition") == "train"}

    sample = random.sample(dags, EVAL_N + JUDGE_N * 4)
    eval_dags  = sample[:EVAL_N]
    judge_pool = sample[EVAL_N:]
    print(f"  {len(dags):,} total DAGs; eval={EVAL_N}, judge pool={len(judge_pool)}")

    # ── Load vLLM once for both decoder + judge ────────────────────────────────
    print("\nLoading Qwen2.5-3B via vLLM...", flush=True)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen2.5-3B-Instruct",
        max_model_len=2048,
        gpu_memory_utilization=0.70,   # leave ~30% for SentenceBERT
        dtype="bfloat16",
        enforce_eager=True,
    )
    tok = llm.get_tokenizer()

    def chat(messages: list[dict], max_tokens: int = 300, temp: float = 0.3) -> list[str]:
        prompts = [
            tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        params  = SamplingParams(temperature=temp, max_tokens=max_tokens)
        outputs = llm.generate(prompts, params)
        return [o.outputs[0].text.strip() for o in outputs]

    # ── Step 1: Decode all eval DAGs ──────────────────────────────────────────
    print(f"\n[1/3] Decoding {EVAL_N} DAGs with LLM...", flush=True)

    batch_size = 50
    decoded_texts = []

    for i in tqdm(range(0, len(eval_dags), batch_size), desc="decode"):
        batch = eval_dags[i:i+batch_size]
        msgs  = [
            [
                {"role": "system", "content": "You write recipe instructions as plain numbered prose. No markdown, no headers, no preamble like 'Certainly' or 'Sure'. Start directly with the first step."},
                {"role": "user",   "content": dag_to_prompt(dag)},
            ]
            for dag in batch
        ]
        raw = chat(msgs, max_tokens=300, temp=0.3)
        # prompt ends with "1." so prepend it back, then strip any leading preamble
        decoded_texts.extend("1. " + t for t in raw)

    # ── Step 2: ROUGE-L evaluation ────────────────────────────────────────────
    print("\n[2/3] Computing ROUGE-L...", flush=True)
    rouge_scores = []
    orig_texts   = []

    for dag, decoded in zip(eval_dags, decoded_texts):
        recipe = recipe_by_id.get(dag["id"])
        if not recipe: continue
        steps  = [s["text"].strip() for s in recipe.get("instructions", []) if s["text"].strip()]
        orig   = " ".join(steps)
        orig_texts.append(orig)
        rouge_scores.append(rouge_l(decoded, orig))

    rouge_arr = np.array(rouge_scores)
    print(f"  ROUGE-L mean:   {rouge_arr.mean():.4f}  (was 0.167 with templates)")
    print(f"  ROUGE-L median: {np.median(rouge_arr):.4f}")
    print(f"  % > 0.20: {(rouge_arr > 0.20).mean():.1%}")
    print(f"  % > 0.30: {(rouge_arr > 0.30).mean():.1%}")

    # ── Step 3: SentenceBERT ──────────────────────────────────────────────────
    print("\n[2/3] SentenceBERT cosine similarity...", flush=True)
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    ea = sbert.encode(orig_texts,    batch_size=256, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)
    eb = sbert.encode(decoded_texts[:len(orig_texts)], batch_size=256,
                      normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    sims = (ea * eb).sum(axis=1)

    dec_shuf = decoded_texts[:len(orig_texts)].copy(); random.shuffle(dec_shuf)
    eb2 = sbert.encode(dec_shuf, batch_size=256, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
    null_sims = (ea * eb2).sum(axis=1)

    print(f"  SentBERT mean:  {sims.mean():.3f}  (was 0.612 with templates)")
    print(f"  Null baseline:  {null_sims.mean():.3f}")
    print(f"  Gap:            {sims.mean()-null_sims.mean():+.3f}  (was +0.206)")

    del sbert  # free GPU for judge

    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

    # ── Step 4: LLM judge ─────────────────────────────────────────────────────
    print(f"\n[3/3] LLM judge ({JUDGE_N} pairs/condition)...", flush=True)

    tp_dags  = judge_pool[:JUDGE_N]
    neg_pool = [decode_dag_simple(d) for d in judge_pool[JUDGE_N:JUDGE_N*2+1]]

    # Decode true-positive DAGs with LLM
    tp_msgs = [
        [{"role":"system","content":"You write recipe instructions as plain numbered prose. No markdown, no headers, no preamble like 'Certainly' or 'Sure'. Start directly with the first step."},
         {"role":"user","content":dag_to_prompt(d)}]
        for d in tp_dags
    ]
    tp_decoded = ["1. " + t for t in chat(tp_msgs, max_tokens=300, temp=0.3)]

    judge_pairs = []
    for i, (dag, dec) in enumerate(zip(tp_dags, tp_decoded)):
        recipe = recipe_by_id.get(dag["id"])
        if not recipe: continue
        steps = [s["text"].strip() for s in recipe.get("instructions",[]) if s["text"].strip()]
        orig  = " ".join(steps[:10])
        judge_pairs.append(("true_pos",     orig, dec))
        judge_pairs.append(("easy_neg",     orig, neg_pool[i % len(neg_pool)]))
        judge_pairs.append(("shuffled_neg", orig, decode_dag_simple(dag, reverse=True)))

    judge_msgs = [
        [{"role":"system","content":"Answer with a single word: YES or NO."},
         {"role":"user","content":JUDGE_PROMPT.format(a=a[:600], b=b[:600])}]
        for _, a, b in judge_pairs
    ]
    judge_answers = chat(judge_msgs, max_tokens=4, temp=0.0)

    by_label = defaultdict(list)
    for (label, _, _), ans in zip(judge_pairs, judge_answers):
        by_label[label].append("YES" if ans.upper().startswith("Y") else "NO")

    tp_yes  = by_label["true_pos"].count("YES")    / max(len(by_label["true_pos"]), 1)
    easy_no = by_label["easy_neg"].count("NO")     / max(len(by_label["easy_neg"]), 1)
    shuf_no = by_label["shuffled_neg"].count("NO") / max(len(by_label["shuffled_neg"]), 1)

    print(f"\n{'='*60}")
    print("LLM DECODER + JUDGE RESULTS")
    print(f"{'='*60}")
    print(f"  ROUGE-L mean:              {rouge_arr.mean():.4f}  (templates: 0.167)")
    print(f"  SentBERT mean:             {sims.mean():.3f}  (templates: 0.612, null: {null_sims.mean():.3f})")
    print(f"  SentBERT gap:              {sims.mean()-null_sims.mean():+.3f}  (templates: +0.206)")
    print(f"  LLM judge TP YES rate:     {tp_yes:.1%}  (templates: 0%)")
    print(f"  LLM judge easy-neg NO:     {easy_no:.1%}")
    print(f"  LLM judge shuffled NO:     {shuf_no:.1%}")

    # ── Save examples ─────────────────────────────────────────────────────────
    with open(OUT / "llm_decode_examples.txt", "w") as f:
        for i, (dag, decoded) in enumerate(zip(eval_dags[:20], decoded_texts[:20])):
            recipe = recipe_by_id.get(dag["id"])
            if not recipe: continue
            steps = [s["text"].strip() for s in recipe.get("instructions",[]) if s["text"].strip()]
            orig  = " ".join(steps)
            rl    = rouge_l(decoded, orig)
            f.write(f"{'='*70}\nExample {i+1}: {recipe.get('title','')}\nROUGE-L={rl:.3f}\n\n")
            f.write(f"ORIGINAL:\n{orig[:500]}\n\nLLM DECODED:\n{decoded[:500]}\n\n")

    with open(OUT / "llm_decode_results.json", "w") as f:
        json.dump({
            "decoder": "Qwen2.5-3B-Instruct",
            "n_eval": len(rouge_scores),
            "rouge_l": {"mean": float(rouge_arr.mean()), "median": float(np.median(rouge_arr)),
                        "pct_gt_020": float((rouge_arr > 0.20).mean()),
                        "pct_gt_030": float((rouge_arr > 0.30).mean())},
            "sbert":   {"mean": float(sims.mean()), "null_mean": float(null_sims.mean()),
                        "gap":  float(sims.mean() - null_sims.mean())},
            "judge":   {"true_pos_yes": float(tp_yes), "easy_neg_no": float(easy_no),
                        "shuffled_neg_no": float(shuf_no)},
            "template_baseline": {"rouge_l": 0.167, "sbert": 0.612, "judge_tp_yes": 0.0},
        }, f, indent=2)

    print(f"\n  Saved → {OUT}/llm_decode_results.json")
    print(f"  Saved → {OUT}/llm_decode_examples.txt")


def decode_dag_simple(dag: dict, reverse: bool = False) -> str:
    """Fallback template decoder for negative examples."""
    proc_nodes = sorted(
        [n for n in dag["nodes"] if n["type"] == "process"],
        key=lambda n: n["step_idx"], reverse=reverse
    )
    node_map = {n["id"]: n for n in dag["nodes"]}
    inputs   = defaultdict(list)
    for e in dag["edges"]:
        if e["label"] in ("input","input_fallback"):
            src = node_map.get(e["src"])
            if src and src["type"]=="ingredient":
                inputs[e["dst"]].append(src["name"].split(",")[0].strip())
    parts = []
    for proc in proc_nodes:
        ings    = inputs.get(proc["id"], ["ingredients"])
        ing_str = ", ".join(ings[:3])
        parts.append(f"{proc['canonical'].capitalize()} {ing_str}.")
    return " ".join(parts)


if __name__ == "__main__":
    main()
