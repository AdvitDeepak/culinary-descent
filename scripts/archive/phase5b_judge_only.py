#!/usr/bin/env python3
"""
Phase 5b: Run just the LLM judge in isolation (no BGE/SBERT — clean GPU).

Uses enforce_eager=True to skip CUDA graph profiling (avoids memory assertion
error when other processes share the GPU). Compile cache from phase5 is reused.

Reads judge pairs from data/eval/judge_pairs.json (built here if not present).
Writes data/eval/judge_results.json + prints final numbers.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

DAGS_DIR  = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/dags")
RECIPE1M  = Path("/home/addeepak/AdvitResearch/cs348k/recipes.json")
OUT       = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/eval")
OUT.mkdir(parents=True, exist_ok=True)

JUDGE_N_EACH = 75
JUDGE_MODEL  = "Qwen/Qwen2.5-3B-Instruct"
random.seed(42)

TEMPLATES = {
    "bake":"Bake {i}.","roast":"Roast {i} in the oven.","broil":"Broil {i}.",
    "grill":"Grill {i}.","toast":"Toast {i}.","saute":"Sauté {i} in a pan.",
    "fry":"Fry {i}.","sear":"Sear {i}.","brown":"Brown {i}.",
    "boil":"Boil {i}.","simmer":"Simmer {i}.","steam":"Steam {i}.",
    "poach":"Poach {i}.","braise":"Braise {i}.","blanch":"Blanch {i}.",
    "chop":"Chop {i}.","dice":"Dice {i}.","slice":"Slice {i}.",
    "mince":"Mince {i}.","grate":"Grate {i}.","peel":"Peel {i}.",
    "crush":"Crush {i}.","mix":"Mix {i}.","stir":"Stir {i}.",
    "whisk":"Whisk {i}.","fold":"Fold in {i}.","blend":"Blend {i}.",
    "beat":"Beat {i}.","knead":"Knead {i}.","toss":"Toss {i}.",
    "season":"Season {i}.","coat":"Coat {i}.","brush":"Brush {i}.",
    "drizzle":"Drizzle {i}.","cool":"Cool {i}.","chill":"Refrigerate {i}.",
    "freeze":"Freeze {i}.","rest":"Rest {i}.","marinate":"Marinate {i}.",
    "reduce":"Reduce {i}.","dissolve":"Dissolve {i}.","melt":"Melt {i}.",
    "drain":"Drain {i}.",
}

def decode_dag(dag):
    node_map = {n["id"]: n for n in dag["nodes"]}
    proc_nodes = sorted([n for n in dag["nodes"] if n["type"]=="process"],
                        key=lambda n: n["step_idx"])
    inputs = defaultdict(list)
    for e in dag["edges"]:
        if e["label"] in ("input","input_fallback"):
            src = node_map.get(e["src"])
            if src and src["type"]=="ingredient":
                inputs[e["dst"]].append(src["name"].split(",")[0].strip())
    sentences = []
    for proc in proc_nodes:
        ings = inputs.get(proc["id"], [])
        ing_str = ", ".join(ings[:3]) if ings else "ingredients"
        sentences.append(TEMPLATES.get(proc["canonical"],"Process {i}.").format(i=ing_str))
    return " ".join(sentences)

def decode_dag_reversed(dag):
    node_map = {n["id"]: n for n in dag["nodes"]}
    proc_nodes = sorted([n for n in dag["nodes"] if n["type"]=="process"],
                        key=lambda n: n["step_idx"], reverse=True)
    inputs = defaultdict(list)
    for e in dag["edges"]:
        if e["label"] in ("input","input_fallback"):
            src = node_map.get(e["src"])
            if src and src["type"]=="ingredient":
                inputs[e["dst"]].append(src["name"].split(",")[0].strip())
    sentences = []
    for proc in proc_nodes:
        ings = inputs.get(proc["id"], [])
        ing_str = ", ".join(ings[:3]) if ings else "ingredients"
        sentences.append(TEMPLATES.get(proc["canonical"],"Process {i}.").format(i=ing_str))
    return " ".join(sentences)


JUDGE_PROMPT = (
    "You are evaluating whether two recipe descriptions are semantically equivalent — "
    "i.e., would following either recipe produce approximately the same dish with the "
    "same key steps and ingredients?\n\n"
    "Focus on: main ingredients, cooking methods, step order, key parameters (temp/time).\n"
    "Ignore differences in phrasing or narrative style.\n\n"
    "Recipe A:\n{a}\n\nRecipe B:\n{b}\n\n"
    "Answer with a single word only: YES or NO"
)

def main():
    pairs_file = OUT / "judge_pairs.json"

    if pairs_file.exists():
        print("Loading pre-built judge pairs...", flush=True)
        with open(pairs_file) as f:
            pairs = json.load(f)
    else:
        print("Building judge pairs...", flush=True)
        dags = []
        with open(DAGS_DIR / "train_dags.jsonl") as f:
            for line in f:
                dags.append(json.loads(line))

        with open(RECIPE1M) as f:
            all_recipes = json.load(f)
        recipe_by_id = {r["id"]: r for r in all_recipes if r.get("partition") == "train"}

        pool = random.sample(dags, JUDGE_N_EACH * 4)
        tp_dags  = pool[:JUDGE_N_EACH]
        neg_pool = pool[JUDGE_N_EACH:JUDGE_N_EACH*3]
        neg_decoded = [decode_dag(d) for d in neg_pool]

        pairs = []
        for i, dag in enumerate(tp_dags):
            recipe = recipe_by_id.get(dag["id"])
            if not recipe: continue
            steps = [s["text"].strip() for s in recipe.get("instructions",[]) if s["text"].strip()]
            orig = " ".join(steps[:10])
            pairs.append({"label":"true_pos",    "a": orig, "b": decode_dag(dag)})
            pairs.append({"label":"easy_neg",    "a": orig, "b": neg_decoded[(i + JUDGE_N_EACH//2) % len(neg_decoded)]})
            pairs.append({"label":"shuffled_neg","a": orig, "b": decode_dag_reversed(dag)})

        with open(pairs_file, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"  Built {len(pairs)} pairs → {pairs_file}")

    # ── Load vLLM in isolation ────────────────────────────────────────────────
    print(f"\nLoading {JUDGE_MODEL} via vLLM (enforce_eager=True)...", flush=True)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=JUDGE_MODEL,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        enforce_eager=True,       # skip CUDA graph profiling → no memory assertion error
    )
    tok    = llm.get_tokenizer()
    params = SamplingParams(temperature=0, max_tokens=8)

    print(f"  Generating {len(pairs)} judge responses in batch...", flush=True)
    prompts = []
    for p in pairs:
        msgs = [
            {"role": "system", "content": "You answer with a single word: YES or NO."},
            {"role": "user",   "content": JUDGE_PROMPT.format(a=p["a"][:700], b=p["b"][:700])},
        ]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    outputs = llm.generate(prompts, params)

    results = []
    for p, out in zip(pairs, outputs):
        raw    = out.outputs[0].text.strip().upper()
        answer = "YES" if raw.startswith("Y") else "NO"
        results.append({"label": p["label"], "answer": answer, "raw": raw})

    # ── Compute accuracy ──────────────────────────────────────────────────────
    by_label = defaultdict(list)
    for r in results:
        by_label[r["label"]].append(r["answer"])

    tp_yes   = by_label["true_pos"].count("YES")    / max(len(by_label["true_pos"]), 1)
    easy_no  = by_label["easy_neg"].count("NO")     / max(len(by_label["easy_neg"]), 1)
    shuf_no  = by_label["shuffled_neg"].count("NO") / max(len(by_label["shuffled_neg"]), 1)
    valid    = easy_no >= 0.80

    print(f"\n{'='*60}")
    print("LLM JUDGE RESULTS (Qwen2.5-3B)")
    print(f"{'='*60}")
    print(f"  True positive  → YES rate:  {tp_yes:.1%}  ({by_label['true_pos'].count('YES')}/{len(by_label['true_pos'])})")
    print(f"  Easy negative  → NO  rate:  {easy_no:.1%}  ({by_label['easy_neg'].count('NO')}/{len(by_label['easy_neg'])})")
    print(f"  Shuffled neg   → NO  rate:  {shuf_no:.1%}  ({by_label['shuffled_neg'].count('NO')}/{len(by_label['shuffled_neg'])})")
    print(f"\n  Judge validity: {'VALID' if valid else 'QUESTIONABLE'} (easy_neg NO rate {'≥' if valid else '<'} 80%)")

    with open(OUT / "judge_results.json", "w") as f:
        json.dump({
            "model": JUDGE_MODEL,
            "n_per_condition": JUDGE_N_EACH,
            "true_pos_yes_rate":  tp_yes,
            "easy_neg_no_rate":   easy_no,
            "shuffled_neg_no_rate": shuf_no,
            "judge_valid": valid,
            "raw_results": results,
        }, f, indent=2)
    print(f"\n  Saved → {OUT}/judge_results.json")


if __name__ == "__main__":
    main()
