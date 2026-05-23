#!/usr/bin/env python3
"""
Phase 10: Constrained encoder with Qwen2.5-14B-Instruct.

Same guided-JSON approach as phase9 but with a 14B model.
Expected improvement: better implicit temp/duration extraction
(culinary world knowledge — "medium heat"=350°F, etc.)

Runs on ~500 recipes as a benchmark first, then optionally full corpus.
Compares against:
  - BGE:     temp 69.3%, dur 51.6%, ing_assign 34.2%
  - Phase9:  temp 26.3%, dur 29.3%, ing_assign 99.8%  (3B model)

With 128 GB unified memory, Qwen2.5-14B (~28 GB) leaves ~81 GB KV cache.
Increase batch_size to 60 since 14B throughput is lower per-token.

Outputs: data/eval/phase10_stats.json
         data/dags/constrained_dags_14b.jsonl  (if --full flag set)
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

RECIPE1M  = Path("/home/addeepak/AdvitResearch/cs348k/recipes.json")
DAGS_DIR  = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/dags")
OUT       = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/eval")

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
            "type": "array", "minItems": 1, "maxItems": 12,
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

SYSTEM_PROMPT = """\
You parse recipe instructions into structured JSON steps.
Only use the cooking verbs from the enum in the schema.

Rules for temp_f and duration_min:
- Explicit numbers: "bake at 350°F" → temp_f: 350, "simmer 20 min" → duration_min: 20
- Implicit heat: "medium heat" → 350, "medium-high heat" → 400, "high heat" → 425, "low heat" → 250
- Boiling/simmering water → temp_f: 212
- Implicit time: "until tender, about 15 min" → 15, "overnight" → 480
- Qualitative only ("until golden", "until fragrant") → null for that field"""

def repair_xgrammar_json(raw: str) -> dict | None:
    """Fix XGrammar/Qwen3 tokenizer apostrophe corruption.

    Bug: Qwen3 tokenizes ' differently from Qwen2.5. XGrammar's JSON string
    state machine confuses the closing " of a string with the ' token, emitting
    `ingredient_name', "  ]` instead of `ingredient_name"]`.
    Pattern: last item in a JSON array ends with ' then has garbage array-close.
    """
    fixed = re.sub(r"',\s*\"\s*\]", '"]', raw)
    fixed = re.sub(r"',\s*\"\s*\],", '"],', fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


def recipe_to_prompt(recipe: dict) -> str:
    steps = [s["text"].strip() for s in recipe.get("instructions", []) if s["text"].strip()]
    ings  = [i.get("text", "").split(",")[0].strip() for i in recipe.get("ingredients", [])]
    steps = [s[:200] for s in steps[:20]]
    inst  = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    return (
        f"Parse these cooking instructions into structured steps.\n\n"
        f"Ingredients: {', '.join(ings[:12])}\n\n"
        f"Instructions:\n{inst}\n\n"
        f"Extract each cooking action with its ingredients, temperature (°F), "
        f"and duration (minutes)."
    )


def run(recipes, llm, tok, params, batch_size, desc="encode"):
    ing_rates = []; temp_rates = []; dur_rates = []
    n_ok = n_fail = n_repaired = 0
    results = []

    for i in tqdm(range(0, len(recipes), batch_size), desc=desc):
        batch = recipes[i:i+batch_size]
        msgs  = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": recipe_to_prompt(r)},
            ]
            for r in batch
        ]
        # enable_thinking=False disables Qwen3's chain-of-thought preamble,
        # which otherwise prepends <think>...</think> and breaks JSON parsing
        prompts = [tok.apply_chat_template(
                       m, tokenize=False, add_generation_prompt=True,
                       enable_thinking=False)
                   for m in msgs]
        outputs = llm.generate(prompts, params)

        for recipe, out in zip(batch, outputs):
            raw = out.outputs[0].text.strip()
            parsed = None
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                repaired = repair_xgrammar_json(raw)
                if repaired:
                    parsed = repaired
                    n_repaired += 1
            if parsed and "steps" in parsed:
                steps = parsed["steps"]
                n_ok += 1
                ing_rates.append(sum(1 for s in steps if s["ingredients"]) / max(len(steps),1))
                temp_rates.append(sum(1 for s in steps if s.get("temp_f") is not None) / max(len(steps),1))
                dur_rates.append(sum(1 for s in steps if s.get("duration_min") is not None) / max(len(steps),1))
                results.append((recipe, steps))
            else:
                n_fail += 1
                results.append((recipe, None))

    parse_rate = n_ok / max(n_ok + n_fail, 1)
    return results, {
        "parse_rate":        parse_rate,
        "n_repaired":        n_repaired,
        "ing_assignment":    float(np.mean(ing_rates)) if ing_rates else 0,
        "temp_capture":      float(np.mean(temp_rates)) if temp_rates else 0,
        "dur_capture":       float(np.mean(dur_rates)) if dur_rates else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",  action="store_true", help="Run on full corpus (default: 500-recipe benchmark)")
    parser.add_argument("--n",     type=int, default=500, help="Benchmark size (default 500)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct",
                        help="Model to use (default: Qwen2.5-14B-Instruct)")
    parser.add_argument("--split", type=str, default="test", choices=["train","val","test"],
                        help="Partition to sample benchmark from (default: test for held-out eval)")
    args = parser.parse_args()

    import random; random.seed(42)

    print("Loading recipes...", flush=True)
    with open(RECIPE1M) as f:
        all_recipes = json.load(f)
    split_recipes = [r for r in all_recipes if r.get("partition") == args.split]

    if args.full:
        recipes = [r for r in all_recipes if r.get("partition") == "train"]
        out_dags = DAGS_DIR / "constrained_dags_14b.jsonl"
        print(f"  Full train corpus: {len(recipes):,} recipes")
    else:
        recipes = random.sample(split_recipes, min(args.n, len(split_recipes)))
        out_dags = None
        print(f"  Benchmark: {len(recipes):,} recipes from '{args.split}' split (held-out)")

    print(f"\nLoading {args.model} via vLLM...", flush=True)
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(
        model=args.model,
        max_model_len=4096,
        gpu_memory_utilization=0.88,   # leave headroom for display/other processes
        dtype="bfloat16",
        enforce_eager=True,
    )
    tok = llm.get_tokenizer()
    params = SamplingParams(
        temperature=0.0,
        max_tokens=2000,
        structured_outputs=StructuredOutputsParams(json=DAG_SCHEMA),
    )

    t0 = time.time()
    results, stats = run(recipes, llm, tok, params, batch_size=60, desc="encode-14b")
    elapsed = time.time() - t0

    stats["model"]       = args.model
    stats["n_recipes"]  = len(recipes)
    stats["elapsed_s"]  = elapsed
    stats["recipes_per_hour"] = len(recipes) / elapsed * 3600
    stats["baselines"]  = {
        "bge":    {"parse_rate": 0.975, "ing_assignment": 0.342, "temp": 0.693, "dur": 0.516},
        "phase9": {"parse_rate": 0.983, "ing_assignment": 0.998, "temp": 0.263, "dur": 0.293},
    }

    model_label = args.model.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"{model_label.upper()} CONSTRAINED ENCODER  (n={len(recipes):,})")
    print(f"{'='*60}")
    print(f"  {'Metric':<30}  {model_label[:8]:>8}  {'3B (p9)':>8}  {'BGE':>8}")
    print(f"  {'-'*58}")
    print(f"  {'Parse rate':<30}  {stats['parse_rate']:>8.1%}  {'98.3%':>8}  {'97.5%':>8}")
    print(f"  {'  (repaired outputs)':<30}  {stats['n_repaired']:>8}  {'':>8}  {'':>8}")
    print(f"  {'Ingredient assignment':<30}  {stats['ing_assignment']:>8.1%}  {'99.8%':>8}  {'34.2%':>8}")
    print(f"  {'Temp capture':<30}  {stats['temp_capture']:>8.1%}  {'26.3%':>8}  {'69.3%':>8}")
    print(f"  {'Duration capture':<30}  {stats['dur_capture']:>8.1%}  {'29.3%':>8}  {'51.6%':>8}")
    print(f"\n  Throughput: {stats['recipes_per_hour']:.0f} recipes/hr")
    if args.full:
        print(f"  ETA for full 35K corpus: {35000/stats['recipes_per_hour']:.1f}h")

    with open(OUT / "phase10_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved → {OUT}/phase10_stats.json")

    if out_dags and results:
        with open(out_dags, "w") as f:
            for recipe, steps in results:
                if steps:
                    f.write(json.dumps({"id": recipe["id"], "title": recipe.get("title",""),
                                        "steps": steps}) + "\n")
        print(f"  DAGs     → {out_dags}")


if __name__ == "__main__":
    main()
