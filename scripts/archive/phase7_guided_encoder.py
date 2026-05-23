#!/usr/bin/env python3
"""
Phase 7: Constrained LLM encoder — guided JSON decoding.

Replace spaCy + BGE nearest-neighbor with a single LLM call whose output
is constrained by guided_json to our 43-type vocabulary.

  NL recipe  →  LLM (guided JSON, vocab-constrained)  →  structured steps  →  RecipeDAG

Key property: the `canonical` field in every step is an enum of the 43 canonical
process types — the model CANNOT generate a process type outside the vocabulary.
This is the encoder-side information bottleneck made explicit.

Evaluates (on 200 recipes vs existing BGE DAGs):
  - JSON parse rate (guided decoding should be ~100%)
  - Process-seq LCS vs BGE encoder (how much the two encoders agree)
  - Ingredient assignment rate (steps that list at least one ingredient)
  - Arg capture rate (steps with temp / duration filled in)

Outputs: data/eval/guided_encoder_results.json
"""

import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

DAGS_DIR = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/dags")
RECIPE1M = Path("/home/addeepak/AdvitResearch/cs348k/recipes.json")
OUT      = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/eval")
OUT.mkdir(parents=True, exist_ok=True)

EVAL_N = 200
SEED   = 42
random.seed(SEED)

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

# JSON schema — canonical is a hard enum; temp/duration are nullable numbers
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


def recipe_to_prompt(recipe: dict) -> str:
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


def main():
    print("Loading recipes + existing BGE DAGs...", flush=True)
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
    sample = random.sample(candidates, EVAL_N)
    print(f"  {EVAL_N} recipes sampled")

    # ── Load vLLM ────────────────────────────────────────────────────────────────
    print("\nLoading Qwen2.5-3B via vLLM (guided JSON)...", flush=True)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen2.5-3B-Instruct",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        enforce_eager=True,
    )
    tok = llm.get_tokenizer()

    from vllm.sampling_params import StructuredOutputsParams
    guided_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        structured_outputs=StructuredOutputsParams(json=DAG_SCHEMA),
    )

    # ── Encode ───────────────────────────────────────────────────────────────────
    print(f"\nEncoding {EVAL_N} recipes with constrained LLM...", flush=True)
    batch_size = 30
    results    = []   # list of (recipe, parsed_steps | None)
    failed     = 0

    for i in tqdm(range(0, len(sample), batch_size), desc="encode"):
        batch = sample[i:i+batch_size]
        msgs  = [
            [
                {"role": "system", "content":
                 "You parse recipe instructions into structured JSON. "
                 "Only use the cooking verbs from the enum in the schema."},
                {"role": "user", "content": recipe_to_prompt(r)},
            ]
            for r in batch
        ]
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in msgs]
        outputs = llm.generate(prompts, guided_params)

        for recipe, out in zip(batch, outputs):
            raw = out.outputs[0].text.strip()
            try:
                parsed = json.loads(raw)
                results.append((recipe, parsed["steps"]))
            except Exception:
                results.append((recipe, None))
                failed += 1

    parse_rate = 1.0 - failed / EVAL_N
    print(f"\n  JSON parse rate: {parse_rate:.1%}  ({EVAL_N-failed}/{EVAL_N} valid)")

    # ── Evaluate ─────────────────────────────────────────────────────────────────
    lcs_scores   = []
    ing_rates    = []
    temp_rates   = []
    dur_rates    = []
    vocab_ok     = 0   # steps where canonical is in our 43 (should be 100% due to constraint)

    for recipe, steps in results:
        if steps is None:
            continue

        # process-seq LCS vs BGE encoder
        bge_dag  = bge_dags[recipe["id"]]
        bge_seq  = [n["canonical"] for n in sorted(
            [n for n in bge_dag["nodes"] if n["type"] == "process"],
            key=lambda n: n["step_idx"]
        )]
        llm_seq  = [s["canonical"] for s in steps]
        lcs_scores.append(process_lcs_f1(llm_seq, bge_seq))

        # ingredient assignment: what fraction of steps list ≥1 ingredient?
        ing_rates.append(sum(1 for s in steps if s["ingredients"]) / max(len(steps), 1))

        # arg capture: steps that have temp / duration
        n = max(len(steps), 1)
        temp_rates.append(sum(1 for s in steps if s.get("temp_f") is not None) / n)
        dur_rates.append(sum(1 for s in steps if s.get("duration_min") is not None) / n)

        # vocab check (should be 100%)
        vocab_ok += sum(1 for s in steps if s["canonical"] in CANONICAL_43)

    total_steps = sum(len(s) for _, s in results if s)

    print(f"\n{'='*60}")
    print("CONSTRAINED ENCODER RESULTS")
    print(f"{'='*60}")
    print(f"  JSON parse rate:                {parse_rate:.1%}   (BGE: 97.5%)")
    print(f"  Vocab constraint satisfied:     {vocab_ok}/{total_steps} steps  "
          f"({vocab_ok/max(total_steps,1):.1%})")
    print(f"  Process-seq LCS vs BGE:         {np.mean(lcs_scores):.3f}   "
          f"(agreement between encoders)")
    print(f"  Steps with ≥1 ingredient:       {np.mean(ing_rates):.1%}   "
          f"(BGE substring match: 34.2%)")
    print(f"  Steps with temperature:         {np.mean(temp_rates):.1%}   "
          f"(BGE regex: 69.3% of recipes)")
    print(f"  Steps with duration:            {np.mean(dur_rates):.1%}   "
          f"(BGE regex: 51.6% of recipes)")

    # save a few examples
    print("\nSample encodings:")
    for recipe, steps in results[:3]:
        if steps is None: continue
        print(f"\n  [{recipe.get('title','')}]")
        for s in steps:
            ing = ", ".join(s["ingredients"][:3]) or "—"
            t   = f"{s['temp_f']}°F" if s.get("temp_f") else "—"
            d   = f"{s['duration_min']}min" if s.get("duration_min") else "—"
            print(f"    {s['canonical']:12s}  ings=[{ing}]  temp={t}  dur={d}")

    with open(OUT / "guided_encoder_results.json", "w") as f:
        json.dump({
            "n_eval": EVAL_N,
            "parse_rate": parse_rate,
            "vocab_constraint_rate": vocab_ok / max(total_steps, 1),
            "process_lcs_vs_bge": float(np.mean(lcs_scores)),
            "ing_assignment_rate": float(np.mean(ing_rates)),
            "temp_capture_rate": float(np.mean(temp_rates)),
            "dur_capture_rate": float(np.mean(dur_rates)),
            "bge_baselines": {
                "parse_rate": 0.975,
                "ing_assignment": 0.342,
                "temp_capture": 0.693,
                "dur_capture": 0.516,
            },
        }, f, indent=2)

    print(f"\n  Saved → {OUT}/guided_encoder_results.json")


if __name__ == "__main__":
    main()
