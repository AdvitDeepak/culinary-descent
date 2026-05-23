#!/usr/bin/env python3
"""
Phase 9: Constrained LLM encoder at full corpus scale (~35K recipes).

Improvements over phase7/phase8:
  - Improved prompt with few-shot examples for temp/duration extraction
    (addresses the §5 weakness: temp 13%, duration 24% in phase8)
  - Checkpointing: saves progress every 500 recipes so crash = resume
  - Processes ALL 34,957 training recipes (phase8 = 278 sample)

Runtime: ~5-7 hours on a single GPU.

Outputs:
  data/dags/constrained_dags.jsonl   — one constrained DAG per line
  data/eval/phase9_stats.json        — aggregate metrics vs BGE baseline
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

RECIPE1M   = Path("/home/addeepak/AdvitResearch/cs348k/recipes.json")
DAGS_DIR   = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/dags")
OUT        = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/eval")
OUT_DAGS   = DAGS_DIR / "constrained_dags.jsonl"
CHECKPOINT = DAGS_DIR / "constrained_dags_checkpoint.txt"
BATCH_SIZE = 40
SEED       = 42

CANONICAL_15 = [
    "bake",      # oven dry: bake, roast, toast
    "grill",     # radiant/flame: grill, broil
    "saute",     # fat stovetop: saute, fry, sear, brown
    "boil",      # high moist: boil, blanch
    "simmer",    # low moist: simmer, braise, poach
    "steam",
    "mix",       # bulk combine: mix, stir, toss, fold
    "whisk",     # aerate: whisk, beat
    "blend",     # mechanical puree: blend, puree
    "knead",
    "chop",      # knife prep: chop, dice, slice, mince, grate, peel, crush
    "marinate",  # passive liquid: marinate, soak, brine
    "chill",     # cold thermal: cool, chill, freeze, rest, refrigerate
    "season",    # finish/apply: season, coat, brush, drizzle
    "reduce",    # liquid ops: reduce, dissolve, drain, melt, strain
]

DAG_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "steps": {
            "type": "array", "minItems": 1, "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "canonical":    {"type": "string", "enum": CANONICAL_15},
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

Type guide — collapse synonyms into the canonical form:
- bake: bake, roast, toast (oven dry heat)
- grill: grill, broil (direct flame or radiant heat)
- saute: saute, fry, sear, brown, pan-fry (fat + stovetop)
- boil: boil, blanch (high-temp moist)
- simmer: simmer, braise, poach, stew (low-temp moist)
- steam: steam
- mix: mix, stir, toss, fold, combine (gentle bulk combine)
- whisk: whisk, beat (aerate or emulsify)
- blend: blend, puree (mechanical size reduction)
- knead: knead
- chop: chop, dice, slice, mince, grate, peel, shred, crush (knife prep)
- marinate: marinate, soak, brine (passive liquid absorption)
- chill: cool, chill, freeze, rest, refrigerate (passive cold)
- season: season, coat, brush, drizzle, sprinkle (apply flavor or fat)
- reduce: reduce, dissolve, drain, melt, strain (liquid transform or removal)

Rules for temp_f and duration_min:
- Extract explicit numbers only: "bake at 350°F" → temp_f: 350, "simmer 20 min" → duration_min: 20
- Implicit heat levels: "medium heat" → temp_f: 325, "high heat" → temp_f: 400, "low heat" → temp_f: 250
- Implicit time phrases: "until tender ~15 min" → duration_min: 15, "overnight" → duration_min: 480
- Qualitative only ("until golden brown", "until fragrant") → null
- Boiling/simmering water: temp_f: 212"""

def recipe_to_prompt(recipe: dict) -> str:
    steps = [s["text"].strip() for s in recipe.get("instructions", []) if s["text"].strip()]
    ings  = [i.get("text", "").split(",")[0].strip() for i in recipe.get("ingredients", [])]
    # truncate to avoid context overflow: 20 steps max, each capped at 200 chars
    steps = [s[:200] for s in steps[:20]]
    inst  = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    return (
        f"Parse these cooking instructions into structured steps.\n\n"
        f"Ingredients: {', '.join(ings[:12])}\n\n"
        f"Instructions:\n{inst}\n\n"
        f"Extract each cooking action with its ingredients, temperature (°F), "
        f"and duration (minutes)."
    )


def main():
    print("Loading Recipe1M training recipes...", flush=True)
    with open(RECIPE1M) as f:
        all_recipes = json.load(f)
    train = [r for r in all_recipes if r.get("partition") == "train"]
    print(f"  {len(train):,} training recipes")

    # Resume from checkpoint if exists
    done_ids = set()
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            done_ids = set(f.read().splitlines())
        print(f"  Resuming: {len(done_ids):,} already done")

    todo = [r for r in train if r["id"] not in done_ids]
    print(f"  {len(todo):,} remaining")

    if not todo:
        print("All done!")
        return

    # Load vLLM
    print("\nLoading Qwen2.5-3B via vLLM...", flush=True)
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(
        model="Qwen/Qwen2.5-3B-Instruct",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        enforce_eager=True,
    )
    tok = llm.get_tokenizer()
    params = SamplingParams(
        temperature=0.0,
        max_tokens=600,
        structured_outputs=StructuredOutputsParams(json=DAG_SCHEMA),
    )

    # Stats
    n_total = n_ok = n_fail = 0
    ing_rates = []; temp_rates = []; dur_rates = []
    t0 = time.time()

    out_f  = open(OUT_DAGS, "a")
    ckpt_f = open(CHECKPOINT, "a")

    for i in tqdm(range(0, len(todo), BATCH_SIZE), desc="encode"):
        batch = todo[i:i+BATCH_SIZE]
        msgs  = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": recipe_to_prompt(r)},
            ]
            for r in batch
        ]
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in msgs]
        outputs = llm.generate(prompts, params)

        for recipe, out in zip(batch, outputs):
            raw = out.outputs[0].text.strip()
            n_total += 1
            try:
                parsed = json.loads(raw)
                steps  = parsed["steps"]
                n_ok  += 1

                ing_r  = sum(1 for s in steps if s["ingredients"]) / max(len(steps), 1)
                temp_r = sum(1 for s in steps if s.get("temp_f") is not None) / max(len(steps), 1)
                dur_r  = sum(1 for s in steps if s.get("duration_min") is not None) / max(len(steps), 1)
                ing_rates.append(ing_r); temp_rates.append(temp_r); dur_rates.append(dur_r)

                record = {
                    "id":    recipe["id"],
                    "title": recipe.get("title", ""),
                    "steps": steps,
                }
                out_f.write(json.dumps(record) + "\n")
            except Exception:
                n_fail += 1

            ckpt_f.write(recipe["id"] + "\n")

        out_f.flush(); ckpt_f.flush()

        # Progress log every 10 batches
        if (i // BATCH_SIZE) % 10 == 0 and n_ok > 0:
            elapsed = time.time() - t0
            rate    = n_total / elapsed
            remaining = (len(todo) - n_total) / rate / 3600
            print(f"\n  [{n_total}/{len(todo)}] parse={n_ok/n_total:.1%}  "
                  f"ing={np.mean(ing_rates):.1%}  "
                  f"temp={np.mean(temp_rates):.1%}  "
                  f"dur={np.mean(dur_rates):.1%}  "
                  f"ETA {remaining:.1f}h", flush=True)

    out_f.close(); ckpt_f.close()

    # Final stats
    stats = {
        "n_total": n_total,
        "n_ok":    n_ok,
        "n_fail":  n_fail,
        "parse_rate": n_ok / max(n_total, 1),
        "ing_assignment_rate": float(np.mean(ing_rates)) if ing_rates else 0,
        "temp_capture_rate":   float(np.mean(temp_rates)) if temp_rates else 0,
        "dur_capture_rate":    float(np.mean(dur_rates)) if dur_rates else 0,
        "bge_baselines": {
            "parse_rate": 0.975,
            "ing_assignment": 0.342,
            "temp_capture": 0.693,
            "dur_capture": 0.516,
        },
        "phase8_constrained_sample": {
            "parse_rate": 0.927,
            "ing_assignment": 0.970,
            "temp_capture": 0.130,
            "dur_capture": 0.235,
        },
    }
    with open(OUT / "phase9_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print("PHASE 9 COMPLETE")
    print(f"{'='*60}")
    print(f"  Recipes processed: {n_total:,}")
    print(f"  Parse rate:        {stats['parse_rate']:.1%}  (BGE: 97.5%, phase8: 92.7%)")
    print(f"  Ing assignment:    {stats['ing_assignment_rate']:.1%}  (BGE: 34.2%, phase8: 97.0%)")
    print(f"  Temp capture:      {stats['temp_capture_rate']:.1%}  (BGE: 69.3%, phase8: 13.0%)")
    print(f"  Dur capture:       {stats['dur_capture_rate']:.1%}  (BGE: 51.6%, phase8: 23.5%)")
    print(f"\n  Constrained DAGs → {OUT_DAGS}")
    print(f"  Stats            → {OUT}/phase9_stats.json")


if __name__ == "__main__":
    main()
