#!/usr/bin/env python3
"""
Phase 9b: Hybrid arg extraction — regex backfill on constrained DAGs.

The constrained LLM encoder excels at semantic structure (canonical type,
ingredient assignment at 99.8%) but is conservative on temp/duration.
This script backfills null temp_f / duration_min using regex on the
*specific instruction sentence* each step was derived from.

Hybrid design:
  LLM  → canonical type + ingredient assignment  (hard semantic task)
  Regex → temp_f + duration_min                  (easy detection task)

Inputs:  data/dags/constrained_dags.jsonl  (phase9 output)
         recipes.json                       (original recipe text)
Outputs: data/dags/constrained_dags_hybrid.jsonl
         data/eval/phase9b_stats.json
"""

import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

RECIPE1M    = Path("/home/addeepak/AdvitResearch/cs348k/recipes.json")
DAGS_DIR    = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/dags")
OUT         = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/eval")
IN_DAGS     = DAGS_DIR / "constrained_dags.jsonl"
OUT_DAGS    = DAGS_DIR / "constrained_dags_hybrid.jsonl"

# ── Regex patterns ────────────────────────────────────────────────────────────

TEMP_PATTERNS = [
    # explicit °F
    (re.compile(r'(\d+)\s*°?\s*f(?:\s|$|\.)', re.I),
     lambda m: float(m.group(1))),
    # explicit °C
    (re.compile(r'(\d+)\s*°?\s*c(?:\s|$|\.)', re.I),
     lambda m: float(m.group(1)) * 9/5 + 32),
    # "350 degrees"
    (re.compile(r'(\d+)\s*degrees?', re.I),
     lambda m: float(m.group(1))),
    # named heat levels
    (re.compile(r'medium[- ]high\s+heat', re.I), lambda m: 400.0),
    (re.compile(r'medium[- ]low\s+heat',  re.I), lambda m: 300.0),
    (re.compile(r'medium\s+heat',         re.I), lambda m: 350.0),
    (re.compile(r'high\s+heat',           re.I), lambda m: 425.0),
    (re.compile(r'low\s+heat',            re.I), lambda m: 250.0),
    # boiling
    (re.compile(r'\bboil(?:ing)?\b',      re.I), lambda m: 212.0),
]

DUR_PATTERNS = [
    # "1 hour 30 minutes" or "1½ hours"
    (re.compile(r'(\d+)\s*(?:to\s*\d+\s*)?hours?\s*(?:and\s*)?(\d+)\s*minutes?', re.I),
     lambda m: float(m.group(1)) * 60 + float(m.group(2))),
    # "2 hours" or "1-2 hours"
    (re.compile(r'(\d+)\s*(?:[-–to]+\s*\d+\s*)?hours?', re.I),
     lambda m: float(m.group(1)) * 60),
    # "45 minutes" or "30-45 minutes"
    (re.compile(r'(\d+)\s*(?:[-–to]+\s*\d+\s*)?minutes?', re.I),
     lambda m: float(m.group(1))),
    # "about 15 mins"
    (re.compile(r'(?:about|approx\.?|~)\s*(\d+)\s*(?:min|minute)', re.I),
     lambda m: float(m.group(1))),
    # "overnight"
    (re.compile(r'\bovernight\b', re.I), lambda m: 480.0),
]


def extract_temp(text: str):
    for pat, fn in TEMP_PATTERNS:
        m = pat.search(text)
        if m:
            try: return fn(m)
            except: continue
    return None


def extract_duration(text: str):
    for pat, fn in DUR_PATTERNS:
        m = pat.search(text)
        if m:
            try: return fn(m)
            except: continue
    return None


def main():
    print("Loading recipes...", flush=True)
    with open(RECIPE1M) as f:
        all_recipes = json.load(f)
    recipe_by_id = {r["id"]: r for r in all_recipes}
    print(f"  {len(recipe_by_id):,} recipes loaded")

    print("Loading constrained DAGs...", flush=True)
    dags = []
    with open(IN_DAGS) as f:
        for line in f:
            dags.append(json.loads(line))
    print(f"  {len(dags):,} constrained DAGs loaded")

    # Track stats
    before_temp = before_dur = after_temp = after_dur = 0
    total_steps = backfill_temp = backfill_dur = 0

    out_records = []

    for dag in tqdm(dags, desc="backfill"):
        recipe = recipe_by_id.get(dag["id"])
        instructions = []
        if recipe:
            instructions = [s["text"].strip() for s in recipe.get("instructions", [])
                            if s["text"].strip()]

        new_steps = []
        for i, step in enumerate(dag["steps"]):
            total_steps += 1
            t_orig = step.get("temp_f")
            d_orig = step.get("duration_min")

            if t_orig is not None: before_temp += 1
            if d_orig is not None: before_dur  += 1

            t_new = t_orig
            d_new = d_orig

            # backfill from the corresponding instruction sentence (and neighbors)
            if instructions:
                # search the step's instruction + adjacent ones for context
                window = instructions[max(0,i-1):i+2]
                ctx = " ".join(window)

                if t_new is None:
                    t_new = extract_temp(ctx)
                    if t_new: backfill_temp += 1

                if d_new is None:
                    d_new = extract_duration(ctx)
                    if d_new: backfill_dur += 1

            if t_new is not None: after_temp += 1
            if d_new is not None: after_dur  += 1

            new_steps.append({**step, "temp_f": t_new, "duration_min": d_new})

        out_records.append({**dag, "steps": new_steps})

    # Write output
    with open(OUT_DAGS, "w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")

    n = len(dags)
    stats = {
        "n_dags": n,
        "total_steps": total_steps,
        "temp_capture": {
            "before": before_temp / total_steps,
            "after":  after_temp  / total_steps,
            "backfilled": backfill_temp,
        },
        "dur_capture": {
            "before": before_dur / total_steps,
            "after":  after_dur  / total_steps,
            "backfilled": backfill_dur,
        },
        "baselines": {
            "bge_temp": 0.693, "bge_dur": 0.516,
            "phase8_temp": 0.130, "phase8_dur": 0.235,
            "phase9_temp": 0.263, "phase9_dur": 0.293,
        },
    }

    print(f"\n{'='*60}")
    print("HYBRID ARG EXTRACTION RESULTS")
    print(f"{'='*60}")
    print(f"  {'Metric':<30}  {'Before':>8}  {'After':>8}  {'BGE':>8}")
    print(f"  {'-'*58}")
    print(f"  {'Temp capture (per-step)':<30}  "
          f"{before_temp/total_steps:>8.1%}  "
          f"{after_temp/total_steps:>8.1%}  "
          f"{'69.3%':>8}")
    print(f"  {'Duration capture (per-step)':<30}  "
          f"{before_dur/total_steps:>8.1%}  "
          f"{after_dur/total_steps:>8.1%}  "
          f"{'51.6%':>8}")
    print(f"\n  Backfilled: {backfill_temp:,} temps, {backfill_dur:,} durations")
    print(f"\n  Saved → {OUT_DAGS}")

    with open(OUT / "phase9b_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved → {OUT}/phase9b_stats.json")


if __name__ == "__main__":
    main()
