#!/usr/bin/env python3
"""
Fast hybrid DAG parser — no LLM required.

Extracts process DAGs from Recipe1M instruction text using:
  1. Step type: priority-ordered keyword rules (same 15 types as LLM encoder)
     + context fallbacks ("place in oven" → bake, "cook on stovetop" → saute)
  2. Ingredients per step: fuzzy substring match of recipe ingredients in step text
  3. Temperature: regex + implicit heat-level lookup table
  4. Duration: regex (explicit minutes/hours + qualitative terms)
  5. Edges: ingredient-flow heuristic (same as build_dag_graph.py)

Covers ALL 93K rated training recipes in ~60 seconds.
Output format identical to LLM encoder (rated_dags_graph.jsonl).

Outputs:
  data/dags/fast_dags.jsonl   — all rated train recipes
  data/eval/fast_dag_stats.json
"""

import json, os
import re
from collections import Counter
from pathlib import Path

import numpy as np

BASE   = Path(__file__).resolve().parent.parent
LAYER1 = Path(os.environ.get("RECIPE1M", str(BASE.parent / "layer1.json")))
JSONL  = BASE / "data/enriched/joined_recipes.jsonl"
OUT    = BASE / "data/dags/fast_dags.jsonl"

# ── Step-type rules (priority order — first match wins) ───────────────────────
# Each entry: (canonical, keyword_triggers, context_patterns)
# context_patterns: regex applied to full step text when no keyword matches

STEP_RULES = [
    ("knead",    ["knead"],
                 [r"knead"]),
    ("marinate", ["marinate", "marinade", "brine", "soak"],
                 [r"let.*sit.*(?:hour|overnight)", r"refrigerate.*(?:hour|overnight)"]),
    ("grill",    ["grill", "barbecue", "char-grill", "griddle", "broil"],
                 [r"(?:direct|open).*flame", r"under.*broiler"]),
    ("steam",    ["steam"],
                 [r"steamer", r"double boiler"]),
    ("reduce",   ["reduce", "cook down", "thicken", "dissolve", "drain", "strain", "melt"],
                 [r"until.*(?:thick|syrup|reduced)"]),
    ("whisk",    ["whisk", " beat ", "whip"],
                 [r"until.*(?:stiff|fluffy|peaks)"]),
    ("blend",    ["blend", "puree", "food processor", "blender", "pulse"],
                 [r"until.*smooth"]),
    ("bake",     ["bake", "roast", " broil", "toast"],
                 [r"preheat.*oven", r"place.*(?:in|into).*oven",
                  r"baking sheet", r"baking dish", r"baking pan"]),
    ("simmer",   ["simmer", "stew", "braise", "poach"],
                 [r"low.*heat.*(?:cook|until)", r"gentle.*heat"]),
    ("boil",     ["boil", "blanch"],
                 [r"bring.*to.*boil", r"boiling water"]),
    ("saute",    ["saute", "sauté", "stir-fry", "pan-fry", " fry ", "sear", "brown", " fry,"],
                 [r"(?:medium|high).*heat", r"(?:skillet|pan|wok).*heat",
                  r"heat.*(?:oil|butter).*(?:pan|skillet)", r"cook.*(?:stirring|until)"]),
    ("chop",     ["chop", "dice", "mince", "slice", " cut ", "grate", "peel", "shred",
                  "crush", "halve", "quarter", "julienne", "zest"],
                 [r"cut.*into", r"slice.*thin"]),
    ("chill",    ["chill", "refrigerate", "freeze", " cool", "rest", "set aside.*(?:cool|room)"],
                 [r"let.*(?:cool|rest|stand)", r"allow.*to.*cool", r"room temperature"]),
    ("season",   ["season", "salt and pepper", "sprinkle", "drizzle", "brush", "coat", "dust",
                  "garnish", "top with"],
                 [r"to taste", r"adjust.*seasoning"]),
    ("mix",      ["mix", "stir", "combine", "toss", "fold", " add ", "incorporate",
                  "blend in", "pour", "transfer"],
                 [r"until.*(?:combined|incorporated|mixed)"]),
]

STEP_TYPE_NAMES = [r[0] for r in STEP_RULES]

# Implicit temperature lookup (°F)
HEAT_LEVEL = {
    "low heat":         250,
    "medium-low heat":  300,
    "medium low heat":  300,
    "medium heat":      350,
    "medium-high heat": 400,
    "medium high heat": 400,
    "high heat":        425,
    "very high heat":   450,
    "boiling":          212,
    "simmer":           200,
}

OVEN_TEMPS = {
    "low oven":    300,
    "medium oven": 350,
    "hot oven":    400,
    "very hot":    450,
}


# ── Extraction helpers ─────────────────────────────────────────────────────────

def classify_step(text: str) -> str | None:
    """Return canonical step type for instruction text, or None if no match."""
    tl = text.lower()
    for canonical, keywords, context_pats in STEP_RULES:
        if any(kw in tl for kw in keywords):
            return canonical
    # Context patterns (second pass — only if no keyword matched)
    for canonical, keywords, context_pats in STEP_RULES:
        if any(re.search(pat, tl) for pat in context_pats):
            return canonical
    return None


def extract_temp(text: str) -> float | None:
    tl = text.lower()
    # Explicit °F
    m = re.search(r'(\d{2,3})\s*°?\s*[Ff]', text)
    if m:
        return float(m.group(1))
    # Explicit °C → convert
    m = re.search(r'(\d{2,3})\s*°?\s*[Cc]', text)
    if m:
        return round(float(m.group(1)) * 9/5 + 32)
    # Heat level
    for phrase, temp in HEAT_LEVEL.items():
        if phrase in tl:
            return float(temp)
    for phrase, temp in OVEN_TEMPS.items():
        if phrase in tl:
            return float(temp)
    return None


def extract_duration(text: str) -> float | None:
    tl = text.lower()
    total = 0.0
    found = False
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(?:to\s*\d+\s*)?(?:hour|hr)', tl):
        total += float(m.group(1)) * 60; found = True
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(?:to\s*\d+\s*)?min', tl):
        total += float(m.group(1)); found = True
    if not found:
        if "overnight" in tl:  return 480.0
        if "all day" in tl:    return 480.0
    return min(total, 600.0) if found else None


def tokenize_ingredient(ing_text: str) -> list[str]:
    """Extract searchable tokens from an ingredient string."""
    # Strip leading quantities: "2 cups chopped onion" → ["chopped", "onion"]
    text = re.sub(r'^\d[\d\s/\-\.]*(?:cup|tbsp|tsp|oz|lb|gram|g|ml|l|tablespoon|teaspoon|pound|ounce|can|package|pkg|bunch|clove|head|stalk|slice|strip)s?\b\.?', '', ing_text, flags=re.I).strip()
    # Remove parentheticals and punctuation
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', ' ', text.lower())
    tokens = [t for t in text.split() if len(t) >= 3
              and t not in {'the', 'and', 'for', 'with', 'into', 'from', 'some', 'cup',
                            'large', 'small', 'medium', 'fresh', 'dried', 'ground',
                            'chopped', 'sliced', 'diced', 'minced', 'grated', 'peeled',
                            'cooked', 'frozen', 'canned', 'optional'}]
    return tokens


def assign_ingredients(step_text: str, all_ingredients: list[str]) -> list[str]:
    """Return subset of recipe ingredients mentioned in this step."""
    tl = step_text.lower()
    matched = []
    for ing_raw in all_ingredients:
        tokens = tokenize_ingredient(ing_raw)
        if tokens and any(tok in tl for tok in tokens):
            # Use normalized short name (first meaningful token after trimming quantity)
            short = re.sub(r'^\d[\d\s/\.]*(?:cup|tbsp|tsp|oz|lb|g|ml|tablespoon|teaspoon|pound|ounce|can|package)s?\b', '', ing_raw, flags=re.I).strip()
            short = re.sub(r',.*', '', short).strip()
            if short:
                matched.append(short[:40])
    return matched[:6]


# ── Edge building (same as build_dag_graph.py) ─────────────────────────────────

def normalize_token(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    if s.endswith("ies"):   s = s[:-3] + "y"
    elif s.endswith("es") and len(s) > 4: s = s[:-2]
    elif s.endswith("s") and len(s) > 3:  s = s[:-1]
    return s

def ing_tokens(ing_list):
    tokens = set()
    for ing in ing_list:
        for part in ing.split():
            tok = normalize_token(part)
            if len(tok) >= 3:
                tokens.add(tok)
    return tokens

def build_edges(steps):
    n = len(steps)
    tok_sets = [ing_tokens(s.get("ingredients", [])) for s in steps]
    edges = []
    parents = [[] for _ in range(n)]
    for j in range(n):
        if not tok_sets[j]: continue
        matched = set()
        for tok in tok_sets[j]:
            for i in range(j - 1, -1, -1):
                if tok in tok_sets[i]:
                    matched.add(i); break
        for i in matched:
            edges.append([i, j])
            parents[j].append(i)
    depths = [0] * n
    for j in range(n):
        if parents[j]:
            depths[j] = max(depths[i] for i in parents[j]) + 1
    for j, s in enumerate(steps):
        s["depth"] = depths[j]
    return edges


# ── Main ───────────────────────────────────────────────────────────────────────

print("Loading rated training recipe IDs...", flush=True)
rated_ids = set()
with open(JSONL) as f:
    for line in f:
        r = json.loads(line)
        if r["n_reviews"] >= 3:
            rated_ids.add(r["id"])
print(f"  {len(rated_ids):,} rated recipes", flush=True)

print("Loading layer1.json...", flush=True)
with open(LAYER1) as f:
    layer1 = json.load(f)
targets = [r for r in layer1 if r["id"] in rated_ids]
print(f"  {len(targets):,} matching records", flush=True)

print("Parsing DAGs...", flush=True)
n_ok = n_skip = 0
step_type_counts: Counter = Counter()
n_steps_list = []
n_matched_ings = []
n_with_temp = []
n_with_dur = []

with open(OUT, "w") as fout:
    for recipe in targets:
        instructions = recipe.get("instructions", [])
        all_ings     = [i.get("text", "") for i in recipe.get("ingredients", [])]

        steps = []
        for step in instructions:
            text = step.get("text", "").strip()
            if not text:
                continue
            canonical = classify_step(text)
            if canonical is None:
                continue
            ings      = assign_ingredients(text, all_ings)
            temp_f    = extract_temp(text)
            dur_min   = extract_duration(text)
            steps.append({
                "canonical":    canonical,
                "ingredients":  ings,
                "temp_f":       temp_f,
                "duration_min": dur_min,
            })

        if not steps:
            n_skip += 1
            continue

        edges = build_edges(steps)
        n_ok += 1
        step_type_counts.update(s["canonical"] for s in steps)
        n_steps_list.append(len(steps))
        n_matched_ings.append(sum(1 for s in steps if s["ingredients"]) / len(steps))
        n_with_temp.append(sum(1 for s in steps if s["temp_f"]) / len(steps))
        n_with_dur.append(sum(1 for s in steps if s["duration_min"]) / len(steps))

        fout.write(json.dumps({
            "id":      recipe["id"],
            "title":   recipe.get("title", ""),
            "steps":   steps,
            "edges":   edges,
            "n_nodes": len(steps),
            "n_edges": len(edges),
        }) + "\n")

print(f"\nDone: {n_ok:,} DAGs written, {n_skip:,} skipped (no recognizable steps)", flush=True)
print(f"Steps per recipe: mean={np.mean(n_steps_list):.1f}  median={np.median(n_steps_list):.1f}  max={max(n_steps_list)}", flush=True)
print(f"Fraction steps with ingredients: {np.mean(n_matched_ings):.1%}", flush=True)
print(f"Fraction steps with temp_f:      {np.mean(n_with_temp):.1%}", flush=True)
print(f"Fraction steps with duration:    {np.mean(n_with_dur):.1%}", flush=True)
print(f"Step type distribution: {dict(step_type_counts.most_common(8))}", flush=True)

import json as _json
stats = {
    "n_dags": n_ok,
    "n_skipped": n_skip,
    "steps_mean": float(np.mean(n_steps_list)),
    "steps_median": float(np.median(n_steps_list)),
    "frac_with_ingredients": float(np.mean(n_matched_ings)),
    "frac_with_temp": float(np.mean(n_with_temp)),
    "frac_with_duration": float(np.mean(n_with_dur)),
    "step_type_distribution": dict(step_type_counts.most_common()),
}
with open(BASE / "data/eval/fast_dag_stats.json", "w") as f:
    _json.dump(stats, f, indent=2)
print(f"\nSaved → {OUT}", flush=True)
print(f"Saved → data/eval/fast_dag_stats.json", flush=True)
