#!/usr/bin/env python3
"""
Build enriched dataset joining Recipe1M + food.com Kaggle data.

Joins by food.com recipe ID (extracted from URL in layer1.json):
  - Cuisine tags, diet tags, difficulty, cooking time (from food.com structured tags)
  - Mean user rating + review count (from RAW_interactions.csv)
  - Nutrition: [calories, fat, sugar, sodium, protein, sat_fat, carbs]
  - Process features: keyword step-type histogram + step count

Result: 228K recipes with ground-truth ratings + cuisine + rich process features.

Outputs:
  data/enriched/joined_recipes.jsonl   — full enriched records
  data/enriched/features.npz           — (N, D) feature matrices
  data/eval/enriched_dataset_stats.json
"""

import json, os, re, ast
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd

BASE    = Path(__file__).resolve().parent.parent
LAYER1  = Path(os.environ.get("RECIPE1M", str(BASE.parent / "layer1.json")))
FC_DIR  = Path(os.environ.get("FOODCOM_DIR", str(BASE.parent / "foodcom_data")))
OUT_DIR = BASE / "data/enriched"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Step-type keyword rules ───────────────────────────────────────────────────

STEP_RULES = [
    ("knead",    ["knead"]),
    ("marinate", ["marinate", "marinade"]),
    ("grill",    ["grill", "barbecue", "char-grill", "griddle"]),
    ("steam",    ["steam"]),
    ("reduce",   ["reduce", "cook down", "thicken"]),
    ("whisk",    ["whisk", "beat", "whip"]),
    ("blend",    ["blend", "puree", "food processor", "blender"]),
    ("bake",     ["bake", "roast", "broil", " oven"]),
    ("simmer",   ["simmer", "stew", "braise"]),
    ("boil",     ["boil", "blanch"]),
    ("saute",    ["saute", "sauté", "stir-fry", "pan-fry", " fry "]),
    ("chop",     ["chop", "dice", "mince", "slice", " cut "]),
    ("chill",    ["chill", "refrigerate", "freeze", " cool"]),
    ("season",   ["season", "salt and pepper"]),
    ("mix",      ["mix", "stir", "combine", "toss", "fold", "add"]),
]
STEP_TYPES = [s for s, _ in STEP_RULES]
N_TYPES    = len(STEP_TYPES)

def step_histogram(instructions):
    counts = np.zeros(N_TYPES, dtype=np.float32)
    for step in instructions:
        text = step.get("text", "").lower()
        for idx, (_, kws) in enumerate(STEP_RULES):
            if any(kw in text for kw in kws):
                counts[idx] += 1
                break
    total = counts.sum()
    return counts / total if total > 0 else counts

def parse_time_minutes(instructions):
    """Extract total cooking time from instruction text (rough heuristic)."""
    total = 0
    for step in instructions:
        text = step.get("text", "").lower()
        for m in re.finditer(r'(\d+)\s*(?:to\s*\d+\s*)?(?:hour|hr)', text):
            total += int(m.group(1)) * 60
        for m in re.finditer(r'(\d+)\s*(?:to\s*\d+\s*)?min', text):
            total += int(m.group(1))
    return min(total, 600)  # cap at 10 hours

# ── Cuisine / diet / tag taxonomy ────────────────────────────────────────────

CUISINE_TAGS = {
    "italian":       ["italian"],
    "mexican":       ["mexican"],
    "asian":         ["chinese", "japanese", "korean", "thai", "vietnamese",
                      "asian", "cambodian"],
    "indian":        ["indian"],
    "american":      ["american", "north-american"],
    "french":        ["french", "european"],
    "mediterranean": ["greek", "mediterranean", "middle-eastern", "moroccan"],
}

DIET_TAGS     = ["vegetarian", "vegan", "low-fat", "low-carb", "low-sodium",
                 "low-calorie", "low-protein", "low-cholesterol", "diabetic",
                 "healthy", "gluten-free"]
DIFFICULTY_TAGS = ["easy", "beginner-cook", "advanced-cook"]
TIME_TAGS       = ["15-minutes-or-less", "30-minutes-or-less",
                   "60-minutes-or-less", "1-to-4-hours", "4-to-5-hours"]
COURSE_TAGS     = ["main-dish", "side-dishes", "desserts", "appetizers",
                   "breakfast", "salads", "soups-stews", "beverages", "snacks"]

def parse_tags(tag_str):
    try:
        tags = ast.literal_eval(tag_str)
    except Exception:
        return {}, [], None, None, None
    tag_set = set(tags)

    cuisine = None
    for c, kws in CUISINE_TAGS.items():
        if any(kw in tag_set for kw in kws):
            cuisine = c
            break

    diet  = [t for t in DIET_TAGS if t in tag_set]
    diff  = next((t for t in DIFFICULTY_TAGS if t in tag_set), None)
    time_tag = next((t for t in TIME_TAGS if t in tag_set), None)
    course = next((t for t in COURSE_TAGS if t in tag_set), None)
    return cuisine, diet, diff, time_tag, course

def parse_nutrition(nutr_str):
    """Returns [calories, fat_pct, sugar_pct, sodium_pct, protein_pct, sat_fat_pct, carbs_pct]"""
    try:
        v = ast.literal_eval(nutr_str)
        return [float(x) for x in v[:7]]
    except Exception:
        return [0.0] * 7

# ── Load food.com Kaggle data ─────────────────────────────────────────────────

print("Loading food.com Kaggle data...", flush=True)
fc_recipes = pd.read_csv(FC_DIR / "RAW_recipes.csv")
fc_inter   = pd.read_csv(FC_DIR / "RAW_interactions.csv")

mean_ratings = (fc_inter.groupby("recipe_id")["rating"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"recipe_id":"id","mean":"mean_rating",
                                 "count":"n_reviews"}))
fc_joined = fc_recipes.merge(mean_ratings, on="id", how="left")
fc_joined["mean_rating"] = fc_joined["mean_rating"].fillna(0)
fc_joined["n_reviews"]   = fc_joined["n_reviews"].fillna(0).astype(int)

# Index by id
fc_by_id = fc_joined.set_index("id").to_dict("index")
print(f"  food.com recipes: {len(fc_by_id):,}", flush=True)

# ── Load Recipe1M and join ────────────────────────────────────────────────────

print("Loading Recipe1M...", flush=True)
with open(LAYER1) as f:
    layer1 = json.load(f)
print(f"  Recipe1M records: {len(layer1):,}", flush=True)

ID_PAT = re.compile(r'food\.com/recipe/(?:[^/]*?-)?(\d+)')

print("Joining...", flush=True)
joined = []
missing = 0
for r in layer1:
    m = ID_PAT.search(r.get("url", ""))
    if not m:
        missing += 1
        continue
    fc_id = int(m.group(1))
    if fc_id not in fc_by_id:
        missing += 1
        continue
    fc = fc_by_id[fc_id]

    instructions = r.get("instructions", [])
    ingredients  = r.get("ingredients", [])

    cuisine, diet, difficulty, time_tag, course = parse_tags(str(fc.get("tags","")))
    nutrition = parse_nutrition(str(fc.get("nutrition", "")))
    hist      = step_histogram(instructions)
    ing_text  = " ".join(i["text"] for i in ingredients)
    instr_time = parse_time_minutes(instructions)

    joined.append({
        "id":          r["id"],
        "fc_id":       fc_id,
        "title":       r["title"],
        "partition":   r.get("partition"),
        "ing_text":    ing_text,
        "n_ingredients": len(ingredients),
        "n_steps":     len(instructions),
        "instr_time_min": instr_time,
        "fc_minutes":  float(fc.get("minutes", 0) or 0),
        "mean_rating": float(fc.get("mean_rating", 0)),
        "n_reviews":   int(fc.get("n_reviews", 0)),
        "cuisine":     cuisine,
        "diet":        diet,
        "difficulty":  difficulty,
        "time_tag":    time_tag,
        "course":      course,
        "nutrition":   nutrition,
        "step_hist":   hist.tolist(),
    })

print(f"  Joined: {len(joined):,} recipes  ({missing:,} without food.com ID)", flush=True)

# ── Stats ─────────────────────────────────────────────────────────────────────

n_rated   = sum(1 for r in joined if r["n_reviews"] >= 3)
n_cuisine = sum(1 for r in joined if r["cuisine"])
ratings   = [r["mean_rating"] for r in joined if r["n_reviews"] >= 3]
print(f"\n  With ≥3 reviews: {n_rated:,}", flush=True)
print(f"  With cuisine tag: {n_cuisine:,}", flush=True)
print(f"  Mean rating (≥3 reviews): {np.mean(ratings):.3f} ± {np.std(ratings):.3f}", flush=True)
cuisine_dist = Counter(r["cuisine"] for r in joined if r["cuisine"])
print(f"  Cuisine distribution: {dict(cuisine_dist.most_common())}", flush=True)

# ── Save enriched records ─────────────────────────────────────────────────────

out_path = OUT_DIR / "joined_recipes.jsonl"
with open(out_path, "w") as f:
    for r in joined:
        f.write(json.dumps(r) + "\n")
print(f"\nSaved → data/enriched/joined_recipes.jsonl ({len(joined):,} records)", flush=True)

# ── Build feature matrices ─────────────────────────────────────────────────────
# For downstream experiments, save structured numpy arrays

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Filter to usable records (≥3 reviews so rating is meaningful)
usable = [r for r in joined if r["n_reviews"] >= 3]
print(f"\nBuilding feature matrices for {len(usable):,} rated recipes...", flush=True)

ids        = [r["id"] for r in usable]
ratings_np = np.array([r["mean_rating"] for r in usable], dtype=np.float32)
step_hists = np.array([r["step_hist"] for r in usable], dtype=np.float32)
nutrition  = np.array([r["nutrition"] for r in usable], dtype=np.float32)

meta = np.array([
    [r["n_steps"], r["n_ingredients"], r["instr_time_min"],
     min(r["fc_minutes"], 600)]
    for r in usable
], dtype=np.float32)

ing_texts = [r["ing_text"] for r in usable]
print("  Fitting TF-IDF on ingredient text...", flush=True)
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2),
                        sublinear_tf=True, min_df=5)
X_ing = tfidf.fit_transform(ing_texts).toarray().astype(np.float32)
print(f"  X_ing: {X_ing.shape}", flush=True)

# Save
np.savez_compressed(
    OUT_DIR / "features.npz",
    ids=np.array(ids),
    ratings=ratings_np,
    step_hists=step_hists,
    nutrition=nutrition,
    meta=meta,
    X_ing=X_ing,
)
print(f"Saved → data/enriched/features.npz", flush=True)

# Save id→cuisine map for later
cuisine_map = {r["id"]: r["cuisine"] for r in usable if r["cuisine"]}
with open(OUT_DIR / "cuisine_map.json", "w") as f:
    json.dump(cuisine_map, f)

# Save vocab for TF-IDF
import pickle
with open(OUT_DIR / "tfidf_vocab.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Stats JSON
stats = {
    "n_joined": len(joined),
    "n_rated_3plus": len(usable),
    "n_with_cuisine": n_cuisine,
    "rating_mean": float(np.mean(ratings)),
    "rating_std":  float(np.std(ratings)),
    "rating_quantiles": {
        "10": float(np.quantile(ratings, .10)),
        "25": float(np.quantile(ratings, .25)),
        "50": float(np.quantile(ratings, .50)),
        "75": float(np.quantile(ratings, .75)),
        "90": float(np.quantile(ratings, .90)),
    },
    "cuisine_distribution": dict(cuisine_dist.most_common()),
    "step_type_names": STEP_TYPES,
    "feature_shapes": {
        "X_ing": list(X_ing.shape),
        "step_hists": list(step_hists.shape),
        "nutrition": list(nutrition.shape),
        "meta": list(meta.shape),
    }
}
with open(BASE / "data/eval/enriched_dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"Saved → data/eval/enriched_dataset_stats.json", flush=True)
print("\nDone.", flush=True)
