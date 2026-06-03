#!/usr/bin/env python3
"""
Cache MiniLM-L6 embeddings for instruction step texts across 93K recipes.

Loads ORIGINAL instruction text from layer1.json (the reconstruction target),
aligned to the steps that survived fast-parsing into fast_dags.jsonl.

For each DAG step we store the original instruction text that produced it,
enabling the VAE to learn to compress real recipe language, not just
canonical-type strings.

Outputs:
  data/models/embeddings/step_embs.npy        — (N_steps, 384) float16, unit-normed
  data/models/embeddings/step_index.json      — {recipe_id: [start_idx, end_idx]}
  data/models/embeddings/recipe_ids.json      — ordered list of recipe IDs (93K)
  data/models/embeddings/step_texts.json      — flat list of original step texts (for uplift)
  data/models/embeddings/ing_embs.npy         — (N_recipes, 384) ingredient embeddings

Runtime: ~20 min on GPU.
"""

import json, os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE    = Path(__file__).resolve().parent.parent
LAYER1  = Path(os.environ.get("RECIPE1M", str(BASE.parent / "layer1.json")))
DAGS    = BASE / "data/dags/fast_dags.jsonl"
OUT_DIR = BASE / "data/models/embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH   = 2048
MODEL   = "sentence-transformers/all-MiniLM-L6-v2"

# ── Load fast DAGs to know which steps survived parsing ────────────────────────
# For each recipe, we need: which step indices in the original recipe correspond
# to which canonical steps in the DAG?
# parse_dags_fast.py iterates instructions in order and appends steps where
# classify_step() returns non-None. We replay that logic here.

import re, sys
sys.path.insert(0, str(BASE / "scripts"))

# Inline step classifier (same rules as parse_dags_fast.py)
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

def classify_step(text):
    tl = text.lower()
    for canonical, keywords, context_pats in STEP_RULES:
        if any(kw in tl for kw in keywords):
            return canonical
    for canonical, keywords, context_pats in STEP_RULES:
        if any(re.search(pat, tl) for pat in context_pats):
            return canonical
    return None


# ── Build set of recipe IDs in fast DAG corpus ────────────────────────────────
print("Loading fast DAG recipe IDs...", flush=True)
dag_ids = set()
with open(DAGS) as f:
    for line in f:
        dag_ids.add(json.loads(line)["id"])
print(f"  {len(dag_ids):,} recipes in fast DAG corpus", flush=True)

# ── Stream layer1.json and collect step texts ──────────────────────────────────
print("Loading layer1.json and extracting step texts...", flush=True)
recipe_ids   = []
step_texts   = []   # flat list of ALL step texts (original language)
step_index   = {}   # recipe_id → [start, end]
ing_sentences = []  # one sentence per recipe for ingredient embedding

with open(LAYER1) as f:
    all_recipes = json.load(f)

for recipe in tqdm(all_recipes, desc="extract steps"):
    rid = recipe["id"]
    if rid not in dag_ids:
        continue

    instructions = recipe.get("instructions", [])
    ingredients  = recipe.get("ingredients", [])

    # Replay classifier to get only the steps that survived parsing
    surviving_texts = []
    for step in instructions:
        text = step.get("text", "").strip()
        if not text:
            continue
        if classify_step(text) is not None:
            surviving_texts.append(text[:300])   # cap length

    if not surviving_texts:
        continue

    # Ingredient sentence
    ing_text = ", ".join(i.get("text","").split(",")[0].strip()
                         for i in ingredients[:15] if i.get("text",""))

    start = len(step_texts)
    step_texts.extend(surviving_texts)
    step_index[rid] = [start, len(step_texts)]
    recipe_ids.append(rid)
    ing_sentences.append(ing_text)

print(f"  {len(recipe_ids):,} recipes kept, {len(step_texts):,} step texts", flush=True)

# ── Load SentenceTransformer ───────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nLoading MiniLM on {device}...", flush=True)
smodel = SentenceTransformer(MODEL, device=device)
smodel.eval()

# ── Embed step texts ───────────────────────────────────────────────────────────
print("Embedding step texts...", flush=True)
step_embs = smodel.encode(
    step_texts,
    batch_size=BATCH,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
    precision="float32",
)
print(f"  step_embs: {step_embs.shape}  ({step_embs.nbytes/1e9:.2f} GB)", flush=True)

# ── Embed ingredient sentences ─────────────────────────────────────────────────
print("Embedding ingredient lists...", flush=True)
ing_embs = smodel.encode(
    ing_sentences,
    batch_size=BATCH,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
    precision="float32",
)
print(f"  ing_embs:  {ing_embs.shape}", flush=True)

# ── Save ───────────────────────────────────────────────────────────────────────
np.save(OUT_DIR / "step_embs.npy", step_embs)
np.save(OUT_DIR / "ing_embs.npy",  ing_embs)
with open(OUT_DIR / "step_index.json", "w") as f:
    json.dump(step_index, f)
with open(OUT_DIR / "recipe_ids.json", "w") as f:
    json.dump(recipe_ids, f)
with open(OUT_DIR / "step_texts.json", "w") as f:
    json.dump(step_texts, f)

print(f"\nSaved to {OUT_DIR}/")
print(f"  step_embs.npy  {step_embs.shape}")
print(f"  ing_embs.npy   {ing_embs.shape}")
print(f"  step_index.json  ({len(step_index)} recipes)")
print(f"  step_texts.json  ({len(step_texts)} texts)")
print("Done.", flush=True)
