#!/usr/bin/env python3
"""
Phase 3: Parse Recipe1M train set into RecipeDAGs.

Pipeline per recipe:
  1. spaCy: extract verb lemmas from each instruction step
  2. BGE-large: embed step verbs, map to nearest canonical process type
  3. Build ProcessNode per step (canonical type + args extracted from text)
  4. Build IngredientNode per ingredient string (open vocabulary)
  5. Edges: ingredient → first step that mentions it; step_i → step_{i+1} (sequential)
  6. Save DAGs as JSON; build transition matrix for data-driven verifier

Outputs:
  data/dags/train_dags.jsonl       — one DAG per line (JSONL)
  data/dags/transition_matrix.json — P(process_B | process_A) over training DAGs
  data/dags/parse_stats.json       — coverage, mapping quality, step counts
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

RECIPE1M   = Path("/home/addeepak/AdvitResearch/cs348k/recipes.json")
OUT        = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/dags")
VOCAB_DIR  = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/vocab_analysis")
OUT.mkdir(parents=True, exist_ok=True)

# ── canonical process taxonomy (43 types) ────────────────────────────────────
TAXONOMY = [
    ("bake",      "dry_heat"),
    ("roast",     "dry_heat"),
    ("broil",     "dry_heat"),
    ("grill",     "dry_heat"),
    ("toast",     "dry_heat"),
    ("saute",     "fat_heat"),
    ("fry",       "fat_heat"),
    ("sear",      "fat_heat"),
    ("brown",     "fat_heat"),
    ("boil",      "moist_heat"),
    ("simmer",    "moist_heat"),
    ("steam",     "moist_heat"),
    ("poach",     "moist_heat"),
    ("braise",    "moist_heat"),
    ("blanch",    "moist_heat"),
    ("chop",      "prep"),
    ("dice",      "prep"),
    ("slice",     "prep"),
    ("mince",     "prep"),
    ("grate",     "prep"),
    ("peel",      "prep"),
    ("crush",     "prep"),
    ("mix",       "combine"),
    ("stir",      "combine"),
    ("whisk",     "combine"),
    ("fold",      "combine"),
    ("blend",     "combine"),
    ("beat",      "combine"),
    ("knead",     "combine"),
    ("toss",      "combine"),
    ("season",    "apply"),
    ("coat",      "apply"),
    ("brush",     "apply"),
    ("drizzle",   "apply"),
    ("cool",      "rest"),
    ("chill",     "rest"),
    ("freeze",    "rest"),
    ("rest",      "rest"),
    ("marinate",  "rest"),
    ("reduce",    "rest"),
    ("dissolve",  "rest"),
    ("melt",      "rest"),
    ("drain",     "rest"),
]
CANONICAL_NAMES = [t[0] for t in TAXONOMY]
GROUP_MAP = {t[0]: t[1] for t in TAXONOMY}

# Override table: BGE nearest-neighbor errors discovered in phase2b
OVERRIDE_MAP = {
    "refrigerate": "chill",
    "thaw":        "cool",
    "defrost":     "cool",
    "reheat":      "simmer",
    "microwave":   "bake",
    "pressure":    "boil",
    "deep-fry":    "fry",
    "deep fry":    "fry",
    "pan-fry":     "fry",
    "pan fry":     "fry",
    "stir-fry":    "saute",
    "stir fry":    "saute",
}

# Generic / structural verbs to skip (not cooking processes)
GENERIC_VERBS = {
    "be", "have", "do", "make", "use", "get", "go", "come", "take", "put",
    "let", "set", "keep", "give", "want", "need", "find", "look", "seem",
    "try", "call", "tell", "ask", "work", "feel", "leave", "bring", "move",
    "start", "continue", "ensure", "allow", "remain", "result", "yield",
    "note", "check", "watch", "test", "taste", "adjust", "enjoy", "serve",
    "prepare", "follow", "read", "turn", "place", "add", "remove", "transfer",
    "pour", "top", "fill", "line", "arrange", "layer",
}

# ── unit / quantity patterns for arg extraction ───────────────────────────────
_TIME_RE   = re.compile(r'(\d+(?:\.\d+)?)\s*(hour|hr|minute|min|second|sec)s?', re.I)
_TEMP_RE   = re.compile(r'(\d{2,4})\s*(?:degrees?|°)\s*([CF])', re.I)
_THICK_RE  = re.compile(r'(\d+(?:/\d+)?)\s*(inch|cm|mm|")', re.I)

def extract_args(text: str) -> dict:
    """Pull quantitative arguments from a step's raw text."""
    args = {}
    # duration
    mins = 0.0
    for m in _TIME_RE.finditer(text):
        val, unit = float(m.group(1)), m.group(2).lower()
        if unit.startswith("hour") or unit.startswith("hr"):
            mins += val * 60
        elif unit.startswith("min"):
            mins += val
        elif unit.startswith("sec"):
            mins += val / 60
    if mins:
        args["duration_min"] = round(mins, 1)
    # temperature
    m = _TEMP_RE.search(text)
    if m:
        val, scale = float(m.group(1)), m.group(2).upper()
        temp_c = val if scale == "C" else (val - 32) * 5 / 9
        args["temp_c"] = round(temp_c, 1)
    # thickness / size
    m = _THICK_RE.search(text)
    if m:
        args["size_note"] = m.group(0).strip()
    return args


def normalize_ingredient(text: str) -> str:
    """Strip quantities/units, lowercase, strip whitespace."""
    text = text.lower().strip()
    # Remove leading quantity patterns: "2 cups", "1/2 tsp", etc.
    text = re.sub(
        r'^\d+[\d/\.\s]*(cup|tsp|tbsp|tablespoon|teaspoon|oz|lb|g|kg|ml|l|pound|ounce|clove|piece|slice|can|package|package|bunch|head|stalk|sprig|pinch|dash|handful|drop)s?\s+',
        '', text, flags=re.I
    )
    return text.strip()


# ── spaCy setup ───────────────────────────────────────────────────────────────
print("Loading spaCy model...", flush=True)
NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat"])


def extract_verbs(doc) -> list[str]:
    verbs = []
    for token in doc:
        if token.pos_ == "VERB" and not token.is_stop:
            lemma = token.lemma_.lower().strip()
            if len(lemma) >= 3 and lemma.isalpha() and lemma not in GENERIC_VERBS:
                # Check override first
                if lemma in OVERRIDE_MAP:
                    verbs.append(OVERRIDE_MAP[lemma])
                else:
                    verbs.append(lemma)
    return verbs


# ── BGE-large embedding ───────────────────────────────────────────────────────
print("Loading BAAI/bge-large-en-v1.5 on GPU...", flush=True)
BGE = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

# Pre-embed all 43 canonical names once
canonical_embs = np.load(VOCAB_DIR / "process_canonical_embs.npy")   # (43, 1024)
assert canonical_embs.shape == (len(CANONICAL_NAMES), 1024), \
    f"Expected ({len(CANONICAL_NAMES)}, 1024), got {canonical_embs.shape}"

# Cache: verb lemma → canonical name (to avoid re-embedding duplicates)
verb_cache: dict[str, tuple[str, float]] = {}

def lemma_to_canonical(lemma: str) -> tuple[str, float]:
    """Map a verb lemma to its nearest canonical process type."""
    if lemma in verb_cache:
        return verb_cache[lemma]
    # If already canonical, return directly
    if lemma in CANONICAL_NAMES:
        verb_cache[lemma] = (lemma, 1.0)
        return lemma, 1.0
    # Check override
    if lemma in OVERRIDE_MAP:
        canon = OVERRIDE_MAP[lemma]
        verb_cache[lemma] = (canon, 1.0)
        return canon, 1.0
    # BGE embedding
    emb = BGE.encode([lemma], normalize_embeddings=True, convert_to_numpy=True)[0]
    sims = canonical_embs @ emb
    idx  = int(np.argmax(sims))
    verb_cache[lemma] = (CANONICAL_NAMES[idx], float(sims[idx]))
    return verb_cache[lemma]


def batch_lemmas_to_canonical(lemmas: list[str]) -> list[tuple[str, float]]:
    """Batch-encode any uncached lemmas, then return results."""
    uncached = [l for l in lemmas if l not in verb_cache and l not in CANONICAL_NAMES and l not in OVERRIDE_MAP]
    if uncached:
        embs = BGE.encode(uncached, normalize_embeddings=True, convert_to_numpy=True, batch_size=256)
        sims_all = embs @ canonical_embs.T           # (n_uncached, 43)
        idxs     = np.argmax(sims_all, axis=1)
        scores   = np.max(sims_all, axis=1)
        for lemma, idx, score in zip(uncached, idxs, scores):
            verb_cache[lemma] = (CANONICAL_NAMES[int(idx)], float(score))
    return [lemma_to_canonical(l) for l in lemmas]


# ── DAG building ──────────────────────────────────────────────────────────────

def build_dag(recipe: dict, step_docs: list) -> dict | None:
    """
    Build a RecipeDAG from one recipe.

    Returns a dict:
      {
        "id":    recipe id,
        "title": recipe title,
        "nodes": [
          {"id": "ing_0", "type": "ingredient", "name": "..."},
          {"id": "proc_0", "type": "process",
           "canonical": "bake", "group": "dry_heat",
           "sim": 0.92, "raw_verbs": ["bake","roast"], "args": {...},
           "step_text": "..."},
          ...
        ],
        "edges": [
          {"src": "ing_0", "dst": "proc_0", "label": "input"},
          {"src": "proc_0", "dst": "proc_1", "label": "seq"},
          ...
        ],
        "n_steps":     int,
        "n_ings":      int,
        "n_proc_nodes": int,
      }
    Returns None if recipe has no parseable process nodes.
    """
    rid   = recipe.get("id", "")
    title = recipe.get("title", "")
    steps = [s["text"] for s in recipe.get("instructions", []) if s["text"].strip()]
    ings  = [item["text"] for item in recipe.get("ingredients", []) if item["text"].strip()]

    if not steps:
        return None

    nodes = []
    edges = []
    node_id_counter = [0]

    def new_id(prefix):
        nid = f"{prefix}_{node_id_counter[0]}"
        node_id_counter[0] += 1
        return nid

    # ── Ingredient nodes ──────────────────────────────────────────────────
    ing_nodes = []
    for raw_ing in ings:
        norm_ing = normalize_ingredient(raw_ing)
        nid = new_id("ing")
        nodes.append({"id": nid, "type": "ingredient", "name": norm_ing, "raw": raw_ing})
        ing_nodes.append(nid)

    # Build lowercase ingredient name set for substring matching
    ing_names_lower = [normalize_ingredient(i).lower() for i in ings]

    # ── Process nodes (one per step) ─────────────────────────────────────
    proc_nodes = []
    for step_idx, (step_text, doc) in enumerate(zip(steps, step_docs)):
        raw_verbs = extract_verbs(doc)
        if not raw_verbs:
            # No cooking verbs found; skip this step
            continue

        # Map all verbs in this step → canonical types
        canon_results = batch_lemmas_to_canonical(raw_verbs)

        # Pick the dominant canonical type (highest frequency among verbs in step,
        # breaking ties by highest similarity score)
        type_scores: dict[str, list[float]] = defaultdict(list)
        for _, (canon, sim) in zip(raw_verbs, canon_results):
            type_scores[canon].append(sim)

        # Score = count × mean_sim for each candidate canonical type
        best_canon = max(type_scores.items(),
                         key=lambda kv: (len(kv[1]), sum(kv[1]) / len(kv[1])))[0]
        best_sim   = max(type_scores[best_canon])

        args = extract_args(step_text)

        nid = new_id("proc")
        nodes.append({
            "id":        nid,
            "type":      "process",
            "canonical": best_canon,
            "group":     GROUP_MAP[best_canon],
            "sim":       round(best_sim, 4),
            "raw_verbs": raw_verbs[:5],  # keep top-5 for inspection
            "args":      args,
            "step_idx":  step_idx,
            "step_text": step_text[:200],
        })
        proc_nodes.append(nid)

    if not proc_nodes:
        return None

    # ── Sequential edges between process nodes ────────────────────────────
    for i in range(len(proc_nodes) - 1):
        edges.append({"src": proc_nodes[i], "dst": proc_nodes[i + 1], "label": "seq"})

    # ── Ingredient → process edges ────────────────────────────────────────
    # For each ingredient, connect it to the FIRST process node that mentions it
    # (substring match in step_text).
    for ing_nid, ing_name in zip(ing_nodes, ing_names_lower):
        short = ing_name.split(",")[0].strip()   # e.g. "butter" from "butter, unsalted"
        if len(short) < 3:
            short = ing_name
        connected = False
        for proc_nid in proc_nodes:
            proc_node = next(n for n in nodes if n["id"] == proc_nid)
            if short in proc_node["step_text"].lower():
                edges.append({"src": ing_nid, "dst": proc_nid, "label": "input"})
                connected = True
                break
        if not connected:
            # Connect to first process node as fallback
            edges.append({"src": ing_nid, "dst": proc_nodes[0], "label": "input_fallback"})

    n_proc  = len(proc_nodes)
    n_ings  = len(ing_nodes)

    return {
        "id":          rid,
        "title":       title,
        "nodes":       nodes,
        "edges":       edges,
        "n_steps":     len(steps),
        "n_ings":      n_ings,
        "n_proc_nodes": n_proc,
    }


# ── Transition matrix ─────────────────────────────────────────────────────────

def build_transition_matrix(dags: list[dict]) -> dict:
    """
    Compute P(B | A) = count(A→B) / count(A) from all sequential process edges.
    """
    co_counts: Counter = Counter()   # (A, B) pair counts
    out_counts: Counter = Counter()  # A count (outgoing)

    for dag in dags:
        node_map = {n["id"]: n for n in dag["nodes"]}
        for edge in dag["edges"]:
            if edge["label"] != "seq":
                continue
            src_node = node_map.get(edge["src"])
            dst_node = node_map.get(edge["dst"])
            if src_node and dst_node:
                A = src_node.get("canonical")
                B = dst_node.get("canonical")
                if A and B:
                    co_counts[(A, B)] += 1
                    out_counts[A] += 1

    # Build matrix dict
    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for (A, B), cnt in co_counts.items():
        matrix[A][B] = cnt / out_counts[A]

    return {
        "transition_probs":  dict(matrix),
        "co_counts":         {f"{a}|{b}": c for (a, b), c in co_counts.most_common()},
        "total_transitions": sum(co_counts.values()),
        "n_canonical_types": len(CANONICAL_NAMES),
        "canonical_names":   CANONICAL_NAMES,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {RECIPE1M}...", flush=True)
    with open(RECIPE1M) as f:
        all_recipes = json.load(f)
    train = [r for r in all_recipes if r.get("partition") == "train"]
    print(f"  {len(train):,} train recipes")

    # ── Collect ALL steps for batch spaCy parse ────────────────────────────
    print("\nCollecting instruction steps...", flush=True)
    step_offsets = []   # (recipe_idx, list of step indices into all_steps)
    all_steps    = []
    for r_idx, recipe in enumerate(train):
        steps = [s["text"] for s in recipe.get("instructions", []) if s["text"].strip()]
        start = len(all_steps)
        all_steps.extend(steps)
        step_offsets.append((r_idx, start, len(all_steps)))
    print(f"  {len(all_steps):,} steps across {len(train):,} recipes", flush=True)

    # ── spaCy batch parse ──────────────────────────────────────────────────
    print("\nParsing with spaCy (batch=256)...", flush=True)
    BATCH = 256
    all_docs = []
    for i in tqdm(range(0, len(all_steps), BATCH), desc="spaCy"):
        all_docs.extend(list(NLP.pipe(all_steps[i:i+BATCH])))
    print(f"  {len(all_docs):,} docs parsed", flush=True)

    # ── Pre-collect unique verb lemmas for bulk BGE encoding ──────────────
    print("\nCollecting unique verb lemmas for bulk embedding...", flush=True)
    unique_lemmas: set[str] = set()
    for doc in all_docs:
        for token in doc:
            if token.pos_ == "VERB" and not token.is_stop:
                lemma = token.lemma_.lower().strip()
                if len(lemma) >= 3 and lemma.isalpha() and lemma not in GENERIC_VERBS:
                    if lemma not in CANONICAL_NAMES and lemma not in OVERRIDE_MAP:
                        unique_lemmas.add(lemma)
    unique_lemmas_list = sorted(unique_lemmas)
    print(f"  {len(unique_lemmas_list):,} unique verb lemmas to embed", flush=True)

    print("\nBulk-encoding verb lemmas with BGE-large...", flush=True)
    if unique_lemmas_list:
        lemma_embs = BGE.encode(
            unique_lemmas_list, batch_size=256,
            normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=True,
        )
        sims_all = lemma_embs @ canonical_embs.T    # (n_lemmas, 43)
        idxs_all = np.argmax(sims_all, axis=1)
        scores_all = np.max(sims_all, axis=1)
        for lemma, idx, score in zip(unique_lemmas_list, idxs_all, scores_all):
            verb_cache[lemma] = (CANONICAL_NAMES[int(idx)], float(score))
    print(f"  verb_cache populated: {len(verb_cache):,} entries", flush=True)

    # ── Parse recipes into DAGs ────────────────────────────────────────────
    print("\nBuilding DAGs...", flush=True)
    dags = []
    n_skipped = 0
    out_file  = OUT / "train_dags.jsonl"

    process_type_counter: Counter = Counter()
    step_counts: list[int]        = []
    proc_counts: list[int]        = []
    sim_scores:  list[float]      = []

    with open(out_file, "w") as fout:
        for r_idx, start, end in tqdm(step_offsets, desc="build DAG"):
            recipe    = train[r_idx]
            step_docs = all_docs[start:end]
            dag       = build_dag(recipe, step_docs)
            if dag is None:
                n_skipped += 1
                continue
            dags.append(dag)
            fout.write(json.dumps(dag) + "\n")

            step_counts.append(dag["n_steps"])
            proc_counts.append(dag["n_proc_nodes"])
            for node in dag["nodes"]:
                if node["type"] == "process":
                    process_type_counter[node["canonical"]] += 1
                    sim_scores.append(node["sim"])

    n_parsed = len(dags)
    print(f"\n  Parsed: {n_parsed:,}  /  {len(train):,}  ({n_parsed/len(train):.1%})")
    print(f"  Skipped (no process verbs): {n_skipped:,}")
    print(f"  Avg steps/recipe:           {np.mean(step_counts):.1f}")
    print(f"  Avg proc nodes/recipe:      {np.mean(proc_counts):.1f}")
    print(f"  Avg mapping sim:            {np.mean(sim_scores):.3f}")
    print(f"  % sim > 0.6:                {(np.array(sim_scores) > 0.6).mean():.1%}")

    print("\n  Process type frequencies (top 20):")
    for ptype, cnt in process_type_counter.most_common(20):
        bar = "█" * (cnt // 500)
        print(f"    {cnt:7,}  {ptype:<15} {bar}")

    # ── Transition matrix ──────────────────────────────────────────────────
    print("\nBuilding transition matrix...", flush=True)
    trans = build_transition_matrix(dags)
    print(f"  Total transitions: {trans['total_transitions']:,}")

    with open(OUT / "transition_matrix.json", "w") as f:
        json.dump(trans, f, indent=2)
    print(f"  Saved → {OUT}/transition_matrix.json")

    # ── Parse stats ────────────────────────────────────────────────────────
    stats = {
        "n_train_recipes":    len(train),
        "n_parsed":           n_parsed,
        "n_skipped":          n_skipped,
        "parse_rate":         n_parsed / len(train),
        "avg_steps":          float(np.mean(step_counts)),
        "avg_proc_nodes":     float(np.mean(proc_counts)),
        "avg_mapping_sim":    float(np.mean(sim_scores)),
        "pct_sim_gt_0_6":     float((np.array(sim_scores) > 0.6).mean()),
        "process_type_freq":  dict(process_type_counter.most_common()),
        "verb_cache_size":    len(verb_cache),
    }
    with open(OUT / "parse_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved → {OUT}/parse_stats.json")
    print(f"  DAGs → {out_file}")

    # ── Print sample transition probabilities ─────────────────────────────
    print("\n  Sample transition probabilities (top 5 successors for common types):")
    for ptype in ["chop", "saute", "simmer", "bake", "mix"]:
        row = trans["transition_probs"].get(ptype, {})
        if row:
            top = sorted(row.items(), key=lambda x: -x[1])[:5]
            print(f"    {ptype} → {top}")

    print("\nDone.")


if __name__ == "__main__":
    main()
