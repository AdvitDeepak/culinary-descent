#!/usr/bin/env python3
"""
Phase 1: Recipe Vocabulary Coverage Curve
------------------------------------------
Answers: what vocabulary size K covers what % of Recipe1M?

Pipeline:
  1. Extract ingredients + operations from all train recipes (no LLM)
  2. Embed ingredients with a sentence transformer (GPU) → better clustering
  3. For each K: greedily pick top-K types, measure % of recipes fully covered
  4. Plot the rate-coverage curve → find the knee

Outputs (data/vocab_analysis/):
  ingredient_counts.json    raw normalized ingredient frequencies
  operation_counts.json     raw normalized operation frequencies
  ingredient_embeddings.npy sentence-transformer embeddings of unique ingredients
  coverage_curve_ops.png    operation-only coverage curve
  coverage_curve_ing.png    ingredient-only coverage curve
  coverage_curve_joint.png  joint curve (ops fixed at 95%, vary ingredients)
  coverage_results.json     all curve data points
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

RECIPE1M = Path("/home/addeepak/AdvitResearch/cs348k/recipes.json")
OUT = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/vocab_analysis")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ingredient normalization
# ---------------------------------------------------------------------------

# Recipe1M uses USDA format: "food_name, descriptor, descriptor, ..."
# We take the part before the first comma as the core ingredient name.

_UNITS = {
    'cup', 'cups', 'tablespoon', 'tablespoons', 'tbsp', 'tbsps',
    'teaspoon', 'teaspoons', 'tsp', 'tsps', 'oz', 'ounce', 'ounces',
    'lb', 'lbs', 'pound', 'pounds', 'g', 'gram', 'grams', 'kg',
    'ml', 'liter', 'liters', 'quart', 'quarts', 'pint', 'pints',
    'gallon', 'gallons', 'can', 'cans', 'package', 'packages', 'pkg',
    'slice', 'slices', 'piece', 'pieces', 'clove', 'cloves',
    'head', 'heads', 'bunch', 'bunches', 'stalk', 'stalks',
    'sprig', 'sprigs', 'pinch', 'dash', 'handful',
}
_STOP = {
    'and', 'or', 'to', 'with', 'without', 'of', 'the', 'a', 'an',
    'in', 'for', 'from', 'at', 'by', 'as', 'on',
}

def normalize_ingredient(text: str) -> str:
    text = text.lower().strip()
    core = text.split(',')[0].strip()                 # drop USDA descriptors
    core = re.sub(r'\d[\d\s./\-]*', '', core).strip() # drop quantities
    words = [w for w in core.split()
             if w not in _UNITS and w not in _STOP and len(w) >= 2]
    result = ' '.join(words).strip()
    return result if result else core[:30]


# ---------------------------------------------------------------------------
# Operation extraction
# ---------------------------------------------------------------------------

_COOKING_VERBS = {
    # dry heat
    'bake', 'roast', 'broil', 'grill', 'toast', 'saute', 'fry', 'sear', 'brown',
    # moist heat
    'boil', 'simmer', 'steam', 'poach', 'braise', 'blanch',
    # prep
    'chop', 'slice', 'dice', 'mince', 'grate', 'shred', 'peel', 'trim', 'cut',
    # combine
    'mix', 'stir', 'combine', 'toss', 'blend', 'whisk', 'beat', 'fold',
    'knead', 'puree', 'cream', 'mash',
    # finish
    'season', 'garnish', 'serve', 'plate', 'drizzle',
    # other common
    'marinate', 'drain', 'strain', 'rinse', 'cook', 'heat', 'cool', 'chill',
    'cover', 'rest', 'melt', 'dissolve', 'reduce', 'caramelize', 'deglaze',
    'smoke', 'cure', 'pickle', 'soak', 'infuse', 'coat', 'brush', 'spread',
    'roll', 'shape', 'press', 'squeeze', 'add', 'pour', 'remove', 'transfer',
}

_OP_SYNONYMS = {
    'sauté': 'saute', 'pan-fry': 'fry', 'stir-fry': 'fry', 'deep-fry': 'fry',
    'whip': 'whisk', 'process': 'blend', 'shred': 'grate', 'julienne': 'slice',
    'cube': 'dice', 'halve': 'cut', 'quarter': 'cut', 'blanch': 'boil',
    'stew': 'braise', 'refrigerate': 'chill', 'freeze': 'chill',
    'warm': 'heat', 'plate': 'serve', 'sprinkle': 'season', 'top': 'garnish',
    'pour': 'add', 'place': 'add', 'transfer': 'remove', 'strain': 'drain',
    'rinse': 'drain', 'wrap': 'cover', 'let': 'rest', 'set': 'rest',
    'dissolve': 'melt', 'coat': 'spread', 'brush': 'spread',
    'caramelize': 'brown', 'soak': 'marinate', 'infuse': 'marinate',
    'steep': 'marinate', 'cure': 'marinate', 'pickle': 'marinate',
}

def extract_ops(text: str) -> list[str]:
    words = re.findall(r'\b[a-z\-]+\b', text.lower())
    seen, ops = set(), []
    for w in words:
        canon = _OP_SYNONYMS.get(w, w)
        if canon in _COOKING_VERBS and canon not in seen:
            ops.append(canon)
            seen.add(canon)
    return ops


# ---------------------------------------------------------------------------
# Greedy coverage curve
# ---------------------------------------------------------------------------

def coverage_at_k(recipe_item_sets: list[set], vocab: set) -> float:
    covered = sum(1 for s in recipe_item_sets if s.issubset(vocab))
    return covered / len(recipe_item_sets)


def greedy_curve(
    recipe_sets: list[set],
    freq: Counter,
    k_values: list[int],
) -> list[tuple[int, float]]:
    """
    For each K in k_values, take top-K items by frequency (greedy),
    compute % of recipes fully covered.
    """
    sorted_items = [item for item, _ in freq.most_common()]
    points = []
    for k in k_values:
        vocab = set(sorted_items[:k])
        cov = coverage_at_k(recipe_sets, vocab)
        points.append((k, cov))
    return points


# ---------------------------------------------------------------------------
# Embedding-based coverage (uses GPU sentence-transformers)
# ---------------------------------------------------------------------------

def embed_ingredients(unique_ings: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print("  Loading sentence-transformer (all-MiniLM-L6-v2) on GPU...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    print(f"  Encoding {len(unique_ings):,} unique ingredient strings...")
    embeddings = model.encode(
        unique_ings,
        batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def embedding_coverage_curve(
    recipe_ing_sets: list[set[str]],
    unique_ings: list[str],
    embeddings: np.ndarray,
    k_values: list[int],
) -> list[tuple[int, float]]:
    """
    For each K: k-means cluster ingredients into K groups,
    map each recipe's ingredients to cluster IDs,
    measure % of recipes fully covered (all cluster IDs present in vocab).

    This is "can we represent this recipe with K ingredient categories?"
    """
    from sklearn.cluster import MiniBatchKMeans

    ing_to_idx = {ing: i for i, ing in enumerate(unique_ings)}

    # Pre-map recipe sets to embedding indices
    recipe_idx_sets = []
    for s in recipe_ing_sets:
        idxs = {ing_to_idx[ing] for ing in s if ing in ing_to_idx}
        recipe_idx_sets.append(idxs)

    points = []
    for k in k_values:
        print(f"  K={k}: clustering...", end=' ', flush=True)
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096, n_init=3)
        labels = km.fit_predict(embeddings)  # label for each unique ingredient

        # Map each recipe's ingredient indices → cluster labels → set of clusters used
        recipe_cluster_sets = [
            {labels[i] for i in idxs} for idxs in recipe_idx_sets
        ]
        # "Covered" = recipe uses a subset of {0..K-1}, always true, so coverage=100%
        # Better question: for each cluster, does the recipe need that cluster?
        # Actually the right metric: with K clusters, ALL ingredients map to a cluster,
        # so 100% of recipes are "representable" — the distortion is the info lost.
        #
        # More useful: compute average intra-cluster distance (distortion) vs K
        distortion = km.inertia_ / len(unique_ings)
        print(f"distortion={distortion:.4f}")
        points.append((k, distortion))

    return points


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {RECIPE1M}...")
    with open(RECIPE1M) as f:
        all_recipes = json.load(f)
    train = [r for r in all_recipes if r.get('partition') == 'train']
    print(f"  {len(train):,} train recipes")

    # ---- Extract ----
    print("\nExtracting ingredients and operations...")
    ing_counts: Counter = Counter()
    op_counts: Counter = Counter()
    recipe_ing_sets: list[set] = []
    recipe_op_sets: list[set] = []

    for recipe in tqdm(train, desc="parsing"):
        ings = set()
        for item in recipe.get('ingredients', []):
            n = normalize_ingredient(item['text'])
            if n:
                ings.add(n)
                ing_counts[n] += 1
        recipe_ing_sets.append(ings)

        ops = set()
        for step in recipe.get('instructions', []):
            for op in extract_ops(step['text']):
                ops.add(op)
                op_counts[op] += 1
        recipe_op_sets.append(ops)

    print(f"\n  Unique ingredient types (after normalization): {len(ing_counts):,}")
    print(f"  Unique operation types:                        {len(op_counts):,}")
    print(f"\n  Top 30 ingredients: {[k for k,_ in ing_counts.most_common(30)]}")
    print(f"\n  Top ops: {[k for k,_ in op_counts.most_common()]}")

    with open(OUT / 'ingredient_counts.json', 'w') as f:
        json.dump(dict(ing_counts.most_common()), f, indent=2)
    with open(OUT / 'operation_counts.json', 'w') as f:
        json.dump(dict(op_counts.most_common()), f, indent=2)

    # ---- Operation coverage curve ----
    print("\n--- Operation coverage curve ---")
    op_k_vals = list(range(1, min(len(op_counts) + 1, 60)))
    op_curve = greedy_curve(recipe_op_sets, op_counts, op_k_vals)
    for k, cov in op_curve:
        if k % 5 == 0 or cov > 0.94:
            print(f"  M={k:3d}: {cov:.1%}")

    # ---- Ingredient coverage curve (frequency-greedy) ----
    print("\n--- Ingredient coverage curve (frequency greedy) ---")
    ing_k_vals = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300,
                  400, 500, 750, 1000, 1500, 2000, 3000, 5000]
    ing_k_vals = [k for k in ing_k_vals if k <= len(ing_counts)]
    ing_curve = greedy_curve(recipe_ing_sets, ing_counts, ing_k_vals)
    for k, cov in ing_curve:
        print(f"  K={k:5d}: {cov:.1%}")

    # ---- Find optimal ops M (95% coverage) ----
    op_95_m = next((k for k, cov in op_curve if cov >= 0.95), len(op_counts))
    print(f"\n  Operations at 95% coverage: M={op_95_m}")

    # ---- Joint curve: fix ops at op_95_m, vary ingredients ----
    print("\n--- Joint coverage curve ---")
    sorted_ops = [item for item, _ in op_counts.most_common()]
    vocab_ops = set(sorted_ops[:op_95_m])
    # Filter recipe_op_sets to only count recipes whose ops are covered
    joint_curve = []
    for k_ing in ing_k_vals:
        sorted_ings = [item for item, _ in ing_counts.most_common()]
        vocab_ing = set(sorted_ings[:k_ing])
        covered = sum(
            1 for ings, ops in zip(recipe_ing_sets, recipe_op_sets)
            if ings.issubset(vocab_ing) and ops.issubset(vocab_ops)
        )
        cov = covered / len(train)
        joint_curve.append((k_ing + op_95_m, cov))
        print(f"  K_total={k_ing+op_95_m:5d} (ing={k_ing}, ops={op_95_m}): {cov:.1%}")

    # ---- Embedding-based ingredient distortion curve ----
    print("\n--- Embedding-based ingredient distortion curve (GPU) ---")
    unique_ings = list(ing_counts.keys())
    try:
        embeddings = embed_ingredients(unique_ings)
        np.save(OUT / 'ingredient_embeddings.npy', embeddings)
        emb_k_vals = [9, 20, 50, 100, 200, 500, 1000]
        emb_curve = embedding_coverage_curve(
            recipe_ing_sets, unique_ings, embeddings, emb_k_vals
        )
    except Exception as e:
        print(f"  Embedding curve skipped: {e}")
        emb_curve = []

    # ---- Save results ----
    results = {
        'n_train': len(train),
        'n_unique_ingredients': len(ing_counts),
        'n_unique_operations': len(op_counts),
        'ops_at_95pct': op_95_m,
        'op_curve': op_curve,
        'ing_curve_greedy': ing_curve,
        'joint_curve': joint_curve,
        'emb_distortion_curve': emb_curve,
    }
    with open(OUT / 'coverage_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # ---- Plots ----
    _plot_op_curve(op_curve)
    _plot_ing_curve(ing_curve, joint_curve)
    if emb_curve:
        _plot_emb_distortion(emb_curve)

    print(f"\nAll outputs saved to {OUT}/")


def _plot_op_curve(op_curve):
    ks = [p[0] for p in op_curve]
    covs = [p[1] * 100 for p in op_curve]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, covs, 'b-o', markersize=5)
    for thresh, color, label in [(80, 'gray', '80%'), (90, 'orange', '90%'), (95, 'red', '95%')]:
        ax.axhline(thresh, color=color, linestyle='--', alpha=0.6, label=label)
        idx = next((i for i, c in enumerate(covs) if c >= thresh), None)
        if idx is not None:
            ax.axvline(ks[idx], color=color, linestyle=':', alpha=0.4)
    ax.set_xlabel('Number of operation types in vocabulary', fontsize=12)
    ax.set_ylabel('% of train recipes fully covered', fontsize=12)
    ax.set_title('Operation Vocabulary Coverage\n(how many cooking verb types to cover Recipe1M?)', fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / 'coverage_curve_ops.png', dpi=150)
    plt.close()
    print(f"  Saved coverage_curve_ops.png")


def _plot_ing_curve(ing_curve, joint_curve):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ingredient-only
    ks = [p[0] for p in ing_curve]
    covs = [p[1] * 100 for p in ing_curve]
    ax = axes[0]
    ax.semilogx(ks, covs, 'b-o', markersize=5)
    for thresh, color in [(70, 'gray'), (80, 'orange'), (90, 'red'), (95, 'purple')]:
        ax.axhline(thresh, color=color, linestyle='--', alpha=0.5, label=f'{thresh}%')
    ax.set_xlabel('Ingredient vocabulary size K (log scale)', fontsize=11)
    ax.set_ylabel('% of recipes with all ingredients covered', fontsize=11)
    ax.set_title('Ingredient Coverage (frequency-greedy)', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Joint
    ks_j = [p[0] for p in joint_curve]
    covs_j = [p[1] * 100 for p in joint_curve]
    ax = axes[1]
    ax.semilogx(ks_j, covs_j, 'g-o', markersize=5)
    for thresh, color in [(70, 'gray'), (80, 'orange'), (90, 'red'), (95, 'purple')]:
        ax.axhline(thresh, color=color, linestyle='--', alpha=0.5, label=f'{thresh}%')
    ax.set_xlabel('Total vocabulary size (ingredients + operations, log scale)', fontsize=11)
    ax.set_ylabel('% of recipes fully covered', fontsize=11)
    ax.set_title('Joint Coverage (ingredients + operations)', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle('Recipe1M Vocabulary Coverage Curves', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / 'coverage_curve_ing.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved coverage_curve_ing.png")


def _plot_emb_distortion(emb_curve):
    ks = [p[0] for p in emb_curve]
    dists = [p[1] for p in emb_curve]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, dists, 'r-o', markersize=6)
    ax.set_xlabel('Number of ingredient clusters K', fontsize=12)
    ax.set_ylabel('Average intra-cluster distortion', fontsize=12)
    ax.set_title('Ingredient Embedding Distortion vs. K\n(rate-distortion curve — lower = better)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / 'coverage_curve_emb.png', dpi=150)
    plt.close()
    print(f"  Saved coverage_curve_emb.png")


if __name__ == '__main__':
    main()
