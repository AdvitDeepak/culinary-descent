#!/usr/bin/env python3
"""
Phase 1b: Embedding-based ingredient rate-distortion curve.

The USDA ingredient format ("spices, pepper, black") causes over-collapsing
when we take only the first part before the comma. Instead, embed the FULL
ingredient string, cluster at varying K, and measure:
  - Intra-cluster distortion (embedding space) — how lossy is the compression?
  - Coverage: with K clusters, what % of train recipes are fully representable?

This gives the true rate-distortion curve over the semantic ingredient space.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm

RECIPE1M = Path("/home/addeepak/AdvitResearch/cs348k/recipes.json")
OUT = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/vocab_analysis")
OUT.mkdir(parents=True, exist_ok=True)

K_VALUES = [9, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]


def main():
    print("Loading Recipe1M...")
    with open(RECIPE1M) as f:
        all_recipes = json.load(f)
    train = [r for r in all_recipes if r.get('partition') == 'train']
    print(f"  {len(train):,} train recipes")

    # ---- Collect ALL raw ingredient strings (before normalization) ----
    print("\nCollecting raw ingredient strings...")
    raw_ing_counts: Counter = Counter()
    recipe_raw_ing_sets: list[set] = []

    for recipe in tqdm(train):
        s = set()
        for item in recipe.get('ingredients', []):
            raw = item['text'].strip().lower()
            if raw:
                raw_ing_counts[raw] += 1
                s.add(raw)
        recipe_raw_ing_sets.append(s)

    unique_strings = list(raw_ing_counts.keys())
    print(f"  {len(unique_strings):,} unique raw ingredient strings")
    print(f"  Top 20: {[s for s,_ in raw_ing_counts.most_common(20)]}")

    # ---- Embed on GPU ----
    print("\nEmbedding ingredient strings on GPU...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    embeddings = model.encode(
        unique_strings,
        batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.save(OUT / 'ingredient_embeddings_raw.npy', embeddings)
    print(f"  Embeddings shape: {embeddings.shape}")

    # String → index map for fast lookup
    str_to_idx = {s: i for i, s in enumerate(unique_strings)}

    # Pre-map recipe ingredient sets to embedding indices
    recipe_idx_sets = [
        {str_to_idx[s] for s in recipe_set if s in str_to_idx}
        for recipe_set in recipe_raw_ing_sets
    ]

    # ---- Frequency-greedy coverage curve (on raw strings) ----
    print("\n--- Frequency-greedy coverage curve (raw strings) ---")
    sorted_strs = [s for s, _ in raw_ing_counts.most_common()]
    freq_k_vals = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
                   10000, len(unique_strings)]
    freq_k_vals = [k for k in freq_k_vals if k <= len(unique_strings)]
    freq_curve = []
    for k in freq_k_vals:
        vocab = set(sorted_strs[:k])
        covered = sum(1 for s in recipe_raw_ing_sets if s.issubset(vocab))
        cov = covered / len(train)
        freq_curve.append((k, cov))
        print(f"  K={k:6d}: {cov:.1%}")

    # ---- Embedding k-means sweep ----
    print("\n--- K-means clustering sweep ---")
    distortion_curve = []
    coverage_curve = []
    cluster_labels_by_k = {}

    valid_k = [k for k in K_VALUES if k <= len(unique_strings)]

    for k in valid_k:
        print(f"  K={k}: fitting k-means...", end=' ', flush=True)
        km = MiniBatchKMeans(
            n_clusters=k, random_state=42, batch_size=4096,
            n_init=5, max_iter=300,
        )
        labels = km.fit_predict(embeddings)
        distortion = float(km.inertia_ / len(unique_strings))

        # Coverage: with K clusters, what % of recipes have all their
        # ingredients represented? (Always 100% since all items get a cluster)
        # More useful: average % of a recipe's ingredients that map to
        # distinct clusters (= how much information is preserved per recipe)
        info_preserved = []
        for idx_set in recipe_idx_sets:
            if not idx_set:
                continue
            n_distinct_clusters = len({labels[i] for i in idx_set})
            info_preserved.append(n_distinct_clusters / len(idx_set))
        avg_info = float(np.mean(info_preserved))

        cluster_labels_by_k[k] = labels.tolist()
        distortion_curve.append((k, distortion))
        coverage_curve.append((k, avg_info))

        # Show centroid exemplars (nearest ingredient to each centroid)
        _, indices = pairwise_distances_argmin_min(km.cluster_centers_, embeddings)
        exemplars = [unique_strings[int(i)] for i in indices[:min(5, k)]]
        print(f"distortion={distortion:.4f}, info_preserved={avg_info:.3f}")
        print(f"    exemplars: {exemplars}")

    # ---- Save ----
    results = {
        'n_unique_raw_strings': len(unique_strings),
        'n_train_recipes': len(train),
        'freq_greedy_curve': freq_curve,
        'kmeans_distortion_curve': distortion_curve,
        'kmeans_info_preserved_curve': coverage_curve,
        'top_ingredients_raw': raw_ing_counts.most_common(200),
    }
    with open(OUT / 'embedding_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # ---- Plots ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Frequency-greedy coverage on raw strings
    ax = axes[0]
    ks = [p[0] for p in freq_curve]
    covs = [p[1] * 100 for p in freq_curve]
    ax.semilogx(ks, covs, 'b-o', markersize=5)
    for thresh, color in [(70, 'gray'), (80, 'orange'), (90, 'red'), (95, 'purple')]:
        ax.axhline(thresh, color=color, linestyle='--', alpha=0.5, label=f'{thresh}%')
    ax.set_xlabel('K (unique ingredient strings in vocabulary)', fontsize=11)
    ax.set_ylabel('% of recipes fully covered', fontsize=11)
    ax.set_title('Frequency-greedy Coverage\n(raw USDA strings)', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 2. K-means distortion vs K
    ax = axes[1]
    ks_e = [p[0] for p in distortion_curve]
    dists = [p[1] for p in distortion_curve]
    ax.plot(ks_e, dists, 'r-o', markersize=6)
    ax.set_xlabel('Number of ingredient clusters K', fontsize=11)
    ax.set_ylabel('Avg intra-cluster distortion (embedding space)', fontsize=11)
    ax.set_title('Embedding Distortion vs. K\n(rate-distortion — lower = better)', fontsize=11)
    ax.grid(True, alpha=0.3)

    # 3. Information preserved vs K
    ax = axes[2]
    ks_i = [p[0] for p in coverage_curve]
    infos = [p[1] * 100 for p in coverage_curve]
    ax.plot(ks_i, infos, 'g-o', markersize=6)
    ax.set_xlabel('Number of ingredient clusters K', fontsize=11)
    ax.set_ylabel('Avg % of recipe ingredients as distinct clusters', fontsize=11)
    ax.set_title('Information Preserved vs. K\n(how much structure survives compression)', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Recipe1M Ingredient Rate-Distortion Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT / 'embedding_rate_distortion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlots saved to {OUT}/embedding_rate_distortion.png")

    # Print summary
    print("\n=== Key findings ===")
    print(f"  Unique raw ingredient strings: {len(unique_strings):,}")
    print(f"\n  Freq-greedy coverage:")
    for target in [0.70, 0.80, 0.90, 0.95, 0.99]:
        for k, cov in freq_curve:
            if cov >= target:
                print(f"    {target:.0%}: K={k:,} strings needed")
                break
    print(f"\n  Embedding distortion knees:")
    for k, d in distortion_curve:
        print(f"    K={k:5d}: distortion={d:.4f}")


if __name__ == '__main__':
    main()
