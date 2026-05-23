#!/usr/bin/env python3
"""
Phase 1c: Coverage curve on real Recipe1M (det_ingrs.json)
-----------------------------------------------------------
Uses det_ingrs.json (1M recipes, natural language ingredient names)
instead of the USDA-normalized nutritional layer.

Only counts ingredients where valid[i] == True.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

DET_INGRS = Path("/home/addeepak/AdvitResearch/cs348k/det_ingrs.json")
OUT = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/vocab_analysis")
OUT.mkdir(parents=True, exist_ok=True)

K_VALUES = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300,
            400, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000]


def normalize(text: str) -> str:
    return text.lower().strip()


def main():
    print(f"Loading {DET_INGRS}...")
    with open(DET_INGRS) as f:
        data = json.load(f)
    print(f"  {len(data):,} recipe entries")

    # ---- Count ingredient frequencies (valid only) ----
    print("\nCounting ingredient frequencies (valid entries only)...")
    ing_counts: Counter = Counter()
    recipe_ing_sets: list[set] = []

    for entry in tqdm(data, desc="scanning"):
        ings = entry.get('ingredients', [])
        valids = entry.get('valid', [True] * len(ings))
        s = set()
        for ing, v in zip(ings, valids):
            if v:
                norm = normalize(ing['text'])
                if norm:
                    ing_counts[norm] += 1
                    s.add(norm)
        recipe_ing_sets.append(s)

    print(f"\n  Unique ingredient strings (valid, normalized): {len(ing_counts):,}")
    print(f"  Total valid ingredient mentions:               {sum(ing_counts.values()):,}")
    print(f"  Average ingredients per recipe:                "
          f"{sum(len(s) for s in recipe_ing_sets)/len(recipe_ing_sets):.1f}")
    print(f"\n  Top 40 ingredients:")
    for name, count in ing_counts.most_common(40):
        print(f"    {count:7,}  {name}")

    # ---- Frequency-greedy coverage curve ----
    print("\n--- Frequency-greedy coverage curve ---")
    sorted_ings = [item for item, _ in ing_counts.most_common()]
    n_recipes = len(data)

    k_vals = [k for k in K_VALUES if k <= len(sorted_ings)]
    k_vals_full = sorted(set(k_vals + [len(sorted_ings)]))

    curve_points = []
    for k in k_vals_full:
        vocab = set(sorted_ings[:k])
        covered = sum(1 for s in recipe_ing_sets if s.issubset(vocab))
        cov = covered / n_recipes
        curve_points.append((k, cov))
        if k in set(k_vals):
            print(f"  K={k:6d}: {cov:.1%} of {n_recipes:,} recipes covered")

    # ---- Find K for coverage targets ----
    print("\n--- Vocabulary size for coverage targets ---")
    for target in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
        hit = next(((k, c) for k, c in curve_points if c >= target), None)
        if hit:
            print(f"  {target:.0%} coverage: K = {hit[0]:,} ingredient types")

    # ---- Embedding distortion sweep (GPU) ----
    print("\n--- Embedding distortion curve (GPU, top 5000 by freq) ---")
    top_n = min(5000, len(sorted_ings))
    top_ings = sorted_ings[:top_n]

    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import MiniBatchKMeans

    print(f"  Embedding top {top_n:,} ingredient strings...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    embeddings = model.encode(
        top_ings, batch_size=512, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    np.save(OUT / 'ingredient_embeddings_real.npy', embeddings)
    str_to_idx = {s: i for i, s in enumerate(top_ings)}

    # For coverage: only count recipes whose ingredients are ALL in top_n
    recipe_idx_sets_top = []
    for s in recipe_ing_sets:
        top_s = {str_to_idx[x] for x in s if x in str_to_idx}
        recipe_idx_sets_top.append(top_s)

    emb_k_vals = [9, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    emb_k_vals = [k for k in emb_k_vals if k <= top_n]

    distortion_curve = []
    info_curve = []
    exemplar_table = {}

    for k in emb_k_vals:
        print(f"  K={k}: fitting...", end=' ', flush=True)
        km = MiniBatchKMeans(
            n_clusters=k, random_state=42, batch_size=4096,
            n_init=5, max_iter=300,
        )
        labels = km.fit_predict(embeddings)
        distortion = float(km.inertia_ / top_n)

        # Average % of distinct clusters used per recipe
        info_vals = []
        for idx_set in recipe_idx_sets_top:
            if not idx_set:
                continue
            n_distinct = len({labels[i] for i in idx_set})
            info_vals.append(n_distinct / len(idx_set))
        avg_info = float(np.mean(info_vals))

        # Find exemplar for each cluster: the ingredient closest to centroid
        centers = km.cluster_centers_
        # centers: (k, 384), embeddings: (top_n, 384)
        # For each center, find the nearest embedding
        sims = centers @ embeddings.T   # (k, top_n)
        nearest_idx = np.argmax(sims, axis=1)  # (k,)
        exemplars = [top_ings[int(i)] for i in nearest_idx[:min(8, k)]]

        distortion_curve.append((k, distortion))
        info_curve.append((k, avg_info))
        exemplar_table[k] = exemplars

        print(f"distortion={distortion:.4f}, info={avg_info:.3f}")
        print(f"    exemplars: {exemplars}")

    # ---- Save ----
    results = {
        'n_recipes': n_recipes,
        'n_unique_ingredients': len(ing_counts),
        'freq_curve': curve_points,
        'emb_distortion_curve': distortion_curve,
        'emb_info_curve': info_curve,
        'top_200_ingredients': ing_counts.most_common(200),
    }
    with open(OUT / 'real_coverage_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # ---- Plots ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Frequency coverage curve
    ax = axes[0]
    ks = [p[0] for p in curve_points]
    covs = [p[1] * 100 for p in curve_points]
    ax.semilogx(ks, covs, 'b-o', markersize=3)
    for thresh, color, label in [
        (70, 'gray', '70%'), (80, 'orange', '80%'),
        (90, 'red', '90%'), (95, 'purple', '95%')
    ]:
        ax.axhline(thresh, color=color, linestyle='--', alpha=0.5, label=label)
        hit = next((k for k, c in zip(ks, covs) if c >= thresh), None)
        if hit:
            ax.axvline(hit, color=color, linestyle=':', alpha=0.3)
    ax.set_xlabel('Vocabulary size K (log scale)', fontsize=11)
    ax.set_ylabel('% of 1M recipes fully covered', fontsize=11)
    ax.set_title('Ingredient Coverage Curve\n(frequency-greedy, real Recipe1M)', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Embedding distortion
    ax = axes[1]
    ks_e = [p[0] for p in distortion_curve]
    dists = [p[1] for p in distortion_curve]
    ax.plot(ks_e, dists, 'r-o', markersize=6)
    ax.set_xlabel('Number of semantic clusters K', fontsize=11)
    ax.set_ylabel('Avg intra-cluster distortion', fontsize=11)
    ax.set_title('Embedding Distortion vs. K\n(rate-distortion — find the elbow)', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Information preserved
    ax = axes[2]
    ks_i = [p[0] for p in info_curve]
    infos = [p[1] * 100 for p in info_curve]
    ax.plot(ks_i, infos, 'g-o', markersize=6)
    ax.set_xlabel('Number of semantic clusters K', fontsize=11)
    ax.set_ylabel('Avg % of recipe ingredients as distinct clusters', fontsize=11)
    ax.set_title('Information Preserved vs. K\n(reconstruction capacity)', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Recipe1M Ingredient Rate-Distortion (det_ingrs.json, 1M recipes)', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT / 'real_rate_distortion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved → {OUT}/real_rate_distortion.png")
    print(f"Results  → {OUT}/real_coverage_results.json")

    print("\n=== Exemplar clusters ===")
    for k, exemplars in sorted(exemplar_table.items()):
        print(f"  K={k}: {exemplars}")


if __name__ == '__main__':
    main()
