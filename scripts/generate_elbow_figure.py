#!/usr/bin/env python3
"""
Generate k-means inertia/silhouette figure for cooking verb embeddings.
Uses only actual cooking verbs (not generic English verbs or ingredients).
Shows why bottom-up clustering can't determine K — and motivates top-down design.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

VOCAB  = Path("data/vocab_analysis/process_vocab.json")
OUT    = Path("data/figures/process_kmeans_elbow.png")

# Whitelist of cooking verbs — anything not on this list is excluded.
# Derived from ACTION_TO_CANONICAL + common recipe corpus verbs.
# Excludes: generic English (desire, like, begin, end, depend), ingredients
# used as verbs (chicken, carrot, butter), and tools (bowl, pan, skillet).
COOKING_VERBS = {
    # heat: dry/radiant
    "bake","roast","broil","grill","toast","char","sear","brown",
    # heat: fat
    "saute","fry","pan-fry","stir-fry","deep-fry",
    # heat: moist
    "boil","simmer","braise","poach","blanch","parboil","steam","stew","scald",
    # heat: other
    "microwave","caramelize","toast","reheat","warm",
    # combine
    "mix","combine","stir","toss","fold","whisk","beat","blend","puree",
    "knead","cream","incorporate","emulsify",
    # prep / knife
    "chop","dice","slice","mince","grate","peel","crush","cut","shred",
    "julienne","halve","quarter","trim","score","chiffonade","shave",
    "grind","mash","crumble","zest","press","pound",
    # passive / liquid
    "marinate","soak","brine","macerate","steep","infuse","cure",
    "cool","chill","freeze","refrigerate","rest","stand",
    # apply / finish
    "season","coat","brush","drizzle","sprinkle","dust","glaze","garnish",
    "baste","rub","dredge","dip","stuff",
    # liquid ops
    "reduce","dissolve","drain","melt","strain","skim","clarify","thicken",
    "deglaze","rinse","wash","absorb","evaporate","condense",
    # other culinary
    "sift","whip","spread","roll","flatten","shape","form","punch","proof",
    "scoop","ladle","pour","transfer","divide","portion","layer","arrange",
    "wrap","seal","cover","uncover","flip","turn","rotate","pat","dry",
    "toast","smoke","pickle","ferment","cure","can","preserve",
    "soften","wilt","reduce","crack","separate","discard","reserve",
    "squeeze","strain","pass","puree","process","blend",
}

# ── Load verbs ─────────────────────────────────────────────────────────────────
print("Loading verb vocabulary...", flush=True)
with open(VOCAB) as f:
    d = json.load(f)

verb_counts = d["verb_counts"]
# Keep cooking verbs that appear in >= 30 recipes
verbs = [(v, c) for v, c in verb_counts.items() if c >= 30 and v in COOKING_VERBS]
all_above_30 = sum(1 for v, c in verb_counts.items() if c >= 30)
print(f"  {len(verbs)} cooking verbs (count >= 30) out of {all_above_30} total verbs >= 30")
print(f"  Filtered out {all_above_30 - len(verbs)} non-cooking words")

verb_list = [v for v, _ in verbs]
print(f"  Cooking verbs: {', '.join(sorted(verb_list))}")

# ── Embed with BGE ─────────────────────────────────────────────────────────────
print("Embedding verbs with BGE...", flush=True)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings = model.encode(verb_list, show_progress_bar=True, normalize_embeddings=True)
print(f"  Embeddings shape: {embeddings.shape}")

# ── Run k-means and compute silhouette across K ────────────────────────────────
n = len(verb_list)
Ks = [k for k in [3, 5, 7, 9, 10, 12, 15, 18, 20, 23, 25, 28, 30, 35, 40, 45, 50] if k < n]

print(f"Running k-means for K in {Ks}...", flush=True)
inertias   = []
silhouettes = []
for k in Ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(embeddings)
    inertias.append(km.inertia_)
    sil = silhouette_score(embeddings, labels, metric='cosine')
    silhouettes.append(sil)
    print(f"  K={k:3d}  inertia={km.inertia_:.1f}  silhouette={sil:.4f}", flush=True)

inertias_arr   = np.array(inertias)
silhouettes_arr = np.array(silhouettes)
Ks_arr = np.array(Ks)

# ── Marginal gain ──────────────────────────────────────────────────────────────
marginal_K    = []
marginal_gain = []
for i in range(1, len(Ks)):
    dk   = Ks_arr[i] - Ks_arr[i-1]
    drop = (inertias_arr[i-1] - inertias_arr[i]) / dk
    mid  = (Ks_arr[i] + Ks_arr[i-1]) / 2
    marginal_K.append(mid)
    marginal_gain.append(drop)

marginal_K    = np.array(marginal_K)
marginal_gain = np.array(marginal_gain)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle(
    f"K-means on cooking verb BGE embeddings (Recipe1M, n={len(verb_list)} cooking verbs)",
    fontsize=11, fontweight="bold"
)

# Left: inertia curve
ax = axes[0]
ax.plot(Ks_arr, inertias_arr, "o-", color="#2171b5", lw=2, ms=5)
ax.axvline(15, color="#e6550d", lw=1.5, ls="--", label="K=15 (chosen)")
ax.set_xlabel("Number of clusters K", fontsize=10)
ax.set_ylabel("Inertia (within-cluster SS)", fontsize=10)
ax.set_title("Inertia vs K\n(no sharp elbow)", fontsize=9)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

# Middle: silhouette score
ax = axes[1]
ax.plot(Ks_arr, silhouettes_arr, "o-", color="#31a354", lw=2, ms=5)
ax.axvline(15, color="#e6550d", lw=1.5, ls="--", label="K=15 (chosen)")
ax.axhline(0, color="#aaa", lw=0.8, ls=":")
ax.set_xlabel("Number of clusters K", fontsize=10)
ax.set_ylabel("Silhouette score (cosine)", fontsize=10)
ax.set_title("Silhouette score vs K\n(near-zero: continuous manifold)", fontsize=9)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

# Right: marginal gain
ax = axes[2]
ax.plot(marginal_K, marginal_gain, "o-", color="#2171b5", lw=2, ms=5)
ax.axvline(15, color="#e6550d", lw=1.5, ls="--", label="K=15 (chosen)")
ax.fill_between(marginal_K, marginal_gain, alpha=0.12, color="#2171b5")
ax.set_xlabel("Number of clusters K", fontsize=10)
ax.set_ylabel("Inertia drop per additional cluster", fontsize=10)
ax.set_title("Marginal improvement per cluster\n(steadily diminishing)", fontsize=9)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {OUT}")

print(f"\nSilhouette scores (near zero = continuous space, no natural K):")
for k, sil in zip(Ks, silhouettes):
    marker = " <-- chosen" if k == 15 else ""
    print(f"  K={k:3d}: {sil:.4f}{marker}")

print(f"\nInertia values:")
for k, inertia in zip(Ks, inertias):
    marker = " <-- chosen" if k == 15 else ""
    print(f"  K={k:3d}: {inertia:.1f}{marker}")
