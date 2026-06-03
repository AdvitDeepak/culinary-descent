#!/usr/bin/env python3
"""
Vocabulary Analysis: find the ideal step-type vocabulary empirically.

Three signals:
  1. spaCy ROOT-verb extraction from 300K sampled recipe steps → ranked cooking verb list
  2. MiniLM + hierarchical clustering of top cooking verbs → dendrogram
  3. Cluster-level coverage curve: at vocabulary size K, what % of recipe steps
     are covered by one of the K semantic clusters? Find the elbow.

Outputs:
  data/eval/vocab_analysis.json
  data/figures/vocab_dendrogram.png
  data/figures/vocab_coverage.png
  data/figures/vocab_frequencies.png
"""

import json, os, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE    = Path(__file__).resolve().parent.parent
LAYER1  = Path(os.environ.get("RECIPE1M", str(BASE.parent / "layer1.json")))
OUT_DIR = BASE / "data/eval"
FIG_DIR = BASE / "data/figures"

# ── Known cooking verbs: expanded seed list ───────────────────────────────────
# These are verbs whose first-word occurrence in a recipe step IS a cooking action.
# Anything not here is either a non-cooking word or too rare/ambiguous to include.
COOKING_VERBS = {
    # heat-based / dry
    "bake", "roast", "broil", "toast", "grill", "barbecue", "char",
    "sear", "brown", "caramelize", "smoke",
    # heat-based / wet
    "boil", "blanch", "parboil", "poach", "simmer", "stew", "braise",
    "steam", "microwave",
    # fat-based
    "saute", "fry", "stir-fry", "deep-fry", "pan-fry",
    # reduce / concentrate
    "reduce", "thicken", "deglaze",
    # mechanical / mix
    "mix", "combine", "stir", "fold", "toss", "incorporate",
    "whisk", "beat", "whip",
    "blend", "puree", "process", "liquefy",
    "knead", "work",
    # cut / prep
    "chop", "dice", "mince", "slice", "julienne", "shred", "grate",
    "peel", "trim", "halve", "quarter", "crush", "smash", "flatten",
    # season / flavor
    "season", "marinate", "brine", "cure", "rub",
    "salt", "pepper",
    # thermal (no heat)
    "chill", "refrigerate", "freeze", "cool",
    # other meaningful ops
    "sift", "drain", "strain", "melt", "cream", "temper", "proof",
    "infuse", "steep", "ferment", "smoke",
    # common generic ops that we keep
    "cook", "heat", "preheat",
}

# Words that appear at start of sentences but are NOT cooking ops
NON_COOKING_FIRST = {
    "add", "place", "put", "use", "make", "let", "set", "get", "keep",
    "allow", "ensure", "note", "try", "go", "come", "do", "serve", "enjoy",
    "reserve", "repeat", "continue", "begin", "start", "finish",
    "transfer", "pour", "turn", "flip", "cover", "remove", "take",
    "bring", "divide", "spoon", "ladle", "top", "fill", "line",
    "arrange", "lay", "press", "hold", "tear", "shape", "form", "roll",
    "wrap", "tie", "store", "prepare", "gather", "portion", "plate",
    "garnish", "assemble", "layer", "stack", "insert", "thread", "dip",
    "coat", "brush", "grease", "spray", "butter", "dust", "oil",
    "drizzle", "sprinkle", "scatter", "drop", "spoon", "ladle",
    "spread", "scrape", "scoop",
    # clearly not verbs
    "in", "a", "an", "the", "for", "to", "with", "on", "at", "from",
    "if", "when", "after", "before", "once", "until", "while", "as",
    "now", "then", "next", "finally", "first", "last", "lightly",
    "gradually", "gently", "carefully", "slowly", "using", "you",
    "this", "makes", "just", "its", "your",
}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Extract cooking verbs with spaCy (ROOT verb + POS check)
# ══════════════════════════════════════════════════════════════════════════════

print("STAGE 1: Loading layer1.json...", flush=True)
with open(LAYER1) as f:
    all_recipes = json.load(f)

train_sentences = []
for r in all_recipes:
    if r.get("partition") != "train":
        continue
    for step in r.get("instructions", []):
        text = step.get("text", "").strip()
        if text:
            train_sentences.append(text)

print(f"  Train sentences: {len(train_sentences):,}", flush=True)

import random
random.seed(42)
N_SAMPLE = 300_000
sample = random.sample(train_sentences, min(N_SAMPLE, len(train_sentences)))
print(f"  Sampled {len(sample):,} for analysis", flush=True)

import spacy
nlp = spacy.load("en_core_web_sm")

print("  Running spaCy (POS + dependency) on sampled sentences...", flush=True)

raw_verb_counts = Counter()   # all first-word verbs (for coverage denominator)
cook_verb_counts = Counter()  # only COOKING_VERBS (for vocabulary analysis)
step_first_verbs = []         # extracted verb per step (or None)

BATCH = 4000
for i in range(0, len(sample), BATCH):
    docs = list(nlp.pipe(sample[i:i+BATCH], batch_size=BATCH))
    for doc in docs:
        verb = None
        # Strategy 1: ROOT token that is a VERB
        for tok in doc:
            if tok.dep_ == "ROOT" and tok.pos_ in ("VERB", "AUX"):
                verb = tok.lemma_.lower()
                break
        # Strategy 2: first non-punct token that is VERB or looks like imperative
        if verb is None or verb in NON_COOKING_FIRST:
            for tok in doc:
                if tok.is_alpha and tok.pos_ == "VERB":
                    verb = tok.lemma_.lower()
                    break
        if verb:
            raw_verb_counts[verb] += 1
            if verb in COOKING_VERBS:
                cook_verb_counts[verb] += 1
        step_first_verbs.append(verb)
    if (i // BATCH) % 10 == 0:
        print(f"    {i+len(docs):>7,} / {len(sample):,}", flush=True)

total_extracted = sum(1 for v in step_first_verbs if v)
total_cooking   = sum(cook_verb_counts.values())
print(f"  Extracted verb: {total_extracted:,}/{len(sample):,} steps", flush=True)
print(f"  Known cooking verbs: {total_cooking:,} ({100*total_cooking/len(sample):.1f}% of all steps)", flush=True)

print(f"\nTop 50 cooking verbs:")
for v, c in cook_verb_counts.most_common(50):
    bar = "█" * (c // 1000)
    print(f"  {v:<18} {c:>7,}  {bar}")

print(f"\nTop 20 OTHER verbs (not in cooking set, for reference):")
other = {v: c for v, c in raw_verb_counts.items() if v not in COOKING_VERBS}
for v, c in Counter(other).most_common(20):
    print(f"  {v:<18} {c:>7,}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Semantic clustering of cooking verbs (MiniLM + hierarchical)
# ══════════════════════════════════════════════════════════════════════════════

print("\nSTAGE 2: Semantic clustering...", flush=True)

# Use all verbs that appear in our cooking set and were actually seen in data
seen_cooking = [v for v in COOKING_VERBS if v in cook_verb_counts]
# Add a few important ones even if low frequency
print(f"  Clustering {len(seen_cooking)} cooking verbs", flush=True)

from sentence_transformers import SentenceTransformer
smodel = SentenceTransformer("all-MiniLM-L6-v2")

# Use richer phrases to help MiniLM disambiguate
phrases = [f"food preparation: {v} the ingredients" for v in seen_cooking]
embs = smodel.encode(phrases, normalize_embeddings=True, show_progress_bar=False)
print(f"  Embeddings: {embs.shape}", flush=True)

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

dist_mat = pdist(embs, metric="cosine")
Z = linkage(dist_mat, method="ward")

# ── Coverage curves ───────────────────────────────────────────────────────────
# Individual curve: top-K verbs by raw frequency → % of steps covered
ranked_cooking = cook_verb_counts.most_common()
cumulative = 0
coverage_individual = []
for k, (v, c) in enumerate(ranked_cooking, 1):
    cumulative += c
    coverage_individual.append((k, cumulative / len(sample)))

# Cluster curve: cut dendrogram at each K, rank the resulting K clusters by
# their total member frequency, then greedily include clusters from most→least
# frequent. Coverage at K = sum of ALL member frequencies of the top-K
# clusters / total steps. This curve is monotonically increasing with K.
MAX_K = min(50, len(seen_cooking) - 1)
cluster_coverage = []
for K in range(1, MAX_K + 1):
    labels_k = fcluster(Z, K, criterion="maxclust")
    # total frequency for each cluster
    cluster_freq = defaultdict(int)
    for verb, lbl in zip(seen_cooking, labels_k):
        cluster_freq[int(lbl)] += cook_verb_counts.get(verb, 0)
    # sort clusters descending by frequency; take top-K of them
    # (when fcluster gives exactly K clusters, all K are already "top")
    total_covered = sum(cluster_freq.values())   # all K clusters included
    cluster_coverage.append((K, total_covered / len(sample)))

# The above is still flat. The RIGHT curve: fix the partition at MAX_K clusters,
# then greedily add the most-frequent clusters one by one.
labels_max = fcluster(Z, MAX_K, criterion="maxclust")
cluster_freq_max = defaultdict(int)
cluster_members_max = defaultdict(list)
for verb, lbl in zip(seen_cooking, labels_max):
    cluster_freq_max[int(lbl)] += cook_verb_counts.get(verb, 0)
    cluster_members_max[int(lbl)].append(verb)
sorted_clusters = sorted(cluster_freq_max.items(), key=lambda x: -x[1])

cumulative_clust = 0
greedy_cluster_coverage = []
for k, (lbl, freq) in enumerate(sorted_clusters, 1):
    cumulative_clust += freq
    greedy_cluster_coverage.append((k, cumulative_clust / len(sample)))

# Also: individual verb greedy coverage but including ALL verbs (not just cooking)
# to show why 100% is not achievable with cooking verbs alone
all_verb_counts = Counter({v: c for v, c in raw_verb_counts.items()})
cumulative_all = 0
coverage_all_verbs = []
for k, (v, c) in enumerate(all_verb_counts.most_common(100), 1):
    cumulative_all += c
    coverage_all_verbs.append((k, cumulative_all / len(sample)))

print("\nGreedy cluster coverage (adding most-frequent cluster each step):")
for K, cov in greedy_cluster_coverage:
    if K in {1, 3, 5, 8, 10, 12, 15, 18, 20, 25, 30}:
        print(f"  K={K:>2} clusters → {cov:.3f} of steps covered")

# Find elbow in greedy cluster coverage curve (second derivative)
cov_arr = np.array([c for _, c in greedy_cluster_coverage])
ks_arr  = np.array([k for k, _ in greedy_cluster_coverage])
if len(cov_arr) > 4:
    d2 = np.gradient(np.gradient(cov_arr))
    elbow_idx = int(np.argmax(-d2[:min(30, len(d2))]))
else:
    elbow_idx = len(cov_arr) // 2
elbow_K   = int(ks_arr[elbow_idx])
elbow_cov = float(cov_arr[elbow_idx])
print(f"\n  Elbow: K={elbow_K} clusters covers {elbow_cov:.3f} of steps")

# Show cluster contents at K=15
labels15 = fcluster(Z, 15, criterion="maxclust")
clusters15 = defaultdict(list)
for verb, lbl in zip(seen_cooking, labels15):
    clusters15[int(lbl)].append(verb)
print("\nSemantic clusters at K=15:")
for lbl, verbs in sorted(clusters15.items()):
    top_v = sorted(verbs, key=lambda v: -cook_verb_counts.get(v, 0))
    freq  = sum(cook_verb_counts.get(v, 0) for v in verbs)
    print(f"  C{lbl:>2} [{freq:>5,}]: {', '.join(top_v)}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Figures
# ══════════════════════════════════════════════════════════════════════════════

print("\nSTAGE 3: Figures...", flush=True)

# ── Figure 1: Coverage curves (individual verbs vs semantic clusters) ─────────
fig, ax = plt.subplots(figsize=(10, 5))

xs_ind = [k for k, _ in coverage_individual[:60]]
ys_ind = [c for _, c in coverage_individual[:60]]
xs_cls = [k for k, _ in cluster_coverage[:40]]
ys_cls = [c for _, c in cluster_coverage[:40]]

xs_ind = [k for k, _ in coverage_individual[:50]]
ys_ind = [c for _, c in coverage_individual[:50]]
xs_cls = [k for k, _ in greedy_cluster_coverage]
ys_cls = [c for _, c in greedy_cluster_coverage]
xs_all = [k for k, _ in coverage_all_verbs[:50]]
ys_all = [c for _, c in coverage_all_verbs[:50]]

ax.plot(xs_ind, ys_ind, color="#1565C0", lw=2,
        label="Top-K individual cooking verbs")
ax.plot(xs_cls, ys_cls, color="#2E7D32", lw=2.5, ls="--",
        label=f"K semantic clusters (greedily ordered, covers all synonyms)")
ax.plot(xs_all, ys_all, color="#9E9E9E", lw=1.5, ls=":",
        label="Top-K ALL verbs (including utility: add/remove/pour...)")
ax.axvline(elbow_K, color="#E53935", lw=1.8, ls="-.",
           label=f"Elbow at K={elbow_K} ({elbow_cov:.1%} coverage)")
ax.axvline(15, color="#FB8C00", lw=1.5, ls="-.",
           alpha=0.7,
           label=f"Our vocabulary K=15 ({ys_cls[14]:.1%} coverage)")
for k in [5, 10, 15, 20]:
    if k <= len(ys_cls):
        ax.text(k + 0.3, ys_cls[k-1] + 0.008, f"{ys_cls[k-1]:.0%}",
                fontsize=8, color="#2E7D32")
ax.fill_between(xs_cls, ys_cls, alpha=0.1, color="#2E7D32")
ax.set_xlabel("Vocabulary size K (number of types)", fontsize=11)
ax.set_ylabel("Fraction of recipe steps covered", fontsize=11)
ax.set_title("Coverage curve: how many cooking operation types\nare needed to cover Recipe1M (720K train steps)?",
             fontsize=11, fontweight="bold")
ax.set_ylim(0, 0.85)
ax.set_xlim(1, 40)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "vocab_coverage.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved → vocab_coverage.png", flush=True)

# ── Figure 2: Dendrogram ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 7))
cut15 = Z[-(15), 2]
cut12 = Z[-(12), 2]
cut20 = Z[-(20), 2]
dendrogram(Z, labels=seen_cooking, ax=ax,
           leaf_rotation=75, leaf_font_size=8.5,
           color_threshold=cut15)
ax.axhline(cut15, color="#E53935", lw=1.8, ls="--", label="K=15 cut")
ax.axhline(cut12, color="#FB8C00", lw=1.3, ls=":",  label="K=12 cut")
ax.axhline(cut20, color="#43A047", lw=1.3, ls="-.", label="K=20 cut")
ax.set_title("Hierarchical clustering of cooking verbs (MiniLM + Ward linkage)\n"
             "Natural cut point determines vocabulary size",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Ward distance", fontsize=10)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "vocab_dendrogram.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved → vocab_dendrogram.png", flush=True)

# ── Figure 3: Frequency bar chart ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
top_n = 40
top_list = cook_verb_counts.most_common(top_n)
labels_b = [v for v, _ in top_list]
counts_b = [c for _, c in top_list]
in_elbow = set(v for v, _ in cook_verb_counts.most_common(elbow_K))
colors_b = ["#1565C0" if v in in_elbow else "#90A4AE" for v in labels_b]
ax.bar(range(len(labels_b)), counts_b, color=colors_b, edgecolor="white", lw=0.5)
ax.set_xticks(range(len(labels_b)))
ax.set_xticklabels(labels_b, rotation=50, ha="right", fontsize=8.5)
ax.axvline(elbow_K - 0.5, color="#E53935", lw=2, ls="--",
           label=f"Elbow (K={elbow_K})")
ax.set_ylabel("Occurrences (300K sampled steps)", fontsize=10)
ax.set_title("Cooking verb frequency — Recipe1M train (720K recipes)\n"
             "Blue = inside elbow vocabulary, grey = beyond",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "vocab_frequencies.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved → vocab_frequencies.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4: Save
# ══════════════════════════════════════════════════════════════════════════════

eval_out = {
    "n_sampled_steps": len(sample),
    "n_total_train_steps": len(train_sentences),
    "top_cooking_verbs": [{"verb": v, "count": int(c)}
                          for v, c in cook_verb_counts.most_common(80)],
    "coverage_individual": [{"k": int(k), "coverage": float(c)}
                             for k, c in coverage_individual[:60]],
    "coverage_clusters_greedy": [{"k": int(k), "coverage": float(c)}
                                 for k, c in greedy_cluster_coverage],
    "elbow_k": elbow_K,
    "elbow_coverage": elbow_cov,
    "coverage_at_k15_clusters": float(greedy_cluster_coverage[14][1]) if len(greedy_cluster_coverage) >= 15 else None,
    "semantic_clusters_k15": {
        f"cluster_{k}": sorted(verbs, key=lambda v: -cook_verb_counts.get(v, 0))
        for k, verbs in sorted(clusters15.items())
    },
}
with open(OUT_DIR / "vocab_analysis.json", "w") as f:
    json.dump(eval_out, f, indent=2)
print(f"\nSaved → data/eval/vocab_analysis.json", flush=True)

print("\n" + "="*60)
print("VOCABULARY ANALYSIS SUMMARY")
print("="*60)
print(f"  Sampled {len(sample):,} train steps from 720K recipes")
print(f"  Elbow:  K={elbow_K} semantic clusters → {elbow_cov:.1%} coverage")
if len(greedy_cluster_coverage) >= 15:
    print(f"  K=15:   {greedy_cluster_coverage[14][1]:.1%} coverage")
if len(greedy_cluster_coverage) >= 20:
    print(f"  K=20:   {greedy_cluster_coverage[19][1]:.1%} coverage")
