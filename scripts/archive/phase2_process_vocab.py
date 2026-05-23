#!/usr/bin/env python3
"""
Phase 2: Process Vocabulary Discovery
--------------------------------------
Extract cooking verbs from Recipe1M instructions using spaCy dependency
parsing — no predefined list, just what actually appears in the corpus.

Goal: find the natural finite process vocabulary of Recipe1M.

Outputs:
  data/vocab_analysis/process_vocab.json   — all verb lemmas + frequencies
  data/vocab_analysis/process_curve.png    — operation coverage curve
  data/vocab_analysis/process_clusters.png — embedding clusters of process types
"""

import json
from collections import Counter
from pathlib import Path

import spacy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

RECIPE1M = Path("/home/addeepak/AdvitResearch/cs348k/recipes.json")
OUT = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/vocab_analysis")
OUT.mkdir(parents=True, exist_ok=True)

# spaCy: use the small model for speed, we only need lemmas + POS
NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

# Non-cooking verbs to filter out — structural/generic language
GENERIC_VERBS = {
    "be", "have", "do", "make", "use", "get", "go", "come", "take", "put",
    "let", "set", "keep", "give", "want", "need", "find", "look", "seem",
    "try", "call", "tell", "ask", "work", "feel", "leave", "bring", "move",
    "start", "continue", "ensure", "allow", "remain", "result", "yield",
    "note", "check", "watch", "test", "taste", "adjust", "enjoy", "serve",
    "prepare", "follow", "read", "turn", "place", "add", "remove", "transfer",
    "pour", "top", "fill", "line", "arrange", "layer",
}


def extract_verbs_from_step(text: str, doc) -> list[str]:
    """Return cooking verb lemmas from a parsed instruction step."""
    verbs = []
    for token in doc:
        if token.pos_ == "VERB" and not token.is_stop:
            lemma = token.lemma_.lower().strip()
            if (len(lemma) >= 3
                    and lemma.isalpha()
                    and lemma not in GENERIC_VERBS):
                verbs.append(lemma)
    return verbs


def main():
    print(f"Loading {RECIPE1M}...")
    with open(RECIPE1M) as f:
        all_recipes = json.load(f)
    train = [r for r in all_recipes if r.get("partition") == "train"]
    print(f"  {len(train):,} train recipes")

    # ---- Collect all instruction steps ----
    all_steps = []
    recipe_step_counts = []
    for recipe in train:
        steps = [s["text"] for s in recipe.get("instructions", []) if s["text"].strip()]
        all_steps.extend(steps)
        recipe_step_counts.append(len(steps))

    print(f"  {len(all_steps):,} instruction steps total")
    print(f"  {sum(recipe_step_counts)/len(recipe_step_counts):.1f} steps per recipe avg")

    # ---- spaCy batch parse ----
    print("\nParsing with spaCy (batch)...")
    verb_counts: Counter = Counter()
    recipe_verb_sets: list[set] = []   # one set per recipe

    step_idx = 0
    recipe_verb_buffer: list[set] = [set() for _ in train]
    r_idx = 0

    # Parse in batches for speed
    BATCH = 256
    all_docs = []
    for i in tqdm(range(0, len(all_steps), BATCH), desc="spaCy"):
        batch = all_steps[i:i+BATCH]
        docs = list(NLP.pipe(batch))
        all_docs.extend(docs)

    # Map docs back to recipes
    doc_idx = 0
    for r_i, recipe in enumerate(train):
        steps = recipe.get("instructions", [])
        recipe_verbs = set()
        for step in steps:
            if step["text"].strip():
                doc = all_docs[doc_idx]
                verbs = extract_verbs_from_step(step["text"], doc)
                for v in verbs:
                    verb_counts[v] += 1
                recipe_verbs.update(verbs)
                doc_idx += 1
        recipe_verb_sets.append(recipe_verbs)

    n_recipes = len(train)
    print(f"\n  Unique verb lemmas extracted: {len(verb_counts):,}")
    print(f"  Total verb mentions:          {sum(verb_counts.values()):,}")
    print(f"\n  Top 60 cooking verbs:")
    for verb, count in verb_counts.most_common(60):
        bar = "█" * (count // 1000)
        print(f"    {count:7,}  {verb:<20} {bar}")

    # ---- Operation coverage curve ----
    print("\n--- Operation coverage curve ---")
    sorted_verbs = [v for v, _ in verb_counts.most_common()]
    k_vals = list(range(1, min(len(sorted_verbs) + 1, 150)))

    curve = []
    for k in k_vals:
        vocab = set(sorted_verbs[:k])
        covered = sum(1 for s in recipe_verb_sets if s.issubset(vocab))
        cov = covered / n_recipes
        curve.append((k, cov))

    print(f"  Total unique verb types: {len(sorted_verbs)}")
    for target in [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]:
        hit = next(((k, c) for k, c in curve if c >= target), None)
        if hit:
            print(f"  {target:.0%} coverage: M = {hit[0]} process types")

    print(f"\n  Coverage by M:")
    for k, cov in curve:
        if k <= 20 or k % 10 == 0:
            print(f"    M={k:4d}: {cov:.1%}")

    # ---- Embed process types for clustering ----
    print("\n--- Embedding process types (GPU) ---")
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    # Use top-N verbs for embedding (all of them if < 500)
    top_verbs = sorted_verbs[:min(500, len(sorted_verbs))]
    embeddings = model.encode(
        top_verbs, batch_size=512, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    print(f"\n  Embedded {len(top_verbs)} verb types")

    # Cluster at a few K values to find natural groupings
    print("\n  K-means clusters of process types:")
    cluster_results = {}
    for k in [5, 9, 15, 20, 30]:
        if k > len(top_verbs):
            continue
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        # Find exemplar (nearest to centroid) for each cluster
        sims = km.cluster_centers_ @ embeddings.T
        nearest = np.argmax(sims, axis=1)
        # Group verbs by cluster
        clusters = {}
        for i, (verb, label) in enumerate(zip(top_verbs, labels)):
            clusters.setdefault(int(label), []).append(verb)
        # Sort clusters by size
        sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))
        exemplars = [top_verbs[int(nearest[k_])] for k_ in range(k)]
        cluster_results[k] = {
            "exemplars": exemplars,
            "clusters": {top_verbs[int(nearest[i])]: sorted_clusters[i][1]
                         for i in range(min(k, len(sorted_clusters)))},
        }
        print(f"\n  K={k} — exemplars: {exemplars}")
        for exemplar, members in list(cluster_results[k]["clusters"].items())[:k]:
            top_members = ", ".join(members[:8])
            print(f"    [{exemplar}]: {top_members}")

    # ---- Save ----
    results = {
        "n_train_recipes": n_recipes,
        "n_unique_verbs": len(verb_counts),
        "verb_counts": dict(verb_counts.most_common()),
        "coverage_curve": curve,
        "coverage_targets": {},
        "clusters": {str(k): v for k, v in cluster_results.items()},
    }
    for target in [0.70, 0.80, 0.90, 0.95, 0.99]:
        hit = next(((k, c) for k, c in curve if c >= target), None)
        if hit:
            results["coverage_targets"][f"{int(target*100)}pct"] = hit[0]
    with open(OUT / "process_vocab.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- Plots ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Coverage curve
    ax = axes[0]
    ks = [p[0] for p in curve]
    covs = [p[1] * 100 for p in curve]
    ax.plot(ks, covs, "b-o", markersize=4)
    for thresh, color, label in [
        (80, "orange", "80%"), (90, "red", "90%"), (95, "purple", "95%")
    ]:
        ax.axhline(thresh, color=color, linestyle="--", alpha=0.5, label=label)
        hit_k = next((k for k, c in zip(ks, covs) if c >= thresh), None)
        if hit_k:
            ax.axvline(hit_k, color=color, linestyle=":", alpha=0.4)
            ax.annotate(f"M={hit_k}", xy=(hit_k, thresh),
                        xytext=(hit_k + 2, thresh - 8), fontsize=8, color=color)
    ax.set_xlabel("Number of process types M", fontsize=12)
    ax.set_ylabel("% of train recipes fully covered", fontsize=12)
    ax.set_title("Process Vocabulary Coverage Curve\n(Recipe1M, 35K train recipes)", fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)

    # Frequency histogram (top 40)
    ax = axes[1]
    top40 = verb_counts.most_common(40)
    verbs_top, counts_top = zip(*top40)
    y_pos = range(len(verbs_top))
    ax.barh(list(y_pos), counts_top, color="steelblue", alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(verbs_top, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency in Recipe1M train set", fontsize=11)
    ax.set_title("Top 40 Cooking Process Types\n(spaCy verb lemma extraction)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Recipe1M Process Vocabulary Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT / "process_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {OUT}/process_curve.png")
    print(f"Saved → {OUT}/process_vocab.json")


if __name__ == "__main__":
    main()
