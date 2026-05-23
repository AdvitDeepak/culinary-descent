#!/usr/bin/env python3
"""
Phase 2b: Lock in the process taxonomy using a large embedding model.

Steps:
1. Embed the top-N corpus verbs with BAAI/bge-large-en-v1.5 (large model, GPU)
2. Cluster into semantic groups to validate our proposed taxonomy
3. Map all 3,059 corpus verbs → canonical process type (nearest neighbor)
4. Show coverage of proposed taxonomy over Recipe1M
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

OUT = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/vocab_analysis")

# ---------------------------------------------------------------------------
# Proposed process taxonomy — 35 canonical process types across 7 groups
# Each entry: (canonical_name, group, required_args, optional_args, notes)
# ---------------------------------------------------------------------------
TAXONOMY = [
    # ── GROUP 1: DRY HEAT (oven / direct heat) ──────────────────────────
    ("bake",       "dry_heat",   ["temp_c", "duration_min"], [],             "oven, enclosed dry heat"),
    ("roast",      "dry_heat",   ["temp_c", "duration_min"], [],             "oven, high temp, uncovered"),
    ("broil",      "dry_heat",   ["duration_min"],           [],             "direct overhead heat"),
    ("grill",      "dry_heat",   ["duration_min"],           [],             "direct flame/coals from below"),
    ("toast",      "dry_heat",   ["duration_min"],           [],             "dry heat, bread/nuts/spices"),

    # ── GROUP 2: STOVETOP FAT HEAT ───────────────────────────────────────
    ("saute",      "fat_heat",   ["duration_min"],           [],             "pan + fat, medium-high heat"),
    ("fry",        "fat_heat",   ["duration_min"],           ["temp_c"],     "shallow or deep fat"),
    ("sear",       "fat_heat",   ["duration_min"],           [],             "very high heat, short time, crust"),
    ("brown",      "fat_heat",   ["duration_min"],           [],             "Maillard reaction, stovetop"),

    # ── GROUP 3: MOIST HEAT (liquid medium) ─────────────────────────────
    ("boil",       "moist_heat", ["duration_min"],           [],             "100°C, rolling boil"),
    ("simmer",     "moist_heat", ["duration_min"],           [],             "below boil, gentle bubbles"),
    ("steam",      "moist_heat", ["duration_min"],           [],             "steam vapor as medium"),
    ("poach",      "moist_heat", ["duration_min"],           ["temp_c"],     "gentle submersion in liquid"),
    ("braise",     "moist_heat", ["duration_min"],           [],             "partial liquid, long low heat"),
    ("blanch",     "moist_heat", ["duration_min"],           [],             "brief boil + ice bath"),

    # ── GROUP 4: MECHANICAL PREP ─────────────────────────────────────────
    ("chop",       "prep",       [],                         [],             "rough cut, no size spec"),
    ("dice",       "prep",       [],                         ["size_mm"],    "uniform cubes"),
    ("slice",      "prep",       [],                         ["thickness_mm"],"flat cuts"),
    ("mince",      "prep",       [],                         [],             "very fine cut"),
    ("grate",      "prep",       [],                         [],             "shred against grater"),
    ("peel",       "prep",       [],                         [],             "remove outer skin/rind"),
    ("crush",      "prep",       [],                         [],             "smash (garlic, peppercorns)"),

    # ── GROUP 5: COMBINE / MIX ───────────────────────────────────────────
    ("mix",        "combine",    [],                         ["duration_min"],"general mixing"),
    ("stir",       "combine",    [],                         ["duration_min"],"mixing during cooking"),
    ("whisk",      "combine",    [],                         ["duration_min"],"incorporate air, smooth"),
    ("fold",       "combine",    [],                         [],             "gentle incorporation, no deflate"),
    ("blend",      "combine",    [],                         ["duration_sec"],"machine blending, smooth"),
    ("beat",       "combine",    [],                         ["duration_min"],"eggs, cream — vigorous"),
    ("knead",      "combine",    ["duration_min"],           [],             "bread dough — gluten develop"),
    ("toss",       "combine",    [],                         [],             "coat lightly, salad/pasta"),

    # ── GROUP 6: APPLY / COAT ────────────────────────────────────────────
    ("season",     "apply",      [],                         [],             "add salt/pepper/spices"),
    ("coat",       "apply",      [],                         [],             "cover surface evenly"),
    ("brush",      "apply",      [],                         [],             "apply liquid with brush"),
    ("drizzle",    "apply",      [],                         [],             "pour in thin stream"),

    # ── GROUP 7: REST / STORE / CHANGE STATE ─────────────────────────────
    ("cool",       "rest",       [],                         ["duration_min"],"bring to room temp"),
    ("chill",      "rest",       [],                         ["duration_min"],"refrigerate until cold"),
    ("freeze",     "rest",       [],                         ["duration_min"],"bring below 0°C"),
    ("rest",       "rest",       [],                         ["duration_min"],"hold off heat (meat, dough)"),
    ("marinate",   "rest",       ["duration_min"],           [],             "soak in flavored liquid"),
    ("reduce",     "rest",       [],                         ["pct"],        "boil down liquid volume"),
    ("dissolve",   "rest",       [],                         [],             "solid → liquid (sugar, gelatin)"),
    ("melt",       "rest",       [],                         [],             "solid → liquid via heat"),
    ("drain",      "rest",       [],                         [],             "remove liquid from solid"),
]

CANONICAL_NAMES = [t[0] for t in TAXONOMY]
GROUP_MAP = {t[0]: t[1] for t in TAXONOMY}
GROUPS = sorted(set(t[1] for t in TAXONOMY))


def main():
    print("Loading corpus verb frequencies...")
    with open(OUT / "process_vocab.json") as f:
        data = json.load(f)
    verb_counts = data["verb_counts"]
    all_verbs = list(verb_counts.keys())
    top_verbs = all_verbs[:500]   # top-500 for embedding
    print(f"  {len(all_verbs):,} total verbs, embedding top {len(top_verbs)}")

    # ── Embed with large model ──────────────────────────────────────────
    print("\nEmbedding with BAAI/bge-large-en-v1.5 (GPU)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

    corpus_embs = model.encode(
        top_verbs, batch_size=256, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    canonical_embs = model.encode(
        CANONICAL_NAMES, batch_size=256,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    np.save(OUT / "process_corpus_embs.npy", corpus_embs)
    np.save(OUT / "process_canonical_embs.npy", canonical_embs)
    print(f"  corpus_embs: {corpus_embs.shape}, canonical_embs: {canonical_embs.shape}")

    # ── Map each corpus verb → nearest canonical type ───────────────────
    print("\nMapping corpus verbs → canonical process types...")
    # sim: (len(top_verbs), len(CANONICAL_NAMES))
    sims = corpus_embs @ canonical_embs.T
    nearest_idx = np.argmax(sims, axis=1)
    nearest_score = np.max(sims, axis=1)

    verb_to_canonical = {}
    canonical_to_verbs = defaultdict(list)
    for verb, idx, score in zip(top_verbs, nearest_idx, nearest_score):
        canon = CANONICAL_NAMES[int(idx)]
        verb_to_canonical[verb] = (canon, float(score))
        canonical_to_verbs[canon].append((verb, float(score), verb_counts.get(verb, 0)))

    print("\nCanonical process → top mapped corpus verbs:")
    for canon_name, group, req_args, opt_args, note in TAXONOMY:
        mapped = sorted(canonical_to_verbs[canon_name], key=lambda x: -x[2])[:8]
        arg_str = ", ".join(req_args + [f"{a}?" for a in opt_args])
        print(f"\n  {canon_name}({arg_str})  [{group}]  — {note}")
        for verb, score, count in mapped:
            marker = "★" if verb == canon_name else " "
            print(f"    {marker} {verb:<20} sim={score:.3f}  n={count:,}")

    # ── Coverage: what % of train recipes have ALL processes in taxonomy ─
    print("\n--- Coverage of proposed taxonomy over Recipe1M train ---")
    recipe_verb_sets = []
    for recipe_data in data.get("coverage_curve", []):
        pass  # coverage_curve doesn't have per-recipe data

    # Recompute from process_vocab coverage_curve data instead
    # Use the coverage curve we already computed but now with canonical mapping
    curve = data["coverage_curve"]
    # Each recipe's "coverage" under the taxonomy = did we map ALL its verbs?
    # We need the per-recipe verb sets — reconstruct from what we saved
    # (We saved verb_counts but not per-recipe sets — use curve as proxy)
    # Best proxy: at M=35 canonical types, what % is covered?
    # From the freq coverage curve at M=35 we got ~22%
    # With canonical mapping, verbs like "stir" map to "stir" → helps;
    # "desire" maps to something → every recipe now maps better
    print("  (Per-recipe sets not saved; showing canonical similarity stats)")
    print(f"  Taxonomy size: {len(CANONICAL_NAMES)} canonical process types")
    print(f"  Groups: {GROUPS}")

    scores_all = nearest_score
    print(f"  Avg mapping similarity (top 500 verbs): {scores_all.mean():.3f}")
    print(f"  % with sim > 0.6: {(scores_all > 0.6).mean():.1%}")
    print(f"  % with sim > 0.5: {(scores_all > 0.5).mean():.1%}")
    print(f"  % with sim > 0.4: {(scores_all > 0.4).mean():.1%}")

    # ── Validate taxonomy with K-means (does data confirm our groupings?) ─
    print("\n--- Validating taxonomy groups via k-means (K=7) ---")
    km = MiniBatchKMeans(n_clusters=7, random_state=42, n_init=20)
    labels = km.fit_predict(canonical_embs)
    sims_c = km.cluster_centers_ @ canonical_embs.T
    nearest_c = np.argmax(sims_c, axis=1)

    cluster_to_processes = defaultdict(list)
    for name, label in zip(CANONICAL_NAMES, labels):
        cluster_to_processes[int(label)].append(name)
    print("  Data-driven clusters vs our groups:")
    for cl, members in sorted(cluster_to_processes.items()):
        groups_in_cluster = [GROUP_MAP[m] for m in members]
        dominant = Counter(groups_in_cluster).most_common(1)[0][0]
        print(f"  Cluster {cl} (dominant={dominant}): {members}")

    # ── Print final taxonomy table ───────────────────────────────────────
    print("\n" + "="*70)
    print("FINAL PROCESS TAXONOMY")
    print("="*70)
    current_group = None
    for canon_name, group, req_args, opt_args, note in TAXONOMY:
        if group != current_group:
            print(f"\n  ── {group.upper()} ──")
            current_group = group
        req = ", ".join(req_args) if req_args else ""
        opt = ", ".join(f"{a}?" for a in opt_args) if opt_args else ""
        args = ", ".join(filter(None, [req, opt]))
        print(f"  {canon_name}({args})")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: canonical types colored by group, x=sim to nearest corpus verb
    ax = axes[0]
    colors = plt.cm.Set1(np.linspace(0, 1, len(GROUPS)))
    group_color = {g: c for g, c in zip(GROUPS, colors)}
    # Embed in 2D with UMAP if available, else PCA
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42)
        all_embs = np.vstack([canonical_embs, corpus_embs[:50]])
        coords = reducer.fit_transform(all_embs)
        can_coords = coords[:len(CANONICAL_NAMES)]
        corp_coords = coords[len(CANONICAL_NAMES):]
        method = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        all_embs = np.vstack([canonical_embs, corpus_embs[:50]])
        coords = PCA(n_components=2).fit_transform(all_embs)
        can_coords = coords[:len(CANONICAL_NAMES)]
        corp_coords = coords[len(CANONICAL_NAMES):]
        method = "PCA"

    for name, group, *_ in TAXONOMY:
        idx = CANONICAL_NAMES.index(name)
        c = group_color[group]
        ax.scatter(*can_coords[idx], color=c, s=120, zorder=3)
        ax.annotate(name, can_coords[idx], fontsize=7, ha='center',
                    xytext=(0, 5), textcoords='offset points')
    # Add corpus top-50 as small dots
    for i, verb in enumerate(top_verbs[:50]):
        ax.scatter(*corp_coords[i], color='gray', s=20, alpha=0.4, zorder=1)
        ax.annotate(verb, corp_coords[i], fontsize=5, color='gray', alpha=0.6)
    # Legend
    for group, color in group_color.items():
        ax.scatter([], [], color=color, label=group, s=80)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(f"Process Taxonomy + Top-50 Corpus Verbs ({method})", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # Right: frequency bar of top canonical-mapped counts
    ax = axes[1]
    canon_freq = Counter()
    for verb, (canon, score) in verb_to_canonical.items():
        canon_freq[canon] += verb_counts.get(verb, 0)
    sorted_canon = sorted(canon_freq.items(), key=lambda x: -x[1])
    names_bar, counts_bar = zip(*sorted_canon)
    colors_bar = [group_color[GROUP_MAP[n]] for n in names_bar]
    y = range(len(names_bar))
    ax.barh(list(y), counts_bar, color=colors_bar, alpha=0.8)
    ax.set_yticks(list(y)); ax.set_yticklabels(names_bar, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative frequency in Recipe1M (top-500 verbs mapped)", fontsize=10)
    ax.set_title("Process Frequency (mapped to canonical types)", fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    # Add group legend
    for group, color in group_color.items():
        ax.barh([], [], color=color, label=group, alpha=0.8)
    ax.legend(fontsize=8)

    plt.suptitle("Recipe1M Process Taxonomy — 35 Canonical Types × 7 Groups", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT / "process_taxonomy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved → {OUT}/process_taxonomy.png")

    # Save mapping
    with open(OUT / "process_taxonomy.json", "w") as f:
        json.dump({
            "taxonomy": [
                {"name": t[0], "group": t[1], "required_args": t[2],
                 "optional_args": t[3], "note": t[4]}
                for t in TAXONOMY
            ],
            "verb_to_canonical": {
                v: {"canonical": c, "similarity": s}
                for v, (c, s) in verb_to_canonical.items()
            },
        }, f, indent=2)
    print(f"Saved → {OUT}/process_taxonomy.json")


if __name__ == "__main__":
    main()
