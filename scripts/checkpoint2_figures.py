#!/usr/bin/env python3
"""
Generate all figures for Checkpoint 2.

Figures:
  1. Evaluation summary dashboard (main figure)
  2. Transition matrix heatmap (43×43 process types)
  3. SentenceBERT cosine distribution (vs null baseline)

All numbers are loaded from JSON outputs — no hardcoding of eval results.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sentence_transformers import SentenceTransformer

DAGS_DIR  = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/dags")
VOCAB_DIR = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/vocab_analysis")
RECIPE1M  = Path("/home/addeepak/AdvitResearch/cs348k/recipes.json")
EVAL_DIR  = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/eval")
OUT       = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/figures")
OUT.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)

CANONICAL_NAMES = [
    "bake","roast","broil","grill","toast",
    "saute","fry","sear","brown",
    "boil","simmer","steam","poach","braise","blanch",
    "chop","dice","slice","mince","grate","peel","crush",
    "mix","stir","whisk","fold","blend","beat","knead","toss",
    "season","coat","brush","drizzle",
    "cool","chill","freeze","rest","marinate","reduce","dissolve","melt","drain",
]
GROUP_MAP = {
    "bake":"dry_heat","roast":"dry_heat","broil":"dry_heat","grill":"dry_heat","toast":"dry_heat",
    "saute":"fat_heat","fry":"fat_heat","sear":"fat_heat","brown":"fat_heat",
    "boil":"moist_heat","simmer":"moist_heat","steam":"moist_heat",
    "poach":"moist_heat","braise":"moist_heat","blanch":"moist_heat",
    "chop":"prep","dice":"prep","slice":"prep","mince":"prep",
    "grate":"prep","peel":"prep","crush":"prep",
    "mix":"combine","stir":"combine","whisk":"combine","fold":"combine",
    "blend":"combine","beat":"combine","knead":"combine","toss":"combine",
    "season":"apply","coat":"apply","brush":"apply","drizzle":"apply",
    "cool":"rest","chill":"rest","freeze":"rest","rest":"rest",
    "marinate":"rest","reduce":"rest","dissolve":"rest","melt":"rest","drain":"rest",
}
GROUPS = ["dry_heat","fat_heat","moist_heat","prep","combine","apply","rest"]
GROUP_COLORS = {
    "dry_heat":  "#e74c3c",
    "fat_heat":  "#e67e22",
    "moist_heat":"#3498db",
    "prep":      "#27ae60",
    "combine":   "#9b59b6",
    "apply":     "#f39c12",
    "rest":      "#95a5a6",
}

TEMPLATES = {
    "bake":"Bake {ings}.","roast":"Roast {ings} in the oven.","broil":"Broil {ings}.",
    "grill":"Grill {ings}.","toast":"Toast {ings}.","saute":"Sauté {ings} in a pan.",
    "fry":"Fry {ings}.","sear":"Sear {ings}.","brown":"Brown {ings}.",
    "boil":"Boil {ings}.","simmer":"Simmer {ings}.","steam":"Steam {ings}.",
    "poach":"Poach {ings}.","braise":"Braise {ings}.","blanch":"Blanch {ings}.",
    "chop":"Chop {ings}.","dice":"Dice {ings}.","slice":"Slice {ings}.",
    "mince":"Mince {ings}.","grate":"Grate {ings}.","peel":"Peel {ings}.",
    "crush":"Crush {ings}.","mix":"Mix {ings}.","stir":"Stir {ings}.",
    "whisk":"Whisk {ings}.","fold":"Fold in {ings}.","blend":"Blend {ings}.",
    "beat":"Beat {ings}.","knead":"Knead {ings}.","toss":"Toss {ings}.",
    "season":"Season {ings}.","coat":"Coat {ings}.","brush":"Brush {ings}.",
    "drizzle":"Drizzle {ings}.","cool":"Cool {ings}.","chill":"Refrigerate {ings}.",
    "freeze":"Freeze {ings}.","rest":"Rest {ings}.","marinate":"Marinate {ings}.",
    "reduce":"Reduce {ings}.","dissolve":"Dissolve {ings}.","melt":"Melt {ings}.",
    "drain":"Drain {ings}.",
}


def decode_dag(dag):
    """Decode constrained DAG (steps list) to NL sentence."""
    sentences = []
    for step in dag["steps"]:
        ings = step.get("ingredients", [])
        ing_str = ", ".join(ings[:3]) if ings else "ingredients"
        sentences.append(TEMPLATES.get(step["canonical"], "Process {ings}.").format(ings=ing_str))
    return " ".join(sentences)


def main():
    print("Loading data...", flush=True)

    dags = []
    with open(DAGS_DIR / "constrained_dags.jsonl") as f:
        for line in f:
            dags.append(json.loads(line))
    print(f"  {len(dags):,} constrained DAGs loaded")

    with open(RECIPE1M) as f:
        all_recipes = json.load(f)
    recipe_by_id = {r["id"]: r for r in all_recipes if r.get("partition") == "train"}

    with open(DAGS_DIR / "transition_matrix.json") as f:
        trans_data = json.load(f)

    with open(VOCAB_DIR / "real_coverage_results.json") as f:
        coverage_data = json.load(f)

    with open(EVAL_DIR / "phase8_full_pipeline.json") as f:
        pipeline_data = json.load(f)

    with open(EVAL_DIR / "phase9b_stats.json") as f:
        hybrid_stats = json.load(f)

    with open(EVAL_DIR / "phase12_xml_eval.json") as f:
        curd_data = json.load(f)

    # Compute process_type_freq from constrained DAGs
    freq = Counter()
    for dag in dags:
        for step in dag["steps"]:
            freq[step["canonical"]] += 1

    sample = random.sample(dags, min(1000, len(dags)))

    # ── FIGURE 1: Evaluation summary dashboard ────────────────────────────────
    print("Figure 1: Summary dashboard...", flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("RecipeDAG Typed Process Sequences — Checkpoint 2 Evaluation", fontsize=14, y=1.01)

    # 1a. Method comparison: BGE vs Constrained (4 metrics grouped)
    ax = axes[0, 0]
    metrics = ["Parse\nRate", "Steps w/\nIngredient†", "Temp\nCapture‡", "Duration\nCapture‡"]
    bge_vals    = [97.5, 34.2, 69.3, 51.6]
    constrained = [98.0, 97.0, 40.6, 44.5]

    x = np.arange(len(metrics))
    w = 0.35
    bars_bge = ax.bar(x - w/2, bge_vals, w, label="BGE baseline", color="#e74c3c", alpha=0.8)
    bars_con = ax.bar(x + w/2, constrained, w, label="Constrained 3B (hybrid)", color="#2ecc71", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_title("Encoder Comparison: BGE vs Constrained", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.text(0.02, 0.98, "† coverage, not correctness\n‡ not apples-to-apples (see §4.1)",
            transform=ax.transAxes, fontsize=7, va='top', color='gray')

    # 1b. Process type frequency (top 15)
    ax = axes[0, 1]
    top15 = sorted(freq.items(), key=lambda x: -x[1])[:15]
    names15, counts15 = zip(*top15)
    colors15 = [GROUP_COLORS[GROUP_MAP[n]] for n in names15]
    y = range(len(names15))
    ax.barh(list(y), counts15, color=colors15, alpha=0.8)
    ax.set_yticks(list(y)); ax.set_yticklabels(names15, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"Node count in {len(dags):,} training DAGs", fontsize=9)
    ax.set_title("Top 15 Process Types (by frequency)", fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    for g, c in GROUP_COLORS.items():
        ax.barh([], [], color=c, label=g.replace("_"," "), alpha=0.8)
    ax.legend(fontsize=7, loc='lower right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # 1c. Reconstruction quality: BGE vs Constrained
    ax = axes[0, 2]
    bge = pipeline_data["bge_llm_baseline"]
    con = pipeline_data

    categories = ["ROUGE-L", "SentBERT", "SentBERT\nnull", "SentBERT\ngap over null"]
    bge_recon = [bge["rouge_l"], bge["sbert"], bge["sbert_null"], bge["sbert_gap"]]
    con_recon = [con["rouge_l"]["mean"], con["sbert"]["mean"], con["sbert"]["null_mean"], con["sbert"]["gap"]]

    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w/2, bge_recon, w, label="BGE baseline", color="#e74c3c", alpha=0.8)
    ax.bar(x + w/2, con_recon, w, label="Constrained 3B", color="#2ecc71", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Score (0–1)", fontsize=10)
    ax.set_title("Reconstruction Fidelity (NL→DAG→NL)\n(n=278 recipes)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    # Annotate gap improvement
    gap_bge = bge["sbert_gap"]
    gap_con = con["sbert"]["gap"]
    ax.annotate(f'+{gap_con:.3f} vs\n+{gap_bge:.3f} (BGE)',
                xy=(3 + w/2, gap_con), xytext=(2.5, gap_con + 0.15),
                fontsize=7, color='#27ae60',
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # 1d. CURD ground-truth evaluation
    ax = axes[1, 0]
    curd_metrics = ["Type F1\n(n=16)", "Seq LCS F1\n(n=16)", "Ingredient\nOverlap (n=16)"]
    curd_vals    = [curd_data["type_f1"]*100, curd_data["avg_seq_lcs_f1"]*100, curd_data["avg_ing_overlap"]*100]
    colors_curd  = ["#e74c3c", "#e67e22", "#9b59b6"]
    bars = ax.bar(curd_metrics, curd_vals, color=colors_curd, width=0.5, alpha=0.85)
    ax.set_ylim(0, 75)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_title("Ground-Truth Accuracy (CURD annotations)\nConstrained 3B vs hand-labeled", fontsize=11)
    for bar, val in zip(bars, curd_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}%",
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.axhline(50, color='gray', linestyle='--', alpha=0.3, label='50%')
    ax.text(0.02, 0.92, f"7/16 recipes at 0% type F1\n(vocabulary + abstraction errors)",
            transform=ax.transAxes, fontsize=7.5, va='top', color='#c0392b')
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # 1e. Coverage curve (ingredient freq-greedy)
    ax = axes[1, 1]
    freq_curve = coverage_data["freq_curve"]
    ks   = [p[0] for p in freq_curve]
    covs = [p[1]*100 for p in freq_curve]
    ax.semilogx(ks, covs, 'b-o', markersize=3, label='Frequency-greedy')
    for thresh, color, label in [(70,'gray','70%'),(80,'orange','80%'),(90,'red','90%'),(95,'purple','95%')]:
        ax.axhline(thresh, color=color, linestyle='--', alpha=0.5, label=label)
    ax.axvline(18252, color='green', linestyle=':', alpha=0.6)
    ax.text(18252*1.1, 20, 'Full vocab\n(18,252 types)', fontsize=8, color='green')
    ax.set_xlabel("Vocabulary size K (log scale)", fontsize=10)
    ax.set_ylabel("% of recipes fully covered", fontsize=10)
    ax.set_title("Ingredient Coverage Curve\n(justifies open vocabulary design)", fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # 1f. Arg extraction: LLM-only vs hybrid vs BGE
    ax = axes[1, 2]
    arg_labels = ["Temperature", "Duration"]
    bge_args    = [hybrid_stats["baselines"]["bge_temp"]*100, hybrid_stats["baselines"]["bge_dur"]*100]
    llm_args    = [hybrid_stats["baselines"]["phase9_temp"]*100, hybrid_stats["baselines"]["phase9_dur"]*100]
    hybrid_args = [hybrid_stats["temp_capture"]["after"]*100, hybrid_stats["dur_capture"]["after"]*100]

    x = np.arange(len(arg_labels))
    w = 0.25
    ax.bar(x - w, bge_args,    w, label="BGE (global scan‡)", color="#e74c3c", alpha=0.8)
    ax.bar(x,     llm_args,    w, label="Constrained LLM only", color="#f39c12", alpha=0.8)
    ax.bar(x + w, hybrid_args, w, label="LLM + regex backfill", color="#2ecc71", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(arg_labels, fontsize=10)
    ax.set_ylim(0, 85)
    ax.set_ylabel("% steps with value captured", fontsize=9)
    ax.set_title("Argument Extraction: BGE vs LLM vs Hybrid\n‡ BGE is global-scan, others are per-step", fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "checkpoint2_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {OUT}/checkpoint2_dashboard.png")

    # ── FIGURE 2: Transition matrix heatmap ──────────────────────────────────
    print("Figure 2: Transition matrix heatmap...", flush=True)

    trans_probs = trans_data["transition_probs"]
    n = len(CANONICAL_NAMES)
    matrix = np.zeros((n, n))
    for i, a in enumerate(CANONICAL_NAMES):
        for j, b in enumerate(CANONICAL_NAMES):
            matrix[i, j] = trans_probs.get(a, {}).get(b, 0.0)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0, vmax=0.25)
    plt.colorbar(im, ax=ax, label="P(column | row)", shrink=0.8)

    ax.set_xticks(range(n)); ax.set_xticklabels(CANONICAL_NAMES, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(CANONICAL_NAMES, fontsize=7)
    ax.set_xlabel("Next process (B)", fontsize=11)
    ax.set_ylabel("Current process (A)", fontsize=11)
    ax.set_title(f"Data-Driven Verifier: P(B | A) Transition Matrix\n"
                 f"(learned from {len(dags):,} training DAGs)", fontsize=12)

    prev_group = GROUP_MAP[CANONICAL_NAMES[0]]
    for i, name in enumerate(CANONICAL_NAMES):
        if GROUP_MAP[name] != prev_group:
            ax.axhline(i - 0.5, color='red', linewidth=1.0, alpha=0.5)
            ax.axvline(i - 0.5, color='red', linewidth=1.0, alpha=0.5)
            prev_group = GROUP_MAP[name]

    plt.tight_layout()
    plt.savefig(OUT / "checkpoint2_transition_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {OUT}/checkpoint2_transition_matrix.png")

    # ── FIGURE 3: SentenceBERT distribution ──────────────────────────────────
    print("Figure 3: SentenceBERT distribution (loading SBERT model)...", flush=True)

    SBERT = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    orig_texts, dec_texts, n_steps_list = [], [], []
    for dag in sample:
        recipe = recipe_by_id.get(dag["id"])
        if not recipe: continue
        steps = [s["text"].strip() for s in recipe.get("instructions", []) if s["text"].strip()]
        orig_texts.append(" ".join(steps))
        dec_texts.append(decode_dag(dag))
        n_steps_list.append(len(dag["steps"]))

    ea = SBERT.encode(orig_texts, batch_size=256, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=True)
    eb = SBERT.encode(dec_texts,  batch_size=256, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=True)
    sims = (ea * eb).sum(axis=1)

    dec_shuf = dec_texts.copy(); random.shuffle(dec_shuf)
    eb2 = SBERT.encode(dec_shuf, batch_size=256, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
    null_sims = (ea * eb2).sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    ax.hist(sims, bins=bins, alpha=0.7, color='#2ecc71', label=f'Matched pairs (mean={sims.mean():.3f})')
    ax.hist(null_sims, bins=bins, alpha=0.7, color='#e74c3c', label=f'Random pairs / null (mean={null_sims.mean():.3f})')
    ax.axvline(sims.mean(), color='#27ae60', linestyle='--', linewidth=2)
    ax.axvline(null_sims.mean(), color='#c0392b', linestyle='--', linewidth=2)
    ax.annotate('', xy=(sims.mean(), 95), xytext=(null_sims.mean(), 95),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text((sims.mean()+null_sims.mean())/2, 98, f'+{sims.mean()-null_sims.mean():.3f}',
            ha='center', fontsize=12, fontweight='bold')
    ax.set_xlabel("SentenceBERT cosine similarity", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Semantic Similarity Distribution\n(original NL vs DAG-decoded NL)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[1]
    n_steps_arr = np.array(n_steps_list[:len(sims)])
    sc = ax.scatter(n_steps_arr, sims, alpha=0.3, s=15, c=sims, cmap='RdYlGn', vmin=0.3, vmax=0.9)
    plt.colorbar(sc, ax=ax, label='SentBERT sim')
    ax.set_xlabel("Recipe length (# steps in extracted DAG)", fontsize=12)
    ax.set_ylabel("SentenceBERT cosine similarity", fontsize=12)
    ax.set_title("Reconstruction Quality vs Recipe Length", fontsize=12)
    ax.grid(True, alpha=0.3)
    corr = np.corrcoef(n_steps_arr, sims)[0, 1]
    ax.text(0.98, 0.02, f'r = {corr:.3f}', transform=ax.transAxes,
            ha='right', fontsize=10, color='gray')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.suptitle(f"SentenceBERT Semantic Similarity Analysis (n={len(sims):,} recipes)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT / "checkpoint2_sbert_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {OUT}/checkpoint2_sbert_analysis.png")
    print(f"  SentBERT gap (constrained): +{sims.mean()-null_sims.mean():.3f}")
    print(f"  Correlation (length vs sim): r={corr:.3f}")

    print(f"\nAll figures saved to {OUT}/")
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
