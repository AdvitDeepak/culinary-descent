#!/usr/bin/env python3
"""
DAG corpus analysis — no GPU needed.
Runs over constrained_dags.jsonl to produce:
  1. Top process-sequence templates (most common recipe "shapes")
  2. Structural similarity retrieval (query → top-5 matches by process LCS)
  3. Complexity distribution stats

Outputs: data/eval/dag_analysis.json, data/figures/dag_corpus_analysis.png
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DAGS_DIR = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/dags")
FIGURES  = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/figures")
OUT      = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/eval")

print("Loading DAGs...", flush=True)
dags = []
with open(DAGS_DIR / "constrained_dags.jsonl") as f:
    for line in f:
        dags.append(json.loads(line))
print(f"  {len(dags):,} constrained DAGs loaded")


def proc_seq(dag):
    return tuple(s["canonical"] for s in dag["steps"])


def n_unique_ings(dag):
    return len({ing.strip().lower() for s in dag["steps"] for ing in s.get("ingredients", [])})


def lcs_f1(a, b):
    m, n = len(a), len(b)
    if not m or not n: return 0.0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p, r = lcs/m, lcs/n
    return 2*p*r/(p+r) if p+r else 0.0


# ── 1. Process sequence templates ─────────────────────────────────────────────
print("\n[1/3] Computing process sequence templates...", flush=True)
seqs = [proc_seq(d) for d in dags]
seq_counts = Counter(seqs)
top_templates = seq_counts.most_common(15)

print(f"\n  Unique sequences:     {len(seq_counts):,}")
print(f"  Top-1 covers:         {top_templates[0][1]/len(dags):.2%} of corpus")
print(f"  Top-10 cover:         {sum(c for _,c in top_templates[:10])/len(dags):.2%} of corpus")
print(f"\n  Top 10 recipe templates:")
for seq, count in top_templates[:10]:
    pct = count / len(dags) * 100
    print(f"    {' → '.join(seq):<55}  {count:5d}  ({pct:.2f}%)")

counts_sorted = sorted(seq_counts.values(), reverse=True)
cumsum = np.cumsum(counts_sorted) / len(dags)
n_to_cover = {
    0.50: int(np.searchsorted(cumsum, 0.50)) + 1,
    0.80: int(np.searchsorted(cumsum, 0.80)) + 1,
    0.90: int(np.searchsorted(cumsum, 0.90)) + 1,
    0.95: int(np.searchsorted(cumsum, 0.95)) + 1,
}
print(f"\n  Sequences needed to cover:")
for pct, n in n_to_cover.items():
    print(f"    {pct:.0%} of corpus → {n:,} unique sequences")

# ── 2. Structural similarity retrieval ────────────────────────────────────────
print("\n[2/3] Structural similarity retrieval demo...", flush=True)
random.seed(42)

candidates = [d for d in dags if 3 <= len(d["steps"]) <= 6]
queries = random.sample(candidates, 5)

retrieval_results = []
print(f"  Scanning {len(dags):,} DAGs for each query...")

for q in queries:
    q_seq = proc_seq(q)
    scores = []
    for d in dags:
        if d["id"] == q["id"]: continue
        score = lcs_f1(q_seq, proc_seq(d))
        scores.append((score, d))
    scores.sort(key=lambda x: -x[0])
    top5 = scores[:5]

    result = {
        "query_title": q.get("title", ""),
        "query_seq": list(q_seq),
        "matches": [
            {"title": d.get("title",""), "seq": list(proc_seq(d)), "lcs": round(s,3)}
            for s, d in top5
        ],
    }
    retrieval_results.append(result)

    print(f"\n  Query: \"{q.get('title','')}\"")
    print(f"    Sequence: {' → '.join(q_seq)}")
    print(f"    Top matches:")
    for m in result["matches"][:3]:
        print(f"      [{m['lcs']:.2f}] \"{m['title']}\"  ({' → '.join(m['seq'])})")

# ── 3. Complexity distribution ─────────────────────────────────────────────────
print("\n[3/3] Complexity distribution...", flush=True)
n_proc = [len(d["steps"]) for d in dags]
n_ings = [n_unique_ings(d) for d in dags]

print(f"  Process nodes: mean={np.mean(n_proc):.1f}  median={np.median(n_proc):.0f}  "
      f"p90={np.percentile(n_proc,90):.0f}  max={max(n_proc)}")
print(f"  Unique ings:   mean={np.mean(n_ings):.1f}  median={np.median(n_ings):.0f}  "
      f"p90={np.percentile(n_ings,90):.0f}  max={max(n_ings)}")

# ── Figures ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(f"RecipeDAG Corpus Analysis (n={len(dags):,} constrained DAGs)", fontsize=12, fontweight="bold")

ax = axes[0]
labels = [" → ".join(s) for s, _ in top_templates[:8]]
labels = [l if len(l) < 40 else l[:38]+"…" for l in labels]
vals   = [c for _, c in top_templates[:8]]
ax.barh(range(len(labels)), vals, color="#2171b5")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("Recipe count")
ax.set_title("Most common process sequences", fontsize=10)
ax.spines[["top","right"]].set_visible(False)

ax = axes[1]
xs = np.arange(1, min(len(counts_sorted)+1, 5001))
ax.plot(xs, cumsum[:5000]*100, color="#2171b5", lw=2)
for pct, n in n_to_cover.items():
    if n <= 5000:
        ax.axhline(pct*100, color="gray", lw=0.8, ls="--")
        ax.axvline(n, color="gray", lw=0.8, ls="--")
        ax.text(n+30, pct*100-2, f"{n:,} seqs\n→ {pct:.0%}", fontsize=7, color="gray")
ax.set_xlabel("# unique sequences (ranked by frequency)")
ax.set_ylabel("% corpus covered")
ax.set_title("Process sequence coverage curve", fontsize=10)
ax.set_xlim(0, 5000)
ax.set_ylim(0, 101)
ax.spines[["top","right"]].set_visible(False)

ax = axes[2]
ax.hist(n_proc, bins=range(1, 16), color="#2171b5", edgecolor="white", lw=0.5, density=True)
ax.axvline(np.mean(n_proc), color="#e6550d", lw=1.5, ls="--", label=f"mean={np.mean(n_proc):.1f}")
ax.set_xlabel("# process steps per recipe")
ax.set_ylabel("Density")
ax.set_title("Recipe complexity distribution", fontsize=10)
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(FIGURES / "dag_corpus_analysis.png", dpi=150, bbox_inches="tight")
print(f"\n  Saved → {FIGURES}/dag_corpus_analysis.png")

# ── Save JSON ─────────────────────────────────────────────────────────────────
results = {
    "n_dags": len(dags),
    "n_unique_sequences": len(seq_counts),
    "top_templates": [{"seq": list(s), "count": c, "pct": c/len(dags)} for s,c in top_templates],
    "coverage": {str(int(k*100))+"pct": v for k, v in n_to_cover.items()},
    "complexity": {
        "proc_nodes": {"mean": float(np.mean(n_proc)), "median": float(np.median(n_proc)),
                       "p90": float(np.percentile(n_proc,90))},
        "unique_ingredients": {"mean": float(np.mean(n_ings)), "median": float(np.median(n_ings)),
                                "p90": float(np.percentile(n_ings,90))},
    },
    "retrieval_examples": retrieval_results,
}
with open(OUT / "dag_analysis.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"  Saved → {OUT}/dag_analysis.json")
