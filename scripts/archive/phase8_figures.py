#!/usr/bin/env python3
"""Generate comparison bar chart for phase8 (constrained vs BGE pipeline)."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/figures")

bge = dict(rouge_l=0.189, sbert=0.619, sbert_gap=0.222, judge_tp=0.013)
con = dict(rouge_l=0.216, sbert=0.673, sbert_gap=0.290, judge_tp=0.0625)

metrics = [
    ("ROUGE-L", "rouge_l", 1.0, False),
    ("SentBERT\nmean", "sbert", 1.0, False),
    ("SentBERT\ngap", "sbert_gap", 0.5, False),
    ("Judge TP\nrate", "judge_tp", 0.15, True),
]

fig, axes = plt.subplots(1, 4, figsize=(11, 4))
fig.suptitle(
    "BGE encoder vs. Constrained LLM encoder\n(shared Qwen2.5-3B decoder)",
    fontsize=12, fontweight="bold", y=1.02,
)

colors = {"BGE + LLM decoder": "#6baed6", "Constrained + LLM decoder": "#2171b5"}

for ax, (label, key, ymax, is_pct) in zip(axes, metrics):
    vals = [bge[key], con[key]]
    bars = ax.bar(
        [0, 1], vals,
        color=[colors["BGE + LLM decoder"], colors["Constrained + LLM decoder"]],
        width=0.5, edgecolor="white", linewidth=1.2,
    )
    for bar, v in zip(bars, vals):
        fmt = f"{v:.1%}" if is_pct else f"{v:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + ymax * 0.02,
            fmt, ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    delta = (con[key] - bge[key]) / bge[key] * 100
    ax.set_title(f"{label}\n(+{delta:.0f}%)", fontsize=10)
    ax.set_ylim(0, ymax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BGE", "Constrained"], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_visible(False)

# legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors["BGE + LLM decoder"], label="BGE + LLM decoder"),
    Patch(facecolor=colors["Constrained + LLM decoder"], label="Constrained + LLM decoder"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.08), fontsize=9, frameon=False)

plt.tight_layout()
out_path = OUT / "phase8_pipeline_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
