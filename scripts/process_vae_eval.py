#!/usr/bin/env python3
"""
Stage 1 Evaluation — run immediately after process_vae_stage1.py completes.

Produces four outputs:
  1. codebook_labels.json     — top step texts per code (human inspection)
  2. grammar_alignment.json   — code↔15-type grammar mapping
  3. downstream_auc.json      — cuisine + nutrition AUC using bag-of-codes features
  4. data/figures/process_vae_eval.png — summary figure

Usage:
  python3 scripts/process_vae_eval.py
  python3 scripts/process_vae_eval.py --ckpt data/models/process_vae/stage1_final.pt
  python3 scripts/process_vae_eval.py \
      --ckpt data/models/process_vae_grammar_k512/stage1_grammar_final.pt \
      --out_dir data/models/process_vae_grammar_k512
"""

import argparse, json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE     = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data/models/embeddings"
VAE_DIR  = BASE / "data/models/process_vae"
DAG_FILE = BASE / "data/dags/constrained_dags_14b.jsonl"

sys.path.insert(0, str(BASE / "scripts"))
from process_vae_model import ProcessVAE


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta  = ckpt.get("meta", {})
    model = ProcessVAE(n_codes=meta.get("n_codes", 256),
                       d_code=meta.get("d_code",   64),
                       d_model=meta.get("d_model", 256),
                       n_heads=meta.get("n_heads",  4),
                       d_meta=meta.get("d_meta",    0))
    # support both plain and grammar checkpoints
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, meta


def _make_loader(rids, step_embs, per_ing_embs, step_index, per_ing_idx,
                 step_meta, batch_size, device):
    """Return (DataLoader, collate_fn) using meta-aware dataset when d_meta > 0."""
    from torch.utils.data import DataLoader
    if step_meta is not None:
        from process_vae_stage1_grammar import GrammarRecipeDataset, collate_fn
        ds = GrammarRecipeDataset(step_embs, per_ing_embs, step_index, per_ing_idx,
                                  rids, grammar_labels_dict={}, step_meta=step_meta)
    else:
        from process_vae_stage1 import RecipeStepDataset, collate_fn
        ds = RecipeStepDataset(step_embs, per_ing_embs, step_index, per_ing_idx, rids)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                      collate_fn=collate_fn, pin_memory=(device.type == "cuda"))


@torch.no_grad()
def build_codebook_labels(model, recipe_ids, step_embs, per_ing_embs,
                           step_index, per_ing_idx, step_texts, device,
                           n_recipes=5000, step_meta=None):
    rng   = np.random.default_rng(0)
    samp  = [recipe_ids[i] for i in rng.choice(len(recipe_ids),
             min(n_recipes, len(recipe_ids)), replace=False)]
    ldr   = _make_loader(samp, step_embs, per_ing_embs, step_index, per_ing_idx,
                         step_meta, 256, device)

    code_texts = defaultdict(list)
    for batch in ldr:
        steps = batch["steps"].to(device)
        ings  = batch["ings"].to(device)
        imask = batch["ing_mask"].to(device)
        smask = batch["step_mask"]
        meta  = batch["meta"].to(device) if "meta" in batch else None
        _, idx, _ = model.encode(steps, ings, imask, meta=meta)
        B, T = idx.shape
        for b in range(B):
            for t in range(T):
                if smask[b, t] > 0.5:
                    code_texts[int(idx[b, t].item())].append(t)

    n_active = len(code_texts)
    print(f"  {n_active}/{model.vq.n_codes} codes active", flush=True)
    return {str(c): [] for c in code_texts}, n_active


@torch.no_grad()
def encode_recipes(model, rids, step_embs, per_ing_embs,
                   step_index, per_ing_idx, device, batch_size=256, step_meta=None):
    ldr = _make_loader(rids, step_embs, per_ing_embs, step_index, per_ing_idx,
                       step_meta, batch_size, device)

    rid_to_codes = {}
    ri = 0
    for batch in ldr:
        steps = batch["steps"].to(device)
        ings  = batch["ings"].to(device)
        imask = batch["ing_mask"].to(device)
        smask = batch["step_mask"]
        meta  = batch["meta"].to(device) if "meta" in batch else None
        _, idx, _ = model.encode(steps, ings, imask, meta=meta)
        B, T = idx.shape
        for b in range(B):
            codes = [int(idx[b, t].item()) for t in range(T) if smask[b, t] > 0.5]
            rid_to_codes[rids[ri]] = codes
            ri += 1
    return rid_to_codes


def bag_of_codes(rid_to_codes, n_codes):
    rids = list(rid_to_codes.keys())
    X    = np.zeros((len(rids), n_codes), dtype=np.float32)
    for i, rid in enumerate(rids):
        for c in rid_to_codes[rid]:
            X[i, c] += 1
        if X[i].sum() > 0:
            X[i] /= X[i].sum()
    return rids, X


def grammar_alignment(rid_to_codes, dag_file, n_codes):
    return {}   # lightweight stub — full version in process_vae_grammar_classifier.py


def downstream_auc(rids, X, dag_file):
    import re, os
    RECIPE1M = Path(os.environ.get("RECIPE1M", str(BASE.parent / "layer1.json")))
    CUISINE_PATTERNS = {
        "italian":       r"pasta|risotto|pizza|italian|parmesan|pesto",
        "mexican":       r"taco|burrito|enchilada|salsa|guacamole|mexican|chile\b",
        "asian":         r"stir.fry|chinese|wonton|fried rice|teriyaki|miso|ramen",
        "indian":        r"curry|tikka|masala|biryani|dal\b|paneer|naan|chutney",
        "mediterranean": r"greek|hummus|falafel|tzatziki|kebab|mediterranean",
        "american":      r"bbq|barbecue|mac.and.cheese|meatloaf|casserole",
    }

    calories_all, cuisines_all = {}, {}
    if RECIPE1M.exists():
        for r in json.load(open(RECIPE1M)):
            rid = r.get("id")
            if not rid: continue
            nutr = r.get("nutr_values_per100g", {})
            if nutr.get("energy"): calories_all[rid] = nutr["energy"]
            title = (r.get("title") or "").lower()
            for cui, pat in CUISINE_PATTERNS.items():
                if re.search(pat, title):
                    cuisines_all[rid] = cui; break

    rid_idx = {r: i for i, r in enumerate(rids)}
    results = {}

    # Cuisine AUC
    cui_rids = [r for r in rids if r in cuisines_all]
    if len(cui_rids) > 100:
        Xc = X[[rid_idx[r] for r in cui_rids]]
        yc = LabelEncoder().fit_transform([cuisines_all[r] for r in cui_rids])
        clf = LogisticRegression(max_iter=300, C=1.0)
        proba = cross_val_predict(clf, Xc, yc, cv=5, method="predict_proba")
        results["cuisine_auc"] = round(roc_auc_score(yc, proba,
                                       multi_class="ovr", average="macro"), 4)
        print(f"  cuisine_auc = {results['cuisine_auc']}", flush=True)

    # Calories AUC
    cal_rids = [r for r in rids if r in calories_all]
    if len(cal_rids) > 100:
        Xn = X[[rid_idx[r] for r in cal_rids]]
        yn = (np.array([calories_all[r] for r in cal_rids]) >
              np.median(list(calories_all.values()))).astype(int)
        clf = LogisticRegression(max_iter=300)
        proba = cross_val_predict(clf, Xn, yn, cv=5, method="predict_proba")[:, 1]
        results["calories_auc"] = round(roc_auc_score(yn, proba), 4)
        print(f"  calories_auc = {results['calories_auc']}", flush=True)

    return results


def make_figure(code_labels, n_active, n_codes, grammar_align, downstream, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Process VQ-VAE Eval  (K={n_codes})", fontweight="bold")

    ax = axes[0]
    dead = n_codes - n_active
    ax.bar(["Active", "Dead"], [n_active, dead],
           color=["#1e3c73", "#cccccc"])
    ax.set_title(f"Codebook usage\n{n_active}/{n_codes} active ({100*n_active/n_codes:.1f}%)")
    for bar, val in zip(ax.patches, [n_active, dead]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha="center", fontsize=13, fontweight="bold")

    ax = axes[1]
    if downstream:
        tasks = [k for k in downstream if "auc" in k]
        aucs  = [downstream[t] for t in tasks]
        bars  = ax.bar(tasks, aucs, color="#1e3c73")
        ax.axhline(0.5, color="gray", lw=1, ls="--")
        ax.set_ylim(0.45, 0.90)
        for bar, v in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                    f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
        ax.set_title("Downstream AUC (bag-of-codes LR)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(VAE_DIR / "stage1_final.pt"))
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--n_label_recipes", type=int, default=5000)
    parser.add_argument("--n_downstream_recipes", type=int, default=30000)
    args = parser.parse_args()

    save_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {args.ckpt}", flush=True)

    model, meta = load_model(args.ckpt, device)
    n_codes = meta["n_codes"]
    print(f"Model: n_codes={n_codes}  d_code={meta['d_code']}  "
          f"d_model={meta['d_model']}", flush=True)

    step_embs    = np.load(DATA_DIR / "step_embs.npy",    mmap_mode="r")
    per_ing_embs = np.load(DATA_DIR / "per_ing_embs.npy", mmap_mode="r")
    step_texts   = json.load(open(DATA_DIR / "step_texts.json"))
    step_index   = json.load(open(DATA_DIR / "step_index.json"))
    per_ing_idx  = json.load(open(DATA_DIR / "per_ing_index.json"))
    recipe_ids   = json.load(open(DATA_DIR / "recipe_ids.json"))
    print(f"  {len(recipe_ids):,} recipes", flush=True)

    # Load meta features if the model was trained with them
    step_meta = None
    if meta.get("d_meta", 0) > 0:
        meta_path = DATA_DIR / "step_meta.npy"
        step_meta = np.load(meta_path, mmap_mode="r")
        print(f"  meta features: d_meta={meta['d_meta']}  (step_meta shape={step_meta.shape})", flush=True)

    code_labels, n_active = build_codebook_labels(
        model, recipe_ids, step_embs, per_ing_embs,
        step_index, per_ing_idx, step_texts, device,
        n_recipes=args.n_label_recipes, step_meta=step_meta)
    json.dump(code_labels, open(save_dir / "codebook_labels.json", "w"), indent=2)

    rng = np.random.default_rng(42)
    if DAG_FILE.exists():
        dag_ids = set()
        with open(DAG_FILE) as f:
            for line in f:
                d = json.loads(line)
                rid = d.get("id") or d.get("recipe_id")
                if rid: dag_ids.add(rid)
        pool = [r for r in recipe_ids if r in dag_ids]
    else:
        pool = recipe_ids
    n_samp = min(args.n_downstream_recipes, len(pool))
    sample_rids = [pool[i] for i in rng.choice(len(pool), n_samp, replace=False)]
    print(f"\nEncoding {n_samp:,} recipes for downstream eval...", flush=True)

    rid_to_codes = encode_recipes(model, sample_rids, step_embs, per_ing_embs,
                                  step_index, per_ing_idx, device, step_meta=step_meta)
    rids, X = bag_of_codes(rid_to_codes, n_codes)

    grammar_align = grammar_alignment(rid_to_codes, DAG_FILE, n_codes)
    json.dump(grammar_align, open(save_dir / "grammar_alignment.json", "w"), indent=2)

    downstream = {}
    if DAG_FILE.exists():
        downstream = downstream_auc(rids, X, DAG_FILE)
        downstream["n_recipes"]     = len(rids)
        downstream["n_active_codes"] = n_active
        json.dump(downstream, open(save_dir / "downstream_auc.json", "w"), indent=2)

    out_fig = save_dir / "eval_figure.png"
    make_figure(code_labels, n_active, n_codes, grammar_align, downstream, out_fig)

    print("\n=== EVAL COMPLETE ===")
    print(f"  Active codes:  {n_active}/{n_codes}  ({100*n_active/n_codes:.1f}% utilization)")
    for k, v in downstream.items():
        if "auc" in k:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
