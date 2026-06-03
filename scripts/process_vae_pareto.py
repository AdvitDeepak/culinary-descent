#!/usr/bin/env python3
"""
Rate-distortion Pareto curve — no retraining needed.

Takes the trained 256-code model and simulates smaller vocabularies by
hierarchically merging similar codebook vectors (agglomerative clustering),
then re-evaluating reconstruction loss and downstream AUC at each k.

Produces the key paper figure:
  X: # codes (8, 15, 32, 64, 128, 185, 256)
  Y1: val reconstruction loss  (rate-distortion)
  Y2: cuisine AUC
  Y3: nutrition AUC

Usage:
  python3 scripts/process_vae_pareto.py
  python3 scripts/process_vae_pareto.py --ks 8 15 32 64 128 256
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os
BASE      = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE / "data/models/embeddings"
VAE_DIR   = BASE / "data/models/process_vae"
DAG_FILE  = BASE / "data/dags/constrained_dags_14b.jsonl"
RECIPE1M  = Path(_os.environ.get("RECIPE1M", str(BASE.parent / "layer1.json")))

import re as _re

CUISINE_PATTERNS = {
    "italian":       r"pasta|risotto|pizza|italian|parmesan|pesto|carbonara|lasagna|gnocchi|tiramisu|bolognese",
    "mexican":       r"taco|burrito|enchilada|salsa|guacamole|mexican|chile\b|quesadilla|fajita|tamale|tortilla",
    "asian":         r"stir.fry|chinese|wonton|fried rice|teriyaki|miso|ramen|soy sauce|pad thai|bok choy|hoisin",
    "indian":        r"curry|tikka|masala|biryani|dal\b|paneer|naan|chutney|tandoori|samosa|garam|turmeric",
    "mediterranean": r"greek|hummus|falafel|tzatziki|kebab|mediterranean|tahini|shawarma|pita|feta|gyro",
    "american":      r"bbq|barbecue|mac.and.cheese|meatloaf|casserole|pot pie|pulled pork|fried chicken|cornbread",
}

def load_recipe1m_labels():
    """Returns (calories_by_rid, cuisines_by_rid) from recipes.json."""
    calories, cuisines = {}, {}
    if not RECIPE1M.exists():
        return calories, cuisines
    with open(RECIPE1M) as f:
        for r in json.load(f):
            rid = r.get("id")
            if not rid: continue
            nutr = r.get("nutr_values_per100g", {})
            if nutr.get("energy"): calories[rid] = nutr["energy"]
            title = (r.get("title") or "").lower()
            for cui, pat in CUISINE_PATTERNS.items():
                if _re.search(pat, title):
                    cuisines[rid] = cui
                    break
    return calories, cuisines

sys.path.insert(0, str(BASE / "scripts"))
from process_vae_model import ProcessVAE
from process_vae_stage1 import RecipeStepDataset, collate_fn


def load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta  = ckpt["meta"]
    model = ProcessVAE(n_codes=meta["n_codes"], d_code=meta["d_code"],
                       d_model=meta["d_model"], n_heads=meta.get("n_heads", 4))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, meta


def merged_codebook(model, k):
    """
    Hierarchically merge the 256 codebook entries down to k clusters.
    Returns a mapping: original_code → merged_code (0..k-1)
    and the merged codebook centroids (k, d_code).
    """
    embed = model.vq.embed.detach().cpu().numpy()   # (256, d_code)
    Z     = linkage(embed, method="ward")
    labels = fcluster(Z, k, criterion="maxclust") - 1   # 0-indexed
    # centroids = mean of original vectors in each cluster
    centroids = np.zeros((k, embed.shape[1]), dtype=np.float32)
    for c in range(k):
        mask = labels == c
        if mask.any():
            centroids[c] = embed[mask].mean(0)
    return labels, centroids   # labels[orig_code] = merged_code


@torch.no_grad()
def eval_at_k(model, val_loader, orig_to_merged, merged_centroids, device):
    """Evaluate reconstruction loss with merged codebook."""
    centroids_t = torch.from_numpy(merged_centroids).to(device)
    total_recon = 0.0; n = 0
    for batch in val_loader:
        steps = batch["steps"].to(device)
        ings  = batch["ings"].to(device)
        smask = batch["step_mask"].to(device)
        imask = batch["ing_mask"].to(device)
        B, T, _ = steps.shape

        # encode to continuous z
        z_q_cont, idx, _ = model.encode(steps, ings, imask, use_vq=True)
        # remap indices through merge map
        idx_flat   = idx.reshape(-1).cpu().numpy()
        merged_idx = torch.tensor([orig_to_merged[i] for i in idx_flat],
                                   device=device)
        z_q_merged = centroids_t[merged_idx].reshape(B, T, -1)

        # decode with merged codes
        recon = model.decode(z_q_merged, ings, imask)

        rn = F.normalize(recon, dim=-1)
        tn = F.normalize(steps, dim=-1)
        cos = (rn * tn).sum(-1)
        recon_loss = ((1 - cos) * smask).sum() / (smask.sum() + 1e-8)
        total_recon += recon_loss.item(); n += 1
    return total_recon / n


@torch.no_grad()
def encode_all(model, val_loader, orig_to_merged, device):
    """Returns (rid_list, code_sequence_list) with merged codes."""
    all_codes = []
    for batch in val_loader:
        steps = batch["steps"].to(device)
        ings  = batch["ings"].to(device)
        imask = batch["ing_mask"].to(device)
        smask = batch["step_mask"]
        _, idx, _ = model.encode(steps, ings, imask, use_vq=True)
        B, T = idx.shape
        for b in range(B):
            codes = []
            for t in range(T):
                if smask[b, t] > 0.5:
                    codes.append(orig_to_merged[int(idx[b, t].item())])
            all_codes.append(codes)
    return all_codes


def downstream_auc_at_k(all_codes, recipe_ids, k, calories_all, cuisines_all):
    """Bag-of-codes AUC for cuisine + nutrition at vocabulary size k."""
    rid_set = set(recipe_ids)
    calories = {r: v for r, v in calories_all.items() if r in rid_set}
    cuisines = {r: v for r, v in cuisines_all.items() if r in rid_set}

    rid_idx = {rid: i for i, rid in enumerate(recipe_ids)}
    X = np.zeros((len(recipe_ids), k), dtype=np.float32)
    for i, codes in enumerate(all_codes):
        for c in codes: X[i, c] += 1
        if X[i].sum() > 0: X[i] /= X[i].sum()

    results = {}
    cui_rids = [r for r in recipe_ids if r in cuisines]
    if len(cui_rids) > 100:
        Xc = X[[rid_idx[r] for r in cui_rids]]
        yc = LabelEncoder().fit_transform([cuisines[r] for r in cui_rids])
        clf = LogisticRegression(max_iter=300, C=1.0)
        proba = cross_val_predict(clf, Xc, yc, cv=3, method="predict_proba")
        results["cuisine_auc"] = roc_auc_score(yc, proba,
                                                multi_class="ovr", average="macro")

    cal_rids = [r for r in recipe_ids if r in calories]
    if len(cal_rids) > 100:
        Xn = X[[rid_idx[r] for r in cal_rids]]
        yn = (np.array([calories[r] for r in cal_rids]) >
              np.median(list(calories.values()))).astype(int)
        clf = LogisticRegression(max_iter=300)
        proba = cross_val_predict(clf, Xn, yn, cv=3, method="predict_proba")[:,1]
        results["calories_auc"] = roc_auc_score(yn, proba)

    return results


def make_pareto_figure(results, out_path):
    ks       = [r["k"] for r in results]
    recons   = [r["recon_loss"] for r in results]
    cuisines = [r.get("cuisine_auc", None) for r in results]
    calories = [r.get("calories_auc", None) for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Rate-Distortion Pareto Curve — Process VAE", fontsize=13,
                 fontweight="bold")

    # Left: recon loss vs k
    ax = axes[0]
    ax.plot(ks, recons, "o-", color="#C44E52", lw=2, ms=7, label="recon loss")
    ax.set_xscale("log"); ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], rotation=45)
    ax.set_xlabel("# codes (vocabulary size)")
    ax.set_ylabel("Reconstruction loss (cosine)")
    ax.set_title("Rate-Distortion\n(lower = better reconstruction)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # Right: downstream AUC vs k
    ax = axes[1]
    if any(c is not None for c in cuisines):
        ax.plot(ks, cuisines, "s-", color="#4C72B0", lw=2, ms=7,
                label="cuisine AUC")
    if any(c is not None for c in calories):
        ax.plot(ks, calories, "^-", color="#55A868", lw=2, ms=7,
                label="calories AUC")
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", label="random")
    ax.set_xscale("log"); ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], rotation=45)
    ax.set_xlabel("# codes (vocabulary size)")
    ax.set_ylabel("AUC")
    ax.set_title("Downstream task AUC vs vocabulary size\n(where does it peak?)")
    ax.set_ylim(0.45, 1.0); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(VAE_DIR / "stage1_final.pt"))
    parser.add_argument("--ks", nargs="+", type=int,
                        default=[8, 15, 32, 64, 128, 185, 256])
    parser.add_argument("--n_val", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  ks={args.ks}", flush=True)

    model, meta = load_model(args.ckpt, device)
    n_codes_full = meta["n_codes"]

    # Load data
    step_embs    = np.load(DATA_DIR / "step_embs.npy",    mmap_mode="r")
    per_ing_embs = np.load(DATA_DIR / "per_ing_embs.npy", mmap_mode="r")
    step_index   = json.load(open(DATA_DIR / "step_index.json"))
    per_ing_idx  = json.load(open(DATA_DIR / "per_ing_index.json"))
    recipe_ids   = json.load(open(DATA_DIR / "recipe_ids.json"))

    # Load labels once; sample val_rids from recipes that have labels
    print("Loading Recipe1M labels...", flush=True)
    calories_all, cuisines_all = load_recipe1m_labels()
    labeled_rids = [r for r in recipe_ids if r in calories_all]
    print(f"  {len(labeled_rids):,} recipes with nutrition labels, "
          f"{sum(1 for r in labeled_rids if r in cuisines_all):,} with cuisine", flush=True)

    rng      = np.random.default_rng(99)
    pool     = labeled_rids if labeled_rids else recipe_ids
    val_rids = [pool[i] for i in
                rng.choice(len(pool), min(args.n_val, len(pool)), replace=False)]
    val_ds   = RecipeStepDataset(step_embs, per_ing_embs,
                                  step_index, per_ing_idx, val_rids)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_fn,
                            pin_memory=(device.type == "cuda"))

    results = []
    for k in sorted(args.ks):
        print(f"\n── k={k} ──────────────────────────────────", flush=True)
        if k >= n_codes_full:
            # Use original codebook as-is
            orig_to_merged = list(range(n_codes_full))
            centroids = model.vq.embed.detach().cpu().numpy()
        else:
            orig_to_merged, centroids = merged_codebook(model, k)

        recon = eval_at_k(model, val_loader, orig_to_merged, centroids, device)
        print(f"  recon_loss={recon:.4f}", flush=True)

        row = {"k": k, "recon_loss": round(recon, 5)}

        if calories_all:
            all_codes = encode_all(model, val_loader, orig_to_merged, device)
            aucs = downstream_auc_at_k(all_codes, val_rids, k, calories_all, cuisines_all)
            row.update({k2: round(v, 4) for k2, v in aucs.items()})
            for k2, v in aucs.items():
                print(f"  {k2}={v:.4f}", flush=True)

        results.append(row)

    json.dump(results, open(VAE_DIR / "pareto_curve.json", "w"), indent=2)

    out_fig = BASE / "data/figures/process_vae_pareto.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    make_pareto_figure(results, out_fig)

    print("\n=== PARETO COMPLETE ===")
    print(f"  Results → {VAE_DIR}/pareto_curve.json")


if __name__ == "__main__":
    main()
