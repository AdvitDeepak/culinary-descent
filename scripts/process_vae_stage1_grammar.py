#!/usr/bin/env python3
"""
Option C: Grammar-supervised VQ-VAE training.

Re-runs Stage 1 Phase 2 + Phase 3 starting from the Phase 1 checkpoint,
with an additional grammar cross-entropy loss on the 35K labeled recipes.

Changes vs plain Stage 1:
  1. A tiny GrammarHead (Linear 64→16) is attached to the VQ encoder output.
  2. For each step t in the batch that belongs to a labeled recipe, we compute
     CrossEntropy( GrammarHead(code_emb[idx[t]]) , grammar_type[t] ).
  3. This loss is added with weight λ (--grammar_weight, default 0.1).
  4. Unlabeled recipes contribute only reconstruction + commitment + entropy loss,
     exactly as in the original Stage 1.

Why this works:
  The EMA codebook is updated to minimise reconstruction loss.  Adding grammar
  supervision shapes which *cluster* each step embedding maps to — semantically
  similar operations (all baking variants) are pushed to cluster together, making
  the code→grammar mapping clean.  The other 965K unlabeled recipes prevent the
  codebook from collapsing to just 15 coarse types.

Training data:
  Labeled:   35,708 recipes (constrained_dags_14b.jsonl), ~3.5% of corpus.
  Unlabeled: ~994K recipes — standard reconstruction loss only.
  Step alignment: grammar DAG step t ↔ learned step t (by position index).
                  Recipes where |T_grammar - T_learned| > 2 get no labels.

Usage:
  # Full run (Phase 2 + 3 from phase1 checkpoint, ~52 min):
  python3 scripts/process_vae_stage1_grammar.py --batch 512 --workers 8

  # Resume if interrupted:
  python3 scripts/process_vae_stage1_grammar.py --batch 512 --workers 8 --resume

  # Weaker grammar signal (safer):
  python3 scripts/process_vae_stage1_grammar.py --grammar_weight 0.05

Outputs:
  data/models/process_vae_grammar/
    stage1_grammar_phase2_best.pt
    stage1_grammar_phase3_best.pt
    stage1_grammar_final.pt          ← use this for Stage 2/3/4
    train_log.jsonl
"""

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

BASE     = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data/models/embeddings"
VAE_DIR  = BASE / "data/models/process_vae"
OUT_DIR  = BASE / "data/models/process_vae_grammar"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAG_GRAMMAR = BASE / "data/dags/constrained_dags_14b.jsonl"

sys.path.insert(0, str(BASE / "scripts"))
from process_vae_model import ProcessVAE

GRAMMAR_TYPES = ["bake","grill","saute","boil","simmer","steam",
                 "mix","whisk","blend","knead","chop","marinate",
                 "chill","season","reduce"]
TYPE2IDX = {t: i for i, t in enumerate(GRAMMAR_TYPES)}
N_GRAMMAR = len(GRAMMAR_TYPES)   # 15 (no "other" — just skip unlabeled steps)
MAX_STEPS = 24
MAX_ING   = 20


# ── Grammar head ───────────────────────────────────────────────────────────────

class GrammarHead(nn.Module):
    """Tiny classifier: VQ code embedding (d_code) → grammar type (15-way)."""
    def __init__(self, d_code=64, n_types=N_GRAMMAR):
        super().__init__()
        self.fc = nn.Linear(d_code, n_types)

    def forward(self, code_embs):
        """code_embs: (N, d_code) → logits (N, n_types)"""
        return self.fc(code_embs)


# ── Grammar label pre-computation ─────────────────────────────────────────────

def build_grammar_labels(recipe_ids, step_index, max_step_diff=2):
    """
    Returns dict: recipe_id → np.array (MAX_STEPS,) int8
      Values: 0..14 = grammar type index,  -1 = no label
    Only recipes in constrained_dags_14b.jsonl are labeled.
    """
    print("Pre-computing grammar labels from 14B DAG file...", flush=True)
    grammar_dags = {}
    with open(DAG_GRAMMAR) as f:
        for line in f:
            d = json.loads(line)
            rid = d.get("id") or d.get("recipe_id")
            grammar_dags[rid] = d

    labels_dict = {}
    n_labeled = n_skipped = 0
    for rid in recipe_ids:
        if rid not in grammar_dags:
            continue
        gsteps = grammar_dags[rid]["steps"]
        T_g = len(gsteps)
        ss, se = step_index[rid]
        T_l = min(se - ss, MAX_STEPS)
        if abs(T_g - T_l) > max_step_diff:
            n_skipped += 1
            continue
        arr = np.full(MAX_STEPS, -1, dtype=np.int8)
        T = min(T_g, T_l)
        for t in range(T):
            gtype = gsteps[t].get("canonical", "")
            if gtype in TYPE2IDX:
                arr[t] = TYPE2IDX[gtype]
        labels_dict[rid] = arr
        n_labeled += 1

    print(f"  {n_labeled:,} labeled recipes  "
          f"({n_skipped:,} skipped, step-count diff > {max_step_diff})", flush=True)
    return labels_dict


# ── Dataset ────────────────────────────────────────────────────────────────────

class GrammarRecipeDataset(Dataset):
    """
    Like RecipeStepDataset but also returns per-step grammar labels.
    grammar_labels_dict: {recipe_id → (MAX_STEPS,) int8} or None for unlabeled.
    step_meta: optional (N_total_steps, d_meta) array of temp/duration features.
    """
    def __init__(self, step_embs, per_ing_embs, step_index, per_ing_index,
                 recipe_ids, grammar_labels_dict, max_steps=MAX_STEPS, max_ing=MAX_ING,
                 step_meta=None):
        self.step_embs     = step_embs
        self.per_ing_embs  = per_ing_embs
        self.step_index    = step_index
        self.per_ing_index = per_ing_index
        self.recipe_ids    = recipe_ids
        self.grammar_labels_dict = grammar_labels_dict
        self.max_steps     = max_steps
        self.max_ing       = max_ing
        self.step_meta     = step_meta

    def __len__(self): return len(self.recipe_ids)

    def __getitem__(self, i):
        rid    = self.recipe_ids[i]
        ss, se = self.step_index[rid]
        n_steps = min(se - ss, self.max_steps)
        steps  = self.step_embs[ss:ss + n_steps]
        is_, ie = self.per_ing_index[rid]
        n_ing  = min(ie - is_, self.max_ing)
        ings   = self.per_ing_embs[is_:is_ + n_ing]
        glabels = self.grammar_labels_dict.get(rid)
        out = {
            "steps":   torch.from_numpy(steps.copy()).float(),
            "ings":    torch.from_numpy(ings.copy()).float(),
            "n_steps": n_steps,
            "n_ings":  n_ing,
            "grammar_labels": torch.from_numpy(
                glabels.copy() if glabels is not None
                else np.full(MAX_STEPS, -1, dtype=np.int8)
            ).long(),
        }
        if self.step_meta is not None:
            out["meta"] = torch.from_numpy(self.step_meta[ss:ss + n_steps].copy()).float()
        return out


def collate_fn(batch):
    B      = len(batch)
    max_T  = max(b["n_steps"] for b in batch)
    max_N  = max(b["n_ings"]  for b in batch)
    steps  = torch.zeros(B, max_T, 384)
    smask  = torch.zeros(B, max_T)
    ings   = torch.zeros(B, max_N, 384)
    imask  = torch.ones(B, max_N, dtype=torch.bool)
    glbls  = torch.full((B, MAX_STEPS), -1, dtype=torch.long)
    has_meta = "meta" in batch[0]
    d_meta = batch[0]["meta"].shape[-1] if has_meta else 0
    meta   = torch.zeros(B, max_T, d_meta) if has_meta else None
    for i, b in enumerate(batch):
        T, N = b["n_steps"], b["n_ings"]
        steps[i, :T] = b["steps"]
        smask[i, :T] = 1.0
        ings[i,  :N] = b["ings"]
        imask[i, :N] = False
        glbls[i]     = b["grammar_labels"]
        if has_meta:
            meta[i, :T] = b["meta"]
    out = {"steps": steps, "ings": ings, "step_mask": smask,
           "ing_mask": imask, "grammar_labels": glbls}
    if has_meta:
        out["meta"] = meta
    return out


# ── Training ───────────────────────────────────────────────────────────────────

def run_epoch(model, grammar_head, loader, optimizer, device,
              use_vq, grammar_weight, log_path, ep, phase):
    model.train(); grammar_head.train()
    totals = {k: 0.0 for k in ["total", "recon", "commit", "entropy", "grammar"]}
    usage_acc = []; n = 0

    for batch in loader:
        steps  = batch["steps"].to(device)
        ings   = batch["ings"].to(device)
        smask  = batch["step_mask"].to(device)
        imask  = batch["ing_mask"].to(device)
        glbls  = batch["grammar_labels"].to(device)   # (B, MAX_STEPS), -1=unlabeled
        meta   = batch["meta"].to(device) if "meta" in batch else None

        out = model(steps, ings, imask, smask, use_vq=use_vq, meta=meta)

        grammar_loss = torch.tensor(0.0, device=device)
        if use_vq and out["indices"] is not None and grammar_weight > 0:
            idx   = out["indices"]            # (B, T)  — T ≤ MAX_STEPS
            B, T  = idx.shape
            # Align grammar labels to the (B, T) code grid
            glab_bt = glbls[:, :T]           # (B, T)
            valid   = (smask[:, :T] > 0.5) & (glab_bt >= 0)  # labeled + real step
            if valid.any():
                valid_codes = idx[valid]      # (N_valid,) code indices
                valid_gtypes= glab_bt[valid]  # (N_valid,) grammar type labels
                code_embs   = model.vq.embed[valid_codes]  # (N_valid, d_code)
                logits       = grammar_head(code_embs)     # (N_valid, 15)
                grammar_loss = F.cross_entropy(logits, valid_gtypes)

        total = out["total"] + grammar_weight * grammar_loss

        optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(grammar_head.parameters()), 1.0)
        optimizer.step()

        totals["total"]   += total.item()
        totals["recon"]   += out["recon"].item()
        totals["commit"]  += out["commit"].item()
        totals["entropy"] += out["entropy"].item()
        totals["grammar"] += grammar_loss.item()
        if use_vq and out["indices"] is not None:
            usage_acc.append(model.vq.usage_fraction(out["indices"]))
        n += 1

    avgs = {k: v / n for k, v in totals.items()}
    avgs["usage"] = float(np.mean(usage_acc)) if usage_acc else 0.0
    avgs["epoch"] = ep; avgs["phase"] = phase; avgs["ts"] = time.time()
    with open(log_path, "a") as f:
        f.write(json.dumps(avgs) + "\n")
    return avgs


@torch.no_grad()
def run_val(model, grammar_head, loader, device, use_vq, grammar_weight):
    model.eval(); grammar_head.eval()
    totals = {"recon": 0.0, "total": 0.0, "grammar": 0.0}
    usage_acc = []; n = 0
    for batch in loader:
        steps = batch["steps"].to(device)
        ings  = batch["ings"].to(device)
        smask = batch["step_mask"].to(device)
        imask = batch["ing_mask"].to(device)
        glbls = batch["grammar_labels"].to(device)
        meta  = batch["meta"].to(device) if "meta" in batch else None
        out   = model(steps, ings, imask, smask, use_vq=use_vq, meta=meta)

        grammar_loss = torch.tensor(0.0, device=device)
        if use_vq and out["indices"] is not None and grammar_weight > 0:
            idx = out["indices"]; B, T = idx.shape
            glab_bt = glbls[:, :T]
            valid   = (smask[:, :T] > 0.5) & (glab_bt >= 0)
            if valid.any():
                code_embs  = model.vq.embed[idx[valid]]
                logits      = grammar_head(code_embs)
                grammar_loss = F.cross_entropy(logits, glab_bt[valid])

        totals["recon"]   += out["recon"].item()
        totals["total"]   += (out["total"] + grammar_weight * grammar_loss).item()
        totals["grammar"] += grammar_loss.item()
        if use_vq and out["indices"] is not None:
            usage_acc.append(model.vq.usage_fraction(out["indices"]))
        n += 1
    avgs = {k: v / n for k, v in totals.items()}
    avgs["usage"] = float(np.mean(usage_acc)) if usage_acc else 0.0
    return avgs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar_weight", type=float, default=0.1,
                        help="λ: weight on grammar cross-entropy loss (default 0.1)")
    parser.add_argument("--n_codes", type=int, default=None,
                        help="Override codebook size from Phase 1 checkpoint (e.g. 512)")
    parser.add_argument("--dag_file", type=str, default=None,
                        help="Path to grammar-labeled DAG JSONL (default: constrained_dags_14b.jsonl). "
                             "Use with expanded labels for stronger supervision.")
    parser.add_argument("--out_dir",  type=str, default=None,
                        help="Override output directory (default: data/models/process_vae_grammar)")
    parser.add_argument("--use_meta",  action="store_true",
                        help="Add temp_f + duration_min as encoder input features (from step_meta.npy)")
    parser.add_argument("--batch",    type=int,   default=512)
    parser.add_argument("--workers",  type=int,   default=8)
    parser.add_argument("--resume",   action="store_true",
                        help="Resume from grammar_phase2_best.pt if it exists")
    parser.add_argument("--val_frac", type=float, default=0.02)
    args = parser.parse_args()

    global DAG_GRAMMAR, OUT_DIR
    if args.dag_file:
        DAG_GRAMMAR = Path(args.dag_file)
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Grammar weight λ={args.grammar_weight}", flush=True)

    # ── Load model config from Phase 1 checkpoint ─────────────────────────────
    p1_ckpt = VAE_DIR / "stage1_phase1_best.pt"
    obj     = torch.load(p1_ckpt, map_location="cpu", weights_only=False)
    meta    = obj.get("meta", {})
    n_codes = args.n_codes if args.n_codes else meta.get("n_codes", 256)
    d_code  = meta.get("d_code",  64)
    d_model = meta.get("d_model", 256)
    n_heads = meta.get("n_heads", 4)
    print(f"Config: n_codes={n_codes}  d_code={d_code}  d_model={d_model}", flush=True)

    # ── Load embeddings ────────────────────────────────────────────────────────
    print("Loading embeddings (mmap)...", flush=True)
    step_embs    = np.load(DATA_DIR / "step_embs.npy",    mmap_mode="r")
    per_ing_embs = np.load(DATA_DIR / "per_ing_embs.npy", mmap_mode="r")
    step_index   = json.load(open(DATA_DIR / "step_index.json"))
    per_ing_idx  = json.load(open(DATA_DIR / "per_ing_index.json"))
    recipe_ids   = json.load(open(DATA_DIR / "recipe_ids.json"))
    print(f"  {len(recipe_ids):,} recipes", flush=True)

    step_meta = None
    d_meta    = 0
    if args.use_meta:
        meta_path = DATA_DIR / "step_meta.npy"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"step_meta.npy not found. Run:\n  python3 scripts/build_step_meta.py")
        step_meta = np.load(meta_path, mmap_mode="r")
        d_meta    = step_meta.shape[1]
        print(f"  step_meta loaded: shape={step_meta.shape}  d_meta={d_meta}", flush=True)

    # ── Grammar labels ─────────────────────────────────────────────────────────
    grammar_labels_dict = build_grammar_labels(recipe_ids, step_index)

    # ── Train/val split ────────────────────────────────────────────────────────
    n_total   = len(recipe_ids)
    n_val     = max(2048, int(n_total * args.val_frac))
    rng       = np.random.default_rng(42)
    perm      = rng.permutation(n_total)
    train_ids = [recipe_ids[i] for i in perm[:n_total - n_val]]
    val_ids   = [recipe_ids[i] for i in perm[n_total - n_val:]]
    print(f"  train={len(train_ids):,}  val={len(val_ids):,}", flush=True)

    # Log how many labeled recipes end up in each split
    n_tr_labeled = sum(1 for r in train_ids if r in grammar_labels_dict)
    n_val_labeled = sum(1 for r in val_ids  if r in grammar_labels_dict)
    print(f"  labeled in train={n_tr_labeled:,}  val={n_val_labeled:,}", flush=True)

    def make_ds(ids):
        return GrammarRecipeDataset(
            step_embs, per_ing_embs, step_index, per_ing_idx, ids,
            grammar_labels_dict, step_meta=step_meta)

    kw = dict(batch_size=args.batch, num_workers=args.workers, collate_fn=collate_fn,
              pin_memory=(device.type == "cuda"),
              persistent_workers=(args.workers > 0))
    train_loader = DataLoader(make_ds(train_ids), shuffle=True,  **kw)
    val_loader   = DataLoader(make_ds(val_ids),   shuffle=False, **kw)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = ProcessVAE(n_codes=n_codes, d_code=d_code,
                       d_model=d_model, n_heads=n_heads, d_meta=d_meta).to(device)
    grammar_head = GrammarHead(d_code=d_code).to(device)
    log_path = OUT_DIR / "train_log.jsonl"

    def save(name, ep, val_recon):
        torch.save({"model": model.state_dict(),
                    "grammar_head": grammar_head.state_dict(),
                    "meta": {"n_codes": n_codes, "d_code": d_code,
                             "d_model": d_model, "n_heads": n_heads,
                             "grammar_weight": args.grammar_weight,
                             "d_meta": d_meta},
                    "ep": ep, "val_recon": val_recon},
                   OUT_DIR / name)
        print(f"  saved → {OUT_DIR}/{name}", flush=True)

    # ── Phase 2: VQ training with grammar loss ─────────────────────────────────
    p2_done = (OUT_DIR / "stage1_grammar_phase2_best.pt").exists()
    if args.resume and p2_done:
        print("\nLoading phase2_best checkpoint...", flush=True)
        obj2 = torch.load(OUT_DIR / "stage1_grammar_phase2_best.pt",
                          map_location="cpu", weights_only=False)
        model.load_state_dict(obj2["model"])
        grammar_head.load_state_dict(obj2["grammar_head"])
    else:
        # Start from Phase 1 weights (skip VQ buffers — size may differ if n_codes changed)
        vq_keys = {"vq.embed", "vq.cluster_size", "vq.embed_avg"}
        filtered = {k: v for k, v in obj["model"].items() if k not in vq_keys}
        if d_meta > 0:
            # step_proj in Phase 1 is Linear(384, d_model); new model has Linear(384+d_meta, d_model).
            # Pad the extra d_meta input columns with zeros so existing knowledge is preserved.
            p1_w = filtered.pop("step_proj.weight")   # (d_model, 384)
            pad  = torch.zeros(p1_w.shape[0], d_meta)
            filtered["step_proj.weight"] = torch.cat([p1_w, pad], dim=1)  # (d_model, 386)
        model.load_state_dict(filtered, strict=False)
        log_path.unlink(missing_ok=True)

        # Warm codebook from Phase 1 activations (same as original stage1.py)
        print("\n=== Phase 2: VQ + grammar training ===", flush=True)
        print("  Warming codebook from Phase 1 activations...", flush=True)
        model.eval()
        warmup = []
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= 30: break
                steps = batch["steps"].to(device)
                ings  = batch["ings"].to(device)
                imask = batch["ing_mask"].to(device)
                meta  = batch["meta"].to(device) if "meta" in batch else None
                B, T, _ = steps.shape
                inp = torch.cat([steps, meta], dim=-1) if meta is not None and model.d_meta > 0 else steps
                x = model.step_proj(inp)
                g = model.ing_proj(ings)
                causal = ProcessVAE._causal_mask(T, x.device)
                h, _ = model.enc_self(x, x, x, attn_mask=causal)
                x = model.enc_norm1(x + h)
                h, _ = model.enc_cross(x, g, g, key_padding_mask=imask)
                x = model.enc_norm2(x + h)
                x = model.enc_norm3(x + model.enc_ff(x))
                warmup.append(model.pre_vq(x).reshape(-1, d_code).cpu())
        warmup = torch.cat(warmup)
        pidx   = torch.randperm(len(warmup))[:n_codes]
        model.vq.embed.data.copy_(warmup[pidx].to(device))
        model.vq.embed_avg.data.copy_(model.vq.embed.data)
        print(f"  Codebook warmed from {len(warmup):,} vectors", flush=True)

        opt   = torch.optim.AdamW(
            list(model.parameters()) + list(grammar_head.parameters()),
            lr=1e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=5e-6)
        best_val = float("inf")

        for ep in range(1, 21):
            t0  = time.time()
            tr  = run_epoch(model, grammar_head, train_loader, opt, device,
                            use_vq=True, grammar_weight=args.grammar_weight,
                            log_path=log_path, ep=ep, phase=2)
            val = run_val(model, grammar_head, val_loader, device,
                          use_vq=True, grammar_weight=args.grammar_weight)
            sched.step()
            print(f"  ep{ep:02d}  recon={tr['recon']:.4f}  usage={tr['usage']:.1%}  "
                  f"grammar={tr['grammar']:.4f}  val={val['recon']:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if val["recon"] < best_val:
                best_val = val["recon"]
                save("stage1_grammar_phase2_best.pt", ep, best_val)

    # ── Phase 3: fine-tune with grammar loss ───────────────────────────────────
    p2_ckpt = OUT_DIR / "stage1_grammar_phase2_best.pt"
    if p2_ckpt.exists():
        obj2 = torch.load(p2_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(obj2["model"])
        grammar_head.load_state_dict(obj2["grammar_head"])

    model.vq.commitment_cost = 0.10
    print("\n=== Phase 3: Fine-tune with grammar loss ===", flush=True)
    opt   = torch.optim.AdamW(
        list(model.parameters()) + list(grammar_head.parameters()),
        lr=3e-5, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-6)
    best_val = float("inf")

    for ep in range(1, 11):
        t0  = time.time()
        tr  = run_epoch(model, grammar_head, train_loader, opt, device,
                        use_vq=True, grammar_weight=args.grammar_weight,
                        log_path=log_path, ep=ep, phase=3)
        val = run_val(model, grammar_head, val_loader, device,
                      use_vq=True, grammar_weight=args.grammar_weight)
        sched.step()
        print(f"  ep{ep:02d}  recon={tr['recon']:.4f}  usage={tr['usage']:.1%}  "
              f"grammar={tr['grammar']:.4f}  val={val['recon']:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if val["recon"] < best_val:
            best_val = val["recon"]
            save("stage1_grammar_phase3_best.pt", ep, best_val)

    # Final checkpoint
    torch.save({"model": model.state_dict(),
                "grammar_head": grammar_head.state_dict(),
                "meta": {"n_codes": n_codes, "d_code": d_code,
                         "d_model": d_model, "n_heads": n_heads,
                         "grammar_weight": args.grammar_weight}},
               OUT_DIR / "stage1_grammar_final.pt")
    print("\nDone. Final model → data/models/process_vae_grammar/stage1_grammar_final.pt",
          flush=True)


if __name__ == "__main__":
    main()
