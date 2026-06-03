#!/usr/bin/env python3
"""
Stage 1: Train Per-step Contextual VQ-VAE on 1M recipes.

Uses pre-cached MiniLM step + ingredient embeddings from data/models/embeddings/.
Learns a discrete codebook = ideal cooking operation vocabulary.

Training phases:
Phase 1 (10 ep, lr=3e-4): continuous bottleneck — encoder+decoder co-adapt
  without VQ pressure. Gives the model time to learn useful structure first.

Phase 2 (20 ep, lr=1e-4): VQ enabled. Codebook warmed from phase-1 activations
  so it starts with reasonable cluster centres. EMA updates stabilise training.

Phase 3 (10 ep, lr=3e-5): fine-tune at low LR with stronger dropout (0.25)
  to compress codebook usage and sharpen code boundaries.

Outputs:
  data/models/process_vae/
    stage1_phase{1,2,3}_best.pt   — encoder + decoder weights
    stage1_final.pt               — final model (use this for Stage 2)
    train_log.jsonl               — per-epoch metrics
    codebook_labels.json          — top-5 step texts per code (for inspection)

Usage:
  # Default: use existing cached embeddings (fast start)
  python3 scripts/process_vae_stage1.py

  # With atomic step splitting (re-embeds compound steps; recommended but ~30 min extra)
  python3 scripts/process_vae_stage1.py --atomic

  # Resume from phase 2 (if phase 1 already done)
  python3 scripts/process_vae_stage1.py --resume

  # Skip to phase 3 (if phase 2 already done)
  python3 scripts/process_vae_stage1.py --phase 3
"""

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE     = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data/models/embeddings"   # pre-cached 1M embeddings
OUT_DIR  = BASE / "data/models/process_vae"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE / "scripts"))
from process_vae_model import ProcessVAE, split_atomic

SNAP_EVERY = 5    # epochs between codebook label snapshots

# ── Dataset ────────────────────────────────────────────────────────────────────

class RecipeStepDataset(Dataset):
    def __init__(self, step_embs, per_ing_embs, step_index, per_ing_index,
                 recipe_ids, max_steps=24, max_ing=20, step_meta=None):
        self.step_embs     = step_embs
        self.per_ing_embs  = per_ing_embs
        self.step_index    = step_index
        self.per_ing_index = per_ing_index
        self.recipe_ids    = recipe_ids
        self.max_steps     = max_steps
        self.max_ing       = max_ing
        self.step_meta     = step_meta  # optional (N_steps, d_meta) array

    def __len__(self): return len(self.recipe_ids)

    def __getitem__(self, i):
        rid    = self.recipe_ids[i]
        ss, se = self.step_index[rid]
        steps  = self.step_embs[ss:se]
        if len(steps) > self.max_steps:
            steps = steps[:self.max_steps]

        is_, ie = self.per_ing_index[rid]
        n_ing   = min(ie - is_, self.max_ing)
        ings    = self.per_ing_embs[is_:is_ + n_ing]

        out = {
            "steps":   torch.from_numpy(steps.copy()).float(),
            "ings":    torch.from_numpy(ings.copy()).float(),
            "n_steps": len(steps),
            "n_ings":  n_ing,
        }
        if self.step_meta is not None:
            meta = self.step_meta[ss : ss + len(steps)]
            out["meta"] = torch.from_numpy(meta.copy()).float()
        return out


def collate_fn(batch):
    B       = len(batch)
    max_T   = max(b["n_steps"] for b in batch)
    max_N   = max(b["n_ings"]  for b in batch)
    d       = 384
    steps   = torch.zeros(B, max_T, d)
    smask   = torch.zeros(B, max_T)
    ings    = torch.zeros(B, max_N, d)
    imask   = torch.ones(B, max_N, dtype=torch.bool)   # True = padding
    has_meta = "meta" in batch[0]
    d_meta  = batch[0]["meta"].shape[-1] if has_meta else 0
    meta    = torch.zeros(B, max_T, d_meta) if has_meta else None
    for i, b in enumerate(batch):
        T, N = b["n_steps"], b["n_ings"]
        steps[i, :T]  = b["steps"]
        smask[i, :T]  = 1.0
        ings[i, :N]   = b["ings"]
        imask[i, :N]  = False
        if has_meta:
            meta[i, :T] = b["meta"]
    out = {"steps": steps, "ings": ings, "step_mask": smask, "ing_mask": imask}
    if has_meta:
        out["meta"] = meta
    return out


# ── Atomic re-embedding (optional) ────────────────────────────────────────────

def rebuild_atomic_embeddings(step_texts_all, step_index, recipe_ids, out_dir):
    """
    Re-split compound steps and re-embed with MiniLM.
    Saves step_embs_atomic.npy + step_index_atomic.json next to existing cache.
    ~25-35 min on GPU for 1M recipes.
    """
    from sentence_transformers import SentenceTransformer
    print("Loading MiniLM for atomic re-embedding...", flush=True)
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                   device="cuda")
    encoder.eval()

    atomic_texts = []
    new_index    = {}
    print("Splitting compound steps...", flush=True)
    for rid in recipe_ids:
        ss, se = step_index[rid]
        start  = len(atomic_texts)
        for raw in step_texts_all[ss:se]:
            for sub in split_atomic(raw):
                atomic_texts.append(sub)
        new_index[rid] = (start, len(atomic_texts))

    print(f"  {len(atomic_texts):,} atomic steps from {len(recipe_ids):,} recipes "
          f"(was {sum(se-ss for ss,se in step_index.values() if True):,})", flush=True)

    print("Embedding with MiniLM (batch 4096)...", flush=True)
    embs = encoder.encode(atomic_texts, batch_size=4096, show_progress_bar=True,
                          convert_to_numpy=True, normalize_embeddings=False)
    embs = embs.astype(np.float16)

    out_emb = out_dir / "step_embs_atomic.npy"
    out_idx = out_dir / "step_index_atomic.json"
    out_txt = out_dir / "step_texts_atomic.json"
    np.save(out_emb, embs)
    json.dump(new_index, open(out_idx, "w"))
    json.dump(atomic_texts, open(out_txt, "w"))
    print(f"  Saved → {out_emb}  ({embs.nbytes/1e9:.1f} GB)", flush=True)
    return embs, new_index, atomic_texts


# ── Helpers ────────────────────────────────────────────────────────────────────

def log_step(path, entry):
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def save_ckpt(model, path, meta):
    torch.save({"model": model.state_dict(), "meta": meta}, path)
    print(f"  saved → {path}", flush=True)


def load_ckpt(model, path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(obj["model"], strict=False)
    return obj.get("meta", {})


@torch.no_grad()
def snapshot_codebook(model, val_loader, step_texts, step_index, recipe_ids,
                      val_ids, out_path, device, n_per_code=5):
    """Collect top-N step texts per code for human inspection."""
    model.eval()
    code_texts = {c: [] for c in range(model.vq.n_codes)}
    for batch in val_loader:
        steps = batch["steps"].to(device)
        ings  = batch["ings"].to(device)
        imask = batch["ing_mask"].to(device)
        smask = batch["step_mask"]
        z_q, idx, _ = model.encode(steps, ings, imask, use_vq=True)
        if idx is None: break
        B, T = idx.shape
        for b in range(B):
            for t in range(T):
                if smask[b, t] < 0.5: continue
                c = int(idx[b, t].item())
                if len(code_texts[c]) < n_per_code:
                    code_texts[c].append(f"[recipe {b} step {t}]")
        if all(len(v) >= n_per_code for v in code_texts.values()):
            break
    usage = model.vq.usage_fraction(idx)
    out = {"usage_fraction": usage, "n_codes": model.vq.n_codes,
           "codes": code_texts}
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"  codebook snapshot → {out_path}  (usage={usage:.1%})", flush=True)


# ── Training loops ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, use_vq, log_path, ep, phase):
    model.train()
    totals = {k: 0.0 for k in ["total", "recon", "commit", "entropy"]}
    usage_acc = []; n = 0
    for batch in loader:
        steps = batch["steps"].to(device)
        ings  = batch["ings"].to(device)
        smask = batch["step_mask"].to(device)
        imask = batch["ing_mask"].to(device)
        out   = model(steps, ings, imask, smask, use_vq=use_vq)
        optimizer.zero_grad()
        out["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for k in totals:
            totals[k] += out[k].item()
        if use_vq and out["indices"] is not None:
            usage_acc.append(model.vq.usage_fraction(out["indices"]))
        n += 1
    avgs = {k: v / n for k, v in totals.items()}
    avgs["usage"] = float(np.mean(usage_acc)) if usage_acc else 0.0
    avgs["epoch"] = ep; avgs["phase"] = phase; avgs["ts"] = time.time()
    log_step(log_path, avgs)
    return avgs


@torch.no_grad()
def run_val(model, loader, device, use_vq):
    model.eval()
    totals = {"recon": 0.0, "total": 0.0}
    usage_acc = []; n = 0
    for batch in loader:
        steps = batch["steps"].to(device)
        ings  = batch["ings"].to(device)
        smask = batch["step_mask"].to(device)
        imask = batch["ing_mask"].to(device)
        out   = model(steps, ings, imask, smask, use_vq=use_vq)
        totals["recon"] += out["recon"].item()
        totals["total"] += out["total"].item()
        if use_vq and out["indices"] is not None:
            usage_acc.append(model.vq.usage_fraction(out["indices"]))
        n += 1
    avgs = {k: v / n for k, v in totals.items()}
    avgs["usage"] = float(np.mean(usage_acc)) if usage_acc else 0.0
    return avgs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",    type=int,   default=0,
                        help="0=all, 1=warmup only, 2=VQ only, 3=finetune only")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip phase 1 if phase1_best.pt exists")
    parser.add_argument("--atomic",   action="store_true",
                        help="Re-split compound steps and re-embed with MiniLM")
    parser.add_argument("--batch",    type=int,   default=512)
    parser.add_argument("--n_codes",  type=int,   default=256)
    parser.add_argument("--d_code",   type=int,   default=64)
    parser.add_argument("--d_model",  type=int,   default=256)
    parser.add_argument("--n_heads",  type=int,   default=4)
    parser.add_argument("--workers",  type=int,   default=8)
    parser.add_argument("--val_frac", type=float, default=0.02)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Config: n_codes={args.n_codes}  d_code={args.d_code}  "
          f"d_model={args.d_model}  batch={args.batch}", flush=True)

    # ── Load embeddings ────────────────────────────────────────────────────────
    print("Loading embeddings (mmap)...", flush=True)
    with open(DATA_DIR / "step_index.json")    as f: step_index    = json.load(f)
    with open(DATA_DIR / "per_ing_index.json") as f: per_ing_index = json.load(f)
    with open(DATA_DIR / "recipe_ids.json")    as f: recipe_ids    = json.load(f)

    if args.atomic:
        atomic_emb_path = DATA_DIR / "step_embs_atomic.npy"
        atomic_idx_path = DATA_DIR / "step_index_atomic.json"
        if atomic_emb_path.exists() and atomic_idx_path.exists():
            print("  Loading existing atomic embeddings...", flush=True)
            step_embs = np.load(atomic_emb_path, mmap_mode="r")
            with open(atomic_idx_path) as f: step_index = json.load(f)
        else:
            step_texts = json.load(open(DATA_DIR / "step_texts.json"))
            step_embs, step_index, _ = rebuild_atomic_embeddings(
                step_texts, step_index, recipe_ids, DATA_DIR)
    else:
        step_embs = np.load(DATA_DIR / "step_embs.npy", mmap_mode="r")

    per_ing_embs = np.load(DATA_DIR / "per_ing_embs.npy", mmap_mode="r")
    print(f"  step_embs:    {step_embs.shape}  "
          f"({step_embs.nbytes/1e9:.1f} GB)", flush=True)
    print(f"  per_ing_embs: {per_ing_embs.shape}  "
          f"({per_ing_embs.nbytes/1e9:.1f} GB)", flush=True)
    print(f"  {len(recipe_ids):,} recipes", flush=True)

    # ── Split train/val ────────────────────────────────────────────────────────
    n_total   = len(recipe_ids)
    n_val     = max(2048, int(n_total * args.val_frac))
    rng       = np.random.default_rng(42)
    perm      = rng.permutation(n_total)
    train_ids = [recipe_ids[i] for i in perm[:n_total - n_val]]
    val_ids   = [recipe_ids[i] for i in perm[n_total - n_val:]]
    print(f"  train={len(train_ids):,}  val={len(val_ids):,}", flush=True)

    def make_ds(ids):
        return RecipeStepDataset(step_embs, per_ing_embs,
                                 step_index, per_ing_index, ids)

    train_loader = DataLoader(make_ds(train_ids), batch_size=args.batch,
                              shuffle=True,  num_workers=args.workers,
                              collate_fn=collate_fn,
                              pin_memory=(device.type == "cuda"),
                              persistent_workers=(args.workers > 0))
    val_loader   = DataLoader(make_ds(val_ids),   batch_size=args.batch,
                              shuffle=False, num_workers=args.workers,
                              collate_fn=collate_fn,
                              pin_memory=(device.type == "cuda"),
                              persistent_workers=(args.workers > 0))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ProcessVAE(n_codes=args.n_codes, d_code=args.d_code,
                       d_model=args.d_model, n_heads=args.n_heads).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}", flush=True)

    ckpt_dir = OUT_DIR
    log_path = ckpt_dir / "train_log.jsonl"
    meta_base = {"n_codes": args.n_codes, "d_code": args.d_code,
                 "d_model": args.d_model, "n_heads": args.n_heads}

    def _save(name, ep, val_recon):
        save_ckpt(model, ckpt_dir / name, {**meta_base, "ep": ep,
                                            "val_recon": val_recon})

    # ── Phase 1: Continuous warmup ─────────────────────────────────────────────
    if args.phase in (0, 1):
        p1_ckpt = ckpt_dir / "stage1_phase1_best.pt"
        print("\n=== Phase 1: Continuous warmup (no VQ) ===", flush=True)
        if args.resume and p1_ckpt.exists():
            print("  Resuming: loading phase1_best.pt", flush=True)
            load_ckpt(model, p1_ckpt)
        else:
            log_path.unlink(missing_ok=True)
            opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-5)
            best_val = float("inf")
            for ep in range(1, 11):
                t0  = time.time()
                tr  = run_epoch(model, train_loader, opt, device, False,
                                log_path, ep, 1)
                val = run_val(model, val_loader, device, False)
                sched.step()
                print(f"  ep{ep:02d}  recon={tr['recon']:.4f}  "
                      f"val={val['recon']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
                if val["recon"] < best_val:
                    best_val = val["recon"]
                    _save("stage1_phase1_best.pt", ep, best_val)

    # ── Phase 2: VQ training ───────────────────────────────────────────────────
    if args.phase in (0, 2):
        p1_ckpt = ckpt_dir / "stage1_phase1_best.pt"
        if p1_ckpt.exists():
            load_ckpt(model, p1_ckpt)

        print("\n=== Phase 2: VQ training ===", flush=True)
        print("  Warming codebook from phase-1 activations...", flush=True)
        model.eval()
        warmup = []
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= 30: break
                steps = batch["steps"].to(device)
                ings  = batch["ings"].to(device)
                imask = batch["ing_mask"].to(device)
                # collect pre-VQ projections as warm-start cluster centres
                B, T, _ = steps.shape
                x = model.step_proj(steps)
                g = model.ing_proj(ings)
                causal = ProcessVAE._causal_mask(T, x.device)
                h, _ = model.enc_self(x, x, x, attn_mask=causal)
                x = model.enc_norm1(x + h)
                h, _ = model.enc_cross(x, g, g, key_padding_mask=imask)
                x = model.enc_norm2(x + h)
                x = model.enc_norm3(x + model.enc_ff(x))
                z = model.pre_vq(x)
                warmup.append(z.reshape(-1, args.d_code).cpu())
        warmup = torch.cat(warmup)
        pidx   = torch.randperm(len(warmup))[:args.n_codes]
        model.vq.embed.data.copy_(warmup[pidx].to(device))
        model.vq.embed_avg.data.copy_(model.vq.embed.data)
        print(f"  Codebook warmed from {len(warmup):,} vectors", flush=True)

        opt   = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=5e-6)
        best_val = float("inf")
        for ep in range(1, 21):
            t0  = time.time()
            tr  = run_epoch(model, train_loader, opt, device, True,
                            log_path, ep, 2)
            val = run_val(model, val_loader, device, True)
            sched.step()
            print(f"  ep{ep:02d}  recon={tr['recon']:.4f}  "
                  f"commit={tr['commit']:.4f}  usage={tr['usage']:.1%}  "
                  f"val={val['recon']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            if val["recon"] < best_val:
                best_val = val["recon"]
                _save("stage1_phase2_best.pt", ep, best_val)
            if ep % SNAP_EVERY == 0:
                try:
                    snapshot_codebook(model, val_loader,
                                      None, step_index, recipe_ids, val_ids,
                                      ckpt_dir / f"codebook_phase2_ep{ep:02d}.json",
                                      device)
                except Exception as e:
                    print(f"  [snapshot failed: {e}]", flush=True)

    # ── Phase 3: Fine-tune ─────────────────────────────────────────────────────
    if args.phase in (0, 3):
        p2_ckpt = ckpt_dir / "stage1_phase2_best.pt"
        if p2_ckpt.exists():
            load_ckpt(model, p2_ckpt)
        model.vq.commitment_cost = 0.10   # relax commitment — codes already stable

        print("\n=== Phase 3: Fine-tune ===", flush=True)
        opt   = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-6)
        best_val = float("inf")
        for ep in range(1, 11):
            t0  = time.time()
            tr  = run_epoch(model, train_loader, opt, device, True,
                            log_path, ep, 3)
            val = run_val(model, val_loader, device, True)
            sched.step()
            print(f"  ep{ep:02d}  recon={tr['recon']:.4f}  "
                  f"usage={tr['usage']:.1%}  val={val['recon']:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if val["recon"] < best_val:
                best_val = val["recon"]
                _save("stage1_phase3_best.pt", ep, best_val)

        save_ckpt(model, ckpt_dir / "stage1_final.pt", meta_base)
        print("\nStage 1 complete. Final model → data/models/process_vae/stage1_final.pt",
              flush=True)

        # Final codebook snapshot for analysis
        try:
            snapshot_codebook(model, val_loader,
                              None, step_index, recipe_ids, val_ids,
                              ckpt_dir / "codebook_final.json", device,
                              n_per_code=10)
        except Exception as e:
            print(f"  [final snapshot failed: {e}]", flush=True)


if __name__ == "__main__":
    main()
