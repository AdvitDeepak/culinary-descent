#!/usr/bin/env python3
"""
Option A: Grammar-type classifier on VQ codes.

Trains a small MLP that maps a VQ code embedding (64-dim) to one of 16 grammar types
(15 grammar types + "other") using the 35K overlap recipes as weak supervision.

Training data construction:
  - Load learned DAGs (VQ codes per step) and grammar DAGs (type per step)
  - Align by step position index (works well for 68.6% of recipes within ±1 step)
  - Collect (code_embedding, grammar_type) pairs
  - ~170K training pairs total

At inference: given a VQ code, predict the most likely grammar type.
This fixes the "wrong/nonsensical label" failure mode (42% of judge losses).

Usage:
  # Train:
  python3 scripts/process_vae_grammar_classifier.py

  # Evaluate alignment on val set:
  python3 scripts/process_vae_grammar_classifier.py --eval

  # Test prediction for a specific code:
  python3 scripts/process_vae_grammar_classifier.py --predict 240
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

BASE       = Path(__file__).resolve().parent.parent
VAE_DIR    = BASE / "data/models/process_vae"
DAG_LEARNED  = BASE / "data/dags/learned_dags.jsonl"
DAG_GRAMMAR  = BASE / "data/dags/constrained_dags_14b.jsonl"

sys.path.insert(0, str(BASE / "scripts"))
from process_vae_model import ProcessVAE

GRAMMAR_TYPES = ["bake","grill","saute","boil","simmer","steam",
                 "mix","whisk","blend","knead","chop","marinate",
                 "chill","season","reduce"]
TYPE2IDX = {t: i for i, t in enumerate(GRAMMAR_TYPES)}
TYPE2IDX["other"] = len(GRAMMAR_TYPES)   # index 15
N_TYPES = len(GRAMMAR_TYPES) + 1         # 16


# ── Model ──────────────────────────────────────────────────────────────────────

class GrammarClassifier(nn.Module):
    """
    Maps VQ code embedding (d_code-dim) → grammar type (16-way).
    Tiny MLP — trains in seconds on CPU.
    """
    def __init__(self, d_code=64, n_types=N_TYPES, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_code, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, n_types),
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, code_embs):
        """code_embs: (N, d_code) → predicted type indices (N,)"""
        with torch.no_grad():
            return self.forward(code_embs).argmax(1)

    def predict_label(self, code_idx, embed):
        """Single code index → grammar type string."""
        emb = embed[code_idx].unsqueeze(0)
        pred = self.predict(emb).item()
        labels = GRAMMAR_TYPES + ["other"]
        return labels[pred]


# ── Training data ──────────────────────────────────────────────────────────────

def build_training_data(vq_embed, max_step_diff=2):
    """
    Align grammar DAG steps to learned DAG steps by position index.
    Returns X (N, d_code) code embeddings and y (N,) grammar type indices.
    """
    print("Loading DAGs...", flush=True)

    learned = {}
    with open(DAG_LEARNED) as f:
        for line in f:
            d = json.loads(line)
            learned[d["id"]] = d

    grammar = {}
    with open(DAG_GRAMMAR) as f:
        for line in f:
            d = json.loads(line)
            rid = d.get("id") or d.get("recipe_id")
            if rid in learned:
                grammar[rid] = d

    print(f"  {len(grammar):,} overlap recipes", flush=True)

    X_list, y_list = [], []
    n_aligned = n_skipped_diff = n_skipped_type = 0

    for rid, gdag in grammar.items():
        ldag = learned[rid]
        g_steps = gdag["steps"]
        l_nodes  = ldag["nodes"]
        T_g, T_l = len(g_steps), len(l_nodes)

        # Skip if step counts diverge too much
        if abs(T_g - T_l) > max_step_diff:
            n_skipped_diff += 1
            continue

        T = min(T_g, T_l)
        for t in range(T):
            gtype = g_steps[t].get("canonical", "")
            code  = l_nodes[t]["code"]
            label = TYPE2IDX.get(gtype, TYPE2IDX["other"])
            X_list.append(vq_embed[code].numpy())
            y_list.append(label)
            n_aligned += 1

    print(f"  {n_aligned:,} (code, grammar_type) pairs collected", flush=True)
    print(f"  {n_skipped_diff:,} recipes skipped (step count diff > {max_step_diff})", flush=True)

    X = torch.from_numpy(np.array(X_list, dtype=np.float32))
    y = torch.tensor(y_list, dtype=torch.long)
    return X, y


# ── Training ───────────────────────────────────────────────────────────────────

def train(args):
    # Load VQ codebook
    print("Loading Stage 1 checkpoint...", flush=True)
    ckpt  = torch.load(VAE_DIR / "stage1_final.pt", map_location="cpu", weights_only=False)
    meta  = ckpt["meta"]
    model = ProcessVAE(n_codes=meta["n_codes"], d_code=meta["d_code"],
                       d_model=meta["d_model"], n_heads=meta.get("n_heads", 4))
    model.load_state_dict(ckpt["model"])
    vq_embed = model.vq.embed.detach().cpu()   # (n_codes, d_code)
    d_code   = meta["d_code"]
    print(f"  Codebook: {meta['n_codes']} codes × {d_code}-dim", flush=True)

    X, y = build_training_data(vq_embed, max_step_diff=args.max_step_diff)

    # Train/val split
    rng   = np.random.default_rng(42)
    perm  = rng.permutation(len(X))
    n_val = max(500, len(X) // 10)
    val_idx = perm[:n_val].tolist()
    tr_idx  = perm[n_val:].tolist()

    tr_ds  = TensorDataset(X[tr_idx], y[tr_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx])
    tr_loader  = DataLoader(tr_ds,  batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    clf = GrammarClassifier(d_code=d_code)
    opt = torch.optim.AdamW(clf.parameters(), lr=3e-4, weight_decay=1e-3)

    print(f"\nTraining on {len(tr_idx):,} pairs, val on {len(val_idx):,}...", flush=True)
    best_val_acc = 0.0

    for ep in range(1, args.epochs + 1):
        clf.train()
        tot_loss = 0.0; n = 0
        for xb, yb in tr_loader:
            logits = clf(xb)
            loss   = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item(); n += 1

        clf.eval()
        correct = total = 0
        per_type = {t: [0, 0] for t in GRAMMAR_TYPES + ["other"]}
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = clf(xb).argmax(1)
                correct += (preds == yb).sum().item()
                total   += len(yb)
                for pred, gt in zip(preds.tolist(), yb.tolist()):
                    labels = GRAMMAR_TYPES + ["other"]
                    per_type[labels[gt]][1] += 1
                    if pred == gt:
                        per_type[labels[gt]][0] += 1

        val_acc = correct / total
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            print(f"  ep{ep:03d}  loss={tot_loss/n:.4f}  val_acc={val_acc:.3f}", flush=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"clf": clf.state_dict(), "d_code": d_code,
                        "types": GRAMMAR_TYPES}, VAE_DIR / "grammar_classifier.pt")

    print(f"\nBest val accuracy: {best_val_acc:.3f}", flush=True)

    # Per-type breakdown
    print("\nPer-type accuracy (val set):")
    labels = GRAMMAR_TYPES + ["other"]
    for t in sorted(per_type, key=lambda t: -per_type[t][1]):
        c, tot = per_type[t]
        if tot > 0:
            print(f"  {t:12s}  {c:4d}/{tot:4d}  ({c/tot:.0%})")

    print(f"\nSaved → {VAE_DIR}/grammar_classifier.pt", flush=True)
    return clf


# ── Inference helpers (imported by eval script) ────────────────────────────────

def load_classifier(device="cpu"):
    """Load trained classifier. Returns (clf, vq_embed, predict_fn)."""
    path = VAE_DIR / "grammar_classifier.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path} — run: python3 scripts/process_vae_grammar_classifier.py")
    ckpt    = torch.load(path, map_location=device, weights_only=False)
    clf     = GrammarClassifier(d_code=ckpt["d_code"])
    clf.load_state_dict(ckpt["clf"])
    clf.eval()

    # Also load VQ embed for lookup
    vae_ckpt = torch.load(VAE_DIR / "stage1_final.pt", map_location=device, weights_only=False)
    vae_meta = vae_ckpt["meta"]
    vae_model = ProcessVAE(n_codes=vae_meta["n_codes"], d_code=vae_meta["d_code"],
                           d_model=vae_meta["d_model"], n_heads=vae_meta.get("n_heads", 4))
    vae_model.load_state_dict(vae_ckpt["model"])
    vq_embed = vae_model.vq.embed.detach()  # (n_codes, d_code)

    labels = GRAMMAR_TYPES + ["other"]

    def predict_type(code_idx: int, confidence_threshold: float = 0.6) -> str | None:
        """
        Returns predicted grammar type if softmax confidence >= threshold, else None.
        Threshold prevents the 102/256 codes that weakly predict 'mix' from
        flooding the output with uninformative labels.
        """
        emb    = vq_embed[code_idx].unsqueeze(0)
        logits = clf(emb)
        probs  = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(1)
        if conf.item() < confidence_threshold:
            return None
        return labels[pred.item()]

    def predict_batch(code_indices, confidence_threshold: float = 0.6) -> list:
        """Returns list of (label | None) per code index."""
        embs   = vq_embed[torch.tensor(code_indices)]
        logits = clf(embs)
        probs  = torch.softmax(logits, dim=-1)
        confs, preds = probs.max(1)
        return [labels[p] if c >= confidence_threshold else None
                for p, c in zip(preds.tolist(), confs.tolist())]

    return clf, vq_embed, predict_type, predict_batch


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--max_step_diff", type=int,   default=2,
                        help="Skip recipes where |T_learned - T_grammar| > this")
    parser.add_argument("--eval",          action="store_true",
                        help="Just print per-type accuracy of saved classifier")
    parser.add_argument("--predict",       type=int,   default=None,
                        help="Print predicted grammar type for a specific code index")
    args = parser.parse_args()

    if args.predict is not None:
        _, _, predict_type, _ = load_classifier()
        print(f"Code {args.predict} → {predict_type(args.predict)}")
        return

    if args.eval:
        clf, vq_embed, predict_type, _ = load_classifier()
        print("Classifier predictions for all 256 codes:")
        for c in range(256):
            pred = predict_type(c)
            print(f"  c{c:03d} → {pred}")
        return

    train(args)


if __name__ == "__main__":
    main()
