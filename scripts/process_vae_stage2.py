#!/usr/bin/env python3
"""
Stage 2: Qwen Decoder — discrete DAG → recipe text.

Freezes the Stage 1 VQ encoder. Trains:
  - code_embedding table  (n_codes × qwen_dim)
  - Qwen2.5-3B-Instruct LoRA (r=16)

Forward pass (cached-code path, default):
  1. Load pre-cached code IDs from tokenized/code_ids.npy
  2. Embed each code as a learnable vector (n_codes × 2048)
  3. Prepend T code-prefix tokens to Qwen's input
  4. Qwen generates full recipe text (title + ingredients + instructions)
  5. LM loss on recipe tokens only (prompt masked to -100)

Usage:
  # Tokenise first (one-time, ~30s):
  python3 scripts/process_vae_stage2.py --mode tokenize

  # Pre-cache code IDs through Stage 1 (one-time, ~5 min):
  python3 scripts/process_vae_stage2.py --mode cache_codes

  # Train (uses cached codes — Stage 1 not on GPU during training):
  python3 scripts/process_vae_stage2.py --mode train --batch 16 --epochs 5

  # Generate from a DAG:
  python3 scripts/process_vae_stage2.py --mode generate --recipe_id <id>
"""

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

BASE       = Path(__file__).resolve().parent.parent
LAYER1     = Path(os.environ.get("RECIPE1M", str(BASE.parent / "layer1.json")))
DATA_DIR   = BASE / "data/models/embeddings"
STAGE1_DIR = BASE / "data/models/process_vae"
OUT_DIR    = BASE / "data/models/process_vae_stage2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TOK_DIR    = OUT_DIR / "tokenized"

# Overridden by --stage1_ckpt / --s2_dir at parse time (see main())
_STAGE1_CKPT_OVERRIDE = None
_S2_DIR_OVERRIDE      = None

QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
QWEN_DIM   = 2048
MAX_LEN    = 512
MAX_STEPS  = 24
MAX_ING    = 20

sys.path.insert(0, str(BASE / "scripts"))
from process_vae_model import ProcessVAE
from process_vae_stage1 import RecipeStepDataset, collate_fn as step_collate

SYSTEM_PROMPT = (
    "You are a culinary writer. Given a cooking process and ingredient list, "
    "write a complete recipe with title, ingredients, and step-by-step instructions."
)


# ── Tokenise pass ──────────────────────────────────────────────────────────────

def tokenize_corpus(recipe_ids, raw_by_id, tokenizer, n_recipes, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pre-tokenising {n_recipes:,} recipes → {out_dir}", flush=True)
    pad_id = tokenizer.pad_token_id or 0
    kept, ids_list, mask_list, lbl_list = [], [], [], []
    t0 = time.time()
    for i, rid in enumerate(recipe_ids[:n_recipes]):
        raw = raw_by_id.get(rid)
        if raw is None: continue
        ings   = [x.get("text","").split(",")[0].strip()
                  for x in raw.get("ingredients",[])[:16] if x.get("text","")]
        instrs = [s.get("text","").strip()
                  for s in raw.get("instructions",[])[:15] if s.get("text","")]
        if not instrs: continue
        ing_text  = ", ".join(ings) if ings else "various ingredients"
        ing_lines = "\n".join(f"- {s}" for s in ings)
        instr_lines = "\n".join(f"{j+1}. {s}" for j,s in enumerate(instrs))
        recipe_text = (f"Title: {raw.get('title','Untitled')}\n\n"
                       f"Ingredients:\n{ing_lines}\n\nInstructions:\n{instr_lines}")
        full_msgs = [{"role": "system",    "content": SYSTEM_PROMPT},
                     {"role": "user",      "content": f"Ingredients: {ing_text}"},
                     {"role": "assistant", "content": recipe_text}]
        pfx_msgs  = [{"role": "system",    "content": SYSTEM_PROMPT},
                     {"role": "user",      "content": f"Ingredients: {ing_text}"}]
        full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False,
                                                   add_generation_prompt=False)
        pfx_text  = tokenizer.apply_chat_template(pfx_msgs,  tokenize=False,
                                                   add_generation_prompt=True)
        full_enc = tokenizer(full_text, max_length=MAX_LEN, truncation=True,
                             padding="max_length")
        pfx_enc  = tokenizer(pfx_text, max_length=MAX_LEN, truncation=True)
        n_prompt = len(pfx_enc["input_ids"])
        ids    = full_enc["input_ids"]
        labels = list(ids)
        for j in range(min(n_prompt, len(labels))): labels[j] = -100
        for j in range(len(labels)):
            if ids[j] == pad_id: labels[j] = -100
        if sum(1 for l in labels if l != -100) < 10: continue
        kept.append(rid)
        ids_list.append(ids)
        mask_list.append(full_enc["attention_mask"])
        lbl_list.append(labels)
        if (i+1) % 10000 == 0:
            print(f"  {i+1:,}/{n_recipes:,}  kept={len(kept):,}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    np.save(out_dir / "input_ids.npy",      np.array(ids_list,   dtype=np.int32))
    np.save(out_dir / "attention_mask.npy", np.array(mask_list,  dtype=np.uint8))
    np.save(out_dir / "labels.npy",         np.array(lbl_list,   dtype=np.int32))
    json.dump(kept, open(out_dir / "recipe_ids.json", "w"))
    print(f"Tokenised {len(kept):,} recipes in {time.time()-t0:.0f}s", flush=True)
    return kept


# ── Dataset ────────────────────────────────────────────────────────────────────

class CachedStage2Dataset(Dataset):
    """Uses pre-cached code IDs — no Stage 1 encoder needed at training time."""
    def __init__(self, code_ids, n_steps, input_ids, attention_mask, labels):
        self.code_ids       = code_ids        # (N, MAX_STEPS) int16
        self.n_steps        = n_steps         # (N,) int8
        self.input_ids      = input_ids
        self.attention_mask = attention_mask
        self.labels         = labels

    def __len__(self): return len(self.code_ids)

    def __getitem__(self, i):
        return {
            "code_ids":       torch.tensor(self.code_ids[i].astype(np.int64)),
            "n_steps":        int(self.n_steps[i]),
            "input_ids":      torch.tensor(self.input_ids[i],      dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[i], dtype=torch.long),
            "labels":         torch.tensor(self.labels[i],         dtype=torch.long),
        }


def collate_cached(batch):
    return {
        "code_ids":       torch.stack([b["code_ids"]       for b in batch]),
        "n_steps":        torch.tensor([b["n_steps"]       for b in batch], dtype=torch.long),
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
    }


class Stage2Dataset(Dataset):
    """Fallback dataset — runs Stage 1 live (slower, higher memory)."""
    def __init__(self, step_embs, per_ing_embs, step_index, per_ing_index,
                 recipe_ids, input_ids, attention_mask, labels):
        self.step_embs     = step_embs
        self.per_ing_embs  = per_ing_embs
        self.step_index    = step_index
        self.per_ing_index = per_ing_index
        self.recipe_ids    = recipe_ids
        self.input_ids     = input_ids
        self.attention_mask= attention_mask
        self.labels        = labels

    def __len__(self): return len(self.recipe_ids)

    def __getitem__(self, i):
        rid    = self.recipe_ids[i]
        ss, se = self.step_index.get(rid, (0, 0))
        steps  = self.step_embs[ss:min(se, ss+MAX_STEPS)]
        is_, ie= self.per_ing_index.get(rid, (0, 0))
        n_ing  = min(ie-is_, MAX_ING)
        ings   = self.per_ing_embs[is_:is_+n_ing]
        return {
            "steps":    torch.from_numpy(steps.copy()).float(),
            "ings":     torch.from_numpy(ings.copy()).float(),
            "n_steps":  len(steps),
            "n_ings":   n_ing,
            "input_ids":      torch.tensor(self.input_ids[i],       dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[i],  dtype=torch.long),
            "labels":         torch.tensor(self.labels[i],          dtype=torch.long),
        }


def collate_stage2(batch):
    B   = len(batch)
    max_T = max(b["n_steps"] for b in batch)
    max_N = max(b["n_ings"]  for b in batch)
    d     = 384
    steps = torch.zeros(B, max_T, d)
    smask = torch.zeros(B, max_T)
    ings  = torch.zeros(B, max_N, d)
    imask = torch.ones(B, max_N, dtype=torch.bool)
    for i, b in enumerate(batch):
        T, N = b["n_steps"], b["n_ings"]
        steps[i, :T] = b["steps"]
        smask[i, :T] = 1.0
        ings[i,  :N] = b["ings"]
        imask[i, :N] = False
    return {
        "steps": steps, "ings": ings, "step_mask": smask, "ing_mask": imask,
        "input_ids":      torch.stack([b["input_ids"]       for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"]  for b in batch]),
        "labels":         torch.stack([b["labels"]          for b in batch]),
    }


# ── Joint model ────────────────────────────────────────────────────────────────

class Stage2Model(nn.Module):
    def __init__(self, encoder, qwen, n_codes, qwen_dim=QWEN_DIM, d_code=64):
        super().__init__()
        self.encoder    = encoder     # frozen ProcessVAE encoder
        self.qwen       = qwen        # Qwen + LoRA
        self.code_emb   = nn.Embedding(n_codes, qwen_dim)
        nn.init.normal_(self.code_emb.weight, std=0.02)

    def forward(self, input_ids, attention_mask, labels,
                code_ids=None, n_steps=None,
                steps=None, ings=None, ing_mask=None):
        """
        Preferred path: pass code_ids (B, T) pre-cached int64 + n_steps (B,) for masking.
        Fallback path: pass steps/ings/ing_mask and run frozen encoder live.
        """
        if code_ids is not None:
            idx = code_ids                                    # (B, T)
            B, T = idx.shape
        else:
            B, T, _ = steps.shape
            with torch.no_grad():
                _, idx, _ = self.encoder.encode(steps, ings, ing_mask, use_vq=True)
            if idx is None:
                idx = torch.zeros(B, T, dtype=torch.long, device=steps.device)

        # Embed codes → prefix tokens
        prefix = self.code_emb(idx)    # (B, T, qwen_dim)

        tok_embs  = self.qwen.get_input_embeddings()(input_ids)
        prefix    = prefix.to(tok_embs.dtype)
        full_embs = torch.cat([prefix, tok_embs], dim=1)

        # Prefix attention mask: 0 for padding code positions (beyond n_steps)
        if n_steps is not None:
            arange   = torch.arange(T, device=attention_mask.device).unsqueeze(0)
            pfx_mask = (arange < n_steps.unsqueeze(1)).to(attention_mask.dtype)
        else:
            pfx_mask = torch.ones(B, T, dtype=attention_mask.dtype,
                                  device=attention_mask.device)
        pfx_lbl   = torch.full((B, T), -100, dtype=labels.dtype,
                               device=labels.device)
        full_mask = torch.cat([pfx_mask,  attention_mask], dim=1)
        full_lbl  = torch.cat([pfx_lbl,   labels],         dim=1)

        out = self.qwen(inputs_embeds=full_embs,
                        attention_mask=full_mask,
                        labels=full_lbl)
        return out.loss


# ── Cache codes (one-time pre-computation) ─────────────────────────────────────

def cache_codes(args):
    """Encode all tokenised recipes through frozen Stage 1, save code_ids.npy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    s1_path = _STAGE1_CKPT_OVERRIDE or (STAGE1_DIR / "stage1_final.pt")
    print(f"Stage 1 ckpt: {s1_path}", flush=True)
    ckpt1 = torch.load(s1_path, map_location="cpu",
                       weights_only=False)
    meta1 = ckpt1["meta"]
    encoder = ProcessVAE(n_codes=meta1["n_codes"], d_code=meta1["d_code"],
                         d_model=meta1["d_model"], n_heads=meta1.get("n_heads", 4))
    encoder.load_state_dict(ckpt1["model"])
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"  n_codes={meta1['n_codes']}  d_code={meta1['d_code']}", flush=True)

    step_embs    = np.load(DATA_DIR / "step_embs.npy",    mmap_mode="r")
    per_ing_embs = np.load(DATA_DIR / "per_ing_embs.npy", mmap_mode="r")
    step_index   = json.load(open(DATA_DIR / "step_index.json"))
    per_ing_idx  = json.load(open(DATA_DIR / "per_ing_index.json"))
    # recipe_ids.json is the same regardless of Stage 1 model — read from original tok dir
    ORIG_TOK_DIR = BASE / "data/models/process_vae_stage2/tokenized"
    text_tok_dir = ORIG_TOK_DIR if _STAGE1_CKPT_OVERRIDE else TOK_DIR
    TOK_DIR.mkdir(parents=True, exist_ok=True)
    tok_ids      = json.load(open(text_tok_dir / "recipe_ids.json"))
    N            = len(tok_ids)

    # Padding slots are zero-initialized (code 0), which is a valid code ID.
    # n_steps.npy records the true step count per recipe so the training forward
    # pass can mask out padding positions via pfx_mask — see Stage2Model.forward().
    all_codes  = np.zeros((N, MAX_STEPS), dtype=np.int16)
    all_nsteps = np.zeros(N,              dtype=np.int8)

    ENC_BATCH = 512
    t0 = time.time()
    for start in range(0, N, ENC_BATCH):
        batch_rids = tok_ids[start:start + ENC_BATCH]
        B = len(batch_rids)
        max_T = 0
        max_N = 0
        items = []
        for rid in batch_rids:
            ss, se  = step_index.get(rid, (0, 0))
            T       = min(se - ss, MAX_STEPS)
            is_, ie = per_ing_idx.get(rid, (0, 0))
            Ni      = min(ie - is_, MAX_ING)
            items.append((ss, T, is_, Ni))
            max_T = max(max_T, T)
            max_N = max(max_N, Ni)
        max_T = max(max_T, 1)
        max_N = max(max_N, 1)

        steps_t = torch.zeros(B, max_T, 384)
        ings_t  = torch.zeros(B, max_N, 384)
        imask   = torch.ones(B, max_N, dtype=torch.bool)
        for i, (ss, T, is_, Ni) in enumerate(items):
            if T > 0:
                steps_t[i, :T] = torch.from_numpy(
                    step_embs[ss:ss + T].copy()).float()
            if Ni > 0:
                ings_t[i, :Ni] = torch.from_numpy(
                    per_ing_embs[is_:is_ + Ni].copy()).float()
                imask[i, :Ni]  = False

        with torch.no_grad():
            _, idx, _ = encoder.encode(
                steps_t.to(device), ings_t.to(device), imask.to(device),
                use_vq=True)
        idx_np = idx.cpu().numpy().astype(np.int16)  # (B, max_T)
        for i, (_, T, _, _) in enumerate(items):
            all_codes[start + i, :idx_np.shape[1]]  = idx_np[i]
            all_nsteps[start + i]                   = T

        if (start // ENC_BATCH + 1) % 20 == 0:
            done = start + B
            print(f"  {done:,}/{N:,}  ({time.time()-t0:.0f}s)", flush=True)

    np.save(TOK_DIR / "code_ids.npy",  all_codes)
    np.save(TOK_DIR / "n_steps.npy",   all_nsteps)
    print(f"Saved code_ids.npy + n_steps.npy  ({N:,} recipes, {time.time()-t0:.0f}s)",
          flush=True)


# ── Training ───────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    code_ids_path = TOK_DIR / "code_ids.npy"
    use_cache     = code_ids_path.exists()

    # Load Stage 1 meta (always needed for n_codes/d_code) + encoder (only if no cache)
    s1_path = _STAGE1_CKPT_OVERRIDE or (STAGE1_DIR / "stage1_final.pt")
    print(f"Loading Stage 1 checkpoint: {s1_path}", flush=True)
    ckpt1 = torch.load(s1_path, map_location="cpu",
                       weights_only=False)
    meta1 = ckpt1["meta"]

    n_heads = meta1.get("n_heads", 4)
    if use_cache:
        print("  Pre-cached code IDs found — Stage 1 NOT loaded onto GPU.", flush=True)
        encoder = ProcessVAE(n_codes=meta1["n_codes"], d_code=meta1["d_code"],
                             d_model=meta1["d_model"], n_heads=n_heads)
        encoder.load_state_dict(ckpt1["model"])
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
        # Keep encoder on CPU — never moved to device
    else:
        print("  No cache found — Stage 1 will run live on GPU (slower).", flush=True)
        encoder = ProcessVAE(n_codes=meta1["n_codes"], d_code=meta1["d_code"],
                             d_model=meta1["d_model"], n_heads=n_heads)
        encoder.load_state_dict(ckpt1["model"])
        encoder.to(device).eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
    print(f"  n_codes={meta1['n_codes']}  d_code={meta1['d_code']}", flush=True)

    # Load Qwen + LoRA
    print(f"Loading {QWEN_MODEL}...", flush=True)
    qwen_base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, dtype=torch.bfloat16, trust_remote_code=True)

    if args.resume and (OUT_DIR / "lora_best").exists():
        from peft import PeftModel as _PeftModel
        print(f"Resuming LoRA from {OUT_DIR / 'lora_best'}", flush=True)
        qwen = _PeftModel.from_pretrained(qwen_base, OUT_DIR / "lora_best",
                                          is_trainable=True).to(device)
    elif args.warm_lora_from and Path(args.warm_lora_from).exists():
        from peft import PeftModel as _PeftModel
        print(f"Warm-starting LoRA from {args.warm_lora_from} (code_emb stays random)",
              flush=True)
        qwen = _PeftModel.from_pretrained(qwen_base, args.warm_lora_from,
                                          is_trainable=True).to(device)
    else:
        lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                              target_modules=["q_proj","v_proj"], lora_dropout=0.05)
        qwen = get_peft_model(qwen_base, lora_cfg).to(device)
    qwen.print_trainable_parameters()

    model = Stage2Model(encoder, qwen, meta1["n_codes"],
                        d_code=meta1["d_code"]).to(device)
    if use_cache:
        model.encoder = model.encoder.cpu()

    # Text tokenisation files (input_ids, labels) always live in the original Stage 2
    # tokenized dir. Only code_ids.npy / n_steps.npy are Stage-1-specific.
    ORIG_TOK_DIR = BASE / "data/models/process_vae_stage2/tokenized"
    text_tok_dir = ORIG_TOK_DIR if _STAGE1_CKPT_OVERRIDE else TOK_DIR
    tok_ids   = json.load(open(text_tok_dir / "recipe_ids.json"))
    input_ids = np.load(text_tok_dir / "input_ids.npy",      mmap_mode="r")
    attn_mask = np.load(text_tok_dir / "attention_mask.npy", mmap_mode="r")
    labels    = np.load(text_tok_dir / "labels.npy",         mmap_mode="r")
    print(f"  {len(tok_ids):,} tokenised recipes", flush=True)

    rng     = np.random.default_rng(42)
    n_total = len(tok_ids)
    n_val   = max(500, int(n_total * 0.03))
    perm    = rng.permutation(n_total)
    tr_idx  = perm[:n_total - n_val].tolist()
    val_idx = perm[n_total - n_val:].tolist()

    if use_cache:
        code_ids_arr = np.load(code_ids_path,          mmap_mode="r")
        n_steps_arr  = np.load(TOK_DIR / "n_steps.npy", mmap_mode="r")

        def make_ds(idx):
            return CachedStage2Dataset(
                code_ids_arr[idx], n_steps_arr[idx],
                input_ids[idx], attn_mask[idx], labels[idx])

        collate_fn = collate_cached
    else:
        step_embs    = np.load(DATA_DIR / "step_embs.npy",    mmap_mode="r")
        per_ing_embs = np.load(DATA_DIR / "per_ing_embs.npy", mmap_mode="r")
        step_index   = json.load(open(DATA_DIR / "step_index.json"))
        per_ing_idx  = json.load(open(DATA_DIR / "per_ing_index.json"))

        def make_ds(idx):
            rids = [tok_ids[i] for i in idx]
            return Stage2Dataset(step_embs, per_ing_embs, step_index, per_ing_idx,
                                 rids, input_ids[idx], attn_mask[idx], labels[idx])

        collate_fn = collate_stage2

    tr_loader  = DataLoader(make_ds(tr_idx),  batch_size=args.batch, shuffle=True,
                            num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(make_ds(val_idx), batch_size=args.batch, shuffle=False,
                            num_workers=4, collate_fn=collate_fn, pin_memory=True)

    opt   = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                        eta_min=1e-6)
    log_path  = OUT_DIR / "train_log.jsonl"
    start_ep  = 1
    best_val  = float("inf")

    if args.resume and (OUT_DIR / "lora_best").exists():
        # Full resume: reload both LoRA (already done above) AND code_emb, continue epoch count
        print("Resuming from lora_best/ + code_emb_best.pt ...", flush=True)
        emb_ckpt = torch.load(OUT_DIR / "code_emb_best.pt", map_location=device,
                               weights_only=False)
        model.code_emb.load_state_dict(emb_ckpt["code_emb"])
        if log_path.exists():
            past = [json.loads(l) for l in open(log_path)]
            if past:
                start_ep = past[-1]["ep"] + 1
                best_val = min(e["val_lm"] for e in past)
                for _ in range(start_ep - 1):
                    sched.step()
        print(f"  Resuming from epoch {start_ep}  best_val={best_val:.4f}", flush=True)
    elif args.warm_lora_from:
        # Warm LoRA start: LoRA already loaded above, code_emb stays randomly initialized.
        # Start fresh from epoch 1 — code_emb has never seen grammar codes before.
        print("  code_emb randomly initialized — will learn grammar code meanings from scratch.",
              flush=True)
        log_path.unlink(missing_ok=True)   # fresh log for this run
    else:
        log_path.unlink(missing_ok=True)

    n_train   = len(tr_loader)
    log_every = max(1, n_train // 20)   # log ~20× per epoch (every 5%)
    mode_tag  = "cached" if use_cache else "live-encoder"
    print(f"\nTraining Stage 2 [{mode_tag}] for {args.epochs} epochs... "
          f"({n_train} steps/ep, starting ep{start_ep})", flush=True)

    for ep in range(start_ep, args.epochs + 1):
        model.train(); tot = 0.0; n = 0; t0 = time.time()
        for batch in tr_loader:
            ids  = batch["input_ids"].to(device)
            att  = batch["attention_mask"].to(device)
            lbl  = batch["labels"].to(device)
            if use_cache:
                cids = batch["code_ids"].to(device)
                ns   = batch["n_steps"].to(device)
                loss = model(ids, att, lbl, code_ids=cids, n_steps=ns)
            else:
                loss = model(ids, att, lbl,
                             steps=batch["steps"].to(device),
                             ings=batch["ings"].to(device),
                             ing_mask=batch["ing_mask"].to(device))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
            if n % log_every == 0:
                elapsed = time.time() - t0
                sps     = n / elapsed
                eta     = (n_train - n) / sps
                print(f"  ep{ep} step {n}/{n_train}  loss={tot/n:.4f}  "
                      f"{sps:.2f} steps/s  eta={eta/60:.1f}min", flush=True)
        sched.step()

        model.eval(); vtot = 0.0; vn = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                att = batch["attention_mask"].to(device)
                lbl = batch["labels"].to(device)
                if use_cache:
                    cids = batch["code_ids"].to(device)
                    ns   = batch["n_steps"].to(device)
                    loss = model(ids, att, lbl, code_ids=cids, n_steps=ns)
                else:
                    loss = model(ids, att, lbl,
                                 steps=batch["steps"].to(device),
                                 ings=batch["ings"].to(device),
                                 ing_mask=batch["ing_mask"].to(device))
                vtot += loss.item(); vn += 1

        tr_lm = tot/n; val_lm = vtot/vn
        entry = {"ep": ep, "train_lm": round(tr_lm, 4),
                 "val_lm": round(val_lm, 4), "ts": time.time()}
        with open(log_path, "a") as f: f.write(json.dumps(entry) + "\n")
        print(f"  ep{ep:02d}  train_lm={tr_lm:.4f}  val_lm={val_lm:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if val_lm < best_val:
            best_val = val_lm
            model.qwen.save_pretrained(OUT_DIR / "lora_best")
            torch.save({"code_emb": model.code_emb.state_dict(),
                        "meta": meta1}, OUT_DIR / "code_emb_best.pt")
            print(f"  saved → {OUT_DIR}/lora_best", flush=True)

    model.qwen.save_pretrained(OUT_DIR / "lora_final")
    torch.save({"code_emb": model.code_emb.state_dict(), "meta": meta1},
               OUT_DIR / "code_emb_final.pt")
    print("\nStage 2 complete.", flush=True)


# ── Generate / eval ────────────────────────────────────────────────────────────

def generate(args):
    """Decode a few recipes from pre-cached code sequences using lora_best/."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt1 = torch.load(STAGE1_DIR / "stage1_final.pt", map_location="cpu",
                       weights_only=False)
    meta1 = ckpt1["meta"]

    print(f"Loading {QWEN_MODEL} + lora_best ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    qwen_base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, dtype=torch.bfloat16, trust_remote_code=True)
    from peft import PeftModel as _PeftModel
    qwen = _PeftModel.from_pretrained(qwen_base, OUT_DIR / "lora_best").to(device)
    qwen.eval()

    emb_ckpt = torch.load(OUT_DIR / "code_emb_best.pt", map_location=device,
                           weights_only=False)
    code_emb = nn.Embedding(meta1["n_codes"], QWEN_DIM).to(device)
    code_emb.load_state_dict(emb_ckpt["code_emb"])
    code_emb.eval()

    tok_ids      = json.load(open(TOK_DIR / "recipe_ids.json"))
    code_ids_arr = np.load(TOK_DIR / "code_ids.npy", mmap_mode="r")
    n_steps_arr  = np.load(TOK_DIR / "n_steps.npy",  mmap_mode="r")

    label_map = {}
    analysis_path = STAGE1_DIR / "codebook_analysis.json"
    if analysis_path.exists():
        for v in json.load(open(analysis_path)).values():
            label_map[v["code"]] = v.get("semantic_label", "?")

    if args.recipe_id:
        rids = [args.recipe_id]
    else:
        rng  = np.random.default_rng(42)
        rids = [tok_ids[i] for i in rng.choice(len(tok_ids), 5, replace=False)]

    tok_set = {r: i for i, r in enumerate(tok_ids)}
    for rid in rids:
        if rid not in tok_set:
            print(f"  {rid}: not in tokenised set"); continue
        idx      = tok_set[rid]
        T        = int(n_steps_arr[idx])
        cids     = code_ids_arr[idx, :T].astype(np.int64)
        code_str = "  ".join(f"c{c:03d}({label_map.get(c,'?')})" for c in cids)
        print(f"\n{'─'*70}")
        print(f"Recipe: {rid}  ({T} steps)")
        print(f"Codes:  {code_str}")

        cids_t   = torch.tensor(cids, dtype=torch.long, device=device).unsqueeze(0)
        prefix   = code_emb(cids_t)                          # (1, T, qwen_dim)

        prompt_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": "Ingredients: (from recipe codes)"},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True)
        prompt_enc  = tokenizer(prompt_text, return_tensors="pt").to(device)

        tok_embs  = qwen.get_input_embeddings()(prompt_enc["input_ids"])
        prefix    = prefix.to(tok_embs.dtype)
        full_embs = torch.cat([prefix, tok_embs], dim=1)
        pfx_mask  = torch.ones(1, T, dtype=torch.long, device=device)
        full_mask = torch.cat([pfx_mask, prompt_enc["attention_mask"]], dim=1)

        with torch.no_grad():
            out = qwen.generate(
                inputs_embeds=full_embs,
                attention_mask=full_mask,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0]  # generate() with inputs_embeds returns only new tokens
        print("Output:")
        print(tokenizer.decode(gen_ids, skip_special_tokens=True))

    print(f"\n{'─'*70}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   choices=["tokenize","cache_codes","train","generate"],
                        default="train")
    parser.add_argument("--n_recipes", type=int, default=50000)
    parser.add_argument("--batch",     type=int, default=16)
    parser.add_argument("--epochs",    type=int, default=5)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--recipe_id", type=str, default=None)
    parser.add_argument("--resume",    action="store_true",
                        help="Resume from lora_best/ checkpoint (loads both LoRA + code_emb)")
    parser.add_argument("--warm_lora_from", type=str, default=None,
                        help="Path to a lora_best/ dir to warm-start LoRA weights from, "
                             "while keeping code_emb randomly initialized. Use this when "
                             "the Stage 1 codebook changed (e.g. grammar retrain) so "
                             "code_emb must re-learn from scratch but LoRA is reusable.")
    parser.add_argument("--stage1_ckpt", type=str, default=None,
                        help="Override Stage 1 checkpoint path (default: process_vae/stage1_final.pt)")
    parser.add_argument("--s2_dir", type=str, default=None,
                        help="Override Stage 2 output dir (default: data/models/process_vae_stage2)")
    args = parser.parse_args()

    # Apply directory/checkpoint overrides globally before any mode runs
    global OUT_DIR, TOK_DIR, STAGE1_DIR, _STAGE1_CKPT_OVERRIDE
    if args.s2_dir:
        OUT_DIR = Path(args.s2_dir)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        TOK_DIR = OUT_DIR / "tokenized"
        # Tokenized text files (input_ids etc.) live in original tokenized dir — share them
        # Only code_ids.npy and n_steps.npy are grammar-specific; everything else is reused
    if args.stage1_ckpt:
        _STAGE1_CKPT_OVERRIDE = Path(args.stage1_ckpt)

    if args.mode == "tokenize":
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
        recipe_ids = json.load(open(DATA_DIR / "recipe_ids.json"))
        print(f"Loading {LAYER1}...", flush=True)
        raw = json.load(open(LAYER1))
        raw_by_id = {r["id"]: r for r in raw}
        tokenize_corpus(recipe_ids, raw_by_id, tokenizer, args.n_recipes, TOK_DIR)

    elif args.mode == "cache_codes":
        cache_codes(args)

    elif args.mode == "train":
        train(args)

    elif args.mode == "generate":
        generate(args)


if __name__ == "__main__":
    main()
