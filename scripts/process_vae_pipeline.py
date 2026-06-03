#!/usr/bin/env python3
"""
Stage 4: End-to-end pipeline — raw recipe text → DAG → (optionally) text out.

This is the complete system:
  1. Split raw text into steps (split_atomic handles compound steps)
  2. Embed steps + ingredients with MiniLM
  3. Stage 1 VQ encoder → discrete codes per step
  4. Stage 3 DAG extractor → explicit graph (step nodes + edges + ingredient sources)
  5. [Optional] Stage 2 Qwen decoder → regenerate recipe text from DAG

The DAG in the middle is interpretable, editable, and queryable.
You can modify it between encode and decode to do recipe editing.

Prerequisites:
  Stage 1: data/models/process_vae/stage1_final.pt          (required)
  Stage 2: data/models/process_vae_stage2/lora_best/        (optional, for decode)
           data/models/process_vae_stage2/code_emb_best.pt  (optional, for decode)

Usage:
  # Encode only — get the DAG
  python3 scripts/process_vae_pipeline.py --recipe "Mix flour and butter. Bake at 350F."

  # Encode + decode — roundtrip
  python3 scripts/process_vae_pipeline.py --recipe "..." --decode

  # From a recipe file
  python3 scripts/process_vae_pipeline.py --recipe_file my_recipe.txt --decode

  # Encode all recipes in a JSONL and save DAGs
  python3 scripts/process_vae_pipeline.py --encode_corpus --out data/dags/learned_dags.jsonl
"""

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

BASE     = Path(__file__).resolve().parent.parent
VAE_DIR  = BASE / "data/models/process_vae"
S2_DIR   = BASE / "data/models/process_vae_stage2"

sys.path.insert(0, str(BASE / "scripts"))
from process_vae_model import ProcessVAE, split_atomic
from process_vae_stage3 import load_encoder, extract_dag, visualise_dag


# ── Text preprocessing ─────────────────────────────────────────────────────────

def parse_recipe_text(raw_text):
    """
    Parse a raw recipe string into (step_texts, ingredient_texts).
    Handles both structured (Ingredients:/Instructions:) and unstructured text.
    """
    lines = [l.strip() for l in raw_text.strip().split("\n") if l.strip()]
    step_texts = []; ing_texts = []
    in_ings = False; in_instrs = False

    for line in lines:
        ll = line.lower()
        if "ingredient" in ll and len(line) < 30:
            in_ings = True; in_instrs = False; continue
        if any(k in ll for k in ["instruction", "direction", "method", "steps"]) \
                and len(line) < 30:
            in_ings = False; in_instrs = True; continue
        # strip list markers
        clean = line.lstrip("•-–0123456789.) ").strip()
        if not clean: continue
        if in_ings:
            ing_texts.append(clean)
        elif in_instrs:
            # split compound steps
            for sub in split_atomic(clean):
                step_texts.append(sub)
        else:
            # unstructured: assume it's all instructions
            for sub in split_atomic(clean):
                step_texts.append(sub)

    if not step_texts:
        # fallback: treat every line as a step
        for line in lines:
            for sub in split_atomic(line):
                step_texts.append(sub)

    return step_texts, ing_texts


def embed_texts(texts, encoder_model, device, batch_size=512):
    """MiniLM-embed a list of strings → (N, 384) float32 numpy array."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs  = encoder_model.encode(batch, convert_to_numpy=True,
                                      show_progress_bar=False,
                                      normalize_embeddings=False)
        all_embs.append(embs.astype(np.float32))
    return np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, 384))


# ── Encoder (Stage 1 + 3) ──────────────────────────────────────────────────────

def encode_recipe(raw_text, vae_model, miniml, device,
                  top_k_edges=2, codebook_labels=None):
    """
    raw_text → explicit DAG.

    Returns
    -------
    dag : dict  (same format as process_vae_stage3.py output)
    step_texts : list[str]
    ing_texts  : list[str]
    """
    step_texts, ing_texts = parse_recipe_text(raw_text)
    if not step_texts:
        raise ValueError("Could not parse any steps from recipe text.")

    step_embs = embed_texts(step_texts, miniml, device)
    ing_embs  = embed_texts(ing_texts,  miniml, device) if ing_texts \
                else np.zeros((1, 384), dtype=np.float32)

    dag = extract_dag(vae_model, step_embs, ing_embs, ing_texts,
                      step_texts, "input_recipe",
                      top_k_edges=top_k_edges)
    return dag, step_texts, ing_texts


def print_dag(dag, codebook_labels=None):
    """Pretty-print a DAG to stdout."""
    print(f"\n{'='*60}")
    print(f"Recipe DAG  ({dag['n_steps']} steps, {len(dag['edges'])} edges)")
    print(f"{'='*60}")
    print("\nNODES (operation sequence):")
    for node in dag["nodes"]:
        code  = node["code"]
        label = ""
        if codebook_labels:
            sample = codebook_labels.get(str(code), [])
            label  = f"  [{sample[0][:40]}]" if sample else ""
        ings = ", ".join(node["ingredients"][:4]) or "—"
        print(f"  step {node['t']:2d}  code={code:3d}{label}")
        print(f"          \"{node['step_text'][:70]}\"")
        print(f"          ingredients: {ings}")

    print("\nSTEP→STEP EDGES (process flow):")
    if dag["edges"]:
        for e in dag["edges"]:
            print(f"  step {e['from']} → step {e['to']}  (w={e['weight']:.3f})")
    else:
        print("  (linear recipe — no branching)")

    print("\nINGREDIENT→STEP SOURCES:")
    by_step = {}
    for se in dag["source_edges"]:
        by_step.setdefault(se["step"], []).append(se["ingredient"])
    for t, ings in sorted(by_step.items()):
        print(f"  step {t}: {', '.join(ings[:6])}")
    print()


# ── Decoder (Stage 2) ──────────────────────────────────────────────────────────

def load_decoder(s2_dir, vae_meta, device):
    """Load Stage 2 Qwen decoder if available."""
    lora_path   = Path(s2_dir) / "lora_best"
    code_path   = Path(s2_dir) / "code_emb_best.pt"
    if not (lora_path.exists() and code_path.exists()):
        return None, None, None

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer  = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    base       = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    qwen       = PeftModel.from_pretrained(base, str(lora_path)).to(device)
    qwen.eval()

    code_ckpt  = torch.load(code_path, map_location="cpu", weights_only=False)
    import torch.nn as nn
    code_emb   = nn.Embedding(vae_meta["n_codes"], 2048)
    code_emb.load_state_dict(code_ckpt["code_emb"])
    code_emb   = code_emb.to(device)

    return qwen, tokenizer, code_emb


@torch.no_grad()
def decode_dag(dag, ing_texts, qwen, tokenizer, code_emb, vae_model, device,
               max_new_tokens=512):
    """
    DAG + ingredient list → generated recipe text via Stage 2 Qwen decoder.
    """
    SYSTEM_PROMPT = (
        "You are a culinary writer. Given a cooking process and ingredient list, "
        "write a complete recipe with title, ingredients, and step-by-step instructions."
    )
    ing_text = ", ".join(ing_texts[:16]) if ing_texts else "various ingredients"

    # Build prefix from code embeddings
    codes = [node["code"] for node in dag["nodes"]]
    idx_t = torch.tensor(codes, dtype=torch.long, device=device).unsqueeze(0)
    prefix = code_emb(idx_t)   # (1, T, 2048)

    # Tokenise prompt
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Ingredients: {ing_text}"}]
    prompt  = tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
    enc     = tokenizer(prompt, return_tensors="pt").to(device)
    tok_emb = qwen.get_input_embeddings()(enc["input_ids"])

    full_embs = torch.cat([prefix, tok_emb], dim=1)
    pfx_mask  = torch.ones(1, prefix.shape[1],
                           dtype=enc["attention_mask"].dtype, device=device)
    full_mask = torch.cat([pfx_mask, enc["attention_mask"]], dim=1)

    out = qwen.generate(
        inputs_embeds=full_embs,
        attention_mask=full_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id)

    # generate() with inputs_embeds returns only newly generated tokens
    generated = out[0]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--recipe",      type=str,
                       help="Raw recipe text (steps as newline-separated or prose)")
    group.add_argument("--recipe_file", type=str,
                       help="Path to text file containing a recipe")
    group.add_argument("--encode_corpus", action="store_true",
                       help="Encode all 1M cached recipes → learned_dags.jsonl")

    parser.add_argument("--decode",     action="store_true",
                        help="Also run Stage 2 Qwen decoder (requires trained Stage 2)")
    parser.add_argument("--visualise",  action="store_true",
                        help="Save a DAG figure")
    parser.add_argument("--ckpt",       default=str(VAE_DIR / "stage1_final.pt"))
    parser.add_argument("--top_k_edges",type=int, default=2)
    parser.add_argument("--out",        default="data/dags/learned_dags.jsonl",
                        help="Output path for --encode_corpus")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load VQ encoder
    vae_model, vae_meta = load_encoder(args.ckpt, device)

    # Load MiniLM
    from sentence_transformers import SentenceTransformer
    print("Loading MiniLM...", flush=True)
    miniml = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                  device=str(device))

    # Load codebook labels if available
    lbl_path = VAE_DIR / "codebook_labels.json"
    codebook_labels = json.load(open(lbl_path)) if lbl_path.exists() else None

    # ── Corpus encode ──────────────────────────────────────────────────────────
    if args.encode_corpus:
        from process_vae_stage3 import extract_all
        DATA_DIR = BASE / "data/models/embeddings"
        step_embs    = np.load(DATA_DIR / "step_embs.npy",    mmap_mode="r")
        per_ing_embs = np.load(DATA_DIR / "per_ing_embs.npy", mmap_mode="r")
        step_texts   = json.load(open(DATA_DIR / "step_texts.json"))
        step_index   = json.load(open(DATA_DIR / "step_index.json"))
        per_ing_idx  = json.load(open(DATA_DIR / "per_ing_index.json"))
        recipe_ids   = json.load(open(DATA_DIR / "recipe_ids.json"))
        layer1 = Path(os.environ.get("RECIPE1M", str(BASE.parent / "layer1.json")))
        ing_by_rid = {}
        if layer1.exists():
            raw = json.load(open(layer1))
            for r in raw:
                ing_by_rid[r["id"]] = [
                    x.get("text","").split(",")[0].strip()
                    for x in r.get("ingredients",[])[:20] if x.get("text","")]
        extract_all(vae_model, recipe_ids, step_embs, per_ing_embs,
                    step_index, per_ing_idx, step_texts, ing_by_rid,
                    Path(args.out), top_k_edges=args.top_k_edges)
        return

    # ── Single recipe ──────────────────────────────────────────────────────────
    if args.recipe_file:
        raw_text = open(args.recipe_file).read()
    else:
        raw_text = args.recipe

    dag, step_texts, ing_texts = encode_recipe(
        raw_text, vae_model, miniml, device,
        top_k_edges=args.top_k_edges, codebook_labels=codebook_labels)

    print_dag(dag, codebook_labels)

    if args.visualise:
        out_fig = BASE / "data/figures/pipeline_dag.png"
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        visualise_dag(dag, codebook_labels, out_fig)

    if args.decode:
        qwen, tokenizer, code_emb = load_decoder(S2_DIR, vae_meta, device)
        if qwen is None:
            print("Stage 2 decoder not found — run process_vae_stage2.py first.")
        else:
            print("\n" + "="*60)
            print("DECODED RECIPE (Stage 2 Qwen output):")
            print("="*60)
            text = decode_dag(dag, ing_texts, qwen, tokenizer,
                              code_emb, vae_model, device)
            print(text)


if __name__ == "__main__":
    main()
