#!/usr/bin/env python3
"""
Stage 3: DAG Extractor — convert Stage 1 soft representations into explicit graphs.

Takes the trained VQ encoder and extracts three things per recipe:
  1. Node labels    : VQ code per step (learned operation type)
  2. Step→step edges: threshold causal self-attention → which step feeds which
  3. Ingredient→step: threshold ingredient cross-attention → ingredient source nodes

Output: one JSON DAG per recipe, replacing constrained_dags_14b.jsonl with a
learned-grammar version that doesn't require a hand-designed 15-type vocabulary.

Edge extraction:
Self-attention attn[t, s] = "how much step t draws from step s output."
We keep edge s→t if attn[t, s] > 1/T (above uniform = step s is a real predecessor).
Additionally keep at most --top_k predecessors per step (sparsity control).

Ingredient assignment:
Cross-attention attn[t, n] = "how much step t uses ingredient n."
Hard assignment: ingredient n → step t* = argmax_t attn[t, n].
This gives explicit bipartite ingredient→step source edges.

Prerequisites:
Stage 1 training must be complete (stage1_final.pt must exist).
Previous phases (1-35, grammar-constrained encoder) are NOT required —
only used optionally for grammar alignment comparison in process_vae_eval.py.

Usage:
  # Extract DAGs for all 1M recipes (saves to data/dags/learned_dags.jsonl)
  python3 scripts/process_vae_stage3.py

  # Quick test on 1000 recipes
  python3 scripts/process_vae_stage3.py --n_recipes 1000 --out data/dags/learned_dags_test.jsonl

  # Inspect a single recipe by id
  python3 scripts/process_vae_stage3.py --recipe_id 000018c8a5
"""

import argparse, json, os, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE     = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data/models/embeddings"
VAE_DIR  = BASE / "data/models/process_vae"
OUT_DIR  = BASE / "data/dags"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE / "scripts"))
from process_vae_model import ProcessVAE


# ── Load model ─────────────────────────────────────────────────────────────────

def load_encoder(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta  = ckpt["meta"]
    model = ProcessVAE(n_codes=meta["n_codes"], d_code=meta["d_code"],
                       d_model=meta["d_model"], n_heads=meta.get("n_heads", 4))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, meta


# ── DAG extraction for one recipe ─────────────────────────────────────────────

@torch.no_grad()
def extract_dag(model, step_embs_np, ing_embs_np, ing_texts,
                step_texts, recipe_id, top_k_edges=2, attn_threshold_factor=1.0):
    """
    Extract an explicit DAG from one recipe.

    Parameters
    ----------
    step_embs_np : (T, 384) float32
    ing_embs_np  : (N, 384) float32
    ing_texts    : list of N ingredient strings
    step_texts   : list of T step strings
    top_k_edges  : max predecessors to keep per step
    attn_threshold_factor : keep edge s→t if attn[t,s] > factor/T

    Returns
    -------
    dag : dict with keys id, nodes, edges, source_edges, step_texts
    """
    device = next(model.parameters()).device
    T = len(step_embs_np)
    N = len(ing_embs_np)
    if T < 1 or N < 1:
        return None

    steps = torch.from_numpy(step_embs_np).float().unsqueeze(0).to(device)  # (1,T,384)
    ings  = torch.from_numpy(ing_embs_np).float().unsqueeze(0).to(device)   # (1,N,384)
    imask = torch.zeros(1, N, dtype=torch.bool, device=device)

    z_q, idx, s_attn, c_attn = model.encode_dag(steps, ings, imask)
    # s_attn: (1,T,T)   c_attn: (1,T,N)
    s_attn = s_attn[0].cpu().numpy()   # (T, T)
    c_attn = c_attn[0].cpu().numpy()   # (T, N)
    codes  = idx[0].cpu().tolist()     # list of T ints

    # ── Step→step edges ───────────────────────────────────────────────────────
    # s_attn[t, s] = attention weight step t pays to step s
    # Only keep causal (s < t) edges above threshold
    threshold = attn_threshold_factor / max(T, 1)
    edges = []
    for t in range(1, T):
        # candidates: steps s < t with attn above threshold
        row = s_attn[t, :t]
        candidates = [(s, float(row[s])) for s in range(t)
                      if row[s] > threshold]
        # keep top_k by attention weight
        candidates.sort(key=lambda x: -x[1])
        for s, w in candidates[:top_k_edges]:
            edges.append({"from": s, "to": t, "weight": round(w, 4)})

    # ── Ingredient→step assignment ────────────────────────────────────────────
    # c_attn[t, n] = attention weight step t pays to ingredient n
    # Hard assign each ingredient to the step that attends to it most
    source_edges = []
    if N > 0:
        ing_step = c_attn.argmax(axis=0)   # (N,) — which step each ingredient goes to
        for n, t in enumerate(ing_step):
            if n < len(ing_texts):
                source_edges.append({
                    "ingredient": ing_texts[n],
                    "step": int(t),
                    "weight": round(float(c_attn[t, n]), 4)
                })

    # ── Nodes ─────────────────────────────────────────────────────────────────
    # Collect which ingredients were assigned to each step
    step_ings = defaultdict(list)
    for se in source_edges:
        step_ings[se["step"]].append(se["ingredient"])

    nodes = []
    for t in range(T):
        nodes.append({
            "t":           t,
            "code":        codes[t],
            "step_text":   step_texts[t] if t < len(step_texts) else "",
            "ingredients": step_ings[t],
        })

    return {
        "id":           recipe_id,
        "nodes":        nodes,
        "edges":        edges,           # step→step flow
        "source_edges": source_edges,   # ingredient→step
        "n_steps":      T,
        "n_ings":       N,
    }


# ── Batch extraction ───────────────────────────────────────────────────────────

MAX_STEPS = 24
MAX_ING   = 20


def _assign_ingredients(ca, step_texts, ing_texts, text_weight=0.0):
    """
    Assign each ingredient to its most likely step.

    Priority order (text_weight > 0):
      1. Verbatim: ingredient name appears as substring in step text → guaranteed
         win; among tied verbatim steps pick highest attention.
      2. Hybrid fallback: (1-α)*attention + α*jaccard word overlap.
    Pure attention (text_weight=0.0): argmax over ca[:, n].

    Returns: (Ni,) int array — step index for each ingredient.
    """
    T, Ni = ca.shape
    if text_weight == 0.0 or not step_texts:
        return ca.argmax(axis=0)

    assignments = np.empty(Ni, dtype=np.int64)
    for n in range(Ni):
        if n >= len(ing_texts):
            assignments[n] = int(ca[:, n].argmax())
            continue

        ing_lower = ing_texts[n].lower()
        ing_words = set(ing_lower.split())

        # Priority 1: verbatim substring match — attention decides among ties
        verbatim = [t for t in range(min(T, len(step_texts)))
                    if ing_lower in step_texts[t].lower()]
        if verbatim:
            assignments[n] = max(verbatim, key=lambda t: ca[t, n])
            continue

        # Priority 2: hybrid attention + jaccard word overlap
        best_t = int(ca[:, n].argmax()); best_score = -1.0
        for t in range(T):
            attn_score = float(ca[t, n])
            if t < len(step_texts):
                step_words = set(step_texts[t].lower().split())
                union      = len(ing_words | step_words)
                jaccard    = len(ing_words & step_words) / union if union else 0.0
            else:
                jaccard = 0.0
            score = (1.0 - text_weight) * attn_score + text_weight * jaccard
            if score > best_score:
                best_score = score; best_t = t
        assignments[n] = best_t
    return assignments


def extract_all(model, recipe_ids, step_embs, per_ing_embs,
                step_index, per_ing_index, step_texts, ing_texts_by_rid,
                out_path, top_k_edges=2, max_recipes=None, batch_size=256,
                text_weight=0.0):
    """
    GPU-batched DAG extraction. Processes `batch_size` recipes per forward pass
    (55x faster than batch=1 on this hardware).

    text_weight: float in [0, 1]. Controls ingredient assignment hybrid (Option B).
      0.0 = pure cross-attention argmax (original behaviour)
      0.3 = 70% attention + 30% text Jaccard overlap (recommended hybrid)
      1.0 = pure text matching
    """
    ids    = recipe_ids[:max_recipes] if max_recipes else recipe_ids
    device = next(model.parameters()).device
    n_ok = 0; n_skip = 0
    t0   = time.time()

    with open(out_path, "w") as f:
        for batch_start in range(0, len(ids), batch_size):
            chunk_rids = ids[batch_start:batch_start + batch_size]

            # ── Gather valid recipes in this chunk ────────────────────────────
            items = []   # (rid, ss, T, is_, Ni)
            for rid in chunk_rids:
                ss, se  = step_index.get(rid, (0, 0))
                if se <= ss: n_skip += 1; continue
                is_, ie = per_ing_index.get(rid, (0, 0))
                Ni      = min(ie - is_, MAX_ING)
                if Ni == 0: n_skip += 1; continue
                T = min(se - ss, MAX_STEPS)
                items.append((rid, ss, T, is_, Ni))

            if not items:
                continue

            B     = len(items)
            max_T = max(T  for _, _, T,  _, _  in items)
            max_N = max(Ni for _, _, _, _, Ni in items)

            # ── Build padded GPU tensors ──────────────────────────────────────
            steps_t = torch.zeros(B, max_T, 384, device=device)
            ings_t  = torch.zeros(B, max_N, 384, device=device)
            imask   = torch.ones( B, max_N, dtype=torch.bool, device=device)

            for i, (rid, ss, T, is_, Ni) in enumerate(items):
                steps_t[i, :T]  = torch.from_numpy(
                    step_embs[ss:ss+T].astype(np.float32))
                ings_t[i,  :Ni] = torch.from_numpy(
                    per_ing_embs[is_:is_+Ni].astype(np.float32))
                imask[i,   :Ni] = False

            # ── Single batched forward pass ───────────────────────────────────
            with torch.no_grad():
                _, idx, s_attn, c_attn = model.encode_dag(steps_t, ings_t, imask)

            idx_np    = idx.cpu().numpy()     # (B, max_T)
            s_attn_np = s_attn.cpu().numpy()  # (B, max_T, max_T)
            c_attn_np = c_attn.cpu().numpy()  # (B, max_T, max_N)

            # ── Unpack per recipe ─────────────────────────────────────────────
            for i, (rid, ss, T, is_, Ni) in enumerate(items):
                codes = idx_np[i, :T].tolist()
                sa    = s_attn_np[i, :T, :T]   # (T, T) — unpadded
                ca    = c_attn_np[i, :T, :Ni]  # (T, Ni) — unpadded
                s_txts = step_texts[ss:ss + T]
                i_txts = ing_texts_by_rid.get(rid, [])[:Ni]

                # Step→step edges
                threshold = 1.0 / max(T, 1)
                edges = []
                for t in range(1, T):
                    row = sa[t, :t]
                    cands = [(s, float(row[s])) for s in range(t) if row[s] > threshold]
                    cands.sort(key=lambda x: -x[1])
                    for s, w in cands[:top_k_edges]:
                        edges.append({"from": s, "to": t, "weight": round(w, 4)})

                # Ingredient→step assignment
                source_edges = []
                if i_txts:
                    ing_step = _assign_ingredients(ca, s_txts, i_txts, text_weight)
                    for n, t in enumerate(ing_step):
                        if n < len(i_txts):
                            source_edges.append({
                                "ingredient": i_txts[n],
                                "step": int(t),
                                "weight": round(float(ca[t, n]), 4),
                            })

                step_ings = defaultdict(list)
                for se_item in source_edges:
                    step_ings[se_item["step"]].append(se_item["ingredient"])

                nodes = [{"t": t, "code": codes[t],
                          "step_text": s_txts[t] if t < len(s_txts) else "",
                          "ingredients": step_ings[t]}
                         for t in range(T)]

                dag = {"id": rid, "nodes": nodes, "edges": edges,
                       "source_edges": source_edges, "n_steps": T, "n_ings": Ni}
                f.write(json.dumps(dag) + "\n")
                n_ok += 1

            total_done = batch_start + len(chunk_rids)
            if total_done % 50000 < batch_size or total_done >= len(ids):
                elapsed = time.time() - t0
                rate    = total_done / elapsed
                eta_min = (len(ids) - total_done) / rate / 60
                print(f"  {total_done:,}/{len(ids):,}  ok={n_ok:,}  "
                      f"({rate:.0f} recipes/s  ~{eta_min:.1f}min left)", flush=True)

    print(f"\nExtracted {n_ok:,} DAGs ({n_skip:,} skipped) → {out_path}", flush=True)
    return n_ok


# ── Visualise a single DAG ────────────────────────────────────────────────────

def visualise_dag(dag, codebook_labels, out_path):
    """Draw ingredient→step→dish DAG for one recipe."""
    nodes    = dag["nodes"]
    edges    = dag["edges"]
    src_edgs = dag["source_edges"]
    T        = dag["n_steps"]

    # Assign x positions by step index, y = 0.5 for all step nodes
    fig, ax = plt.subplots(figsize=(max(10, T * 1.5), 6))
    ax.set_xlim(-1, T + 0.5); ax.set_ylim(-0.5, 2.5); ax.axis("off")
    ax.set_title(f"Recipe DAG: {dag['id']}", fontsize=11, fontweight="bold")

    step_pos = {t: (t, 1.0) for t in range(T)}

    # Draw step nodes
    palette = plt.cm.tab20(np.linspace(0, 1, 20))
    for node in nodes:
        t     = node["t"]
        code  = node["code"]
        x, y  = step_pos[t]
        label = (codebook_labels.get(str(code), ["?"])[0][:20]
                 if codebook_labels else f"c{code}")
        color = palette[code % 20]
        circ  = plt.Circle((x, y), 0.3, color=color, alpha=0.85, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y + 0.02, f"c{code}", ha="center", va="center",
                fontsize=7, fontweight="bold", zorder=4)
        ax.text(x, y - 0.38, label[:18], ha="center", va="top",
                fontsize=6, color="#333", zorder=4)

    # Draw step→step edges
    for e in edges:
        xs, ys = step_pos[e["from"]]
        xt, yt = step_pos[e["to"]]
        ax.annotate("", xy=(xt - 0.3, yt), xytext=(xs + 0.3, ys),
                    arrowprops=dict(arrowstyle="-|>", color="#555",
                                   lw=1.5 * e["weight"] * 4,
                                   mutation_scale=12), zorder=2)

    # Draw ingredient nodes (top row) and source edges
    ing_positions = {}
    ings_shown    = [se for se in src_edgs if se["weight"] > 0.05][:T * 3]
    for j, se in enumerate(ings_shown):
        xp = se["step"] + (j % 3 - 1) * 0.25
        yp = 2.2
        ing_positions[se["ingredient"]] = (xp, yp)
        ax.text(xp, yp, se["ingredient"][:12], ha="center", va="bottom",
                fontsize=5.5, color="#226",
                bbox=dict(boxstyle="round,pad=0.1", fc="#ddf", ec="#aac", lw=0.5))
        xt, yt = step_pos[se["step"]]
        ax.annotate("", xy=(xt, yt + 0.3), xytext=(xp, yp - 0.1),
                    arrowprops=dict(arrowstyle="-|>", color="#88a",
                                   lw=0.8, mutation_scale=8), zorder=1)

    # Dish sink node
    ax.text(T - 0.5, -0.3, "🍽 dish", ha="center", va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffd", ec="#aa8", lw=1.5))
    xt, yt = step_pos[T - 1]
    ax.annotate("", xy=(T - 0.5, -0.15), xytext=(xt, yt - 0.3),
                arrowprops=dict(arrowstyle="-|>", color="#aa8",
                               lw=1.5, mutation_scale=12), zorder=2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"DAG figure → {out_path}", flush=True)


# ── Stats ──────────────────────────────────────────────────────────────────────

def dag_stats(out_path, sample=10000):
    """Print summary statistics over extracted DAGs."""
    edges_per_recipe, ings_per_step, steps_per_recipe = [], [], []
    n = 0
    with open(out_path) as f:
        for line in f:
            if n >= sample: break
            dag = json.loads(line)
            edges_per_recipe.append(len(dag["edges"]))
            steps_per_recipe.append(dag["n_steps"])
            for node in dag["nodes"]:
                ings_per_step.append(len(node["ingredients"]))
            n += 1
    print(f"\nDAG statistics (n={n:,}):")
    print(f"  steps/recipe: mean={np.mean(steps_per_recipe):.1f}  "
          f"median={np.median(steps_per_recipe):.0f}")
    print(f"  edges/recipe: mean={np.mean(edges_per_recipe):.1f}  "
          f"(branching factor: {np.mean(edges_per_recipe)/max(np.mean(steps_per_recipe),1):.2f})")
    print(f"  ings/step:    mean={np.mean(ings_per_step):.1f}  "
          f"median={np.median(ings_per_step):.0f}")
    no_edge = sum(1 for e in edges_per_recipe if e == 0)
    print(f"  linear recipes (0 branches): {no_edge/n:.1%}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       default=str(VAE_DIR / "stage1_final.pt"))
    parser.add_argument("--out",        default=str(OUT_DIR / "learned_dags.jsonl"))
    parser.add_argument("--n_recipes",  type=int, default=None,
                        help="Limit to first N recipes (default: all 1M)")
    parser.add_argument("--top_k_edges",type=int, default=2,
                        help="Max predecessor edges per step")
    parser.add_argument("--recipe_id",  type=str, default=None,
                        help="Inspect + visualise a single recipe by id")
    parser.add_argument("--ing_file",   type=str, default=None,
                        help="JSON file mapping recipe_id→[ingredient strings]. "
                             "If absent, ingredient text will be empty.")
    parser.add_argument("--batch",       type=int,   default=256,
                        help="Recipes per GPU forward pass (default 256, ~55x faster than 1)")
    parser.add_argument("--text_weight", type=float, default=0.0,
                        help="Option B: ingredient assignment text weight α (0=pure attn, 0.3=hybrid)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    model, meta = load_encoder(args.ckpt, device)
    print(f"Encoder: n_codes={meta['n_codes']}  d_code={meta['d_code']}", flush=True)

    print("Loading embeddings...", flush=True)
    step_embs    = np.load(DATA_DIR / "step_embs.npy",    mmap_mode="r")
    per_ing_embs = np.load(DATA_DIR / "per_ing_embs.npy", mmap_mode="r")
    step_texts   = json.load(open(DATA_DIR / "step_texts.json"))
    step_index   = json.load(open(DATA_DIR / "step_index.json"))
    per_ing_idx  = json.load(open(DATA_DIR / "per_ing_index.json"))
    recipe_ids   = json.load(open(DATA_DIR / "recipe_ids.json"))

    # Optional: ingredient text for source node labels
    ing_texts_by_rid = {}
    if args.ing_file and Path(args.ing_file).exists():
        ing_texts_by_rid = json.load(open(args.ing_file))
    else:
        # Try to load from layer1.json (ingredient names)
        layer1 = Path(os.environ.get("RECIPE1M", str(BASE.parent / "layer1.json")))
        if layer1.exists():
            print("Loading ingredient names from layer1.json...", flush=True)
            raw = json.load(open(layer1))
            for r in raw:
                ings = [x.get("text","").split(",")[0].strip()
                        for x in r.get("ingredients",[])[:20]
                        if x.get("text","")]
                ing_texts_by_rid[r["id"]] = ings
            print(f"  Loaded ingredient names for {len(ing_texts_by_rid):,} recipes",
                  flush=True)

    # Single-recipe inspection mode
    if args.recipe_id:
        rid = args.recipe_id
        ss, se = step_index.get(rid, (0, 0))
        is_, ie = per_ing_idx.get(rid, (0, 0))
        n_ing = min(ie - is_, 20)
        dag = extract_dag(
            model,
            step_embs[ss:se].astype(np.float32),
            per_ing_embs[is_:is_+n_ing].astype(np.float32),
            ing_texts_by_rid.get(rid, [])[:n_ing],
            step_texts[ss:se],
            rid, top_k_edges=args.top_k_edges)
        if dag is None:
            print("Could not extract DAG (recipe too short or no ingredients)")
            return
        print(json.dumps(dag, indent=2))
        # Try to load codebook labels for better viz
        lbl_path = VAE_DIR / "codebook_labels.json"
        lbl = json.load(open(lbl_path)) if lbl_path.exists() else {}
        out_fig = BASE / f"data/figures/dag_{rid}.png"
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        visualise_dag(dag, lbl, out_fig)
        return

    # Batch extraction
    n_str = "all" if args.n_recipes is None else f"{args.n_recipes:,}"
    print(f"Extracting DAGs for {n_str} recipes...", flush=True)
    print(f"Batch size: {args.batch} recipes/forward-pass  "
          f"text_weight={args.text_weight}", flush=True)
    n_ok = extract_all(
        model, recipe_ids, step_embs, per_ing_embs,
        step_index, per_ing_idx, step_texts, ing_texts_by_rid,
        Path(args.out), top_k_edges=args.top_k_edges,
        max_recipes=args.n_recipes, batch_size=args.batch,
        text_weight=args.text_weight)

    dag_stats(args.out)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
