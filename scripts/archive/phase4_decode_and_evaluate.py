#!/usr/bin/env python3
"""
Phase 4: Decode RecipeDAGs → reconstructed NL and evaluate reconstruction quality.

Pipeline:
  1. Load train_dags.jsonl (from phase3)
  2. For each DAG: topological sort process nodes → fill templates → concat NL
  3. Compute ROUGE-L between original NL and reconstructed NL
  4. Compute verifier acceptance rate using learned transition matrix

Outputs:
  data/eval/reconstruction_results.json  — ROUGE-L stats + examples
  data/eval/verifier_results.json        — transition-matrix verifier stats
  data/eval/examples.txt                 — 20 side-by-side reconstructions
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

DAGS_DIR   = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/dags")
RECIPE1M   = Path("/home/addeepak/AdvitResearch/cs348k/recipes.json")
OUT        = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/eval")
OUT.mkdir(parents=True, exist_ok=True)

# ── Decoding templates (one per canonical process type) ──────────────────────
# Format: {ingredients} = comma-joined ingredient list, {args} = args string

TEMPLATES = {
    # dry_heat
    "bake":     "Bake {ingredients}{args}.",
    "roast":    "Roast {ingredients}{args} in the oven.",
    "broil":    "Broil {ingredients}{args} under the broiler.",
    "grill":    "Grill {ingredients}{args} over direct heat.",
    "toast":    "Toast {ingredients}{args} until golden.",
    # fat_heat
    "saute":    "Sauté {ingredients} in a pan over medium-high heat{args}.",
    "fry":      "Fry {ingredients}{args} until cooked through.",
    "sear":     "Sear {ingredients} in a hot pan{args} to form a crust.",
    "brown":    "Brown {ingredients} in a pan{args} until golden.",
    # moist_heat
    "boil":     "Bring {ingredients} to a boil and cook{args}.",
    "simmer":   "Simmer {ingredients} over low heat{args}.",
    "steam":    "Steam {ingredients}{args}.",
    "poach":    "Poach {ingredients} in liquid{args}.",
    "braise":   "Braise {ingredients} with liquid over low heat{args}.",
    "blanch":   "Blanch {ingredients} briefly in boiling water{args}, then transfer to ice bath.",
    # prep
    "chop":     "Chop {ingredients}.",
    "dice":     "Dice {ingredients}{args}.",
    "slice":    "Slice {ingredients}{args}.",
    "mince":    "Mince {ingredients} finely.",
    "grate":    "Grate {ingredients}.",
    "peel":     "Peel {ingredients}.",
    "crush":    "Crush {ingredients}.",
    # combine
    "mix":      "Mix {ingredients} together{args}.",
    "stir":     "Stir {ingredients}{args}.",
    "whisk":    "Whisk {ingredients} until smooth{args}.",
    "fold":     "Fold {ingredients} gently to combine.",
    "blend":    "Blend {ingredients} until smooth{args}.",
    "beat":     "Beat {ingredients}{args}.",
    "knead":    "Knead {ingredients}{args} until dough is smooth.",
    "toss":     "Toss {ingredients} to coat.",
    # apply
    "season":   "Season {ingredients} with salt and pepper.",
    "coat":     "Coat {ingredients} evenly.",
    "brush":    "Brush {ingredients} over the surface.",
    "drizzle":  "Drizzle {ingredients} over the top.",
    # rest
    "cool":     "Let {ingredients} cool{args}.",
    "chill":    "Refrigerate {ingredients}{args} until cold.",
    "freeze":   "Freeze {ingredients}{args}.",
    "rest":     "Let {ingredients} rest{args}.",
    "marinate": "Marinate {ingredients}{args}.",
    "reduce":   "Reduce {ingredients} by simmering until thickened.",
    "dissolve": "Dissolve {ingredients} completely.",
    "melt":     "Melt {ingredients} until liquid.",
    "drain":    "Drain {ingredients}.",
}


def format_args(args: dict) -> str:
    """Convert args dict to a human-readable suffix."""
    parts = []
    if "temp_c" in args:
        f = round(args["temp_c"] * 9/5 + 32)
        parts.append(f"at {f}°F ({args['temp_c']}°C)")
    if "duration_min" in args:
        m = args["duration_min"]
        if m >= 60:
            h = int(m // 60)
            rem = int(m % 60)
            parts.append(f"for {h}h {rem}m" if rem else f"for {h} hour{'s' if h>1 else ''}")
        else:
            parts.append(f"for {int(m)} minute{'s' if m != 1 else ''}")
    if "size_note" in args:
        parts.append(f"({args['size_note']})")
    return " " + ", ".join(parts) if parts else ""


def decode_dag(dag: dict, recipe_ing_map: dict) -> str:
    """
    Reconstruct natural language from a DAG via topological sort + templates.

    recipe_ing_map: node_id → ingredient name (for ingredient nodes)
    """
    node_map = {n["id"]: n for n in dag["nodes"]}

    # Get ordered process nodes (already sequential from phase3)
    proc_nodes = sorted(
        [n for n in dag["nodes"] if n["type"] == "process"],
        key=lambda n: n["step_idx"]
    )

    # For each process node, find its input ingredients (via "input" / "input_fallback" edges)
    proc_inputs: dict[str, list[str]] = defaultdict(list)
    for edge in dag["edges"]:
        if edge["label"] in ("input", "input_fallback"):
            src = node_map.get(edge["src"])
            if src and src["type"] == "ingredient":
                proc_inputs[edge["dst"]].append(src["name"])

    sentences = []
    for proc in proc_nodes:
        canon    = proc["canonical"]
        template = TEMPLATES.get(canon, "Process {ingredients}{args}.")
        ings     = proc_inputs.get(proc["id"], [])
        # Use up to 3 ingredients in the reconstruction to keep it brief
        if ings:
            ing_str = ", ".join(ings[:3]) + (" and other ingredients" if len(ings) > 3 else "")
        else:
            ing_str = "the ingredients"
        arg_str  = format_args(proc.get("args", {}))
        sentence = template.format(ingredients=ing_str, args=arg_str)
        sentences.append(sentence)

    return " ".join(sentences)


def original_nl(recipe: dict) -> str:
    """Extract original NL from recipe instructions."""
    steps = [s["text"].strip() for s in recipe.get("instructions", []) if s["text"].strip()]
    return " ".join(steps)


# ── ROUGE-L implementation ────────────────────────────────────────────────────

def _lcs(a: list, b: list) -> int:
    """Longest common subsequence length."""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    # Use O(min(m,n)) space
    if m < n:
        a, b, m, n = b, a, n, m
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev = curr
    return prev[n]


def rouge_l(hyp: str, ref: str) -> float:
    """ROUGE-L F1 score (unigram tokens, lowercased)."""
    h = re.findall(r'\w+', hyp.lower())
    r = re.findall(r'\w+', ref.lower())
    if not h or not r:
        return 0.0
    lcs = _lcs(h, r)
    prec = lcs / len(h)
    rec  = lcs / len(r)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def token_f1(hyp: str, ref: str) -> float:
    """Unigram token F1 (bag-of-words overlap)."""
    h = set(re.findall(r'\w+', hyp.lower()))
    r = set(re.findall(r'\w+', ref.lower()))
    if not h or not r:
        return 0.0
    inter = len(h & r)
    prec  = inter / len(h)
    rec   = inter / len(r)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ── Data-driven verifier ──────────────────────────────────────────────────────

def load_transition_matrix():
    with open(DAGS_DIR / "transition_matrix.json") as f:
        data = json.load(f)
    return data["transition_probs"]   # {A: {B: prob}}


FORBIDDEN_THRESHOLD = 0.001   # pairs with P < threshold in training → flag as suspicious

def verify_dag(dag: dict, trans: dict) -> dict:
    """
    Check a DAG against the learned transition matrix.

    Returns:
      valid:       True if no forbidden transitions
      n_edges:     total seq edges checked
      n_flagged:   transitions with P < threshold
      flagged:     list of (A, B, P) tuples
    """
    node_map = {n["id"]: n for n in dag["nodes"]}
    flagged  = []
    n_edges  = 0

    for edge in dag["edges"]:
        if edge["label"] != "seq":
            continue
        src = node_map.get(edge["src"])
        dst = node_map.get(edge["dst"])
        if not (src and dst):
            continue
        A = src.get("canonical")
        B = dst.get("canonical")
        if not (A and B):
            continue
        n_edges += 1
        prob = trans.get(A, {}).get(B, 0.0)
        if prob < FORBIDDEN_THRESHOLD:
            flagged.append((A, B, prob))

    return {
        "valid":     len(flagged) == 0,
        "n_edges":   n_edges,
        "n_flagged": len(flagged),
        "flagged":   flagged,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load DAGs
    print("Loading DAGs...", flush=True)
    dags = []
    with open(DAGS_DIR / "train_dags.jsonl") as f:
        for line in f:
            dags.append(json.loads(line))
    print(f"  {len(dags):,} DAGs loaded")

    # Load original recipes indexed by id
    print("Loading original recipes...", flush=True)
    with open(RECIPE1M) as f:
        all_recipes = json.load(f)
    recipe_by_id = {r["id"]: r for r in all_recipes if r.get("partition") == "train"}
    print(f"  {len(recipe_by_id):,} train recipes")

    # Load transition matrix
    trans = load_transition_matrix()
    print(f"  Transition matrix loaded ({len(trans)} source types)")

    # ── Evaluate on ALL parsed DAGs ────────────────────────────────────────
    print("\nEvaluating reconstruction + verification...", flush=True)
    rouge_scores = []
    f1_scores    = []
    verify_results = []
    examples     = []

    for dag in tqdm(dags, desc="eval"):
        rid    = dag["id"]
        recipe = recipe_by_id.get(rid)
        if recipe is None:
            continue

        ref  = original_nl(recipe)
        hyp  = decode_dag(dag, {})

        rl = rouge_l(hyp, ref)
        f1 = token_f1(hyp, ref)
        rouge_scores.append(rl)
        f1_scores.append(f1)

        vr = verify_dag(dag, trans)
        verify_results.append(vr)

        if len(examples) < 20:
            examples.append({
                "id":        rid,
                "title":     recipe.get("title", ""),
                "original":  ref[:600],
                "decoded":   hyp[:600],
                "rouge_l":   round(rl, 4),
                "token_f1":  round(f1, 4),
                "verify":    vr,
            })

    rouge_arr = np.array(rouge_scores)
    f1_arr    = np.array(f1_scores)
    valid_arr = np.array([v["valid"] for v in verify_results])

    print(f"\n{'='*60}")
    print("RECONSTRUCTION QUALITY")
    print(f"{'='*60}")
    print(f"  N evaluated:         {len(rouge_arr):,}")
    print(f"  ROUGE-L  mean:       {rouge_arr.mean():.4f}")
    print(f"  ROUGE-L  median:     {np.median(rouge_arr):.4f}")
    print(f"  ROUGE-L  p25/p75:    {np.percentile(rouge_arr,25):.4f} / {np.percentile(rouge_arr,75):.4f}")
    print(f"  Token-F1 mean:       {f1_arr.mean():.4f}")
    print(f"  Token-F1 median:     {np.median(f1_arr):.4f}")
    print(f"  % ROUGE-L > 0.10:    {(rouge_arr > 0.10).mean():.1%}")
    print(f"  % ROUGE-L > 0.20:    {(rouge_arr > 0.20).mean():.1%}")
    print(f"  % ROUGE-L > 0.30:    {(rouge_arr > 0.30).mean():.1%}")

    print(f"\n{'='*60}")
    print("DATA-DRIVEN VERIFIER")
    print(f"{'='*60}")
    print(f"  % DAGs passing verifier: {valid_arr.mean():.1%}")
    print(f"  Threshold P < {FORBIDDEN_THRESHOLD}")
    # Most common flagged transitions
    from collections import Counter
    flag_counter = Counter()
    for vr in verify_results:
        for A, B, p in vr["flagged"]:
            flag_counter[(A, B)] += 1
    print(f"\n  Most-flagged transitions (train set — these are genuinely rare):")
    for (A, B), cnt in flag_counter.most_common(10):
        print(f"    {A} → {B}: flagged in {cnt} DAGs")

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        "n_evaluated":        len(rouge_arr),
        "rouge_l_mean":       float(rouge_arr.mean()),
        "rouge_l_median":     float(np.median(rouge_arr)),
        "rouge_l_p25":        float(np.percentile(rouge_arr, 25)),
        "rouge_l_p75":        float(np.percentile(rouge_arr, 75)),
        "rouge_l_gt_0_10":    float((rouge_arr > 0.10).mean()),
        "rouge_l_gt_0_20":    float((rouge_arr > 0.20).mean()),
        "rouge_l_gt_0_30":    float((rouge_arr > 0.30).mean()),
        "token_f1_mean":      float(f1_arr.mean()),
        "token_f1_median":    float(np.median(f1_arr)),
        "verifier_pass_rate": float(valid_arr.mean()),
        "verifier_threshold": FORBIDDEN_THRESHOLD,
        "flagged_transitions": {f"{a}→{b}": c for (a,b), c in flag_counter.most_common()},
        "examples":           examples,
    }
    with open(OUT / "reconstruction_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {OUT}/reconstruction_results.json")

    # ── Write human-readable examples ─────────────────────────────────────
    with open(OUT / "examples.txt", "w") as f:
        for i, ex in enumerate(examples):
            f.write(f"{'='*70}\n")
            f.write(f"Example {i+1}: {ex['title']}\n")
            f.write(f"ROUGE-L={ex['rouge_l']:.3f}  Token-F1={ex['token_f1']:.3f}  "
                    f"Valid={ex['verify']['valid']}\n")
            f.write(f"\nORIGINAL:\n{ex['original']}\n")
            f.write(f"\nDECODED:\n{ex['decoded']}\n\n")
    print(f"  Saved → {OUT}/examples.txt")

    # ── Quick distribution check ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ROUGE-L DISTRIBUTION")
    for lb, ub in [(0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,1.0)]:
        pct = ((rouge_arr >= lb) & (rouge_arr < ub)).mean()
        bar = "█" * int(pct * 40)
        print(f"  [{lb:.1f},{ub:.1f}): {pct:.1%}  {bar}")


if __name__ == "__main__":
    main()
