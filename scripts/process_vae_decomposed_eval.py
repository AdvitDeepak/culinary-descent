#!/usr/bin/env python3
"""
Decomposed LLM-as-judge evaluation.

Asks three targeted sub-questions separately instead of one overall question:
  Q1 (labels):      Which DAG uses more accurate operation type labels?
  Q2 (ingredients): Which DAG assigns ingredients to the correct steps?
  Q3 (structure):   Which DAG better captures the dependency structure?

Isolates which dimension drives the overall 55.5% win rate.

Usage:
  python3 scripts/process_vae_decomposed_eval.py --n 50 --seed 42
"""
import json, re, sys, os, argparse
from pathlib import Path
from collections import Counter

import numpy as np
import anthropic

BASE = Path(__file__).resolve().parent.parent

DAG_LEARNED  = BASE / "data/dags/learned_dags_grammar.jsonl"
DAG_GRAMMAR  = BASE / "data/dags/constrained_dags_14b.jsonl"
LAYER1       = Path(os.environ.get("RECIPE1M", str(BASE.parent / "layer1.json")))
CODEBOOK     = BASE / "data/models/process_vae_grammar/codebook_analysis.json"

CLEAR_LABELS = {"preheat-oven","bake-timed","knead","whisk","fold",
                "season","simmer","mix-general","prepare-pan","roll-dough","marinate"}
NOISE_LABELS = {"serve","instruction-note"}

JUDGE_MODEL  = "claude-sonnet-4-6"

# ── Three targeted prompts ─────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an expert chef evaluating recipe process representations. "
    "You give concise, accurate judgements. Answer only the specific question asked."
)

SHARED_HEADER = """\
RECIPE: {title}
Original instructions:
{instructions}

---
DAG A ({a_n} steps):
{a_dag}

---
DAG B ({b_n} steps):
{b_dag}

---
"""

Q1_LABELS = SHARED_HEADER + """\
QUESTION (labels only): Considering ONLY the operation type labels used for each step \
(e.g., mix, bake, simmer, knead) — which DAG uses more semantically accurate labels \
that match what the step actually describes? Ignore ingredient assignments and \
dependency edges.

Start with "A" or "B", then one sentence."""

Q2_INGREDIENTS = SHARED_HEADER + """\
QUESTION (ingredients only): Considering ONLY the ingredient assignments — which DAG \
places ingredients at the steps where they are actually first introduced or processed \
in the recipe? Ignore operation type labels and dependency edges.

Start with "A" or "B", then one sentence."""

Q3_STRUCTURE = SHARED_HEADER + """\
QUESTION (structure only): Considering ONLY the dependency structure — which DAG \
better captures the correct sequence of steps and, where applicable, identifies \
operations that can happen in parallel? Ignore specific operation labels and \
ingredient details.

Start with "A" or "B", then one sentence."""

QUESTIONS = [
    ("labels",      "Operation type label accuracy",     Q1_LABELS),
    ("ingredients", "Ingredient-step assignment",         Q2_INGREDIENTS),
    ("structure",   "Dependency structure / sequence",    Q3_STRUCTURE),
]

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(n, seed):
    learned, grammar = {}, {}
    with open(DAG_LEARNED) as f:
        for line in f:
            d = json.loads(line); learned[d["id"]] = d
    with open(DAG_GRAMMAR) as f:
        for line in f:
            d = json.loads(line)
            rid = d.get("id") or d.get("recipe_id")
            if rid in learned: grammar[rid] = d

    raw = {}
    with open(LAYER1) as f:
        for r in json.load(f):
            if r["id"] in grammar: raw[r["id"]] = r

    overlap = [r for r in grammar if r in raw
               and len(grammar[r]["steps"]) >= 3
               and learned[r]["n_steps"] >= 3]
    rng = np.random.default_rng(seed)
    sample = [overlap[i] for i in rng.choice(len(overlap), min(n, len(overlap)), replace=False)]
    return [(rid, learned[rid], grammar[rid], raw[rid]) for rid in sample]


def load_labels():
    with open(CODEBOOK) as f:
        cb = json.load(f)
    return {int(k): v["semantic_label"] for k, v in cb.items()}


# ── DAG formatting ─────────────────────────────────────────────────────────────

def fmt_learned(dag, label_map):
    lines = []
    for i, node in enumerate(dag["nodes"]):
        lbl = label_map.get(node.get("code"), "")
        show = lbl if (lbl in CLEAR_LABELS and lbl not in NOISE_LABELS) else ""
        txt  = node.get("step_text", "").strip()[:120]
        ings = node.get("ingredients", [])
        line = f"Step {i+1}"
        if show: line += f" [{show}]"
        if txt:  line += f": {txt}"
        if ings: line += f"\n  Ingredients: {', '.join(ings[:5])}"
        lines.append(line)
    edges = dag.get("edges", [])
    if edges:
        deps = [f"Step {e['from']+1}→Step {e['to']+1}" for e in edges[:12]]
        lines.append(f"Dependencies: {', '.join(deps)}")
    return "\n".join(lines)


def fmt_grammar(dag):
    lines = []
    for i, s in enumerate(dag["steps"]):
        canon = s.get("canonical", "?")
        ings  = s.get("ingredients", [])
        temp  = f"  Temp: {s['temp_f']}°F" if s.get("temp_f") else ""
        dur   = f"  Duration: {s['duration_min']}min" if s.get("duration_min") else ""
        line  = f"Step {i+1} [{canon}]"
        if ings: line += f"\n  Ingredients: {', '.join(ings[:5])}"
        if temp: line += temp
        if dur:  line += dur
        lines.append(line)
    return "\n".join(lines)


# ── Judge call ─────────────────────────────────────────────────────────────────

def ask_judge(client, prompt_text):
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=120,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt_text}],
    )
    text = resp.content[0].text.strip()
    m = re.match(r"^\s*([AB])\b", text, re.IGNORECASE)
    if m: return m.group(1).upper(), text
    m = re.search(r"\b([AB])\b", text, re.IGNORECASE)
    return (m.group(1).upper() if m else "?"), text


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",  type=str,
                        default=str(BASE / "data/eval/decomposed_eval.json"))
    args = parser.parse_args()

    print(f"Decomposed eval: n={args.n}, seed={args.seed}", flush=True)
    records   = load_data(args.n, args.seed)
    label_map = load_labels()
    client    = anthropic.Anthropic()

    results = []
    # Track per-dimension wins
    dim_wins   = {k: 0 for k,_,_ in QUESTIONS}
    dim_total  = {k: 0 for k,_,_ in QUESTIONS}

    # Balanced swap: first half learned=A, second half learned=B
    half = args.n // 2

    for idx, (rid, l_dag, g_dag, raw) in enumerate(records):
        learned_is_a = idx < half
        a_is = "learned" if learned_is_a else "grammar"

        title = raw.get("title", rid)
        instrs = " ".join(raw.get("instructions", {}).get("instructions", []))[:600]
        l_fmt  = fmt_learned(l_dag, label_map)
        g_fmt  = fmt_grammar(g_dag)

        a_fmt = l_fmt if learned_is_a else g_fmt
        b_fmt = g_fmt if learned_is_a else l_fmt
        a_n   = l_dag["n_steps"] if learned_is_a else len(g_dag["steps"])
        b_n   = len(g_dag["steps"]) if learned_is_a else l_dag["n_steps"]

        rec = {"rid": rid, "title": title, "learned_is_a": learned_is_a,
               "n_learned": l_dag["n_steps"], "n_grammar": len(g_dag["steps"]),
               "dims": {}}

        dim_results = []
        for dim_key, dim_label, tmpl in QUESTIONS:
            prompt = tmpl.format(title=title, instructions=instrs,
                                 a_dag=a_fmt, b_dag=b_fmt,
                                 a_n=a_n, b_n=b_n)
            choice, rationale = ask_judge(client, prompt)
            winner = "learned" if ((choice == "A") == learned_is_a) else "grammar"
            rec["dims"][dim_key] = {"choice": choice, "winner": winner,
                                    "rationale": rationale[:200]}
            dim_wins[dim_key]  += (winner == "learned")
            dim_total[dim_key] += 1
            dim_results.append(f"{dim_key}={winner[0].upper()}")

        results.append(rec)
        print(f"  [{idx+1:3d}/{args.n}]  {title[:40]:<40}  "
              f"{' | '.join(dim_results)}", flush=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"DECOMPOSED EVAL RESULTS  (n={args.n})")
    print(f"{'='*60}")
    for dim_key, dim_label, _ in QUESTIONS:
        wins = dim_wins[dim_key]
        n    = dim_total[dim_key]
        print(f"  {dim_label:<42} {wins}/{n} = {wins/n:.1%}")

    from scipy import stats
    print(f"\n  p-values (binomial, H0=50%):")
    for dim_key, dim_label, _ in QUESTIONS:
        wins = dim_wins[dim_key]
        n    = dim_total[dim_key]
        p = stats.binomtest(wins, n, 0.5, alternative='greater').pvalue
        print(f"  {dim_label:<42} p={p:.4f}")

    out = {"n": args.n, "seed": args.seed,
           "summary": {k: {"wins": dim_wins[k], "n": dim_total[k],
                           "rate": dim_wins[k]/dim_total[k]}
                       for k,_,_ in QUESTIONS},
           "records": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
