#!/usr/bin/env python3
"""
Phase 13: Close the adaptation loop — DAG-guided vegan adaptation vs direct LLM.

Pipeline under test:
  NL recipe
    → [constrained encoder]   DAG with per-step ingredient attribution
    → [edit step]             swap animal ingredients per step
    → [LLM decoder]           adapted NL recipe

Compared against:
  NL recipe → [direct LLM prompt] → adapted NL recipe  (no DAG)

Evaluation on 50 non-vegan recipes:
  1. Constraint satisfaction: % of animal ingredients eliminated in output
  2. Step coverage: % of adaptation steps correctly reflected in output
  3. Fluency: SentBERT(original, adapted) as proxy for coherence

The DAG approach should be more reliable at (1) because attribution is explicit —
the editor knows exactly which step uses which ingredient. The direct LLM prompt
may miss an ingredient used in one of several steps, or incorrectly alter steps
that don't involve animal products.

Outputs:
  data/eval/phase13_adaptation_loop.json
  data/eval/phase13_examples.txt
"""

import json
import random
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

DAGS_DIR = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/dags")
RECIPE1M = Path("/home/addeepak/AdvitResearch/cs348k/recipes.json")
OUT      = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/eval")
IN_DAGS  = DAGS_DIR / "constrained_dags.jsonl"

EVAL_N = 50
SEED   = 42
random.seed(SEED)

# ── Vegan knowledge (same as phase11) ─────────────────────────────────────────

ANIMAL_INGREDIENTS = {
    "chicken","beef","pork","lamb","turkey","bacon","ham","sausage",
    "ground beef","ground pork","ground turkey","ground chicken",
    "steak","brisket","ribs","lard","duck","veal","venison",
    "fish","salmon","tuna","shrimp","crab","lobster","clam","oyster",
    "scallop","anchovy","sardine","cod","tilapia","halibut",
    "butter","milk","cream","cheese","parmesan","cheddar","mozzarella",
    "heavy cream","sour cream","cream cheese","whipping cream","half and half",
    "evaporated milk","condensed milk","buttermilk","yogurt",
    "egg","eggs","egg yolk","egg white",
    "gelatin","honey","lard","ghee",
}

VEGAN_SUBS = {
    "butter": "vegan butter",
    "milk": "oat milk",
    "cream": "coconut cream",
    "heavy cream": "coconut cream",
    "egg": "flax egg",
    "eggs": "flax eggs",
    "cheese": "vegan cheese",
    "parmesan": "nutritional yeast",
    "chicken": "tofu",
    "beef": "lentils",
    "pork": "tempeh",
    "bacon": "smoked tempeh",
    "honey": "maple syrup",
    "lard": "coconut oil",
    "yogurt": "coconut yogurt",
    "sour cream": "cashew cream",
    "cream cheese": "vegan cream cheese",
}


def find_animal(ing: str):
    ing_lower = ing.lower()
    for animal in ANIMAL_INGREDIENTS:
        if animal in ing_lower:
            return animal
    return None


def apply_vegan_subs(dag: dict) -> tuple[dict, list]:
    """Return (edited_dag, substitution_log). Edits in place on a copy."""
    import copy
    edited = copy.deepcopy(dag)
    log = []
    for step in edited["steps"]:
        new_ings = []
        for ing in step.get("ingredients", []):
            animal = find_animal(ing)
            if animal:
                sub = VEGAN_SUBS.get(animal, f"plant-based {animal}")
                log.append({"step_verb": step["canonical"], "removed": ing, "added": sub})
                new_ings.append(sub)
            else:
                new_ings.append(ing)
        step["ingredients"] = new_ings
    return edited, log


# ── DAG → decoder prompt ──────────────────────────────────────────────────────

def dag_to_prompt(dag: dict, title: str = "") -> str:
    steps  = dag["steps"]
    lines  = []
    all_ings = []
    for i, s in enumerate(steps, 1):
        verb = s["canonical"]
        ings = s.get("ingredients") or []
        temp = s.get("temp_f")
        dur  = s.get("duration_min")
        all_ings.extend(ings)
        line = f"{i}. {verb}"
        if ings:
            line += f" [{', '.join(ings[:4])}]"
        if temp is not None:
            line += f" at {int(temp)}°F"
        if dur is not None:
            if dur >= 60:
                h = int(dur // 60); rem = int(dur % 60)
                line += f" for {h}h {rem}m" if rem else f" for {h}hr"
            else:
                line += f" for {int(dur)} min"
        lines.append(line)

    # deduplicate ingredients
    seen = set(); unique_ings = []
    for ing in all_ings:
        if ing.lower() not in seen:
            seen.add(ing.lower()); unique_ings.append(ing)

    step_str = "\n".join(lines)
    ing_str  = ", ".join(unique_ings[:12])
    title_str = f" for '{title}'" if title else ""

    return (
        f"Convert this vegan recipe outline{title_str} into plain cooking instructions. "
        f"No markdown, no headers, no preamble. Start immediately with step 1.\n\n"
        f"Ingredients: {ing_str}\n\n"
        f"Steps:\n{step_str}\n\n"
        f"Instructions ({len(steps)} steps, plain prose, no lists or headers):\n1."
    )


def direct_vegan_prompt(original_nl: str) -> str:
    return (
        f"Rewrite this recipe to be fully vegan. Replace all animal products "
        f"(meat, dairy, eggs, honey) with plant-based alternatives. "
        f"Keep the same cooking method and structure. Plain prose, no markdown.\n\n"
        f"Original recipe:\n{original_nl}\n\n"
        f"Vegan version:\n"
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def constraint_satisfaction(text: str) -> tuple[float, list]:
    """Return (fraction of animal ingredients absent from text, list found)."""
    found = []
    for animal in sorted(ANIMAL_INGREDIENTS):
        if re.search(r'\b' + re.escape(animal) + r'\b', text, re.IGNORECASE):
            found.append(animal)
    # satisfaction = fraction absent (higher is better)
    total_checked = len(ANIMAL_INGREDIENTS)
    satisfaction = (total_checked - len(found)) / total_checked
    return satisfaction, found


def sentbert_sim(texts_a: list, texts_b: list, model) -> list:
    ea = model.encode(texts_a, convert_to_numpy=True, normalize_embeddings=True)
    eb = model.encode(texts_b, convert_to_numpy=True, normalize_embeddings=True)
    return (ea * eb).sum(axis=1).tolist()


def get_original_nl(dag: dict, recipe_by_id: dict) -> str:
    recipe = recipe_by_id.get(dag.get("id"))
    if not recipe:
        return ""
    steps = [s["text"].strip() for s in recipe.get("instructions", []) if s["text"].strip()]
    return " ".join(steps[:10])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading DAGs + recipes...", flush=True)
    dags = []
    with open(IN_DAGS) as f:
        for line in f:
            dags.append(json.loads(line))
    print(f"  {len(dags):,} DAGs loaded")

    with open(RECIPE1M) as f:
        all_recipes = json.load(f)
    recipe_by_id = {r["id"]: r for r in all_recipes}
    print(f"  {len(recipe_by_id):,} recipes loaded")

    # Filter to non-vegan DAGs with ≥1 animal ingredient
    non_vegan = []
    for dag in dags:
        for step in dag["steps"]:
            if any(find_animal(ing) for ing in step.get("ingredients", [])):
                non_vegan.append(dag)
                break

    sample = random.sample(non_vegan, min(EVAL_N, len(non_vegan)))
    print(f"\n  Sampled {len(sample)} non-vegan recipes for adaptation eval\n")

    print("Loading Qwen2.5-3B via vLLM...", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", max_model_len=2048,
               gpu_memory_utilization=0.85, dtype="bfloat16", enforce_eager=True)
    tok = llm.get_tokenizer()
    decode_params  = SamplingParams(temperature=0.0, max_tokens=400)
    direct_params  = SamplingParams(temperature=0.0, max_tokens=500)

    print("Loading SentBERT...", flush=True)
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ── Apply vegan edits ──────────────────────────────────────────────────────
    edited_dags  = []
    sub_logs     = []
    for dag in sample:
        edited, log = apply_vegan_subs(dag)
        edited_dags.append(edited)
        sub_logs.append(log)

    # ── DAG-guided decode ──────────────────────────────────────────────────────
    print("\nDecoding edited DAGs → NL (DAG-guided approach)...", flush=True)
    dag_prompts = []
    for dag in edited_dags:
        title = dag.get("title", "")
        raw_prompt = dag_to_prompt(dag, title)
        msgs = [
            {"role": "system", "content": "You write clear, concise cooking instructions from structured recipe outlines."},
            {"role": "user",   "content": raw_prompt},
        ]
        dag_prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "1.")

    dag_outputs = llm.generate(dag_prompts, decode_params)
    dag_results = ["1." + o.outputs[0].text.strip() for o in dag_outputs]

    # ── Direct LLM baseline ───────────────────────────────────────────────────
    print("Running direct LLM baseline (no DAG)...", flush=True)
    direct_prompts = []
    original_nls   = []
    for dag in sample:
        nl = get_original_nl(dag, recipe_by_id)
        original_nls.append(nl)
        msgs = [
            {"role": "system", "content": "You are a helpful cooking assistant."},
            {"role": "user",   "content": direct_vegan_prompt(nl)},
        ]
        direct_prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    direct_outputs = llm.generate(direct_prompts, direct_params)
    direct_results = [o.outputs[0].text.strip() for o in direct_outputs]

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating...", flush=True)

    dag_sat  = []; dag_found  = []
    dir_sat  = []; dir_found  = []
    for dr, dg in zip(dag_results, direct_results):
        s, f = constraint_satisfaction(dr); dag_sat.append(s); dag_found.append(f)
        s, f = constraint_satisfaction(dg); dir_sat.append(s); dir_found.append(f)

    # SentBERT: adapted vs original (coherence proxy)
    dag_sim  = sentbert_sim(dag_results,    original_nls, sbert)
    dir_sim  = sentbert_sim(direct_results, original_nls, sbert)

    # Step coverage: did the output mention the substitution ingredients?
    def coverage(result, log):
        if not log: return 1.0
        hits = sum(1 for e in log if e["added"].lower() in result.lower())
        return hits / len(log)

    dag_cov = [coverage(r, l) for r, l in zip(dag_results, sub_logs)]
    dir_cov = [coverage(r, l) for r, l in zip(direct_results, sub_logs)]

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"VEGAN ADAPTATION: DAG-GUIDED vs DIRECT LLM PROMPT  (n={len(sample)})")
    print(f"{'='*65}")
    print(f"  {'Metric':<40}  {'DAG':>8}  {'Direct':>8}")
    print(f"  {'-'*58}")
    print(f"  {'Constraint satisfaction (animal-free %)':<40}  {np.mean(dag_sat):>8.1%}  {np.mean(dir_sat):>8.1%}")
    print(f"  {'Sub ingredient coverage':<40}  {np.mean(dag_cov):>8.1%}  {np.mean(dir_cov):>8.1%}")
    print(f"  {'SentBERT sim to original':<40}  {np.mean(dag_sim):>8.3f}  {np.mean(dir_sim):>8.3f}")

    # How often does DAG beat direct on constraint satisfaction?
    dag_wins = sum(1 for d, di in zip(dag_sat, dir_sat) if d > di)
    tie      = sum(1 for d, di in zip(dag_sat, dir_sat) if d == di)
    dir_wins = sum(1 for d, di in zip(dag_sat, dir_sat) if d < di)
    print(f"\n  Per-recipe: DAG better={dag_wins}, tie={tie}, direct better={dir_wins}")

    # Animal ingredients still found in output (failures)
    dag_leaks  = [item for f in dag_found  for item in f]
    dir_leaks  = [item for f in dir_found  for item in f]
    print(f"\n  Animal ingredients leaked into output:")
    print(f"    DAG-guided: {len(dag_leaks)} total ({set(dag_leaks[:10])})")
    print(f"    Direct LLM: {len(dir_leaks)} total ({set(dir_leaks[:10])})")

    # ── Save examples ─────────────────────────────────────────────────────────
    with open(OUT / "phase13_examples.txt", "w") as f:
        for i, (dag, orig, dr, dg, log) in enumerate(
                zip(sample, original_nls, dag_results, direct_results, sub_logs)):
            f.write(f"{'='*70}\n")
            f.write(f"Recipe {i+1}: {dag.get('title','')}\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"[ORIGINAL]\n{orig[:400]}\n\n")
            f.write(f"[EDIT PLAN] ({len(log)} substitutions)\n")
            for e in log:
                f.write(f"  {e['step_verb']}: '{e['removed']}' → '{e['added']}'\n")
            f.write(f"\n[DAG-GUIDED OUTPUT]\n{dr[:500]}\n\n")
            f.write(f"[DIRECT LLM OUTPUT]\n{dg[:500]}\n\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    results = {
        "n": len(sample),
        "dag_guided": {
            "constraint_satisfaction": float(np.mean(dag_sat)),
            "sub_coverage":            float(np.mean(dag_cov)),
            "sentbert_vs_original":    float(np.mean(dag_sim)),
            "animal_leaks":            len(dag_leaks),
            "leak_types":              list(set(dag_leaks)),
        },
        "direct_llm": {
            "constraint_satisfaction": float(np.mean(dir_sat)),
            "sub_coverage":            float(np.mean(dir_cov)),
            "sentbert_vs_original":    float(np.mean(dir_sim)),
            "animal_leaks":            len(dir_leaks),
            "leak_types":              list(set(dir_leaks)),
        },
        "per_recipe_wins": {
            "dag_better": dag_wins, "tie": tie, "direct_better": dir_wins,
        },
    }
    with open(OUT / "phase13_adaptation_loop.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {OUT}/phase13_adaptation_loop.json")
    print(f"  Examples → {OUT}/phase13_examples.txt")


if __name__ == "__main__":
    main()
