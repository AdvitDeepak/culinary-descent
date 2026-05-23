#!/usr/bin/env python3
"""
Phase 11: Downstream application demos on constrained DAGs.

Shows WHY the DSL is useful beyond compression quality metrics.
Runs entirely on constrained_dags.jsonl — no GPU required.

Applications:
  1. Structural Query Engine  — process-level predicates on 35K recipes
  2. Recipe Adaptation Planner — surgical ingredient substitution using
                                  per-step ingredient assignment

Outputs: data/eval/phase11_applications.json
         Printed results table
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DAGS_DIR = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/dags")
OUT      = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/eval")
IN_DAGS  = DAGS_DIR / "constrained_dags.jsonl"

# ── Vegan substitution knowledge ─────────────────────────────────────────────

ANIMAL_INGREDIENTS = {
    # meat
    "chicken", "beef", "pork", "lamb", "turkey", "bacon", "ham", "sausage",
    "ground beef", "ground pork", "ground turkey", "ground chicken",
    "steak", "brisket", "ribs", "lard", "duck", "veal", "venison",
    # seafood
    "fish", "salmon", "tuna", "shrimp", "crab", "lobster", "clam", "oyster",
    "scallop", "anchovy", "sardine", "cod", "tilapia", "halibut",
    # dairy
    "butter", "milk", "cream", "cheese", "parmesan", "cheddar", "mozzarella",
    "heavy cream", "sour cream", "cream cheese", "whipping cream", "half and half",
    "evaporated milk", "condensed milk", "buttermilk", "yogurt",
    # eggs
    "egg", "eggs", "egg yolk", "egg white",
    # gelatin / misc
    "gelatin", "honey", "lard", "ghee",
}

VEGAN_SUBS = {
    "butter": "vegan butter or coconut oil",
    "milk": "oat milk or almond milk",
    "cream": "coconut cream",
    "heavy cream": "coconut cream",
    "egg": "flax egg (1 tbsp flaxseed + 3 tbsp water)",
    "eggs": "flax eggs",
    "cheese": "nutritional yeast or vegan cheese",
    "parmesan": "nutritional yeast",
    "chicken": "tofu or chickpeas",
    "beef": "lentils or Beyond Beef",
    "pork": "tempeh",
    "bacon": "coconut bacon or smoked tempeh",
    "honey": "maple syrup or agave",
    "lard": "coconut oil",
    "yogurt": "coconut yogurt",
    "sour cream": "cashew cream",
    "cream cheese": "vegan cream cheese",
}


def load_dags():
    dags = []
    with open(IN_DAGS) as f:
        for line in f:
            dags.append(json.loads(line))
    return dags


# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION 1: Structural Query Engine
# ══════════════════════════════════════════════════════════════════════════════

def sequence(dag):
    return [s["canonical"] for s in dag["steps"]]


def query_precedes(dags, verb_a, verb_b, max_gap=3):
    """Find recipes where verb_a appears before verb_b (within max_gap steps)."""
    hits = []
    for dag in dags:
        seq = sequence(dag)
        for i, v in enumerate(seq):
            if v == verb_a:
                window = seq[i+1 : i+1+max_gap]
                if verb_b in window:
                    hits.append(dag)
                    break
    return hits


def query_contains_all(dags, verbs):
    """Find recipes containing ALL of the given verbs."""
    verb_set = set(verbs)
    return [d for d in dags if verb_set.issubset(set(sequence(d)))]


def query_no_heat(dags):
    """Find recipes that use ZERO heat application steps."""
    heat = {"bake","roast","broil","grill","toast","saute","fry","sear","brown",
            "boil","simmer","steam","poach","braise","blanch"}
    return [d for d in dags if not any(v in heat for v in sequence(d))]


def query_max_steps(dags, n):
    return [d for d in dags if len(dag["steps"]) <= n for dag in [d]]


def query_ingredient_in_step(dags, ingredient_keyword, step_verb):
    """Find recipes where step_verb acts on an ingredient matching keyword."""
    hits = []
    for dag in dags:
        for step in dag["steps"]:
            if step["canonical"] == step_verb:
                if any(ingredient_keyword.lower() in ing.lower()
                       for ing in step.get("ingredients", [])):
                    hits.append(dag)
                    break
    return hits


# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION 2: Recipe Adaptation Planner
# ══════════════════════════════════════════════════════════════════════════════

def adaptation_plan(dag, constraint="vegan"):
    """
    Return a surgical edit plan to make the recipe satisfy a dietary constraint.
    Per-step ingredient assignment makes this precise: we know exactly which step
    uses which ingredient.
    """
    steps = dag["steps"]
    edits = []
    affected_steps = 0

    for i, step in enumerate(steps):
        step_edits = []
        for ing in step.get("ingredients", []):
            ing_lower = ing.lower()
            # check against animal ingredient list
            matched = None
            for animal in ANIMAL_INGREDIENTS:
                if animal in ing_lower:
                    matched = animal
                    break
            if matched:
                sub = VEGAN_SUBS.get(matched, f"plant-based {matched} substitute")
                step_edits.append({
                    "remove": ing,
                    "add": sub,
                    "reason": f"'{matched}' is not vegan",
                })
        if step_edits:
            affected_steps += 1
            edits.append({
                "step_index": i,
                "verb": step["canonical"],
                "substitutions": step_edits,
            })

    already_compliant = len(edits) == 0
    return {
        "title": dag.get("title", ""),
        "n_steps": len(steps),
        "affected_steps": affected_steps,
        "already_compliant": already_compliant,
        "edits": edits,
    }


def main():
    print("Loading DAGs...", flush=True)
    dags = load_dags()
    print(f"  {len(dags):,} constrained DAGs loaded\n")

    results = {}

    # ──────────────────────────────────────────────────────────────────────────
    # APPLICATION 1: Structural Queries
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 65)
    print("APPLICATION 1: STRUCTURAL QUERY ENGINE")
    print("=" * 65)
    print("These queries are structural predicates on process sequences.")
    print("They are impossible to express accurately with keyword search.\n")

    queries = [
        ("marinate → grill",   query_precedes(dags, "marinate", "grill"),
         "Recipes where marinating precedes grilling"),
        ("marinate → bake",    query_precedes(dags, "marinate", "bake"),
         "Recipes where marinating precedes baking"),
        ("knead → bake",       query_precedes(dags, "knead", "bake"),
         "Bread recipes: knead before bake"),
        ("boil → drain → season", query_contains_all(dags, ["boil","drain","season"]),
         "Pasta-style: boil + drain + season"),
        ("reduce after saute", query_precedes(dags, "saute", "reduce"),
         "Pan-sauce technique: saute then reduce"),
        ("no-heat recipes",    query_no_heat(dags),
         "Recipes requiring zero heat (salads, raw, no-cook)"),
        ("chill precedes serve",query_precedes(dags, "chill", "season"),
         "Cold-prep recipes: chill before final seasoning"),
    ]

    query_results = []
    for name, hits, desc in queries:
        pct = len(hits) / len(dags) * 100
        sample = hits[:3]
        print(f"  Query: {desc}")
        print(f"  Matches: {len(hits):,} / {len(dags):,} ({pct:.1f}%)")
        print(f"  Examples: {', '.join(d['title'] for d in sample[:3])}")
        print()
        query_results.append({
            "query": name,
            "description": desc,
            "n_matches": len(hits),
            "pct": round(pct, 2),
            "examples": [{"title": d["title"], "seq": sequence(d)} for d in sample],
        })

    # Contrasting keyword query (simulate what text search would do)
    # "marinate" appears in title or step text — but that doesn't tell us ORDER
    marinate_grill = query_precedes(dags, "marinate", "grill")
    grill_marinate = query_precedes(dags, "grill", "marinate")
    print(f"  ORDER MATTERS: 'marinate→grill' = {len(marinate_grill):,} recipes, "
          f"'grill→marinate' = {len(grill_marinate):,} recipes")
    print(f"  (keyword 'marinate+grill' cannot distinguish these)\n")

    results["structural_queries"] = query_results
    results["order_matters"] = {
        "marinate_then_grill": len(marinate_grill),
        "grill_then_marinate": len(grill_marinate),
        "note": "keyword search cannot distinguish order; DAG sequence can",
    }

    # ──────────────────────────────────────────────────────────────────────────
    # APPLICATION 2: Recipe Adaptation Planner
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 65)
    print("APPLICATION 2: RECIPE ADAPTATION PLANNER (vegan)")
    print("=" * 65)
    print("Per-step ingredient assignment (97% accuracy) enables surgical")
    print("dietary adaptation: we know WHICH STEP uses WHICH ingredient.\n")

    # Corpus-wide adaptation stats
    plans = [adaptation_plan(d) for d in dags]
    already_vegan = sum(1 for p in plans if p["already_compliant"])
    one_step_fix  = sum(1 for p in plans if not p["already_compliant"] and p["affected_steps"] == 1)
    easy_fix      = sum(1 for p in plans if not p["already_compliant"] and p["affected_steps"] <= 2)

    print(f"  Corpus vegan adaptation analysis ({len(dags):,} recipes):")
    print(f"    Already vegan:          {already_vegan:,}  ({already_vegan/len(dags):.1%})")
    print(f"    1 step needs editing:   {one_step_fix:,}  ({one_step_fix/len(dags):.1%})")
    print(f"    ≤2 steps need editing:  {easy_fix:,}  ({easy_fix/len(dags):.1%})")
    print()

    # Show 3 detailed adaptation examples
    # Pick: one already vegan, one easy fix, one complex fix
    vegan_ex  = next(d for d, p in zip(dags, plans) if p["already_compliant"] and len(d["steps"]) >= 3)
    easy_ex   = next(d for d, p in zip(dags, plans) if not p["already_compliant"] and p["affected_steps"] == 1)
    complex_ex= next(d for d, p in zip(dags, plans) if p["affected_steps"] >= 3)

    examples = []
    for dag, label in [(vegan_ex, "Already vegan"), (easy_ex, "1-step fix"), (complex_ex, "Multi-step fix")]:
        plan = adaptation_plan(dag)
        print(f"  [{label}] '{dag['title']}'")
        print(f"  Process sequence: {' → '.join(sequence(dag))}")
        if plan["already_compliant"]:
            print(f"  ✓ No changes needed")
        else:
            for edit in plan["edits"]:
                print(f"  Step {edit['step_index']+1} ({edit['verb']}):")
                for sub in edit["substitutions"]:
                    print(f"    Replace '{sub['remove']}' → {sub['add']}")
        print()
        examples.append({"label": label, "plan": plan, "sequence": sequence(dag)})

    results["adaptation"] = {
        "corpus_stats": {
            "n_dags": len(dags),
            "already_vegan": already_vegan,
            "one_step_fix": one_step_fix,
            "easy_fix_le2_steps": easy_fix,
        },
        "examples": examples,
    }

    # ──────────────────────────────────────────────────────────────────────────
    # APPLICATION 3 (bonus): Process complexity segmentation
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 65)
    print("APPLICATION 3: COMPLEXITY-BASED RECIPE SEGMENTATION")
    print("=" * 65)

    heat_verbs  = {"bake","roast","broil","grill","toast","saute","fry","sear",
                   "brown","boil","simmer","steam","poach","braise","blanch"}
    prep_verbs  = {"chop","dice","slice","mince","grate","peel","crush"}
    finish_verbs= {"season","coat","brush","drizzle","reduce","dissolve","melt","drain"}

    def complexity(dag):
        seq = sequence(dag)
        n = len(seq)
        unique = len(set(seq))
        has_heat = any(v in heat_verbs for v in seq)
        has_prep = any(v in prep_verbs for v in seq)
        # rough complexity score
        return n + unique * 0.5 + (2 if has_heat else 0) + (1 if has_prep else 0)

    scores = [(complexity(d), d) for d in dags]
    scores.sort(key=lambda x: x[0])

    easy  = [d for s, d in scores if s <= 4]
    medium= [d for s, d in scores if 4 < s <= 8]
    hard  = [d for s, d in scores if s > 8]

    print(f"  Easy   (score ≤4):   {len(easy):,}  ({len(easy)/len(dags):.1%})")
    print(f"  Medium (4 < s ≤ 8): {len(medium):,}  ({len(medium)/len(dags):.1%})")
    print(f"  Hard   (score > 8): {len(hard):,}  ({len(hard)/len(dags):.1%})")
    print(f"\n  Simplest recipes (≤3 steps, no heat):")
    no_heat_simple = [d for d in easy if len(d["steps"]) <= 2 and not any(v in heat_verbs for v in sequence(d))]
    for d in no_heat_simple[:5]:
        print(f"    '{d['title']}': {' → '.join(sequence(d))}")
    print(f"\n  Most complex recipes (score > 12):")
    very_hard = [d for s, d in scores if s > 12]
    for d in very_hard[:5]:
        print(f"    '{d['title']}': {' → '.join(sequence(d))}")

    results["complexity_segmentation"] = {
        "easy": len(easy), "medium": len(medium), "hard": len(hard),
        "simplest_examples": [{"title": d["title"], "seq": sequence(d)} for d in no_heat_simple[:5]],
        "hardest_examples": [{"title": d["title"], "seq": sequence(d)} for d in very_hard[:5]],
    }

    print(f"\n{'='*65}")
    print("WHY THE DAG ENABLES THESE AND RAW TEXT DOESN'T")
    print(f"{'='*65}")
    print("""
  Query 1 (structural): "marinate → grill" requires knowing process ORDER.
    Raw text: grep for both words → cannot determine sequence.
    DAG:      sequence predicate on canonical step list → exact.

  Query 2 (adaptation): "which step uses butter so I can replace it?"
    Raw text: 'butter' appears in "Add butter, milk..." → ambiguous which
              cooking action it belongs to; NLP needed to resolve.
    DAG:      step.ingredients is already assigned per-step at 97% accuracy
              → direct lookup, no NLP needed.

  Query 3 (complexity): "find a weeknight-friendly version of beef stew"
    Raw text: no notion of structural complexity; word count ≠ cooking effort.
    DAG:      step count + process type diversity + heat presence → computable.
    """)

    with open(OUT / "phase11_applications.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {OUT}/phase11_applications.json")


if __name__ == "__main__":
    main()
