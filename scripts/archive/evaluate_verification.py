#!/usr/bin/env python3
"""
Verifier Recall Evaluation
---------------------------
Metric 2 of 2 for our north-star evaluation:

  "Given recipes with KNOWN violations, what % does the verifier catch?"

This script builds a SYNTHETIC test set with one recipe per constraint predicate
— each recipe designed to trigger exactly one specific violation — plus a set of
valid "control" recipes that should produce no false positives.

Why synthetic?
~~~~~~~~~~~~~~
We don't have a labeled dataset of Recipe1M recipes annotated with specific
violations. Building synthetically lets us:
  1. Guarantee ground-truth labels (we know exactly which violation exists)
  2. Test each predicate independently (no confounded violations)
  3. Run without any API calls or external data (fully offline)

This mirrors how MoVer evaluates temporal predicates: construct examples
that isolate each predicate, measure recall per predicate, measure false
positive rate on clean examples.

Predicate inventory (from constraints/verifier.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  dag_structure              [RFG][MILK]  — structural DAG invariants
  raw_protein_before_plate   [FDA]        — protein must be heat-treated
  solid_before_blend         [DC]         — solid ingredient directly blended
  no_heat_after_plate        [MILK]       — heat operations after PLATE
  moist_heat_needs_liquid    [PHYS]       — BOIL/SIMMER/BRAISE/POACH needs liquid
  combine_needs_multiple_inputs [MILK]    — COMBINE/MIX/TOSS need ≥2 inputs
  no_dangling_outputs        [MILK]       — all operation outputs must be consumed
  season_needs_target        [DC]         — SEASON needs a non-seasoning input

Usage
-----
    python scripts/evaluate_verification.py
    python scripts/evaluate_verification.py --verbose
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from collections import defaultdict

from culinary_descent.constraints.verifier import verify, ConstraintViolation
from culinary_descent.dsl.recipe_dag import RecipeDAG, Operation, RecipeNodeType
from culinary_descent.dsl.vocabulary import (
    DishCategory, IngredientCategory, OperationType, Ingredient, INGREDIENTS
)


# ---------------------------------------------------------------------------
# Test case builder helpers
# ---------------------------------------------------------------------------

def _ing(key: str) -> Ingredient:
    """Look up ingredient by key or build a minimal one."""
    return INGREDIENTS.get(key, Ingredient(key.replace("_", " "), IngredientCategory.VEGETABLE))


def _valid_pasta_arrabbiata() -> RecipeDAG:
    """Control: valid pasta arrabbiata — no proteins, should pass all constraints.

    Avoids the carbonara edge-case where eggs are "cooked" by residual heat
    (not expressible in our DSL), which would produce a spurious raw_protein flag.
    """
    dag = RecipeDAG("Pasta Arrabbiata", DishCategory.PASTA)
    dag.add_ingredient("n_penne",   _ing("penne"))
    dag.add_ingredient("n_water",   _ing("water"))
    dag.add_ingredient("n_tomato",  _ing("tomato"))
    dag.add_ingredient("n_garlic",  _ing("garlic"))
    dag.add_ingredient("n_chili",   _ing("chili_flakes"))
    dag.add_ingredient("n_oil",     _ing("olive_oil"))
    dag.add_ingredient("n_salt",    _ing("salt"))
    dag.add_operation("op_boil",    Operation(OperationType.BOIL))
    dag.add_operation("op_chop",    Operation(OperationType.CHOP))
    dag.add_operation("op_saute",   Operation(OperationType.SAUTE))
    dag.add_operation("op_combine", Operation(OperationType.COMBINE))
    dag.add_operation("op_season",  Operation(OperationType.SEASON))
    dag.add_operation("op_plate",   Operation(OperationType.PLATE))
    dag.add_dish_output("dish")
    dag.add_edge("n_penne",   "op_boil")
    dag.add_edge("n_water",   "op_boil")
    dag.add_edge("n_garlic",  "op_chop")
    dag.add_edge("n_oil",     "op_saute")
    dag.add_edge("op_chop",   "op_saute")
    dag.add_edge("n_tomato",  "op_saute")
    dag.add_edge("n_chili",   "op_saute")
    dag.add_edge("op_boil",   "op_combine")
    dag.add_edge("op_saute",  "op_combine")
    dag.add_edge("op_combine","op_season")
    dag.add_edge("n_salt",    "op_season")
    dag.add_edge("op_season", "op_plate")
    dag.add_edge("op_plate",  "dish")
    return dag


def _valid_stir_fry() -> RecipeDAG:
    """Control: valid chicken stir fry — should pass all constraints."""
    dag = RecipeDAG("Chicken Stir Fry", DishCategory.STIR_FRY)
    dag.add_ingredient("n_chicken",  _ing("chicken_breast"))
    dag.add_ingredient("n_pepper",   _ing("bell_pepper"))
    dag.add_ingredient("n_garlic",   _ing("garlic"))
    dag.add_ingredient("n_soy",      _ing("soy_sauce"))
    dag.add_ingredient("n_oil",      _ing("sesame_oil"))
    dag.add_operation("op_slice_c",  Operation(OperationType.SLICE))
    dag.add_operation("op_slice_p",  Operation(OperationType.SLICE))
    dag.add_operation("op_mince",    Operation(OperationType.MINCE))
    dag.add_operation("op_saute",    Operation(OperationType.SAUTE))
    dag.add_operation("op_toss",     Operation(OperationType.TOSS))
    dag.add_operation("op_plate",    Operation(OperationType.PLATE))
    dag.add_dish_output("dish")
    dag.add_edge("n_chicken", "op_slice_c")
    dag.add_edge("n_pepper",  "op_slice_p")
    dag.add_edge("n_garlic",  "op_mince")
    dag.add_edge("n_oil",     "op_saute")
    dag.add_edge("op_slice_c","op_saute")
    dag.add_edge("op_mince",  "op_saute")
    dag.add_edge("op_saute",  "op_toss")
    dag.add_edge("op_slice_p","op_toss")
    dag.add_edge("n_soy",     "op_toss")
    dag.add_edge("op_toss",   "op_plate")
    dag.add_edge("op_plate",  "dish")
    return dag


# ---------------------------------------------------------------------------
# Violation-triggering test cases (one per predicate)
# ---------------------------------------------------------------------------

def _violate_raw_protein() -> RecipeDAG:
    """raw_protein_before_plate: chicken goes to COMBINE without any heat."""
    dag = RecipeDAG("Raw Chicken Salad", DishCategory.SALAD)
    dag.add_ingredient("n_chicken",  _ing("chicken_breast"))   # is_protein=True
    dag.add_ingredient("n_lettuce",  _ing("lettuce"))
    dag.add_ingredient("n_oil",      _ing("olive_oil"))
    dag.add_operation("op_combine",  Operation(OperationType.COMBINE))
    dag.add_operation("op_plate",    Operation(OperationType.PLATE))
    dag.add_dish_output("dish")
    # chicken goes directly to combine — no heat
    dag.add_edge("n_chicken", "op_combine")
    dag.add_edge("n_lettuce", "op_combine")
    dag.add_edge("n_oil",     "op_combine")
    dag.add_edge("op_combine","op_plate")
    dag.add_edge("op_plate",  "dish")
    return dag


def _violate_solid_before_blend() -> RecipeDAG:
    """solid_before_blend: whole carrot goes directly into BLEND."""
    dag = RecipeDAG("Carrot Soup", DishCategory.SOUP)
    dag.add_ingredient("n_carrot",   _ing("carrot"))       # solid, not liquid
    dag.add_ingredient("n_broth",    _ing("chicken_broth"))
    dag.add_operation("op_blend",    Operation(OperationType.BLEND))
    dag.add_operation("op_plate",    Operation(OperationType.PLATE))
    dag.add_dish_output("dish")
    dag.add_edge("n_carrot",  "op_blend")   # solid → blend directly
    dag.add_edge("n_broth",   "op_blend")
    dag.add_edge("op_blend",  "op_plate")
    dag.add_edge("op_plate",  "dish")
    return dag


def _violate_no_heat_after_plate() -> RecipeDAG:
    """no_heat_after_plate: SAUTE appears as a successor of PLATE."""
    dag = RecipeDAG("Post-Plate Saute", DishCategory.EGGS)
    dag.add_ingredient("n_eggs",   _ing("eggs"))
    dag.add_ingredient("n_butter", _ing("butter"))
    dag.add_operation("op_plate",  Operation(OperationType.PLATE))
    dag.add_operation("op_saute",  Operation(OperationType.SAUTE))   # after plate!
    dag.add_dish_output("dish")
    dag.add_edge("n_eggs",   "op_plate")
    dag.add_edge("n_butter", "op_plate")
    dag.add_edge("op_plate", "op_saute")   # heat after plating
    dag.add_edge("op_saute", "dish")
    return dag


def _violate_moist_heat_no_liquid() -> RecipeDAG:
    """moist_heat_needs_liquid: BOIL with only solid ingredient inputs."""
    dag = RecipeDAG("Dry Boiled Pasta", DishCategory.PASTA)
    dag.add_ingredient("n_pasta",   _ing("spaghetti"))   # solid, not liquid
    dag.add_operation("op_boil",    Operation(OperationType.BOIL))
    dag.add_operation("op_plate",   Operation(OperationType.PLATE))
    dag.add_dish_output("dish")
    dag.add_edge("n_pasta",  "op_boil")   # boil without water
    dag.add_edge("op_boil",  "op_plate")
    dag.add_edge("op_plate", "dish")
    return dag


def _violate_combine_single_input() -> RecipeDAG:
    """combine_needs_multiple_inputs: MIX with only 1 input."""
    dag = RecipeDAG("Self Mix", DishCategory.EGGS)
    dag.add_ingredient("n_eggs",  _ing("eggs"))
    dag.add_operation("op_mix",   Operation(OperationType.MIX))   # needs ≥2
    dag.add_operation("op_plate", Operation(OperationType.PLATE))
    dag.add_dish_output("dish")
    dag.add_edge("n_eggs",   "op_mix")    # only one input
    dag.add_edge("op_mix",   "op_plate")
    dag.add_edge("op_plate", "dish")
    return dag


def _violate_dangling_output() -> RecipeDAG:
    """no_dangling_outputs: CHOP node output goes nowhere."""
    dag = RecipeDAG("Dangling Chop", DishCategory.SALAD)
    dag.add_ingredient("n_lettuce",  _ing("lettuce"))
    dag.add_ingredient("n_carrot",   _ing("carrot"))
    dag.add_ingredient("n_tomato",   _ing("tomato"))
    dag.add_operation("op_chop",     Operation(OperationType.CHOP))   # dangling
    dag.add_operation("op_toss",     Operation(OperationType.TOSS))
    dag.add_operation("op_plate",    Operation(OperationType.PLATE))
    dag.add_dish_output("dish")
    dag.add_edge("n_carrot",  "op_chop")        # op_chop has no successors → dangling
    dag.add_edge("n_lettuce", "op_toss")
    dag.add_edge("n_tomato",  "op_toss")
    dag.add_edge("op_toss",   "op_plate")
    dag.add_edge("op_plate",  "dish")
    return dag


def _violate_season_no_target() -> RecipeDAG:
    """season_needs_target: SEASON with only seasoning ingredients as input."""
    dag = RecipeDAG("Season of Seasonings", DishCategory.SALAD)
    dag.add_ingredient("n_salt",     _ing("salt"))        # SEASONING
    dag.add_ingredient("n_pepper",   _ing("black_pepper")) # SEASONING
    dag.add_ingredient("n_lettuce",  _ing("lettuce"))
    dag.add_operation("op_season",   Operation(OperationType.SEASON))
    dag.add_operation("op_toss",     Operation(OperationType.TOSS))
    dag.add_operation("op_plate",    Operation(OperationType.PLATE))
    dag.add_dish_output("dish")
    # SEASON only has seasonings as inputs (no base food)
    dag.add_edge("n_salt",    "op_season")
    dag.add_edge("n_pepper",  "op_season")
    dag.add_edge("op_season", "op_toss")
    dag.add_edge("n_lettuce", "op_toss")
    dag.add_edge("op_toss",   "op_plate")
    dag.add_edge("op_plate",  "dish")
    return dag


def _violate_dag_structure() -> RecipeDAG:
    """dag_structure: ingredient with no outgoing edges (unused ingredient)."""
    dag = RecipeDAG("Unused Ingredient", DishCategory.PASTA)
    dag.add_ingredient("n_pasta",  _ing("spaghetti"))
    dag.add_ingredient("n_water",  _ing("water"))
    dag.add_ingredient("n_garlic", _ing("garlic"))   # UNUSED — no outgoing edge
    dag.add_operation("op_boil",   Operation(OperationType.BOIL))
    dag.add_operation("op_plate",  Operation(OperationType.PLATE))
    dag.add_dish_output("dish")
    dag.add_edge("n_pasta",  "op_boil")
    dag.add_edge("n_water",  "op_boil")
    # n_garlic has no edge → structural violation
    dag.add_edge("op_boil",  "op_plate")
    dag.add_edge("op_plate", "dish")
    return dag


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    name: str
    recipe: RecipeDAG
    expected_violation: str   # rule name expected, or "none" for valid recipes


TEST_CASES: list[TestCase] = [
    # ── Valid controls ──────────────────────────────────────────────────────
    TestCase("valid:arrabbiata",        _valid_pasta_arrabbiata(),     "none"),
    TestCase("valid:stir_fry",          _valid_stir_fry(),             "none"),

    # ── One violation each ──────────────────────────────────────────────────
    TestCase("viol:raw_protein",        _violate_raw_protein(),        "raw_protein_before_plate"),
    TestCase("viol:solid_before_blend", _violate_solid_before_blend(), "solid_before_blend"),
    TestCase("viol:heat_after_plate",   _violate_no_heat_after_plate(),"no_heat_after_plate"),
    TestCase("viol:moist_no_liquid",    _violate_moist_heat_no_liquid(),"moist_heat_needs_liquid"),
    TestCase("viol:combine_single",     _violate_combine_single_input(),"combine_needs_multiple_inputs"),
    TestCase("viol:dangling_output",    _violate_dangling_output(),    "no_dangling_outputs"),
    TestCase("viol:season_no_target",   _violate_season_no_target(),   "season_needs_target"),
    TestCase("viol:dag_structure",      _violate_dag_structure(),      "dag_structure"),
]


def run_recall_eval(verbose: bool = False) -> dict:
    tp = 0   # expected violation, verifier caught it
    fp = 0   # expected valid, verifier flagged it (false alarm)
    fn = 0   # expected violation, verifier missed it

    per_predicate: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fn": 0})

    print("\nVERIFIER RECALL EVALUATION")
    print("=" * 70)
    print(f"{'Test case':<35} {'Expected':<35} {'Caught'}")
    print("-" * 70)

    for tc in TEST_CASES:
        result = verify(tc.recipe)
        caught_rules = {v.rule for v in result.violations}

        if tc.expected_violation == "none":
            # Control: should be valid
            if result.is_valid:
                status = "✓ no false positive"
            else:
                status = f"✗ FALSE POSITIVE: {', '.join(caught_rules)}"
                fp += 1
        else:
            # Violation case
            if tc.expected_violation in caught_rules:
                status = f"✓ caught"
                tp += 1
                per_predicate[tc.expected_violation]["tp"] += 1
            else:
                status = f"✗ MISSED (got: {caught_rules or 'nothing'})"
                fn += 1
                per_predicate[tc.expected_violation]["fn"] += 1

            if verbose and result.violations:
                for v in result.violations:
                    print(f"    [{v.rule}] {v.message[:60]}")

        expected_short = tc.expected_violation[:33]
        print(f"  {tc.name:<33} {expected_short:<35} {status}")

    n_violation_cases = sum(1 for tc in TEST_CASES if tc.expected_violation != "none")
    recall = tp / n_violation_cases if n_violation_cases else 0.0

    print("=" * 70)
    print(f"\nOverall recall:  {tp}/{n_violation_cases} = {recall:.1%}")
    print(f"False positives: {fp} (out of {len(TEST_CASES) - n_violation_cases} valid controls)")

    if per_predicate:
        print("\nPer-predicate recall:")
        for rule, counts in sorted(per_predicate.items()):
            hit = "✓" if counts["fn"] == 0 else "✗ MISSED"
            print(f"  {rule:<40} {hit}")

    return {
        "n_total": len(TEST_CASES),
        "n_violation_cases": n_violation_cases,
        "n_control_cases": len(TEST_CASES) - n_violation_cases,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "recall": recall,
        "per_predicate": dict(per_predicate),
    }


def main():
    parser = argparse.ArgumentParser(description="Verifier recall evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full violation messages")
    args = parser.parse_args()

    run_recall_eval(verbose=args.verbose)


if __name__ == "__main__":
    main()
