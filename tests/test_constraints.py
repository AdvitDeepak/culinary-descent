"""Tests for the constraint verifier — both positive (valid) and negative (invalid) cases."""

import pytest
from culinary_descent.dsl.recipe_dag import RecipeDAG, Operation
from culinary_descent.dsl.vocabulary import DishCategory, OperationType, INGREDIENTS
from culinary_descent.constraints.verifier import verify, ConstraintViolation


# ---------------------------------------------------------------------------
# Raw protein without heat → violation
# ---------------------------------------------------------------------------

def test_raw_protein_into_combine_is_violation():
    dag = RecipeDAG("Raw Chicken Pasta", DishCategory.PASTA)
    dag.add_ingredient("i_chicken", INGREDIENTS["chicken_breast"])
    dag.add_ingredient("i_spaghetti", INGREDIENTS["spaghetti"])
    dag.add_ingredient("i_water", INGREDIENTS["water"])
    dag.add_operation("op_boil",    Operation(OperationType.BOIL,    {}))
    dag.add_operation("op_combine", Operation(OperationType.COMBINE, {}))
    dag.add_operation("op_plate",   Operation(OperationType.PLATE,   {}))
    dag.add_dish_output("out")
    dag.add_edge("i_chicken",   "op_combine")   # raw protein → combine!
    dag.add_edge("i_spaghetti", "op_boil")
    dag.add_edge("i_water",     "op_boil")
    dag.add_edge("op_boil",     "op_combine")
    dag.add_edge("op_combine",  "op_plate")
    dag.add_edge("op_plate",    "out")

    result = verify(dag)
    assert not result.is_valid
    rules = [v.rule for v in result.violations]
    assert "raw_protein_before_plate" in rules


def test_cooked_protein_is_valid():
    dag = RecipeDAG("Chicken Pasta", DishCategory.PASTA)
    dag.add_ingredient("i_chicken", INGREDIENTS["chicken_breast"])
    dag.add_ingredient("i_spaghetti", INGREDIENTS["spaghetti"])
    dag.add_ingredient("i_water", INGREDIENTS["water"])
    dag.add_operation("op_sear",    Operation(OperationType.SEAR,    {}))
    dag.add_operation("op_boil",    Operation(OperationType.BOIL,    {}))
    dag.add_operation("op_combine", Operation(OperationType.COMBINE, {}))
    dag.add_operation("op_plate",   Operation(OperationType.PLATE,   {}))
    dag.add_dish_output("out")
    dag.add_edge("i_chicken",   "op_sear")      # protein is cooked first
    dag.add_edge("i_spaghetti", "op_boil")
    dag.add_edge("i_water",     "op_boil")
    dag.add_edge("op_sear",     "op_combine")   # cooked chicken → combine (ok)
    dag.add_edge("op_boil",     "op_combine")
    dag.add_edge("op_combine",  "op_plate")
    dag.add_edge("op_plate",    "out")

    result = verify(dag)
    protein_violations = [v for v in result.violations if v.rule == "raw_protein_before_combine"]
    assert len(protein_violations) == 0


# ---------------------------------------------------------------------------
# Heat after PLATE → violation
# ---------------------------------------------------------------------------

def test_heat_after_plate_is_violation():
    dag = RecipeDAG("Plate Then Sear", DishCategory.EGGS)
    dag.add_ingredient("i_eggs", INGREDIENTS["eggs"])
    dag.add_ingredient("i_butter", INGREDIENTS["butter"])
    dag.add_operation("op_saute", Operation(OperationType.SAUTE, {}))
    dag.add_operation("op_plate", Operation(OperationType.PLATE, {}))
    dag.add_operation("op_sear",  Operation(OperationType.SEAR,  {}))  # after plate!
    dag.add_dish_output("out")
    dag.add_edge("i_eggs",   "op_saute")
    dag.add_edge("i_butter", "op_saute")
    dag.add_edge("op_saute", "op_plate")
    dag.add_edge("op_plate", "op_sear")
    dag.add_edge("op_sear",  "out")

    result = verify(dag)
    rules = [v.rule for v in result.violations]
    assert "no_heat_after_plate" in rules


# ---------------------------------------------------------------------------
# Moist heat without liquid → violation
# ---------------------------------------------------------------------------

def test_boil_without_liquid_is_violation():
    dag = RecipeDAG("Dry Boil", DishCategory.EGGS)
    dag.add_ingredient("i_eggs", INGREDIENTS["eggs"])
    dag.add_operation("op_boil",  Operation(OperationType.BOIL,  {}))
    dag.add_operation("op_plate", Operation(OperationType.PLATE, {}))
    dag.add_dish_output("out")
    dag.add_edge("i_eggs",   "op_boil")
    dag.add_edge("op_boil",  "op_plate")
    dag.add_edge("op_plate", "out")

    result = verify(dag)
    rules = [v.rule for v in result.violations]
    assert "moist_heat_needs_liquid" in rules


def test_boil_with_liquid_is_valid():
    dag = RecipeDAG("Boiled Egg", DishCategory.EGGS)
    dag.add_ingredient("i_eggs",  INGREDIENTS["eggs"])
    dag.add_ingredient("i_water", INGREDIENTS["water"])
    dag.add_operation("op_boil",  Operation(OperationType.BOIL,  {}))
    dag.add_operation("op_plate", Operation(OperationType.PLATE, {}))
    dag.add_dish_output("out")
    dag.add_edge("i_eggs",   "op_boil")
    dag.add_edge("i_water",  "op_boil")
    dag.add_edge("op_boil",  "op_plate")
    dag.add_edge("op_plate", "out")

    result = verify(dag)
    liquid_violations = [v for v in result.violations if v.rule == "moist_heat_needs_liquid"]
    assert len(liquid_violations) == 0


# ---------------------------------------------------------------------------
# COMBINE with one input → violation
# ---------------------------------------------------------------------------

def test_combine_one_input_is_violation():
    dag = RecipeDAG("Solo Combine", DishCategory.PASTA)
    dag.add_ingredient("i_spaghetti", INGREDIENTS["spaghetti"])
    dag.add_ingredient("i_water", INGREDIENTS["water"])
    dag.add_operation("op_boil",    Operation(OperationType.BOIL,    {}))
    dag.add_operation("op_combine", Operation(OperationType.COMBINE, {}))
    dag.add_operation("op_plate",   Operation(OperationType.PLATE,   {}))
    dag.add_dish_output("out")
    dag.add_edge("i_spaghetti", "op_boil")
    dag.add_edge("i_water",     "op_boil")
    dag.add_edge("op_boil",     "op_combine")   # only one input to COMBINE
    dag.add_edge("op_combine",  "op_plate")
    dag.add_edge("op_plate",    "out")

    result = verify(dag)
    rules = [v.rule for v in result.violations]
    assert "combine_needs_multiple_inputs" in rules


# ---------------------------------------------------------------------------
# Solid ingredient directly into BLEND → violation
# ---------------------------------------------------------------------------

def test_solid_directly_into_blend_is_violation():
    dag = RecipeDAG("Raw Blend", DishCategory.SOUP)
    dag.add_ingredient("i_carrot", INGREDIENTS["carrot"])
    dag.add_ingredient("i_tomato", INGREDIENTS["tomato"])
    dag.add_operation("op_blend",  Operation(OperationType.BLEND,  {}))
    dag.add_operation("op_plate",  Operation(OperationType.PLATE,  {}))
    dag.add_dish_output("out")
    dag.add_edge("i_carrot", "op_blend")   # solid, unprepped → blend
    dag.add_edge("i_tomato", "op_blend")
    dag.add_edge("op_blend", "op_plate")
    dag.add_edge("op_plate", "out")

    result = verify(dag)
    rules = [v.rule for v in result.violations]
    assert "solid_before_blend" in rules


# ---------------------------------------------------------------------------
# Dangling operation (no successors) → violation
# ---------------------------------------------------------------------------

def test_dangling_operation_is_violation():
    dag = RecipeDAG("Dangling Op", DishCategory.EGGS)
    dag.add_ingredient("i_eggs",   INGREDIENTS["eggs"])
    dag.add_ingredient("i_butter", INGREDIENTS["butter"])
    dag.add_operation("op_saute",   Operation(OperationType.SAUTE, {}))
    dag.add_operation("op_plate",   Operation(OperationType.PLATE, {}))
    dag.add_operation("op_orphan",  Operation(OperationType.ROAST, {}))  # never connected downstream
    dag.add_dish_output("out")
    dag.add_edge("i_eggs",   "op_saute")
    dag.add_edge("i_butter", "op_saute")
    dag.add_edge("op_saute", "op_plate")
    dag.add_edge("op_plate", "out")
    # op_orphan has no successors but also no predecessors (just floating)

    result = verify(dag)
    rules = [v.rule for v in result.violations]
    assert "no_dangling_outputs" in rules
