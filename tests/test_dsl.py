"""Tests for the DSL layer: RecipeDAG construction, serialization, and validation."""

import pytest
from culinary_descent.dsl.vocabulary import (
    DishCategory, IngredientCategory, OperationType, INGREDIENTS, OPERATIONS,
    REWRITE_GROUPS, OperationGroup,
)
from culinary_descent.dsl.recipe_dag import RecipeDAG, Operation, RecipeNode, RecipeNodeType


# ---------------------------------------------------------------------------
# Vocabulary tests
# ---------------------------------------------------------------------------

def test_all_operations_have_metadata():
    for op_type in OperationType:
        assert op_type in OPERATIONS, f"OperationType.{op_type.name} missing from OPERATIONS"
        meta = OPERATIONS[op_type]
        assert meta.group in OperationGroup
        assert meta.min_inputs >= 0


def test_rewrite_groups_partition_operations():
    """Every OperationType should appear in exactly one REWRITE_GROUPS entry."""
    seen = set()
    for group_ops in REWRITE_GROUPS.values():
        for op in group_ops:
            assert op not in seen, f"{op} appears in multiple REWRITE_GROUPS"
            seen.add(op)
    for op in OperationType:
        assert op in seen, f"OperationType.{op.name} not in any REWRITE_GROUPS"


def test_liquid_ingredients_are_marked():
    liquids = [name for name, ing in INGREDIENTS.items() if ing.is_liquid]
    assert len(liquids) >= 5, "Expected at least 5 liquid ingredients in vocabulary"


def test_protein_ingredients_are_marked():
    proteins = [name for name, ing in INGREDIENTS.items() if ing.is_protein]
    assert len(proteins) >= 5


# ---------------------------------------------------------------------------
# RecipeDAG construction
# ---------------------------------------------------------------------------

def _simple_dag() -> RecipeDAG:
    dag = RecipeDAG("Test Dish", DishCategory.EGGS)
    dag.add_ingredient("i_eggs", INGREDIENTS["eggs"])
    dag.add_ingredient("i_butter", INGREDIENTS["butter"])
    dag.add_operation("op_saute", Operation(OperationType.SAUTE, {"minutes": 3}))
    dag.add_operation("op_plate", Operation(OperationType.PLATE, {}))
    dag.add_dish_output("out")
    dag.add_edge("i_eggs",   "op_saute")
    dag.add_edge("i_butter", "op_saute")
    dag.add_edge("op_saute", "op_plate")
    dag.add_edge("op_plate", "out")
    return dag


def test_dag_ingredient_nodes():
    dag = _simple_dag()
    assert len(dag.ingredient_nodes()) == 2
    names = dag.ingredient_names()
    assert "eggs" in names
    assert "butter" in names


def test_dag_operation_nodes():
    dag = _simple_dag()
    ops = dag.operation_types()
    assert OperationType.SAUTE in ops
    assert OperationType.PLATE in ops


def test_dag_single_dish_output():
    dag = _simple_dag()
    assert dag.dish_output_node() is not None


def test_dag_topological_sort_is_valid():
    dag = _simple_dag()
    order = dag.topological_sort()
    assert len(order) == len(dag.nodes())
    # Ingredients must come before operations
    pos = {nid: i for i, nid in enumerate(order)}
    for src, dsts in dag._adj.items():
        for dst in dsts:
            assert pos[src] < pos[dst], f"Edge {src}→{dst} violates topological order"


def test_dag_cycle_detection():
    dag = RecipeDAG("Cyclic", DishCategory.EGGS)
    dag.add_operation("op_a", Operation(OperationType.SAUTE, {}))
    dag.add_operation("op_b", Operation(OperationType.PLATE, {}))
    dag.add_dish_output("out")
    dag._nodes["op_a"]  # exists
    dag._adj["op_a"].append("op_b")
    dag._adj["op_b"].append("op_a")  # cycle!
    dag._radj["op_b"].append("op_a")
    dag._radj["op_a"].append("op_b")
    dag._adj["op_b"].append("out")
    dag._radj["out"].append("op_b")

    with pytest.raises(ValueError, match="cycle"):
        dag.topological_sort()


# ---------------------------------------------------------------------------
# Text serialization
# ---------------------------------------------------------------------------

def test_to_text_contains_dish_name():
    dag = _simple_dag()
    text = dag.to_text()
    assert "Test Dish" in text


def test_to_text_contains_ingredients():
    dag = _simple_dag()
    text = dag.to_text()
    assert "eggs" in text
    assert "butter" in text


def test_to_text_contains_operations():
    dag = _simple_dag()
    text = dag.to_text()
    assert "saute" in text or "saut" in text


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

def test_json_roundtrip():
    dag = _simple_dag()
    reloaded = RecipeDAG.from_json(dag.to_json())
    assert reloaded.dish_name == dag.dish_name
    assert reloaded.category == dag.category
    assert set(reloaded.ingredient_names()) == set(dag.ingredient_names())
    assert set(op.value for op in reloaded.operation_types()) == \
           set(op.value for op in dag.operation_types())


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

def test_validate_structure_valid_dag():
    dag = _simple_dag()
    errors = dag.validate_structure()
    assert errors == [], f"Unexpected errors in valid DAG: {errors}"


def test_validate_structure_no_output():
    dag = RecipeDAG("No Output", DishCategory.EGGS)
    dag.add_ingredient("i_eggs", INGREDIENTS["eggs"])
    dag.add_operation("op_plate", Operation(OperationType.PLATE, {}))
    dag.add_edge("i_eggs", "op_plate")
    errors = dag.validate_structure()
    assert any("DISH_OUTPUT" in e for e in errors)


def test_validate_structure_unused_ingredient():
    dag = RecipeDAG("Unused Ing", DishCategory.EGGS)
    dag.add_ingredient("i_eggs", INGREDIENTS["eggs"])
    dag.add_ingredient("i_butter", INGREDIENTS["butter"])  # never connected
    dag.add_operation("op_plate", Operation(OperationType.PLATE, {}))
    dag.add_dish_output("out")
    dag.add_edge("i_eggs",   "op_plate")
    dag.add_edge("op_plate", "out")
    errors = dag.validate_structure()
    assert any("unused" in e.lower() or "butter" in e.lower() for e in errors)
