"""
Cooking Constraint Verifier — MoVer-style predicate checking for recipe DAGs.

Sources: [MILK] Tasse & Smith CMU-LTI-08-005 2008, [RFG] Yamakata LREC 2020,
[FDA] FSMA HACCP guidelines, [PHYS] McGee "On Food and Cooking" §5 2004,
[DC] design choice. Each predicate is tagged with its source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..dsl.recipe_dag import RecipeDAG, RecipeNodeType
from ..dsl.vocabulary import OperationType, OperationGroup, OPERATIONS


@dataclass
class ConstraintViolation:
    rule: str
    message: str
    node_ids: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        nodes = ", ".join(self.node_ids) if self.node_ids else "—"
        return f"[{self.rule}] {self.message} (nodes: {nodes})"


@dataclass
class VerificationResult:
    is_valid: bool
    violations: list[ConstraintViolation]

    def __str__(self) -> str:
        if self.is_valid:
            return "✓ Recipe is valid (no constraint violations)"
        lines = [f"✗ Recipe has {len(self.violations)} violation(s):"]
        for v in self.violations:
            lines.append(f"  · {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Predicate 1: DAG structural invariants
# Source: [RFG] — Recipe Flow Graphs are DAGs with a single root (final dish)
#                 and leaf nodes (raw ingredients); Yamakata 2020 §2, Mori 2014 §2
#         [MILK] — world state is well-founded: ingredients must be created before
#                  use, all ingredients should be served (§3 operational semantics)
# ---------------------------------------------------------------------------

def _check_dag_structure(dag: RecipeDAG) -> list[ConstraintViolation]:
    """DAG structural invariants from the Recipe Flow Graph representation.

    Checks:
    - Exactly one DISH_OUTPUT node (the recipe root) [RFG]
    - No cycles (DAG property required by both [RFG] and [MILK]) [RFG][MILK]
    - No unused ingredients — every ingredient must be consumed [MILK serve]
    """
    errors = dag.validate_structure()
    return [ConstraintViolation(rule="dag_structure", message=e) for e in errors]


# ---------------------------------------------------------------------------
# Predicate 2: Raw protein must be cooked before reaching the dish output
# Source: [FDA] — FDA FSMA HACCP guidelines specify minimum internal temperatures
#                 for food safety; proteins that haven't passed through any heat
#                 operation are "raw" by definition and unsafe to serve
# ---------------------------------------------------------------------------

def _check_raw_protein_cooked(dag: RecipeDAG) -> list[ConstraintViolation]:
    """Every protein must pass through at least one heat operation.

    Checks whether every path from a protein ingredient leaf to the DISH_OUTPUT
    root includes at least one DRY_HEAT or MOIST_HEAT operation.

    Grounded in [FDA]: FDA Model Food Code §3-401 specifies minimum internal
    temperatures for poultry (165°F), ground meat (160°F), fish (145°F), and
    eggs (until yolk/white are firm). An ingredient that reaches the final dish
    without any heat operation is definitionally raw.

    Note: 'protein' here corresponds to INGREDIENTS where is_protein=True,
    which is set based on the FDA FSMA food safety classification.
    """
    violations: list[ConstraintViolation] = []
    heat_groups = {OperationGroup.DRY_HEAT, OperationGroup.MOIST_HEAT}
    output_node = dag.dish_output_node()
    if output_node is None:
        return violations

    def _has_uncooked_path(start_id: str) -> bool:
        """DFS: is there any path from start to output with no heat operation?"""
        stack = [(start_id, False)]  # (node_id, seen_heat_on_this_path)
        visited: set[tuple[str, bool]] = set()
        while stack:
            nid, seen_heat = stack.pop()
            if (nid, seen_heat) in visited:
                continue
            visited.add((nid, seen_heat))
            node = dag.get_node(nid)
            is_heat = (
                node.node_type == RecipeNodeType.OPERATION
                and node.operation is not None
                and OPERATIONS[node.operation.op_type].group in heat_groups
            )
            now_heat = seen_heat or is_heat
            if nid == output_node.node_id:
                if not now_heat:
                    return True
                continue
            for succ_id in dag.successors(nid):
                stack.append((succ_id, now_heat))
        return False

    for ing_node in dag.ingredient_nodes():
        if not ing_node.ingredient or not ing_node.ingredient.is_protein:
            continue
        if _has_uncooked_path(ing_node.node_id):
            violations.append(ConstraintViolation(
                rule="raw_protein_before_plate",
                message=(
                    f"[FDA] Protein '{ing_node.ingredient.name}' reaches the dish "
                    f"without passing through any heat operation"
                ),
                node_ids=[ing_node.node_id],
            ))
    return violations


# ---------------------------------------------------------------------------
# Predicate 3: Solid ingredients should be prepared before blending
# Source: [DC] Design Choice — blending whole solid vegetables (carrots, etc.)
#              without any preparation is physically possible but produces poor
#              results and is absent from all Recipe1M recipes in the BLEND context.
#              This rule is labeled as [DC] because no published paper formally
#              defines this constraint; it is a reasonable engineering assumption.
# ---------------------------------------------------------------------------

def _check_solid_before_blend(dag: RecipeDAG) -> list[ConstraintViolation]:
    """[DC] Solid non-liquid ingredients should be prepared before blending.

    A non-liquid ingredient directly entering BLEND without any PREP upstream is
    flagged. This is a design choice: while physically possible, it does not appear
    in Recipe1M BLEND contexts.
    """
    violations: list[ConstraintViolation] = []
    for op_node in dag.operation_nodes():
        if op_node.operation is None or op_node.operation.op_type != OperationType.BLEND:
            continue
        for pred_id in dag.predecessors(op_node.node_id):
            pred = dag.get_node(pred_id)
            if (pred.node_type == RecipeNodeType.INGREDIENT
                    and pred.ingredient and not pred.ingredient.is_liquid):
                violations.append(ConstraintViolation(
                    rule="solid_before_blend",
                    message=(
                        f"[DC] Solid ingredient '{pred.ingredient.name}' enters BLEND "
                        f"without any preparation"
                    ),
                    node_ids=[pred_id, op_node.node_id],
                ))
    return violations


# ---------------------------------------------------------------------------
# Predicate 4: PLATE terminates cooking; heat operations may not follow it
# Source: [MILK] — serve() is the final instruction; §3.2 states that after
#                  serve(), no more operations should transform the ingredient.
#         [DC]   — the boundary between PLATE and post-plating ops (GARNISH, REST)
#                  is our extension of the MILK serve() terminal semantics.
# ---------------------------------------------------------------------------

def _check_plate_is_terminal(dag: RecipeDAG) -> list[ConstraintViolation]:
    """[MILK] No heat operations may follow PLATE; heat after plating is invalid.

    Grounded in MILK's serve() semantics: once an ingredient is served (plated),
    its state should not be further modified. Post-plating operations like
    GARNISH and REST are permitted (they do not transform the dish's heat state).
    """
    violations: list[ConstraintViolation] = []
    disallowed_after_plate = {OperationGroup.DRY_HEAT, OperationGroup.MOIST_HEAT}

    for op_node in dag.operation_nodes():
        if op_node.operation is None or op_node.operation.op_type != OperationType.PLATE:
            continue
        visited = set()
        stack = list(dag.successors(op_node.node_id))
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            node = dag.get_node(nid)
            if node.node_type == RecipeNodeType.OPERATION and node.operation:
                if OPERATIONS[node.operation.op_type].group in disallowed_after_plate:
                    violations.append(ConstraintViolation(
                        rule="no_heat_after_plate",
                        message=(
                            f"[MILK/serve] Heat operation '{node.operation.op_type.value}' "
                            f"appears after PLATE"
                        ),
                        node_ids=[op_node.node_id, nid],
                    ))
            stack.extend(dag.successors(nid))
    return violations


# ---------------------------------------------------------------------------
# Predicate 5: Moist heat requires a liquid input
# Source: [PHYS] — thermodynamic necessity; BOIL/SIMMER/BRAISE/POACH transfer
#                  heat through a liquid medium (water, broth, wine).
#         [MILK] — cook() action: "Cooking or heating in any way (boil, bake, fry)"
#                  The moist/liquid subcategory requires a liquid parameter.
# ---------------------------------------------------------------------------

def _check_moist_heat_has_liquid(dag: RecipeDAG) -> list[ConstraintViolation]:
    """[PHYS] Moist-heat operations require at least one liquid input.

    Grounded in basic thermodynamics [PHYS]: boiling and simmering transfer
    heat through convection in a liquid medium. An operation node for BOIL/
    SIMMER/BRAISE/POACH with no liquid predecessor is physically incoherent.
    """
    violations: list[ConstraintViolation] = []
    moist_ops = {OperationType.BOIL, OperationType.SIMMER,
                 OperationType.BRAISE, OperationType.POACH}

    for op_node in dag.operation_nodes():
        if op_node.operation is None or op_node.operation.op_type not in moist_ops:
            continue
        has_liquid = False
        for pred_id in dag.predecessors(op_node.node_id):
            pred = dag.get_node(pred_id)
            if (pred.node_type == RecipeNodeType.INGREDIENT
                    and pred.ingredient and pred.ingredient.is_liquid):
                has_liquid = True
                break
            if pred.node_type == RecipeNodeType.OPERATION:
                has_liquid = True  # upstream op output carries its liquid context
                break
        if not has_liquid:
            violations.append(ConstraintViolation(
                rule="moist_heat_needs_liquid",
                message=(
                    f"[PHYS] '{op_node.operation.op_type.value}' has no liquid input — "
                    f"moist heat requires a liquid medium"
                ),
                node_ids=[op_node.node_id],
            ))
    return violations


# ---------------------------------------------------------------------------
# Predicate 6: Combining operations require ≥2 inputs
# Source: [MILK] — combine(ingredient_set, ...) requires ingredient_set ≥ 2
#                  (§3.1.2: "ingredient_set = syntactic sugar for ≥2 ingredients")
# ---------------------------------------------------------------------------

def _check_combine_has_multiple_inputs(dag: RecipeDAG) -> list[ConstraintViolation]:
    """[MILK] COMBINE/MIX/TOSS require at least 2 distinct inputs.

    Grounded in MILK's combine() predicate: "ingredient_set ins" must have
    cardinality ≥ 2. Combining a single ingredient with nothing is semantically
    incoherent — it is at most a transformation, not a combination.
    """
    violations: list[ConstraintViolation] = []
    combine_ops = {OperationType.COMBINE, OperationType.MIX, OperationType.TOSS}

    for op_node in dag.operation_nodes():
        if op_node.operation is None or op_node.operation.op_type not in combine_ops:
            continue
        n_inputs = len(dag.predecessors(op_node.node_id))
        if n_inputs < 2:
            violations.append(ConstraintViolation(
                rule="combine_needs_multiple_inputs",
                message=(
                    f"[MILK/combine] '{op_node.operation.op_type.value}' has "
                    f"{n_inputs} input(s); requires ≥2"
                ),
                node_ids=[op_node.node_id],
            ))
    return violations


# ---------------------------------------------------------------------------
# Predicate 7: No dangling operation outputs (all outputs must be consumed)
# Source: [MILK] — "All ingredients in I should eventually be served or deleted"
#                  (§2, world state final condition).  An operation whose output
#                  is never consumed is equivalent to a recipe step that produces
#                  food that is then discarded — not a valid recipe.
# ---------------------------------------------------------------------------

def _check_no_dangling_outputs(dag: RecipeDAG) -> list[ConstraintViolation]:
    """[MILK] Every operation's output must eventually reach the dish output.

    Grounded in MILK's world-state final condition: all ingredients must be
    served. An operation node with no successors (other than DISH_OUTPUT) means
    its output is discarded, which is not a valid recipe state.
    """
    violations: list[ConstraintViolation] = []
    output_node = dag.dish_output_node()
    output_id = output_node.node_id if output_node else None

    for node in dag.operation_nodes():
        if not dag.successors(node.node_id) and node.node_id != output_id:
            violations.append(ConstraintViolation(
                rule="no_dangling_outputs",
                message=(
                    f"[MILK/serve] Operation '{node.label()}' ({node.node_id}) "
                    f"has no successors — output is never consumed"
                ),
                node_ids=[node.node_id],
            ))
    return violations


# ---------------------------------------------------------------------------
# Predicate 8: SEASON must have a non-seasoning target to season
# Source: [DC] — Design choice grounded in semantic consistency: SEASON is a
#               modifier operation (MILK's "set" analogue for flavor state).
#               Seasoning a seasoning (e.g., adding pepper to pepper) is
#               semantically incoherent and never appears in Recipe1M.
# ---------------------------------------------------------------------------

def _check_season_has_target(dag: RecipeDAG) -> list[ConstraintViolation]:
    """[DC] SEASON must have at least one non-seasoning predecessor.

    Design choice: SEASON is a flavor modifier applied to a dish component;
    it is semantically incoherent for SEASON to operate on only seasonings.
    This does not appear in Recipe1M's SEASON contexts.
    """
    violations: list[ConstraintViolation] = []
    from ..dsl.vocabulary import IngredientCategory

    for op_node in dag.operation_nodes():
        if op_node.operation is None or op_node.operation.op_type != OperationType.SEASON:
            continue
        has_non_seasoning_input = False
        for pred_id in dag.predecessors(op_node.node_id):
            pred = dag.get_node(pred_id)
            if pred.node_type == RecipeNodeType.INGREDIENT and pred.ingredient:
                if pred.ingredient.category != IngredientCategory.SEASONING:
                    has_non_seasoning_input = True
                    break
            if pred.node_type == RecipeNodeType.OPERATION:
                has_non_seasoning_input = True
                break
        if not has_non_seasoning_input:
            violations.append(ConstraintViolation(
                rule="season_needs_target",
                message="[DC] SEASON has no non-seasoning input",
                node_ids=[op_node.node_id],
            ))
    return violations


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_PREDICATES: list[Callable[[RecipeDAG], list[ConstraintViolation]]] = [
    _check_dag_structure,        # [RFG][MILK]
    _check_raw_protein_cooked,   # [FDA]
    _check_solid_before_blend,   # [DC]
    _check_plate_is_terminal,    # [MILK]
    _check_moist_heat_has_liquid,# [PHYS]
    _check_combine_has_multiple_inputs,  # [MILK]
    _check_no_dangling_outputs,  # [MILK]
    _check_season_has_target,    # [DC]
]


def verify(dag: RecipeDAG) -> VerificationResult:
    """Run all constraint predicates and return a VerificationResult.

    All predicates run even when earlier ones fail, giving a complete
    violation list for repair.
    """
    all_violations: list[ConstraintViolation] = []
    for predicate in _PREDICATES:
        all_violations.extend(predicate(dag))
    return VerificationResult(
        is_valid=len(all_violations) == 0,
        violations=all_violations,
    )
