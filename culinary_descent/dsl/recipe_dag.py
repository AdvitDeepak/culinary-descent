"""
RecipeDAG — the core program representation.

Leaf-to-root DAG: ingredient leaves → operation nodes → single DISH_OUTPUT root.
Edges represent material flow (source is consumed by destination).
to_text() serializes to a natural-language description via topological sort.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterator, Optional

from .vocabulary import (
    DishCategory,
    Ingredient,
    IngredientCategory,
    OperationMeta,
    OperationType,
    INGREDIENTS,
    OPERATIONS,
)


class RecipeNodeType(Enum):
    INGREDIENT  = "ingredient"
    OPERATION   = "operation"
    DISH_OUTPUT = "dish_output"


@dataclass
class Operation:
    """An instantiated cooking operation with optional parameters.

    Attributes
    ----------
    op_type:
        The type of operation (must be a key in OPERATIONS).
    params:
        Optional named parameters, e.g. {"minutes": 10, "temp_c": 180}.
        No schema enforcement at this level; the verifier checks semantics.
    """
    op_type: OperationType
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def meta(self) -> OperationMeta:
        return OPERATIONS[self.op_type]

    def to_dict(self) -> dict:
        return {"op_type": self.op_type.value, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> "Operation":
        return cls(op_type=OperationType(d["op_type"]), params=d.get("params", {}))


@dataclass
class RecipeNode:
    """A single node in the RecipeDAG.

    Attributes
    ----------
    node_id:
        Unique string identifier within its DAG (e.g. "n0", "n1").
    node_type:
        INGREDIENT, OPERATION, or DISH_OUTPUT.
    ingredient:
        Set only when node_type == INGREDIENT.
    operation:
        Set only when node_type == OPERATION.
    """
    node_id: str
    node_type: RecipeNodeType
    ingredient: Optional[Ingredient] = None
    operation: Optional[Operation] = None

    def __post_init__(self):
        if self.node_type == RecipeNodeType.INGREDIENT:
            assert self.ingredient is not None, f"Node {self.node_id}: ingredient nodes require an ingredient"
        elif self.node_type == RecipeNodeType.OPERATION:
            assert self.operation is not None, f"Node {self.node_id}: operation nodes require an operation"

    def label(self) -> str:
        """Human-readable label for visualization and debugging."""
        if self.node_type == RecipeNodeType.INGREDIENT:
            return self.ingredient.name  # type: ignore[union-attr]
        elif self.node_type == RecipeNodeType.OPERATION:
            return self.operation.op_type.value  # type: ignore[union-attr]
        else:
            return "DISH"

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"node_id": self.node_id, "node_type": self.node_type.value}
        if self.ingredient:
            d["ingredient"] = {
                "name": self.ingredient.name,
                "category": self.ingredient.category.value,
                "is_protein": self.ingredient.is_protein,
                "is_liquid": self.ingredient.is_liquid,
            }
        if self.operation:
            d["operation"] = self.operation.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RecipeNode":
        node_type = RecipeNodeType(d["node_type"])
        ingredient = None
        operation = None
        if "ingredient" in d:
            ing = d["ingredient"]
            ingredient = Ingredient(
                name=ing["name"],
                category=IngredientCategory(ing["category"]),
                is_protein=ing.get("is_protein", False),
                is_liquid=ing.get("is_liquid", False),
            )
        if "operation" in d:
            operation = Operation.from_dict(d["operation"])
        return cls(
            node_id=d["node_id"],
            node_type=node_type,
            ingredient=ingredient,
            operation=operation,
        )


class RecipeDAG:
    """A directed acyclic graph representing a recipe program.

    Edges are stored in the direction of material flow: source → destination
    means "source's output is consumed by destination."  The root node
    (DISH_OUTPUT) has no outgoing edges.

    Construction
    ~~~~~~~~~~~~
    Use add_ingredient(), add_operation(), add_dish_output(), and add_edge()
    to build a DAG programmatically.  Use from_dict() to deserialize from JSON.

    Text serialization
    ~~~~~~~~~~~~~~~~~~
    to_text() produces a CLIP-encodable description.  The operation sequence is
    derived from a topological sort so that operations appear in a valid
    cooking order.
    """

    def __init__(self, dish_name: str, category: DishCategory):
        self.dish_name = dish_name
        self.category = category
        self._nodes: dict[str, RecipeNode] = {}
        # Adjacency list: _adj[src] = list of dst node IDs
        self._adj: dict[str, list[str]] = defaultdict(list)
        # Reverse adjacency: _radj[dst] = list of src node IDs
        self._radj: dict[str, list[str]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_ingredient(self, node_id: str, ingredient: Ingredient) -> "RecipeDAG":
        self._nodes[node_id] = RecipeNode(node_id, RecipeNodeType.INGREDIENT, ingredient=ingredient)
        return self

    def add_operation(self, node_id: str, operation: Operation) -> "RecipeDAG":
        self._nodes[node_id] = RecipeNode(node_id, RecipeNodeType.OPERATION, operation=operation)
        return self

    def add_dish_output(self, node_id: str) -> "RecipeDAG":
        self._nodes[node_id] = RecipeNode(node_id, RecipeNodeType.DISH_OUTPUT)
        return self

    def add_edge(self, src_id: str, dst_id: str) -> "RecipeDAG":
        """Add a directed edge from src to dst (material flows src → dst)."""
        assert src_id in self._nodes, f"Source node '{src_id}' not found"
        assert dst_id in self._nodes, f"Destination node '{dst_id}' not found"
        self._adj[src_id].append(dst_id)
        self._radj[dst_id].append(src_id)
        return self

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def nodes(self) -> list[RecipeNode]:
        return list(self._nodes.values())

    def get_node(self, node_id: str) -> RecipeNode:
        return self._nodes[node_id]

    def successors(self, node_id: str) -> list[str]:
        """Nodes that consume the output of node_id."""
        return self._adj[node_id]

    def predecessors(self, node_id: str) -> list[str]:
        """Nodes whose output is consumed by node_id."""
        return self._radj[node_id]

    def ingredient_nodes(self) -> list[RecipeNode]:
        return [n for n in self._nodes.values() if n.node_type == RecipeNodeType.INGREDIENT]

    def operation_nodes(self) -> list[RecipeNode]:
        return [n for n in self._nodes.values() if n.node_type == RecipeNodeType.OPERATION]

    def dish_output_node(self) -> Optional[RecipeNode]:
        outputs = [n for n in self._nodes.values() if n.node_type == RecipeNodeType.DISH_OUTPUT]
        return outputs[0] if outputs else None

    def ingredient_names(self) -> list[str]:
        return [n.ingredient.name for n in self.ingredient_nodes() if n.ingredient]  # type: ignore[union-attr]

    def operation_types(self) -> list[OperationType]:
        return [n.operation.op_type for n in self.operation_nodes() if n.operation]  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def topological_sort(self) -> list[str]:
        """Return node IDs in topological order (leaves first, root last).

        Uses Kahn's algorithm.  Raises ValueError if a cycle is detected,
        which would mean the DSL constraint that recipes are DAGs is violated.
        """
        in_degree: dict[str, int] = {nid: len(self._radj[nid]) for nid in self._nodes}
        queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order: list[str] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for succ in self._adj[nid]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(order) != len(self._nodes):
            raise ValueError(
                f"RecipeDAG '{self.dish_name}' contains a cycle — not a valid DAG."
            )
        return order

    # ------------------------------------------------------------------
    # Text serialization
    # ------------------------------------------------------------------

    def to_text(self) -> str:
        """Serialize to a natural-language description.

        Format:
            "A dish of {name}. Ingredients: {comma-separated list}.
             Preparation: {topologically ordered operation descriptions}."
        """
        ingredient_list = ", ".join(self.ingredient_names())

        ops_in_order: list[str] = []
        for nid in self.topological_sort():
            node = self._nodes[nid]
            if node.node_type != RecipeNodeType.OPERATION:
                continue
            op = node.operation
            assert op is not None
            # Collect ingredient/operation labels for inputs to this operation
            input_labels = [self._nodes[pred].label() for pred in self._radj[nid]]
            input_str = " and ".join(input_labels) if input_labels else "ingredients"
            ops_in_order.append(
                op.meta.natural_language.format(inputs=input_str)
            )

        prep_text = "; ".join(ops_in_order) if ops_in_order else "combine all ingredients"
        return (
            f"A dish of {self.dish_name}. "
            f"Ingredients: {ingredient_list}. "
            f"Preparation: {prep_text}."
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "dish_name": self.dish_name,
            "category": self.category.value,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [[src, dst] for src, dsts in self._adj.items() for dst in dsts],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "RecipeDAG":
        dag = cls(dish_name=d["dish_name"], category=DishCategory(d["category"]))
        for node_dict in d["nodes"]:
            node = RecipeNode.from_dict(node_dict)
            dag._nodes[node.node_id] = node
        for src, dst in d["edges"]:
            dag._adj[src].append(dst)
            dag._radj[dst].append(src)
        return dag

    @classmethod
    def from_json(cls, s: str) -> "RecipeDAG":
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # Structural validation (not constraint checking — see verifier.py)
    # ------------------------------------------------------------------

    def validate_structure(self) -> list[str]:
        """Check basic DAG structural invariants.  Returns a list of error strings.

        These are structural errors, not cooking logic errors.  Cooking logic
        errors (raw protein, wrong operation order, etc.) are caught by
        constraints/verifier.py.
        """
        errors: list[str] = []

        # Must have exactly one DISH_OUTPUT node
        outputs = [n for n in self._nodes.values() if n.node_type == RecipeNodeType.DISH_OUTPUT]
        if len(outputs) != 1:
            errors.append(f"Expected 1 DISH_OUTPUT node, found {len(outputs)}")

        # No cycles
        try:
            self.topological_sort()
        except ValueError as e:
            errors.append(str(e))

        # DISH_OUTPUT must have no outgoing edges
        for out_node in outputs:
            if self._adj[out_node.node_id]:
                errors.append("DISH_OUTPUT node has outgoing edges")

        # Every ingredient must have at least one outgoing edge (must be used)
        for node in self.ingredient_nodes():
            if not self._adj[node.node_id]:
                errors.append(f"Ingredient '{node.label()}' ({node.node_id}) has no outgoing edges — unused")

        return errors

    def __repr__(self) -> str:
        return (
            f"RecipeDAG(dish='{self.dish_name}', category={self.category.value}, "
            f"ingredients={len(self.ingredient_nodes())}, "
            f"operations={len(self.operation_nodes())})"
        )
