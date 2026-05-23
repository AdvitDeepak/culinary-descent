"""
LLM-based Recipe Parser (Perplexity Sonar)
------------------------------------------
Converts free-text Recipe1M recipes into typed RecipeDAG programs.

The parser is the "compiler front-end" for our DSL evaluation:
  NL recipe text → RecipeDAG → constraint verifier → coverage/recall metrics

Design choices
~~~~~~~~~~~~~~
- We use Perplexity Sonar because it has strong instruction-following for 
structured output tasks (and I also have lots of free credits loll)
- The prompt provides the FULL vocabulary so the LLM maps to exact tokens;
  it does not generate free-form ingredient/operation names.
- Out-of-vocabulary recipes are marked "out_of_scope" — this is deliberate:
  we want to MEASURE vocabulary coverage, not hide failures behind fuzzy matching.
- A recipe is "parse-failed" (not "out_of_scope") if the LLM returns malformed
  JSON or an invalid DAG structure.  These failures are tracked separately.

Usage
-----
    export PERPLEXITY_API_KEY=pplx-...
    parser = LLMParser()
    result = parser.parse(title="Spaghetti Carbonara",
                          ingredients=["spaghetti", "eggs", "guanciale", "parmesan"],
                          instructions=["Boil pasta. Fry guanciale. Whisk eggs..."])
    if result.dag:
        print(result.dag.to_text())
    else:
        print(result.failure_reason)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from ..dsl.recipe_dag import RecipeDAG
from ..dsl.vocabulary import (
    INGREDIENTS,
    OPERATIONS,
    DishCategory,
    IngredientCategory,
    OperationType,
    Ingredient,
)


# ---------------------------------------------------------------------------
# Vocabulary context (injected into every prompt)
# ---------------------------------------------------------------------------

def _build_vocab_block() -> str:
    """Build the vocabulary section of the system prompt."""
    ing_by_cat: dict[str, list[str]] = {}
    for key, ing in INGREDIENTS.items():
        cat = ing.category.value.upper()
        ing_by_cat.setdefault(cat, []).append(key)

    ing_lines = []
    for cat, keys in sorted(ing_by_cat.items()):
        ing_lines.append(f"  {cat}: {', '.join(sorted(keys))}")

    op_by_group: dict[str, list[str]] = {}
    for op, meta in OPERATIONS.items():
        group = meta.group.value.upper()
        op_by_group.setdefault(group, []).append(op.value)

    op_lines = []
    for group, ops in sorted(op_by_group.items()):
        op_lines.append(f"  {group}: {', '.join(sorted(ops))}")

    return (
        "ALLOWED INGREDIENTS (use exact key names):\n"
        + "\n".join(ing_lines)
        + "\n\nALLOWED OPERATIONS (use exact values):\n"
        + "\n".join(op_lines)
        + "\n\nALLOWED CATEGORIES: pasta, eggs, salad, stir_fry, soup"
    )


_VOCAB_BLOCK = _build_vocab_block()

_SYSTEM_PROMPT = f"""You are a recipe DAG parser. Convert free-text recipes into a structured RecipeDAG JSON.

{_VOCAB_BLOCK}

OUTPUT FORMAT — respond with ONLY a JSON object, no prose, no markdown:
{{
  "dish_name": "...",
  "category": "pasta|eggs|salad|stir_fry|soup|out_of_scope",
  "nodes": [
    {{"node_id": "n0", "node_type": "ingredient", "ingredient": {{"name": "...", "category": "protein|vegetable|allium|grain|dairy|fat|seasoning|liquid|fruit", "is_protein": false, "is_liquid": false}}}},
    {{"node_id": "n1", "node_type": "operation", "operation": {{"op_type": "saute", "params": {{}}}}}},
    {{"node_id": "nN", "node_type": "dish_output"}}
  ],
  "edges": [["n0", "n1"], ["n1", "nN"]]
}}

RULES:
1. Map every recipe ingredient to the nearest ALLOWED INGREDIENT key.
   If an ingredient has no reasonable match (exotic spice, unusual cut), omit it.
   If >50% of main ingredients have no match, set category to "out_of_scope".
2. Map every cooking verb to the nearest ALLOWED OPERATION value.
3. Infer DAG edges from instruction order: ingredients flow into their first operation,
   operations flow into subsequent operations, all paths must reach dish_output.
4. Every ingredient node must have exactly one outgoing edge.
5. Every operation node (except the one feeding dish_output) must have one outgoing edge.
6. Exactly one dish_output node at the end.
7. The ingredient's "name" field should be the human-readable name (spaces allowed),
   "category" should match IngredientCategory enum values exactly.
8. If the recipe category is not pasta/eggs/salad/stir_fry/soup, set to "out_of_scope"
   and return minimal nodes (just dish_output is fine for out_of_scope).
"""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    """Result of attempting to parse a recipe into a RecipeDAG."""
    dag: Optional[RecipeDAG] = None
    category: Optional[str] = None          # detected category (even if out_of_scope)
    failure_reason: Optional[str] = None    # None means success
    raw_response: str = ""

    @property
    def success(self) -> bool:
        return self.dag is not None

    @property
    def out_of_scope(self) -> bool:
        return self.category == "out_of_scope"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class LLMParser:
    """Parse free-text recipes into RecipeDAG using Perplexity Sonar.

    Parameters
    ----------
    model:
        Perplexity model name. "sonar" is faster; "sonar-pro" is more accurate.
    max_retries:
        Number of API retries on transient errors.
    retry_delay:
        Seconds to wait between retries.
    """

    def __init__(
        self,
        model: str = "sonar",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        api_key = os.environ.get("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("Set PERPLEXITY_API_KEY environment variable")

        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai",
        )
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def parse(
        self,
        title: str,
        ingredients: list[str],
        instructions: list[str],
    ) -> ParseResult:
        """Parse a recipe into a RecipeDAG.

        Parameters
        ----------
        title:
            Recipe title (e.g. "Spaghetti Carbonara").
        ingredients:
            List of ingredient strings as they appear in the recipe
            (e.g. ["2 cups flour", "1/2 tsp salt"]).
        instructions:
            List of instruction strings, one per step.
        """
        user_msg = self._build_user_message(title, ingredients, instructions)
        raw = self._call_api(user_msg)
        if raw is None:
            return ParseResult(failure_reason="api_error", raw_response="")

        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_user_message(
        self,
        title: str,
        ingredients: list[str],
        instructions: list[str],
    ) -> str:
        ing_block = "\n".join(f"  - {i}" for i in ingredients)
        instr_block = "\n".join(f"  {j+1}. {s}" for j, s in enumerate(instructions))
        return (
            f"TITLE: {title}\n\n"
            f"INGREDIENTS:\n{ing_block}\n\n"
            f"INSTRUCTIONS:\n{instr_block}"
        )

    def _call_api(self, user_msg: str) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.0,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    return None
        return None

    def _parse_response(self, raw: str) -> ParseResult:
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return ParseResult(failure_reason=f"json_parse_error: {e}", raw_response=raw)

        category = data.get("category", "out_of_scope")
        if category == "out_of_scope":
            return ParseResult(category="out_of_scope", failure_reason=None, raw_response=raw)

        try:
            cat_enum = DishCategory(category)
        except ValueError:
            return ParseResult(
                category=category,
                failure_reason=f"unknown_category: {category}",
                raw_response=raw,
            )

        # Normalize ingredient node data to match expected vocabulary attributes
        for node in data.get("nodes", []):
            if node.get("node_type") == "ingredient" and "ingredient" in node:
                ing = node["ingredient"]
                # Resolve name and attributes from vocabulary if key matches
                vocab_key = ing.get("name", "").lower().replace(" ", "_")
                if vocab_key in INGREDIENTS:
                    v = INGREDIENTS[vocab_key]
                    ing["name"] = v.name
                    ing["category"] = v.category.value
                    ing["is_protein"] = v.is_protein
                    ing["is_liquid"] = v.is_liquid
                else:
                    # Use LLM-provided values but validate category
                    cat_str = ing.get("category", "vegetable")
                    try:
                        IngredientCategory(cat_str)
                    except ValueError:
                        ing["category"] = "vegetable"
                    ing.setdefault("is_protein", False)
                    ing.setdefault("is_liquid", False)

            if node.get("node_type") == "operation" and "operation" in node:
                op = node["operation"]
                op_val = op.get("op_type", "")
                try:
                    OperationType(op_val)
                except ValueError:
                    # LLM sometimes returns a group name (e.g. "finish") instead of
                    # a specific op — map to the most common op in that group
                    _group_defaults = {
                        "prep": "chop", "dry_heat": "saute", "moist_heat": "simmer",
                        "combine": "mix", "finish": "plate",
                    }
                    fallback = _group_defaults.get(op_val.lower())
                    if fallback:
                        op["op_type"] = fallback
                    else:
                        return ParseResult(
                            category=category,
                            failure_reason=f"unknown_op_type: {op_val}",
                            raw_response=raw,
                        )

        try:
            dag = RecipeDAG.from_dict(data)
        except Exception as e:
            return ParseResult(
                category=category,
                failure_reason=f"dag_construction_error: {e}",
                raw_response=raw,
            )

        return ParseResult(dag=dag, category=category, raw_response=raw)
