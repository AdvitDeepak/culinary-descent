#!/usr/bin/env python3
"""
DSL Coverage Evaluation on Recipe1M
--------------------------------------
Metric 1 of 2 for our north-star evaluation:

  "What % of Recipe1M recipes parse into valid DSL programs?"

This script samples recipes from Recipe1M, calls the LLM parser
(Perplexity Sonar) on each, runs the constraint verifier, and reports:

  - Parse success rate    : % that produce a RecipeDAG (not malformed JSON / OOV)
  - Validity rate         : % of parsed DAGs that pass all 8 constraint predicates
  - Category distribution : how many recipes fall in each of our 5 categories
  - Violation breakdown   : which constraint predicates fire most often
  - Out-of-scope rate     : % outside our 5 dish categories

Usage
-----
    export PERPLEXITY_API_KEY=pplx-...
    python scripts/evaluate_coverage.py \\
        --recipe1m-path /path/to/recipe1m \\
        --n 200 \\
        --out data/coverage_results.json

    # Dry-run with 10 recipes (no API cost):
    python scripts/evaluate_coverage.py --recipe1m-path /path/to/recipe1m --n 10

Recipe1M format expected
------------------------
    /path/to/recipe1m/
        layer1.json      — list of {id, partition, title, ingredients, instructions}
        det_ingrs.json   — parallel list of {id, ingredients, valid}
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from culinary_descent.parser.llm_parser import LLMParser
from culinary_descent.constraints.verifier import verify


def load_recipe1m(recipe1m_path: Path, partition: str = "test") -> list[dict]:
    """Load Recipe1M layer1.json, filtered to the given partition."""
    layer1_path = recipe1m_path / "layer1.json"
    if not layer1_path.exists():
        raise FileNotFoundError(f"layer1.json not found at {layer1_path}")

    print(f"Loading {layer1_path} ...")
    with open(layer1_path) as f:
        data = json.load(f)

    recipes = [r for r in data if r.get("partition") == partition]
    print(f"  {len(recipes):,} recipes in '{partition}' partition")
    return recipes


def sample_recipes(recipes: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Uniform random sample of n recipes."""
    import random
    rng = random.Random(seed)
    if n >= len(recipes):
        return recipes
    return rng.sample(recipes, n)


def run_coverage_eval(
    recipe1m_path: Path,
    n: int,
    partition: str,
    delay_between_calls: float,
    verbose: bool,
) -> dict:
    recipes = load_recipe1m(recipe1m_path, partition)
    sample = sample_recipes(recipes, n)

    parser = LLMParser(model="sonar")

    results = {
        "n_attempted": len(sample),
        "n_parse_success": 0,
        "n_out_of_scope": 0,
        "n_parse_failed": 0,
        "n_valid": 0,
        "n_invalid": 0,
        "categories": Counter(),
        "violation_counts": Counter(),
        "failure_reasons": Counter(),
        "per_recipe": [],
    }

    for i, recipe in enumerate(sample):
        title = recipe.get("title", "")
        ingredients = [ing["text"] for ing in recipe.get("ingredients", [])]
        instructions = [step["text"] for step in recipe.get("instructions", [])]

        if verbose:
            print(f"\n[{i+1}/{len(sample)}] {title[:60]}")

        result = parser.parse(title=title, ingredients=ingredients, instructions=instructions)

        entry = {
            "recipe_id": recipe.get("id", ""),
            "title": title,
            "category": result.category,
            "success": result.success,
            "out_of_scope": result.out_of_scope,
            "failure_reason": result.failure_reason,
            "is_valid": None,
            "violations": [],
        }

        if result.out_of_scope:
            results["n_out_of_scope"] += 1
            if verbose:
                print(f"  → out_of_scope")

        elif not result.success:
            results["n_parse_failed"] += 1
            results["failure_reasons"][result.failure_reason] += 1
            if verbose:
                print(f"  → parse_failed: {result.failure_reason}")

        else:
            results["n_parse_success"] += 1
            results["categories"][result.category] += 1

            vresult = verify(result.dag)
            entry["is_valid"] = vresult.is_valid
            entry["violations"] = [v.rule for v in vresult.violations]

            if vresult.is_valid:
                results["n_valid"] += 1
                if verbose:
                    print(f"  → VALID  ({result.category})")
            else:
                results["n_invalid"] += 1
                for v in vresult.violations:
                    results["violation_counts"][v.rule] += 1
                if verbose:
                    viols = ", ".join(v.rule for v in vresult.violations[:3])
                    print(f"  → INVALID ({result.category}): {viols}")

        results["per_recipe"].append(entry)

        if i < len(sample) - 1:
            time.sleep(delay_between_calls)

    return results


def print_summary(results: dict):
    n = results["n_attempted"]
    n_in_scope = results["n_parse_success"] + results["n_invalid"] + results["n_valid"]

    print("\n" + "=" * 60)
    print("DSL COVERAGE EVALUATION — SUMMARY")
    print("=" * 60)
    print(f"Recipes attempted:     {n}")
    print(f"Out-of-scope:          {results['n_out_of_scope']:4d}  ({100*results['n_out_of_scope']/n:.1f}%)")
    print(f"Parse failed:          {results['n_parse_failed']:4d}  ({100*results['n_parse_failed']/n:.1f}%)")
    print(f"Parse success:         {results['n_parse_success']:4d}  ({100*results['n_parse_success']/n:.1f}%)")

    if results["n_parse_success"] > 0:
        n_ps = results["n_parse_success"]
        print(f"  → Constraint-valid:  {results['n_valid']:4d}  ({100*results['n_valid']/n_ps:.1f}% of parsed)")
        print(f"  → Constraint-invalid:{results['n_invalid']:4d}  ({100*results['n_invalid']/n_ps:.1f}% of parsed)")

    if results["categories"]:
        print("\nCategory distribution (of parsed recipes):")
        for cat, count in results["categories"].most_common():
            print(f"  {cat:<12} {count:4d}")

    if results["violation_counts"]:
        print("\nViolation breakdown (of invalid parsed recipes):")
        for rule, count in results["violation_counts"].most_common():
            print(f"  {rule:<35} {count:4d}")

    if results["failure_reasons"]:
        print("\nParse failure reasons:")
        for reason, count in results["failure_reasons"].most_common():
            print(f"  {str(reason)[:55]:<55} {count:4d}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="DSL coverage evaluation on Recipe1M")
    parser.add_argument("--recipe1m-path", required=True,
                        help="Path to Recipe1M data directory (layer1.json + det_ingrs.json)")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of recipes to sample (default: 200)")
    parser.add_argument("--partition", default="test",
                        help="Recipe1M partition to sample from (default: test)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between API calls (default: 0.5)")
    parser.add_argument("--out", default="data/coverage_results.json",
                        help="Output JSON path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results = run_coverage_eval(
        recipe1m_path=Path(args.recipe1m_path),
        n=args.n,
        partition=args.partition,
        delay_between_calls=args.delay,
        verbose=args.verbose,
    )

    print_summary(results)

    out_path = Path(__file__).parent.parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Don't serialize Counter objects directly
    results["categories"] = dict(results["categories"])
    results["violation_counts"] = dict(results["violation_counts"])
    results["failure_reasons"] = dict(results["failure_reasons"])

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
