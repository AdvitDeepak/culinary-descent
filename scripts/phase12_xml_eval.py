#!/usr/bin/env python3
"""
Phase 12: Ground-truth evaluation against CURD annotated recipes.

The annotated_recipes/ folder contains 261 XML files with hand-annotated
cooking DAGs in a formal action language:
  cook(ing_in, tool, ing_out, result_desc, "action text")
  mix(ingredients, result, desc)
  cut(ing, result, "cut action")
  combine(...), separate(...), etc.

We:
  1. Parse each XML → ground-truth process sequence + ingredient assignments
  2. Match by title to recipes.json (26 direct matches found)
  3. Look up our constrained LLM encoder output for those 26 recipes
  4. Compare: canonical type precision/recall, ingredient assignment accuracy

This is the first evaluation against human-annotated ground truth rather
than round-trip proxies (SentBERT/ROUGE-L).

Outputs: data/eval/phase12_xml_eval.json
         data/eval/phase12_xml_eval.txt   (human-readable report)
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict

import xml.etree.ElementTree as ET

RECIPE1M   = Path("/home/addeepak/AdvitResearch/cs348k/recipes.json")
XML_DIR    = Path("/home/addeepak/AdvitResearch/cs348k/annotated_recipes")
DAGS_FILE  = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/dags/constrained_dags.jsonl")
OUT        = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/eval")

CANONICAL_43 = [
    "bake","roast","broil","grill","toast",
    "saute","fry","sear","brown",
    "boil","simmer","steam","poach","braise","blanch",
    "chop","dice","slice","mince","grate","peel","crush",
    "mix","stir","whisk","fold","blend","beat","knead","toss",
    "season","coat","brush","drizzle",
    "cool","chill","freeze",
    "rest","marinate",
    "reduce","dissolve","melt","drain",
]
CANONICAL_SET = set(CANONICAL_43)

# ── Mapping from XML action text → canonical type ─────────────────────────────
# Extracted from XML action text strings (5th arg of cook(), or the op name)
ACTION_TO_CANONICAL = {
    # heat verbs
    "bake": "bake", "roast": "roast", "broil": "broil", "grill": "grill",
    "toast": "toast", "saute": "saute", "sauté": "saute", "fry": "fry",
    "sear": "sear", "brown": "brown", "boil": "boil", "simmer": "simmer",
    "steam": "steam", "poach": "poach", "braise": "braise", "blanch": "blanch",
    # prep verbs
    "chop": "chop", "dice": "dice", "slice": "slice", "mince": "mince",
    "grate": "grate", "peel": "peel", "crush": "crush", "cut": "chop",
    "shred": "grate", "julienne": "slice", "halve": "chop",
    # combine / mix
    "mix": "mix", "combine": "mix", "stir": "stir", "whisk": "whisk",
    "fold": "fold", "blend": "blend", "beat": "beat", "knead": "knead",
    "toss": "toss", "mix in": "mix", "stir in": "stir",
    # finish
    "season": "season", "coat": "coat", "brush": "brush", "drizzle": "drizzle",
    # temperature change
    "cool": "cool", "chill": "chill", "freeze": "freeze",
    "refrigerate": "chill", "let cool": "cool",
    # rest
    "rest": "rest", "marinate": "marinate", "let rest": "rest",
    "let stand": "rest", "stand": "rest",
    # other
    "reduce": "reduce", "dissolve": "dissolve", "melt": "melt",
    "drain": "drain", "heat": "saute", "cook": "saute",
    "heat oil": "saute", "heat butter": "saute",
}


def action_text_to_canonical(text: str):
    """Map an XML action text string to our canonical type."""
    t = text.lower().strip().rstrip(".")
    # direct lookup
    if t in ACTION_TO_CANONICAL:
        return ACTION_TO_CANONICAL[t]
    # check if any canonical verb appears in the text
    for verb in CANONICAL_43:
        if re.search(r'\b' + verb + r'\b', t):
            return verb
    # check action map keys
    for key, val in ACTION_TO_CANONICAL.items():
        if key in t:
            return val
    return None


def parse_xml(fpath: Path):
    """
    Parse a CURD XML file into:
      - steps: list of {canonical, ingredients, action_text, original_text}
      - all_ingredients: set of ingredient strings (from create_ing)
    """
    try:
        tree = ET.parse(fpath)
        root = tree.getroot()
    except ET.ParseError:
        # fallback: read raw and parse manually
        content = fpath.read_text(errors='replace')
        content = re.sub(r'&(?!amp;|lt;|gt;|apos;|quot;)', '&amp;', content)
        root = ET.fromstring(content)

    # First pass: collect ingredient definitions
    ing_defs = {}   # ingN -> text
    for line in root.findall('line'):
        ann = line.findtext('annotation', '')
        orig = line.findtext('originaltext', '').strip()
        m = re.match(r'create_ing\((ing\d+),\s*"([^"]+)"\)', ann)
        if m:
            ing_id, ing_text = m.group(1), m.group(2)
            # strip quantities — take last word chunk that's a noun
            # simple heuristic: drop leading digits/fractions/units
            clean = re.sub(r'^[\d\s./½¼¾⅓⅔]+', '', ing_text).strip()
            clean = re.sub(r'^\w+\s+(of\s+)?', '', clean) if len(clean.split()) > 2 else clean
            ing_defs[ing_id] = clean or ing_text

    steps = []
    seen_actions = set()  # deduplicate same action on same original text

    for line in root.findall('line'):
        ann  = line.findtext('annotation', '').strip()
        orig = line.findtext('originaltext', '').strip()

        # cook(ing_in, tool, ing_out, result_desc, "action_text")
        m = re.match(r'cook\((.+)\)', ann)
        if m:
            args = m.group(1)
            # extract quoted action text (last quoted string)
            quoted = re.findall(r'"([^"]*)"', args)
            action_text = quoted[-1] if quoted else ""
            result_desc = quoted[-2] if len(quoted) >= 2 else ""
            canonical = action_text_to_canonical(action_text) or action_text_to_canonical(result_desc)

            # extract input ingredient IDs
            ing_ids = re.findall(r'ing\d+', args.split(',')[0])
            # also look in set notation {ing0, ing1, ...}
            ing_ids += re.findall(r'ing\d+', args)
            ing_ids = list(dict.fromkeys(ing_ids))  # dedup preserve order
            ings = [ing_defs.get(i, i) for i in ing_ids]

            key = (orig[:60], canonical)
            if canonical and key not in seen_actions:
                seen_actions.add(key)
                steps.append({
                    "canonical": canonical,
                    "ingredients": ings[:6],
                    "action_text": action_text,
                    "original_text": orig,
                    "source": "cook",
                })

        # mix(ingredients_set, result, desc)
        m = re.match(r'mix\((.+)\)', ann)
        if m and orig not in [s["original_text"] for s in steps]:
            args = m.group(1)
            ing_ids = re.findall(r'ing\d+', args)
            ings = [ing_defs.get(i, i) for i in ing_ids[:6]]
            key = (orig[:60], "mix")
            if key not in seen_actions:
                seen_actions.add(key)
                steps.append({
                    "canonical": "mix",
                    "ingredients": ings,
                    "action_text": "mix",
                    "original_text": orig,
                    "source": "mix",
                })

        # cut(ing, result, "action")
        m = re.match(r'cut\((.+)\)', ann)
        if m:
            args = m.group(1)
            quoted = re.findall(r'"([^"]*)"', args)
            action_text = quoted[0] if quoted else "chop"
            canonical = action_text_to_canonical(action_text) or "chop"
            ing_ids = re.findall(r'ing\d+', args)
            ings = [ing_defs.get(i, i) for i in ing_ids[:6]]
            key = (orig[:60], canonical)
            if key not in seen_actions:
                seen_actions.add(key)
                steps.append({
                    "canonical": canonical,
                    "ingredients": ings,
                    "action_text": action_text,
                    "original_text": orig,
                    "source": "cut",
                })

    return {
        "steps": steps,
        "all_ingredients": list(ing_defs.values()),
        "n_ing_defs": len(ing_defs),
    }


def compare_sequences(gt_seq, pred_seq):
    """LCS-based sequence similarity."""
    m, n = len(gt_seq), len(pred_seq)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if gt_seq[i-1] == pred_seq[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / n if n > 0 else 0
    recall    = lcs / m if m > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    return {"lcs": lcs, "precision": precision, "recall": recall, "f1": f1,
            "gt_len": m, "pred_len": n}


def ingredient_overlap(gt_ings, pred_ings):
    """Token-level ingredient overlap."""
    gt_tokens  = set(' '.join(gt_ings).lower().split())
    pred_tokens= set(' '.join(pred_ings).lower().split())
    # remove stopwords
    stops = {'and','the','a','an','of','in','with','to','for','or','cup','cups',
             'tablespoon','tablespoons','teaspoon','teaspoons','pound','pounds',
             'ounce','ounces','large','small','medium','fresh','dried','ground'}
    gt_tokens   -= stops
    pred_tokens -= stops
    if not gt_tokens:
        return 1.0
    return len(gt_tokens & pred_tokens) / len(gt_tokens)


def main():
    print("Loading recipes.json...", flush=True)
    with open(RECIPE1M) as f:
        recipes = json.load(f)
    title_to_recipe = {r.get('title','').lower().strip(): r for r in recipes}
    id_to_recipe    = {r['id']: r for r in recipes}
    print(f"  {len(recipes):,} recipes")

    print("Loading constrained DAGs...", flush=True)
    dag_by_id = {}
    with open(DAGS_FILE) as f:
        for line in f:
            d = json.loads(line)
            dag_by_id[d['id']] = d
    print(f"  {len(dag_by_id):,} DAGs")

    # ── Match XMLs to recipes ─────────────────────────────────────────────────
    print("\nMatching XMLs to recipes...", flush=True)
    matched = []
    unmatched = []
    for fname in sorted(os.listdir(XML_DIR)):
        if not fname.endswith('.xml'):
            continue
        title_guess = fname.replace('.rcp_tagged.xml','').replace('-',' ').lower()
        recipe = title_to_recipe.get(title_guess)
        fpath  = XML_DIR / fname
        gt     = parse_xml(fpath)
        if recipe and recipe['id'] in dag_by_id and gt['steps']:
            matched.append({
                "fname":    fname,
                "recipe":   recipe,
                "dag":      dag_by_id[recipe['id']],
                "gt":       gt,
            })
        else:
            unmatched.append(fname)
    print(f"  Matched: {len(matched)} / {len(matched)+len(unmatched)}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = []
    type_tp = type_fp = type_fn = 0
    seq_f1s = []
    ing_overlaps = []

    report_lines = []
    report_lines.append("GROUND-TRUTH EVALUATION: Constrained Encoder vs CURD Annotations")
    report_lines.append("=" * 72)
    report_lines.append(f"Matched recipes: {len(matched)}")
    report_lines.append("")

    for item in matched:
        recipe = item["recipe"]
        gt     = item["gt"]
        pred   = item["dag"]

        gt_seq   = [s["canonical"] for s in gt["steps"] if s["canonical"] in CANONICAL_SET]
        pred_seq = [s["canonical"] for s in pred["steps"]]

        seq_sim  = compare_sequences(gt_seq, pred_seq)
        seq_f1s.append(seq_sim["f1"])

        # Per-step type accuracy: for each GT step, is it in the prediction?
        gt_types   = set(gt_seq)
        pred_types = set(pred_seq)
        tp = len(gt_types & pred_types)
        fp = len(pred_types - gt_types)
        fn = len(gt_types - pred_types)
        type_tp += tp; type_fp += fp; type_fn += fn

        # Ingredient overlap (whole-recipe level)
        gt_ings   = gt["all_ingredients"]
        pred_ings = [ing for s in pred["steps"] for ing in s.get("ingredients", [])]
        ing_ov    = ingredient_overlap(gt_ings, pred_ings)
        ing_overlaps.append(ing_ov)

        # Per-step comparison
        step_comparisons = []
        for gs in gt["steps"]:
            gc = gs["canonical"]
            if gc not in CANONICAL_SET:
                continue
            # find closest matching pred step by canonical type
            pred_match = next((ps for ps in pred["steps"] if ps["canonical"] == gc), None)
            step_comparisons.append({
                "gt_canonical":   gc,
                "gt_action_text": gs["action_text"],
                "gt_original":    gs["original_text"][:100],
                "gt_ingredients": gs["ingredients"],
                "pred_canonical": pred_match["canonical"] if pred_match else "MISSING",
                "pred_ingredients": pred_match.get("ingredients", []) if pred_match else [],
                "type_match": pred_match is not None,
            })

        result = {
            "title":     recipe.get("title"),
            "id":        recipe["id"],
            "gt_seq":    gt_seq,
            "pred_seq":  pred_seq,
            "seq_sim":   seq_sim,
            "ing_overlap": ing_ov,
            "type_tp": tp, "type_fp": fp, "type_fn": fn,
            "steps":     step_comparisons,
        }
        results.append(result)

        # Report block
        report_lines.append(f"RECIPE: {recipe.get('title')}")
        report_lines.append(f"  GT   sequence: {' → '.join(gt_seq)}")
        report_lines.append(f"  Pred sequence: {' → '.join(pred_seq)}")
        report_lines.append(f"  Seq F1: {seq_sim['f1']:.2f}  "
                             f"(P={seq_sim['precision']:.2f} R={seq_sim['recall']:.2f})  "
                             f"Ing overlap: {ing_ov:.2f}")
        # Flag mismatches
        missing = [t for t in gt_types if t not in pred_types]
        extra   = [t for t in pred_types if t not in gt_types]
        if missing: report_lines.append(f"  MISSING from pred: {missing}")
        if extra:   report_lines.append(f"  EXTRA in pred:     {extra}")
        report_lines.append("")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    prec = type_tp / (type_tp + type_fp) if (type_tp + type_fp) > 0 else 0
    rec  = type_tp / (type_tp + type_fn) if (type_tp + type_fn) > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    avg_seq_f1  = sum(seq_f1s) / len(seq_f1s) if seq_f1s else 0
    avg_ing_ov  = sum(ing_overlaps) / len(ing_overlaps) if ing_overlaps else 0

    summary = {
        "n_matched":        len(matched),
        "type_precision":   round(prec, 4),
        "type_recall":      round(rec, 4),
        "type_f1":          round(f1, 4),
        "avg_seq_lcs_f1":   round(avg_seq_f1, 4),
        "avg_ing_overlap":  round(avg_ing_ov, 4),
        "recipes":          results,
    }

    print(f"\n{'='*60}")
    print("GROUND-TRUTH EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Matched recipes:          {len(matched)}")
    print(f"  Canonical type precision: {prec:.1%}")
    print(f"  Canonical type recall:    {rec:.1%}")
    print(f"  Canonical type F1:        {f1:.1%}")
    print(f"  Avg sequence LCS F1:      {avg_seq_f1:.1%}")
    print(f"  Avg ingredient overlap:   {avg_ing_ov:.1%}")
    print()
    print("  Comparison notes:")
    print("  - Type precision: what fraction of encoder's predicted types are GT-valid")
    print("  - Type recall:    what fraction of GT process types the encoder captures")
    print("  - Seq F1:         LCS alignment of full canonical sequence (order matters)")

    report_lines.insert(3, f"Type precision:   {prec:.1%}")
    report_lines.insert(4, f"Type recall:      {rec:.1%}")
    report_lines.insert(5, f"Type F1:          {f1:.1%}")
    report_lines.insert(6, f"Avg sequence F1:  {avg_seq_f1:.1%}")
    report_lines.insert(7, f"Avg ing overlap:  {avg_ing_ov:.1%}")
    report_lines.insert(8, "")

    with open(OUT / "phase12_xml_eval.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(OUT / "phase12_xml_eval.txt", "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n  Saved → {OUT}/phase12_xml_eval.json")
    print(f"  Saved → {OUT}/phase12_xml_eval.txt")


if __name__ == "__main__":
    main()
