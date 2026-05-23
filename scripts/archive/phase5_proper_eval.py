#!/usr/bin/env python3
"""
Phase 5: Proper evaluation of RecipeDAG information capture.

Metrics:
  1. Process-sequence LCS  — round-trip: encode NL → DAG → decode → re-encode → compare seqs
  2. Ingredient coverage   — what % of original ingredients appear in DAG
  3. Arg recall            — what % of time/temp mentions in text are captured in DAG args
  4. SentenceBERT cosine   — semantic similarity of original vs decoded NL (vs ROUGE-L)
  5. LLM judge (vLLM)      — binary YES/NO on (orig, decoded) with adversarial baselines

Adversarial baselines for judge validation:
  - True positive:    (orig_NL,  decoded_NL from same recipe)
  - Easy negative:    (orig_NL,  decoded_NL from random different recipe)
  - Shuffled negative:(orig_NL,  decoded_NL but process steps in reverse order)

Outputs:
  data/eval/proper_eval_results.json
  data/eval/judge_results.json
  data/eval/proper_eval_report.txt
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DAGS_DIR  = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/dags")
RECIPE1M  = Path("/home/addeepak/AdvitResearch/cs348k/recipes.json")
VOCAB_DIR = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/vocab_analysis")
OUT       = Path("/home/addeepak/AdvitResearch/cs348k/culinary-descent/data/eval")
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_N       = 2000   # for structural metrics
JUDGE_N_EACH   = 75     # per condition (true pos, easy neg, shuffled neg)
RANDOM_SEED    = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── taxonomy (needed for re-encoding) ────────────────────────────────────────
TAXONOMY = [
    ("bake","dry_heat"),("roast","dry_heat"),("broil","dry_heat"),
    ("grill","dry_heat"),("toast","dry_heat"),("saute","fat_heat"),
    ("fry","fat_heat"),("sear","fat_heat"),("brown","fat_heat"),
    ("boil","moist_heat"),("simmer","moist_heat"),("steam","moist_heat"),
    ("poach","moist_heat"),("braise","moist_heat"),("blanch","moist_heat"),
    ("chop","prep"),("dice","prep"),("slice","prep"),("mince","prep"),
    ("grate","prep"),("peel","prep"),("crush","prep"),("mix","combine"),
    ("stir","combine"),("whisk","combine"),("fold","combine"),("blend","combine"),
    ("beat","combine"),("knead","combine"),("toss","combine"),("season","apply"),
    ("coat","apply"),("brush","apply"),("drizzle","apply"),("cool","rest"),
    ("chill","rest"),("freeze","rest"),("rest","rest"),("marinate","rest"),
    ("reduce","rest"),("dissolve","rest"),("melt","rest"),("drain","rest"),
]
CANONICAL_NAMES = [t[0] for t in TAXONOMY]
OVERRIDE_MAP = {
    "refrigerate":"chill","thaw":"cool","defrost":"cool",
    "reheat":"simmer","microwave":"bake","pressure":"boil",
    "stir-fry":"saute","stir fry":"saute","pan-fry":"fry","pan fry":"fry",
    "deep-fry":"fry","deep fry":"fry",
}
GENERIC_VERBS = {
    "be","have","do","make","use","get","go","come","take","put","let","set",
    "keep","give","want","need","find","look","seem","try","call","tell","ask",
    "work","feel","leave","bring","move","start","continue","ensure","allow",
    "remain","result","yield","note","check","watch","test","taste","adjust",
    "enjoy","serve","prepare","follow","read","turn","place","add","remove",
    "transfer","pour","top","fill","line","arrange","layer",
}
_TIME_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(hour|hr|minute|min|second|sec)s?', re.I)
_TEMP_RE = re.compile(r'(\d{2,4})\s*(?:degrees?|°)\s*([CF])', re.I)

TEMPLATES = {
    "bake":"Bake {ings}{args}.","roast":"Roast {ings}{args} in the oven.",
    "broil":"Broil {ings}{args}.","grill":"Grill {ings}{args}.",
    "toast":"Toast {ings}{args}.","saute":"Sauté {ings}{args} in a pan.",
    "fry":"Fry {ings}{args}.","sear":"Sear {ings}{args}.",
    "brown":"Brown {ings}{args}.","boil":"Boil {ings}{args}.",
    "simmer":"Simmer {ings}{args}.","steam":"Steam {ings}{args}.",
    "poach":"Poach {ings}{args}.","braise":"Braise {ings}{args}.",
    "blanch":"Blanch {ings}{args}.","chop":"Chop {ings}.",
    "dice":"Dice {ings}{args}.","slice":"Slice {ings}{args}.",
    "mince":"Mince {ings}.","grate":"Grate {ings}.",
    "peel":"Peel {ings}.","crush":"Crush {ings}.",
    "mix":"Mix {ings}{args}.","stir":"Stir {ings}{args}.",
    "whisk":"Whisk {ings}{args}.","fold":"Fold in {ings}.",
    "blend":"Blend {ings}{args}.","beat":"Beat {ings}{args}.",
    "knead":"Knead {ings}{args}.","toss":"Toss {ings}.",
    "season":"Season {ings}.","coat":"Coat {ings}.",
    "brush":"Brush {ings}.","drizzle":"Drizzle {ings}.",
    "cool":"Cool {ings}{args}.","chill":"Refrigerate {ings}{args}.",
    "freeze":"Freeze {ings}{args}.","rest":"Rest {ings}{args}.",
    "marinate":"Marinate {ings}{args}.","reduce":"Reduce {ings}.",
    "dissolve":"Dissolve {ings}.","melt":"Melt {ings}.",
    "drain":"Drain {ings}.",
}

def short_ing(name: str) -> str:
    """Take first component before comma for cleaner display."""
    return name.split(",")[0].strip()

def format_args(args: dict) -> str:
    parts = []
    if "temp_c" in args:
        f = round(args["temp_c"] * 9/5 + 32)
        parts.append(f"at {f}°F")
    if "duration_min" in args:
        m = args["duration_min"]
        if m >= 60:
            h = int(m // 60); rem = int(m % 60)
            parts.append(f"for {h}h {rem}m" if rem else f"for {h}hr")
        else:
            parts.append(f"for {int(m)} min")
    return " " + ", ".join(parts) if parts else ""

def decode_dag(dag: dict) -> str:
    node_map = {n["id"]: n for n in dag["nodes"]}
    proc_nodes = sorted(
        [n for n in dag["nodes"] if n["type"] == "process"],
        key=lambda n: n["step_idx"]
    )
    proc_inputs: dict[str, list[str]] = defaultdict(list)
    for edge in dag["edges"]:
        if edge["label"] in ("input", "input_fallback"):
            src = node_map.get(edge["src"])
            if src and src["type"] == "ingredient":
                proc_inputs[edge["dst"]].append(short_ing(src["name"]))
    sentences = []
    for proc in proc_nodes:
        canon   = proc["canonical"]
        tmpl    = TEMPLATES.get(canon, "Process {ings}{args}.")
        ings    = proc_inputs.get(proc["id"], [])
        ing_str = ", ".join(ings[:3]) if ings else "ingredients"
        if len(ings) > 3:
            ing_str += " and more"
        arg_str = format_args(proc.get("args", {}))
        sentences.append(tmpl.format(ings=ing_str, args=arg_str))
    return " ".join(sentences)

def decode_dag_shuffled(dag: dict) -> str:
    """Decode with process nodes in reverse order (adversarial negative)."""
    node_map = {n["id"]: n for n in dag["nodes"]}
    proc_nodes = sorted(
        [n for n in dag["nodes"] if n["type"] == "process"],
        key=lambda n: n["step_idx"]
    )
    proc_nodes = list(reversed(proc_nodes))  # reverse order
    proc_inputs: dict[str, list[str]] = defaultdict(list)
    for edge in dag["edges"]:
        if edge["label"] in ("input", "input_fallback"):
            src = node_map.get(edge["src"])
            if src and src["type"] == "ingredient":
                proc_inputs[edge["dst"]].append(short_ing(src["name"]))
    sentences = []
    for proc in proc_nodes:
        canon   = proc["canonical"]
        tmpl    = TEMPLATES.get(canon, "Process {ings}{args}.")
        ings    = proc_inputs.get(proc["id"], [])
        ing_str = ", ".join(ings[:3]) if ings else "ingredients"
        if len(ings) > 3:
            ing_str += " and more"
        arg_str = format_args(proc.get("args", {}))
        sentences.append(tmpl.format(ings=ing_str, args=arg_str))
    return " ".join(sentences)


# ── LCS helper ────────────────────────────────────────────────────────────────
def lcs_len(a: list, b: list) -> int:
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    if m < n: a, b, m, n = b, a, n, m
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j-1] + 1 if a[i-1] == b[j-1] else max(curr[j-1], prev[j])
        prev = curr
    return prev[n]

def lcs_score(a: list, b: list) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return lcs_len(a, b) / max(len(a), len(b))


# ── Arg recall from raw text ──────────────────────────────────────────────────
def extract_args_from_text(text: str) -> dict:
    args = {}
    mins = 0.0
    for m in _TIME_RE.finditer(text):
        val, unit = float(m.group(1)), m.group(2).lower()
        if unit.startswith("hour") or unit.startswith("hr"): mins += val * 60
        elif unit.startswith("min"): mins += val
        elif unit.startswith("sec"): mins += val / 60
    if mins: args["duration_min"] = round(mins, 1)
    m = _TEMP_RE.search(text)
    if m:
        val, scale = float(m.group(1)), m.group(2).upper()
        args["temp_c"] = round(val if scale == "C" else (val-32)*5/9, 1)
    return args


# ── spaCy + BGE re-encoder (for round-trip) ───────────────────────────────────
print("Loading spaCy + BGE for re-encoding...", flush=True)
NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
BGE = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
canonical_embs = np.load(VOCAB_DIR / "process_canonical_embs.npy")
verb_cache: dict[str, str] = {}

def text_to_process_seq(text: str) -> list[str]:
    """Extract process type sequence from NL text."""
    doc = NLP(text)
    seq = []
    for token in doc:
        if token.pos_ == "VERB" and not token.is_stop:
            lemma = token.lemma_.lower().strip()
            if len(lemma) < 3 or not lemma.isalpha() or lemma in GENERIC_VERBS:
                continue
            if lemma in OVERRIDE_MAP:
                seq.append(OVERRIDE_MAP[lemma])
                continue
            if lemma in CANONICAL_NAMES:
                seq.append(lemma)
                continue
            if lemma not in verb_cache:
                emb = BGE.encode([lemma], normalize_embeddings=True, convert_to_numpy=True)[0]
                sims = canonical_embs @ emb
                verb_cache[lemma] = CANONICAL_NAMES[int(np.argmax(sims))]
            seq.append(verb_cache[lemma])
    return seq


# ── SentenceBERT cosine similarity ────────────────────────────────────────────
SBERT = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

def batch_cosine_sim(texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    embs_a = SBERT.encode(texts_a, batch_size=256, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    embs_b = SBERT.encode(texts_b, batch_size=256, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    return (embs_a * embs_b).sum(axis=1)


# ── LLM judge (vLLM) ─────────────────────────────────────────────────────────
JUDGE_MODEL    = "Qwen/Qwen2.5-3B-Instruct"
JUDGE_GPU_FRAC = 0.55   # leave room for BGE/SBERT already loaded

JUDGE_PROMPT = """\
You are evaluating whether two recipe descriptions are semantically equivalent — \
i.e., would following either recipe produce approximately the same dish with approximately \
the same key steps and ingredients?

Focus on:
- Are the main ingredients the same?
- Are the cooking methods (baking, frying, simmering, etc.) the same?
- Are the steps in roughly the same order?
- Are key parameters (temperature, cook time) compatible?

Ignore differences in phrasing, narrative style, or level of detail.

Recipe A:
{recipe_a}

Recipe B:
{recipe_b}

Answer with a single word only: YES or NO"""

def build_judge_prompt(a: str, b: str) -> str:
    # Truncate to keep prompts manageable
    return JUDGE_PROMPT.format(recipe_a=a[:800], recipe_b=b[:800])

def run_judge_batch(pairs: list[tuple[str, str, str]]) -> list[dict]:
    """
    pairs: list of (label, text_a, text_b)
    Returns list of {label, answer, prompt}

    NOTE: Call this AFTER releasing BGE/SBERT GPU memory.
    """
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    from vllm import LLM, SamplingParams

    print(f"\nLoading judge model {JUDGE_MODEL}...", flush=True)
    llm = LLM(
        model=JUDGE_MODEL,
        max_model_len=2048,
        gpu_memory_utilization=JUDGE_GPU_FRAC,
        dtype="bfloat16",
    )
    params = SamplingParams(temperature=0, max_tokens=8)

    prompts = []
    for label, a, b in pairs:
        # Format as chat
        messages = [
            {"role": "system", "content": "You answer with a single word: YES or NO."},
            {"role": "user",   "content": build_judge_prompt(a, b)},
        ]
        # Use tokenizer chat template
        tok = llm.get_tokenizer()
        prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_str)

    print(f"  Running {len(prompts)} judge queries in batch...", flush=True)
    outputs = llm.generate(prompts, params)

    results = []
    for (label, a, b), out in zip(pairs, outputs):
        raw = out.outputs[0].text.strip().upper()
        answer = "YES" if raw.startswith("Y") else "NO"
        results.append({"label": label, "answer": answer, "raw": raw})

    # Free GPU memory before returning
    import gc, torch
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading DAGs...", flush=True)
    dags = []
    with open(DAGS_DIR / "train_dags.jsonl") as f:
        for line in f:
            dags.append(json.loads(line))
    print(f"  {len(dags):,} DAGs")

    print("Loading original recipes...", flush=True)
    with open(RECIPE1M) as f:
        all_recipes = json.load(f)
    recipe_by_id = {r["id"]: r for r in all_recipes if r.get("partition") == "train"}

    # Sample for structural metrics
    sample_dags = random.sample(dags, min(SAMPLE_N, len(dags)))
    print(f"  Sampled {len(sample_dags)} DAGs for structural eval")

    # ── METRIC 1: Ingredient step-assignment + Arg recall ────────────────────
    print("\n[1/5] Ingredient step-assignment rate + arg recall...", flush=True)

    # Ingredient capture is 100% by construction (DAG is built from recipe.ingredients).
    # The interesting metric is STEP ASSIGNMENT: % of ingredient→process edges that
    # matched via substring (good) vs fell back to step-0 (uninformative).
    ing_match_rates  = []   # per recipe: % of edges that are "input" not "input_fallback"
    arg_recall_dur   = []
    arg_recall_tmp   = []

    for dag in tqdm(sample_dags, desc="step-assign+arg"):
        recipe = recipe_by_id.get(dag["id"])
        if not recipe:
            continue

        # Ingredient step-assignment accuracy
        ing_edges = [e for e in dag["edges"] if e["label"] in ("input", "input_fallback")]
        if ing_edges:
            good = sum(1 for e in ing_edges if e["label"] == "input")
            ing_match_rates.append(good / len(ing_edges))

        # Arg recall: for each step with a time/temp mention, did DAG capture it?
        steps = [s["text"] for s in recipe.get("instructions", []) if s["text"].strip()]
        proc_nodes = sorted(
            [n for n in dag["nodes"] if n["type"] == "process"],
            key=lambda n: n["step_idx"]
        )
        for step_text, proc in zip(steps, proc_nodes):
            ref_args = extract_args_from_text(step_text)
            dag_args = proc.get("args", {})
            if "duration_min" in ref_args:
                arg_recall_dur.append(1.0 if "duration_min" in dag_args else 0.0)
            if "temp_c" in ref_args:
                arg_recall_tmp.append(1.0 if "temp_c" in dag_args else 0.0)

    print(f"  Ingredient capture rate:    100%  (by construction — open vocabulary)")
    print(f"  Ingredient step-assignment: {np.mean(ing_match_rates):.1%} matched via substring  (n={len(ing_match_rates)} recipes)")
    print(f"  Arg recall (duration):      {np.mean(arg_recall_dur):.1%}  (n={len(arg_recall_dur)} steps with time mention)")
    print(f"  Arg recall (temperature):   {np.mean(arg_recall_tmp):.1%}  (n={len(arg_recall_tmp)} steps with temp mention)")

    # ── METRIC 3: Process-sequence LCS (round-trip) ──────────────────────────
    print("\n[2/5] Process-sequence LCS (round-trip)...", flush=True)

    # For a subsample — re-encoding via BGE is expensive
    rt_sample = random.sample(sample_dags, min(500, len(sample_dags)))
    lcs_scores = []

    for dag in tqdm(rt_sample, desc="round-trip LCS"):
        # Original process sequence from DAG
        seq_orig = [
            n["canonical"]
            for n in sorted([n for n in dag["nodes"] if n["type"] == "process"],
                            key=lambda n: n["step_idx"])
        ]
        if not seq_orig:
            continue

        # Decode DAG → NL
        decoded_nl = decode_dag(dag)

        # Re-encode decoded NL → process sequence
        seq_rt = text_to_process_seq(decoded_nl)

        lcs_scores.append(lcs_score(seq_orig, seq_rt))

    print(f"  Process-sequence LCS (round-trip): {np.mean(lcs_scores):.3f} ± {np.std(lcs_scores):.3f}")
    print(f"  Median: {np.median(lcs_scores):.3f}")
    print(f"  % > 0.5: {(np.array(lcs_scores) > 0.5).mean():.1%}")
    print(f"  % > 0.8: {(np.array(lcs_scores) > 0.8).mean():.1%}")

    # ── METRIC 4: SentenceBERT cosine sim ────────────────────────────────────
    print("\n[3/5] SentenceBERT semantic similarity...", flush=True)

    sbert_sample = random.sample(sample_dags, min(1000, len(sample_dags)))
    orig_texts   = []
    decoded_texts= []
    for dag in sbert_sample:
        recipe = recipe_by_id.get(dag["id"])
        if not recipe:
            continue
        steps = [s["text"].strip() for s in recipe.get("instructions", []) if s["text"].strip()]
        orig_texts.append(" ".join(steps))
        decoded_texts.append(decode_dag(dag))

    sbert_sims = batch_cosine_sim(orig_texts, decoded_texts)
    print(f"  SentenceBERT cosine sim: {sbert_sims.mean():.3f} ± {sbert_sims.std():.3f}")
    print(f"  Median: {np.median(sbert_sims):.3f}")
    print(f"  % > 0.5: {(sbert_sims > 0.5).mean():.1%}")
    print(f"  % > 0.7: {(sbert_sims > 0.7).mean():.1%}")

    # Baseline: random pair similarity (null hypothesis)
    shuffled_texts = decoded_texts.copy()
    random.shuffle(shuffled_texts)
    null_sims = batch_cosine_sim(orig_texts[:len(shuffled_texts)], shuffled_texts)
    print(f"  Null (random pairs) cosine sim: {null_sims.mean():.3f}  (gap: {sbert_sims.mean()-null_sims.mean():+.3f})")

    # Free BGE + SBERT from GPU before loading vLLM judge
    import gc, torch
    global BGE, SBERT
    del BGE, SBERT
    gc.collect()
    torch.cuda.empty_cache()
    print("  GPU memory freed for vLLM judge.", flush=True)

    # ── METRIC 5: LLM judge with adversarial baselines ───────────────────────
    print("\n[4/5] Building LLM judge pairs...", flush=True)

    judge_dags = random.sample(dags, min(JUDGE_N_EACH * 4, len(dags)))
    judge_pairs: list[tuple[str, str, str]] = []

    # True positives
    tp_dags = judge_dags[:JUDGE_N_EACH]
    for dag in tp_dags:
        recipe = recipe_by_id.get(dag["id"])
        if not recipe:
            continue
        steps = [s["text"].strip() for s in recipe.get("instructions", []) if s["text"].strip()]
        orig  = " ".join(steps[:10])
        dec   = decode_dag(dag)
        judge_pairs.append(("true_pos", orig, dec))

    # Easy negatives: decoded NL from a completely different recipe
    neg_pool = judge_dags[JUDGE_N_EACH:JUDGE_N_EACH*3]
    neg_decoded = [decode_dag(d) for d in neg_pool]
    for i, dag in enumerate(tp_dags):
        if i >= len(neg_pool):
            break
        recipe = recipe_by_id.get(dag["id"])
        if not recipe:
            continue
        steps = [s["text"].strip() for s in recipe.get("instructions", []) if s["text"].strip()]
        orig  = " ".join(steps[:10])
        other_dec = neg_decoded[(i + JUDGE_N_EACH // 2) % len(neg_decoded)]  # different recipe
        judge_pairs.append(("easy_neg", orig, other_dec))

    # Shuffled negatives: same recipe but reverse process order
    for dag in tp_dags:
        recipe = recipe_by_id.get(dag["id"])
        if not recipe:
            continue
        steps = [s["text"].strip() for s in recipe.get("instructions", []) if s["text"].strip()]
        orig     = " ".join(steps[:10])
        shuffled = decode_dag_shuffled(dag)
        judge_pairs.append(("shuffled_neg", orig, shuffled))

    print(f"  Total judge pairs: {len(judge_pairs)}")
    by_label = defaultdict(list)
    for label, a, b in judge_pairs:
        by_label[label].append((a, b))
    for lbl, items in by_label.items():
        print(f"    {lbl}: {len(items)}")

    print("\n[5/5] Running vLLM judge...", flush=True)
    judge_results = run_judge_batch(judge_pairs)

    # ── Compute judge accuracy ────────────────────────────────────────────────
    by_label_results = defaultdict(list)
    for r in judge_results:
        by_label_results[r["label"]].append(r["answer"])

    print("\n" + "="*60)
    print("LLM JUDGE RESULTS")
    print("="*60)

    # Expected: true_pos → YES, easy_neg/shuffled_neg → NO
    tp_answers     = by_label_results["true_pos"]
    easy_answers   = by_label_results["easy_neg"]
    shuffle_answers= by_label_results["shuffled_neg"]

    tp_yes    = tp_answers.count("YES") / max(len(tp_answers), 1)
    easy_no   = easy_answers.count("NO") / max(len(easy_answers), 1)
    shuf_no   = shuffle_answers.count("NO") / max(len(shuffle_answers), 1)

    print(f"\n  True positive  → YES rate:  {tp_yes:.1%}  ({tp_answers.count('YES')}/{len(tp_answers)})")
    print(f"  Easy negative  → NO  rate:  {easy_no:.1%}  ({easy_answers.count('NO')}/{len(easy_answers)})")
    print(f"  Shuffled neg   → NO  rate:  {shuf_no:.1%}  ({shuffle_answers.count('NO')}/{len(shuffle_answers)})")

    # Judge validity: only meaningful if easy_neg NO rate is high (>80%)
    judge_valid = easy_no >= 0.80
    print(f"\n  Judge validity: {'VALID' if judge_valid else 'QUESTIONABLE'} (easy_neg NO rate {'≥' if judge_valid else '<'} 80%)")
    if judge_valid:
        print(f"  → DAG encoding rated equivalent by LLM: {tp_yes:.1%} of true positives")
        print(f"  → Shuffled order correctly rejected:    {shuf_no:.1%}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY — RecipeDAG Information Capture")
    print("="*60)
    print(f"  Ingredient capture:             100% (open vocabulary, by construction)")
    print(f"  Ingredient step-assignment:     {np.mean(ing_match_rates):.1%}")
    print(f"  Arg recall (duration):          {np.mean(arg_recall_dur):.1%}")
    print(f"  Arg recall (temperature):       {np.mean(arg_recall_tmp):.1%}")
    print(f"  Process-seq LCS (round-trip):   {np.mean(lcs_scores):.3f}  (median {np.median(lcs_scores):.3f})")
    print(f"  SentenceBERT cosine sim:        {sbert_sims.mean():.3f}  (null: {null_sims.mean():.3f})")
    print(f"  LLM judge TP acceptance:        {tp_yes:.1%}")
    print(f"  LLM judge easy-neg rejection:   {easy_no:.1%}")
    print(f"  LLM judge shuffled rejection:   {shuf_no:.1%}")
    print(f"  [reference] ROUGE-L (phase 4):  0.167")

    # ── Save ──────────────────────────────────────────────────────────────────
    results = {
        "n_sample": len(sample_dags),
        "ingredient_capture_rate": 1.0,  # 100% by construction (open vocabulary)
        "ingredient_step_assignment": {
            "mean": float(np.mean(ing_match_rates)),
            "std":  float(np.std(ing_match_rates)),
            "note": "% of ingredient→process edges matched via substring (vs fallback to step-0)",
        },
        "arg_recall": {
            "duration_mean": float(np.mean(arg_recall_dur)) if arg_recall_dur else None,
            "duration_n":    len(arg_recall_dur),
            "temp_mean":     float(np.mean(arg_recall_tmp)) if arg_recall_tmp else None,
            "temp_n":        len(arg_recall_tmp),
        },
        "process_lcs_roundtrip": {
            "mean":    float(np.mean(lcs_scores)),
            "median":  float(np.median(lcs_scores)),
            "std":     float(np.std(lcs_scores)),
            "pct_gt_05": float((np.array(lcs_scores) > 0.5).mean()),
            "pct_gt_08": float((np.array(lcs_scores) > 0.8).mean()),
        },
        "sbert_cosine_sim": {
            "mean":    float(sbert_sims.mean()),
            "median":  float(np.median(sbert_sims)),
            "std":     float(sbert_sims.std()),
            "null_mean": float(null_sims.mean()),
            "gap":     float(sbert_sims.mean() - null_sims.mean()),
            "pct_gt_05": float((sbert_sims > 0.5).mean()),
            "pct_gt_07": float((sbert_sims > 0.7).mean()),
        },
        "llm_judge": {
            "model":              JUDGE_MODEL,
            "n_per_condition":    JUDGE_N_EACH,
            "true_pos_yes_rate":  float(tp_yes),
            "easy_neg_no_rate":   float(easy_no),
            "shuffled_neg_no_rate": float(shuf_no),
            "judge_valid":        judge_valid,
        },
        "rouge_l_phase4_reference": 0.167,
    }
    with open(OUT / "proper_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(OUT / "judge_results.json", "w") as f:
        json.dump(judge_results, f, indent=2)

    print(f"\n  Saved → {OUT}/proper_eval_results.json")
    print(f"  Saved → {OUT}/judge_results.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
