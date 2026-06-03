#!/usr/bin/env python3
"""
Visualize learned VQ-VAE DAGs — clean per-recipe figure, one recipe per output.
"""
import json, textwrap, os
from pathlib import Path
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE       = Path(__file__).resolve().parent.parent
LAYER1     = str(Path(os.environ.get("RECIPE1M", str(BASE.parent / "layer1.json"))))
DAG_FILE   = str(BASE / "data/dags/learned_dags_grammar.jsonl")
OUT_DIR    = str(BASE / "data/figures")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_IDS = {
    "00a1e3bbc5": "Tomato Shrimp Fettuccine",
    "000dc3e7e7": "Penne with Sausage & Kale",
    "b61b4dd004": "Blueberry-Balsamic Grilled Chicken Salad",
}

STEP_CLR = "#2E6FBF"
ING_CLR  = "#E8A838"
EDGE_CLR = "#444"
ING_EDGE = "#C48010"

STOPWORDS = {"the","a","an","in","on","at","to","with","and","or","for",
             "of","is","it","this","that","into","until","over","from",
             "let","just","when","then","now","after","before","well"}

def _build_code_labels():
    """Map each VQ code → short semantic label (grammar type or dominant verb)."""
    from collections import Counter
    labels_path = "data/models/process_vae_grammar/codebook_labels.json"
    ga_path     = "data/models/process_vae_grammar/grammar_alignment.json"
    cb = json.load(open(labels_path))
    ga = json.load(open(ga_path))

    code_to_grammar = {}
    for gtype, info in ga.items():
        for entry in info["top_codes"]:
            c, pct = entry["code"], entry["pct"]
            if c not in code_to_grammar or pct > code_to_grammar[c][1]:
                code_to_grammar[c] = (gtype.split("-")[0], pct)

    def best_verb(texts):
        counts = Counter()
        for t in texts:
            for w in t.strip().split()[:3]:
                w = w.lower().rstrip(".,;:")
                if w not in STOPWORDS and w.isalpha() and len(w) > 2:
                    counts[w] += 1
                    break
        return counts.most_common(1)[0][0] if counts else "?"

    result = {}
    for code_str, step_texts in cb.items():
        code = int(code_str)
        if code in code_to_grammar:
            result[code] = code_to_grammar[code][0]
        else:
            result[code] = best_verb(step_texts)
    return result

CODE_LABELS = _build_code_labels()

def wrap(txt, w=22):
    return "\n".join(textwrap.wrap(txt.strip(), w))

def draw_recipe(dag, title, out_path, recipe_text=None):
    nodes     = dag["nodes"]
    edges     = dag["edges"]
    src_edges = dag.get("source_edges", [])
    T = len(nodes)

    G = nx.DiGraph()
    for n in nodes:
        G.add_node(f"s{n['t']}", kind="step", code=n["code"],
                   text=n["step_text"], t=n["t"])

    # Step→step edges — only gap ≤ 3, single strongest predecessor per node
    from collections import defaultdict
    by_target = defaultdict(list)
    for e in edges:
        gap = e["to"] - e["from"]
        if 0 < gap <= 3 and e["weight"] > 0.15:
            by_target[e["to"]].append((e["weight"], e["from"]))
    for tgt, preds in by_target.items():
        preds.sort(reverse=True)
        w, src = preds[0]   # single strongest only
        G.add_edge(f"s{src}", f"s{tgt}", weight=w, kind="step")

    # Top ingredient sources (by weight, max 1 per step)
    best_src = {}
    for se in sorted(src_edges, key=lambda x: -x["weight"]):
        step = se["step"]
        if step not in best_src and se["weight"] > 0.15:
            best_src[step] = se
    for step, se in best_src.items():
        ikey = f"i{step}"
        G.add_node(ikey, kind="ing", text=se["ingredient"])
        G.add_edge(ikey, f"s{step}", weight=se["weight"], kind="ing")

    step_nodes = [f"s{n['t']}" for n in nodes]
    ing_nodes  = [k for k in G.nodes if k.startswith("i")]

    # ── Layout: meanwhile-aware two-track placement ───────────────────────────
    # Find where "meanwhile" starts — that node and its successors go to track 1
    meanwhile_start = None
    for n in nodes:
        if "meanwhile" in n["step_text"].lower():
            meanwhile_start = n["t"]
            break

    # BFS forward from meanwhile_start in the gap-filtered graph to find track-1 nodes
    track1 = set()
    if meanwhile_start is not None:
        track1.add(meanwhile_start)
        frontier = [meanwhile_start]
        adj = defaultdict(list)
        for e in edges:
            if 0 < e["to"] - e["from"] <= 3 and e["weight"] > 0.15:
                adj[e["from"]].append(e["to"])
        visited = set(track1)
        while frontier:
            cur = frontier.pop()
            for nxt in adj[cur]:
                if nxt not in visited and nxt not in [
                    n["t"] for n in nodes
                    if sum(1 for e in edges
                           if e["to"] == n["t"] and e["from"] < meanwhile_start
                           and e["weight"] > 0.20) > 0  # merge node: has strong pre-meanwhile pred
                ]:
                    visited.add(nxt); track1.add(nxt); frontier.append(nxt)

    x_scale = 2.2
    y_gap   = 2.8
    pos_all = {}
    for n in nodes:
        t_idx = n["t"]
        y = -y_gap if t_idx in track1 else 0.0
        pos_all[f"s{t_idx}"] = (t_idx * x_scale, y)

    # Ingredient nodes float above their step's track
    for k in ing_nodes:
        t_idx = int(k[1:])
        sx, sy = pos_all.get(f"s{t_idx}", (t_idx * x_scale, 0))
        pos_all[k] = (sx, sy + 2.2)

    # ── Figure: DAG left, recipe text right ──────────────────────────────────
    W = max(18, T * 1.6)
    fig = plt.figure(figsize=(W + 5, 7))
    ax_dag = fig.add_axes([0.0, 0.0, 0.72, 1.0])   # DAG takes 72% width
    ax_txt = fig.add_axes([0.74, 0.0, 0.25, 1.0])   # text panel 25%
    ax = ax_dag

    step_edges = [(u,v) for u,v,d in G.edges(data=True) if d.get("kind")=="step"]
    ing_edges  = [(u,v) for u,v,d in G.edges(data=True) if d.get("kind")=="ing"]

    ns = max(1200, 3800 - 100*T)
    fs = max(5.5, 8.5 - 0.18*T)

    nx.draw_networkx_nodes(G, pos_all, nodelist=step_nodes,
                           node_color=STEP_CLR, node_size=ns, ax=ax, alpha=0.93)
    nx.draw_networkx_nodes(G, pos_all, nodelist=ing_nodes,
                           node_color=ING_CLR, node_size=ns*0.45,
                           node_shape="D", ax=ax, alpha=0.88)

    if step_edges:
        ws = [G[u][v]["weight"] for u,v in step_edges]
        wmax = max(ws)
        lws = [1.0 + 3.0*(w/wmax) for w in ws]
        nx.draw_networkx_edges(G, pos_all, edgelist=step_edges, width=lws,
                               edge_color=EDGE_CLR, arrows=True, arrowsize=20,
                               ax=ax, min_source_margin=20, min_target_margin=20)

    if ing_edges:
        nx.draw_networkx_edges(G, pos_all, edgelist=ing_edges, width=1.2,
                               edge_color=ING_EDGE, arrows=True, arrowsize=12,
                               ax=ax, style="dashed",
                               min_source_margin=8, min_target_margin=18)

    # Step labels: code + semantic type label
    step_labels = {}
    for n in nodes:
        nid = f"s{n['t']}"
        sem = CODE_LABELS.get(n["code"], "?")
        step_labels[nid] = f"[{n['code']}]\n{sem}"
    nx.draw_networkx_labels(G, pos_all, labels=step_labels,
                            font_size=fs, font_color="white", ax=ax)

    ing_labels = {}
    for k in ing_nodes:
        ing_labels[k] = wrap(G.nodes[k]["text"][:28], 16)
    nx.draw_networkx_labels(G, pos_all, labels=ing_labels,
                            font_size=fs - 0.8, font_color="#6b3f00", ax=ax)

    p1 = mpatches.Patch(color=STEP_CLR, label="Process step  [VQ code]")
    p2 = mpatches.Patch(color=ING_CLR,  label="Ingredient entry point")
    ax.legend(handles=[p1, p2], fontsize=9, loc="lower right")

    ax.set_title(
        f"{title}\n"
        "Step→step edges: causal self-attention  |  "
        "Ingredient edges: cross-attention  |  No LLM at inference",
        fontsize=11, fontweight="bold", pad=10)
    ax.axis("off")

    # ── Right panel: original recipe text ────────────────────────────────────
    ax_txt.axis("off")
    ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)
    ax_txt.add_patch(plt.Rectangle((0, 0), 1, 1, color="#F5F5F5", zorder=0))
    ax_txt.text(0.5, 0.98, "Original Recipe", ha="center", va="top",
                fontsize=10, fontweight="bold", color="#222",
                transform=ax_txt.transAxes)
    ax_txt.axhline(0.955, color="#aaa", linewidth=0.8)

    if recipe_text:
        lines = [f"{i+1}. {s}" for i, s in enumerate(recipe_text)]
        body  = "\n\n".join(lines)
    else:
        body = "(text not available)"

    ax_txt.text(0.04, 0.93, body, ha="left", va="top",
                fontsize=7.2, color="#333", wrap=True,
                transform=ax_txt.transAxes,
                linespacing=1.55,
                bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F5", ec="none"))

    plt.savefig(out_path, dpi=145, bbox_inches="tight")
    plt.close()
    print(f"  {out_path}")


def main():
    # Load original recipe texts from layer1.json
    print("Loading layer1.json for recipe text...")
    recipe_texts = {}
    with open(LAYER1) as f:
        for rec in json.load(f):
            if rec["id"] in TARGET_IDS:
                steps = [s["text"] for s in rec.get("instructions", [])]
                recipe_texts[rec["id"]] = steps
            if len(recipe_texts) == len(TARGET_IDS):
                break

    targets = {}
    with open(DAG_FILE) as f:
        for line in f:
            d = json.loads(line)
            if d["id"] in TARGET_IDS:
                targets[d["id"]] = d
            if len(targets) == len(TARGET_IDS):
                break

    for rid, dag in targets.items():
        slug = TARGET_IDS[rid].lower().replace(" ", "_")
        out  = os.path.join(OUT_DIR, f"vq_dag_{slug}.png")
        draw_recipe(dag, TARGET_IDS[rid], out, recipe_text=recipe_texts.get(rid))


if __name__ == "__main__":
    main()
