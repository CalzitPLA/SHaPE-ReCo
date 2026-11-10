#!/usr/bin/env python3
"""
LaTeX/TikZ figures, tables and number macros for docs/codemap_modules.tex, from the import graph that
scripts/codemap_graph.py reads out of the source:

    python3 scripts/make_codemap_tex.py      ->  docs/generated/codemap_*.tex

The layout of each graph is computed by Graphviz (`dot -Tjson`); the drawing is TikZ, so it is native LaTeX
(fonts, colours, scaling) and needs no image files. Figures show a transitive reduction of the imports (an edge
that is implied by a longer path is left out); the tables list every module with its real fan-in and fan-out.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import networkx as nx

import codemap_graph

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "docs" / "generated"

FILL = {"gen": "gen", "train": "train", "reco": "reco", "script": "script", "nodes": "nodes", "tool": "tool"}
NAME = {"gen": "dataset\\_gen/src", "train": "training", "reco": "reco\\_engine", "script": "scripts",
        "nodes": "shape\\_reco nodes"}
CHAR_IN = 0.066                     # width of one \scriptsize\ttfamily character, inches (with margin)


def tex(text: str) -> str:
    for a, b in (("—", "---"), ("–", "--"), ("→", "$\\to$"), ("≥", "$\\ge$"), ("≤", "$\\le$"), ("×", "$\\times$"),
                 ("·", "."), ("’", "'"), ("‘", "'"), ("“", "``"), ("”", "''"), ("°", "$^\\circ$"), ("`", "'")):
        text = text.replace(a, b)
    text = text.encode("ascii", "ignore").decode()
    out = []
    for ch in text:
        out.append({"_": "\\_", "%": "\\%", "&": "\\&", "#": "\\#", "$": "\\$", "{": "\\{", "}": "\\}",
                    "~": "\\textasciitilde{}", "^": "\\textasciicircum{}", "\\": "\\textbackslash{}"}.get(ch, ch))
    s = "".join(out).replace("\\_", "\\_\\allowbreak{}").replace("/", "/\\allowbreak{}").replace("=", "=\\allowbreak{}")
    return s.replace("\\$\\to\\$", "$\\to$").replace("\\$\\ge\\$", "$\\ge$").replace("\\$\\le\\$", "$\\le$") \
            .replace("\\$\\times\\$", "$\\times$").replace("\\$^\\textasciicircum{}\\circ\\$", "$^\\circ$")


def short(mod: str) -> str:
    return mod.split(".", 1)[1]


# ------------------------------------------------------------------------------------------ graphviz -> tikz

def layout(nodes, edges, clusters=None, rankdir="TB", nodesep=0.18, ranksep=0.42):
    """nodes: {id: (width_in, height_in)}; edges: [(a, b)]; clusters: {name: (label, [ids])}. -> dot JSON."""
    lines = ["digraph G {", f"rankdir={rankdir}; nodesep={nodesep}; ranksep={ranksep}; splines=true; "
             "node [shape=box, fixedsize=true, label=\"\"]; edge [arrowhead=none];"]
    placed = set()
    for cname, (label, ids) in (clusters or {}).items():
        lines.append(f'subgraph "cluster_{cname}" {{ label="{label}"; margin=10;')
        for i in ids:
            w, h = nodes[i]
            lines.append(f'"{i}" [width={w:.3f}, height={h:.3f}];')
            placed.add(i)
        lines.append("}")
    for i, (w, h) in nodes.items():
        if i not in placed:
            lines.append(f'"{i}" [width={w:.3f}, height={h:.3f}];')
    for a, b in edges:
        lines.append(f'"{a}" -> "{b}";')
    lines.append("}")
    res = subprocess.run(["dot", "-Tjson"], input="\n".join(lines), capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def tikz(graph, nodes_tex, edge_style, cluster_labels=None, max_width_bp=440.0):
    """dot JSON -> TikZ. nodes_tex: {id: (tex label, style)}; edge_style: {(a, b): style}."""
    x0, y0, x1, y1 = (float(v) for v in graph["bb"].split(","))
    width = x1 - x0
    scale = min(1.0, max_width_bp / width)
    by_gvid = {o["_gvid"]: o for o in graph["objects"]}
    out = [f"\\begin{{tikzpicture}}[x=1bp,y=1bp,scale={scale:.3f},transform shape,cmnode/.style={{draw,rounded corners=1.5pt,"
           "inner sep=1.5pt,font=\\scriptsize\\ttfamily,align=center,line width=0.5pt},"
           "cmcluster/.style={draw=black!40,rounded corners=3pt,dashed,line width=0.5pt},"
           "cmedge/.style={-{Stealth[length=1.6mm,width=1.2mm]},line width=0.45pt,black!65},"
           "cmlazy/.style={dashed}]"]
    for o in graph["objects"]:
        if "pos" in o or not o.get("bb"):
            continue
        a, b, c, d = (float(v) for v in o["bb"].split(","))
        out.append(f"\\draw[cmcluster] ({a:.1f},{b:.1f}) rectangle ({c:.1f},{d:.1f});")
        label = (cluster_labels or {}).get(o["name"].replace("cluster_", ""), "")
        if label:
            out.append(f"\\node[anchor=north west,font=\\scriptsize\\itshape,text=black!65] at ({a + 3:.1f},{d - 1:.1f}) {{{label}}};")
    for e in graph.get("edges", []):
        a, b = by_gvid[e["tail"]]["name"], by_gvid[e["head"]]["name"]
        style = edge_style.get((a, b), "")
        pts = [(float(p[0]), float(p[1])) for op in e["_draw_"] if op["op"] == "b" for p in op["points"]]
        seg = f"({pts[0][0]:.1f},{pts[0][1]:.1f})"
        for i in range(1, len(pts), 3):
            (p1, p2, p3) = pts[i:i + 3]
            seg += f" .. controls ({p1[0]:.1f},{p1[1]:.1f}) and ({p2[0]:.1f},{p2[1]:.1f}) .. ({p3[0]:.1f},{p3[1]:.1f})"
        out.append(f"\\draw[cmedge{',' + style if style else ''}] {seg};")
    for o in graph["objects"]:
        if "pos" not in o:
            continue
        px, py = (float(v) for v in o["pos"].split(","))
        w, h = float(o["width"]) * 72, float(o["height"]) * 72
        label, style = nodes_tex[o["name"]]
        out.append(f"\\node[cmnode,{style},minimum width={w:.1f}bp,minimum height={h:.1f}bp] at ({px:.1f},{py:.1f}) {{{label}}};")
    out.append("\\end{tikzpicture}")
    return "\n".join(out)


def reduced(edges):
    """Transitive reduction of [(a, b, lazy)]; edges of a cyclic graph are returned unchanged."""
    g = nx.DiGraph()
    g.add_edges_from((a, b) for a, b, _ in edges)
    if not nx.is_directed_acyclic_graph(g):
        return edges, False
    keep = set(nx.transitive_reduction(g).edges())
    return [e for e in edges if (e[0], e[1]) in keep], True


def node_style(kind, entry=False):
    return f"fill={FILL[kind]}" + (",line width=1.2pt" if entry else "")


# ------------------------------------------------------------------------------------------------ figures

def fig_module_graph(g, modules, extra_nodes=None, extra_edges=None, clusters=None, cluster_labels=None, dagger=()):
    """Graph of `modules` (list of ids) with their import edges (reduced); extra_* add group nodes."""
    info = g["modules"]
    ids = list(modules)
    edges = [(e["src"], e["dst"], e["lazy"]) for e in g["edges"] if e["src"] in ids and e["dst"] in ids]
    edges, ok = reduced(edges)
    nodes, nodes_tex, styles = {}, {}, {}
    for m in ids:
        label = tex(short(m)) + ("$^\\dagger$" if m in dagger else "")
        nodes[m] = (len(short(m)) * CHAR_IN + 0.14 + (0.06 if m in dagger else 0), 0.24)
        nodes_tex[m] = (label, node_style(info[m]["kind"], info[m]["entry"]))
    all_edges = [(a, b) for a, b, _ in edges]
    for a, b, lazy in edges:
        if lazy:
            styles[(a, b)] = "cmlazy"
    for gid, (label, kind, width) in (extra_nodes or {}).items():
        nodes[gid] = (width, 0.34)
        nodes_tex[gid] = (label, node_style(kind))
    for a, b in extra_edges or []:
        all_edges.append((a, b))
    graph = layout(nodes, all_edges, clusters)
    return tikz(graph, nodes_tex, styles, cluster_labels), ok


def fig_overview(g):
    info, edges = g["modules"], g["edges"]
    cross = defaultdict(int)
    for e in edges:
        a, b = info[e["src"]]["kind"], info[e["dst"]]["kind"]
        if a != b:
            cross[(a, b)] += 1
    counts = defaultdict(int)
    for m in info.values():
        counts[m["kind"]] += 1
    runtime = defaultdict(set)
    for r in g["runtime"]:
        runtime[(info[r["src"]]["kind"], r["dst"])].add(r["src"])
    nodes = {k: (1.55, 0.5) for k in ("gen", "train", "reco", "script", "nodes")}
    nodes.update({"FreeCAD": (1.1, 0.3), "FreeCAD macros": (1.3, 0.3), "stltovoxel": (1.1, 0.3)})
    tex_nodes = {k: (f"{NAME[k]}\\\\{counts[k]} modules", node_style(k)) for k in ("gen", "train", "reco", "script", "nodes")}
    tex_nodes.update({t: (tex(t), "fill=tool,dashed") for t in ("FreeCAD", "FreeCAD macros", "stltovoxel")})
    e_list, styles = [], {}
    for (a, b), n in cross.items():
        e_list.append((a, b))
    for (kind, tool), mods in runtime.items():
        e_list.append((kind, tool))
        styles[(kind, tool)] = "densely dotted"
    graph = layout(nodes, e_list, rankdir="TB", nodesep=0.35, ranksep=0.55)
    # edge labels: number of importing modules / import statements, placed at the curve midpoint
    code = tikz(graph, tex_nodes, styles)
    by_gvid = {o["_gvid"]: o for o in graph["objects"]}
    labels = []
    for e in graph["edges"]:
        a, b = by_gvid[e["tail"]]["name"], by_gvid[e["head"]]["name"]
        pts = [(float(p[0]), float(p[1])) for op in e["_draw_"] if op["op"] == "b" for p in op["points"]]
        mid = pts[len(pts) // 2]
        n = cross.get((a, b)) or len(runtime.get((a, b), ()))
        labels.append(f"\\node[fill=white,inner sep=0.5pt,font=\\tiny] at ({mid[0]:.1f},{mid[1]:.1f}) {{{n}}};")
    return code.replace("\\end{tikzpicture}", "\n".join(labels) + "\n\\end{tikzpicture}"), cross, runtime


def fig_dsm(g):
    """Which modules each script uses: rows scripts, columns modules; filled = import at module level, open = inside a function."""
    info = g["modules"]
    scripts = sorted(m for m, i in info.items() if i["kind"] == "script")
    used = defaultdict(dict)
    for e in g["edges"]:
        if info[e["src"]]["kind"] == "script" and info[e["dst"]]["kind"] != "script":
            used[e["src"]][e["dst"]] = e["lazy"]
    scripts = [s for s in scripts if used[s]]
    cols = sorted({d for s in scripts for d in used[s]}, key=lambda m: (["gen", "train", "reco"].index(info[m]["kind"]), m))
    cell = 0.36
    out = ["\\begin{tikzpicture}[x=1cm,y=1cm,font=\\scriptsize]"]
    for j, c in enumerate(cols):
        out.append(f"\\node[rotate=90,anchor=west,font=\\scriptsize\\ttfamily,inner sep=1pt] at ({j * cell + cell / 2:.2f},0.05) {{{tex(short(c))}}};")
    top = 0.0
    for i, s in enumerate(scripts):
        y = -i * cell - cell / 2
        out.append(f"\\node[anchor=east,font=\\scriptsize\\ttfamily,inner sep=2pt] at (-0.05,{y:.2f}) {{{tex(short(s))}}};")
        for j, c in enumerate(cols):
            fill = FILL[info[c]["kind"]]
            out.append(f"\\draw[gray!35,line width=0.3pt] ({j * cell:.2f},{y - cell / 2:.2f}) rectangle +({cell},{cell});")
            if c in used[s]:
                if used[s][c]:
                    out.append(f"\\draw[fill=white,draw={'black!70'},line width=0.6pt] ({j * cell + 0.1:.2f},{y - cell / 2 + 0.1:.2f}) rectangle +({cell - 0.2:.2f},{cell - 0.2:.2f});")
                else:
                    out.append(f"\\draw[fill=black!70,draw=black!70] ({j * cell + 0.1:.2f},{y - cell / 2 + 0.1:.2f}) rectangle +({cell - 0.2:.2f},{cell - 0.2:.2f});")
    # colour bar of the package of each column
    for j, c in enumerate(cols):
        out.append(f"\\fill[{FILL[info[c]['kind']]}] ({j * cell:.2f},{-len(scripts) * cell - 0.05:.2f}) rectangle +({cell},-0.12);")
    out.append("\\end{tikzpicture}")
    return "\n".join(out), len(scripts), len(cols)


# ------------------------------------------------------------------------------------------------- tables

def table(g, kind):
    info = g["modules"]
    rows = sorted(m for m, i in info.items() if i["kind"] == kind)
    rt = defaultdict(set)
    for r in g["runtime"]:
        rt[r["src"]].add(r["dst"])
    lines = ["\\begingroup\\scriptsize", "\\begin{longtable}{@{}p{4.0cm}rrr>{\\raggedright\\arraybackslash}p{2.2cm}>{\\raggedright\\arraybackslash}p{5.8cm}@{}}", "\\toprule",
             "Module & LOC & in & out & needs & Purpose (first sentence of the docstring) \\\\", "\\midrule", "\\endhead"]
    for m in rows:
        i = info[m]
        needs = ", ".join(sorted(set(i.get("external", [])) | {t.replace("FreeCAD macros", "macros") for t in rt.get(m, [])}))
        name = f"\\texttt{{{tex(short(m))}}}" + ("$^{*}$" if i["entry"] else "")
        lines.append(f"{name} & {i['loc']} & {i['fan_in']} & {i['fan_out']} & {tex(needs)} & {tex(i['summary'])} \\\\")
    lines += ["\\bottomrule", "\\end{longtable}", "\\endgroup"]
    return "\n".join(lines)


def main():
    g = codemap_graph.build()
    (ROOT / "docs" / "data" / "codemap_graph.json").write_text(json.dumps(g, indent=1))
    GEN.mkdir(exist_ok=True)
    info = g["modules"]
    kinds = defaultdict(list)
    for m, i in info.items():
        kinds[i["kind"]].append(m)

    # overview
    code, cross, runtime = fig_overview(g)
    (GEN / "codemap_overview.tex").write_text(code)

    # dataset_gen/src + training
    gt = kinds["gen"] + kinds["train"]
    code, ok1 = fig_module_graph(g, gt, clusters={"gen": ("dataset\\_gen/src", kinds["gen"]), "train": ("training", kinds["train"])},
                                 cluster_labels={"gen": "dataset\\_gen/src", "train": "training"})
    (GEN / "codemap_gentrain.tex").write_text(code)

    # reco_engine, with the node package as one group on top and the modules it reaches at the bottom
    reach = sorted({e["dst"] for e in g["edges"] if info[e["src"]]["kind"] == "reco" and info[e["dst"]]["kind"] in ("gen", "train")})
    node_targets = sorted({e["dst"] for e in g["edges"] if info[e["src"]]["kind"] == "nodes" and info[e["dst"]]["kind"] == "reco"})
    reco = kinds["reco"] + reach
    code, ok2 = fig_module_graph(g, reco, dagger=set(node_targets))
    (GEN / "codemap_reco.tex").write_text(code)

    # scripts
    code, n_rows, n_cols = fig_dsm(g)
    (GEN / "codemap_scripts.tex").write_text(code)

    # tables
    for kind in ("gen", "train", "reco", "script"):
        (GEN / f"codemap_table_{kind}.tex").write_text(table(g, kind))

    # numbers
    fan = sorted(((i["fan_in"], m) for m, i in info.items() if i["kind"] in ("gen", "train", "reco")), reverse=True)
    lazy = sum(e["lazy"] for e in g["edges"])
    order = {"gen": 0, "train": 1, "reco": 2, "nodes": 3, "script": 3}
    upward = [(e["src"], e["dst"]) for e in g["edges"] if order[info[e["src"]]["kind"]] < order[info[e["dst"]]["kind"]]]
    ext = defaultdict(int)
    for i in info.values():
        for x in i.get("external", []):
            ext[x] += 1
    loc = {k: sum(info[m]["loc"] for m in ms) for k, ms in kinds.items()}
    orphans = [m for m, i in info.items() if i["fan_in"] == 0 and not i["entry"] and i["kind"] in ("gen", "train", "reco")]
    macros = {
        "cmModules": len(info), "cmEdges": len(g["edges"]), "cmLazy": lazy, "cmUpward": len(upward),
        "cmLocGen": loc["gen"], "cmLocTrain": loc["train"], "cmLocReco": loc["reco"], "cmLocScript": loc["script"], "cmLocNodes": loc["nodes"],
        "cmNGen": len(kinds["gen"]), "cmNTrain": len(kinds["train"]), "cmNReco": len(kinds["reco"]), "cmNScript": len(kinds["script"]), "cmNNodes": len(kinds["nodes"]),
        "cmTopIn": ", ".join(f"\\texttt{{{tex(short(m))}}} ({n})" for n, m in fan[:5]),
        "cmOrphans": ", ".join(f"\\texttt{{{tex(short(m))}}}" for m in orphans) or "none",
        "cmTorch": ext.get("torch", 0), "cmFreeCADmods": len({r["src"] for r in g["runtime"] if r["dst"] in ("FreeCAD", "FreeCAD macros")}),
        "cmStlMods": len({r["src"] for r in g["runtime"] if r["dst"] == "stltovoxel"}),
        "cmRecoToTrain": cross[("reco", "train")], "cmRecoToGen": cross[("reco", "gen")], "cmTrainToGen": cross[("train", "gen")],
        "cmScriptRows": n_rows, "cmScriptCols": n_cols, "cmNodesToReco": len(node_targets),
    }
    (GEN / "codemap_macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    print("wrote docs/generated/codemap_*.tex;", "reduction ok:", ok1, ok2, "; upward edges:", upward)


if __name__ == "__main__":
    main()
