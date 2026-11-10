#!/usr/bin/env python3
"""
Draw a FreeCAD Nodes-workbench graph (a saved .json) as a figure: nodes at their saved positions, sockets named from the node classes, wires as curves.

    python3 scripts/make_node_graph_figure.py [graph.json] [out.pdf]        # default: example_vector_field.json -> docs/figures/vf_nodes.pdf
"""
import ast
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NODES = ROOT / "freecad_nodes" / "shape_reco"
COLORS = {"Text": "#f3f0e0", "Peel (Vector Field)": "#cfe3f5", "Export Program": "#d6ecd2", "Field Grade": "#f7dcc4", "Load Mesh": "#e6e0f0"}


def socket_names(module):
    """(input names, output names) of a shape_reco node, read from its source."""
    tree = ast.parse((NODES / f"{module}.py").read_text())
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in ("inputs_init_list", "outputs_init_list"):
            found[node.arg] = [e.elts[0].value for e in node.value.elts]
    return found.get("inputs_init_list", []), found.get("outputs_init_list", [])


def draw(graph_path, out_path):
    d = json.loads(Path(graph_path).read_text())
    W, ROW = 200.0, 26.0
    geo, where = {}, {}
    for n in d["nodes"]:
        module = n["op_code"].split("'")[1].split(".")[0]
        ins, outs = socket_names(module) if module.startswith("shape_reco_") else ([], ["value"])
        if n["title"] == "Text":
            w, h = 250.0, 30.0
        else:
            w, h = W, ROW * max(len(ins), len(outs), 1) + 34.0
        x, y = n["pos_x"], n["pos_y"]
        geo[n["id"]] = (x, y, w, h)
        for k, s in enumerate(n["inputs"]):
            where[s["id"]] = (x, y + 30.0 + ROW * k + ROW / 2)
        for k, s in enumerate(n["outputs"]):
            where[s["id"]] = (x + w, y + (h / 2 if n["title"] == "Text" else 30.0 + ROW * k + ROW / 2))
        geo[n["id"]] += (n, ins, outs)
    xs = [g[0] for g in geo.values()] + [g[0] + g[2] for g in geo.values()]
    ys = [g[1] for g in geo.values()] + [g[1] + g[3] for g in geo.values()]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for e in d["edges"]:
        (x0, y0), (x1, y1) = where[e["start"]], where[e["end"]]
        dx = max(60.0, abs(x1 - x0) * 0.45)
        ax.add_patch(PathPatch(MPath([(x0, y0), (x0 + dx, y0), (x1 - dx, y1), (x1, y1)], [1, 4, 4, 4]), fill=False, ec="#555", lw=1.0, zorder=1))
    for x, y, w, h, n, ins, outs in geo.values():
        title = n["title"]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8", fc=COLORS.get(title, "#eee"), ec="#333", lw=1.0, zorder=2))
        if title == "Text":
            v = n["content"].get("value", "")
            ax.text(x + 8, y + h / 2, (v if len(v) < 30 else v[:27] + "...") if not v.startswith("VECTOR FIELD") else "note: " + v[:24] + "...",
                    va="center", fontsize=6.5, family="monospace", zorder=3)
            continue
        ax.text(x + w / 2, y + 12, title, ha="center", va="center", fontsize=8, weight="bold", zorder=3)
        for k, name in enumerate(ins):
            ax.plot(*where[n["inputs"][k]["id"]], "o", ms=4, color="#333", zorder=3)
            ax.text(x + 8, where[n["inputs"][k]["id"]][1], name, va="center", fontsize=6.5, zorder=3)
        for k, name in enumerate(outs):
            ax.plot(*where[n["outputs"][k]["id"]], "o", ms=4, color="#333", zorder=3)
            ax.text(x + w - 8, where[n["outputs"][k]["id"]][1], name, va="center", ha="right", fontsize=6.5, zorder=3)
    ax.set_xlim(min(xs) - 20, max(xs) + 20)
    ax.set_ylim(max(ys) + 20, min(ys) - 20)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "freecad_nodes/examples/example_vector_field.json")
    out = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "docs/figures/vf_nodes.pdf")
    draw(src, out)
    print("wrote", out)
