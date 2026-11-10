#!/usr/bin/env python3
"""
Import graph of the project's Python modules, read from the source with `ast` (nothing typed by hand).

    python3 scripts/codemap_graph.py            # prints a summary and writes docs/data/codemap_graph.json

Scope: the packages the training and the reconstruction actually run through -
    gen    dataset_gen/src          sample generation (the module names are `src.*`)
    train  training                 the three CNN trainers and their shared plumbing
    reco   reco_engine              inference and the reconstruction loop
    script scripts                  evaluation, probes, comparison and report scripts
    nodes  freecad_nodes/shape_reco the FreeCAD "Nodes" UI (23 files, grouped as one unit in the figures)
Not in scope: tests, archiv, the legacy FeatureNet and dashboard directories.

An edge A -> B means A imports B. `lazy` = the import sits inside a function (paid only when that function
runs: typically torch or FreeCAD helpers). External tools that are called as subprocesses (FreeCAD, stltovoxel)
are found from string constants and recorded as `runtime` edges.
"""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PACKAGES = {                       # kind -> (directory, module-name prefix)
    "gen": (ROOT / "dataset_gen" / "src", "src"),
    "train": (ROOT / "training", "training"),
    "reco": (ROOT / "reco_engine", "reco_engine"),
    "script": (ROOT / "scripts", "scripts"),
    "nodes": (ROOT / "freecad_nodes" / "shape_reco", "shape_reco"),
}


def runtime_tool(text: str):
    """External tool a string constant names as a command or file: the executable `freecadcmd` (or a path ending
    in it), the `stltovoxel` executable, or a `*.FCMacro` file. Prose (docstrings, help texts) never matches."""
    if len(text) > 200 or "\n" in text:
        return None
    base = text.rsplit("/", 1)[-1].strip()
    if base.lower() == "freecadcmd":
        return "FreeCAD"
    if base == "stltovoxel":
        return "stltovoxel"
    if base.endswith(".FCMacro") and " " not in base:
        return "FreeCAD macros"
    return None


EXTERNAL = ("torch", "numpy", "scipy", "PIL", "FreeCAD", "matplotlib", "stl", "trimesh", "PySide2", "PySide6", "Part")


def module_files():
    out = {}
    for kind, (directory, prefix) in PACKAGES.items():
        for path in sorted(directory.glob("*.py")):
            if path.name == "__init__.py" or path.name in ("codemap_graph.py", "make_codemap_tex.py"):   # not the generators
                continue
            out[f"{prefix}.{path.stem}"] = (kind, path)
    return out


def docstring_summary(tree: ast.Module, limit: int = 150) -> str:
    """First sentence of the module docstring (its first paragraph, joined), at most `limit` characters."""
    doc = (ast.get_docstring(tree) or "").strip()
    paragraph = " ".join(doc.split("\n\n")[0].split())
    cut = paragraph.find(". ")
    text = paragraph[:cut + 1] if 0 < cut < limit else paragraph
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + " ..."
    return text


class Collector(ast.NodeVisitor):
    def __init__(self):
        self.imports = []          # (dotted name, [imported names], lazy)
        self.strings = []
        self.depth = 0

    def _fn(self, node):
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    visit_FunctionDef = visit_AsyncFunctionDef = _fn

    def visit_Import(self, node):
        for a in node.names:
            self.imports.append((a.name, [], self.depth > 0, 0))

    def visit_ImportFrom(self, node):
        self.imports.append((node.module or "", [a.name for a in node.names], self.depth > 0, node.level))

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            self.strings.append(node.value)

    def visit_Expr(self, node):                                   # a bare string statement is a docstring: skip
        if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            self.generic_visit(node)


def resolve(dotted: str, names, level: int, current: str, known: set, aliases: dict):
    """Internal module ids that an import statement refers to."""
    if level:                                                     # relative: relative to the current package
        base = current.rsplit(".", 1)[0].split(".")
        base = base[:len(base) - (level - 1)]
        dotted = ".".join(base + ([dotted] if dotted else []))
    if dotted.startswith("dataset_gen."):
        dotted = dotted[len("dataset_gen."):]
    hits = []
    candidates = [dotted] + [f"{dotted}.{n}" for n in names]
    for cand in candidates:
        cand = aliases.get(cand, cand)
        if cand in known and cand not in hits:
            hits.append(cand)
    return hits


def build():
    files = module_files()
    known = set(files)
    aliases = {}
    for mod, (kind, _) in files.items():                         # scripts import their siblings by bare name
        if kind == "script":
            aliases[mod.split(".", 1)[1]] = mod
        if kind == "nodes":
            aliases[mod.split(".", 1)[1]] = mod
    info, edges, runtime, external = {}, [], [], defaultdict(set)
    for mod, (kind, path) in files.items():
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        col = Collector()
        col.visit(tree)
        entry = any(isinstance(n, ast.If) and "__main__" in ast.dump(n.test) for n in tree.body)
        info[mod] = dict(kind=kind, path=str(path.relative_to(ROOT)), loc=len(src.splitlines()), entry=entry,
                         summary=docstring_summary(tree))
        seen = {}
        for dotted, names, lazy, level in col.imports:
            hits = resolve(dotted, names, level, mod, known, aliases)
            for h in hits:
                if h != mod:
                    seen[h] = seen.get(h, True) and lazy          # solid wins over lazy when both occur
            if not hits:
                top = dotted.split(".")[0]
                if top in EXTERNAL:
                    external[mod].add(top)
        edges += [dict(src=mod, dst=h, lazy=lz) for h, lz in seen.items()]
        tools = {t for t in map(runtime_tool, col.strings) if t}
        runtime += [dict(src=mod, dst=t) for t in sorted(tools)]
    for mod, ex in external.items():
        info[mod]["external"] = sorted(ex)
    indeg = defaultdict(int)
    for e in edges:
        indeg[e["dst"]] += 1
    for mod in info:
        info[mod]["fan_in"] = indeg[mod]
        info[mod]["fan_out"] = sum(1 for e in edges if e["src"] == mod)
    return dict(modules=info, edges=edges, runtime=runtime)


if __name__ == "__main__":
    g = build()
    out = ROOT / "docs" / "data" / "codemap_graph.json"
    out.write_text(json.dumps(g, indent=1))
    kinds = defaultdict(int)
    for m in g["modules"].values():
        kinds[m["kind"]] += 1
    print("modules:", dict(kinds), " import edges:", len(g["edges"]), " lazy:", sum(e["lazy"] for e in g["edges"]),
          " runtime:", len(g["runtime"]))
    cross = defaultdict(int)
    for e in g["edges"]:
        cross[(g["modules"][e["src"]]["kind"], g["modules"][e["dst"]]["kind"])] += 1
    for (a, b), n in sorted(cross.items()):
        print(f"  {a:6s} -> {b:6s} {n}")
    orphans = [m for m, i in g["modules"].items() if i["fan_in"] == 0 and not i["entry"] and i["kind"] in ("gen", "train", "reco")]
    print("no importer and no __main__:", orphans)
    print("runtime:", sorted({(r['src'], r['dst']) for r in g["runtime"]}))
