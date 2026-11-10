#!/usr/bin/env python3
"""
How many stored programs the torus guard (reco_engine/torus_guard.py, T-29) would have hit, per rule.

Sources (whatever exists):
  trees    ~/shape_reco_results/real_trees/programs/<series>.json - the program trees of the real-tree study
           (scripts/real_tree_study.py: ABC known and pilots with and without sub-parts, liner stages, review frames,
           the frame); the part's diagonal from the input's STL (ABC: <stl-dir>/<id>_*_step_<n>.stl), the liner's
           from its run in the run store
  runs     ~/shape_reco_results/<part>/<run>/run.json - every run of the run store (its "program"), the diagonal
           from the input named in the run
  proposed ~/shape_reco_results/real_trees/abc/<run>/<part>/program.json - the ABC results the trees were made of,
           with every torus they hold anywhere: in the program and among the `alternatives` each operation keeps
           (the cascade's other branches, the loop's other candidates) - tori the loop was offered

and the real torus faces of the inputs (--faces-json, default docs/data/torus_faces_abc500.json: every torus face
of the 500 ABC STEP files of the external set with its part's diagonal; made by --faces STEP_DIR, one freecadcmd
run over the directory - 35 s for the 500 files on server_b, 2026-10-01): what the guard would reject of the
geometry real parts hold.

A torus counts under the FIRST rule it breaks (spindle/horn before near-horn before oversized), a program under
every rule one of its tori breaks. The diagonal is that of the input's bounding box: from an STL of the same stem
(next to a STEP input or in --stl-dir), from --diag STEM=VALUE, else from FreeCAD's BoundBox of the STEP (one
freecadcmd run per such input). Without a diagonal the size rule is not applied to that program (listed).

    python3 scripts/torus_guard_census.py [--json out.json] [--diag Liner_LINER_001-LINER_001=1648.2]
    python3 scripts/torus_guard_census.py --faces <dir of .step> --faces-json out.json    (freecadcmd: only here)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

from reco_engine import torus_guard as tg  # noqa: E402

RESULTS = Path.home() / "shape_reco_results"
STL_DIR = Path.home() / "Documents" / "Uni" / "thesis" / "external_parts" / "abc" / "stl"
BBOX_SCRIPT = """import os, Part
s = Part.Shape(); s.read(os.environ["STEP_IN"])
print("DIAGONAL", s.BoundBox.DiagonalLength)
"""
FACES_JSON = ROOT / "docs" / "data" / "torus_faces_abc500.json"
FACES_SCRIPT = """import json, os, Part
out = []
names = sorted(n for n in os.listdir(os.environ["STEP_DIR"]) if n.lower().endswith((".step", ".stp")))
for n in names:
    rec = dict(part=n)
    try:
        s = Part.Shape(); s.read(os.path.join(os.environ["STEP_DIR"], n))
        rec.update(solids=len(s.Solids), diagonal=s.BoundBox.DiagonalLength, faces=len(s.Faces),
                   tori=[[f.Surface.MajorRadius, f.Surface.MinorRadius, f.Area] for f in s.Faces
                         if type(f.Surface).__name__ == "Toroid"])
    except Exception as exc:
        rec["error"] = str(exc)[:200]
    out.append(rec)
json.dump(out, open(os.environ["FACES_OUT"], "w"))
"""


class Diagonals:
    """The bounding-box diagonal of an input file, cached per path."""

    def __init__(self, stl_dir: Path, given: dict):
        self.stl_dir, self.given, self.cache = stl_dir, dict(given), {}

    def of(self, path) -> float | None:
        if path is None:
            return None
        p = Path(str(path))
        if p.stem in self.given:
            return self.given[p.stem]
        if str(p) in self.cache:
            return self.cache[str(p)]
        d = None
        for stl in ([p] if p.suffix.lower() == ".stl" else [p.with_suffix(".stl"), self.stl_dir / (p.stem + ".stl")]):
            if stl.is_file():
                import trimesh
                b = trimesh.load(str(stl), force="mesh").bounds
                d = float(np.linalg.norm(b[1] - b[0]))
                break
        if d is None and p.suffix.lower() in (".step", ".stp") and p.is_file():
            d = self._freecad(p)
        self.cache[str(p)] = d
        return d

    @staticmethod
    def _freecad(step: Path) -> float | None:
        from src.closest_point_field import find_freecadcmd
        fc = find_freecadcmd()
        if fc is None:
            return None
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "bbox.py"
            script.write_text(BBOX_SCRIPT)
            out = subprocess.run([fc, str(script)], capture_output=True, text=True, timeout=600,
                                 env={"STEP_IN": str(step), "PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
        for line in (out.stdout or "").splitlines():
            if line.startswith("DIAGONAL"):
                return float(line.split()[1])
        return None


def schema_tori(prim: dict, cut: bool):
    """(major, minor, under a cut) of every torus of a schema primitive dict (a leaf or a composite's parts)."""
    if not isinstance(prim, dict):
        return
    if prim.get("kind") == "torus":
        d = prim.get("dims_mm") or {}
        yield float(d.get("major_radius", np.nan)), float(d.get("minor_radius", np.nan)), cut
    for q in prim.get("parts") or []:
        yield from schema_tori(q, cut)


def all_tori(x, cut: bool = False):
    """(major, minor, under a cut) of every torus anywhere in a result dict (operations, parts, alternatives)."""
    if isinstance(x, dict):
        if x.get("kind") == "torus" and isinstance(x.get("dims_mm"), dict):
            d = x["dims_mm"]
            yield float(d.get("major_radius", np.nan)), float(d.get("minor_radius", np.nan)), cut
        for k, v in x.items():
            yield from all_tori(v, cut or (k == "primitive" and x.get("op") == "cut"))
    elif isinstance(x, list):
        for v in x:
            yield from all_tori(v, cut)


def judge(tori, diagonal):
    """[(major, minor, cut, rule or None, text)] of a program's tori."""
    out = []
    for R, r, cut in tori:
        found = tg.torus_defect(R, r, diagonal)
        out.append((R, r, cut, found.rule if found else None, found.text if found else ""))
    return out


def tree_input(series: str, liner_runs: dict, stl_dir: Path):
    """The input file of a program tree's series (see the module's text), or None."""
    parts = series.split("_")
    if parts[0] in ("known", "pilot2"):
        ident, step = parts[-2], parts[-1]
        found = sorted(stl_dir.glob(f"{ident}_*_step_{step}.stl"))
        return found[0] if found else None
    if parts[0] == "liner":
        n = int(parts[1].replace("ops", ""))
        return liner_runs.get(n)
    return None


def census(results: Path, stl_dir: Path, given: dict) -> dict:
    diags = Diagonals(stl_dir, given)
    runs = sorted(results.glob("*/*/run.json"))
    run_programs, liner_runs = [], {}
    for rj in runs:
        try:
            d = json.loads(rj.read_text())
        except Exception:
            continue
        ops = d.get("program") or []
        path = (d.get("input") or {}).get("path")
        if rj.parent.parent.name.startswith("Liner") and len(ops) >= 2:
            liner_runs.setdefault(len(ops), path)
        run_programs.append((str(rj.parent.relative_to(results)), path, ops))
    out = {}
    trees = []
    for f in sorted((results / "real_trees" / "programs").glob("*.json")):
        prog = json.loads(f.read_text())
        tori = list(tg.program_tori(prog))
        path = tree_input(f.stem, liner_runs, stl_dir) if tori else None
        trees.append((f.stem, path, tori))
    out["trees"] = _tally([(name, path, tori) for name, path, tori in trees], diags)
    out["runs"] = _tally([(name, path, [t for o in ops for t in schema_tori(o.get("primitive"), o.get("op") == "cut")])
                          for name, path, ops in run_programs], diags)
    proposed = []
    for f in sorted((results / "real_trees" / "abc").glob("*/*/program.json")):
        d = json.loads(f.read_text())
        proposed.append((f"{f.parent.parent.name}/{f.parent.name}", stl_dir / (f.parent.name + ".stl"),
                         list(all_tori(d.get("operations") or []))))
    out["proposed"] = _tally(proposed, diags)
    return out


def _tally(programs, diags) -> dict:
    rules = Counter()
    tori_by_rule = Counter()
    hits = defaultdict(list)
    no_diagonal = []
    with_tori, hit = 0, 0
    rows = []
    for name, path, tori in programs:
        if not tori:
            continue
        with_tori += 1
        diag = diags.of(path)
        if diag is None:
            no_diagonal.append(name)
        judged = judge(tori, diag)
        broken = sorted({rule for *_, rule, _ in judged if rule})
        hit += bool(broken)
        for rule in broken:
            rules[rule] += 1
            hits[rule].append(name)
        for R, r, cut, rule, text in judged:
            tori_by_rule[rule or "admitted"] += 1
            rows.append(dict(program=name, major=R, minor=r, op="cut" if cut else "add", diagonal=diag,
                             gap=(R - r) / R if R else None, size=(R + r) / diag if diag else None, rule=rule))
    return dict(programs=len(programs), with_tori=with_tori, hit=hit, programs_by_rule=dict(rules),
                tori_by_rule=dict(tori_by_rule), hits=dict(hits), no_diagonal=no_diagonal, tori=rows)


def face_census(step_dir: Path, out: Path) -> None:
    """Every torus face of the STEP files in `step_dir` with its part's diagonal, as JSON (FreeCAD)."""
    from src.closest_point_field import find_freecadcmd
    fc = find_freecadcmd()
    if fc is None:
        raise SystemExit("--faces needs freecadcmd")
    with tempfile.TemporaryDirectory() as tmp:
        script, raw = Path(tmp) / "faces.py", Path(tmp) / "faces.json"
        script.write_text(FACES_SCRIPT)
        subprocess.run([fc, str(script)], check=False, timeout=7200, capture_output=True,
                       env={"STEP_DIR": str(step_dir), "FACES_OUT": str(raw), "PATH": "/usr/bin:/bin",
                            "HOME": str(Path.home())})
        rows = json.loads(raw.read_text())
    head = json.dumps(dict(step_dir=str(step_dir), parts=len(rows), errors=sum(1 for r in rows if "error" in r)))
    tori = ",\n".join(json.dumps(r) for r in rows if r.get("tori"))           # one part per line
    out.write_text(head[:-1] + ', "with_tori": [\n' + tori + "\n]}\n")


def face_summary(path: Path) -> dict:
    """Per rule the real torus faces of a face census (`face_census`) the guard would reject."""
    d = json.loads(path.read_text())
    by_rule, parts, n = Counter(), defaultdict(set), 0
    gaps, sizes = [], []
    for rec in d["with_tori"]:
        for R, r, _area in rec["tori"]:
            n += 1
            found = tg.torus_defect(R, r, rec.get("diagonal"))
            by_rule[found.rule if found else "admitted"] += 1
            if found:
                parts[found.rule].add(rec["part"])
            else:
                gaps.append((R - r) / R)
                sizes.append((R + r) / rec["diagonal"])
    return dict(parts=d["parts"], parts_with_tori=len(d["with_tori"]), faces=n, by_rule=dict(by_rule),
                parts_by_rule={k: len(v) for k, v in parts.items()},
                admitted_min_gap=min(gaps) if gaps else None, admitted_max_size=max(sizes) if sizes else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--stl-dir", type=Path, default=STL_DIR)
    ap.add_argument("--diag", action="append", default=[], metavar="STEM=VALUE",
                    help="the bounding-box diagonal of an input, by its file stem")
    ap.add_argument("--json", type=Path, help="write the census (every torus with its rule) here")
    ap.add_argument("--list", type=int, default=12, help="programs named per rule (default 12)")
    ap.add_argument("--faces", type=Path, metavar="STEP_DIR", help="make the face census of these STEP files first")
    ap.add_argument("--faces-json", type=Path, default=FACES_JSON, help=f"the face census (default {FACES_JSON})")
    a = ap.parse_args(argv)
    if a.faces:
        face_census(a.faces, a.faces_json)
    given = {k: float(v) for k, v in (s.split("=", 1) for s in a.diag)}
    out = census(a.results, a.stl_dir, given)
    print(f"torus guard: near-horn below (R - r) / R = {tg.NEAR_HORN_GAP:g}, oversized above R + r = "
          f"{tg.OVERSIZE_RATIO:g} x the part's diagonal")
    for key, label in (("trees", "program trees (real_trees/programs)"), ("runs", "run store (<part>/<run>/run.json)"),
                       ("proposed", "ABC results with their alternatives (real_trees/abc/*/*/program.json)")):
        t = out[key]
        print(f"\n{label}: {t['programs']} programs, {t['with_tori']} with a torus, {t['hit']} hit")
        for rule in tg.RULES:
            if rule in t["programs_by_rule"]:
                names = t["hits"][rule]
                more = f" ... ({len(names) - a.list} more)" if len(names) > a.list else ""
                print(f"  {rule:13s} {t['programs_by_rule'][rule]:4d} programs, "
                      f"{t['tori_by_rule'].get(rule, 0):4d} tori: " + ", ".join(names[:a.list]) + more)
        print(f"  {'admitted':13s} {t['tori_by_rule'].get('admitted', 0):4d} tori")
        if t["no_diagonal"]:
            print("  without a diagonal (size rule not applied): " + ", ".join(t["no_diagonal"]))
        ok = [r for r in t["tori"] if not r["rule"]]
        if ok:
            gaps = [r["gap"] for r in ok]
            sizes = [r["size"] for r in ok if r["size"] is not None]
            print(f"  admitted tori: smallest (R - r) / R {min(gaps):.4f}, largest (R + r) / diagonal "
                  f"{max(sizes) if sizes else float('nan'):.3f}")
    if a.faces_json and a.faces_json.is_file():
        f = face_summary(a.faces_json)
        out["faces"] = f
        print(f"\nreal torus faces ({a.faces_json.name}): {f['parts']} parts, {f['parts_with_tori']} with a torus "
              f"face, "
              f"{f['faces']} torus faces")
        for rule in tg.RULES + ("admitted",):
            if rule in f["by_rule"]:
                print(f"  {rule:13s} {f['by_rule'][rule]:5d} faces" + (f" in {f['parts_by_rule'][rule]} parts"
                                                                          if rule in f["parts_by_rule"] else ""))
        print(f"  admitted faces: smallest (R - r) / R {f['admitted_min_gap']:.4f}, largest (R + r) / diagonal "
              f"{f['admitted_max_size']:.3f}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
