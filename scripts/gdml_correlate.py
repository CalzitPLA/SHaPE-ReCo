#!/usr/bin/env python3
"""
One program, four readings: does the GDML that a simulation reads describe the same solid as the program JSON that
FreeCAD builds? (the user's request 2026-09-30: "correlate our freecad json with gdml")

    python3 scripts/gdml_correlate.py <program.json ...> [--out DIR] [--points 200000] [--aegir DIR] [--union multi]

For every program JSON (reco_field.export.program_to_dict: the tree FreeCAD's builder executes):
  1. seeded points, uniform in the program's bounds (2 % margin);
  2. the verdict inside / outside of four readers:
       program   the SDF tree (reco_field.export.program_from_dict, src.sdf_primitives): the pipeline's own semantics
       freecad   FreeCAD 1.0 builds the JSON with reco_field/freecad_builder.py's `build` (the WHOLE result, all solids)
                 and answers isInside - the reference, the solid our STEP files come from
       gdml      our exporter (reco_field/gdml_export.py) read by ROOT's TGeoManager (FindNode)
       aegir     aegir's converter (scripts/sandbox_geometry/make_gdml.py in the aegir checkout, --origin 0 0 0,
                 --union as given) read by ROOT the same way
  3. per pair of readers the points they judge differently, leaving out the points within 1e-4 of the diagonal of
     the program's surface (a verdict there is a matter of rounding), and the Monte Carlo volume of each reader;
     FreeCAD's exact volume and its number of solids beside them.

Writes <out>/<stem>/{points.npy, verdicts.npz, ours.gdml, aegir.gdml, freecad.json} and <out>/correlation.{json,md}.
ROOT comes from aegir's pixi environment "bench" (the laptop has no other ROOT); nothing is written into aegir's
checkout. A reader that fails is reported with its error, the others still run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "dataset_gen"))

from reco_field.export import find_freecadcmd, program_from_dict  # noqa: E402
from reco_field import gdml_export as ge  # noqa: E402

AEGIR = Path(os.environ.get("AEGIR_ROOT") or Path.home() / "Documents" / "Sandbox_aegir" / "aegir")
READERS = ("program", "freecad", "gdml", "aegir")

FREECAD_HELPER = r'''
import json, os, sys
import numpy as np
import FreeCAD as App
# the builder's functions without its closing main() call (as reco_field/exact_check.py takes them)
src = open(os.path.join(os.environ["BUILDER_DIR"], "freecad_builder.py")).read()
ns = dict(__name__="freecad_builder_functions")
exec(compile(src[:src.rindex("main()")], "freecad_builder.py", "exec"), ns)
class fb:
    build = staticmethod(ns["build"])
d = json.load(open(os.environ["PROGRAM_JSON"]))
pts = np.load(os.environ["POINTS_NPY"])
out = dict()
try:
    shape = fb.build(d).removeSplitter()
    solids = sorted(shape.Solids, key=lambda s: s.Volume, reverse=True)
    out.update(volume=float(sum(s.Volume for s in solids)), n_solids=len(solids),
               biggest=float(solids[0].Volume) if solids else 0.0, valid=bool(shape.isValid()))
    tol = 1e-7 * max(1.0, float(shape.BoundBox.DiagonalLength))
    v = np.array([shape.isInside(App.Vector(*map(float, p)), tol, True) for p in pts], dtype=bool)
    np.save(os.environ["VERDICT_NPY"], v)
except Exception as e:
    out["error"] = "%s: %s" % (type(e).__name__, e)
json.dump(out, open(os.environ["RESULT_JSON"], "w"))
'''

ROOT_HELPER = r'''
import sys
import numpy as np
import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kWarning
gdml, pts_npy, out_npy, volume = sys.argv[1:5]
geo = ROOT.TGeoManager.Import(gdml)
if not geo:
    raise SystemExit("ROOT could not read " + gdml)
pts = np.load(pts_npy) * 0.1                       # TGeo's length unit is cm
v = np.zeros(len(pts), dtype=bool)
for i, (x, y, z) in enumerate(pts):
    node = geo.FindNode(float(x), float(y), float(z))
    v[i] = bool(node) and node.GetVolume().GetName() == volume
np.save(out_npy, v)
'''


def pixi() -> str:
    import shutil
    return shutil.which("pixi") or str(Path.home() / ".pixi" / "bin" / "pixi")


def run_root(gdml: Path, pts_npy: Path, out_npy: Path, volume: str, work: Path) -> str:
    helper = work / "root_reader.py"
    helper.write_text(ROOT_HELPER)
    r = subprocess.run([pixi(), "run", "--manifest-path", str(AEGIR / "pixi.toml"), "-e", "bench", "python", str(helper),
                        str(gdml), str(pts_npy), str(out_npy), volume], capture_output=True, text=True, timeout=3600)
    return "" if r.returncode == 0 and out_npy.exists() else (r.stderr.strip().splitlines() or ["failed"])[-1][:300]


def run_freecad(program: Path, pts_npy: Path, out_npy: Path, work: Path) -> dict:
    helper = work / "freecad_reader.py"
    helper.write_text(FREECAD_HELPER)
    res = work / "freecad.json"
    env = dict(os.environ, PROGRAM_JSON=str(program), POINTS_NPY=str(pts_npy), VERDICT_NPY=str(out_npy),
               RESULT_JSON=str(res), BUILDER_DIR=str(ROOT_DIR / "reco_field"))
    r = subprocess.run([find_freecadcmd() or "freecadcmd", str(helper)], capture_output=True, text=True, env=env,
                       timeout=3600)
    if not res.exists():
        return dict(error=(r.stderr.strip().splitlines() or ["freecadcmd failed"])[-1][:300])
    return json.loads(res.read_text())


def correlate(program: Path, out: Path, n: int, union: str) -> dict:
    stem = program.stem if program.stem not in ("program", "model") else program.parent.name
    work = out / stem
    work.mkdir(parents=True, exist_ok=True)
    d = json.loads(program.read_text())
    d = d.get("program", d) if "type" not in d else d
    tree = program_from_dict(d)
    lo, hi = (np.asarray(b, float) for b in tree.bounds())
    pad = 0.02 * (hi - lo)
    rng = np.random.default_rng(0)
    pts = lo - pad + rng.random((n, 3)) * (hi - lo + 2 * pad)
    np.save(work / "points.npy", pts)
    diag = float(np.linalg.norm(hi - lo))
    sdf = tree.sdf(pts)
    verdict = {"program": sdf <= 0}
    far = np.abs(sdf) > 1e-4 * diag
    info = dict(program=str(program), points=n, near_surface=int((~far).sum()), diag_mm=diag,
                box_volume=float(np.prod(hi - lo + 2 * pad)), errors={}, seconds={})

    t = time.time()
    fc = run_freecad(work / "program.json" if (work / "program.json").write_text(json.dumps(d)) else program,
                     work / "points.npy", work / "freecad.npy", work)
    info["freecad"] = fc
    info["seconds"]["freecad"] = round(time.time() - t, 1)
    if "error" in fc:
        info["errors"]["freecad"] = fc["error"]
    else:
        verdict["freecad"] = np.load(work / "freecad.npy")

    t = time.time()
    try:
        g = ge.export_gdml(tree, work / "ours.gdml", name="part")
        info["gdml_solids"] = g["solids"]
        err = run_root(work / "ours.gdml", work / "points.npy", work / "gdml.npy", "part_volume", work)
        if err:
            info["errors"]["gdml"] = err
        else:
            verdict["gdml"] = np.load(work / "gdml.npy")
    except Exception as e:
        info["errors"]["gdml"] = f"{type(e).__name__}: {e}"
    info["seconds"]["gdml"] = round(time.time() - t, 1)

    t = time.time()
    conv = AEGIR / "scripts" / "sandbox_geometry" / "make_gdml.py"
    r = subprocess.run([sys.executable, str(conv), "csg", str(work / "program.json"), str(work / "aegir.gdml"),
                        "--origin", "0", "0", "0", "--union", union, "--material", "Iron_part"],
                       capture_output=True, text=True, timeout=600)
    if r.returncode:
        info["errors"]["aegir"] = (r.stderr.strip().splitlines() or ["converter failed"])[-1][:300]
    else:
        # the converter refers to Geant4 NIST materials, which ROOT does not know: define the two it uses
        text = (work / "aegir.gdml").read_text().replace(
            "<materials/>", '<materials>\n    <material name="Iron_part" Z="26"><D value="7.874"/><atom value="55.845"/>'
            '</material>\n    <material name="G4_AIR" Z="7"><D value="0.0012"/><atom value="14.007"/></material>\n'
            '  </materials>')
        (work / "aegir_root.gdml").write_text(text)
        err = run_root(work / "aegir_root.gdml", work / "points.npy", work / "aegir.npy", "PartVolume", work)
        if err:
            info["errors"]["aegir"] = err
        else:
            verdict["aegir"] = np.load(work / "aegir.npy")
    info["seconds"]["aegir"] = round(time.time() - t, 1)

    np.savez_compressed(work / "verdicts.npz", **verdict)
    box = info["box_volume"]
    info["mc_volume"] = {k: float(v.mean() * box) for k, v in verdict.items()}
    info["differ"] = {f"{a}|{b}": int((verdict[a][far] != verdict[b][far]).sum())
                      for i, a in enumerate(READERS) for b in READERS[i + 1:] if a in verdict and b in verdict}
    info["far_points"] = int(far.sum())
    (work / "correlation.json").write_text(json.dumps(info, indent=1))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("programs", nargs="+")
    ap.add_argument("--out", default=str(Path.home() / "shape_reco_results" / "gdml_correlation"))
    ap.add_argument("--points", type=int, default=200000)
    ap.add_argument("--union", default="multi", choices=("multi", "binary", "balanced", "as-is"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    rows = []
    for p in a.programs:
        info = correlate(Path(p), out, a.points, a.union)
        rows.append(info)
        fc = info.get("freecad", {})
        print(f"{Path(p).stem}: differ {info['differ']}  FreeCAD V {fc.get('volume', float('nan')):.6g} "
              f"({fc.get('n_solids', '?')} solids)  errors {info['errors'] or 'none'}", flush=True)
    (out / "correlation.json").write_text(json.dumps(rows, indent=1))
    lines = ["| program | points (far) | FreeCAD V mm3 | solids | MC V program | differ program/FreeCAD | "
             "program/GDML | program/aegir | FreeCAD/GDML | GDML/aegir | errors |", "|" + "---|" * 11]
    for r in rows:
        dd, fc = r["differ"], r.get("freecad", {})
        g = lambda k: str(dd.get(k, "-"))
        lines.append(f"| {Path(r['program']).stem} | {r['points']} ({r['far_points']}) | "
                     f"{fc.get('volume', float('nan')):.6g} | {fc.get('n_solids', '-')} | "
                     f"{r['mc_volume'].get('program', float('nan')):.6g} | {g('program|freecad')} | {g('program|gdml')} | "
                     f"{g('program|aegir')} | {g('freecad|gdml')} | {g('gdml|aegir')} | "
                     f"{'; '.join(f'{k}: {v}' for k, v in r['errors'].items()) or ''} |")
    (out / "correlation.md").write_text("\n".join(lines) + "\n")
    print(out / "correlation.md")


if __name__ == "__main__":
    main()
