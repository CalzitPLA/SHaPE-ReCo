#!/usr/bin/env python3
"""The IoU and its Sobolev relatives against the errors of every cascade stage, for every primitive kind as the
stage-2 tool.

    python3 scripts/iou_stage_study.py [--out docs/reports/iou_two_stage] [--pitch 0.5] [--jobs 4]

Parts: the box 60 x 40 x 20 (stage 1) fused with or cut by one tool (stage 2) centred on its top face, the tool one of
the kinds the cascade knows (box, cylinder, cone, sphere, torus; axis z) - 10 two-stage trees. Errors, one per cascade
stage (reco_engine/cascade.py: TYPE -> ROTATION -> DIMENSIONS -> POSITION) plus the missing stage:
  type       the tool replaced by each other kind of the SAME volume at the same centre      (4 per part)
  rotation   the tool tilted about x through its centre by 2, 5, 10, 20, 45, 90 deg
  dimensions the tool scaled about its centre by 1.05, 1.1, 1.2, 1.4
  position   the tool moved in x by 0.5, 1, 2, 4, 8, 16 mm
  missing    the tree cut after stage 1
Measures per pair: the exact OCCT IoU of the whole part (reco_field.exact_check.exact_residual), the piece IoU (the true
piece A delta box against the model's piece, what the loop's per-component and feature acceptance score; parity grid),
and the relative Sobolev distances D_s of scripts/iou_sobolev_study.py (chi at s = -1, 0, 0.45; phi at s = 1).
Writes data/stages.json, data/stage_*.tex, fig/stage_*.pdf|png. Reuses the helpers of iou_sobolev_study.py.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import iou_sobolev_study as base                                     # noqa: E402

KINDS = ("box", "cylinder", "cone", "sphere", "torus")
OPS = ("fuse", "cut")
C = np.array([30.0, 20.0, 20.0])                                     # the tool's centre: the middle of the top face
BOX = dict(lo=(0.0, 0.0, 0.0), size=(60.0, 40.0, 20.0))
# canonical tools, centred at the origin, axis z
CANON = dict(box=dict(a=14.0, b=14.0, h=20.0), cylinder=dict(r=8.0, h=20.0), cone=dict(r1=10.0, r2=4.0, h=20.0),
             sphere=dict(r=10.0), torus=dict(R=10.0, r=4.0))
SWEEPS = dict(rotation=[2, 5, 10, 20, 45, 90], dimensions=[1.05, 1.1, 1.2, 1.4], position=[0.5, 1, 2, 4, 8, 16])
STAGE_UNIT = dict(rotation="deg", dimensions="scale", position="mm")
MEASURES = [("1-IoU", "1 - IoU (exact, part)"), ("1-IoU piece", "1 - IoU (piece)"), ("chi_-1", "$D_{-1}(\\chi)$"),
            ("chi_0", "$D_0(\\chi)$"), ("chi_0.45", "$D_{0.45}(\\chi)$"), ("phi_1", "$D_1(\\varphi)$")]


def canon_volume(kind):
    p = CANON[kind]
    if kind == "box":
        return p["a"] * p["b"] * p["h"]
    if kind == "cylinder":
        return math.pi * p["r"] ** 2 * p["h"]
    if kind == "cone":
        return math.pi * p["h"] / 3 * (p["r1"] ** 2 + p["r1"] * p["r2"] + p["r2"] ** 2)
    if kind == "sphere":
        return 4 / 3 * math.pi * p["r"] ** 3
    return 2 * math.pi ** 2 * p["R"] * p["r"] ** 2


def tool(kind, scale=1.0, tilt=0.0, shift=0.0):
    return dict(kind=kind, scale=float(scale), tilt=float(tilt), shift=float(shift))


def same_volume(kind, as_kind):
    """the tool of kind `kind` scaled to the volume of the canonical `as_kind` tool (the type error)."""
    return tool(kind, scale=(canon_volume(as_kind) / canon_volume(kind)) ** (1 / 3))


def cases():
    """(part, error-stage, magnitude/label, model tool or None)"""
    out = []
    for kind in KINDS:
        for op in OPS:
            part = f"{kind}-{op}"
            out.append((part, "exact", 0, tool(kind)))
            out.append((part, "missing", 0, None))
            for other in KINDS:
                if other != kind:
                    out.append((part, "type", other, same_volume(other, kind)))
            for st, vals in SWEEPS.items():
                for v in vals:
                    out.append((part, st, v, tool(kind, **{"rotation": dict(tilt=v), "dimensions": dict(scale=v),
                                                         "position": dict(shift=v)}[st])))
    return out


# ------------------------------------------------------------------------------------------------ meshes


def tool_manifold(t, segments=96):
    import manifold3d as m3
    k, p = t["kind"], CANON[t["kind"]]
    if k == "box":
        m = m3.Manifold.cube([p["a"], p["b"], p["h"]], True)
    elif k == "cylinder":
        m = m3.Manifold.cylinder(p["h"], p["r"], p["r"], 180, True)
    elif k == "cone":
        m = m3.Manifold.cylinder(p["h"], p["r1"], p["r2"], 180, True)
    elif k == "sphere":
        m = m3.Manifold.sphere(p["r"], segments)
    else:
        m = m3.Manifold.revolve(m3.CrossSection.circle(p["r"], 48).translate([p["R"], 0.0]), segments)
    return m.scale([t["scale"]] * 3).rotate([t["tilt"], 0.0, 0.0]).translate(list(C + [t["shift"], 0, 0]))


def box_manifold():
    import manifold3d as m3
    return m3.Manifold.cube(list(BOX["size"])).translate(list(BOX["lo"]))


def part_manifold(op, t):
    b = box_manifold()
    if t is None:
        return b
    return b + tool_manifold(t) if op == "fuse" else b - tool_manifold(t)


def piece_manifold(op, t):
    """what stage 2 changes: the tool outside the box (fuse) or inside it (cut)."""
    if t is None:
        return None
    return tool_manifold(t) - box_manifold() if op == "fuse" else tool_manifold(t) ^ box_manifold()


# ------------------------------------------------------------------------------------------------ STEP + OCCT

FC_SCRIPT = r'''
import json, os, Part, FreeCAD as App
job = json.load(open(os.environ["STAGE_JOB"]))
C = App.Vector(*job["centre"])
def tool(t, canon):
    k, p = t["kind"], canon[t["kind"]]
    if k == "box":
        s = Part.makeBox(p["a"], p["b"], p["h"], App.Vector(-p["a"] / 2, -p["b"] / 2, -p["h"] / 2))
    elif k == "cylinder":
        s = Part.makeCylinder(p["r"], p["h"], App.Vector(0, 0, -p["h"] / 2))
    elif k == "cone":
        s = Part.makeCone(p["r1"], p["r2"], p["h"], App.Vector(0, 0, -p["h"] / 2))
    elif k == "sphere":
        s = Part.makeSphere(p["r"])
    else:
        s = Part.makeTorus(p["R"], p["r"])
    s = s.copy()
    s.scale(t["scale"], App.Vector(0, 0, 0))
    s.rotate(App.Vector(0, 0, 0), App.Vector(1, 0, 0), t["tilt"])
    s.translate(C + App.Vector(t["shift"], 0, 0))
    return s
for item in job["items"]:
    b = Part.makeBox(60, 40, 20)
    t = item["tool"]
    if t is None:
        s = b
    else:
        s = b.fuse(tool(t, job["canon"])) if item["op"] == "fuse" else b.cut(tool(t, job["canon"]))
        s = s.removeSplitter()
    s.exportStep(item["path"])
'''


def step_name(part, stage, mag):
    return f"{part}__{stage}_{mag}.step".replace(" ", "_")


def write_steps(items):
    from reco_field.export import find_freecadcmd
    with tempfile.TemporaryDirectory() as tmp:
        script, job = Path(tmp) / "build.py", Path(tmp) / "job.json"
        script.write_text(FC_SCRIPT)
        job.write_text(json.dumps(dict(items=items, centre=C.tolist(), canon=CANON)))
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env["STAGE_JOB"] = str(job)
        subprocess.run([find_freecadcmd(), str(script)], env=env, check=True, capture_output=True, timeout=3600)


def exact_one(args):
    inp, model = args
    from reco_field.exact_check import exact_residual
    r = exact_residual(inp, model)
    return {k: r.get(k) for k in ("iou", "r_plus", "r_minus", "model_genus", "fuzzy", "error", "input_volume")}


# ------------------------------------------------------------------------------------------------ grid measures

_G = {}


def grid_measures(part, rows, pitch):
    """all grid measures of one part's models (one process per part: the grid, spectrum and input stay in memory)."""
    kind, op = part.split("-")
    G = base.Grid((-16, -16, -16), (76, 56, 54), pitch)
    spec = base.Spectrum(G)
    A = base.occupancy(part_manifold(op, tool(kind)), G)
    phiA = base.signed_distance(A, G.pitch)
    PA = spec.power(A)
    PphiA = spec.power(phiA)
    nA = {s: spec.norm2(PA, s) for s in (-1.0, 0.0, 0.45)}
    nphi = spec.norm2(PphiA, 1.0)
    pA = base.occupancy(piece_manifold(op, tool(kind)), G)
    out = []
    for stage, mag, t in rows:
        B = base.occupancy(part_manifold(op, t), G)
        e = A.astype(np.float64) - B
        P = spec.power(e)
        Pp = spec.power(phiA - base.signed_distance(B, G.pitch))
        pm = piece_manifold(op, t)
        pB = base.occupancy(pm, G) if pm is not None and not pm.is_empty() else np.zeros_like(pA)
        union = int(np.count_nonzero(pA | pB))
        out.append(dict(stage=stage, mag=mag, **{f"chi_{s:g}": math.sqrt(spec.norm2(P, s) / nA[s]) for s in nA},
                        phi_1=math.sqrt(spec.norm2(Pp, 1.0) / nphi),
                        piece_iou=int(np.count_nonzero(pA & pB)) / union if union else 1.0,
                        voxel_iou=base.voxel_iou(A, B), piece_volume=float(pA.sum()) * pitch ** 3))
    return part, out


# ------------------------------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/reports/iou_two_stage"))
    ap.add_argument("--pitch", type=float, default=0.5)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    sdir = out / "parts" / "stages"
    sdir.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    (out / "fig").mkdir(exist_ok=True)
    cs = cases()

    # ---- exact: STEP of every model, one freecadcmd; then exact_residual per pair (in parallel processes)
    cache = out / "data" / "stages_exact.json"
    exact = json.loads(cache.read_text()) if cache.is_file() else {}
    todo = [c for c in cs if f"{c[0]}|{c[1]}|{c[2]}" not in exact]
    if todo:
        items = [dict(op=p.split("-")[1], tool=t, path=str(sdir / step_name(p, st, m))) for p, st, m, t in cs]
        print(f"freecadcmd: {len(items)} STEP files", flush=True)
        write_steps(items)
        jobs = [(str(sdir / step_name(p, "exact", 0)), str(sdir / step_name(p, st, m))) for p, st, m, _ in todo]
        with ProcessPoolExecutor(args.jobs) as ex:
            for (p, st, m, _), r in zip(todo, ex.map(exact_one, jobs)):
                exact[f"{p}|{st}|{m}"] = r
                print("exact", p, st, m, r.get("iou"), r.get("error") or "", flush=True)
        cache.write_text(json.dumps(exact, indent=1))
    bad = {k: v for k, v in exact.items() if v.get("error") or v.get("iou") is None}
    print("exact errors:", len(bad), list(bad)[:5])
    for p in {c[0] for c in cs}:
        assert abs(exact[f"{p}|exact|0"]["iou"] - 1) < 1e-6, p

    # ---- grid measures, one process per part
    gcache = out / "data" / "stages_grid.json"
    grid = json.loads(gcache.read_text()) if gcache.is_file() else {}
    parts = sorted({c[0] for c in cs} - set(grid))
    if parts:
        with ProcessPoolExecutor(min(args.jobs, len(parts))) as ex:
            futs = [ex.submit(grid_measures, p, [(st, m, t) for pp, st, m, t in cs if pp == p], args.pitch) for p in parts]
            for f in futs:
                p, rows = f.result()
                grid[p] = rows
                print("grid", p, flush=True)
                gcache.write_text(json.dumps(grid, indent=1))

    # ---- one table of everything
    rows = []
    for p, st, m, _ in cs:
        g = next(r for r in grid[p] if r["stage"] == st and str(r["mag"]) == str(m))
        ex = exact[f"{p}|{st}|{m}"]
        rows.append({"part": p, "kind": p.split("-")[0], "op": p.split("-")[1], "stage": st, "mag": m,
                     # a failed exact boolean (exact_residual returns an error, never a number): the 0.5 mm parity
                     # grid's IoU stands in (within 0.0024 of the exact one on every pair of the first study)
                     "1-IoU": 1 - (ex["iou"] if ex.get("iou") is not None else g["voxel_iou"]),
                     "exact_failed": ex.get("iou") is None, "r_plus": ex.get("r_plus"),
                     "r_minus": ex.get("r_minus"), "genus": ex.get("model_genus"),
                     "1-IoU piece": 1 - g["piece_iou"], "voxel_iou": g["voxel_iou"],
                     "chi_-1": g["chi_-1"], "chi_0": g["chi_0"], "chi_0.45": g["chi_0.45"], "phi_1": g["phi_1"],
                     "piece_volume": g["piece_volume"]})
    (out / "data" / "stages.json").write_text(json.dumps(rows, indent=1))
    analyse(out, rows)


# ------------------------------------------------------------------------------------------------ analysis


def val(rows, part, stage, mag, key):
    for r in rows:
        if r["part"] == part and r["stage"] == stage and str(r["mag"]) == str(mag):
            return r[key]
    raise KeyError((part, stage, mag, key))


def equivalent_shift(rows, part, key, value):
    """the position error (mm, log-interpolated on the sweep) that the measure rates as bad as `value`."""
    xs = np.array([0.0] + SWEEPS["position"], float)
    ys = np.array([0.0] + [val(rows, part, "position", d, key) for d in SWEEPS["position"]], float)
    ys = np.maximum.accumulate(ys)                          # a plateau stays a plateau
    if value >= ys[-1]:
        return float("inf")
    return float(np.interp(value, ys, xs))


def analyse(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base.style(plt)
    parts = [f"{k}-{o}" for k in KINDS for o in OPS]
    keys = [k for k, _ in MEASURES]
    lab = dict(MEASURES)
    res = {}

    # (1) sweeps: median and range over the parts, per stage and measure
    fig, axs = plt.subplots(len(keys), 3, figsize=(7.4, 9.6), sharex="col")
    for j, st in enumerate(("position", "rotation", "dimensions")):
        vals = SWEEPS[st]
        x = [v - 1 if st == "dimensions" else v for v in vals]
        for i, key in enumerate(keys):
            ax = axs[i, j]
            for k, kind in enumerate(KINDS):
                for o, op in enumerate(OPS):
                    if kind == "sphere" and st == "rotation":
                        continue                          # a sphere has no rotation error (IoU 1 by symmetry)
                    y = [max(val(rows, f"{kind}-{op}", st, v, key), 1e-5) for v in vals]
                    ax.plot(x, y, color=base.SERIES[k], lw=1.0, ls="-" if op == "fuse" else "--",
                            marker=base.MARKERS[k], ms=2.5, label=f"{kind}" if o == 0 else None)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in x]); ax.minorticks_off()
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(lab[key], fontsize=8)
            if i == len(keys) - 1:
                ax.set_xlabel({"position": "shift [mm]", "rotation": "tilt [deg]",
                               "dimensions": "scale $-$ 1"}[st], fontsize=8)
            if i == 0:
                ax.set_title(f"{st} stage", fontsize=9)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, -0.015),
               title="tool (solid: fuse, dashed: cut)", title_fontsize=7.5)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out / "fig" / "stage_sweeps.pdf"); plt.close(fig)

    # (2) slopes (log-log, per part) and plateaus: does the measure keep growing with the error?
    slopes, plateau = {}, {}
    for st in ("position", "rotation", "dimensions"):
        vals = np.array(SWEEPS[st], float)
        x = vals - 1 if st == "dimensions" else vals
        for key in keys:
            ss, pl = [], []
            for p in parts:
                if p.startswith("sphere") and st == "rotation":
                    continue
                y = np.array([val(rows, p, st, v, key) for v in SWEEPS[st]], float)
                m = y > 0
                if m.sum() >= 2:
                    ss.append(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])
                pl.append(y[-1] / y[-2] if y[-2] > 0 else float("nan"))   # growth over the last step of the sweep
            slopes[(st, key)] = (float(np.median(ss)), float(np.min(ss)), float(np.max(ss)))
            plateau[(st, key)] = float(np.median(pl))
    # (3) part dependence: the spread of the measure over the 10 parts for one fixed error per stage
    fixed = dict(position=2, rotation=10, dimensions=1.1)
    spread = {}
    for st, v in fixed.items():
        for key in keys:
            y = np.array([val(rows, p, st, v, key) for p in parts
                          if not (p.startswith("sphere") and st == "rotation")], float)
            spread[(st, key)] = float(y.max() / y.min()) if y.min() > 0 else float("inf")
    # (4) type errors as equivalent shifts, and the missing stage's rank among all errors of its part
    eq = {}
    for key in keys:
        e = []
        for p in parts:
            for other in KINDS:
                if other == p.split("-")[0]:
                    continue
                e.append(equivalent_shift(rows, p, key, val(rows, p, "type", other, key)))
        eq[key] = e
    miss_rank = {}
    for key in keys:
        rk = []
        for p in parts:
            vs = [r[key] for r in rows if r["part"] == p and r["stage"] != "exact"]
            m = val(rows, p, "missing", 0, key)
            rk.append(1 + sum(v > m + 1e-12 for v in vs))
        miss_rank[key] = rk
    res = dict(slopes={f"{a}|{b}": v for (a, b), v in slopes.items()},
               plateau={f"{a}|{b}": v for (a, b), v in plateau.items()},
               spread={f"{a}|{b}": v for (a, b), v in spread.items()}, fixed=fixed,
               equivalent_shift=eq, missing_rank=miss_rank,
               n_errors_per_part=len([r for r in rows if r["part"] == parts[0] and r["stage"] != "exact"]))
    res["exact_failed"] = [f"{r['part']}|{r['stage']}|{r['mag']}" for r in rows if r.get("exact_failed")]
    (out / "data" / "stages_result.json").write_text(json.dumps(res, indent=1))

    # tables
    L = []
    for key in keys:
        cells = []
        for st in ("position", "rotation", "dimensions"):
            md, lo, hi = slopes[(st, key)]
            cells.append(f"{md:.2f} ({lo:.2f}--{hi:.2f})")
            cells.append(f"{plateau[(st, key)]:.2f}")
        L.append(f"{lab[key]} & " + " & ".join(cells) + " \\\\")
    (out / "data" / "stage_slopes.tex").write_text("\n".join(L + ["\\bottomrule"]) + "\n")
    L = []
    for key in keys:
        e = np.array(eq[key])
        fin = e[np.isfinite(e)]
        L.append(f"{lab[key]} & " + " & ".join(f"{spread[(st, key)]:.1f}" for st in fixed) +
                 f" & {np.median(fin) if len(fin) else float('nan'):.2f} & {np.min(fin) if len(fin) else float('nan'):.2f}--"
                 f"{np.max(fin) if len(fin) else float('nan'):.2f} & {int((~np.isfinite(e)).sum())}/{len(e)} & "
                 f"{np.median(miss_rank[key]):.0f} ({min(miss_rank[key])}--{max(miss_rank[key])}) \\\\")
    (out / "data" / "stage_summary.tex").write_text("\n".join(L + ["\\bottomrule"]) + "\n")
    # the full per-part table of the fixed errors and the type errors under the exact IoU
    L = []
    for p in parts:
        cells = [f"{val(rows, p, 'missing', 0, 'piece_volume'):.0f}", f"{1 - val(rows, p, 'missing', 0, '1-IoU'):.4f}"]
        for st, v in fixed.items():
            cells.append("--" if (p.startswith("sphere") and st == "rotation") else
                         f"{1 - val(rows, p, st, v, '1-IoU'):.4f}")
        ty = [(o, 1 - val(rows, p, "type", o, "1-IoU")) for o in KINDS if o != p.split("-")[0]]
        best = max(ty, key=lambda t: t[1])
        cells.append(f"{best[0]} {best[1]:.4f}")
        cells.append(f"{min(t[1] for t in ty):.4f}")
        L.append(f"{p} & " + " & ".join(cells) + " \\\\")
    (out / "data" / "stage_parts.tex").write_text("\n".join(L + ["\\bottomrule"]) + "\n")

    # (5) figure: equivalent shift of every type error, per measure
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    for i, key in enumerate(keys):
        e = np.array(eq[key])
        fin = np.where(np.isfinite(e), e, 24.0)
        jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(e))
        ax.scatter(i + jit, fin, s=10, color=base.SERIES[i], edgecolors="none", alpha=0.85)
        ax.plot([i - 0.3, i + 0.3], [np.median(fin)] * 2, color=base.INK, lw=1.5)
    ax.axhline(16, color=base.MUTED, lw=0.8, ls=":")
    ax.text(len(keys) - 0.5, 17.5, "beyond the sweep (> 16 mm)", fontsize=7, color=base.MUTED, ha="right")
    ax.set_yscale("log"); ax.set_ylim(0.05, 40)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([lab[k] for k in keys], fontsize=7.5)
    ax.set_ylabel("type error = shift of [mm]")
    fig.tight_layout(); fig.savefig(out / "fig" / "stage_type_equivalent.pdf"); plt.close(fig)

    # (6) gallery: the 10 parts and one error of each stage on the cylinder-cut part
    fig = plt.figure(figsize=(7.4, 3.3))
    for i, p in enumerate(parts):
        kind, op = p.split("-")
        ax = fig.add_subplot(2, 5, (i // 2) + 1 + 5 * (i % 2), projection="3d")
        base.draw_mesh(ax, part_manifold(op, tool(kind)), "#9a9992", 1.0)
        base.frame3d(ax, lo=(-2, -2, -2), hi=(62, 42, 34))
        ax.set_title(p, fontsize=8)
    fig.subplots_adjust(wspace=0, hspace=0.05)
    fig.savefig(out / "fig" / "stage_parts.png"); plt.close(fig)
    fig = plt.figure(figsize=(7.4, 1.9))
    ex = [("missing", 0, None), ("type", "box", same_volume("box", "cylinder")), ("rotation", 10, tool("cylinder", tilt=10)),
          ("dimensions", 1.1, tool("cylinder", scale=1.1)), ("position", 2, tool("cylinder", shift=2))]
    A = part_manifold("cut", tool("cylinder"))
    for j, (st, m, t) in enumerate(ex):
        B = part_manifold("cut", t)
        ax = fig.add_subplot(1, 5, j + 1, projection="3d")
        base.draw_mesh(ax, A, "#9a9992", 0.10)
        base.draw_mesh(ax, A - B, base.RPLUS, 0.95)
        base.draw_mesh(ax, B - A, base.RMINUS, 0.95)
        base.frame3d(ax, lo=(-2, -2, -2), hi=(62, 42, 34))
        ax.set_title(f"{st} ({m})" if m else st, fontsize=8)
        ax.text2D(0.5, 0.02, f"IoU {1 - val(rows, 'cylinder-cut', st, m, '1-IoU'):.4f}", transform=ax.transAxes,
                  ha="center", fontsize=7)
    fig.subplots_adjust(wspace=0)
    fig.savefig(out / "fig" / "stage_errors_cylinder_cut.png"); plt.close(fig)
    print(json.dumps(dict(slopes=res["slopes"], plateau=res["plateau"], spread=res["spread"],
                          eq_median={k: float(np.median(np.where(np.isfinite(v), v, 99))) for k, v in eq.items()},
                          missing_rank=miss_rank), indent=0))


if __name__ == "__main__":
    main()
