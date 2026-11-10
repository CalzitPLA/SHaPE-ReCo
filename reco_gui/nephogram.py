#!/usr/bin/env python3
"""
The comparison export for the user's decision (the user's rule of 2026-10-15: when asked to keep a result or try
another path, export the guess and the input, compute the surface distance and show it as an HSV nephogram in FreeCAD
or CloudCompare): both meshes as STL, and both as PLY with a per-vertex `distance` scalar (CloudCompare reads it as a
scalar field) and vertex colours from blue (0) over green and yellow to red (the scale, default the 95th percentile of
the larger way), with the statistics in stats.json and a README. Qt-free; the GUI's decision bar calls it in a thread.

    info = write_nephogram(input_mesh, model_mesh, out_dir, name="part")    # MeshData-like: .vertices, .triangles
    cmd = cloudcompare_command()                                           # ["flatpak", "run", "org.cloudcompare.CloudCompare"] or None
    open_in_cloudcompare(info["files"]["ply"])                             # Popen, the GUI is not blocked

The distance of a vertex is the exact distance to the other mesh's surface (src.closest_point_field.SurfaceField, the
same query the loop's channels use), not a vertex-to-vertex distance.
"""

from __future__ import annotations

import colorsys
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "dataset_gen") not in sys.path:
    sys.path.insert(0, str(ROOT / "dataset_gen"))


BANDS = 8                        # the discrete scale (the user's rule 2026-10-15: colours linked to values): bands of
                                 # scale / BANDS each, the last one open at the top


def band_of(distance: np.ndarray, scale: float, bands: int = BANDS) -> np.ndarray:
    """The band index 0 .. bands-1 of every distance: k = floor(d / (scale / bands)), the last band open."""
    width = max(float(scale), 1e-12) / bands
    return np.clip(np.floor(np.asarray(distance, float) / width), 0, bands - 1).astype(int)


def band_colour(k: int, bands: int = BANDS) -> np.ndarray:
    """RGB (0..255) of band k: HSV hue 240 deg (blue) for the first band to 0 deg (red) for the last."""
    t = k / max(1, bands - 1)
    return np.round(np.array(colorsys.hsv_to_rgb((1.0 - t) * 2.0 / 3.0, 1.0, 1.0)) * 255).astype(np.uint8)


def legend(scale: float, bands: int = BANDS, unit: str = "mm") -> List[Dict]:
    """The discrete legend: [{band, lo, hi, text, rgb}] with hi None for the open last band."""
    width = max(float(scale), 1e-12) / bands
    out = []
    for k in range(bands):
        lo, hi = k * width, (k + 1) * width if k < bands - 1 else None
        text = f"{lo:.3g} - {hi:.3g} {unit}" if hi is not None else f"> {lo:.3g} {unit}"
        out.append(dict(band=k, lo=float(lo), hi=None if hi is None else float(hi), text=text, rgb=[int(c) for c in band_colour(k, bands)]))
    return out


def hsv_colours(distance: np.ndarray, scale: float, bands: int = BANDS) -> np.ndarray:
    """(n, 4) uint8 RGBA, DISCRETE: every distance gets the colour of its band (blue for the first band at 0 to red
    for the last, open band at `scale` and beyond), so every colour on the mesh is one value band of the legend."""
    k = band_of(distance, scale, bands)
    table = np.array([band_colour(i, bands) for i in range(bands)], np.uint8)
    out = np.empty((len(k), 4), np.uint8)
    out[:, :3] = table[k]
    out[:, 3] = 255
    return out


def triangle_colours(vertex_distance: np.ndarray, triangles: np.ndarray, scale: float) -> np.ndarray:
    """(t, 4) uint8 colours per triangle: the mean of its vertices' distances through `hsv_colours` (the 3D view colours
    per triangle, reco_gui.cascade_session.ColouredMesh)."""
    d = np.asarray(vertex_distance, float)[np.asarray(triangles, np.int64)].mean(axis=1)
    return hsv_colours(d, scale)


def _stats(d: np.ndarray) -> Dict[str, float]:
    d = np.asarray(d, float)
    return dict(n=int(len(d)), mean=float(d.mean()), median=float(np.median(d)), p95=float(np.percentile(d, 95)),
                max=float(d.max()))


MAX_EXACT_POINTS = 50_000        # exact B-rep distances per direction at most (about 4 minutes in freecadcmd); above
                                 # it a random subset is exact and the rest comes from the mesh field, said in the README


def write_histogram(info: Dict, path, unit: str = "mm") -> Optional[str]:
    """The histograms that belong to a nephogram (the user's rule 2026-10-29: "each nephogram has always corresponding
    histograms"): the share of the vertices per distance band, both ways (input -> model: what is missing; model ->
    input: what is invented), in the bands' own colours, with n, mean, p95 and max. PNG; None without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    lg, stats = info.get("legend") or legend(info["scale"]), info["stats"]
    cols = [np.array(b["rgb"]) / 255.0 for b in lg]
    labels = [(f"{b['lo']:.3g}-{b['hi']:.3g}" if b["hi"] is not None else f"> {b['lo']:.3g}") for b in lg]
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 4.8), sharex=True)
    for ax, (key, title) in zip(axes, (("input_to_model", "input -> model: what is missing"),
                                        ("model_to_input", "model -> input: what is invented"))):
        st = stats[key]
        share = np.array(st.get("band_share") or [0.0] * len(lg)) * 100.0
        ax.bar(range(len(share)), share, color=cols, edgecolor="none")
        for i, v in enumerate(share):
            if v >= 1:
                ax.text(i, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=8, color="#333333")
        ax.set_xticks(range(len(share)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 112)
        ax.set_title(title, fontsize=10)
        ax.text(0.98, 0.96, f"n = {st['n']}\nmean {st['mean']:.3g} {unit}\np95 {st['p95']:.3g} {unit}\nmax {st['max']:.3g} {unit}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#333333")
        ax.set_ylabel("vertices (%)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1].set_xlabel(f"distance band ({unit})")
    fig.suptitle(f"{info.get('name', 'part')}: {len(lg)} bands of {info['scale'] / len(lg):.3g} {unit}, the nephogram's colours",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(str(path), dpi=130)
    plt.close(fig)
    return str(path)


def write_nephogram(input_mesh, model_mesh, out_dir, name: str = "part", scale: Optional[float] = None,
                    input_path=None, model_step=None, max_exact_points: int = MAX_EXACT_POINTS) -> Dict:
    """Writes <out_dir>/{input,model}.stl, {input_to_model,model_to_input}.ply (colour + scalar `distance`),
    stats.json and README.txt. Returns dict(dir, files, stats, scale, method). With `input_path` (the input's STEP,
    or its STL) and `model_step` (the program built as STEP) the distances are EXACT: every vertex of a mesh to the
    other side's B-rep by OCCT (reco_field.exact_check.point_distances; the user's rule 2026-10-15: between STEP
    files, not meshes), signed by the material flag; without them, the exact closest-point field of the meshes."""
    import trimesh
    from src.closest_point_field import SurfaceField
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meshes = {}
    for key, m in (("input", input_mesh), ("model", model_mesh)):
        v = np.asarray(m.vertices, float)
        f = np.asarray(m.triangles, np.int64)
        meshes[key] = (v, f, SurfaceField(v, f))
    dist = {"input_to_model": meshes["model"][2].query(meshes["input"][0]).distance,
            "model_to_input": meshes["input"][2].query(meshes["model"][0]).distance}
    dist = {k: np.abs(np.asarray(d, float)) for k, d in dist.items()}
    method, exact_note, inside = "mesh field (exact closest point on the tessellations)", "", {}
    if input_path is not None and model_step is not None and Path(str(model_step)).is_file():
        from reco_field.exact_check import point_distances
        notes = []
        for key, pts_key, target in (("input_to_model", "input", model_step), ("model_to_input", "model", input_path)):
            pts = meshes[pts_key][0]
            idx = None
            if len(pts) > max_exact_points:
                idx = np.sort(np.random.default_rng(0).choice(len(pts), max_exact_points, replace=False))
            r = point_distances(str(target), pts if idx is None else pts[idx], signed=True)
            if "error" in r:
                notes.append(f"{key}: {r['error']} - the mesh field stands in")
                continue
            d = dist[key].copy()
            if idx is None:
                d[:] = r["distance"]
                inside[key] = r["inside"]
            else:
                d[idx] = r["distance"]
                notes.append(f"{key}: exact at {len(idx)} of {len(pts)} vertices (the cap), the rest from the mesh field")
            dist[key] = d
        method = "exact B-rep (OCCT distToShape per vertex, signed by isInside)" + (" with notes" if notes else "")
        exact_note = "\n".join(notes)
    stats = {k: _stats(d) for k, d in dist.items()}
    if scale is None:
        scale = max(stats["input_to_model"]["p95"], stats["model_to_input"]["p95"], 1e-9)
    files: Dict[str, List[str]] = dict(stl=[], ply=[])
    for key in ("input", "model"):
        v, f, _ = meshes[key]
        stl = out / f"{key}.stl"
        trimesh.Trimesh(v, f, process=False).export(str(stl))
        files["stl"].append(str(stl))
    for key, src in (("input_to_model", "input"), ("model_to_input", "model")):
        v, f, _ = meshes[src]
        tm = trimesh.Trimesh(v, f, process=False)
        tm.visual.vertex_colors = hsv_colours(dist[key], scale)
        tm.vertex_attributes["distance"] = dist[key].astype(np.float32)
        ply = out / f"{key}.ply"
        tm.export(str(ply), encoding="binary")
        files["ply"].append(str(ply))
    info = dict(dir=str(out), name=name, files=files, stats=stats, scale=float(scale), bands=BANDS, legend=legend(scale),
                method=method, method_notes=exact_note, seconds=time.time() - t0,
                inside_share={k: float(np.mean(v)) for k, v in inside.items()},
                chamfer=float(0.5 * (stats["input_to_model"]["mean"] + stats["model_to_input"]["mean"])))
    for key in ("input_to_model", "model_to_input"):            # how much of the mesh falls in every band
        counts = np.bincount(band_of(dist[key], scale), minlength=BANDS)
        stats[key]["band_share"] = [float(c) / max(1, len(dist[key])) for c in counts]
    info["files"]["histogram"] = write_histogram(info, out / "histogram.png")   # always with the nephogram
    (out / "stats.json").write_text(json.dumps(info, indent=1))
    info["distance"] = dist                                   # the per-vertex distances (arrays; not in stats.json)
    info["files"]["macro"] = write_freecad_macro(info)
    (out / "README.txt").write_text(
        f"{name}: the input and the reconstruction (the guess), exported {time.strftime('%Y-%m-%d %H:%M')}\n"
        f"distances: {method}" + (f"\n{exact_note}" if exact_note else "") + "\n\n"
        f"input.stl, model.stl            the two meshes\n"
        f"input_to_model.ply              the input mesh, every vertex coloured by its distance to the model's surface\n"
        f"model_to_input.ply              the model mesh, every vertex coloured by its distance to the input's surface\n"
        f"histogram.png                   the share of the vertices per band, both ways (always with the nephogram)\n"
        f"colour scale, discrete          {BANDS} bands of {scale / BANDS:.4g} each, blue -> green -> yellow -> red (HSV hue 240 -> 0 deg):\n"
        + "".join(f"    band {b['band']}: {b['text']:>22s}   rgb {tuple(b['rgb'])}   input {stats['input_to_model']['band_share'][b['band']]:.1%}, "
                  f"model {stats['model_to_input']['band_share'][b['band']]:.1%} of the vertices\n" for b in info["legend"])
        + 
        f"PLY scalar field 'distance'     CloudCompare: Properties > Scalar field 'distance', colour scale as you like\n"
        f"FreeCAD                         `freecad nephogram.FCMacro` (or File > Macro): the PLYs as Mesh objects with Coloring on\n\n"
        f"input -> model: mean {stats['input_to_model']['mean']:.4g}, median {stats['input_to_model']['median']:.4g}, "
        f"p95 {stats['input_to_model']['p95']:.4g}, max {stats['input_to_model']['max']:.4g}\n"
        f"model -> input: mean {stats['model_to_input']['mean']:.4g}, median {stats['model_to_input']['median']:.4g}, "
        f"p95 {stats['model_to_input']['p95']:.4g}, max {stats['model_to_input']['max']:.4g}\n"
        f"two-way chamfer (mean of the means): {info['chamfer']:.4g}\n")
    return info


RUN_DIR = "nephogram"             # a stored run's nephogram: <run dir>/nephogram/ (stats.json, histogram.png, PLYs, macro)


class _Mesh:
    def __init__(self, vertices, triangles):
        self.vertices, self.triangles = np.asarray(vertices, float), np.asarray(triangles, np.int64)


def stored_nephogram(run_dir, load_mesh=None, timeout: int = 900) -> Dict:
    """The nephogram (and its histograms) of a stored run, into <run dir>/nephogram/: the input (its file) against
    the model - the run's model.stl when it was saved with one, else the program built by FreeCAD (most GUI runs are
    saved without the rebuild) - with exact distances against the STEPs where the input is one."""
    from reco_engine import benchmark as bm
    from reco_engine import run_store
    run = run_store.load_run(run_dir)
    d = Path(run["path"])
    inp = (run.get("input") or {}).get("path")
    if not inp or not Path(inp).is_file():
        raise FileNotFoundError(f"the run's input is not there: {inp}")
    if load_mesh is None:
        from src.closest_point_field import load_mesh
    m = run.get("model") or {}
    stl = d / m["stl"] if m.get("stl") and (d / m["stl"]).is_file() else None
    step = d / m["step"] if m.get("step") and (d / m["step"]).is_file() else None
    if stl is None:
        result = bm.result_of(run)
        if result is None:
            raise ValueError("the run has no readable program")
        from reco_engine.result_to_program import program_model
        from reco_field.export import build_in_freecad
        ex = build_in_freecad(program_model(result), d / RUN_DIR, "model_built", timeout=timeout)
        stl, step = Path(ex.stl), Path(ex.step)
    iv, itri = load_mesh(str(inp))
    mv, mtri = load_mesh(str(stl))
    exact = str(inp).lower().endswith((".step", ".stp"))
    return write_nephogram(_Mesh(iv, itri), _Mesh(mv, mtri), d / RUN_DIR, name=str(run.get("name") or d.name),
                           input_path=str(inp) if exact else None, model_step=str(step) if exact and step else None)


MACRO = r"""# nephogram.FCMacro - written by SHaPe ReCo (reco_gui/nephogram.py) on {date}: the input and the guess (the
# reconstruction) of {name} coloured by their surface distance. Run it in FreeCAD (File > Macro, or
# `freecad nephogram.FCMacro`): the two PLY meshes are imported as Mesh objects and their view property Coloring is
# switched on, which shows the per-vertex colours FreeCAD's importer (Mesh.insert) read from the PLY into VertexColors. Colour scale: blue 0 -> green -> yellow
# -> red {scale:.4g} {unit} (HSV hue 240 -> 0 deg).
import FreeCAD, Mesh
doc = FreeCAD.newDocument("nephogram_{name}")
items = [("input_to_model.ply", "input, coloured by its distance to the guess", True),
         ("model_to_input.ply", "guess, coloured by its distance to the input", False),
         ("input.stl", "input (plain)", False), ("model.stl", "guess (plain)", False)]
objs = []
for file, label, visible in items:
    path = r"{folder}/" + file
    before = set(o.Name for o in doc.Objects)
    try:
        Mesh.insert(path, doc.Name)        # the importer keeps the PLY's colours as the property VertexColors
    except Exception as exc:               # (verified in freecadcmd 1.0 on 2026-10-15; Mesh.Mesh(path) drops them)
        FreeCAD.Console.PrintError("nephogram: could not read " + path + ": " + str(exc) + "\n")
        continue
    new = [o for o in doc.Objects if o.Name not in before]
    if not new:
        continue
    obj = new[-1]
    obj.Label = label
    objs.append((obj, visible))
doc.recompute()
try:
    import FreeCADGui
    for obj, visible in objs:
        vo = obj.ViewObject
        if vo is None:
            continue
        vo.Visibility = visible
        if "VertexColors" in obj.PropertiesList or "FaceColors" in obj.PropertiesList:
            vo.Coloring = True                                   # the per-vertex colours from the file
        else:
            vo.ShapeColor = (0.75, 0.75, 0.75)
            vo.Transparency = 60
    FreeCADGui.SendMsgToActiveView("ViewFit")
    FreeCADGui.activeDocument().activeView().viewIsometric()
except ImportError:
    pass
try:                                       # the histograms belong to the nephogram: shown beside it
    import os, ImageGui
    if os.path.isfile(r"{folder}/histogram.png"):
        ImageGui.open(r"{folder}/histogram.png")
except Exception:
    pass
FreeCAD.Console.PrintMessage("nephogram {name}: the discrete colour scale\n{legend_lines}")
FreeCAD.Console.PrintMessage(
    "nephogram {name}: colour scale blue 0 -> red {scale:.4g} {unit}; input -> guess mean {i_mean:.4g} max {i_max:.4g}; "
    "guess -> input mean {m_mean:.4g} max {m_max:.4g}; two-way chamfer {chamfer:.4g}. Show/hide the four objects in the tree; "
    "the plain meshes are the uncoloured copies. If a coloured mesh shows grey, set its view property Coloring to true.\n")
"""


def write_freecad_macro(info: Dict, unit: str = "mm") -> str:
    """nephogram.FCMacro beside the export (see MACRO); returns its path."""
    s = info["stats"]
    folder = str(Path(info["dir"]).resolve())
    legend_lines = "".join(f"    band {b['band']}: {b['text']}   rgb {tuple(b['rgb'])}\\n" for b in info.get("legend", legend(info["scale"])))
    text = MACRO.format(date=time.strftime("%Y-%m-%d %H:%M"), name=info.get("name", "part"), scale=info["scale"], unit=unit,
                        legend_lines=legend_lines,
                        folder=folder, i_mean=s["input_to_model"]["mean"], i_max=s["input_to_model"]["max"],
                        m_mean=s["model_to_input"]["mean"], m_max=s["model_to_input"]["max"], chamfer=info["chamfer"])
    path = Path(info["dir"]) / "nephogram.FCMacro"
    path.write_text(text)
    return str(path)


def freecad_command() -> Optional[List[str]]:
    """The FreeCAD GUI here: `freecad` / `FreeCAD` on the PATH (Debian: /usr/bin/freecad), else None. freecadcmd is
    not it (no GUI)."""
    for exe in ("freecad", "FreeCAD"):
        p = shutil.which(exe)
        if p:
            return [p]
    return None


def open_in_freecad(macro: str) -> Optional[subprocess.Popen]:
    """FreeCAD with the macro run at start (FreeCAD executes a .FCMacro / .py given on its command line); None when
    FreeCAD is not installed. The user's tool: started on the user's click, never by the framework on its own."""
    cmd = freecad_command()
    if cmd is None:
        return None
    return subprocess.Popen(cmd + [str(macro)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cloudcompare_command() -> Optional[List[str]]:
    """How CloudCompare is started here: a binary on the PATH, else the flatpak org.cloudcompare.CloudCompare, else None."""
    for exe in ("CloudCompare", "cloudcompare"):
        p = shutil.which(exe)
        if p:
            return [p]
    flatpak = shutil.which("flatpak")
    if flatpak:
        try:
            out = subprocess.run([flatpak, "list", "--app", "--columns=application"], capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        if "org.cloudcompare.CloudCompare" in out:
            return [flatpak, "run", "org.cloudcompare.CloudCompare"]
    return None


def open_in_cloudcompare(files: Sequence[str]) -> Optional[subprocess.Popen]:
    """CloudCompare with `files` loaded (the PLYs: scalar field `distance`); None when it is not installed."""
    cmd = cloudcompare_command()
    if cmd is None:
        return None
    return subprocess.Popen(cmd + [str(f) for f in files], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
