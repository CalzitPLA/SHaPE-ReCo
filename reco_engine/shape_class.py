#!/usr/bin/env python3
"""
Shape classes (CLASS_V1): one versioned key per part from its classical topology, its surface family and the complexity
of its CSG tree - so reconstruction results can be counted per kind of geometry instead of per part
(scripts/shape_atlas.py; plan of 2026-10-29: "find out on which geometry types we lack").

    d = describe("part.step")           # the facts: Betti numbers, faces, carrier surfaces, counts, mesh descriptors
    t = tree_facts(program_dict)        # a known tree (truth.program_json / aegir): leaves, depth, operations, partial leaves
    c = classify(d, t)                  # dict(key "g1.M.C2-3", T, S, C, c_source, flags)

The three axes:
  T  topology - the Betti numbers of the solid: b0 solids, b1 the genus summed over all shells (chi(S) = chi(dS) / 2,
     half lives half dies), b2 the inner shells (voids). `multi` for b0 > 1, else a genus bin (GENUS_BINS) with a
     suffix `v` for b2 >= 1. b1 is also split into `b1_holes` (handles a through hole stack explains, hole_stack.py) and
     `b1_rest` (window/frame handles) - a descriptor, not part of the key.
  S  surface family - from the faces' types and area shares: P planes only, A axial / turned (curved area share >=
     1/3, of it >= 80 % on one axis line), D doubly curved (sphere + torus >= 5 % of the area), M planes with curved
     features (the rest), F free-form faces present (out of the loop's scope). Flag `blend`: narrow cylinder or
     torus strips (at most 0.3 of the full surface) - fillets and rounds on a real part, sectors in a random tree.
  C  complexity of the CSG tree - leaf-count bins (LEAF_BINS) of the known tree; without one, the leaves estimated from
     the B-rep's carrier surfaces by a calibration on the known trees (`calib`, scripts/shape_atlas.py calibrate), or -
     without a calibration that passes - honest bins of the carrier surfaces themselves (CARRIER_BINS, c_source "brep").

STEP input is read by freecadcmd once (reco_field.exact_check.exact_facts) and cached in ~/.cache/shape_reco/shape_class/
($SHAPE_RECO_SHAPE_CLASS_CACHE, "off" disables), keyed by the file's bytes and the job script; STL input is a marked
fallback (source "mesh": Betti numbers from the mesh, no faces, S "?").
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

CLASS_VERSION = "v1"
GENUS_BINS = ((0, 0, "g0"), (1, 1, "g1"), (2, 3, "g2-3"), (4, 9, "g4-9"), (10, None, "g10+"))
LEAF_BINS = ((1, 1, "C1"), (2, 3, "C2-3"), (4, 7, "C4-7"), (8, 15, "C8-15"), (16, None, "C16+"))
CARRIER_BINS = ((1, 6, "K1-6"), (7, 15, "K7-15"), (16, 40, "K16-40"), (41, 100, "K41-100"), (101, None, "K101+"))
FAMILIES = ("P", "A", "M", "D", "F")
AXIAL = ("cylinder", "cone", "torus")
CURVED = ("cylinder", "cone", "sphere", "torus")
TURNED_CURVED_SHARE = 1.0 / 3.0     # A: curved area share at least this ...
TURNED_LINE_SHARE = 0.8             # ... and of the curved area at least this on one axis line
DOUBLY_SHARE = 0.05                 # D: sphere + torus area share at least this
STRIP_COVERAGE = 0.3                # a cylinder or torus face covering at most this share of its surface: a blend
THIN_REL = 0.02                     # flag thin: the 5th percentile wall thickness below this share of the diagonal
CALIB_MIN_ACCURACY = 0.6            # a leaf estimator is used only when it gets at least this share of the bins right
LINE_COS = math.cos(math.radians(1.0))
FULL_TURN = 2.0 * math.pi


def _bin(x, bins) -> Optional[str]:
    if x is None:
        return None
    for lo, hi, name in bins:
        if x >= lo and (hi is None or x <= hi):
            return name
    return None


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _cache_dir() -> Optional[Path]:
    env = os.environ.get("SHAPE_RECO_SHAPE_CLASS_CACHE", "").strip()
    if env.lower() in ("0", "off", "no", "false"):
        return None
    return Path(env) if env else Path.home() / ".cache" / "shape_reco" / "shape_class"


# ---------------------------------------------------------------------------------------------------- the faces

def _canonical(a: np.ndarray) -> np.ndarray:
    a = _unit(a)
    return a if a[int(np.argmax(np.abs(a)))] > 0 else -a


def axis_lines(faces: List[Dict], tol: float) -> List[Dict]:
    """The distinct axis lines of the axial faces (cylinder, cone, torus), the one with the most area first:
    [dict(axis, foot, area, faces)]; two faces share a line when their axes are parallel within 1 degree and the second's
    centre lies within `tol` of the first's line."""
    lines: List[Dict] = []
    for i, f in sorted(enumerate(faces), key=lambda q: -q[1]["area"]):
        if f["type"] not in AXIAL or "axis" not in f or "center" not in f:
            continue
        a, c = _canonical(f["axis"]), np.asarray(f["center"], float)
        for ln in lines:
            off = c - ln["foot"]
            if abs(a @ ln["axis"]) >= LINE_COS and np.linalg.norm(off - ln["axis"] * (off @ ln["axis"])) <= tol:
                ln["area"] += f["area"]
                ln["faces"].append(i)
                break
        else:
            lines.append(dict(axis=a, foot=c - (c @ a) * a, area=float(f["area"]), faces=[i]))
    return sorted(lines, key=lambda ln: -ln["area"])


def carrier_surfaces(faces: List[Dict], tol: float) -> Counter:
    """The distinct surfaces the faces lie on, counted by kind (as start_shape.surface_inventory groups a STEP's faces:
    planes by normal and offset, cylinders by axis line and radius, ...); a free-form face counts as its own surface."""
    keys = set()
    for i, f in enumerate(faces):
        t = f["type"]
        if t == "plane":
            n = _canonical(f["normal"])
            keys.add(("plane", tuple(np.round(n, 3)), round(float(n @ np.asarray(f["point"], float)) / tol)))
        elif t in AXIAL and "axis" in f and "center" in f:
            a, c = _canonical(f["axis"]), np.asarray(f["center"], float)
            foot = c - (c @ a) * a
            if t == "cylinder":
                shape = (round(float(f["radius"]) / tol),)
            elif t == "cone":
                apex = np.asarray(f.get("apex", f["center"]), float)
                shape = (tuple(np.round(apex / tol)), round(float(f.get("semi_angle", 0.0)), 3))
            else:
                shape = (tuple(np.round(c / tol)), round(float(f.get("major_radius", 0.0)) / tol),
                         round(float(f.get("minor_radius", 0.0)) / tol))
            keys.add((t, tuple(np.round(a, 3)), tuple(np.round(foot / tol)), shape))
        elif t == "sphere" and "center" in f:
            keys.add(("sphere", tuple(np.round(np.asarray(f["center"], float) / tol)), round(float(f["radius"]) / tol)))
        else:
            keys.add(("other", i))
    return Counter(k[0] for k in keys)


def angular_coverage(f: Dict) -> Optional[float]:
    """The share of the full surface a cylinder or torus face covers: a cylinder's area over 2 pi r times its reach
    along the axis, a torus's over 4 pi^2 R r."""
    if f["type"] == "torus" and f.get("major_radius") and f.get("minor_radius"):
        return float(f["area"]) / (FULL_TURN ** 2 * float(f["major_radius"]) * float(f["minor_radius"]))
    if f["type"] != "cylinder" or not f.get("along") or not f.get("radius"):
        return None
    h = float(f["along"][1]) - float(f["along"][0])
    return float(f["area"]) / (FULL_TURN * float(f["radius"]) * h) if h > 0 else None


def through_holes(faces: List[Dict], V: Optional[np.ndarray] = None) -> int:
    """The through holes of a part: hole stacks (reco_engine.hole_stack) of its concave cylinder and cone faces, open at
    both ends. hole_stacks reads a piece (the hole's material: faces convex, caps looking out of it), so the part's
    concave faces are passed as convex and its planes with the normal turned round."""
    from reco_engine.hole_stack import hole_stacks
    piece = []
    for f in faces:
        if f["type"] in ("cylinder", "cone") and f.get("convex") is False:
            piece.append(dict(f, convex=True, origin="input"))
        elif f["type"] == "plane":
            piece.append(dict(f, normal=[-float(x) for x in f["normal"]], origin="input"))
    stacks = hole_stacks(piece, V)
    return sum(1 for s in stacks if not s["closed_lo"] and not s["closed_hi"])


def surface_facts(faces: List[Dict], diag: float, V: Optional[np.ndarray] = None) -> Dict:
    """Face kinds, area shares, carrier surfaces, axis lines, blends, through holes and the surface family S."""
    tol = 1e-3 * max(diag, 1e-9)
    kinds = Counter(f["type"] for f in faces)
    area: Dict[str, float] = {}
    for f in faces:
        area[f["type"]] = area.get(f["type"], 0.0) + float(f["area"])
    total = sum(area.values()) or 1.0
    share = {k: v / total for k, v in area.items()}
    curved = sum(share.get(k, 0.0) for k in CURVED)
    lines = axis_lines(faces, tol)
    curved_area = sum(area.get(k, 0.0) for k in CURVED)
    line_share = (lines[0]["area"] / curved_area) if lines and curved_area > 0 else 0.0
    blends = sum(1 for f in faces if (angular_coverage(f) or 1.0) <= STRIP_COVERAGE)
    concave = sum(1 for f in faces if f["type"] in CURVED and f.get("convex") is False)
    free = kinds.get("other", 0)
    if free:
        family = "F"
    elif not curved_area:
        family = "P"
    elif curved >= TURNED_CURVED_SHARE and line_share >= TURNED_LINE_SHARE:
        family = "A"
    elif share.get("sphere", 0.0) + share.get("torus", 0.0) >= DOUBLY_SHARE:
        family = "D"
    else:
        family = "M"
    carriers = carrier_surfaces(faces, tol)
    try:
        holes = through_holes(faces, V)
    except Exception as e:                                     # a descriptor, never a reason to lose the part
        holes = None
        print(f"through_holes: {type(e).__name__}: {e}")
    return dict(face_kinds=dict(kinds), area_share=share, curved_share=curved, axis_lines=len(lines),
                main_line_share=line_share, blends=blends, concave_curved=concave, free_form=free, family=family,
                carriers=int(sum(carriers.values())), carriers_by_kind=dict(carriers), through_holes=holes)


# ---------------------------------------------------------------------------------------------------- the mesh

def mesh_descriptors(V: np.ndarray, F: np.ndarray, rays: int = 2000) -> Dict:
    """Size-free shape descriptors of a closed mesh (as docs/reports/abc_benchmark/abc_shape.py's): convexity (volume
    over the hull's), fill of the oriented box, thickness_rel = 2 V / (A d), aspect, sphericity, and tau_rel - the 5th
    percentile wall thickness (src.local_grids.wall_thickness) over the diagonal."""
    import trimesh
    from src.local_grids import wall_thickness
    m = trimesh.Trimesh(np.asarray(V, float), np.asarray(F), process=True)
    vol, area = abs(float(m.volume)), float(m.area)
    ext = np.sort(m.bounding_box_oriented.primitive.extents)
    d = float(np.linalg.norm(m.extents))
    out = dict(diag=d, watertight=bool(m.is_watertight), aspect=float(ext[2] / max(ext[0], 1e-12)))
    if vol > 0 and area > 0:
        out.update(convexity=vol / max(float(m.convex_hull.volume), 1e-30), fill_obb=vol / max(float(np.prod(ext)), 1e-30),
                   thickness_rel=2 * vol / (area * d), sphericity=math.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / area)
    try:
        _, _, tau = wall_thickness(m.vertices, m.faces, n=rays)
        tau = tau[np.isfinite(tau) & (tau > 0)]
        out["tau_rel"] = float(np.percentile(tau, 5)) / d if len(tau) else None
    except Exception as e:
        print(f"wall_thickness: {type(e).__name__}: {e}")
    return out


def mesh_betti(V: np.ndarray, F: np.ndarray) -> Dict:
    """b0, b1, b2 of a closed mesh without a B-rep: its connected pieces, outer (positive volume) or voids (negative,
    for a consistently wound mesh), b1 the genus summed over the pieces; b1 None when a piece is open."""
    import trimesh
    m = trimesh.Trimesh(np.asarray(V, float), np.asarray(F), process=True)
    b0 = b2 = 0
    b1: Optional[int] = 0
    shells = []
    for p in m.split(only_watertight=False):
        if p.volume >= 0:
            b0 += 1
        else:
            b2 += 1
        if not p.is_watertight:
            b1 = None
            continue
        g = int(round((2 - p.euler_number) / 2))
        shells.append(g)
        b1 = None if b1 is None else b1 + g
    return dict(b0=b0, b1=b1, b2=b2, shells=b0 + b2, shell_genus=sorted(shells, reverse=True))


# ---------------------------------------------------------------------------------------------------- describe

def _step_facts(path: Path, timeout: int) -> Dict:
    """exact_facts of a STEP file plus the path of its tessellation, cached by the file's bytes and the job script."""
    from reco_field.exact_check import SCRIPT, exact_facts
    key = hashlib.sha1(path.read_bytes() + SCRIPT.encode() + CLASS_VERSION.encode()).hexdigest()
    folder = _cache_dir()
    out = (folder / key) if folder is not None else Path(tempfile.mkdtemp(prefix="shape_class_"))
    js, stl = out / "facts.json", out / "mesh.stl"
    if js.is_file() and stl.is_file():
        facts = json.loads(js.read_text())
    else:
        out.mkdir(parents=True, exist_ok=True)
        facts = exact_facts(path, stl_out=stl, timeout=timeout)
        if "error" in facts:
            return facts
        tmp = js.with_suffix(".tmp")
        tmp.write_text(json.dumps(facts))
        tmp.replace(js)
    facts["stl"], facts["cache"] = str(stl), str(out)
    return facts


EP_SCRIPT = r'''
import json, os, Part
job = json.load(open(os.environ["EP_JOB"]))
shape = Part.read(job["step"])
solids = []
for s in shape.Solids:
    V, F, S = len(s.Vertexes), len(s.Faces), len(s.Shells)
    E = sum(1 for e in s.Edges if not e.Degenerated)
    L = sum(len(f.Wires) for f in s.Faces)
    solids.append(dict(shells=S, genus=S - (V - E + F - (L - F)) / 2.0))
json.dump(dict(solids=solids), open(job["out"], "w"))
'''


def euler_poincare(path: Path, folder: Path, timeout: int = 300) -> Optional[Dict]:
    """b0, b1, b2 from the B-rep's own counts, V - E + F - (L - F) = 2 (S - G) per solid (L wires, S shells; degenerated
    edges not counted; docs/reports/step_construction_review/step_facts.py) - the fallback when the tessellation is not
    closed (1.8 % of the ABC parts, 2026-10-29). Equal to the tessellation's b1 on 60 of 60 gate parts. Cached as ep.json
    beside facts.json; None when FreeCAD gives no answer or a genus is not an integer."""
    import subprocess
    from reco_field.export import find_freecadcmd
    js = folder / "ep.json"
    if not js.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        script, job = folder / "ep.py", folder / "ep_job.json"
        script.write_text(EP_SCRIPT)
        job.write_text(json.dumps(dict(step=str(path), out=str(js))))
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env["EP_JOB"] = str(job)
        try:
            subprocess.run([find_freecadcmd(), str(script)], env=env, capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, TypeError):
            return None
        if not js.is_file():
            return None
    solids = json.loads(js.read_text())["solids"]
    g = [s["genus"] for s in solids]
    if not solids or any(abs(x - round(x)) > 1e-6 for x in g):
        return None
    return dict(b0=len(solids), b1=int(round(sum(g))), b2=sum(s["shells"] - 1 for s in solids))


def describe(path, timeout: int = 900) -> Dict:
    """The facts of one part (see the module docstring); {"error": ...} when FreeCAD cannot read it."""
    from reco_engine.brep import is_step
    path = Path(path)
    if is_step(path):
        facts = _step_facts(path, timeout)
        if "error" in facts:
            return dict(path=str(path), source="step", error=facts["error"])
        import trimesh
        mesh = trimesh.load(facts["stl"], force="mesh")
        V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        lo, hi = (np.asarray(b, float) for b in facts["bbox"])
        diag = float(np.linalg.norm(hi - lo))
        betti, method = facts.get("betti") or {}, "tessellation"
        if betti.get("b1") is None:
            betti, method = (euler_poincare(path, Path(facts["cache"])) or {}), "euler-poincare"
        d = dict(version=CLASS_VERSION, path=str(path), source="step", valid=facts.get("valid"),
                 volume=facts.get("volume"), area=facts.get("area"), **facts.get("counts", {}),
                 b0=betti.get("b0"), b1=betti.get("b1"), b2=betti.get("b2"), shell_genus=betti.get("shells"),
                 betti_method=method if betti.get("b1") is not None else None)
        d.update(surface_facts(facts["faces"], diag, V))
    else:
        import trimesh
        mesh = trimesh.load(path, force="mesh")
        mesh.merge_vertices()
        V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        d = dict(version=CLASS_VERSION, path=str(path), source="mesh", family="?", **mesh_betti(V, F))
    d.update(mesh_descriptors(V, F))
    if d.get("b1") is not None and d["b1"] >= 0 and d.get("through_holes") is not None:
        d["b1_holes"] = min(int(d["b1"]), int(d["through_holes"]))
        d["b1_rest"] = int(d["b1"]) - d["b1_holes"]
    return d


def step_of_program(program: Dict, timeout: int = 300) -> Path:
    """A STEP of a known tree that has none (the v5 truths come as meshes only): built by FreeCAD
    (reco_field.export.build_program) into the cache, keyed by the program. Raises when FreeCAD cannot build it."""
    from reco_field.export import build_program
    key = hashlib.sha1(json.dumps(program, sort_keys=True).encode()).hexdigest()
    folder = (_cache_dir() or Path(tempfile.mkdtemp(prefix="shape_class_"))) / "programs"
    step = folder / f"{key}.step"
    if not step.is_file():
        ex = build_program(program, folder / key, "truth", timeout=timeout)
        Path(ex.step).replace(step)
    return step


# ---------------------------------------------------------------------------------------------------- known trees

def _subtype(p: Dict) -> str:
    """aegir's leaf subtypes (make_tree_study.py): box/full, cylinder full|sector|tube, cone full|sector, sphere
    full|sector|band|hollow, torus full|sector|band, halfspace/full. A leaf cut in u is a sector (also when v is cut,
    as the v5 sectors), one cut in v only a band."""
    t = p["type"].lower()

    def cut(key, span):
        r = p.get(key)
        return r is not None and abs((float(r[1]) - float(r[0])) - span) > 1e-6

    if t in ("cylinder", "sphere") and float(p.get("inner_radius") or 0.0) > 0:
        return f"{t}/{'tube' if t == 'cylinder' else 'hollow'}"
    if cut("u", FULL_TURN):
        return f"{t}/sector"
    if (t == "sphere" and cut("v", math.pi)) or (t == "torus" and cut("v", FULL_TURN)):
        return f"{t}/band"
    return f"{t}/full"


def tree_facts(program: Dict) -> Dict:
    """The complexity of a known CSG tree (reco_field.export.program_to_dict format): leaves, depth (edges from the root
    to the deepest leaf), operations by kind, leaf kinds and subtypes, partial leaves (sector, band, tube, hollow), and
    the distinct axis directions of the axial leaves (cylinder, cone, torus: the local z of their rotation)."""
    ops: Counter = Counter()
    subtypes: Counter = Counter()
    axes: List[np.ndarray] = []
    depth = 0

    def walk(n: Dict, level: int):
        nonlocal depth
        t = n["type"]
        if t == "Placed":
            p = n["primitive"]
            subtypes[_subtype(p)] += 1
            depth = max(depth, level)
            if p["type"].lower() in AXIAL:
                a = _canonical(np.asarray(n["rotation"], float)[:, 2])
                if not any(abs(a @ b) >= LINE_COS for b in axes):
                    axes.append(a)
            return
        if t in ("Union", "Intersection", "Subtraction"):
            ops[t] += 1
            walk(n["a"], level + 1)
            walk(n["b"], level + 1)
            return
        subtypes[t.lower() + "/full"] += 1                    # a bare primitive (no placement) as a leaf
        depth = max(depth, level)

    walk(program, 0)
    leaves = int(sum(subtypes.values()))
    kinds = Counter(s.split("/")[0] for s in subtypes.elements())
    return dict(leaves=leaves, depth=depth, ops=dict(ops), leaf_kinds=dict(kinds), subtypes=dict(subtypes),
                partial=int(sum(n for s, n in subtypes.items() if not s.endswith("/full"))), axis_dirs=len(axes))


# ---------------------------------------------------------------------------------------------------- the key

def estimate_leaves(d: Dict, calib: Optional[Dict]) -> Optional[float]:
    """The leaves of an unknown tree from the B-rep by the calibration of scripts/shape_atlas.py: log2 leaves = a + b
    log2 carriers (+ c log2 (1 + concave curved faces)). None without a calibration that passed."""
    if not calib or not calib.get("ok") or not d.get("carriers"):
        return None
    x = calib["a"] + calib["b"] * math.log2(d["carriers"]) + calib.get("c", 0.0) * math.log2(1 + (d.get("concave_curved") or 0))
    return float(2 ** x)


def topology_class(d: Dict) -> str:
    if d.get("b0") is None or d.get("b1") is None:
        return "g?"
    if d["b0"] > 1 or d["b1"] < 0:        # b1 < 0: more pieces in the tessellation than shells - lumps, several bodies
        return "multi"
    return _bin(int(d["b1"]), GENUS_BINS) + ("v" if (d.get("b2") or 0) > 0 else "")


def classify(d: Dict, tree: Optional[Dict] = None, calib: Optional[Dict] = None) -> Dict:
    """The class of a described part: dict(key, T, S, C, c_source, leaves (known or estimated), flags, version)."""
    T, S = topology_class(d), d.get("family") or "?"
    if tree is not None:
        leaves, source, C = tree["leaves"], "tree", _bin(tree["leaves"], LEAF_BINS)
    else:
        leaves = estimate_leaves(d, calib)
        if leaves is not None:
            source, C = "estimate", _bin(max(1, int(round(leaves))), LEAF_BINS)
        else:
            source, C = "brep", _bin(d.get("carriers"), CARRIER_BINS) or "K?"
    flags = []
    if (d.get("blends") or 0) > 0:
        flags.append("blend")
    if (d.get("b2") or 0) > 0:
        flags.append("void")
    if (d.get("b1_holes") or 0) > 0:
        flags.append("holes")
    if d.get("tau_rel") is not None and d["tau_rel"] < THIN_REL:
        flags.append("thin")
    if tree is not None and tree.get("partial"):
        flags.append("partial")
    return dict(version=CLASS_VERSION, key=f"{T}.{S}.{C}", T=T, S=S, C=C, c_source=source, leaves=leaves, flags=flags)
