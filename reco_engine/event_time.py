#!/usr/bin/env python3
"""
The event time of a stored run (the user, 2026-10-29: "benchmark their event-time with aegir and then select via
nephogram/histogram and benchmark like indicated in the presentation"): the run's program as GDML, scanned by aegir
(Geant4 through the SHiP framework; scripts/aegir_benchmark.py starts it), and the navigation time per event that
crossing the part costs - measured as the sneak-preview deck's table and docs/reports/real_trees did it:

    time per event = (wall(N_hi) - wall(N_lo)) / (N_hi - N_lo)          per round (start-up and geometry load cancel)
    navigation     = time per event (program) - time per event (empty baseline: the same world and gun, the part
                     replaced by a 0.1 mm box outside the beam)
    reported       = the median over the measured rounds and their range; round 0 is a warm-up

    r = event_time(run_dir)                         # writes <run_dir>/aegir/event_time.json, returns it
    r = read_event_time(run_dir)                    # the stored result or None
    p = placement(lo, hi)                           # gun, origin, beam axis for any part

aegir's guns are fixed cones (workflows/sandbox.libsonnet, GUNS). A part that is none of the parts aegir knows
(scripts/aegir_benchmark.py PLACEMENTS) is centred on the origin, its beam runs along its thinnest extent, and the gun
is the tightest cone that still covers its cross-section at its front face. All runs of one part get the same
placement (from the input's bounds), so their event times compare; the absolute number depends on the gun.
A benchmark is a heavy job: it waits for the shared lock (~/.cache/shape_reco/heavy.lock).
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "dataset_gen") not in sys.path:
    sys.path.insert(0, str(ROOT / "dataset_gen"))
EVENTS = (50_000, 300_000)          # the two event counts of the difference quotient (as docs/reports/real_trees)
ROUNDS = 3                          # measured rounds (one warm-up round comes before them); measured 2026-10-29 on an ABC
                                    # plate (9 primitives): navigation 0.63 us against a baseline of 13.1 us per event, one
                                    # round of 20k/100k events took 13 s - the noise needs the larger counts and the median
BASELINE_HALF = 0.05                # mm: the half size of the baseline's box, outside the beam
OUT_DIR = "aegir"
RESULT_JSON = "event_time.json"
# aegir's guns, workflows/sandbox.libsonnet (read 2026-10-29): name -> (vertex z in mm, cone half angle in rad)
GUNS = {"blind_hole": (-500.0, 0.02), "convention_test": (-500.0, 0.25), "stack": (-1000.0, 0.15),
        "study": (-1000.0, 0.35), "liner": (-1000.0, 0.75), "frame": (-1400.0, 1.2)}
AXES = "xyz"


def _scripts():
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import aegir_benchmark as ab
    return ab


def cone_radius(gun: str, z: float) -> float:
    """The radius of the gun's cone at world z (mm)."""
    vz, theta = GUNS[gun]
    return max(0.0, (z - vz) * math.tan(theta))


def placement(lo, hi) -> Dict:
    """The gun and the frame for a part of bounds lo..hi (source frame, mm): the beam along the thinnest extent, the
    part's centre in the origin, the tightest gun whose cone covers the cross-section at the front face and whose vertex
    lies before the part. dict(gun, origin, beam_axis, cross_radius, half_length, covered)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ext = hi - lo
    axis = int(np.argmin(ext))
    across = [ext[i] for i in range(3) if i != axis]
    r, half = 0.5 * float(math.hypot(*across)), 0.5 * float(ext[axis])
    fits = [(cone_radius(g, -half), g) for g, (vz, _) in GUNS.items() if vz < -half and cone_radius(g, -half) >= r]
    gun, covered = (min(fits)[1], True) if fits else (max(GUNS, key=lambda g: cone_radius(g, -half)), False)
    return dict(gun=gun, origin=((lo + hi) / 2).tolist(), beam_axis=AXES[axis], cross_radius=r, half_length=half,
                covered=covered)


def per_event(walls: Dict[tuple, float], events: Sequence[int] = EVENTS) -> Dict[int, float]:
    """{round: seconds per event} of one geometry from {(round, events): wall seconds}, the difference quotient."""
    lo, hi = events
    return {r: (walls[(r, hi)] - walls[(r, lo)]) / (hi - lo) for r in sorted({r for r, _ in walls})
            if (r, lo) in walls and (r, hi) in walls}


def navigation(program: Dict[tuple, float], baseline: Optional[Dict[tuple, float]], events: Sequence[int] = EVENTS,
               warmup: int = 0) -> Dict:
    """The navigation per event (us) of the measured rounds: program minus baseline per round, the median and range;
    without a baseline the program's own time per event."""
    p = {r: t for r, t in per_event(program, events).items() if r != warmup}
    b = {r: t for r, t in per_event(baseline, events).items() if r != warmup} if baseline else {}
    nav = [1e6 * (p[r] - b.get(r, 0.0)) for r in sorted(p) if not b or r in b]
    if not nav:
        return dict(rounds_counted=0)
    return dict(rounds_counted=len(nav), navigation_us=statistics.median(nav), min_us=min(nav), max_us=max(nav),
                per_event_us=statistics.median(1e6 * v for v in p.values()),
                baseline_us=statistics.median(1e6 * v for v in b.values()) if b else None, baseline=bool(b))


def read_event_time(run_dir) -> Optional[Dict]:
    p = Path(run_dir) / OUT_DIR / RESULT_JSON
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _input_bounds(run: Dict):
    """The input's bounds (the same placement for every run of the part); None when the input cannot be read."""
    path = (run.get("input") or {}).get("path")
    if not path or not Path(path).is_file():
        return None
    try:
        from src.closest_point_field import load_mesh
        v, _ = load_mesh(path)
        return v.min(axis=0), v.max(axis=0)
    except Exception:
        return None


def write_geometries(result, out: Path, stem: str, bounds=None) -> Dict:
    """The program and its empty baseline as GDML under <out>/geometry/sandbox/ (aegir finds them through the variable
    `root`); dict(place, program (export_gdml's info), baseline (info or None when the box cannot sit outside the beam))."""
    from reco_engine.result_to_program import program_model
    from reco_field.gdml_export import BEAM_AXES, export_gdml
    from src import sdf_primitives as sp
    model = program_model(result)
    if model is None:
        raise ValueError("the run has no accepted operation")
    lo, hi = bounds if bounds is not None else model.bounds()
    place = placement(lo, hi)
    common = dict(material="G4_Fe", world="G4_AIR", margin_mm=1000.0, origin=place["origin"], beam_axis=place["beam_axis"],
                  volume_name="PartVolume", world_name="World", physvol_name="PartPhys")
    geo = out / "geometry" / "sandbox"
    prog = export_gdml(model, geo / f"{stem}.gdml", name=stem, **common)
    W = np.asarray(prog["world_half"], float)
    corner = np.array([W[0] - 5.0, W[1] - 5.0, 0.0])                      # world frame, beside the beam
    base = None
    if math.hypot(corner[0], corner[1]) > cone_radius(place["gun"], 0.0) + 10.0:
        G = BEAM_AXES[place["beam_axis"]]
        p = np.asarray(place["origin"], float) + G.T @ corner                # x = G^T x_world + origin
        box = sp.Placed(sp.Box((BASELINE_HALF,) * 3), translation=p)
        base = export_gdml(box, geo / f"{stem}_baseline.gdml", name=f"{stem}_baseline", world_half=W, **common)
    return dict(place=place, program=prog, baseline=base)


def event_time(run_dir, events: Sequence[int] = EVENTS, rounds: int = ROUNDS, aegir=None, timeout: float = 1800.0,
               lock: bool = True, progress: Optional[Callable[[str], None]] = None) -> Dict:
    """Benchmark one stored run with aegir (see the module docstring); writes and returns <run_dir>/aegir/event_time.json."""
    from reco_engine import run_store
    ab = _scripts()
    say = progress or (lambda _t: None)
    run_dir = Path(run_dir)
    run = run_store.load_run(run_dir)
    result, name = ab.load_result(str(run_dir))
    aegir = Path(aegir or ab.AEGIR).resolve()
    if not (aegir / "pixi.toml").is_file():
        raise FileNotFoundError(f"no aegir checkout at {aegir} (set $AEGIR_ROOT)")
    out = run_dir / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    stem = "run_" + "".join(c if c.isalnum() else "_" for c in str(run.get("name") or name))[:60]
    geo = write_geometries(result, out, stem, _input_bounds(run))
    gun = geo["place"]["gun"]
    geometries = [stem] + ([f"{stem}_baseline"] if geo["baseline"] else [])
    lk = None
    if lock:
        import fcntl
        ab.LOCK.parent.mkdir(parents=True, exist_ok=True)
        lk = open(ab.LOCK, "w")
        say("aegir: waiting for the shared lock of heavy jobs ...")
        fcntl.flock(lk, fcntl.LOCK_EX)
    walls = {g: {} for g in geometries}
    runs = []
    t0 = time.time()
    try:
        for r in range(rounds + 1):                                        # interleaved: every geometry, both counts
            for g in geometries:
                for n in events:
                    say(f"aegir: {run.get('name')} round {r}{' (warm-up)' if r == 0 else ''}, {g.replace(stem, 'program')}, "
                        f"{n} events ...")
                    res = ab.run_scan(aegir, ab.scan_command(aegir, out, gun, g, n, 1), out / f"{g}_r{r}_{n}.log", timeout)
                    runs.append(dict(geometry=g, round=r, events=n, **res))
                    if res["exit_code"] == 0:
                        walls[g][(r, n)] = res["seconds"]
    finally:
        if lk is not None:
            lk.close()
    nav = navigation(walls[stem], walls.get(f"{stem}_baseline"), events)
    report = dict(run=str(run_dir), name=run.get("name"), aegir=str(aegir), placement=geo["place"], events=list(events),
                  rounds=rounds, geometry=dict(program=geo["program"], baseline=geo["baseline"]), runs=runs,
                  seconds=time.time() - t0, created=time.strftime("%Y-%m-%dT%H:%M:%S"), **nav)
    (out / RESULT_JSON).write_text(json.dumps(report, indent=1, default=str))
    return report
