#!/usr/bin/env python3
"""
Gate G3: the add-cut loop (reco_engine.residual_loop) end to end on the fixed part set, one part per process,
resumable. This is the run the thesis numbers come from; `scripts/g3_report.py` compares two runs and writes the
LaTeX macros.

    python3 scripts/g3_run.py --label s6full                 # the 60 parts, 2 jobs x 2 threads, nice 10, CPU only
    python3 scripts/g3_run.py --label smoke --parts dataset_gen/examples/blind_hole.stl --jobs 1
    python3 scripts/g3_run.py --label s6full --list          # which parts are still missing, nothing is run

Settings (the ad-hoc night runner of 2026-09-24, unchanged): type source "registry", max_resolution 128, CPU only
(CUDA_VISIBLE_DEVICES=""), 2 threads per job (OMP/MKL/OPENBLAS/SHAPE_RECO_KDTREE_WORKERS), nice 10, per-part timeout
1500 s for the frame and the beam suite and 600 s for the rest. Parts: the frame STL, dataset_gen/examples/beams/*.stl,
dataset_gen/examples/*.stl except the generator sample box_i000000.

Output: <out>/<label>/<parent>__<stem>.json per part (schema "g3.part/1"), <out>/<label>/run.json (settings, commit,
registry and checkpoint fingerprints, one entry per invocation) and <out>/<label>/freecad/<stem>/ (the built solids,
gitignored). A part whose record exists is skipped (--retry-failed reruns failed and timed-out ones).

Units (D-008): every length is reported in mm. The frame STL is at 1/100 scale; UNIT_TABLE scales it and asserts
its raw extent, so the day the STL is fixed at the source this runner stops instead of scaling it twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
SCHEMA = "g3.part/1"
FRAME_STL = Path(os.environ.get("SHAPE_RECO_FRAME_STL",
                                "~/shape_reco_parts/benchmark_parts/1.0/Unbenannt-Frame_4_x_6.stl"))
DEFAULTS = dict(type_source="registry", max_resolution=128, threads=2, nice=10, jobs=2,
                timeout_long=1500, timeout_short=600)

# mm per file unit for inputs that are NOT in mm, keyed by file name, with the raw longest side that identifies the
# file (relative tolerance 1e-3). The frame STL's coordinates are exactly 1/100 of the STEP's (docs/FRAME_FAILURE.md,
# D-008): raw 67 units = 6700 mm.
UNIT_TABLE: Dict[str, Tuple[float, float]] = {"Unbenannt-Frame_4_x_6.stl": (100.0, 67.0)}
PLAUSIBLE_MM = (1.0, 50000.0)        # longest side of any benchmark part in mm; outside: a unit error


class UnitError(ValueError):
    pass


def mm_per_unit(path, raw_max_extent: Optional[float]) -> float:
    """mm per file unit of `path`. STEP is mm by definition; an STL is mm unless UNIT_TABLE names it, and then its raw
    extent must match the table's (else the file changed, e.g. D-008 fixed at the source: update the table). The
    scaled longest side must lie in PLAUSIBLE_MM. `raw_max_extent` None: the table alone (old runs, no file)."""
    name = Path(str(path)).name
    scale = 1.0
    if name in UNIT_TABLE and Path(str(path)).suffix.lower() == ".stl":
        scale, expect = UNIT_TABLE[name]
        if raw_max_extent is not None and abs(raw_max_extent - expect) > 1e-3 * expect:
            raise UnitError(f"{name}: longest side {raw_max_extent:g} units, UNIT_TABLE expects {expect:g} "
                            f"(x{scale:g} = mm). The file changed (D-008 fixed at the source?): update UNIT_TABLE.")
    if raw_max_extent is not None and not PLAUSIBLE_MM[0] <= raw_max_extent * scale <= PLAUSIBLE_MM[1]:
        raise UnitError(f"{name}: longest side {raw_max_extent * scale:g} mm is outside {PLAUSIBLE_MM} - unit error?")
    return scale


def part_key(path) -> str:
    p = Path(str(path))
    return f"{p.parent.name}__{p.stem}"


def default_parts(repo: Path = REPO, frame: Path = FRAME_STL) -> List[Tuple[Path, int]]:
    """(part, timeout s), longest first so the tail of a parallel run is short."""
    ex = repo / "dataset_gen" / "examples"
    parts = [(frame, DEFAULTS["timeout_long"])]
    parts += [(p, DEFAULTS["timeout_long"]) for p in sorted((ex / "beams").glob("*.stl"))]
    parts += [(p, DEFAULTS["timeout_short"]) for p in sorted(ex.glob("*.stl")) if "box_i000000" not in p.name]
    return parts


def rel(path) -> str:
    p = Path(str(path))
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def sha256(path, chunk: int = 1 << 20) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(chunk), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def git_state(repo: Path = REPO) -> Dict:
    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    return dict(commit=git("rev-parse", "HEAD") or None, branch=git("rev-parse", "--abbrev-ref", "HEAD") or None,
                dirty=bool(git("status", "--porcelain", "--untracked-files=no")))


def model_fingerprint(repo: Path = REPO) -> Dict:
    """The registry file and every checkpoint it names (gitignored, so the commit alone does not pin them)."""
    reg = repo / "models" / "registry.json"
    out = dict(registry=rel(reg), registry_sha256=sha256(reg), checkpoints={})
    try:
        data = json.loads(reg.read_text())
    except Exception:
        return out
    for e in data["entries"] if isinstance(data, dict) else data:
        ck = e.get("checkpoint") or e.get("path")
        if not ck or ck in out["checkpoints"]:
            continue
        p = repo / ck
        files = sorted(q for q in (p.rglob("*") if p.is_dir() else [p])
                       if q.is_file() and q.suffix in (".pth", ".pt", ".joblib", ".json"))
        h = hashlib.sha256()
        for q in files:
            h.update(str(q.relative_to(p) if p.is_dir() else q.name).encode())
            h.update((sha256(q) or "missing").encode())
        out["checkpoints"][ck] = h.hexdigest() if files else None
    return out


def _cpu() -> float:
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime        # freecadcmd children are waited for: counted


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def summarize(result, grade: Dict, scale: float) -> Dict:
    """The per-part numbers from a ReconstructionResult (`result.to_dict()`-able) and the grade dict of the build
    (reco_engine.residual_loop._grade: valid, n_solids, chamfer, chamfer_max, exact_error, sampled_iou, ...).
    Lengths x `scale` (mm per unit), volumes x scale^3."""
    ops = [o for o in result.operations if getattr(o, "accepted", True)]
    genus, in_genus = _num(result.genus), _num(result.input_genus)
    ch, chmax = _num(grade.get("chamfer", result.chamfer_mm)), _num(grade.get("chamfer_max"))
    rp, rm = _num(result.exact_r_plus), _num(result.exact_r_minus)
    return dict(
        n_ops=len(ops), ops=[f"{o.op}:{o.primitive.kind}" for o in ops],
        freecad_valid=bool(grade.get("valid")) if grade else None, n_solids=grade.get("n_solids"),
        build_error=grade.get("error"),
        chamfer_units=ch, chamfer_mm=None if ch is None else ch * scale,
        chamfer_max_mm=None if chmax is None else chmax * scale,
        exact_iou=_num(result.exact_iou), exact_method=result.exact_method, exact_error=grade.get("exact_error"),
        sampled_iou=_num(grade.get("sampled_iou")),
        r_plus_mm3=None if rp is None else rp * scale ** 3, r_minus_mm3=None if rm is None else rm * scale ** 3,
        r_plus_solids=result.exact_r_plus_solids,
        genus=genus, input_genus=in_genus,
        genus_match=None if genus is None or in_genus is None else round(genus) == round(in_genus),
        grid_pitch_over_tau=_num(result.grid_pitch_over_tau),
        residual_voxels_pos=result.residual_voxels_pos, residual_voxels_neg=result.residual_voxels_neg)


def raw_max_extent(part: Path) -> Optional[float]:
    if part.suffix.lower() != ".stl":
        return None
    import trimesh
    m = trimesh.load(str(part), force="mesh")
    return float(max(m.extents))


def default_reconstruct(part: Path, settings: Dict, out_dir: Path):
    """The loop with the registry's type stage limited to settings["type_source"]; returns (result, grade dict).
    The grade dict is the one residual_loop._grade fills (the builder's return value is updated in place with the
    exact judge's output), so exact_error / sampled_iou / n_solids are kept, which the result object drops."""
    for p in (REPO / "dataset_gen", REPO):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from reco_engine import cascade
    from reco_engine.residual_loop import ResidualConfig, freecad_grade, reconstruct
    grade: Dict = {}

    def builder(model, od, name, field):
        grade.update(freecad_grade(model, od, name, field))
        return grade

    reg = cascade.filter_type_entries(cascade.load_default_registry(), settings["type_source"])
    cfg = ResidualConfig(registry=reg, max_resolution=settings["max_resolution"], freecad=True, out_dir=out_dir,
                         builder=builder)
    return reconstruct(part, cfg), grade


def run_one(part: Path, label: str, out_root: Path, settings: Dict,
            reconstruct_fn: Optional[Callable] = None, git: Optional[Dict] = None) -> Dict:
    """Reconstruct one part and write <out_root>/<label>/<key>.json. Never raises (a failure is a record)."""
    part = Path(part)
    dest = out_root / label / f"{part_key(part)}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    here = git or git_state()
    # the commit the run was launched at (the parent passes it); the tree can move while a 60-part run is going
    launched = dict(commit=os.environ.get("G3_COMMIT") or here.get("commit"),
                    dirty=os.environ.get("G3_DIRTY", str(here.get("dirty"))) == "True")
    git = dict(commit=launched["commit"], dirty=launched["dirty"] or bool(here.get("dirty")))
    row = dict(schema=SCHEMA, key=part_key(part), part=rel(part), part_sha256=sha256(part), label=label,
               settings=dict(settings), commit=git.get("commit"), dirty=git.get("dirty"),
               commit_at_part=here.get("commit"), commit_drift=here.get("commit") != launched["commit"],
               host=platform.node(), started=datetime.now().isoformat(timespec="seconds"), timeout=False)
    t0, c0 = time.time(), _cpu()
    try:
        ext = raw_max_extent(part)
        scale = mm_per_unit(part, ext)
        row.update(mm_per_unit=scale, extent_mm=None if ext is None else ext * scale)
        res, grade = (reconstruct_fn or default_reconstruct)(part, settings, out_root / label / "freecad" / part_key(part))
        row.update(ok=True, **summarize(res, grade, scale))
        d = res.to_dict()
        for k in ("step_path", "stl_path"):
            if d.get(k):
                d[k] = rel(d[k])
        row["result"] = d
    except Exception as exc:
        row.update(ok=False, error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-2000:])
    row.update(wall_seconds=round(time.time() - t0, 1), cpu_seconds=round(_cpu() - c0, 1),
               cpu_seconds_process=round(_cpu(), 1))
    dest.write_text(json.dumps(row, default=str, indent=1))
    return row


def line(row: Dict) -> str:
    if row.get("timeout"):
        return f"{row['label']} {row['key']}: TIMEOUT after {row.get('wall_seconds')} s"
    if not row.get("ok"):
        return f"{row['label']} {row['key']}: FAIL {str(row.get('error'))[:120]} ({row.get('wall_seconds')} s)"
    iou = row.get("exact_iou")
    return (f"{row['label']} {row['key']}: ok {row['wall_seconds']} s wall, {row['cpu_seconds']} s CPU, ops {row['n_ops']}, "
            f"chamfer {row['chamfer_mm']:.4g} mm, IoU {'-' if iou is None else f'{iou:.4f}'} ({row.get('exact_method')}), "
            f"genus {row.get('genus')} ({row.get('input_genus')}), valid {row.get('freecad_valid')}"
            if row.get("chamfer_mm") is not None else
            f"{row['label']} {row['key']}: ok {row['wall_seconds']} s, ops {row['n_ops']}, no build ({row.get('build_error')})")


def pending(parts: List[Tuple[Path, int]], run_dir: Path, retry_failed: bool = False) -> List[Tuple[Path, int]]:
    out = []
    for p, t in parts:
        f = run_dir / f"{part_key(p)}.json"
        if f.is_file():
            if not retry_failed:
                continue
            try:
                if json.loads(f.read_text()).get("ok"):
                    continue
            except Exception:
                pass
        out.append((p, t))
    return out


def child_env(threads: int, git: Optional[Dict] = None) -> Dict[str, str]:
    env = dict(os.environ)
    if git:
        env.update(G3_COMMIT=str(git.get("commit") or ""), G3_DIRTY=str(bool(git.get("dirty"))))
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), SHAPE_RECO_KDTREE_WORKERS=str(threads), PYTHONWARNINGS="ignore")
    return env


def launch(part: Path, timeout: int, label: str, out_root: Path, settings: Dict, git: Optional[Dict] = None) -> Dict:
    """One part in its own process group under nice; on timeout the whole group (python + freecadcmd) is killed
    and a timeout record written."""
    cmd = ["nice", "-n", str(settings["nice"]), sys.executable, str(Path(__file__).resolve()), "--one", str(part),
           "--label", label, "--out", str(out_root), "--type-source", settings["type_source"],
           "--max-resolution", str(settings["max_resolution"]), "--threads", str(settings["threads"]),
           "--nice", str(settings["nice"])]
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=child_env(settings["threads"], git), cwd=str(REPO), start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
        g = git or git_state()
        row = dict(schema=SCHEMA, key=part_key(part), part=rel(part), part_sha256=sha256(part), label=label,
                   settings=dict(settings), commit=g["commit"], dirty=g["dirty"], host=platform.node(), ok=False,
                   timeout=True, error=f"timeout after {timeout} s", wall_seconds=round(time.time() - t0, 1),
                   cpu_seconds=None)
        (out_root / label / f"{part_key(part)}.json").write_text(json.dumps(row, indent=1))
        return row
    dest = out_root / label / f"{part_key(part)}.json"
    if dest.is_file():
        return json.loads(dest.read_text())
    return dict(key=part_key(part), label=label, ok=False, error=f"crash (exit {proc.returncode}): {out[-300:]}",
                wall_seconds=round(time.time() - t0, 1))


def update_manifest(run_dir: Path, settings: Dict, parts: List[Tuple[Path, int]], git: Dict) -> Dict:
    f = run_dir / "run.json"
    m = json.loads(f.read_text()) if f.is_file() else dict(schema="g3.run/1", label=run_dir.name, invocations=[])
    m.setdefault("invocations", []).append(dict(
        started=datetime.now().isoformat(timespec="seconds"), host=platform.node(), python=sys.version.split()[0],
        settings=settings, **git, models=model_fingerprint(),
        parts=[dict(part=rel(p), key=part_key(p), timeout=t, sha256=sha256(p)) for p, t in parts]))
    commits = {i.get("commit") for i in m["invocations"]}
    m["mixed_commits"] = len(commits) > 1
    f.write_text(json.dumps(m, indent=1))
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "data" / "g3")
    ap.add_argument("--parts", nargs="*", type=Path, default=None, help="default: the 60-part G3 set")
    ap.add_argument("--timeout", type=int, default=None, help="per part, overrides the 1500/600 s defaults")
    ap.add_argument("--jobs", type=int, default=DEFAULTS["jobs"])
    ap.add_argument("--threads", type=int, default=DEFAULTS["threads"])
    ap.add_argument("--nice", type=int, default=DEFAULTS["nice"])
    ap.add_argument("--type-source", default=DEFAULTS["type_source"], choices=("registry", "cnn", "descriptors"))
    ap.add_argument("--max-resolution", type=int, default=DEFAULTS["max_resolution"])
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the pending parts and exit")
    ap.add_argument("--one", type=Path, default=None, help=argparse.SUPPRESS)      # the per-part child
    a = ap.parse_args(argv)
    settings = dict(type_source=a.type_source, max_resolution=a.max_resolution, threads=a.threads, nice=a.nice)
    out_root = a.out.resolve()
    if a.one is not None:
        print(line(run_one(a.one, a.label, out_root, settings)), flush=True)
        return 0
    if a.parts:
        parts = [(p.resolve(), a.timeout or (DEFAULTS["timeout_long"] if "beams" in p.parts or "Frame" in p.name
                                             else DEFAULTS["timeout_short"])) for p in a.parts]
    else:
        parts = [(p, a.timeout or t) for p, t in default_parts()]
    missing = [p for p, _ in parts if not p.is_file()]
    if missing:
        print(f"missing parts: {[str(p) for p in missing]}", file=sys.stderr)
        return 2
    run_dir = out_root / a.label
    run_dir.mkdir(parents=True, exist_ok=True)
    todo = pending(parts, run_dir, a.retry_failed)
    if a.list:
        print("\n".join(f"{t}\t{p}" for p, t in todo) or "nothing pending")
        return 0
    git = git_state()
    if git["dirty"]:
        print("warning: the working tree has uncommitted changes to tracked files; records carry dirty=true",
              file=sys.stderr)
    m = update_manifest(run_dir, settings, parts, git)
    if m["mixed_commits"]:
        print("warning: this run dir holds records from more than one commit (run.json: mixed_commits)", file=sys.stderr)
    print(f"=== G3 {a.label} {datetime.now():%T}: {len(todo)} of {len(parts)} parts pending, {a.jobs} jobs x "
          f"{a.threads} threads, nice {a.nice}, {a.type_source}, max_res {a.max_resolution}, commit "
          f"{(git['commit'] or '?')[:8]}{' (dirty)' if git['dirty'] else ''}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        futs = [pool.submit(launch, p, t, a.label, out_root, settings, git) for p, t in todo]
        for f in as_completed(futs):
            print(line(f.result()), flush=True)
    done = len(list(run_dir.glob("*__*.json")))
    print(f"=== G3 {a.label} done {datetime.now():%T}: {done} records in {rel(run_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
