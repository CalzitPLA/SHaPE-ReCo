#!/usr/bin/env python3
"""
The store of run results: every reconstruction run that is worth comparing later (the frame, the liner, ...) is saved as
one directory, and two runs can be compared.

    <store>/<part stem>/<YYYYmmdd-HHMMSS>_<short commit>[_<label>]/
        run.json      what ran (input by content, code commit, every setting), the program, the metrics, the timings
        steps.jsonl   one line per step: index, stage, phase, message, seconds
        model.step    copies of the built model, when the run built one
        model.stl
        view.png      a screenshot of the 3D view (from the GUI)
        program_recovered.json   only beside a run whose program was cut when it was saved (before 2026-10-01):
                      the program rebuilt from the feature phase's own record (scripts/recover_run_program.py)

The store is `$SHAPE_RECO_RESULTS` if set, else `~/shape_reco_results/`: outside the repository and its worktrees, so that
every worktree and branch writes to the same store and nothing is lost when a worktree is removed.

`save_run` never overwrites a run and writes atomically (into `<dir>.tmp`, renamed at the end). A metric that was not
computed is null in run.json, never 0. `list_runs`, `load_run` and `diff_runs` read only run.json. No Qt, no torch;
scripts/compare_runs.py is the command line.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = 1
RUN_JSON, STEPS_JSONL = "run.json", "steps.jsonl"
RECOVERED_PROGRAM = "program_recovered.json"   # beside a run whose program was cut when it was saved (`truncated`)
MAX_ARRAY = 64                      # elements: a longer array is recorded by its shape, not its values
TOLERANCE = 1e-6                    # diff_runs: numbers closer than this (absolute, or relative to the larger) are equal

# ReconstructionResult fields that are metrics, under the names the loop uses
RESULT_METRICS = ("residual_voxels_pos", "residual_voxels_neg", "recut_voxels", "chamfer_mm", "exact_iou", "exact_method",
                  "exact_r_plus", "exact_r_minus", "exact_r_plus_solids", "genus", "input_genus", "grid_pitch_over_tau")


def store_root() -> Path:
    env = os.environ.get("SHAPE_RECO_RESULTS")
    return Path(env).expanduser() if env else Path.home() / "shape_reco_results"


# ------------------------------------------------------------------------------------------------ plain JSON

MAX_NODES = 10_000_000              # plain(limit=None): more nodes than this is a cycle (or no program)


def plain(obj, depth: int = 0, limit: Optional[int] = 12):
    """`obj` as JSON data: dataclasses and dicts by field, paths as text, numpy scalars as numbers, short arrays as
    lists (long ones by their shape), non-finite numbers as null, anything else by its name or text. Below `limit`
    levels an object is written as its text (settings and step infos).

    `limit=None` (a PROGRAM): no level is cut and every array is written in full, at any depth, without recursion
    (`_plain_tree`). A composite tool is a chain of booleans, one level per operand, two JSON levels per level of the
    tree: the limit of 12 (until 2026-09-29) cut the operands of a rounded pocket or a hole stack below its fifth level
    down to text, the limit of 400 (until 2026-10-01) every operand below its 198th, and dataclasses.asdict, which
    to_dict used, fails with a RecursionError at about 500 levels; a run cut so cannot be built again (`truncated`)."""
    if limit is None:
        return _plain_tree(obj)
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if depth > limit:
        return str(obj)[:200]
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: plain(getattr(obj, f.name, None), depth + 1, limit) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): plain(v, depth + 1, limit) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [plain(v, depth + 1, limit) for v in obj]
    if hasattr(obj, "shape") and hasattr(obj, "tolist"):                 # numpy, without importing it
        if getattr(obj, "ndim", 1) == 0:
            return plain(obj.tolist(), depth + 1, limit)
        if getattr(obj, "size", 0) > MAX_ARRAY:
            return {"array": [int(n) for n in obj.shape], "dtype": str(getattr(obj, "dtype", ""))}
        return plain(obj.tolist(), depth + 1, limit)
    if hasattr(obj, "item") and callable(obj.item):                      # a numpy scalar
        try:
            return plain(obj.item(), depth + 1, limit)
        except Exception:
            pass
    if callable(obj):
        return f"<{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', type(obj).__name__)}>"
    return str(obj)[:200]


def _plain_tree(obj):
    """plain(obj, limit=None): the same conversion with an explicit stack instead of recursion, nothing cut, arrays in
    full, the order of fields and keys kept."""
    out = [None]
    stack = [(obj, out, 0)]                        # (object, the container its value goes into, its key there)
    nodes = 0
    while stack:
        o, holder, key = stack.pop()
        nodes += 1
        if nodes > MAX_NODES:
            raise ValueError(f"plain: more than {MAX_NODES} nodes - a cycle?")
        if o is None or isinstance(o, (bool, int, str)):
            holder[key] = o
            continue
        if isinstance(o, float):
            holder[key] = o if math.isfinite(o) else None
            continue
        if isinstance(o, Path):
            holder[key] = str(o)
            continue
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            items = [(f.name, getattr(o, f.name, None)) for f in dataclasses.fields(o)]
        elif isinstance(o, dict):
            items = [(str(k), v) for k, v in o.items()]
        elif isinstance(o, (list, tuple, set, frozenset)):
            items = list(enumerate(o))
        elif hasattr(o, "shape") and hasattr(o, "tolist"):                 # numpy (any size), without importing it
            stack.append((o.tolist(), holder, key))
            continue
        elif hasattr(o, "item") and callable(o.item):                      # a numpy scalar
            try:
                stack.append((o.item(), holder, key))
                continue
            except Exception:
                items = None
        else:
            items = None
        if items is None:
            holder[key] = (f"<{getattr(o, '__module__', '')}.{getattr(o, '__qualname__', type(o).__name__)}>"
                           if callable(o) else str(o)[:200])
            continue
        if isinstance(o, (list, tuple, set, frozenset)):
            value = [None] * len(items)
        else:
            value = {k: None for k, _ in items}                 # the keys in their order; the values follow
        holder[key] = value
        stack.extend((v, value, k) for k, v in reversed(items))
    return out[0]


# a stored program: where a list or a dict is expected (a string there is an object that was cut to its text)
_PRIMITIVE_LISTS = ("parts", "alternatives", "sources")
_PRIMITIVE_FIELDS = ("quaternion", "center_mm", "dims_mm", "u", "v") + _PRIMITIVE_LISTS


def truncated(program) -> List[str]:
    """The nodes of a stored program that were cut down to their text when it was saved (run store before 2026-10-01,
    `plain`): their paths, e.g. "[2].primitive.parts[0].parts[1]", in the program's order. Empty for a whole program."""
    out: List[str] = []
    stack = [(o, f"[{i}]", "operation") for i, o in reversed(list(enumerate(program or [])))]
    while stack:
        node, path, what = stack.pop()
        if isinstance(node, str):
            out.append(path)
            continue
        if not isinstance(node, dict):
            continue
        if what == "operation":
            if "primitive" in node:
                stack.append((node["primitive"], f"{path}.primitive", "primitive"))
            continue
        if what == "source":
            if isinstance(node.get("detail"), str):
                out.append(f"{path}.detail")
            continue
        later = []
        for key in _PRIMITIVE_FIELDS:
            value = node.get(key)
            if isinstance(value, str):
                out.append(f"{path}.{key}")
            elif key in _PRIMITIVE_LISTS and isinstance(value, list):
                kind = "source" if key == "sources" else "primitive"
                later += [(x, f"{path}.{key}[{j}]", kind) for j, x in enumerate(value)]
        stack.extend(reversed(later))
    return out


def file_identity(path) -> Dict[str, object]:
    """The input by content: path, size in bytes, sha256 (null when the file cannot be read)."""
    p = Path(path)
    out = dict(path=str(p), name=p.name, size=None, sha256=None)
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        out.update(size=p.stat().st_size, sha256=h.hexdigest())
    except OSError:
        pass
    return out


def git_state(repo=None) -> Dict[str, object]:
    """The code that ran: commit, branch, and whether the working tree had uncommitted changes (with the files)."""
    repo = Path(repo) if repo else ROOT

    def git(*args):
        try:
            r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=20)
            return r.stdout if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    commit = (git("rev-parse", "HEAD") or "").strip() or None
    branch = (git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None
    status = git("status", "--porcelain", "--untracked-files=no")
    modified = [line[3:].strip() for line in (status or "").splitlines() if line.strip()]
    return dict(commit=commit, short=commit[:8] if commit else None, branch=branch,
                dirty=bool(modified) if status is not None else None, modified=modified, repository=str(repo))


def peak_rss_mb() -> Optional[float]:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0          # Linux: kB
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------ collecting

def _states(run) -> list:
    return list(getattr(run, "states", None) or ())


def _result(run):
    """The ReconstructionResult (or the other methods' result object) of a session or of the result itself."""
    if hasattr(run, "operations") or not hasattr(run, "states"):
        return run
    for state in reversed(_states(run)):
        if getattr(state, "result", None) is not None:
            return state.result
    return None


def _method(run, settings) -> Optional[str]:
    method = getattr(settings, "method", None) if settings is not None else None
    if isinstance(settings, dict):
        method = settings.get("method")
    if method:
        return str(method)
    states = _states(run)
    if states:
        return "field" if hasattr(states[-1], "vectors") else "cascade" if hasattr(states[-1], "program") else "voxel"
    return "cascade" if hasattr(run, "operations") else None


def _source(primitive: dict) -> str:
    """Where a primitive came from: the mode of its last "type" source (the feature phase appends its own), else of its
    first source."""
    sources = primitive.get("sources") or []
    typed = [s for s in sources if s.get("stage") == "type"]
    chosen = (typed[-1] if typed else sources[0]) if sources else None
    return str(chosen.get("mode")) if chosen else "unknown"


def _count(items) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return dict(sorted(out.items()))


def _info_of(states, stage: str) -> Optional[dict]:
    for state in reversed(states):
        if getattr(getattr(state, "step", None), "stage", None) == stage:
            return dict(getattr(state, "info", None) or {})
    return None


def collect_program(run, method: Optional[str]) -> List[dict]:
    """The operations in order, by the existing serialisation: ReconstructionResult.to_dict() (the cascade; accepted
    operations only), the vector field's OpInfo records, the voxel pipeline's result. Written in full at any depth
    (plain(limit=None)); the operations of a dataclass result are read field by field, not through to_dict
    (dataclasses.asdict recurses two Python frames per level of a composite)."""
    result = _result(run)
    if result is not None and hasattr(result, "operations") and hasattr(result, "to_dict"):
        if dataclasses.is_dataclass(result) and not isinstance(result, type):
            ops = plain(list(result.operations), limit=None)
        else:
            ops = plain(result.to_dict(), limit=None).get("operations", [])
        return [o for o in ops if o.get("accepted", True)]
    states = _states(run)
    if states and hasattr(states[-1], "ops"):
        return [plain(o, limit=None) for o in states[-1].ops]
    if result is not None and hasattr(result, "added"):
        return ([dict(op="add", primitive=plain(p, limit=None)) for p in result.added]
                + [dict(op="cut", primitive=plain(p, limit=None)) for p in getattr(result, "cut", ())])
    return []


def collect_metrics(run, method: Optional[str], program: List[dict]) -> Dict[str, object]:
    """The metrics that exist for the run, under the names the loop reports them; what was not computed is None."""
    result, states = _result(run), _states(run)
    done = _info_of(states, "done") or {}
    grade = done.get("grade") if isinstance(done.get("grade"), dict) else (_info_of(states, "grade") or {}).get("grade")
    grade = grade if isinstance(grade, dict) else {}
    stats = done.get("stats") if isinstance(done.get("stats"), dict) else None
    m: Dict[str, object] = {name: plain(getattr(result, name, None)) for name in RESULT_METRICS}
    valid = grade.get("valid")
    if valid is None and program:
        valid = program[-1].get("freecad_valid")
    m.update(
        voxel_iou=plain(stats.get("iou")) if stats else None,
        solution_stats=plain(stats),
        chamfer_mean=plain(grade.get("chamfer", grade.get("symmetric_mean", getattr(result, "chamfer_mm", None)))),
        chamfer_max=plain(grade.get("chamfer_max", grade.get("max"))),
        freecad_valid=plain(valid), n_solids=plain(grade.get("n_solids")), grade=plain(grade) or None,
        missing_volume=plain(getattr(result, "exact_r_plus", None)),
        invented_volume=plain(getattr(result, "exact_r_minus", None)),
        input_voxels=plain(done.get("input_voxels")), fallback_boxes=plain(done.get("fallback_boxes")),
        stopped=plain(done.get("stopped")),
        n_operations=len(program),
        operations_by_kind=_count(f"{o.get('op')}:{(o.get('primitive') or o).get('kind')}" for o in program),
        operations_by_source=_count(_source(o.get("primitive") or {}) for o in program) if method == "cascade" else None,
        feature_phase=plain(_info_of(states, "feature_done")))
    feature = m["feature_phase"]
    if feature:                                          # the exact volumes of the phase are newer than the grade's
        for name in ("missing_volume", "invented_volume"):
            if feature.get(name) is not None:
                m[name] = feature[name]
        if m.get("exact_iou") is None:
            m["exact_iou"] = feature.get("exact_iou")
    if states and hasattr(states[-1], "explained"):      # the vector field's own numbers
        last = states[-1]
        m.update(explained=plain(last.explained), invented=plain(getattr(last, "invented", None)),
                 stopped=plain(getattr(last, "stopped", None)) or m["stopped"])
        g = getattr(last, "grade", None)
        if isinstance(g, dict):
            m.update(chamfer_mean=plain(g.get("symmetric_mean")), grade=plain(g),
                     chamfer_max=plain(max((g.get(k) or {}).get("max", 0.0) for k in ("input_to_model", "model_to_input"))
                                       if g.get("input_to_model") and g.get("model_to_input") else None))
    return m


def collect_steps(run) -> List[dict]:
    out = []
    for state in _states(run):
        step = getattr(state, "step", None)
        seconds = [getattr(x, "wall", None) for x in getattr(state, "timings", ()) if getattr(x, "depth", 1) == 0]
        info = getattr(state, "info", None) or {}
        out.append(dict(index=int(getattr(state, "index", len(out))), stage=getattr(step, "stage", None),
                        phase=getattr(step, "phase", None), message=getattr(step, "message", None),
                        seconds=plain(info["seconds"]) if isinstance(info, dict) and info.get("seconds") is not None
                        else (plain(sum(s for s in seconds if s is not None)) if seconds else None)))
    return out


def collect_timings(run) -> Dict[str, object]:
    rec = getattr(run, "recorder", None)
    out = dict(wall=None, cpu=None, sections=None, peak_rss_mb=plain(peak_rss_mb()))
    if rec is not None:
        try:
            elapsed = rec.elapsed()
            out.update(wall=plain(elapsed.get("wall")), cpu=plain(elapsed.get("cpu")),
                       sections=[plain(t.as_dict()) for t in rec.summary()])
        except Exception as exc:
            out["error"] = str(exc)[-200:]
    return out


def collect(run, settings, input_path, label: Optional[str] = None, extra: Optional[dict] = None, repo=None) -> dict:
    """run.json as a dict (without the files of the model)."""
    method = _method(run, settings)
    program = collect_program(run, method)
    states = _states(run)
    result = _result(run)
    finished = bool(getattr(run, "finished", False)) if states or hasattr(run, "finished") else True
    ident = file_identity(input_path)
    return dict(
        schema=SCHEMA, part=Path(str(input_path)).stem, label=label or None,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"), time=time.time(), host=socket.gethostname(),
        input=ident, code=git_state(repo), method=method, finished=finished, steps=len(states) or None,
        settings=dict(gui=plain(settings) if settings is not None else None, config=plain(getattr(run, "config", None))),
        program=program, metrics=collect_metrics(run, method, program), timings=collect_timings(run),
        model=dict(step=plain(getattr(result, "step_path", None)), stl=plain(getattr(result, "stl_path", None)), note=None),
        extra=plain(extra) if extra else None)


# ------------------------------------------------------------------------------------------------ saving

def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", str(text)).strip("-")[:60]


def part_key(text) -> str:
    """What makes two spellings one part: letters and digits only, lower case ('Liner-LINER-001' and
    'Liner_LINER_001' are the same part)."""
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def part_directories(part, store=None) -> List[Path]:
    """The store's directories of a part: the same key (part_key); when there is none, the ones whose key starts with
    the given one ('liner'), and then the ones that contain it."""
    root = Path(store) if store else store_root()
    if not root.is_dir():
        return []
    key = part_key(part)
    found = [(part_key(d.name), d) for d in sorted(root.iterdir()) if d.is_dir()]
    for test in (lambda k: k == key, lambda k: bool(key) and k.startswith(key), lambda k: bool(key) and key in k):
        hits = [d for k, d in found if test(k)]
        if hits:
            return hits
    return []


def part_directory(part, store=None) -> Path:
    """Where the runs of a part are saved: the directory that holds the part already under another spelling (the one
    with the most runs), else a new one."""
    root = Path(store) if store else store_root()
    key = part_key(part)
    same = [d for d in (sorted(root.iterdir()) if root.is_dir() else []) if d.is_dir() and key and part_key(d.name) == key]
    if same:
        return max(same, key=lambda d: sum(1 for r in d.iterdir() if (r / RUN_JSON).is_file()))
    return root / (_slug(part) or "part")


def relabel(path, label: str) -> Path:
    """Give a saved run a label (or another one) instead of saving it a second time: the directory is renamed and
    run.json follows. Returns the new directory."""
    run_dir = Path(path)
    run = json.loads((run_dir / RUN_JSON).read_text())
    slug = _slug(label)
    if not slug:
        return run_dir
    stem = run_dir.name
    old = _slug(run.get("label") or "")
    if old and stem.endswith("_" + old):
        stem = stem[:-len(old) - 1]
    final, n = run_dir.with_name(f"{stem}_{slug}"), 1
    while final.exists() and final != run_dir:
        n += 1
        final = run_dir.with_name(f"{stem}_{slug}-{n}")
    run["label"], run["name"] = label, final.name
    (run_dir / RUN_JSON).write_text(json.dumps(run, indent=1, allow_nan=False))
    if final != run_dir:
        os.rename(run_dir, final)
    return final


def save_run(session_or_result, settings, input_path, label: Optional[str] = None, extra: Optional[dict] = None,
             repo=None, store=None, screenshot=None) -> Path:
    """Write the run into a new directory of the store and return it. `session_or_result`: a StepSession (its states,
    recorder and config are read) or a ReconstructionResult. `screenshot(path) -> bool` writes view.png (the GUI's 3D
    view, called in the caller's thread); a view that was not written is simply left out."""
    run = collect(session_or_result, settings, input_path, label=label, extra=extra, repo=repo)
    base = part_directory(run["part"], store=store)
    base.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}_{run['code']['short'] or 'nocommit'}" + (f"_{_slug(label)}" if label and _slug(label) else "")
    final, n = base / stem, 1
    while final.exists() or final.with_name(final.name + ".tmp").exists():     # never overwrite: a second run this second
        n += 1
        final = base / f"{stem}-{n}"
    tmp = final.with_name(final.name + ".tmp")
    tmp.mkdir()
    try:
        notes = []
        for key, name in (("step", "model.step"), ("stl", "model.stl")):
            source = run["model"][key]
            if source and Path(source).is_file():
                shutil.copyfile(source, tmp / name)
                run["model"][key] = name
                run["model"][f"{key}_source"] = str(source)
            else:
                notes.append(f"no {name}: " + (f"{source} does not exist" if source else "the run built none"))
                run["model"][key] = None
        run["model"]["note"] = "; ".join(notes) or None
        document = ((run.get("metrics") or {}).get("feature_phase") or {}).get("fcstd")
        run["model"]["fcstd"] = None                # the feature phase's FreeCAD document (input, model, pieces, tools)
        if document and Path(document).is_file():
            shutil.copyfile(document, tmp / "features.FCStd")
            run["model"]["fcstd"], run["model"]["fcstd_source"] = "features.FCStd", str(document)
        run["view"] = None
        if screenshot is not None:
            try:
                if screenshot(str(tmp / "view.png")) and (tmp / "view.png").is_file() and (tmp / "view.png").stat().st_size:
                    run["view"] = "view.png"
                else:
                    (tmp / "view.png").unlink(missing_ok=True)
            except Exception:
                (tmp / "view.png").unlink(missing_ok=True)
        run["name"] = final.name
        with open(tmp / STEPS_JSONL, "w") as f:
            for row in collect_steps(session_or_result):
                f.write(json.dumps(row) + "\n")
        (tmp / RUN_JSON).write_text(json.dumps(run, indent=1, allow_nan=False))
        os.rename(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def import_run_part(run_dir, input_path, label: Optional[str] = None, extra: Optional[dict] = None, store=None) -> Path:
    """A run of scripts/run_part.py (record.json, program.json, steps.jsonl, model.stl/.step, rating.json - e.g. a
    server run fetched by scripts/remote_run.py) as a run of the store, so that the window's Benchmark tab, Open,
    Compare, the nephogram and the event time treat it as any run (the user, 2026-10-29: "how to use the run in the gui:
    i want to benchmark"). `input_path` is the LOCAL input: the store finds a part's runs by its content (sha256).
    The run's own settings become the path fields of the GUI (start shape, mode, sub-parts, type source, registry),
    so its path variant shows. Never overwrites; returns the new directory."""
    run_dir = Path(run_dir)
    rec = json.loads((run_dir / "record.json").read_text())
    prog = json.loads((run_dir / "program.json").read_text()) if (run_dir / "program.json").is_file() else {}
    rs = rec.get("settings") or {}
    comp = rec.get("comparison") or {}
    exact, sampled = comp.get("exact") or {}, comp.get("sampled") or {}
    model = rec.get("model") or {}
    program = [o for o in (prog.get("operations") or []) if o.get("accepted", True)]
    registry = rs.get("registry")
    if registry:                                            # a server path of the checkout: the same file here
        local = ROOT / "models" / Path(str(registry)).name
        registry = str(local) if local.is_file() else registry
    gui = dict(method="cascade", input_stl=str(input_path), cascade_start_shape=rs.get("start") or "legacy",
               cascade_mode=rs.get("mode") or "loop", cascade_subparts=bool(rs.get("subparts")),
               cascade_type_source=rs.get("type_source") or "registry", cascade_registry=registry,
               cascade_feature_phase=bool(rs.get("features", True)),
               cascade_feature_time_budget_s=rs.get("feature_time_budget"))
    commit = str(rec.get("commit") or "").split()[0] if rec.get("commit") else None
    commit = None if commit in (None, "unknown") else commit
    seconds = rec.get("seconds") or {}
    metrics = dict({name: None for name in RESULT_METRICS},
                   exact_iou=exact.get("iou"), sampled_iou=sampled.get("iou"), voxel_iou=(rec.get("run") or {}).get("voxel_iou"),
                   missing_volume=exact.get("missing"), invented_volume=exact.get("invented"),
                   freecad_valid=bool(model.get("one_valid_solid")), n_solids=model.get("solids"),
                   stopped=(rec.get("run") or {}).get("stopped"), n_operations=len(program),
                   operations_by_kind=_count(f"{o.get('op')}:{(o.get('primitive') or {}).get('kind')}" for o in program),
                   operations_by_source=model.get("by_source"), chamfer_mean=None, chamfer_max=None, grade=None,
                   solution_stats=None, input_voxels=(rec.get("run") or {}).get("input_voxels"), fallback_boxes=None,
                   feature_phase=None)
    run = dict(schema=SCHEMA, part=Path(str(input_path)).stem, label=label or f"server {rec.get('host')}",
               timestamp=str(rec.get("started") or time.strftime("%Y-%m-%dT%H:%M:%S")), time=time.time(), host=rec.get("host"),
               input=file_identity(input_path), method="cascade", finished=rec.get("state") == "ended",
               code=dict(commit=commit, short=commit[:8] if commit else None, branch="server", dirty=False, modified=[],
                         repository=None),
               steps=None, settings=dict(gui=gui, config=None, run_part=plain(rs)), program=plain(program, limit=None),
               metrics=metrics, timings=dict(wall=seconds.get("reconstruction"), cpu=None, sections=None, peak_rss_mb=None,
                                             all=seconds.get("all")),
               model=dict(step=None, stl=None, note=None),
               extra=plain(dict(extra or {}, run_part_dir=str(run_dir), record_state=rec.get("state"))))
    base = part_directory(run["part"], store=store)
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = f"{stamp}_{run['code']['short'] or 'nocommit'}_{_slug(run['label'])}"
    final, n = base / stem, 1
    while final.exists() or final.with_name(final.name + ".tmp").exists():
        n += 1
        final = base / f"{stem}-{n}"
    tmp = final.with_name(final.name + ".tmp")
    tmp.mkdir()
    try:
        for key, name in (("step", "model.step"), ("stl", "model.stl")):
            if (run_dir / name).is_file():
                shutil.copyfile(run_dir / name, tmp / name)
                run["model"][key] = name
        for name in ("rating.json", "record.json", "program.json", "path.jsonl", "compare.json"):
            if (run_dir / name).is_file():
                shutil.copyfile(run_dir / name, tmp / name)
        rows, prev = [], 0.0
        if (run_dir / "steps.jsonl").is_file():
            for line in (run_dir / "steps.jsonl").read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                t = float(r.get("seconds") or prev)
                rows.append(dict(index=r.get("index"), stage=r.get("kind"), phase=r.get("phase"), message=r.get("message"),
                                 seconds=max(0.0, t - prev)))
                prev = t
        run["steps"] = len(rows) or None
        run["name"] = final.name
        with open(tmp / STEPS_JSONL, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        (tmp / RUN_JSON).write_text(json.dumps(run, indent=1, allow_nan=False, default=str))
        os.rename(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


# ------------------------------------------------------------------------------------------------ reading

def load_run(path) -> dict:
    """run.json of a run directory (or the file itself), with "path" = its directory. A program that was cut to text
    when it was saved (`truncated`, runs before 2026-10-01) is taken from RECOVERED_PROGRAM beside it when there is one
    (scripts/recover_run_program.py; run.json itself is never rewritten): "program_source" names the file the program
    came from, "program_truncated" counts the nodes cut in run.json."""
    p = Path(path)
    p = p / RUN_JSON if p.is_dir() else p
    run = json.loads(p.read_text())
    if not isinstance(run, dict):
        raise ValueError(f"{p}: not a run (run.json holds a {type(run).__name__})")
    run["path"] = str(p.parent)
    cut = truncated(run.get("program"))
    run["program_source"], run["program_truncated"] = RUN_JSON, len(cut)
    recovered = p.parent / RECOVERED_PROGRAM
    if cut and recovered.is_file():
        data = json.loads(recovered.read_text())
        ops = data.get("program") if isinstance(data, dict) else data
        if isinstance(ops, list) and ops and not truncated(ops):
            run["program"], run["program_source"] = ops, RECOVERED_PROGRAM
    return run


def list_runs(part: Optional[str] = None, store=None, sha256: Optional[str] = None) -> List[dict]:
    """The runs of the store, newest first (of one part, or of one input by content); half-written ones are left out."""
    root = Path(store) if store else store_root()
    out = []
    if not root.is_dir():
        return out
    wanted = None if part is None else set(part_directories(part, store=root))
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or (wanted is not None and directory not in wanted):
            continue
        for run_dir in directory.iterdir():
            if run_dir.name.endswith(".tmp") or not (run_dir / RUN_JSON).is_file():
                continue
            try:
                run = load_run(run_dir)
            except (OSError, ValueError):
                continue
            if sha256 is None or (run.get("input") or {}).get("sha256") == sha256:
                out.append(run)
    return sorted(out, key=_when, reverse=True)


def _when(run: dict) -> tuple:
    """The order of the runs: the time of saving (seconds since the epoch, exact within a second too), then the name."""
    return (float(run.get("time") or 0.0), str(run.get("timestamp")), str(run.get("name")))


def resolve(spec, store=None) -> Path:
    """A run directory from a path or from `part:latest`, `part:previous`, `part:<n>` (the n-th newest, 1 = latest)."""
    text = str(spec)
    if Path(text).exists():
        return Path(text)
    if ":" in text:
        part, which = text.rsplit(":", 1)
        runs = list_runs(part, store=store)
        n = {"latest": 1, "previous": 2}.get(which) or (int(which) if which.isdigit() else 0)
        if n < 1:
            raise ValueError(f"{spec}: expected part:latest, part:previous or part:<n>")
        if len(runs) < n:
            raise ValueError(f"{spec}: the store has {len(runs)} run(s) of {part!r}")
        return Path(runs[n - 1]["path"])
    raise ValueError(f"{spec}: no such run directory")


def previous_run(run: dict, store=None) -> Optional[dict]:
    """The newest earlier run of the same input (same sha256), None when there is none."""
    sha = (run.get("input") or {}).get("sha256")
    if not sha:
        return None
    for other in list_runs(store=store, sha256=sha):
        if _when(other) < _when(run):
            return other
    return None


# ------------------------------------------------------------------------------------------------ comparing

def _flat(d, prefix: str = "") -> Dict[str, object]:
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = d
    return out


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _same(a, b, tolerance: float) -> bool:
    if _number(a) and _number(b):
        return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return all(_same(x, y, tolerance) for x, y in zip(a, b))
    return a == b


def _op_parameters(o: dict) -> Dict[str, object]:
    p = o.get("primitive") if isinstance(o.get("primitive"), dict) else o
    out = {}
    for key in ("center_mm", "quaternion", "u", "v", "combine", "label"):
        if p.get(key) is not None:
            out[key] = p[key]
    for group in ("dims_mm", "params"):
        for k, v in (p.get(group) or {}).items():
            out[f"{group}.{k}"] = v
    return out


def _op_name(o: dict) -> str:
    p = o.get("primitive") if isinstance(o.get("primitive"), dict) else o
    return f"{o.get('op')}:{p.get('kind')}"


def diff_runs(a, b, tolerance: float = TOLERANCE) -> dict:
    """What differs between two runs (dicts of run.json, or run directories): the settings that differ, every metric
    side by side with the difference, the programs aligned operation by operation, the code of both. Pure data."""
    a = a if isinstance(a, dict) else load_run(a)
    b = b if isinstance(b, dict) else load_run(b)
    sa, sb = _flat(a.get("settings") or {}), _flat(b.get("settings") or {})
    settings = [dict(key=k, a=sa.get(k), b=sb.get(k)) for k in sorted(set(sa) | set(sb))
                if not _same(sa.get(k), sb.get(k), tolerance)]
    ma, mb = _flat(a.get("metrics") or {}), _flat(b.get("metrics") or {})
    metrics = []
    for k in sorted(set(ma) | set(mb)):
        x, y = ma.get(k), mb.get(k)
        metrics.append(dict(key=k, a=x, b=y, difference=(y - x) if _number(x) and _number(y) else None,
                            changed=not _same(x, y, tolerance)))
    pa, pb = list(a.get("program") or []), list(b.get("program") or [])
    operations = []
    for i in range(max(len(pa), len(pb))):
        x, y = (pa[i] if i < len(pa) else None), (pb[i] if i < len(pb) else None)
        if x is None or y is None:
            operations.append(dict(index=i, status="only in b" if x is None else "only in a",
                                   a=_op_name(x) if x else None, b=_op_name(y) if y else None, parameters=[]))
            continue
        if _op_name(x) != _op_name(y):
            operations.append(dict(index=i, status="different operation", a=_op_name(x), b=_op_name(y), parameters=[]))
            continue
        qa, qb = _op_parameters(x), _op_parameters(y)
        changed = [dict(key=k, a=qa.get(k), b=qb.get(k),
                        difference=(qb[k] - qa[k]) if _number(qa.get(k)) and _number(qb.get(k)) else None)
                   for k in sorted(set(qa) | set(qb)) if not _same(qa.get(k), qb.get(k), tolerance)]
        operations.append(dict(index=i, status="changed" if changed else "same", a=_op_name(x), b=_op_name(y),
                               parameters=changed))
    ia, ib = a.get("input") or {}, b.get("input") or {}
    return dict(
        a=dict(path=a.get("path"), name=a.get("name"), timestamp=a.get("timestamp"), label=a.get("label"),
               finished=a.get("finished"), code=a.get("code")),
        b=dict(path=b.get("path"), name=b.get("name"), timestamp=b.get("timestamp"), label=b.get("label"),
               finished=b.get("finished"), code=b.get("code")),
        same_input=bool(ia.get("sha256")) and ia.get("sha256") == ib.get("sha256"),
        same_code=bool((a.get("code") or {}).get("commit")) and (a.get("code") or {}).get("commit") == (b.get("code") or {}).get("commit")
        and not (a.get("code") or {}).get("dirty") and not (b.get("code") or {}).get("dirty"),
        settings=settings, metrics=metrics, operations=operations,
        summary=dict(settings_changed=len(settings), metrics_changed=sum(1 for m in metrics if m["changed"]),
                     operations_a=len(pa), operations_b=len(pb),
                     operations_changed=sum(1 for o in operations if o["status"] != "same")))
