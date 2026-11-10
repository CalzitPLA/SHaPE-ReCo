#!/usr/bin/env python3
"""
The path a reconstruction run took, written while it runs (`path.jsonl`): which operation entered or left the program at
which step, and every candidate the loop looked at with its primitive, its gain and the reason it was taken or not.
steps.jsonl only has the step kinds and messages; the validation cycle (reco_engine/validation.py) needs the operation
index of every step to find the step at which a reconstruction went wrong (the separation point).

    rec = PathRecorder(out / "path.jsonl")
    for i, step in enumerate(residual_steps(...)):
        rec.feed(i, step, seconds)
    rec.close()
    log = load_path(out / "path.jsonl")         # events, the final program's operations and where each one came from
    result = program_from_path(log)             # the program of a run that was stopped from outside (a timeout)

Events (one JSON object per line, flushed per line):
  grid          origin (world centre of cell [0, 0, 0]), pitch, shape - the input step's grid, to turn voxel boxes
                into world boxes (index -> world is origin + index x pitch on every axis, cascade.grid_frame)
  step          index, seconds, kind, phase (loop / feature), branch (main / classic), n_ops (accepted operations of
                the branch after the step), and what the step's info carries: op, primitive, gain, ok, reason, score,
                source, piece_id, iteration, stopped, ..., bbox_world for a voxel box
  op_added      op_index, op, primitive, fingerprint, net_gain, step index - an operation that entered the program
  op_removed    op_index, fingerprint, step index - an operation that left it (a failed feature confirmation truncates
                the program, residual_loop.py:3565; the classic solution replaces it, solution_choice.py:261)
  alternative   the first add's pruned alternatives (rank, primitive), at n_ops 0
The operation changes come from comparing each step's snapshot of the accepted operations with the previous snapshot of
the same branch, so they hold however the loop changed the program.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

KEEP = ("op", "gain", "ok", "reason", "surface", "score", "precision", "recall", "source", "piece_id", "iteration",
        "stopped", "kept", "volume", "share", "voxels", "net_gain", "recut", "member", "verdict", "label", "why")


def prim_dict(p) -> Optional[dict]:
    """A schema Primitive as compact JSON (what rebuilds it: kind, pose, dims, segment, composite parts)."""
    if p is None:
        return None
    if isinstance(p, dict):
        return {k: p[k] for k in ("kind", "quaternion", "center_mm", "dims_mm", "u", "v", "label", "parts", "combine") if k in p}
    if not hasattr(p, "kind") or not hasattr(p, "center_mm"):
        return dict(kind=type(p).__name__)
    return dict(kind=p.kind, quaternion=[float(x) for x in p.quaternion], center_mm=[float(x) for x in p.center_mm],
                dims_mm={k: float(v) for k, v in (p.dims_mm or {}).items()},
                u=[float(x) for x in p.u] if p.u is not None else None, v=[float(x) for x in p.v] if p.v is not None else None,
                label=p.label, parts=[prim_dict(q) for q in (p.parts or [])], combine=p.combine)


def fingerprint(op: str, prim: Optional[dict]) -> str:
    def rounded(x):
        if isinstance(x, float):
            return round(x, 9)
        if isinstance(x, list):
            return [rounded(y) for y in x]
        if isinstance(x, dict):
            return {k: rounded(v) for k, v in sorted(x.items()) if k != "label"}
        return x
    return hashlib.sha1(json.dumps([op, rounded(prim)], sort_keys=True).encode()).hexdigest()[:16]


def jsonable(v, depth: int = 0):
    if depth > 6:
        return None
    if isinstance(v, (bool, int, str)) or v is None:
        return v
    if isinstance(v, float):
        return v if np.isfinite(v) else None
    if isinstance(v, np.generic):
        return jsonable(v.item(), depth)
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 12 else None
    if isinstance(v, (list, tuple)):
        return [jsonable(x, depth + 1) for x in v[:32]]
    if isinstance(v, dict):
        return {str(k): jsonable(x, depth + 1) for k, x in list(v.items())[:32]}
    if hasattr(v, "kind") and hasattr(v, "center_mm"):
        return prim_dict(v)
    return str(v)[:200]


class PathRecorder:
    def __init__(self, path):
        self.path = Path(path)
        self.f = open(self.path, "w")
        self.prev: Dict[str, List[str]] = {}
        self.origin: Optional[np.ndarray] = None
        self.pitch: Optional[float] = None

    def close(self):
        if not self.f.closed:
            self.f.close()

    def _write(self, ev: dict):
        self.f.write(json.dumps(ev) + "\n")

    def _grid(self, spec):
        try:
            from reco_engine import cascade
            c = cascade.grid_centres(spec)
            self.origin, self.pitch = c[0, 0, 0].copy(), float(c[1, 0, 0, 0] - c[0, 0, 0, 0])
            self._write(dict(event="grid", origin=self.origin.tolist(), pitch=self.pitch, shape=list(c.shape[:3])))
        except Exception as exc:                                          # the path is advisory: never break the run
            self._write(dict(event="grid", error=f"{type(exc).__name__}: {exc}"[:200]))

    def _bbox_world(self, bbox):
        if self.origin is None or bbox is None:
            return None
        try:
            lo, hi = (np.asarray(b, float) for b in bbox)
            return [(self.origin + (lo - 0.5) * self.pitch).tolist(), (self.origin + (hi + 0.5) * self.pitch).tolist()]
        except Exception:
            return None

    def feed(self, index: int, step, seconds: float) -> List[dict]:
        info = step.info or {}
        if self.origin is None and info.get("spec") is not None:
            self._grid(info["spec"])
        branch = str(info.get("solution") or "main")
        ops = [o for o in (step.result.operations if step.result is not None else []) if o.accepted]
        dicts = [prim_dict(o.primitive) for o in ops]
        fps = [fingerprint(o.op, d) for o, d in zip(ops, dicts)]
        prev = self.prev.get(branch, [])
        k = 0
        while k < min(len(prev), len(fps)) and prev[k] == fps[k]:
            k += 1
        events = []
        ev = dict(event="step", index=index, seconds=round(float(seconds), 3), kind=step.kind,
                  phase=str(info.get("phase", "loop")), branch=branch, n_ops=len(fps))
        for key in KEEP:
            if key in info:
                ev[key] = jsonable(info[key])
        if info.get("primitive") is not None:
            ev["primitive"] = prim_dict(info["primitive"])
        if "bbox" in info:
            ev["bbox_world"] = self._bbox_world(info["bbox"])
        msg = info.get("message")
        if msg:
            ev["message"] = str(msg)[:300]
        events.append(ev)
        for j in range(len(prev) - 1, k - 1, -1):
            events.append(dict(event="op_removed", op_index=j, fingerprint=prev[j], index=index, branch=branch,
                               kind=step.kind))
        for j in range(k, len(fps)):
            o = ops[j]
            events.append(dict(event="op_added", op_index=j, op=o.op, primitive=dicts[j], fingerprint=fps[j],
                               net_gain=float(o.net_gain or 0.0), index=index, branch=branch, kind=step.kind,
                               source=str(info.get("source") or step.kind)))
        if step.kind == "add" and info.get("primitive") is not None and k == 0:
            for rank, alt in enumerate(getattr(info["primitive"], "alternatives", None) or []):
                events.append(dict(event="alternative", index=index, rank=rank, op="add", primitive=prim_dict(alt)))
        self.prev[branch] = fps
        for e in events:
            self._write(e)
        self.f.flush()
        return events


# ------------------------------------------------------------------------------------------------------- reading

@dataclass
class PathLog:
    events: List[dict]
    grid: Optional[dict]
    branch: str                                    # the branch of the final program
    final: List[dict] = field(default_factory=list)   # op_added events of the final program, in order (op_index 0..n-1)
    removed: List[dict] = field(default_factory=list)  # every op_removed event (withdrawn operations)
    candidates: List[dict] = field(default_factory=list)   # step events that carry a primitive and are not accepted ops
    stopped: Optional[str] = None


def load_path(path) -> PathLog:
    events = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    grid = next((e for e in events if e.get("event") == "grid"), None)
    steps = [e for e in events if e.get("event") == "step"]
    done = next((e for e in reversed(steps) if e.get("kind") == "done"), None)
    branch = (done or (steps[-1] if steps else {})).get("branch", "main")
    # the classic solution is the final program when the solution step says so
    sol = next((e for e in reversed(steps) if e.get("kind") == "solution"), None)
    if sol is not None and str(sol.get("kept", "")).startswith("classic"):
        branch = "classic"
    current: Dict[int, dict] = {}
    for e in events:
        if e.get("branch", "main") != branch:
            continue
        if e.get("event") == "op_added":
            current[e["op_index"]] = e
        elif e.get("event") == "op_removed":
            current.pop(e["op_index"], None)
    final = [current[i] for i in sorted(current)]
    removed = [e for e in events if e.get("event") == "op_removed"]
    candidates = [e for e in steps if e.get("primitive") is not None and e.get("kind") != "add"] + \
        [dict(e, kind="alternative") for e in events if e.get("event") == "alternative"]
    return PathLog(events, grid, branch, final, removed, candidates,
                   (done or {}).get("stopped") if done else None)


def program_from_path(log: PathLog, input_path: str = ""):
    """The final program as a ReconstructionResult (for a run that ended without writing program.json)."""
    from reco_engine.result_schema import ReconstructionResult
    return ReconstructionResult.from_dict(dict(input_path=input_path, operations=[
        dict(op=e["op"], primitive=dict(e["primitive"], parts=e["primitive"].get("parts") or []), net_gain=e.get("net_gain", 0.0))
        for e in log.final]))
