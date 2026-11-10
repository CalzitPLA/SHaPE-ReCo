#!/usr/bin/env python3
"""
Follow a reconstruction that runs on a server (scripts/remote_run.py; the user, 2026-10-29: "compute on the server but
preview live on the computer"). The mirror's path.jsonl grows while the server run writes its own (ssh tail -F); every
second the new lines are read: a step becomes a log line, an operation that entered or left the program rebuilds the
model (reco_engine.run_path.program_from_path -> result_to_program.program_model -> a mesh of its SDF, in a thread) and
the window shows it over the input part.

    f = ServerFollow(mirror_dir)
    f.stepLine.connect(log); f.modelReady.connect(show); f.finished.connect(done); f.start()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PyQt5 import QtCore

MESH_RESOLUTION = 96            # the model's mesh: a preview, the exact model comes with `remote_run.py fetch`
CANDIDATE_RESOLUTION = 40       # a candidate's mesh: what is being tried, roughly (the user, 2026-10-29: "not 100 % live,
                                # but roughly show which shapes we are investigating")
SHAPE_KINDS = ("candidate", "accepted", "rejected", "add", "member_op", "feature_candidate", "feature_accepted")


def shape_caption(e: dict) -> str:
    """What a step with a primitive is doing, in a few words: 'step 82: accepted cut box (gain 9242)'."""
    p = e.get("primitive") or {}
    verdict = {"candidate": "trying", "feature_candidate": "trying", "accepted": "accepted", "feature_accepted": "accepted",
               "rejected": "rejected", "add": "first add", "member_op": "member"}.get(e.get("kind"), e.get("kind"))
    if e.get("kind") in ("candidate", "feature_candidate") and e.get("ok") is False:
        verdict = "refused"
    extra = (f"gain {e['gain']}" if e.get("gain") is not None else
             f"score {float(e['score']):.3f}" if e.get("score") is not None else "")
    why = f" - {str(e.get('reason'))[:60]}" if e.get("reason") and verdict in ("refused", "rejected") else ""
    return (f"step {int(e.get('index', 0)) + 1}: {verdict} {e.get('op') or ''} {p.get('kind')}"
            + (f" ({extra})" if extra else "") + why).replace("  ", " ")


def primitive_mesh(prim: dict, resolution: int = CANDIDATE_RESOLUTION):
    """(vertices, triangles) of one primitive of a path event (a composite included), or None."""
    from reco_engine.result_schema import ReconstructionResult
    from reco_engine.result_to_program import primitive_to_sp
    from reco_field.peel import mesh_from_sdf
    d = dict(prim, parts=prim.get("parts") or [])
    p = ReconstructionResult.from_dict(dict(input_path="", operations=[dict(op="add", primitive=d)])).operations[0].primitive
    v, f = mesh_from_sdf(primitive_to_sp(p), resolution=resolution)
    return (v, f) if len(f) else None


def step_line(e: dict) -> str:
    """The log line of a step event (as the window's own: index, phase, kind, message)."""
    return (f"[server] {int(e.get('index', 0)) + 1:3d} {str(e.get('phase', '')):5s} {str(e.get('kind', '')):10s} "
            f"{str(e.get('message', ''))[:300]}  ({float(e.get('seconds', 0.0)):.0f} s)")


def model_mesh(path_jsonl: Path, resolution: int = MESH_RESOLUTION):
    """(vertices, triangles, number of operations) of the program the path describes so far, or None without one."""
    from reco_engine.run_path import load_path, program_from_path
    from reco_engine.result_to_program import program_model
    from reco_field.peel import mesh_from_sdf
    log = load_path(path_jsonl)
    result = program_from_path(log)
    model = program_model(result)
    if model is None:
        return None
    v, f = mesh_from_sdf(model, resolution=resolution)
    return v, f, len(result.operations)


class _MeshThread(QtCore.QThread):
    done = QtCore.pyqtSignal(object)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            self.done.emit(model_mesh(self.path))
        except Exception as exc:                                    # a half-written line or an empty program
            self.done.emit(exc)


class _ShapeThread(QtCore.QThread):
    done = QtCore.pyqtSignal(object, object)                        # the event, (v, f) or None

    def __init__(self, event: dict, parent=None):
        super().__init__(parent)
        self.event = event

    def run(self):
        try:
            self.done.emit(self.event, primitive_mesh(self.event["primitive"]))
        except Exception:
            self.done.emit(self.event, None)


class ServerFollow(QtCore.QObject):
    """Polls a mirror directory (link.json, path.jsonl) and reports what the server run did: every step as a line, the
    model after every operation, the latest primitive it tried (candidateReady) and the leftover it works on
    (regionReady: a component's box)."""
    stepLine = QtCore.pyqtSignal(str)
    modelReady = QtCore.pyqtSignal(object, object, str)            # vertices, triangles, caption
    candidateReady = QtCore.pyqtSignal(object, object, str, str)   # vertices, triangles, caption, op ("add" | "cut")
    regionReady = QtCore.pyqtSignal(object, object, str)           # lo, hi (world), caption
    finished = QtCore.pyqtSignal(str)

    def __init__(self, mirror, parent=None, interval_ms: int = 1000, idle_end_s: float = 120.0):
        super().__init__(parent)
        self.mirror = Path(mirror).expanduser()
        self.link = json.loads((self.mirror / "link.json").read_text()) if (self.mirror / "link.json").is_file() else {}
        self.path = self.mirror / "path.jsonl"
        self._pos, self._rest, self._steps, self._ops_dirty, self._done = 0, b"", 0, False, False
        self._mesh_pending = False                                  # a mesh is computed or its signal not delivered yet
        self._idle, self._idle_end = 0.0, idle_end_s
        self._interval = interval_ms
        self._thread: Optional[_MeshThread] = None
        self._shape_thread: Optional[_ShapeThread] = None
        self._latest_shape: Optional[dict] = None                   # the newest event with a primitive not shown yet
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)

    @property
    def caption(self) -> str:
        return f"server run {self.link.get('id', self.mirror.name)} on {self.link.get('host', '?')}"

    def start(self):
        self.timer.start(self._interval)
        self.poll()

    def stop(self):
        self.timer.stop()

    def new_events(self):
        """The complete lines added since the last call, parsed (a partial last line waits for the next call)."""
        if not self.path.is_file():
            return []
        with open(self.path, "rb") as fh:
            fh.seek(self._pos)
            data = fh.read()
        self._pos += len(data)
        data = self._rest + data
        lines = data.split(b"\n")
        self._rest = lines.pop()
        out = []
        for line in lines:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        return out

    def poll(self):
        events = self.new_events()
        self._idle = 0.0 if events else self._idle + self._interval / 1000.0
        for e in events:
            kind = e.get("event")
            if kind == "step":
                self._steps += 1
                self.stepLine.emit(step_line(e))
                if e.get("kind") == "done":
                    self._done = True
            elif kind in ("op_added", "op_removed"):
                self._ops_dirty = True
            if kind == "step" and e.get("primitive") and e.get("kind") in SHAPE_KINDS:
                self._latest_shape = e
            if kind == "step" and e.get("bbox_world"):
                lo, hi = e["bbox_world"]
                self.regionReady.emit(lo, hi, f"step {int(e.get('index', 0)) + 1}: {e.get('op') or ''} component of "
                                              f"{e.get('voxels')} voxels")
        if self._latest_shape is not None and (self._shape_thread is None or not self._shape_thread.isRunning()):
            ev, self._latest_shape = self._latest_shape, None       # the newest only: roughly live, never behind
            self._shape_thread = _ShapeThread(ev, self)
            self._shape_thread.done.connect(self._shape_done)
            self._shape_thread.start()
        if self._ops_dirty and not self._mesh_pending:
            self._ops_dirty, self._mesh_pending = False, True
            self._thread = _MeshThread(self.path, self)
            self._thread.done.connect(self._mesh_done)
            self._thread.start()
        if self._idle >= self._idle_end and not self._done and self._process_alive():
            self._idle = 0.0                                        # a long step: the run is computing, not gone
        if (self._done or self._idle >= self._idle_end) and not self._ops_dirty and not self._mesh_pending:
            self.timer.stop()
            self.finished.emit("done" if self._done else f"no new line for {self._idle_end:.0f} s")

    def _process_alive(self) -> bool:
        """Is the server run's process still there? (One ssh, only after `idle_end_s` without a line: the bracket's
        follow ended after 120 s of one long step while the server still computed, the user's session of 2026-10-29.)"""
        if not self.link.get("host") or not self.link.get("pid"):
            return False
        import subprocess
        try:
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", self.link["host"], f"kill -0 {int(self.link['pid'])} 2>/dev/null "
                                f"&& echo alive || echo gone"], capture_output=True, text=True, timeout=20)
            return r.stdout.strip() == "alive"
        except Exception:
            return False

    def _shape_done(self, event, mesh):
        if mesh is not None:
            self.candidateReady.emit(mesh[0], mesh[1], shape_caption(event), str(event.get("op") or "add"))

    def _mesh_done(self, out):
        self._mesh_pending = False
        if isinstance(out, Exception) or out is None:
            return
        v, f, n = out
        self.modelReady.emit(v, f, f"{self.caption}\nstep {self._steps}, {n} operations" + (" - finished" if self._done else ""))
