#!/usr/bin/env python3
"""
Reconstructions on a server, started from the window (the user, 2026-10-29: "how to start a server run via the gui";
"the server runs should be possible for every part to save computing time"). The run bar's "Run on" says where:
this computer, server_a or server_b. On a server, Start runs scripts/run_part.py there (scripts/remote_run.py: the laptop's
HEAD deployed once per commit, niced) with the window's settings - the path (start shape, mode, sub-parts, type source,
registry), the feature phase and its budget, the starting part of step 2 - and the window follows it live. Every server
run of the part that ends is fetched and stored in the run store (run_store.import_run_part), so the Benchmark tab,
Open, Compare, the nephogram and the event time treat it as any run. The alternatives (setting "reconstructions per
part") start together on the server once the first run is not exact - in parallel, not one after the other.

    args = server_args(settings)                        # run_part options of the window's settings
    runs = ServerRuns(parent); runs.stored.connect(...); runs.start(host, settings, entry, variant)
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

from PyQt5 import QtCore

ROOT = Path(__file__).resolve().parents[1]
HOSTS = (("local", "this computer"), ("auto", "a server: the least loaded"), ("server_a", "server_a (32 cores)"),
         ("server_b", "server_b (20 cores)"), ("server_c", "server_c (16 cores)"), ("server_d", "server_d (16 cores)"))
INTERACTIVE_NICE = 0    # a run started in the window comes before the niced batch jobs (nice 10) on a shared server
POLL_MS = 15_000


def _remote_run():
    import importlib.util
    spec = importlib.util.spec_from_file_location("remote_run", ROOT / "scripts" / "remote_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def server_args(s) -> str:
    """The run_part options of the window's settings (RunSettings): the path fields, the feature phase and its budget.
    A registry inside the repository is passed by its path in the deployed checkout (models/<name>)."""
    args = []
    if (s.cascade_start_shape or "legacy") != "legacy":
        args += ["--start", f"'{s.cascade_start_shape}'"]
    if (s.cascade_mode or "loop") != "loop":
        args += ["--mode", s.cascade_mode]
    if s.cascade_subparts:
        args += ["--subparts", "--feature-subparts"]
    if (s.cascade_type_source or "registry") != "registry":
        args += ["--type-source", s.cascade_type_source]
    if s.cascade_registry:
        reg = Path(str(s.cascade_registry))
        args += ["--registry", f"models/{reg.name}" if (ROOT / "models" / reg.name).is_file() else str(reg)]
    if not s.cascade_feature_phase:
        args.append("--no-features")
    if s.cascade_feature_time_budget_s:
        args += ["--feature-time-budget", f"{float(s.cascade_feature_time_budget_s):g}"]
    return " ".join(args)


def write_first_guess(entry: Optional[dict]) -> Optional[Path]:
    """The window's starting part (StartPartDialog.entry()) as a first_guess.json for run_part --first-guess."""
    if not entry:
        return None
    folder = Path.home() / ".cache" / "shape_reco" / "first_guess"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"first_guess_{time.strftime('%Y%m%d-%H%M%S')}_{abs(hash(json.dumps(entry, sort_keys=True))) % 10**6}.json"
    path.write_text(json.dumps(dict(schema="first_guess/1", ordered=False, shapes=[entry]), indent=1))
    return path


def import_mirror(mirror) -> Optional[Path]:
    """A fetched server run (<mirror>/run) into the run store, once (<mirror>/stored.json remembers where)."""
    from reco_engine import run_store
    mirror = Path(mirror)
    marker = mirror / "stored.json"
    if marker.is_file():
        return Path(json.loads(marker.read_text())["path"])
    if not (mirror / "run" / "record.json").is_file():
        return None
    link = json.loads((mirror / "link.json").read_text())
    step = Path(link.get("step") or "")
    input_path = step if step.is_file() else mirror / "input.step"
    d = run_store.import_run_part(mirror / "run", input_path, label=f"server {link.get('host')} {link.get('variant') or ''}".strip(),
                                  extra=dict(server=link))
    marker.write_text(json.dumps(dict(path=str(d))))
    return d


class _Starter(QtCore.QThread):
    done = QtCore.pyqtSignal(object, str)                         # mirror (or None), error
    said = QtCore.pyqtSignal(str)                                 # what start_run reports (deploy, load, pid)

    def __init__(self, kwargs, parent=None):
        super().__init__(parent)
        self.kwargs = kwargs

    def run(self):
        try:
            self.done.emit(_remote_run().start_run(say=self.said.emit, **self.kwargs), "")
        except Exception as exc:
            self.done.emit(None, str(exc)[-400:])


class _Collector(QtCore.QThread):
    done = QtCore.pyqtSignal(object, object, str)                 # mirror, store dir (or None), error

    def __init__(self, mirror, parent=None):
        super().__init__(parent)
        self.mirror = mirror

    def run(self):
        try:
            rr = _remote_run()
            link = rr.read_link(self.mirror)
            if rr.run_state(link)["process"] == "running":
                self.done.emit(self.mirror, None, "running")
                return
            if not (Path(self.mirror) / "run" / "record.json").is_file():
                rr.cmd_fetch(type("A", (), {"mirror": str(self.mirror)})())
            self.done.emit(self.mirror, import_mirror(self.mirror), "")
        except Exception as exc:
            self.done.emit(self.mirror, None, str(exc)[-400:])


class ServerRuns(QtCore.QObject):
    """The server runs started from the window: started in a thread, polled every POLL_MS, fetched and stored when
    they end. Signals: started(mirror, variant), stored(mirror, store dir, variant), failed(text)."""
    started = QtCore.pyqtSignal(object, str)
    stored = QtCore.pyqtSignal(object, object, str)
    failed = QtCore.pyqtSignal(str)
    said = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, poll_ms: int = POLL_MS):
        super().__init__(parent)
        self.pending: Dict[str, str] = {}                         # mirror -> variant
        self.hosts: Dict[str, str] = {}                           # mirror -> the host it runs on ("auto" resolved)
        self._threads = []
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(poll_ms)

    def start(self, host: str, settings, entry: Optional[dict] = None, variant: str = "", name: Optional[str] = None):
        from reco_engine import benchmark
        step = Path(str(settings.input_stl))
        kwargs = dict(step=str(step), host=host, extra=server_args(settings), first_guess=write_first_guess(entry),
                      truth=benchmark.truth_for(step), name=name or (f"{step.stem}_{variant}" if variant else step.stem),
                      nice=INTERACTIVE_NICE)
        t = _Starter(kwargs, self)
        t.said.connect(self.said.emit)
        t.done.connect(lambda mirror, err, v=variant: self._started(mirror, err, v))
        self._threads = [x for x in self._threads if x.isRunning()] + [t]
        t.start()

    def _started(self, mirror, err: str, variant: str):
        if mirror is None:
            self.failed.emit(f"the server run did not start: {err}")
            return
        link_path = Path(mirror) / "link.json"
        link = json.loads(link_path.read_text())
        link["variant"] = variant
        self.hosts[str(mirror)] = link.get("host")
        link_path.write_text(json.dumps(link, indent=1))
        self.pending[str(mirror)] = variant
        self.started.emit(mirror, variant)

    def close(self, wait_ms: int = 5000):
        """The window closes: no more polling; a thread that fetches or starts is waited for (the server runs go on and
        can be followed again: File > Follow a server run)."""
        self.timer.stop()
        for t in self._threads:
            if t.isRunning():
                t.wait(wait_ms)

    def poll(self):
        for mirror in list(self.pending):
            if any(isinstance(t, _Collector) and t.mirror == mirror and t.isRunning() for t in self._threads):
                continue
            t = _Collector(mirror, self)
            t.done.connect(self._collected)
            self._threads = [x for x in self._threads if x.isRunning()] + [t]
            t.start()

    def _collected(self, mirror, store_dir, err: str):
        if err == "running":
            return
        variant = self.pending.pop(str(mirror), "")
        if err:
            self.failed.emit(f"server run {Path(mirror).name}: {err}")
        self.stored.emit(mirror, store_dir, variant)


def with_variant(settings, variant: str):
    """The window's settings with a path variant's fields (path_policy.apply) - an alternative's settings."""
    from reco_engine import path_policy
    fields = {k: v for k, v in path_policy.apply({}, variant).items() if hasattr(settings, k)}
    return replace(settings, **fields)
