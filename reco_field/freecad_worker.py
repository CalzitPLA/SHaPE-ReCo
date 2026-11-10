#!/usr/bin/env python3
"""Run FreeCAD scripts in a LONG-LIVED freecadcmd instead of one process per call (opt-in).

Every build of the data generator (the primitive macro, `export.build_in_freecad`, ...) starts `freecadcmd` once; the
start alone costs about 1.75 s (measured on the laptop), and a generated sample makes about three such calls - roughly
half of a sample's time once the field runs on the GPU. With SHAPE_RECO_FREECAD_WORKER=1, `run_script` hands the same
script + environment to one persistent freecadcmd per host process (`freecad_worker_server.py`), which runs it with
runpy under exactly that environment and working directory, and is restarted every `RESTART_EVERY` jobs (FreeCAD
documents a script forgets to close cannot pile up) or after a timeout. Without the variable, `run_script` is a plain
`subprocess.run` of the same command - the behaviour every caller had before.

    rc, out = run_script(freecad, script_path, env, cwd=None, timeout=300)
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

SERVER = Path(__file__).resolve().parent / "freecad_worker_server.py"
RESTART_EVERY = 50


def enabled() -> bool:
    return os.environ.get("SHAPE_RECO_FREECAD_WORKER", "0") == "1"


class _Worker:
    def __init__(self, freecad: str):
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        self.proc = subprocess.Popen([freecad, str(SERVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        self.jobs = 0

    def run(self, script: str, env: Dict[str, str], cwd: Optional[str], timeout: float) -> Tuple[int, str]:
        tag = uuid.uuid4().hex
        self.proc.stdin.write(json.dumps(dict(script=str(script), env=dict(env), cwd=cwd, tag=tag)) + "\n")
        self.proc.stdin.flush()
        lines, result = [], {}

        def read():
            for line in self.proc.stdout:
                if line.startswith("@@DONE " + tag):
                    result["rc"] = int(line.split()[2])
                    return
                lines.append(line)
        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(timeout)
        self.jobs += 1
        if "rc" not in result:                              # timeout or the worker died: the caller sees a failure
            self.close()
            return -9, "".join(lines) + f"\n[freecad worker: no answer within {timeout:.0f} s]"
        return result["rc"], "".join(lines)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self):
        try:
            self.proc.kill()
        except Exception:
            pass


_workers: Dict[str, _Worker] = {}
_lock = threading.Lock()


def _close_all():
    for w in _workers.values():
        w.close()


atexit.register(_close_all)


def run_script(freecad: str, script, env: Dict[str, str], cwd: Optional[str] = None,
               timeout: Optional[float] = None) -> Tuple[int, str]:
    """(return code, combined stdout/stderr) of `freecadcmd script` under `env` and `cwd`."""
    if not enabled():
        proc = subprocess.run([freecad, str(script)], cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return proc.returncode, proc.stdout
    with _lock:
        w = _workers.get(freecad)
        if w is None or not w.alive() or w.jobs >= RESTART_EVERY:
            if w is not None:
                w.close()
            w = _workers[freecad] = _Worker(freecad)
        return w.run(script, env, cwd, timeout or 600.0)
