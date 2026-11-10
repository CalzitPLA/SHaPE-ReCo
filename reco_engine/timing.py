#!/usr/bin/env python3
"""
Where the time goes: named, nested sections with wall and CPU time.

    from reco_engine.timing import timed
    with timed("input/channels"):
        ...

Every section records its wall time and the process's CPU time (user + system of all threads - torch and BLAS run their
own - plus the child processes that finished inside it, e.g. freecadcmd). A section opened inside another gets the
path "outer/inner". Nothing is recorded unless a `Recorder` is active (`with Recorder() as rec:`), so the loop costs
nothing extra when nobody asks. `rec.take()` returns the sections closed since the last take (one step's worth);
`rec.summary()` aggregates the whole run by path: calls, wall, CPU, self time (minus the sub-sections), sorted by wall.
In a GUI the CPU also counts the GUI thread (drawing), so there it is an upper bound.
"""

from __future__ import annotations

import contextvars
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

_ACTIVE: contextvars.ContextVar = contextvars.ContextVar("reco_timing_recorder", default=None)


def cpu_seconds() -> float:
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


@dataclass
class Section:
    path: str
    wall: float
    cpu: float
    depth: int


@dataclass
class Total:
    path: str
    calls: int = 0
    wall: float = 0.0
    cpu: float = 0.0
    self_wall: float = 0.0                       # wall minus the direct sub-sections

    def as_dict(self) -> dict:
        return asdict(self)


class Recorder:
    def __init__(self):
        self.sections: List[Section] = []
        self._stack: List[str] = []
        self._open: List[float] = []                 # start times of the open sections, parallel to _stack
        self._taken = 0
        self._token = None
        self.t0 = time.time()
        self.cpu0 = cpu_seconds()

    def __enter__(self) -> "Recorder":
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, *exc):
        _ACTIVE.reset(self._token)
        return False

    def activate(self) -> "Recorder":
        """For a generator that cannot hold a `with` across its yields: the recorder stays active in this context."""
        _ACTIVE.set(self)
        return self

    @contextmanager
    def section(self, name: str):
        path = "/".join(self._stack + [name])
        self._stack.append(name)
        w0, c0 = time.time(), cpu_seconds()
        self._open.append(w0)
        try:
            yield
        finally:
            self._open.pop()
            self._stack.pop()
            self.sections.append(Section(path, time.time() - w0, cpu_seconds() - c0, len(self._stack)))

    def current(self) -> Optional[Tuple[str, float]]:
        """The innermost section that is open right now and the seconds it has been open; None between sections.
        For a progress display in another thread (read only; a section closing meanwhile gives None)."""
        try:
            return "/".join(self._stack), time.time() - self._open[-1]
        except IndexError:
            return None

    def take(self) -> List[Section]:
        """The sections closed since the last call."""
        new = self.sections[self._taken:]
        self._taken = len(self.sections)
        return new

    def summary(self, sections: Optional[List[Section]] = None) -> List[Total]:
        totals: Dict[str, Total] = {}
        for s in self.sections if sections is None else sections:
            t = totals.setdefault(s.path, Total(s.path))
            t.calls += 1
            t.wall += s.wall
            t.cpu += s.cpu
            t.self_wall += s.wall
        for s in self.sections if sections is None else sections:
            parent = s.path.rpartition("/")[0]
            if parent in totals:
                totals[parent].self_wall -= s.wall
        return sorted(totals.values(), key=lambda t: -t.wall)

    def elapsed(self) -> Dict[str, float]:
        return dict(wall=time.time() - self.t0, cpu=cpu_seconds() - self.cpu0)


@contextmanager
def timed(name: str):
    """A section of the active recorder; nothing when none is active."""
    rec = _ACTIVE.get()
    if rec is None:
        yield
        return
    with rec.section(name):
        yield


def format_sections(sections: List[Section], limit: int = 6, min_wall: float = 0.05) -> str:
    """'type_stage 3.1 s (CPU 9.0), channels 1.2 s (CPU 1.2), ...' - the top-level sections of one step, biggest first."""
    top = {}
    for s in sections:
        if s.depth == min(x.depth for x in sections):
            w, c = top.get(s.path, (0.0, 0.0))
            top[s.path] = (w + s.wall, c + s.cpu)
    items = sorted(top.items(), key=lambda kv: -kv[1][0])
    items = [(p, w, c) for p, (w, c) in items if w >= min_wall][:limit]
    return ", ".join(f"{p} {w:.1f} s (CPU {c:.1f})" for p, w, c in items)


def format_summary(totals: List[Total], elapsed: Optional[Dict[str, float]] = None, limit: int = 25) -> str:
    """A fixed-width table of the run's sections, biggest first."""
    wall_all = elapsed["wall"] if elapsed else (max((t.wall for t in totals), default=0.0) or 1.0)
    lines = [f"{'section':44s} {'calls':>6s} {'wall s':>8s} {'share':>6s} {'self s':>8s} {'CPU s':>8s} {'cores':>6s}"]
    for t in totals[:limit]:
        cores = t.cpu / t.wall if t.wall > 0.05 else float("nan")
        lines.append(f"{t.path[:44]:44s} {t.calls:6d} {t.wall:8.1f} {t.wall / wall_all:6.1%} {t.self_wall:8.1f} "
                     f"{t.cpu:8.1f} {cores:6.1f}")
    if elapsed:
        lines.append(f"{'(whole run)':44s} {'':6s} {elapsed['wall']:8.1f} {1:6.1%} {'':8s} {elapsed['cpu']:8.1f} "
                     f"{elapsed['cpu'] / max(elapsed['wall'], 1e-9):6.1f}")
    return "\n".join(lines)


def timed_fn(name: Optional[str] = None):
    """Decorator: the function's calls are sections of the active recorder (named `name`, default the function name).
    Not for generators: their time would include what the consumer does between the yields."""
    def wrap(fn):
        import functools
        label = name or fn.__name__

        @functools.wraps(fn)
        def inner(*a, **k):
            rec = _ACTIVE.get()
            if rec is None:
                return fn(*a, **k)
            with rec.section(label):
                return fn(*a, **k)
        return inner
    return wrap
