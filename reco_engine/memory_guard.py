#!/usr/bin/env python3
"""
The memory watcher (the user's rule 2026-10-15: the framework must not be killed by the machine running out of
memory - the laptop has 7 GB, a grid, a field and a FreeCAD build beside the GUI fill it). Qt-free, used by the
loop (between rounds and pieces: the run stops cleanly with the model it has) and by the GUI's heartbeat (the run is
paused and the mesh caches dropped before the system reaper acts).

    guard = MemoryGuard(min_free_mb=1200)         # or $SHAPE_RECO_MIN_FREE_MB
    reason = guard.check()                        # None, or "memory low: 980 MB available (< 1200 MB)"
    guard.available_mb()                          # MemAvailable of /proc/meminfo (psutil's number when present)

The check reads one file; it is cheap enough for every round. A reading that fails (no /proc, no psutil) is "unknown"
and never stops anything.
"""

from __future__ import annotations

import gc
import os
import time
from typing import Callable, List, Optional

DEFAULT_MIN_FREE_MB = 1200.0


def available_mb() -> Optional[float]:
    """Memory the machine can still give out, in MB: psutil's `available`, else MemAvailable of /proc/meminfo."""
    try:
        import psutil
        return float(psutil.virtual_memory().available) / 1048576.0
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def default_min_free_mb() -> float:
    try:
        return float(os.environ.get("SHAPE_RECO_MIN_FREE_MB", DEFAULT_MIN_FREE_MB))
    except ValueError:
        return DEFAULT_MIN_FREE_MB


class MemoryGuard:
    """`check()` -> a reason when the available memory is below `min_free_mb` (after one try to free memory through
    the registered `release` callbacks and the garbage collector), else None. `min_interval_s` spaces the readings."""

    def __init__(self, min_free_mb: Optional[float] = None, min_interval_s: float = 0.5,
                 reader: Optional[Callable[[], Optional[float]]] = None):
        self.min_free_mb = float(min_free_mb if min_free_mb is not None else default_min_free_mb())
        self.min_interval_s = float(min_interval_s)
        self.reader = reader or (lambda: available_mb())      # resolved at call time (tests patch the module's reader)
        self.releases: List[Callable[[], None]] = []
        self.last: Optional[float] = None
        self.last_t = 0.0
        self.low_count = 0

    def available(self) -> Optional[float]:
        now = time.time()
        if self.last is None or now - self.last_t >= self.min_interval_s:
            self.last, self.last_t = self.reader(), now
        return self.last

    def release(self) -> None:
        """Free what can be freed: the registered caches, then the garbage collector."""
        for fn in list(self.releases):
            try:
                fn()
            except Exception:
                pass
        gc.collect()

    def check(self) -> Optional[str]:
        if self.min_free_mb <= 0:
            return None
        avail = self.available()
        if avail is None or avail >= self.min_free_mb:
            self.low_count = 0
            return None
        self.release()
        self.last, self.last_t = self.reader(), time.time()
        avail = self.last
        if avail is None or avail >= self.min_free_mb:
            return None
        self.low_count += 1
        return f"memory low: {avail:.0f} MB available (< {self.min_free_mb:.0f} MB)"
