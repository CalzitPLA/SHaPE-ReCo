"""Where `freecadcmd` is: one rule for the whole project (2026-10-01, T-04). Standard library only.

    1. $SHAPE_RECO_FREECADCMD - a path or a command name. The institute servers have no /usr/bin/freecadcmd; their FreeCAD
       is the extracted 1.0 AppImage behind /data/user/shape_reco/bin/freecadcmd (server/server_a/run_tests.sh sets
       the variable). On server_b the variable also keeps the system's FreeCAD 0.20 at /usr/bin/freecadcmd out.
    2. `freecadcmd` on PATH.
    3. /usr/bin/freecadcmd (the laptop's FreeCAD 1.0).

A variable that names no executable is an error of the setup: it is reported (a warning) and NO other FreeCAD is taken in
its place, so a run never quietly builds with a FreeCAD nobody chose; callers then say "freecadcmd not found".
"""
from __future__ import annotations

import os
import shutil
import warnings
from typing import Optional

ENV = "SHAPE_RECO_FREECADCMD"
FALLBACK = "/usr/bin/freecadcmd"


def find_freecadcmd() -> Optional[str]:
    """The freecadcmd to run (rule above), or None when there is none."""
    chosen = os.environ.get(ENV, "").strip()
    if chosen:
        found = shutil.which(os.path.expanduser(chosen))
        if found is None:
            warnings.warn(f"{ENV}={chosen!r} is not an executable: no FreeCAD is used in its place", stacklevel=2)
        return found
    return shutil.which("freecadcmd") or (FALLBACK if os.path.isfile(FALLBACK) else None)


def freecadcmd_or_default() -> str:
    """find_freecadcmd() as text for a default (a command-line option, a GUI field): when there is none, the variable's
    own value or the fallback path, so the caller's "not found" message names what was looked for."""
    return find_freecadcmd() or os.environ.get(ENV, "").strip() or FALLBACK
