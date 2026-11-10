#!/usr/bin/env bash
# One-time setup on the institute server (server_a) for running the test suite there. Idempotent. Runs ON the server:
#   bash server/server_a/setup_tests.sh
# Everything lies below $ROOT (/data/user/shape_reco: the directory the institute gave for software):
#   venv_tests/   Python 3.11 venv: torch cu128 (server/setup_venv.sh), server/requirements.txt, and what only the
#                 tests need (PyQt5, vtk, psutil, h5py, numba, pandas)
#   tools/parallel  GNU parallel, the controller of the runs (one perl script; no root needed)
#   bin/freecadcmd  wrapper of the extracted FreeCAD 1.0.0 AppImage (set up by the comparison batch, 2026-09-29)
set -euo pipefail
ROOT=${ROOT:-/data/user/shape_reco}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$ROOT/tools" "$ROOT/bin" "$ROOT/tmp"

if [ ! -x "$ROOT/tools/parallel" ]; then
  cd "$ROOT/tools"
  curl -sSL --max-time 300 -o parallel.tar.bz2 https://ftp.gnu.org/gnu/parallel/parallel-latest.tar.bz2
  tar xjf parallel.tar.bz2
  cp parallel-*/src/parallel "$ROOT/tools/parallel"
  chmod +x "$ROOT/tools/parallel"
  rm -rf parallel-*/ parallel.tar.bz2
fi
export PARALLEL_HOME="$ROOT/tools/parallel_home"                        # its own files below $ROOT, not in the NFS home
mkdir -p "$PARALLEL_HOME" && touch "$PARALLEL_HOME/will-cite"           # GNU parallel is cited in docs/RUNNING.md
"$ROOT/tools/parallel" --version | head -1

if [ -x "$ROOT/freecad/squashfs-root/usr/bin/freecadcmd" ]; then
  cat > "$ROOT/bin/freecadcmd" <<'WRAP'
#!/bin/bash
# freecadcmd of the extracted AppImage (FreeCAD 1.0.0), headless, with its user files kept below /data/user
ROOT=/data/user/shape_reco
export HOME=$ROOT/fc_home
export XDG_CONFIG_HOME=$HOME/.config XDG_DATA_HOME=$HOME/.local/share XDG_CACHE_HOME=$HOME/.cache
export TMPDIR=${TMPDIR:-$ROOT/tmp}
export QT_QPA_PLATFORM=offscreen
unset PYTHONPATH PYTHONHOME DISPLAY WAYLAND_DISPLAY
mkdir -p "$HOME" "$TMPDIR"
exec "$ROOT/freecad/squashfs-root/usr/bin/freecadcmd" "$@"
WRAP
  chmod +x "$ROOT/bin/freecadcmd"
  echo "freecadcmd: $ROOT/bin/freecadcmd"
else
  echo "no FreeCAD below $ROOT/freecad: the tests that need freecadcmd will skip"
fi

cd "$HERE"
VENV="$ROOT/venv_tests" bash server/setup_venv.sh
"$ROOT/venv_tests/bin/pip" install -q PyQt5==5.15.11 vtk psutil h5py numba pandas
"$ROOT/venv_tests/bin/python" -c "import PyQt5.QtWidgets, vtk, pytest; print('PyQt5, vtk', vtk.vtkVersion.GetVTKVersion(), 'pytest', pytest.__version__)"
