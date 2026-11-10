#!/usr/bin/env bash
# SHaPe ReCo installer (Linux): a Python virtual environment with the packages, the command `shape-reco` and a desktop
# entry. It installs the checkout it lies in, in place, and writes only to the environment's directory and ~/.local.
#
#   ./install.sh                          the window and the command line; packages from PyPI
#   ./install.sh --with-torch             plus PyTorch, for the trained networks (checkpoints are not in the repository)
#   ./install.sh --system-site-packages   the environment sees the distribution's packages (PyQt5, VTK, ...)
#   ./install.sh --venv DIR               the environment in DIR (default ~/.local/share/shape-reco/venv)
#   ./install.sh --no-gui                 the command line only (no PyQt5 / VTK, no desktop entry)
#   ./install.sh --check                  check an installation: imports, FreeCAD, one reconstruction of an example part
#   ./install.sh --uninstall              remove the command, the desktop entry, the icon and the environment
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV="$HOME/.local/share/shape-reco/venv"
BIN="$HOME/.local/bin"
APPS="$HOME/.local/share/applications"
ICONS="$HOME/.local/share/icons/hicolor/scalable/apps"
ICONS_PNG="$HOME/.local/share/icons/hicolor/256x256/apps"
PY=${PYTHON:-python3}
TORCH=0 SYSTEM=0 GUI=1 CHECK=0 UNINSTALL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --with-torch) TORCH=1 ;;
    --system-site-packages) SYSTEM=1 ;;
    --venv) VENV=$(realpath -m "$2"); shift ;;
    --no-gui) GUI=0 ;;
    --check) CHECK=1 ;;
    --uninstall) UNINSTALL=1 ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning: %s\033[0m\n' "$*" >&2; }

find_freecad() {      # the freecadcmd the reconstruction will use (dataset_gen/src/freecad_path.py looks the same way)
  for c in "${SHAPE_RECO_FREECADCMD:-}" "$(command -v freecadcmd 2>/dev/null)" "$(command -v FreeCADCmd 2>/dev/null)" \
           "$(command -v freecad.cmd 2>/dev/null)"; do
    [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}

check() {
  local py="$VENV/bin/python" ok=1
  [ -x "$py" ] || { echo "no environment at $VENV - run ./install.sh first" >&2; exit 1; }
  say "packages"
  "$py" - <<'PYEOF' || ok=0
import importlib, sys
need = ["numpy", "scipy", "trimesh", "igl", "manifold3d", "sklearn", "joblib", "matplotlib", "PIL", "psutil", "stl", "yaml"]
opt = ["PyQt5", "vtk", "torch"]
bad = []
for m in need + opt:
    try:
        mod = importlib.import_module(m)
        print(f"  ok       {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  {'MISSING ' if m in need else 'absent  '} {m}" + ("" if m in opt else f" ({e})"))
        bad += [m] if m in need else []
sys.exit(1 if bad else 0)
PYEOF
  say "FreeCAD"
  if fc=$(find_freecad); then
    echo "  $fc: $("$fc" --version 2>/dev/null | head -1)"
  else
    warn "no freecadcmd found: the window starts, but no solid can be built (see README.md, Installation)"; ok=0
  fi
  if [ -n "${fc:-}" ]; then
    say "one reconstruction (dataset_gen/examples/beams/s05_rhs.step, the analytic registry, a few minutes)"
    local tmp; tmp=$(mktemp -d)
    if (cd "$ROOT" && SHAPE_RECO_FREECADCMD="$fc" "$py" scripts/run_part.py dataset_gen/examples/beams/s05_rhs.step \
          --stl dataset_gen/examples/beams/s05_rhs.stl --out "$tmp/run" --registry models/registry_analytic.json \
          --feature-time-budget 120 > "$tmp/run.log" 2>&1); then
      "$py" - "$tmp/run/record.json" <<'PYEOF' || ok=0
import json, sys
r = json.load(open(sys.argv[1]))
c = r.get("comparison") or {}
iou = (c.get("exact") or {}).get("iou") or (c.get("sampled") or {}).get("iou")
m = r.get("model") or {}
print(f"  {r.get('state')}: IoU {iou}, {m.get('operations')} operation(s), one valid solid: {m.get('one_valid_solid')}")
sys.exit(0 if r.get("state") == "ended" and iou is not None and iou > 0.9 else 1)
PYEOF
    else
      warn "the reconstruction failed - the log: $tmp/run.log"; ok=0
    fi
  fi
  [ $ok = 1 ] && say "the installation works" || { warn "the installation is incomplete (see above)"; exit 1; }
}

if [ $UNINSTALL = 1 ]; then
  say "removing the command, the desktop entry, the icon and $VENV"
  rm -f "$BIN/shape-reco" "$APPS/shape-reco.desktop" "$ICONS/shape-reco.svg" "$ICONS_PNG/shape-reco.png"
  case "$VENV" in "$HOME"/*) rm -rf "$VENV"; rmdir "$(dirname "$VENV")" 2>/dev/null || true ;;
                 *) warn "not removing $VENV (outside your home directory)" ;; esac
  command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" 2>/dev/null || true
  exit 0
fi
if [ $CHECK = 1 ]; then check; exit 0; fi

say "Python"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
  || { echo "Python 3.11 or newer is needed ($("$PY" --version 2>&1)); set PYTHON=/path/to/python3.x" >&2; exit 1; }
"$PY" -m venv --help >/dev/null 2>&1 \
  || { echo "the venv module is missing (Debian/Ubuntu: sudo apt install python3-venv)" >&2; exit 1; }
echo "  $("$PY" --version) at $(command -v "$PY")"

say "the environment: $VENV"
mkdir -p "$(dirname "$VENV")"
args=(); [ $SYSTEM = 1 ] && args+=(--system-site-packages)
[ -x "$VENV/bin/python" ] || "$PY" -m venv "${args[@]}" "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip wheel
say "packages (requirements.txt$([ $GUI = 1 ] && echo ', requirements-gui.txt')$([ $TORCH = 1 ] && echo ', torch'))"
"$VENV/bin/python" -m pip install --quiet -r "$ROOT/requirements.txt"
[ $GUI = 1 ] && "$VENV/bin/python" -m pip install --quiet -r "$ROOT/requirements-gui.txt"
[ $TORCH = 1 ] && "$VENV/bin/python" -m pip install --quiet torch

FC=""
if FC=$(find_freecad); then
  say "FreeCAD: $FC ($("$FC" --version 2>/dev/null | head -1))"
else
  warn "no freecadcmd found. Install FreeCAD 1.0 (Debian/Ubuntu: sudo apt install freecad; or the AppImage from
         https://www.freecad.org) and, if freecadcmd is not on the PATH, set SHAPE_RECO_FREECADCMD before running."
fi

say "the command: $BIN/shape-reco"
mkdir -p "$BIN"
FCLINE=""          # the found freecadcmd unless the user sets one (built here: quotes inside \${FC:+...} are removed)
[ -n "$FC" ] && FCLINE="[ -n \"\${SHAPE_RECO_FREECADCMD:-}\" ] || export SHAPE_RECO_FREECADCMD=\"$FC\""
cat > "$BIN/shape-reco" <<LAUNCHER
#!/usr/bin/env bash
# SHaPe ReCo, installed by $ROOT/install.sh
#   shape-reco [part.stl|part.step] [--full] [window options]   the window (python3 -m reco_gui --help)
#   shape-reco run part.step --out DIR [options]                 one reconstruction with records (scripts/run_part.py)
#   shape-reco reconstruct --input-stl part.step --workdir DIR   the headless loop, prints the result JSON
#   shape-reco peel part.stl --grade                             the vector-field route
#   shape-reco check                                             check this installation
ROOT="$ROOT"
PY="$VENV/bin/python"
$FCLINE
cd "\$ROOT"
case "\${1:-}" in
  run) shift; exec "\$PY" scripts/run_part.py "\$@" ;;
  reconstruct) shift; exec "\$PY" -m reco_engine.reconstruct_cli "\$@" ;;
  peel) shift; exec "\$PY" scripts/peel_case.py "\$@" ;;
  check) exec "\$ROOT/install.sh" --check --venv "$VENV" ;;
  -h|--help|help) sed -n '3,7p' "\$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac
win=maximized; args=()
for a in "\$@"; do
  case "\$a" in
    --full) win=full ;;
    *.stl|*.STL|*.step|*.STEP|*.stp|*.STP) args+=(--stl "\$(realpath "\$a")") ;;
    *) args+=("\$a") ;;
  esac
done
exec "\$PY" -m reco_gui --method cascade --window "\$win" "\${args[@]}"
LAUNCHER
chmod +x "$BIN/shape-reco"

if [ $GUI = 1 ]; then
  say "the desktop entry: $APPS/shape-reco.desktop"
  mkdir -p "$APPS" "$ICONS" "$ICONS_PNG"
  cp "$ROOT/packaging/shape-reco.svg" "$ICONS/shape-reco.svg"   # the logo badge (packaging/logo/make_logo.py --3d)
  cp "$ROOT/packaging/shape-reco.png" "$ICONS_PNG/shape-reco.png"
  cat > "$APPS/shape-reco.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=SHaPe ReCo
GenericName=Shape reconstruction
Comment=Reconstruct an STL/STEP part as CAD primitives and booleans
Exec=$BIN/shape-reco %f
Icon=shape-reco
Terminal=false
Categories=Science;Engineering;
MimeType=model/stl;model/step;application/sla;
Keywords=CAD;CSG;STEP;STL;reconstruction;
StartupNotify=true
Actions=FullScreen;

[Desktop Action FullScreen]
Name=Full screen
Exec=$BIN/shape-reco --full %f
DESKTOP
  command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" 2>/dev/null || true
fi

case ":$PATH:" in *":$BIN:"*) ;; *) warn "$BIN is not on your PATH: add it, or start $BIN/shape-reco directly" ;; esac
say "done - check the installation with: shape-reco check"
