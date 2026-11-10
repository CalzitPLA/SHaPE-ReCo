#!/usr/bin/env bash
# One-time pip setup of the shared venv on the NFS home (docs/SERVER_TRAINING.md). Idempotent; rerun after a pin change.
#   bash server/setup_venv.sh            # -> ~/shape_reco_venv, then: source ~/shape_reco_venv/bin/activate
# torch: cu128 wheels (sm_89 RTX 5000 Ada on server_a, sm_120 RTX 5050 on server_b; driver 610 / CUDA 13.3 runs them).
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-$HOME/shape_reco_venv}
TORCH=${TORCH:-torch==2.7.1}
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel
"$VENV/bin/pip" install "$TORCH" --index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/pip" install -r server/requirements.txt
"$VENV/bin/python" - <<'PY'
import importlib, torch
for m in ["numpy", "scipy", "trimesh", "rtree", "stl", "igl", "manifold3d", "sklearn", "joblib", "matplotlib", "networkx"]:
    importlib.import_module(m)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
command -v "$VENV/bin/stltovoxel" >/dev/null && echo "stltovoxel ok"
command -v freecadcmd >/dev/null && echo "freecadcmd: $(command -v freecadcmd)" || echo "no freecadcmd on $(hostname) (generation runs on server_b/server_e/server_f)"
