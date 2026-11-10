#!/bin/bash
# Deploy the shape_reco custom node package into FreeCAD's installed
# Nodes workbench (github.com/j8sr0230/Nodes), so it shows up in the
# workbench's node-creation menu under the "Shape Reco" category.
#
# The Nodes workbench itself lives outside this repo (it's a FreeCAD
# addon, managed by FreeCAD's Addon Manager), so the node source is
# kept here under version control and copied out on demand rather than
# being edited in place.

set -euo pipefail

FREECAD_NODES_MOD="${FREECAD_NODES_MOD:-$HOME/.local/share/FreeCAD/Mod/Nodes}"
TARGET="$FREECAD_NODES_MOD/nodes/shape_reco"

if [ ! -d "$FREECAD_NODES_MOD" ]; then
    echo "FreeCAD Nodes workbench not found at: $FREECAD_NODES_MOD" >&2
    echo "Install it first via FreeCAD's Addon Manager, or set FREECAD_NODES_MOD." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$TARGET"
# nodes removed from the repository (e.g. route C's voxel nodes, 2026-09-30) would stay installed and fail to import
for f in "$TARGET"/shape_reco_*.py; do
    [ -e "$f" ] || continue
    [ -e "$SCRIPT_DIR/shape_reco/$(basename "$f")" ] || { rm -f "$f"; echo "removed stale node: $(basename "$f")"; }
done
cp "$SCRIPT_DIR"/shape_reco/*.py "$TARGET/"
# The installed copy no longer sits inside the repo, so record where the repo is.
(cd "$SCRIPT_DIR/.." && pwd) > "$TARGET/_sandbox_root.txt"

echo "Installed shape_reco nodes to: $TARGET"
echo "Restart FreeCAD (or reload the Nodes workbench) to pick them up."
