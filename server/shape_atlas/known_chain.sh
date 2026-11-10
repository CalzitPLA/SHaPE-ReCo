#!/usr/bin/env bash
# shape atlas: build a known-tree corpus (validate_known.py build: STEP + STL + the truth gate) and run what passed
# through rerun.sh with its truths (every run rated); on server_b or server_a.
#   known_chain.sh <corpus dir> <run name> <jobs>
set -u
corpus=$1; name=$2; jobs=$3
ROOT=/data/user/shape_reco
CODE=${SHAPE_ATLAS_CODE:-$ROOT/wt/shape_atlas_20261029b}
OUT=$HOME/data/shape_reco/shape_atlas
export PATH="$ROOT/bin:$ROOT/venv_tests/bin:$PATH" SHAPE_RECO_FREECADCMD=$ROOT/bin/freecadcmd
export XDG_CONFIG_HOME="$corpus/.xdg_config" XDG_DATA_HOME="$corpus/.xdg_data" QT_QPA_PLATFORM=offscreen
(cd "$CODE" && nice -n 10 python scripts/validate_known.py build "$corpus") > "$OUT/build_$name.log" 2>&1
rc=$?
echo "$(date -Is) build exit $rc" >> "$OUT/build_$name.log"
if [ "$rc" -ne 0 ] || [ ! -s "$corpus/list.txt" ]; then echo "nothing to run" >> "$OUT/build_$name.log"; exit 1; fi
"$(dirname "$0")/rerun.sh" "$name" "$jobs" "$corpus/list.txt" "$corpus" "$corpus/truths"
