#!/usr/bin/env bash
# shape atlas: a class-stratified list through the gate's best variant of 2026-10-16 (tree_lg = --mode tree after the
# legacy start, the exact one-solid gate on), with the gate's settings (gate_paths/gate_run_e.sh); server_a or server_b only
# (server_c/server_d run 3-4x slower in every stage, RUNNING 3.4).
#   rerun.sh <run name> <jobs> <stems list> <parts dir: step/<stem>.step, stl/<stem>.stl> [truths dir]
#   SHAPE_ATLAS_EXTRA: other run_part options instead of "--mode tree", e.g. "--subparts" (the ABC batch of 30 Sep)
#   SHAPE_ATLAS_MAX_LOAD: GNU parallel's load cap (default nproc; the ABC batch of 30 Sep ran 8 jobs under 75 %)
#   -> ~/data/shape_reco/shape_atlas/runs_<name>/<stem>/{record,rating,...}.json, launch_<name>.log
set -u
name=$1; jobs=$2; list=$3; parts=$4; truths=${5:-}
ROOT=/data/user/shape_reco
CODE=${SHAPE_ATLAS_CODE:-$ROOT/wt/shape_atlas_20261029b}            # a git-archive checkout with a COMMIT file
OUT=$HOME/data/shape_reco/shape_atlas
mkdir -p "$OUT/runs_$name"
cd "$CODE"
env TRUTHS="$truths" JOBS="$jobs" GPUS=0 TIMEOUT=3600 RECO_TIMEOUT=1800 COMPARE_TIMEOUT=600 BUDGET=900 POINTS=500000 \
    MAX_LOAD="${SHAPE_ATLAS_MAX_LOAD:-$(nproc)}" THREADS=2 EXTRA="${SHAPE_ATLAS_EXTRA---mode tree}" \
    bash server/server_a/run_parts.sh "$OUT/runs_$name" "$list" "$parts" > "$OUT/launch_$name.log" 2>&1
echo "$(date -Is) $name exit $?" >> "$OUT/launch_$name.log"
