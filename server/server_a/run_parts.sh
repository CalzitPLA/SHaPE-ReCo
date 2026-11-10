#!/usr/bin/env bash
# A list of parts through scripts/run_part.py on the institute server, one process per part, started by GNU parallel
# only while the machine has room (as server/server_a/run_tests.sh). Runs ON the server, detached by the caller:
#   bash server/server_a/run_parts.sh <run directory> <list of stems> [parts directory]
# The parts directory (default ~/data/shape_reco/parts/abc) has step/<stem>.step and, where the manifest made
# them, stl/<stem>.stl. A rerun into the same run directory repeats what failed or did not run (--resume-failed).
#   JOBS 8, MAX_LOAD 75%, TIMEOUT 2700 s per part (TERM: the record says "killed" and the stage), RECO_TIMEOUT 1800 s
#   for the loop alone (it ends with what it has and the model is still measured), COMPARE_TIMEOUT 600 s for the
#   sampled comparison, BUDGET 900 s feature phase, POINTS 500000 (2026-10-15: 2e6 points cost 25 of the gate's 36
#   timeouts; iou_se is recorded, so 5e5 is the default and 2e6 a choice), GPUS 2. No memory guard: GNU parallel's
#   --memfree killed the youngest job silently (28 of the A/B bench's 34 failures) - size JOBS by the machine instead
#   (MIN_FREE sets it back, e.g. MIN_FREE=24G).
#   EXTRA: further options of scripts/run_part.py, e.g. EXTRA=--subparts
#   TRUTHS: a directory of <stem>.json known CSG trees (reco_engine/truth.py): each run is rated (--truth, rating.json)
#   CPUSET: cores for the parts (taskset -c), e.g. CPUSET=8-13 on a shared machine; THREADS per part (default 4)
# A part without step/<stem>.step runs from its stl/<stem>.stl (the validation cycle's mesh-only truths).
# Output: <run>/<stem>/record.json ..., <run>/joblog.<list>.tsv, <run>/DONE.<list>
set -uo pipefail
ROOT=${ROOT:-/data/user/shape_reco}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${1:?run directory}
LIST=${2:?list of stems}
export PARTS=${3:-$HOME/data/shape_reco/parts/abc}
TAG=$(basename "$LIST" .txt)
mkdir -p "$RUN/tmp" "$RUN/cache"
cd "$HERE"
{ echo "host $(hostname), $(nproc) cores, list $TAG ($(wc -l < "$LIST") parts), started $(date -Is)"; uptime; free -g | sed -n 2p;
  echo "EXTRA '${EXTRA:-}'"; echo "JOBS ${JOBS:-8} MAX_LOAD ${MAX_LOAD:-75%} MIN_FREE ${MIN_FREE:-none} TIMEOUT ${TIMEOUT:-2700} RECO_TIMEOUT ${RECO_TIMEOUT:-1800} COMPARE_TIMEOUT ${COMPARE_TIMEOUT:-600} BUDGET ${BUDGET:-900} POINTS ${POINTS:-500000} GPUS ${GPUS:-2}";
  echo "commit $(cat "$HERE/.commit" 2>/dev/null || git -C "$HERE" rev-parse HEAD 2>/dev/null)"; } > "$RUN/machine.$TAG.txt"
rm -f "$RUN/DONE.$TAG"
export PATH="$ROOT/bin:$ROOT/venv_tests/bin:$PATH"
export SHAPE_RECO_FREECADCMD=${SHAPE_RECO_FREECADCMD:-$ROOT/bin/freecadcmd}   # dataset_gen/src/freecad_path.py
export PARALLEL_HOME="$ROOT/tools/parallel_home"
mkdir -p "$PARALLEL_HOME" && touch "$PARALLEL_HOME/will-cite"
export TMPDIR="$RUN/tmp" XDG_CACHE_HOME="$RUN/cache" MPLCONFIGDIR="$RUN/cache/matplotlib" QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg
export SHAPE_RECO_WALL_CACHE="$RUN/cache/wall_thickness"        # 2026-09-29: the home directory's quota was full
export SHAPE_RECO_BREP_CACHE="$RUN/cache/brep" SHAPE_RECO_CACHE="$RUN/cache/shape_reco"   # their defaults are below ~
export XDG_CONFIG_HOME="$RUN/cache/xdg_config" XDG_DATA_HOME="$RUN/cache/xdg_data"        # FreeCAD 1.0 writes there
T=${THREADS:-4}                                                      # threads per part: JOBS x THREADS <= the cores
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T
export RUN BUDGET=${BUDGET:-900} POINTS=${POINTS:-500000} GPUS=${GPUS:-2} EXTRA=${EXTRA:-} TRUTHS=${TRUTHS:-} CPUSET=${CPUSET:-}
export RECO_TIMEOUT=${RECO_TIMEOUT:-1800} COMPARE_TIMEOUT=${COMPARE_TIMEOUT:-600}

one() {   # stem, parallel's job slot
  stem=$1; slot=$2
  if [ "$GPUS" -gt 0 ]; then export CUDA_VISIBLE_DEVICES=$(( (slot - 1) % GPUS )); else export CUDA_VISIBLE_DEVICES=""; fi
  mkdir -p "$RUN/$stem"
  input="$PARTS/step/$stem.step"; [ -f "$input" ] || input="$PARTS/stl/$stem.stl"
  truth=(); [ -n "$TRUTHS" ] && [ -f "$TRUTHS/$stem.json" ] && truth=(--truth "$TRUTHS/$stem.json")
  pin=(); [ -n "$CPUSET" ] && pin=(taskset -c "$CPUSET")
  "${pin[@]}" python scripts/run_part.py "$input" --stl "$PARTS/stl/$stem.stl" --out "$RUN/$stem" \
      --feature-time-budget "$BUDGET" --points "$POINTS" --reconstruction-timeout "$RECO_TIMEOUT" \
      --compare-timeout "$COMPARE_TIMEOUT" "${truth[@]}" $EXTRA > "$RUN/$stem/run.log" 2>&1
}
export -f one

memfree=(); [ -n "${MIN_FREE:-}" ] && memfree=(--memfree "$MIN_FREE")
"$ROOT/tools/parallel" --jobs "${JOBS:-8}" --load "${MAX_LOAD:-75%}" "${memfree[@]}" --nice 10 \
    --timeout "${TIMEOUT:-2700}" --joblog "$RUN/joblog.$TAG.tsv" --resume-failed --delay 3 one {} {%} :::: "$LIST"
echo "exit $? finished $(date -Is)" > "$RUN/DONE.$TAG"
