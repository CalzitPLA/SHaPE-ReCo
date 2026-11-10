#!/usr/bin/env bash
# The whole test suite on the institute server, one pytest process per test file, started by GNU parallel ONLY WHILE
# THE MACHINE HAS ROOM: a new file starts when the load is below $MAX_LOAD and $MIN_FREE of memory are free, at most
# $JOBS at once, every process niced. Runs ON the server, detached by the caller:
#   bash server/server_a/run_tests.sh [out directory] [pytest file ...]
# Rerun with the same out directory: only the files that failed or did not run are run again (--resume-failed).
#   JOBS      files at once, at most                      (default 8)
#   MAX_LOAD  no new file while the 1-minute load is above (default 75% = 24 of server_a's 32 cores; others use it too)
#   MIN_FREE  no new file below this much free memory      (default 16G)
#   TIMEOUT   s per file                                  (default 7200)
#   GPU       1: the tests may use CUDA (SHAPE_RECO_TEST_GPU) (default 0: CPU only, as on the laptop)
#   SHAPE_RECO_FREECADCMD  the freecadcmd the tests use        (default $ROOT/bin/freecadcmd)
#   TEST_TMP  local directory for the processes' TMPDIR       (default /tmp; keep it short, see TMPDIR below)
# Pinned to some cores (taskset): give MAX_LOAD as a number - GNU parallel takes the % of the cores it may use, so
# 75% of 4 pinned cores is a load of 3, which other users' jobs alone can exceed: no file ever starts (2026-10-01).
# Output: <out>/joblog.tsv (GNU parallel's log: start, seconds, exit code per file), <out>/xml/<file>.xml (JUnit),
# <out>/log/<file>.log, <out>/DONE when all is through. server/server_a/summarise_tests.py makes the table.
set -uo pipefail
ROOT=${ROOT:-/data/user/shape_reco}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-$HOME/data/shape_reco/tests/$(date +%Y%m%d-%H%M%S)}
shift || true
mkdir -p "$OUT/xml" "$OUT/log" "$OUT/tmp"
cd "$HERE"
if [ $# -gt 0 ]; then printf '%s\n' "$@" > "$OUT/files.txt"; else ls tests/test_*.py > "$OUT/files.txt"; fi
git -C "$HERE" rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || cp "$HERE/.commit" "$OUT/commit.txt" 2>/dev/null || true
{ echo "host $(hostname), $(nproc) cores, started $(date -Is)"; uptime; free -g | sed -n 2p; } > "$OUT/machine.txt"
rm -f "$OUT/DONE"

export PATH="$ROOT/bin:$ROOT/venv_tests/bin:$PATH"
# FreeCAD: the extracted 1.0 AppImage, named by SHAPE_RECO_FREECADCMD (dataset_gen/src/freecad_path.py: the variable, then
# PATH, then /usr/bin/freecadcmd - which server_a lacks and which on server_b is the system's FreeCAD 0.20; 2026-10-01, T-04)
export SHAPE_RECO_FREECADCMD=${SHAPE_RECO_FREECADCMD:-$ROOT/bin/freecadcmd}
export PARALLEL_HOME="$ROOT/tools/parallel_home"          # 2026-09-29: GNU parallel could not write below the NFS home
mkdir -p "$PARALLEL_HOME" && touch "$PARALLEL_HOME/will-cite"
export MPLCONFIGDIR="$OUT/cache/matplotlib"
# TMPDIR short and local, pytest's directories below $OUT/tmp (--basetemp). Under $OUT - a long NFS path - the AF_UNIX
# sockets of multiprocessing (pymp-*/listener-*) pass the 108-byte limit: a DataLoader worker's batch is dropped with
# "OSError: AF_UNIX path too long" and the test waits for it for ever at 0 % CPU (T-06, 2026-10-01: test_gpu_encode).
mkdir -p "${TEST_TMP:-/tmp}"
export TMPDIR=$(mktemp -d "${TEST_TMP:-/tmp}/srt.XXXXXX")
echo "TMPDIR $TMPDIR" >> "$OUT/machine.txt"
export SHAPE_RECO_TEST_GPU=${GPU:-0} SHAPE_RECO_TEST_THREADS=${THREADS:-4}
# SHAPE_RECO_CACHE stays unset: set, it switches the prediction cache on, and the tests with stub models fail in it
# (2026-09-29: 22 failures of test_loop_guards, test_octree_search, test_void_fit were this runner's)
export XDG_CACHE_HOME="$OUT/cache" QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg
export OUT

one() {   # one test file: its JUnit file, its log, pytest's exit code
  f=$1; n=$(basename "$f" .py)
  python -m pytest "$f" -q -p no:cacheprovider --basetemp="$OUT/tmp/$n" --junitxml="$OUT/xml/$n.xml" -o junit_family=xunit2 \
      > "$OUT/log/$n.log" 2>&1
  rc=$?
  [ $rc -eq 5 ] && rc=0            # no test collected: not a failure of the suite
  exit $rc
}
export -f one

"$ROOT/tools/parallel" --jobs "${JOBS:-8}" --load "${MAX_LOAD:-75%}" --memfree "${MIN_FREE:-16G}" --nice 10 \
    --timeout "${TIMEOUT:-7200}" --joblog "$OUT/joblog.tsv" --resume-failed --delay 2 one {} :::: "$OUT/files.txt"
rc=$?
rm -rf "$TMPDIR"                    # the processes' own temporary files; pytest's stay below $OUT/tmp
echo "exit $rc finished $(date -Is)" > "$OUT/DONE"
