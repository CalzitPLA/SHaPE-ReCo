#!/usr/bin/env bash
# shape atlas: describe a list or corpus on a server (niced, memory capped); usage, in ~/data/shape_reco/shape_atlas:
#   describe_run.sh <name> <jobs> <scripts/shape_atlas.py describe args...>   -> desc_<name>.jsonl, <name>.log
set -u
ROOT=/data/user/shape_reco
CODE=${SHAPE_ATLAS_CODE:-$ROOT/wt/shape_atlas_20261029}            # a git-archive checkout with a COMMIT file
OUT=$HOME/data/shape_reco/shape_atlas
name=$1; jobs=$2; shift 2
export PATH="$ROOT/bin:$ROOT/venv_tests/bin:$PATH"
export SHAPE_RECO_FREECADCMD=$ROOT/bin/freecadcmd
export SHAPE_RECO_SHAPE_CLASS_CACHE=$ROOT/cache/shape_class
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# a cap per process: on 2026-10-29 two ABC parts held 52 and 77 GB in freecadcmd until the 900 s timeout
ulimit -v $((${SHAPE_ATLAS_MEM_GB:-12} * 1024 * 1024))
cd $CODE
echo "start $(date -Is) host $(hostname) commit $(cat COMMIT)" >> $OUT/$name.log
nice -n 10 python scripts/shape_atlas.py describe "$@" --out $OUT/desc_$name.jsonl --jobs $jobs >> $OUT/$name.log 2>&1
rc=$?
echo "end $(date -Is) exit $rc" >> $OUT/$name.log
