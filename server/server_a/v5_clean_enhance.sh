#!/usr/bin/env bash
# Data enhancement of the clean v5 set (2026-10-10), on server_a: what the per-sample error analysis of the k4 type nets
# asked for (scripts/type_predictions.py; docs/reports/server_runs_2026-10-10/).
#   1. hygiene view  $OUT/hygiene_v1: every split minus the cones under 2 voxels of taper (scripts/dataset_hygiene.py)
#   2. increment t1  $OUT/train/t1: the main generator for cylinder and cone only, T1_N (1000) per subclass, the flags
#                    of server/run_plan.sh gen, seeds 400000 + shard (no other increment uses them), field on the CPU
#   3. trainings, dirlog_inside, the settings of train_one.sh, two seeds each:
#        k4  s1       the unchanged set (the second seed of the baseline; s0 is the curve's k4)
#        k4h s0, s1   the hygiene view of inc1..inc4
#        k5t s0, s1   the hygiene view of inc1..inc4 + t1
#      every model card scores the hygiene test view AND the unchanged test set.
# Resumable: a finished unit has a stamp in $OUT/.done or a model card. Start detached:
#   nohup setsid bash $OUT/analysis/v5_clean_enhance.sh > $OUT/logs/enhance_chain.log 2>&1 &
source $HOME/data/shape_reco/v5_clean/env.sh
cd $CODE
export PYTHONPATH=$CODE
A=$OUT/analysis; H=$OUT/hygiene_v1
say() { echo "$(date +%F_%T) $*"; }

hygiene() { nice -n 10 python3 $A/dataset_hygiene.py --out $H --base $OUT --src "$@"; }

gen_t1() {
  # BUG found 2026-10-10 12:30 (the first run of this chain): the `wait` at the end waited for EVERY background job of
  # the shell, the two training subshells among them, and those wait for the marker this function's caller writes -
  # a deadlock after the three trainings that needed no marker (k4 s1, k4h s0, k4h s1). Unblocked by hand (the t1
  # hygiene view and the marker written from a shell); fixed here: wait for the generator PIDs only.
  local N=${T1_N:-1000} SH=${T1_SHARDS:-6} seed=400000 dir=$OUT/train/t1 ki=1 k s pids=""
  mkdir -p "$dir"
  for k in cylinder cone; do
    local total=$((2 * N)) per=$(( (2 * N + SH - 1) / SH ))
    for ((s = 0; s < SH; s++)); do
      local c=$(( total - s * per < per ? total - s * per : per )); [ "$c" -le 0 ] && continue
      local sh=$(( ki * 100 + s )) tag="train_t1_main_${k}_$s"
      [ -f "$OUT/.done/$tag" ] && continue
      ( export CUDA_VISIBLE_DEVICES="" SHAPE_RECO_FIELD_BACKEND=igl SHAPE_RECO_FREECAD_WORKER=1 SHAPE_RECO_VOXELIZER=parity OMP_NUM_THREADS=2
        cd dataset_gen && nice -n 10 python3 src/dataset_generator.py --mode type --topologies "$k" --kinds "$k" --count-per-type "$c" \
          --macro-dir FCMacros --output-dir "$dir/shard$sh" --freecad "$FREECAD" --seed $((seed + sh)) --seed-offset $((seed + sh)) \
          --voxelize --voxel-scale-mode fixed --grid-mm 40 --with-field --segment-prob 0.5 --carve-prob 0.4 --cut-prob 0.3 \
          --hollow-prob 0 --emit-negative-residual --require-manifold --min-voxels 200 $NOISE > "$OUT/logs/$tag.log" 2>&1 \
          && touch "$OUT/.done/$tag" ) &
      pids="$pids $!"
    done
    ki=$((ki + 1))
  done
  [ -n "$pids" ] && wait $pids
  return 0
}

train() {   # train <name> <gpu> <seed> <train roots ...> -- <val root> -- <test roots ...>
  local name=$1 gpu=$2 seed=$3; shift 3
  local d=$CKPT/type_dirlog_inside_${name}_s${seed}_clean R="" first=1 val tests
  while [ "$1" != "--" ]; do [ $first = 1 ] && R="--dataset-root $1" || R="$R --extra-root $1"; first=0; shift; done; shift
  val=$1; shift 2; tests="$*"
  [ -f "$d/type_cnn/model_card.json" ] && { say "skip $d (done)"; return 0; }
  mkdir -p "$d"
  say "start $d on cuda:$gpu"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 nice -n 10 python3 -m training.train_type_cnn --hierarchical --input dirlog_inside \
    --gpu-encode --amp $R --val-roots "$val" --test-roots $tests --epochs "${EPOCHS:-60}" --seed "$seed" --num-workers "${LOADERS:-4}" \
    --oversample cut=1:2 --checkpoint-dir "$d/type_cnn" > "$d.log" 2>&1
  say "end $d rc=$?"
}

RAW="$OUT/train/inc1 $OUT/train/inc2 $OUT/train/inc3 $OUT/train/inc4"
HYG="$H/train/inc1 $H/train/inc2 $H/train/inc3 $H/train/inc4"
wait_t1() { while [ ! -f "$OUT/.done/ENHANCE_T1" ]; do sleep 60; done; }

say "hygiene view of the existing set"
hygiene $OUT/val/val $OUT/test/test $RAW || { say "hygiene failed"; exit 1; }

( train k4h 0 0 $HYG -- $H/val/val -- $H/test/test $OUT/test/test
  train k4h 0 1 $HYG -- $H/val/val -- $H/test/test $OUT/test/test
  wait_t1
  train k5t 0 1 $HYG $H/train/t1 -- $H/val/val -- $H/test/test $OUT/test/test ) &
( train k4 1 1 $RAW -- $OUT/val/val -- $H/test/test $OUT/test/test
  wait_t1
  train k5t 1 0 $HYG $H/train/t1 -- $H/val/val -- $H/test/test $OUT/test/test ) &

say "increment t1: generation"
gen_t1; gen_t1                                   # the second pass redoes only the shards without a stamp
say "increment t1: $(find $OUT/train/t1 -name '*.bin' | wc -l) voxel files; hygiene view"
hygiene $OUT/train/t1 && touch "$OUT/.done/ENHANCE_T1"
say "ENHANCE_T1"
wait
touch "$OUT/.done/ENHANCE_V1"
say "ENHANCE-CHAIN-DONE"
