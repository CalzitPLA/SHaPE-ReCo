#!/usr/bin/env bash
# The v5 dataset for the hierarchical TypeNet (5 parents, 24 subclasses), every solid built in FreeCAD and written with
# dataset_gen/src/sample_writer (voxels + .field.npz + dataset_metadata.jsonl). Three generators, disjoint seeds per split:
#   dataset_generator.py  full / sector / band of the 5 kinds, carved and planar-cut variants, negative residuals (negres)
#   ibeam_generator.py    box/ibeam (DIN profiles), their plates (box/full, source ibeam_plate) and envelopes
#   special_generator.py  hollow (5 parents), cylinder/tube, box/rhs, frame (5 parents)
#
#   bash scripts/make_dataset_v5.sh --small        # about 20 / 5 / 10 per subclass, with .field.npz -> .../v5/small
#   bash scripts/make_dataset_v5.sh --medium       # 120 / 25 / 25 per subclass, occupancy only (no field) -> .../v5/medium
#
# --medium is the smallest set a proper training can use, sized for the local RTX 4050: the small A/B found no gain
# for the exact-field input (0.39 vs 0.41 parent accuracy) at 5x the loading cost, so the field is skipped
# (the most expensive part of generation, 40-80 s per sample); voxels are the ground truth.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
case "${1:-}" in
  --small)  SIZE=small;  FIELD_MAIN="--with-field"; FIELD_OTHER="" ;;
  --medium) SIZE=medium; FIELD_MAIN="";             FIELD_OTHER="--no-field" ;;
  *) echo "usage: $0 --small | --medium" >&2; exit 2 ;;
esac
OUT=$ROOT/dataset_gen/local_dataset_v5/$SIZE
LOGS=$OUT/logs
mkdir -p "$LOGS"
FREECAD=${FREECAD:-${SHAPE_RECO_FREECADCMD:-$(command -v freecadcmd || echo /usr/bin/freecadcmd)}}   # dataset_gen/src/freecad_path.py
WORKERS=${WORKERS:-6}

# per split: base count (one subclass), seed
declare -A N=([train]=20 [val]=5 [test]=10)
[ "$SIZE" = medium ] && declare -A N=([train]=120 [val]=25 [test]=25)
declare -A NPROF=([train]=1 [val]=1 [test]=1)
[ "$SIZE" = medium ] && declare -A NPROF=([train]=6 [val]=5 [test]=3)
declare -A SEED=([train]=1000 [val]=2000 [test]=3000)

gen_split() {
  local split=$1 n=${N[$1]} seed=${SEED[$1]} dir=$OUT/$1
  mkdir -p "$dir"
  local pids=()
  # 1. whole kinds: box full; cylinder/cone full+sector; sphere/torus full+sector+band (hollow comes from the special
  #    generator, so --hollow-prob 0). Carves and planar cuts vary the full bodies; every carve also writes its negres piece.
  local k count seg shard=0
  for spec in "box 1 0.0" "cylinder 2 0.5" "cone 2 0.5" "sphere 3 0.67" "torus 3 0.67"; do
    read -r k count seg <<<"$spec"
    ( cd dataset_gen && python3 src/dataset_generator.py --mode type --topologies "$k" --kinds "$k" \
        --count-per-type $((count * n)) --macro-dir FCMacros --output-dir "$dir/shard$shard" --freecad "$FREECAD" \
        --seed $((seed + shard)) --seed-offset $((seed + shard)) --voxelize --voxel-scale-mode fixed --grid-mm 40 \
        $FIELD_MAIN --segment-prob "$seg" --carve-prob 0.4 --cut-prob 0.3 --hollow-prob 0 --emit-negative-residual \
        > "$LOGS/${split}_main_$k.log" 2>&1 ) &
    pids+=($!)
    shard=$((shard + 1))
  done
  # 2. I-beams: one beam per profile of the default 20 (train), fewer profiles for val/test; plates and envelopes.
  local profiles=default
  [ "$split" = val ] && profiles="IPE 200,HEA 300,HEB 400,HEM 300,IPE 450"
  [ "$split" = test ] && profiles="IPE 160,IPE 360,HEA 200,HEA 500,HEB 240,HEB 500,HEM 200,IPE 600,HEA 600,I 200"
  python3 dataset_gen/src/ibeam_generator.py --out "$dir" --shard 10 --n-per-profile ${NPROF[$split]} --profiles "$profiles" \
    --seed $((seed + 10)) --freecad "$FREECAD" --plate-fraction 0.5 --envelope-fraction 0.25 $FIELD_OTHER \
    > "$LOGS/${split}_ibeam.log" 2>&1 &
  pids+=($!)
  # 3. special subclasses: hollow x5, tube, rhs, frame x5
  python3 -m dataset_gen.src.special_generator --out "$dir" --shard 20 --n-per-subclass "$n" --seed $((seed + 20)) \
    --workers "$WORKERS" --freecad "$FREECAD" $FIELD_OTHER > "$LOGS/${split}_special.log" 2>&1 &
  pids+=($!)
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  echo "[$split] generators finished (failures: $fail) $(date +%T)"
}

t0=$(date +%s)
for split in train val test; do gen_split "$split"; done
echo "generation took $(( ($(date +%s) - t0) / 60 )) min"

# verification: manifest load, subclass counts, field checks
for split in train val test; do
  python3 - "$OUT/$split" <<'EOF'
import sys, collections
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, "dataset_gen")
from training.manifest import load_manifest
from training.subclass_schema import filter_records, labels_of, PARENTS, FLAT_SUBCLASSES
root = Path(sys.argv[1])
recs = load_manifest(root)
kept, skipped = filter_records(recs)
c = collections.Counter(FLAT_SUBCLASSES[labels_of(r)[1]] for r in kept)
need_field = "/small/" in str(root)
missing = [r for r in kept if not (r.stl_path.exists() and r.voxel_path and r.voxel_path.exists()
                                   and (not need_field or r.voxel_path.with_suffix(".field.npz").exists()))]
print(f"{root.name}: {len(kept)} records, skipped {dict(skipped)}, missing files {len(missing)}")
print("  " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
EOF
done
