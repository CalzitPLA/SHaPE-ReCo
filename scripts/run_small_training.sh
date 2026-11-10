#!/bin/bash
# One-shot small-scale generation + training run, to get real (if modest)
# checkpoints for all three CNNs in place quickly - not production quality,
# just enough for the reconstruction pipeline / FreeCAD nodes to have
# something real to run against.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
SANDBOX_ROOT="$(pwd)"

TOPOLOGIES="box cylinder sphere cone torus"

DATASET_ROOT="$SANDBOX_ROOT/dataset_gen/local_dataset"
CHECKPOINT_ROOT="$SANDBOX_ROOT/checkpoints"

FREECAD=${FREECAD:-${SHAPE_RECO_FREECADCMD:-$(command -v freecadcmd || echo /usr/bin/freecadcmd)}}   # dataset_gen/src/freecad_path.py

echo "=== [1/6] Generating TYPE dataset ==="
cd "$SANDBOX_ROOT/dataset_gen"
python3 -m src.dataset_generator --mode type --topologies $TOPOLOGIES \
  --count-per-type 40 --macro-dir ./FCMacros --output-dir "$DATASET_ROOT/type" \
  --freecad "$FREECAD" --voxelize --voxel-resolution 64 --resume

echo "=== [2/6] Generating MEASURE dataset ==="
python3 -m src.dataset_generator --mode measure --topologies $TOPOLOGIES \
  --measure-ratios-per-type 8 --count-per-measure 5 --macro-dir ./FCMacros --output-dir "$DATASET_ROOT/measure" \
  --freecad "$FREECAD" --voxelize --voxel-resolution 64 --resume

echo "=== [3/6] Generating ROTATION dataset ==="
python3 -m src.dataset_generator --mode rotation --topologies $TOPOLOGIES \
  --rotation-groups-per-type 8 --count-per-rotation 5 --macro-dir ./FCMacros --output-dir "$DATASET_ROOT/rotation" \
  --freecad "$FREECAD" --voxelize --voxel-resolution 64 --resume

cd "$SANDBOX_ROOT"

echo "=== [4/6] Training Type CNN ==="
python3 -m training.train_type_cnn \
  --dataset-root "$DATASET_ROOT/type" --checkpoint-dir "$CHECKPOINT_ROOT/type_cnn" \
  --resolution 64 --batch-size 16 --epochs 50 --patience 15 --val-fraction 0.2

echo "=== [5/6] Training Rotation CNN ==="
python3 -m training.train_rotation_cnn \
  --dataset-root "$DATASET_ROOT/rotation" --checkpoint-dir "$CHECKPOINT_ROOT/rotation_cnn" \
  --resolution 64 --batch-size 16 --epochs 50 --patience 15 --val-fraction 0.2

echo "=== [6/6] Training Dimension CNN ==="
python3 -m training.train_dimension_cnn \
  --dataset-root "$DATASET_ROOT/measure" --checkpoint-dir "$CHECKPOINT_ROOT/dimension_cnn" \
  --resolution 64 --batch-size 16 --epochs 50 --patience 15 --val-fraction 0.2

echo "=== DONE ==="
