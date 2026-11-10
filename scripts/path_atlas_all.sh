#!/usr/bin/env bash
# The path atlas over every benchmark where reconstruction variants were compared on the same parts (2026-10-11, the
# user: "see the benchmarks of the reconstruction variations"), from $SHAPE_RECO_RESULTS (~/shape_reco_results):
#   gate      validation/gate_paths/runs_<variant>/<part>   60 known trees x 7 path variants (2026-10-10), rated
#   liner     liner_compare/<variant>                        the liner, 9 variants (2026-10-10)
#   examples  cubes_examples/<part>/<variant>                12 example parts, bare (fused_k4) vs cubes (cubes_k4)
#   ab        aifree_bench/ab_20261002/{orig,aifree}/<part>  99 parts, bare vs the AI-free type stage (2026-10-02)
#   abc       abc/abc_known{,_subparts}/<part>               332 known ABC parts, bare vs sub-parts (2026-09-30)
# -> docs/data/path_atlas_all.jsonl, its compare table and map. The GUI's Benchmark tab reads the jsonl.
set -eu
cd "$(dirname "$0")/.."
R=${SHAPE_RECO_RESULTS:-$HOME/shape_reco_results}
python3 scripts/path_atlas.py build \
  "gate=$R/validation/gate_paths/runs_*/*@variant/part" \
  "liner=$R/liner_compare/*" \
  "examples=$R/cubes_examples/*/*" \
  "ab=$R/aifree_bench/ab_20261002/*/*@variant/part" \
  "abc=$R/abc/abc_known*/*@variant/part" \
  --truths "$R/validation/gate_paths/truths" \
  --rename orig=bare aifree=analytic abc_known=bare abc_known_subparts=sub-parts fused_k4=bare cubes_k4=cubes \
  --out docs/data/path_atlas_all.jsonl
python3 scripts/path_atlas.py compare docs/data/path_atlas_all.jsonl --md docs/data/path_atlas_all_compare.md | tail -8
python3 scripts/path_atlas.py map docs/data/path_atlas_all.jsonl --out docs/reports/path_atlas/path_atlas_all.png --domains gate,liner,examples,ab
