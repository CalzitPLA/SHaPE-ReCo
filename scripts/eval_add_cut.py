#!/usr/bin/env python3
"""
Run the residual add-cut loop (reco_engine/residual_loop.py) on a list of parts and print, per part, the program (op, kind,
dimensions, which mode answered each stage), the residual left in voxels, the FreeCAD validity and the two-way chamfer of
the FreeCAD solid to the input (reco_field.export.export_check).

    python3 scripts/eval_add_cut.py                                   # dataset_gen/examples/bs_*.stl and cy_*.stl
    python3 scripts/eval_add_cut.py dataset_gen/examples/bs_box.stl dataset_gen/examples/cy_box_bore.stl --json out.json
    python3 scripts/eval_add_cut.py --no-freecad                      # the loop only (no build, no chamfer)

Results are cached per part (sha1 of the file + the settings) in --cache-dir, so a re-run of a long list only pays for
the parts that changed; --no-cache recomputes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reco_engine.residual_loop import ResidualConfig, reconstruct  # noqa: E402
from reco_engine.result_schema import ReconstructionResult  # noqa: E402


def _key(path: Path, cfg: ResidualConfig, freecad: bool) -> str:
    h = hashlib.sha1(path.read_bytes())
    h.update(json.dumps(dict(res=cfg.resolution, pad=cfg.pad_fraction, k=cfg.top_k, fc=freecad, v=1)).encode())
    reg = ROOT / "models" / "registry.json"
    if reg.is_file():
        h.update(reg.read_bytes())
    return h.hexdigest()[:16]


def _fmt_dims(d: dict) -> str:
    return ", ".join(f"{k}={v:.2f}" for k, v in d.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="*", type=Path)
    ap.add_argument("--resolution", type=int, default=48)
    ap.add_argument("--no-freecad", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("/tmp/eval_add_cut"), help="FreeCAD output directory")
    ap.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "shape_reco" / "add_cut")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--json", type=Path, default=None, help="write all results here")
    args = ap.parse_args()
    parts = args.parts or sorted((ROOT / "dataset_gen" / "examples").glob("bs_*.stl")) + \
        sorted((ROOT / "dataset_gen" / "examples").glob("cy_*.stl"))

    rows = []
    for part in parts:
        cfg = ResidualConfig(resolution=args.resolution, freecad=not args.no_freecad, out_dir=args.out / part.stem)
        cache = args.cache_dir / f"{part.stem}_{_key(part, cfg, not args.no_freecad)}.json"
        t0 = time.time()
        if cache.is_file() and not args.no_cache:
            d = json.loads(cache.read_text())
            res, seconds = ReconstructionResult.from_dict(d["result"]), d["seconds"]
        else:
            res = reconstruct(part, cfg)
            seconds = time.time() - t0
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(dict(result=res.to_dict(), seconds=seconds)))
        valid = res.operations[-1].freecad_valid if res.operations else None
        print(f"\n== {part.name}: {len(res.operations)} ops, residual +{res.residual_voxels_pos} / -{res.residual_voxels_neg} "
              f"voxels, FreeCAD valid {valid}, chamfer {res.chamfer_mm if res.chamfer_mm is None else f'{res.chamfer_mm:.4f} mm'}, "
              f"{seconds:.0f} s")
        for o in res.operations:
            p = o.primitive
            modes = "/".join(s.mode for s in p.sources)
            print(f"   {o.op:3s} {p.kind:8s} {_fmt_dims(p.dims_mm)}  centre=({', '.join(f'{c:.2f}' for c in p.center_mm)})"
                  f"  gain {o.net_gain:.0f}  [{modes}]")
        rows.append(dict(part=part.name, ops=[(o.op, o.primitive.kind) for o in res.operations],
                         residual_pos=res.residual_voxels_pos, residual_neg=res.residual_voxels_neg,
                         chamfer_mm=res.chamfer_mm, valid=valid, seconds=seconds, result=res.to_dict()))
    ch = [r["chamfer_mm"] for r in rows if r["chamfer_mm"] is not None]
    if ch:
        print(f"\n{len(rows)} parts, mean chamfer {sum(ch) / len(ch):.4f} mm, valid {sum(bool(r['valid']) for r in rows)}/{len(rows)}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
