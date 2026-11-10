"""dataset_gen/src/din_profiles.py: the self-check (area from dims vs tabulated), the frame identification, three_boxes."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import din_profiles as dp  # noqa: E402


def test_series_coverage():
    counts = {s: len(dp.select(s)) for s in ("IPE", "HEA", "HEB", "HEM", "I")}
    assert counts["IPE"] >= 18 and counts["HEA"] >= 19 and counts["HEB"] >= 19
    assert counts["HEM"] >= 3 and counts["I"] >= 3
    assert min(p.h for p in dp.select("IPE")) == 80 and max(p.h for p in dp.select("IPE")) == 600


@pytest.mark.parametrize("name", [n for n, p in dp.PROFILES.items() if p.flange_slope == 0])
def test_parallel_flange_area_within_1_percent(name):
    assert abs(dp.area_deviation(dp.PROFILES[name])) <= 0.01


def test_nothing_dropped_silently_and_tapered_flagged():
    assert not dp.DROPPED                                    # a failing row would be listed here, not kept
    for name, p in dp.PROFILES.items():
        status = dp.self_check()[name][1]
        if p.series == "I":
            assert p.flange_slope == pytest.approx(0.14) and status == "flagged" and name in dp.FLAGS
        else:
            assert p.flange_slope == 0 and status == "ok"


def test_naming_and_standards():
    hea500 = dp.PROFILES["HEA 500"]
    assert (hea500.h, hea500.b) == (490, 300)                # the HE-A 500 is 490 deep
    assert "DIN 1025-3" in hea500.standard and "EN 10365" in hea500.standard
    assert "DIN 1025-5" in dp.PROFILES["IPE 300"].standard


def test_frame_identification():
    p, d = dp.nearest(490, 300, 12, 23)
    assert p.name == "HEA 500" and d == 0
    assert dp.PROFILES["HEA 500"].area_cm2 == pytest.approx(197.6, rel=0.002)
    # 350 x 300, tw 12, tf 17.5: the HE-A 360 envelope, non-standard web (10 -> 12). nearest() scores the LARGEST
    # relative error, so the 20 % web error makes HEA 400 (10 % in h) "nearer" - documented behaviour.
    p, d = dp.nearest(350, 300, 12, 17.5)
    assert p.name == "HEA 400" and d == pytest.approx(40 / 390)
    hea360 = dp.PROFILES["HEA 360"]
    assert (hea360.h, hea360.b, hea360.tf, hea360.tw) == (350, 300, 17.5, 10)
    frame = dp.IProfile("frame", "HEA", "", 350, 300, 12, 17.5, 27, 149.1)
    assert dp.area_from_dims(frame) == pytest.approx(149.1, rel=0.001)
    assert dp.nearest(350, 300, 10, 17.5, series="HEA")[0].name == "HEA 360"


def test_select_filters():
    ps = dp.select(series=["HEA", "HEB"], h_min=300, h_max=400)
    assert ps and all(p.series in ("HEA", "HEB") and 300 <= p.h <= 400 for p in ps)
    assert dp.select("HEM", h_min=10_000) == []


def test_three_boxes():
    p = dp.PROFILES["HEA 500"]
    boxes = dp.three_boxes(p, 1000.0)
    assert len(boxes) == 3
    (ct, ht), (cb, hb), (cw, hw) = boxes
    assert ct[1] > 0 > cb[1] and np.allclose(cw, 0)
    assert np.allclose(ht, [150, 11.5, 500]) and np.allclose(hw, [6, 222, 500])
    # plates touch: top flange's lower face = web's upper face
    assert ct[1] - ht[1] == pytest.approx(cw[1] + hw[1])
    assert ct[1] + ht[1] == pytest.approx(p.h / 2)
    # plate cross-sections = area_from_dims minus the four fillets
    plates = sum(4 * h[0] * h[1] for _, h in boxes) / 100
    assert plates == pytest.approx(dp.area_from_dims(p) - (4 - math.pi) * p.r ** 2 / 100)


def test_same_rolls_same_inner_depth():
    """HE-A, HE-B and HE-M of one size are rolled with the same web: the clear depth between the flanges, h - 2 tf, is
    equal across the three series (an independent check of h and tf, not used to write the table)."""
    for n in (100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 400, 450, 500, 550, 600):
        depths = {s: dp.PROFILES[f"{s} {n}"].h - 2 * dp.PROFILES[f"{s} {n}"].tf
                  for s in ("HEA", "HEB", "HEM") if f"{s} {n}" in dp.PROFILES}
        assert len(set(depths.values())) == 1, (n, depths)
