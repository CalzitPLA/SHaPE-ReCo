"""reco_engine/event_time.py: the placement of any part in aegir's guns, and the navigation per event from the
difference quotient of two event counts minus the empty baseline (no aegir run: the walls are given)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

from reco_engine import event_time as et  # noqa: E402


def test_placement_beam_along_the_thinnest_extent_and_the_tightest_covering_gun():
    p = et.placement([-482.6, -104.8, 0.0], [482.6, 104.8, 12.7])           # an ABC plate, 965 x 210 x 12.7 mm
    assert (p["beam_axis"], p["gun"], p["covered"]) == ("z", "liner", True)
    assert p["origin"] == pytest.approx([0.0, 0.0, 6.35])
    assert et.cone_radius(p["gun"], -p["half_length"]) >= p["cross_radius"]
    small = et.placement([0, 0, 0], [20, 20, 30])                              # a 20 x 20 x 30 block: thinnest is x
    assert small["beam_axis"] == "x" and small["gun"] == "convention_test" and small["covered"]
    huge = et.placement([-3350, -2490, -500], [3350, 2490, 500])               # no gun covers it: the widest, said so
    assert huge["gun"] == "frame" and not huge["covered"]


def test_navigation_is_the_difference_quotient_minus_the_baseline_median_over_measured_rounds():
    lo, hi = et.EVENTS
    def walls(per_us, start, rounds=(0, 1, 2, 3)):                            # wall = start-up + events x time per event
        return {(r, n): start + n * per_us[r] * 1e-6 for r in rounds for n in (lo, hi)}
    prog = walls({0: 99.0, 1: 14.0, 2: 13.8, 3: 14.2}, 1.0)                    # round 0 (warm-up) is not counted
    base = walls({0: 99.0, 1: 13.0, 2: 13.1, 3: 13.0}, 0.9)
    n = et.navigation(prog, base)
    assert n["rounds_counted"] == 3 and n["baseline"]
    assert n["navigation_us"] == pytest.approx(1.0) and n["min_us"] == pytest.approx(0.7) and n["max_us"] == pytest.approx(1.2)
    assert n["per_event_us"] == pytest.approx(14.0) and n["baseline_us"] == pytest.approx(13.0)
    raw = et.navigation(prog, None)                                            # no baseline: the program's own time
    assert raw["navigation_us"] == pytest.approx(14.0) and raw["baseline_us"] is None
    assert et.navigation({}, None) == dict(rounds_counted=0)
