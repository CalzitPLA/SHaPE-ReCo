"""reco_engine.timing: nested sections, per-step takes, the run summary, and nothing recorded without a recorder."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reco_engine.timing import Recorder, format_sections, format_summary, timed, timed_fn  # noqa: E402


@timed_fn("work")
def work(seconds):
    with timed("inner"):
        time.sleep(seconds)
    return seconds


def test_sections_nest_and_sum():
    with Recorder() as rec:
        with timed("step"):
            work(0.02)
            work(0.03)
        first = rec.take()
        work(0.01)
        second = rec.take()
    assert [s.path for s in first] == ["step/work/inner", "step/work", "step/work/inner", "step/work", "step"]
    assert [s.path for s in second] == ["work/inner", "work"]
    totals = {t.path: t for t in rec.summary()}
    assert totals["step/work"].calls == 2 and abs(totals["step/work"].wall - 0.05) < 0.03
    assert totals["step"].self_wall < totals["step"].wall - 0.04                    # the sub-sections are not "self"
    assert format_sections(first).startswith("step ") and "work" in format_sections(second, min_wall=0.0) and "(whole run)" in format_summary(rec.summary(), rec.elapsed())


def test_nothing_is_recorded_without_a_recorder():
    rec = Recorder()                                                              # created, never activated
    assert work(0.0) == 0.0
    assert rec.sections == []


def test_current_names_the_open_section():
    with Recorder() as rec:
        assert rec.current() is None
        with timed("step"):
            with timed("inner"):
                time.sleep(0.02)
                path, seconds = rec.current()
                assert path == "step/inner" and 0.015 < seconds < 0.5
            assert rec.current()[0] == "step"
        assert rec.current() is None
