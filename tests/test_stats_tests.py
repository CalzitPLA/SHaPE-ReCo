"""reco_engine.stats_tests: the statistics of the reconstruction protocol, against reference values."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reco_engine import stats_tests as st  # noqa: E402


def test_mcnemar_exact_matches_the_binomial_reference():
    # 12 breaks, 28 rescues: two-sided 0.01659 (binomial, n = 40), one-sided half of it
    assert st.mcnemar_exact(12, 28) == pytest.approx(0.016589, abs=1e-5)
    assert st.mcnemar_exact(12, 28, "greater") == pytest.approx(0.0082945, abs=1e-6)
    assert st.mcnemar_exact(28, 12, "greater") > 0.99
    assert st.mcnemar_exact(2, 4) == pytest.approx(0.6875)                  # the gate: tree_lg against bare
    assert st.mcnemar_exact(0, 0) == 1.0


def test_discordant_table_and_flip_rate():
    a = [True, True, False, False, True, False]
    b = [True, False, True, False, True, True]
    t = st.discordant(a, b)
    assert t == dict(both=2, breaks=1, rescues=2, neither=1, n=6)
    f = st.flip_rate(a, b)
    assert f["flips"] == 3 and f["rate"] == 0.5 and f["lo"] < 0.5 < f["hi"]
    with pytest.raises(ValueError):
        st.discordant([True], [True, False])


def test_wilson_reference_values():
    lo, hi = st.wilson(28, 60)                  # bare's good rate on the gate
    assert lo == pytest.approx(0.3456, abs=1e-3) and hi == pytest.approx(0.5917, abs=1e-3)
    assert st.wilson(0, 0) == (0.0, 1.0) and st.wilson(0, 10)[0] == 0.0 and st.wilson(10, 10)[1] == pytest.approx(1.0)


def test_holm_is_monotone_and_capped():
    q = st.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert q["a"] == pytest.approx(0.04) and q["c"] == pytest.approx(0.09) and q["b"] == pytest.approx(0.09)
    assert q["d"] == pytest.approx(0.5)
    assert st.holm({"x": 0.6, "y": 0.7}) == {"x": 1.0, "y": 1.0}


def test_sample_size_closed_form_and_the_noise_floor():
    assert st.mcnemar_sample_size(0.2, 0.1) == 155                           # Connor 1987, 154.6 rounded up
    assert st.mcnemar_sample_size(0.2, 0.1, two_sided=False) < 155
    with pytest.raises(ValueError):
        st.mcnemar_sample_size(0.05, 0.1)                                    # a change larger than the discordance
    assert st.net_change_floor(2 / 60, 60) == pytest.approx(1.96 * np.sqrt(2), rel=1e-3)


def test_paired_bootstrap_and_wilcoxon():
    a = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0] * 6, float)
    b = a.copy()
    b[np.nonzero(a == 0)[0][:6]] = 1.0                                         # six rescues
    d = st.paired_bootstrap_delta(a, b, n=2000)
    assert d["delta"] == pytest.approx(0.1) and d["lo"] > 0 and d["hi"] < 0.25
    assert st.noninferior(d["lo"], 0.03) and not st.noninferior(-0.05, 0.03)
    w = st.wilcoxon_paired([0.5, 0.6, 0.7, 0.8], [0.6, 0.7, 0.8, 0.9])
    assert w["n"] == 4 and w["median_diff"] == pytest.approx(0.1) and w["p"] < 0.2
    assert st.wilcoxon_paired([1.0, 2.0], [1.0, 2.0])["p"] == 1.0
    assert st.wilcoxon_paired([1.0, float("nan")], [2.0, 3.0])["n"] == 1
