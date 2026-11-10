"""dataset_gen/src/freecad_path.py: which freecadcmd the project runs - $SHAPE_RECO_FREECADCMD, then `freecadcmd` on PATH,
then /usr/bin/freecadcmd - and that the defaults of the generator, the GUI settings and the I-beam generator follow it
(T-04, 2026-10-01: the institute servers have no /usr/bin/freecadcmd; server_b's is FreeCAD 0.20)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import freecad_path as fp  # noqa: E402


def _exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def no_fallback(monkeypatch):
    """No /usr/bin/freecadcmd, whatever this machine has."""
    real = fp.os.path.isfile
    monkeypatch.setattr(fp.os.path, "isfile", lambda p: False if p == fp.FALLBACK else real(p))


def test_the_variable_wins_over_path(tmp_path, monkeypatch, no_fallback):
    chosen = _exe(tmp_path / "appimage" / "freecadcmd")
    _exe(tmp_path / "bin" / "freecadcmd")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setenv(fp.ENV, str(chosen))
    assert fp.find_freecadcmd() == str(chosen)
    monkeypatch.setenv(fp.ENV, "freecadcmd")                      # a command name is looked up on PATH
    assert fp.find_freecadcmd() == str(tmp_path / "bin" / "freecadcmd")


def test_without_the_variable_path_then_the_fallback(tmp_path, monkeypatch, no_fallback):
    monkeypatch.delenv(fp.ENV, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    assert fp.find_freecadcmd() is None
    assert fp.freecadcmd_or_default() == fp.FALLBACK                # a default names what was looked for
    _exe(tmp_path / "bin" / "freecadcmd")
    assert fp.find_freecadcmd() == str(tmp_path / "bin" / "freecadcmd")


def test_a_variable_naming_nothing_takes_no_other_freecad(tmp_path, monkeypatch, no_fallback):
    _exe(tmp_path / "bin" / "freecadcmd")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setenv(fp.ENV, str(tmp_path / "missing" / "freecadcmd"))
    with pytest.warns(UserWarning, match=fp.ENV):
        assert fp.find_freecadcmd() is None
    with pytest.warns(UserWarning):
        assert fp.freecadcmd_or_default() == str(tmp_path / "missing" / "freecadcmd")


def test_the_defaults_follow_the_rule(tmp_path, monkeypatch):
    chosen = _exe(tmp_path / "fc" / "freecadcmd")
    monkeypatch.setenv(fp.ENV, str(chosen))
    from reco_gui.dataset_gen_setup import DatasetGenSettings, build_args
    from reco_gui.setup import RunSettings
    assert DatasetGenSettings().freecad == str(chosen)
    assert RunSettings(input_stl=tmp_path / "x.stl").freecad == str(chosen)
    cmd = build_args(DatasetGenSettings(), tmp_path / "out", seed=0)
    assert cmd[cmd.index("--freecad") + 1] == str(chosen)
    from src import ibeam_generator
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        raise RuntimeError("stop here")
    monkeypatch.setattr(ibeam_generator.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="stop here"):
        ibeam_generator.run_freecad([dict(id=0)], workdir=tmp_path)
    assert seen["cmd"][0] == str(chosen)
