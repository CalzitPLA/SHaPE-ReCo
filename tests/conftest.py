"""Suite-wide defaults (2026-09-24): the tests are cheap citizens on a shared laptop.

- CPU only unless SHAPE_RECO_TEST_GPU=1: several library paths default to cuda (closest_point_field.field_backend "auto",
  whose _knn_gpu reserves 30 % of the free GPU memory; cascade, model_registry, fused_predict, train_common), so a plain
  test run took 1.4 GB of GPU memory (test_add_cut_loop, 2026-09-24) next to a production training. CUDA reads
  CUDA_VISIBLE_DEVICES at initialisation, so setting it here works even if torch is already imported.
- Thread cap: OMP/MKL/OpenBLAS/numexpr and torch default to SHAPE_RECO_TEST_THREADS (4) instead of all 16 cores.
- Qt offscreen by default (the GUI tests never need the display).
- Markers: `slow` (measured >= 20 s per test or per file, list below; `-m "not slow"` is the quick loop) and `gpu`
  (exercises CUDA; meaningful only with SHAPE_RECO_TEST_GPU=1).
"""
import os

GPU = os.environ.get("SHAPE_RECO_TEST_GPU") == "1"
if not GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
THREADS = os.environ.get("SHAPE_RECO_TEST_THREADS", "4")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, THREADS)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

# measured 2026-09-24 (pytest --durations, other sessions' per-file sweeps; GPU available at the time, so on CPU these
# are if anything slower). Whole files: wall time of the file. Node ids: the test's own call/setup time.
SLOW_FILES = {
    "test_add_cut_loop.py",        # 614 s
    "test_ibeam.py",               # 422 s
    "test_field_channels.py",      # 127 s
    "test_field_flags.py",         # 107 s
    "test_bounding_box_mode.py",   # 40 s
    "test_freecad_nodes.py",       # 30 s
    "test_closest_point_field.py", # 23 s
}
SLOW_TESTS = {
    "test_inscale.py::test_noise_prob_mixes_noisy_and_clean_samples_in_every_class_and_kind",  # 111 s
    "test_reco_field_fit.py::test_full_bodies_get_no_segment",                                 # 80 s
    "test_reco_field_fit.py::test_example_parts_base_primitive_and_leftover",                  # 49 s
    "test_reco_field_fit.py::test_sweep_angle_u_is_recovered",                                 # 45 s each
    "test_reco_field_fit.py::test_a_cylinder_with_a_flat_is_not_a_sector",                     # 25 s
    "test_reco_field_fit.py::test_the_right_kind_wins",                                        # 23 s (sphere)
    "test_reco_field_fit.py::test_torus_is_recovered_with_pose_and_size",                      # 23 s
    "test_reco_gui_field.py::test_steps_come_in_the_documented_order",                         # 79 s (setup)
    "test_reco_gui_field.py::test_autorun_reconstructs_live_without_any_click",                # 32 s
    "test_reco_gui_field.py::test_report_and_timeline_rows",                                   # 29 s
    # measured 2026-09-27 on workstation (scripts/suite_by_file.py, 6 files at a time: upper bounds)
    "test_train_fused.py::test_step_is_finite_and_the_network_can_overfit_a_tiny_batch",       # 36 s (180 steps)
    "test_feature_phase.py::test_the_loop_without_the_phase_is_unchanged_and_with_it_ends_in_the_phase",  # 27 s
    "test_feature_phase.py::test_a_bore_below_the_voxel_pitch_is_found_by_the_phase",          # 25 s
    "test_mesh_clusters.py::test_mesh_clusters_off_by_default_does_not_change_peel_result",    # 26 s
    "test_octree_clusters.py::test_octree_off_by_default_does_not_change_peel_result",         # 26 s
    "test_special_generator.py::test_real_freecad_builds",                                     # 21 s
}
GPU_TESTS = {
    "test_closest_point_field.py::test_gpu_knn_gives_the_kdtree_candidate_sets",
    "test_gpu_encode.py",
}


def pytest_addoption(parser):
    """--run-slow: run the tests marked slow (skipped otherwise). The option used to come only from a pytest plugin of
    the laptop's system (the astropy family): in a clean environment the slow tests ran by default (found 2026-10-30,
    testing install.sh)."""
    try:
        parser.addoption("--run-slow", action="store_true", default=False, help="run the tests marked slow")
    except ValueError:          # the system plugin defines it already
        pass


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: measured >= 20 s (tests/conftest.py); deselect with -m 'not slow'")
    config.addinivalue_line("markers", "gpu: exercises CUDA; runs on the GPU only with SHAPE_RECO_TEST_GPU=1")
    try:
        import torch
        torch.set_num_threads(int(THREADS))
    except Exception:           # torch missing: the torch tests skip themselves
        pass


def _keys(item):
    fname = item.path.name
    base = item.nodeid.split("[", 1)[0]                     # parametrised ids share the base name
    return fname, f"{fname}::{base.rsplit('::', 1)[-1]}"


def pytest_collection_modifyitems(config, items):
    run_slow = bool(config.getoption("--run-slow", default=False))
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        fname, key = _keys(item)
        if fname in SLOW_FILES or key in SLOW_TESTS:
            item.add_marker(pytest.mark.slow)
        if not run_slow and "slow" in item.keywords:
            item.add_marker(skip_slow)
        if fname in GPU_TESTS or key in GPU_TESTS:
            item.add_marker(pytest.mark.gpu)


@pytest.fixture(autouse=True, scope="session")
def _private_wall_cache(tmp_path_factory):
    """Tests never read or fill the user's wall-thickness cache (~/.cache/shape_reco/wall_thickness): a hit there would
    hide a measurement a test counts, and fake parts do not belong in it. Session-wide, so module-scoped fixtures that
    fold a run (test_reco_gui_cascade.session) are covered too; a test may still point it elsewhere itself."""
    old = os.environ.get("SHAPE_RECO_WALL_CACHE")
    os.environ["SHAPE_RECO_WALL_CACHE"] = str(tmp_path_factory.mktemp("wall_cache"))
    yield
    if old is None:
        os.environ.pop("SHAPE_RECO_WALL_CACHE", None)
    else:
        os.environ["SHAPE_RECO_WALL_CACHE"] = old
