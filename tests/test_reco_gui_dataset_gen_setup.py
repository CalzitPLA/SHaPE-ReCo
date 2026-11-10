"""reco_gui.dataset_gen_setup: turning DatasetGenSettings into src.dataset_generator command lines (one per shard)
and summarizing a (possibly still-running) output directory - the pure-logic half of the dataset-generation GUI,
no Qt needed."""
import json
import sys
from pathlib import Path

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT))

from reco_gui.dataset_gen_setup import DatasetGenSettings, build_args, shard_jobs, summarize_output  # noqa: E402


def test_single_shard_by_default():
    s = DatasetGenSettings(output_dir=Path("out"), seed=7)
    jobs = shard_jobs(s)
    assert len(jobs) == 1
    assert jobs[0].output_dir == Path("out") and jobs[0].label == "(single run)"
    assert "--seed" in jobs[0].args and jobs[0].args[jobs[0].args.index("--seed") + 1] == "7"


def test_many_shards_get_their_own_subdir_and_offset_seed():
    s = DatasetGenSettings(output_dir=Path("out"), seed=1000, shards=4)
    jobs = shard_jobs(s)
    assert [j.label for j in jobs] == ["shard0", "shard1", "shard2", "shard3"]
    assert [j.output_dir for j in jobs] == [Path("out/shard0"), Path("out/shard1"), Path("out/shard2"), Path("out/shard3")]
    seeds = [j.args[j.args.index("--seed") + 1] for j in jobs]
    assert seeds == ["1000", "1001", "1002", "1003"]


def test_type_mode_carries_count_per_type_not_the_other_modes_counts():
    s = DatasetGenSettings(mode="type", count_per_type=250)
    args = build_args(s, Path("o"), 1)
    assert "--count-per-type" in args and args[args.index("--count-per-type") + 1] == "250"
    assert "--measure-ratios-per-type" not in args and "--rotation-groups-per-type" not in args


def test_fixed_mode_carries_hollow_and_carve_and_noise_flags():
    s = DatasetGenSettings(voxel_scale_mode="fixed", hollow_prob=1.0, hollow_wall=(0.1, 0.25), carve_prob=0.5,
                           cut_prob=0.6, require_manifold=True, noise_salt_prob=0.001)
    args = build_args(s, Path("o"), 1)
    assert args[args.index("--hollow-prob") + 1] == "1.0"
    assert args[args.index("--hollow-wall") + 1: args.index("--hollow-wall") + 3] == ["0.1", "0.25"]
    assert args[args.index("--carve-prob") + 1] == "0.5"
    assert "--require-manifold" in args
    assert args[args.index("--noise-salt-prob") + 1] == "0.001"


def test_fit_mode_skips_in_scale_only_flags():
    s = DatasetGenSettings(voxel_scale_mode="fit", hollow_prob=1.0)                # would be nonsense in "fit" mode
    args = build_args(s, Path("o"), 1)
    assert "--voxel-scale-mode" in args and args[args.index("--voxel-scale-mode") + 1] == "fit"
    assert "--hollow-prob" not in args and "--grid-mm" not in args


def test_no_voxelize_skips_every_voxel_flag():
    s = DatasetGenSettings(voxelize=False)
    args = build_args(s, Path("o"), 1)
    assert "--voxelize" not in args and "--voxel-resolution" not in args


def test_export_slurm_job_short_circuits_everything_else():
    s = DatasetGenSettings(export_slurm_job="job.sh", hollow_prob=1.0)
    args = build_args(s, Path("o"), 1)
    assert args[-2:] == ["--export-slurm-job", "job.sh"]
    assert "--hollow-prob" not in args


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['{"record_type": "dataset_header", "count_per_type": 10}']
    lines += [json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n")


def test_summarize_output_counts_topologies_and_placement_kinds(tmp_path):
    _write_jsonl(tmp_path / "shard0" / "box" / "dataset_metadata.jsonl", [
        {"topology": "box", "placement": {"hollow": {"wall_voxels": 3}}},
        {"topology": "box", "placement": {"carve": {"cutter": "sphere"}}},
        {"topology": "box", "placement": {"cuts": [{"keep": 0.5}]}},
        {"topology": "box", "placement": {}},
    ])
    _write_jsonl(tmp_path / "shard0" / "sphere" / "dataset_metadata.jsonl", [
        {"topology": "sphere", "placement": {"hollow": {"wall_voxels": 2}}},
    ])
    summary = summarize_output(tmp_path)
    assert summary["total"] == 5
    assert summary["per_topology"] == {"box": 4, "sphere": 1}
    assert summary["hollow"] == 2 and summary["carve"] == 1 and summary["cut"] == 1


def test_summarize_output_on_an_empty_or_missing_directory(tmp_path):
    assert summarize_output(tmp_path / "does_not_exist")["total"] == 0
    (tmp_path / "empty").mkdir()
    assert summarize_output(tmp_path / "empty")["total"] == 0
