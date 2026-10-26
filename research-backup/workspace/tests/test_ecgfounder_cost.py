"""Small benchmark validation/guard tests; never execute its full workload."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/benchmark_ecgfounder_cost.py"
REQUIREMENTS = (
    ROOT
    / "reports/ml-investigation/next-cycle/ecgfounder-cycle-20261013/repair-panel-20261014-b/full-cost-20261015/requirements.json"
)


def benchmark():
    assert SCRIPT.exists(), "Synthetic cost benchmark absent"
    spec = importlib.util.spec_from_file_location("ecgfounder_cost_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_recipe_accepts_reviewed_shapes_and_rejects_mutation():
    module = benchmark()
    recipe = json.loads(REQUIREMENTS.read_text())
    module.validate_recipe(recipe)
    for key, value in (
        ("primary_heads", 5),
        ("null_heads", 56),
        ("augmented_columns", 2094),
        ("bootstrap_samples", 100),
        ("scoring_batch_rows", 8192),
    ):
        changed = copy.deepcopy(recipe)
        changed["fixed_work"][key] = value
        with pytest.raises(ValueError):
            module.validate_recipe(changed)


@pytest.mark.parametrize("kind", ["whole", "phase", "rss", "output", "storage"])
def test_budget_enforces_stop_ceilings_without_calling_benchmark(kind):
    module = benchmark()
    budget = module.Budget(started=0.0, monotonic=lambda: 0.0)
    budget.stage("synthetic", now=0.0)
    kwargs = dict(now=0.0, rss=1, output_bytes=1, storage={"allocated_bytes": 1, "logical_bytes": 1})
    if kind == "whole":
        budget.started = -1799.0
        kwargs["now"] = 2.0
    elif kind == "phase":
        kwargs["now"] = 1201.0
    elif kind == "rss":
        kwargs["rss"] = 10_000_000_001
    elif kind == "output":
        kwargs["output_bytes"] = 250_000_001
    else:
        kwargs["storage"]["allocated_bytes"] = 24_750_000_001
    with pytest.raises((TimeoutError, ValueError)):
        budget.check(**kwargs)


def test_atomic_failure_receipt_retains_missing_bounds_and_original_failure(tmp_path):
    module = benchmark()
    value = module.base_receipt({"source.py": "a" * 64}, "b" * 64)
    value.update(status="failed", error="original synthetic error", completed=7)
    path = tmp_path / "receipt.json"
    module.atomic_json(path, value)
    saved = json.loads(path.read_text())
    assert saved["error"] == "original synthetic error" and saved["completed"] == 7
    assert saved["evidence_class"] == "local-software-benchmark-only"
    assert not saved["full_run_eligible"] and saved["actual_outcome_fits"] == 0
    assert len(saved["accepted_remote_upper_seconds"]) == 11
    assert all(v is None for v in saved["accepted_remote_upper_seconds"].values())
    assert not path.with_suffix(".partial").exists()


def test_producer_timer_keeps_original_error_and_elapsed():
    module = benchmark()
    clock = iter([1.0, 3.0])

    def broken():
        raise ValueError("original producer error")
        yield

    measured = module.TimedProducer(iter(broken()), monotonic=lambda: next(clock))
    with pytest.raises(ValueError, match="original producer error"):
        next(measured)
    assert measured.seconds == 2.0 and measured.completed == 0 and measured.attempted == 1


def test_tiny_writer_fixture_uses_verified_reader_and_restores_real_pins(tmp_path):
    module = benchmark()
    from cardiotwin import ecgfounder_compare as compare

    pinned = (compare.CHECKPOINT_BYTES, compare.CHECKPOINT_SHA256)
    synthetic = module.SyntheticRows(8, record_count=1, people=1)
    output, args, accounting = module.write_fixture(tmp_path, synthetic, rejected=False)
    with module.open_synthetic(args) as data:
        assert isinstance(data, compare.VerifiedTables)
        assert data.batch([0, 1])[1].shape == (2, 1024)
        assert not data.delta.flags.writeable
        assert (compare.CHECKPOINT_BYTES, compare.CHECKPOINT_SHA256) == pinned
    assert accounting["table_hash_passes"] == 2
    assert accounting["table_hash_bytes"] > 0
    assert json.loads((output / "receipt.json").read_text())["complete"]
    args["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError), module.open_synthetic(args):
        pass
    assert (compare.CHECKPOINT_BYTES, compare.CHECKPOINT_SHA256) == pinned


def test_fixed_pairs_are_unique_disjoint_roles_and_correct_shape():
    module = benchmark()
    rows = module.SyntheticRows(8192)
    for pattern in range(2):
        pairs, stop = module.synthetic_pairs(rows, pattern)
        assert len(pairs) == 241 and len(stop) == 4
        indices = [i for p in pairs for i in (p.positive_index, p.negative_index)]
        assert len(set(indices)) == 482
        assert len({p.group_id for p in pairs}) == 119
        assert {p.group_id for p in pairs}.isdisjoint({p.group_id for p in stop})
        assert all(
            rows[p.positive_index]["timestamp"] - rows[p.negative_index]["timestamp"] == 3000.0 for p in pairs
        )


def test_synthetic_chronology_has_fixed_counts_known_classes_and_no_features():
    module = benchmark()
    rows, events = module.chronology(2000, 1800, 1)
    assert len(rows) == 2000 and len(events) == 241
    assert sum(r["original_accepted"] for r in rows) == 1800
    assert {r["label"] for r in rows if r["accepted"] and r["label"] in (0, 1)} == {0, 1}
    assert any(not r["accepted"] and r["original_accepted"] for r in rows)
    assert all("features" not in r for r in rows)
    assert any(r["label_reason"] == "unknown_tail" for r in rows)


def test_bootstrap_fixture_retains_all_ten_denominators_and_negative_only_people():
    module = benchmark()
    first, second = module.bootstrap_people()
    assert len(first) == len(second) == 121
    assert all(len(p) == 11 for p in first)
    assert any(p["eligible_events"] == 0 for p in first)
    for a, b in zip(first, second, strict=True):
        for key in (
            "eligible_events",
            "eligible_seconds",
            "monitored_seconds",
            "eligible_negative_seconds",
            "available_seconds",
        ):
            assert a[key] == b[key]


def test_source_binding_requires_exact_dependency_paths(tmp_path):
    module = benchmark()
    with pytest.raises(ValueError):
        module.verify_sources({"unrelated.py": "a" * 64}, root=tmp_path)


def test_declared_source_manifest_can_bind_every_real_dependency():
    module = benchmark()
    expected = {name: module.sha(ROOT / name) for name in module.DEPENDENCIES}
    module.verify_sources(expected)


def test_fixed_recipe_rejects_a_pattern_substitution():
    module = benchmark()
    recipe = json.loads(REQUIREMENTS.read_text())
    recipe["synthetic_recipe"]["solver_pattern_names"][1] = "easier_pattern"
    with pytest.raises(ValueError):
        module.validate_recipe(recipe)


def test_phase_start_does_not_renew_whole_benchmark_clock():
    module = benchmark()
    budget = module.Budget(started=0.0)
    budget.stage("later", now=1200.0)
    with pytest.raises(TimeoutError):
        budget.check(now=1801.0)


def test_fit_counter_keeps_completed_count_and_original_nonconvergence():
    module = benchmark()
    counts = {}

    def original(*args):
        raise ValueError("original fit nonconvergence")

    with pytest.raises(ValueError, match="original fit nonconvergence"):
        module.counted_fit(original, counts, "null", [[1.0]], [1], [1.0], 42)
    assert counts["null"] == {"attempted": 1, "completed": 0}


def test_failed_score_batch_retains_completed_rows_and_progress():
    from types import SimpleNamespace

    import numpy as np

    module = benchmark()
    counts, progress = {}, []

    def logits(timing, data, indices):
        if len(indices) == 1:
            raise ValueError("original later batch failure")
        return np.zeros(len(indices))

    with pytest.raises(ValueError, match="original later batch failure"):
        module._score_count(
            SimpleNamespace(logits=logits),
            None,
            SimpleNamespace(rows=range(8)),
            4097,
            counts,
            report=lambda event, **details: progress.append(details),
        )
    assert counts == dict(
        batches_attempted=2,
        batches_completed=1,
        rows_attempted=4097,
        rows_completed=4096,
        applicability="tiled_compute_only",
    )
    assert progress[-1]["scoring"] == counts
    assert progress[0]["scoring"]["rows_completed"] == 0


@pytest.mark.parametrize(
    "limit", ["output", "allocated", "logical", "rss", "missing_peak", "publication", None]
)
@pytest.mark.parametrize("event", ["complete", "failed"])
def test_terminal_message_enforces_final_resources_and_keeps_original_error(
    tmp_path, monkeypatch, limit, event
):
    from types import SimpleNamespace

    module = benchmark()
    recipe = json.loads(REQUIREMENTS.read_text())
    recipe_path = tmp_path / module.RECIPE_PATH
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(json.dumps(recipe))
    output = recipe_path.parent / "benchmarks" / "tiny-supervisor"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "verify_sources", lambda *args: None)
    calls = []

    def storage(roots):
        calls.append(roots)
        value = dict(allocated_bytes=0, logical_bytes=0)
        if len(calls) > 1:
            if len(roots) == 1 and limit == "output":
                value["logical_bytes"] = 250_000_001
            elif len(roots) == 1 and limit == "publication":
                value["logical_bytes"] = 250_000_000
            elif len(roots) == 3 and limit in ("allocated", "logical"):
                value[limit + "_bytes"] = 24_750_000_001
        return value

    message = dict(event=event, phases={}, peak_rss_bytes=10_000_000_001 if limit == "rss" else 1234)
    if limit == "missing_peak":
        message.pop("peak_rss_bytes")
    if event == "failed":
        message.update(error="ValueError: original worker failure", traceback="original traceback")

    class Queue:
        def get(self, **kwargs):
            return dict(message)

        def close(self):
            pass

    class Process:
        pid, exitcode = 123, 0

        def start(self):
            pass

        def join(self, **kwargs):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr(module, "storage_usage", storage)
    monkeypatch.setattr(
        module.mp, "get_context", lambda kind: SimpleNamespace(Queue=Queue, Process=lambda **kw: Process())
    )
    expected = {module.RECIPE_PATH: "a" * 64}
    if limit or event == "failed":
        with pytest.raises(RuntimeError):
            module.run_benchmark(output, recipe, expected, "a" * 64)
    else:
        module.run_benchmark(output, recipe, expected, "a" * 64)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["final_resources"]["peak_rss_bytes"] == message.get("peak_rss_bytes", 0)
    assert "output_storage" in receipt["final_resources"] and "project_storage" in receipt["final_resources"]
    assert receipt["status"] == (
        "complete_software_workload_only"
        if event == "complete" and limit is None
        else "failed"
        if event == "failed"
        else "partial"
    )
    if event == "failed":
        assert receipt["error"] == "ValueError: original worker failure"
        assert receipt["traceback"] == "original traceback"
        if limit and limit != "missing_peak":
            assert receipt["secondary_errors"]


def test_numerical_runtime_identity_and_dependency_files_are_bound(tmp_path):
    module = benchmark()
    assert {"pyproject.toml", "uv.lock"} <= set(module.DEPENDENCIES)
    identity = module.runtime_metadata(tmp_path)
    assert set(identity["numerical_versions"]) == {"numpy", "scipy", "scikit-learn"}
    assert all(identity["numerical_versions"].values())
    assert identity["threadpools"] and identity["cpu"]
    assert identity["filesystem"]["block_size"] > 0
    if identity["filesystem"]["type"] is None:
        assert identity["filesystem"]["error"]
    else:
        assert identity["filesystem"]["type"] != "/"


def test_macos_filesystem_uses_exact_mounted_device_and_plist(tmp_path, monkeypatch):
    import plistlib

    module = benchmark()
    mount = tmp_path / "mounted volume"
    mount.mkdir()
    calls = []
    monkeypatch.setattr(module.sys, "platform", "darwin")

    def command(args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "/bin/df":
            return f"Filesystem 512-blocks Used Available Capacity Mounted on\n/dev/disk3s5 100 20 80 20% {mount}\n"
        assert args == ["/usr/sbin/diskutil", "info", "-plist", "/dev/disk3s5"]
        return plistlib.dumps(
            dict(
                FilesystemType="apfs",
                FilesystemName="APFS",
                MountPoint=str(mount),
                DeviceIdentifier="disk3s5",
            )
        )

    monkeypatch.setattr(module.subprocess, "check_output", command)
    value = module.filesystem_metadata(tmp_path)
    assert value["type"] == "apfs" and value["mountpoint"] == str(mount)
    assert value["device_identifier"] == "disk3s5" and "error" not in value
    assert calls[0][0] == ["/bin/df", "-P", str(tmp_path)]
    assert all(kwargs["timeout"] == 2 for _, kwargs in calls)


@pytest.mark.parametrize(
    "failure",
    ["denied", "timeout", "malformed_df", "malformed_plist", "wrong_mount", "wrong_device", "missing_type"],
)
def test_macos_filesystem_unavailable_is_explicit(tmp_path, monkeypatch, failure):
    import plistlib

    module = benchmark()
    monkeypatch.setattr(module.sys, "platform", "darwin")

    def command(args, **kwargs):
        if failure == "denied":
            raise PermissionError("original disk observation denied")
        if failure == "timeout":
            raise module.subprocess.TimeoutExpired(args, 2)
        if args[0] == "/bin/df":
            return (
                "malformed"
                if failure == "malformed_df"
                else f"header\n/dev/disk3s5 100 20 80 20% {tmp_path}\n"
            )
        if failure == "malformed_plist":
            return b"not a plist"
        return plistlib.dumps(
            dict(
                FilesystemType="" if failure == "missing_type" else "apfs",
                MountPoint="/wrong" if failure == "wrong_mount" else str(tmp_path),
                DeviceIdentifier="wrong" if failure == "wrong_device" else "disk3s5",
            )
        )

    monkeypatch.setattr(module.subprocess, "check_output", command)
    value = module.filesystem_metadata(tmp_path)
    assert value["type"] is None and value["error"]
    assert value["device"] == tmp_path.stat().st_dev and value["block_size"] > 0


def test_linux_filesystem_keeps_native_stat_format(tmp_path, monkeypatch):
    module = benchmark()
    monkeypatch.setattr(module.sys, "platform", "linux")
    calls = []

    def command(args, **kwargs):
        calls.append((args, kwargs))
        return "ext2/ext3\n"

    monkeypatch.setattr(module.subprocess, "check_output", command)
    assert module.filesystem_metadata(tmp_path)["type"] == "ext2/ext3"
    assert calls[0][0] == ["stat", "-f", "-c", "%T", str(tmp_path)]


def test_evaluation_counter_distinguishes_attempts_and_completed_row_visits():
    module = benchmark()
    counts, progress = {}, []

    def evaluate(rows, *, fail):
        if fail:
            raise ValueError("original policy error")
        return len(rows)

    def report(event, **details):
        progress.append(details)

    assert module.counted_evaluation(evaluate, counts, "policy", range(3), fail=False, report=report) == 3
    with pytest.raises(ValueError, match="original policy error"):
        module.counted_evaluation(evaluate, counts, "policy", range(2), fail=True, report=report)
    assert counts["policy"] == dict(calls_attempted=2, calls_completed=1, rows_attempted=5, rows_completed=3)
    assert progress[-1]["evaluations"] == counts
