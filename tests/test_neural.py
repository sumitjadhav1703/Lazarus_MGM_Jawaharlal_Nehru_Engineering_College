import numpy as np
import pytest

torch = pytest.importorskip("torch")
from cardiotwin.neural import (  # noqa: E402
    RRTCN,
    CausalConv1d,
    SequenceNormalizer,
    WaveformCNN,
    rr_sequence,
    waveform_sequence,
)


def test_temporal_convolution_cannot_read_future_tokens():
    layer = CausalConv1d(2, 3, kernel_size=3, dilation=4).eval()
    left = torch.randn(1, 2, 50)
    right = left.clone()
    right[:, :, 25:] += 100
    torch.testing.assert_close(layer(left)[:, :, :25], layer(right)[:, :, :25])


def test_optional_models_forward_and_backward():
    torch.set_num_threads(1)
    for model, shape in ((RRTCN(16), (2, 4, 80)), (WaveformCNN(16), (2, 3, 1000))):
        logits = model(torch.randn(*shape))
        assert logits.shape == (2,)
        logits.sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_normalization_is_fitting_only_and_retains_mask():
    fitting = np.array([[1, 2, 3], [0, 0, 0], [0.1, 0.2, 0.3], [1, 1, 1]], dtype=float)
    normalizer = SequenceNormalizer().fit([fitting])
    mean = normalizer.mean.copy()
    transformed = normalizer.transform(fitting * 10)
    np.testing.assert_equal(normalizer.mean, mean)
    np.testing.assert_equal(transformed[-1], fitting[-1] * 10)
    fitting[:, 0] = 0
    assert (normalizer.transform(fitting)[:, 0] == 0).all()


def test_sequences_preserve_missing_intervals(monkeypatch):
    monkeypatch.setattr("cardiotwin.features._peaks", lambda x, fs: np.arange(1, 29, 0.8))
    signal = np.ones((3000, 2))
    sequence = rr_sequence(signal, 100, 30, 60)
    assert sequence.shape == (4, 60)
    signal[500:800] = np.nan
    gapped = rr_sequence(signal, 100, 30, 60)
    assert gapped[-1].sum() < sequence[-1].sum()
    wave = waveform_sequence(signal, 100, 30)
    assert wave.shape == (3, 3000)
    assert (wave[:2, 500:800] == 0).all()
    assert (wave[-1, 500:800] == 0).all()


def test_causal_antialias_filter_reduces_out_of_band_signal():
    fs = 200
    t = np.arange(fs * 30) / fs
    noise = np.sin(2 * np.pi * 80 * t)
    waveform = waveform_sequence(np.column_stack((noise, noise)), fs, 30)
    assert np.std(waveform[0, 100:]) < 0.03


def test_one_epoch_training_is_group_disjoint():
    from cardiotwin.neural import fit_network, network_scores

    fit = [
        dict(patient_id=f"p{i}", group_id=f"g{i}", accepted=True, label=label)
        for i in range(2)
        for label in (0, 1)
    ]
    stop = [dict(patient_id="hold", group_id="hold", accepted=True, label=label) for label in (0, 1)]

    def provider(row):
        values = np.ones((4, 80), dtype=np.float32)
        values[0] = 0.7 + row["label"] * 0.2
        values[1] = 0.02
        values[2] = np.linspace(0, 1, 80)
        return values

    model, normalizer, details = fit_network(fit, stop, provider, max_epochs=1, batch_size=2, device="cpu")
    assert details["epochs"] == 1
    assert not set(details["fit_groups"]) & set(details["early_stop_groups"])
    assert np.isfinite(network_scores(model, normalizer, stop, provider)).all()
    with pytest.raises(ValueError, match="disjoint"):
        fit_network(fit, fit, provider, max_epochs=1, device="cpu")


@pytest.mark.parametrize("weighting,expected_epoch", [("patient_only", 1), ("patient_class", 2)])
def test_rare_event_stopping_objective_restores_matching_checkpoint(monkeypatch, weighting, expected_epoch):
    from cardiotwin import neural

    class BiasModel(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return self.bias.expand(len(x))

    class TwoEpochOptimizer:
        def __init__(self, parameters, **kwargs):
            self.bias = next(iter(parameters))
            self.epoch = 0

        def zero_grad(self, **kwargs):
            self.bias.grad = None

        def step(self):
            probability = (0.01, 0.2)[self.epoch]
            with torch.no_grad():
                self.bias.fill_(np.log(probability / (1 - probability)))
            self.epoch += 1

    monkeypatch.setattr(neural, "RRTCN", BiasModel)
    monkeypatch.setattr(torch.optim, "AdamW", TwoEpochOptimizer)
    fitting = [dict(patient_id="fit", accepted=True, label=label) for label in (0, 1)]
    stopping = [dict(patient_id="stop", accepted=True, label=int(i == 100)) for i in range(101)]
    model, _, details = neural.fit_network(
        fitting,
        stopping,
        lambda row: np.ones((4, 8), dtype=np.float32),
        max_epochs=2,
        batch_size=128,
        device="cpu",
        early_stop_weighting=weighting,
    )
    assert details["best_epoch"] == expected_epoch
    assert torch.sigmoid(model.bias).item() == pytest.approx((0.01, 0.2)[expected_epoch - 1])
    assert details["early_stop_weighting"] == weighting
    assert details["early_stop_positive_windows"] == 1
    history = details["epoch_history"]
    assert (
        history[0]["early_stop_group_macro_natural_loss"] < history[1]["early_stop_group_macro_natural_loss"]
    )
    assert (history[0]["early_stop_loss"] < history[1]["early_stop_loss"]) is (expected_epoch == 1)


@pytest.mark.parametrize("artifact_failure", [False, True])
@pytest.mark.parametrize("custom_cache", [False, True])
@pytest.mark.parametrize("early_stop_weighting", ["patient_only", "patient_class"])
@pytest.mark.parametrize(
    "representation,model_name", [("rr", "rr_tcn_16"), ("waveform_resnet", "waveform_resnet_16")]
)
def test_neural_summary_preserves_group_evidence_and_all_monitoring_burden(
    tmp_path,
    monkeypatch,
    capsys,
    artifact_failure,
    custom_cache,
    early_stop_weighting,
    representation,
    model_name,
):
    import json
    from types import SimpleNamespace

    from cardiotwin import neural
    from cardiotwin.config import load_config
    from cardiotwin.investigation import _digest

    config = load_config()
    config["alert"]["cooldown_seconds"] = 0
    rows = [
        dict(
            patient_id=f"p{i}",
            group_id=f"g{i}",
            record_id=f"r{i}",
            timestamp=float(t),
            duration_seconds=2400,
            accepted=t != 900,
            label=int(1200 <= t <= 1680) if t < 1800 else None,
            label_reason="current_AF" if 1800 <= t < 2000 else "unknown" if t >= 2000 else None,
            reason="quality" if t == 900 else None,
        )
        for i in range(24)
        for t in range(600, 2400, 30)
    ]
    events = [dict(patient_id=f"p{i}", group_id=f"g{i}", record_id=f"r{i}", onset=1800) for i in range(24)]
    provider = SimpleNamespace(close=lambda: None)
    cache_paths = []

    def sequence_provider(*args, **kwargs):
        assert args[2] == ("rr" if representation == "rr" else "waveform")
        cache_paths.append(args[3])
        return provider

    monkeypatch.setattr(neural, "record_sequence_provider", sequence_provider)
    monkeypatch.setattr(
        neural,
        "fit_network",
        lambda *args, **kwargs: (
            torch.nn.Linear(1, 1),
            SimpleNamespace(mean=np.zeros(3), scale=np.ones(3)),
            {},
        ),
    )
    monkeypatch.setattr(
        neural,
        "network_scores",
        lambda model, normalizer, supplied, provider, **kwargs: np.array(
            [
                3.0 if row["label"] == 1 or row["timestamp"] in (1800, 1830, 1860, 2100, 2130, 2160) else -3.0
                for row in supplied
                if row["accepted"]
            ]
        ),
    )
    if artifact_failure:

        def fail_save(*args, **kwargs):
            raise RuntimeError("simulated artifact write failure")

        monkeypatch.setattr(torch, "save", fail_save)
    result = neural.run_neural_ladder(
        rows,
        events,
        [],
        config,
        tmp_path / "screen",
        specs=[(representation, 16)],
        source_contract="raw-content",
        cache_contract={"sha256": "development-content"},
        early_stop_weighting=early_stop_weighting,
        **({"sequence_cache_path": tmp_path / "shared-rr.sqlite"} if custom_cache else {}),
    )
    assert cache_paths == (
        [tmp_path / ("shared-rr.sqlite" if custom_cache else "screen/rr-sequences.sqlite")]
        if representation == "rr"
        else [None]
    )
    progress_events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    fold_ends = [
        e for e in progress_events if e["stage"] == "neural.fold" and e["status"] in ("done", "failed")
    ]
    assert len(fold_ends) == 3
    assert all(e["status"] == ("failed" if artifact_failure else "done") for e in fold_ends)
    assert progress_events[-1]["stage"] == "neural.candidate"
    assert progress_events[-1]["status"] == ("failed" if artifact_failure else "done")
    assert progress_events[-1]["completed"] == (0 if artifact_failure else 3)
    assert progress_events[-1]["complete_outer_folds"] == (0 if artifact_failure else 3)
    assert progress_events[-1]["attempted_folds"] == 3
    for stage in (
        "neural.calibration_scoring",
        "neural.calibration",
        "neural.policy_scoring",
        "neural.policy_search",
        "neural.outer_scoring",
        "neural.outer_evaluation",
        "neural.artifacts",
    ):
        assert sum(e["stage"] == stage and e["status"] == "started" for e in progress_events) == 3
        status = "failed" if artifact_failure and stage == "neural.artifacts" else "done"
        assert sum(e["stage"] == stage and e["status"] == status for e in progress_events) == 3
    if artifact_failure:
        assert result["combined"] == {}
        assert len(result["failures"]) == 3
        return
    report = result["combined"][model_name]
    assert report["complete_outer_folds"] == 3
    assert report["cost_seconds"] > 0
    assert len(report["per_group_brier"]) == 24
    assert np.mean(list(report["per_group_brier"].values())) == pytest.approx(
        report["probability"]["group_macro_brier"]
    )
    assert report["forecast_coverage"] == pytest.approx(59 / 60)
    assert report["known_excluded_warnings"] > 0
    assert report["unknown_outcome_warnings"] > 0
    assert (
        report["warnings"]
        == report["matched_events"]
        + report["false_warnings"]
        + report["known_excluded_warnings"]
        + report["unknown_outcome_warnings"]
    )
    experiments = [
        json.loads(line) for line in (tmp_path / "screen" / "experiments.jsonl").read_text().splitlines()
    ]
    assert report["monitored_hours"] == pytest.approx(
        sum(item["report"]["monitored_hours"] for item in experiments)
    )
    assert report["rejection_hours"]["quality"] == pytest.approx(24 * 30 / 3600)
    assert result["sha256"] == _digest({key: value for key, value in result.items() if key != "sha256"})
    assert result["development_cache_contract"] == {"sha256": "development-content"}
    assert result["source_contract"] == "raw-content"
    assert result["extraction_contract"]["target"] == config["target"]
    checkpoint = torch.load(tmp_path / f"screen/fold-0-{model_name}.pt", weights_only=True)
    assert checkpoint["representation"] == representation
    assert set(checkpoint) == {
        "state_dict",
        "normalizer_mean",
        "normalizer_scale",
        "representation",
        "width",
        "config",
        "feature_version",
        "source_contract",
        "deployment_ready",
        "early_stop_weighting",
    }
    assert checkpoint["early_stop_weighting"] == result["early_stop_weighting"] == early_stop_weighting
    assert checkpoint["deployment_ready"] is False


@pytest.mark.parametrize("limit", [1, 64_000])
def test_sequence_cache_cap_preserves_existing_reads_and_uncached_values(tmp_path, limit):
    import sqlite3

    from cardiotwin.config import load_config
    from cardiotwin.data import RecordInfo
    from cardiotwin.neural import record_sequence_provider

    config = load_config()
    config["target"]["history_seconds"] = 30
    fs = 100
    t = np.arange(fs * 60) / fs
    signal = np.exp(-(((t % 0.8 - 0.4) / 0.025) ** 2))
    signal = np.column_stack((signal, signal))
    path = tmp_path / "signal.npz"
    np.savez(path, signal=signal)
    record = RecordInfo("r", "p", str(path), fs, 60, False, {}, [])
    row30 = dict(patient_id="p", record_id="r", timestamp=30)
    row60 = dict(patient_id="p", record_id="r", timestamp=60)
    database = tmp_path / "rr.sqlite"
    original = record_sequence_provider([record], config, cache_path=database, source_contract="fixture")
    expected30 = original(row30).copy()
    original.close()
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM sequences").fetchone()[0] == 1
    uncached = record_sequence_provider([record], config, source_contract="fixture")
    expected60 = uncached(row60).copy()
    uncached.close()
    capped = record_sequence_provider(
        [record],
        config,
        cache_path=database,
        source_contract="fixture",
        max_cache_bytes=limit,
    )
    np.testing.assert_array_equal(capped(row30), expected30)
    np.testing.assert_array_equal(capped(row60), expected60)
    assert capped.cache_diagnostics()["cap_hit"] is True
    assert capped.cache_diagnostics()["writes"] == 0
    capped.close()
    capped.close()
    assert capped.cache_diagnostics()["closed"] is True
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM sequences").fetchone()[0] == 1


def test_sequence_provider_checks_deadline_before_extracting(tmp_path):
    import time

    from cardiotwin.config import load_config
    from cardiotwin.data import RecordInfo
    from cardiotwin.neural import record_sequence_provider

    record = RecordInfo("r", "p", str(tmp_path / "missing.npz"), 100, 1200, False, {}, [])
    provider = record_sequence_provider(
        [record],
        load_config(),
        source_contract="fixture",
        deadline=time.monotonic() - 1,
    )
    try:
        with pytest.raises(TimeoutError, match="budget"):
            provider(dict(patient_id="p", record_id="r", timestamp=600))
    finally:
        provider.close()


def test_sequence_cache_close_error_preserves_original_failure(tmp_path, monkeypatch):
    import sqlite3

    from cardiotwin.config import load_config
    from cardiotwin.data import RecordInfo
    from cardiotwin.neural import record_sequence_provider

    class CommitFailure(sqlite3.Connection):
        def commit(self):
            self.commits = getattr(self, "commits", 0) + 1
            if self.commits > 1:
                raise sqlite3.OperationalError("simulated disk full")
            super().commit()

    connect = sqlite3.connect
    connections = []

    def failing_connect(*args, **kwargs):
        connection = connect(*args, **kwargs, factory=CommitFailure)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", failing_connect)
    record = RecordInfo("r", "p", str(tmp_path / "unused.npz"), 100, 1200, False, {}, [])
    provider = record_sequence_provider(
        [record],
        load_config(),
        cache_path=tmp_path / "rr.sqlite",
        source_contract="fixture",
    )
    with pytest.raises(ValueError, match="original failure"):
        try:
            raise ValueError("original failure")
        finally:
            provider.close()
    assert provider.cache_diagnostics()["errors"] == ["simulated disk full"]
    assert provider.cache_diagnostics()["closed"] is True
    with pytest.raises(sqlite3.ProgrammingError):
        connections[0].execute("SELECT 1")


def test_waveform_batch_matches_individual_values_reads_on_owner_and_closes(tmp_path, monkeypatch):
    import threading
    from contextlib import contextmanager

    from cardiotwin import data, neural
    from cardiotwin.config import load_config

    config = load_config()
    config["target"]["history_seconds"] = 30
    fs = 200
    t = np.arange(fs * 60) / fs
    signal = np.column_stack((np.sin(t * 6), np.cos(t * 5)))
    signal[1000:1500, 0] = np.nan
    path = tmp_path / "signal.npz"
    np.savez(path, signal=signal)
    record = data.RecordInfo("r", "p", str(path), fs, 60, False, {}, [])
    rows = [dict(patient_id="p", record_id="r", timestamp=t, label=i % 2) for i, t in enumerate((60, 30, 40))]
    owner = threading.get_ident()
    read_threads, filter_threads = [], []
    original_reader, original_sequence = data.cached_signal_reader, neural.waveform_sequence

    @contextmanager
    def checked_reader():
        with original_reader() as read:

            def checked_read(*args, **kwargs):
                read_threads.append(threading.get_ident())
                return read(*args, **kwargs)

            yield checked_read

    def checked_sequence(*args, **kwargs):
        filter_threads.append(threading.get_ident())
        return original_sequence(*args, **kwargs)

    monkeypatch.setattr(data, "cached_signal_reader", checked_reader)
    monkeypatch.setattr(neural, "waveform_sequence", checked_sequence)
    provider = neural.record_sequence_provider(
        [record],
        config,
        representation="waveform",
        source_contract="fixture",
    )
    try:
        individual = [provider(row).copy() for row in rows]
        batched = provider.batch(rows)
        for expected, actual in zip(individual, batched, strict=True):
            np.testing.assert_array_equal(actual, expected)
        normalizer = neural.SequenceNormalizer().fit(individual)
        dataset = neural.SequenceDataset(rows, provider, normalizer, weights=[1, 2, 3])
        x, y, weights = next(iter(torch.utils.data.DataLoader(dataset, batch_size=3)))
        torch.testing.assert_close(x, torch.stack([dataset[i][0] for i in range(3)]), rtol=0, atol=0)
        torch.testing.assert_close(y, torch.tensor([0.0, 1.0, 0.0]))
        torch.testing.assert_close(weights, torch.tensor([1.0, 2.0, 3.0]))
        assert set(read_threads) == {owner}
        workers = set(filter_threads) - {owner}
        assert 1 <= len(workers) <= 4
        with pytest.raises(ValueError, match="64"):
            provider.batch([rows[0]] * 65)
    finally:
        provider.close()
    assert not any(thread.name.startswith("cardiotwin-waveform") for thread in threading.enumerate())
    with pytest.raises(RuntimeError, match="closed"):
        provider.batch(rows)


def test_waveform_batch_checks_deadline_before_any_source_read(tmp_path):
    import time

    from cardiotwin.config import load_config
    from cardiotwin.data import RecordInfo
    from cardiotwin.neural import record_sequence_provider

    record = RecordInfo("r", "p", str(tmp_path / "missing.npz"), 100, 1200, False, {}, [])
    provider = record_sequence_provider(
        [record],
        load_config(),
        representation="waveform",
        source_contract="fixture",
        deadline=time.monotonic() - 1,
    )
    try:
        with pytest.raises(TimeoutError, match="budget"):
            provider.batch([dict(patient_id="p", record_id="r", timestamp=600)])
    finally:
        provider.close()


def test_warm_sequence_cache_still_refuses_source_contract_mismatch(tmp_path):
    from cardiotwin.config import load_config
    from cardiotwin.neural import record_sequence_provider

    path = tmp_path / "warm-rr.sqlite"
    config = load_config()
    provider = record_sequence_provider([], config, cache_path=path, source_contract="source-a")
    provider.close()
    with pytest.raises(ValueError, match="contract differs"):
        record_sequence_provider([], config, cache_path=path, source_contract="source-b")


def test_warm_sequence_cache_does_not_allow_reusing_experiment_output(tmp_path):
    from cardiotwin.config import load_config
    from cardiotwin.neural import run_neural_ladder

    output = tmp_path / "previous-output"
    output.mkdir()
    (output / "summary.json").write_text("previous evidence")
    with pytest.raises(ValueError, match="fresh empty directory"):
        run_neural_ladder([], [], [], load_config(), output, sequence_cache_path=tmp_path / "warm.sqlite")
    assert (output / "summary.json").read_text() == "previous evidence"


@pytest.mark.parametrize("specs,weighting", [([("rr", 16)] * 2, "patient_only"), (None, "invalid")])
def test_invalid_neural_recipe_rejected_before_output_creation(tmp_path, specs, weighting):
    from cardiotwin.config import load_config
    from cardiotwin.neural import run_neural_ladder

    output = tmp_path / "unused"
    with pytest.raises(ValueError, match="unique|Early-stop"):
        run_neural_ladder([], [], [], load_config(), output, specs=specs, early_stop_weighting=weighting)
    assert not output.exists()


@pytest.mark.parametrize("weighting,stop_labels", [("invalid", [0, 1]), ("patient_class", [0])])
def test_invalid_stopping_recipe_does_not_read_signals(weighting, stop_labels):
    from cardiotwin.neural import fit_network

    def forbidden(row):
        pytest.fail("Invalid stopping inputs must fail before signal reads")

    fit = [dict(patient_id="fit", accepted=True, label=label) for label in (0, 1)]
    stop = [dict(patient_id="stop", accepted=True, label=label) for label in stop_labels]
    with pytest.raises(ValueError, match="weighting|both classes"):
        fit_network(fit, stop, forbidden, device="cpu", early_stop_weighting=weighting)


def test_oom_retry_keeps_stopping_objective(monkeypatch):
    from cardiotwin import neural

    calls = []

    def fit(*args):
        calls.append((args[8], args[-1]))
        if len(calls) == 1:
            raise RuntimeError("out of memory: fixture")
        return torch.nn.Linear(1, 1), None, dict(epochs=1, early_stop_metric="balanced fixture")

    monkeypatch.setattr(neural, "_fit_network", fit)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    _, _, report = neural.fit_network([], [], None, batch_size=4, early_stop_weighting="patient_class")
    assert calls == [(4, "patient_class"), (2, "patient_class")]
    assert report["early_stop_metric"] == "balanced fixture"
    assert report["oom_retries"] == [{"failed_batch_size": 4, "reason": "out of memory: fixture"}]


@pytest.mark.parametrize(
    "candidate,representation", [("waveform16", "waveform"), ("waveform_resnet16", "waveform_resnet")]
)
def test_neural_cli_forwards_explicit_warm_sequence_cache(tmp_path, monkeypatch, candidate, representation):
    import json
    import runpy
    import sys
    from pathlib import Path

    from cardiotwin import neural
    from cardiotwin.config import load_config

    cli = runpy.run_path(str(Path(__file__).parents[1] / "scripts/run_investigation.py"))["main"]
    index = dict(
        records=[],
        allowed_groups=[],
        rows_sha256="rows",
        input_sha256="input",
        sha256="cache",
        config=load_config(),
    )
    split = tmp_path / "split.json"
    split.write_text(json.dumps(dict(record_contracts={}, development_groups=[], group_ids={}, seed=42)))
    monkeypatch.setitem(cli.__globals__, "read_cache", lambda *args: ([], [], index))
    monkeypatch.setitem(cli.__globals__, "list_records", lambda *args: [])
    monkeypatch.setitem(cli.__globals__, "validate_live_inputs", lambda *args: None)
    monkeypatch.setitem(cli.__globals__, "sequence_cache_capacity", lambda *args: 100)
    captured = {}

    def ladder(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "dispatched"}

    monkeypatch.setattr(neural, "run_neural_ladder", ladder)
    cache_path = tmp_path / "warm.sqlite"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_investigation.py",
            "neural",
            "--cache",
            "unused",
            "--split",
            str(split),
            "--data",
            "unused",
            "--output",
            str(tmp_path / "output"),
            "--sequence-cache",
            str(cache_path),
            "--models",
            candidate,
            "--early-stop-weighting",
            "patient_class",
        ],
    )
    cli()
    assert captured["sequence_cache_path"] == cache_path
    assert captured["specs"] == [(representation, 16)]
    assert captured["early_stop_weighting"] == "patient_class"


def test_normalization_progress_reports_counts_and_failure_without_payload(capsys):
    import json

    arrays = [np.ones((4, 8), dtype=np.float32)] * 2
    normalizer = SequenceNormalizer().fit(iter(arrays), total=2)
    np.testing.assert_array_equal(normalizer.mean, np.ones(3))
    with pytest.raises(ValueError, match="finite"):
        SequenceNormalizer().fit([np.full((4, 8), np.nan)], total=1)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(event["stage"], event["status"]) for event in events] == [
        ("neural.normalization", "started"),
        ("neural.normalization", "done"),
        ("neural.normalization", "started"),
        ("neural.normalization", "failed"),
    ]
    assert events[1]["completed"] == events[1]["total"] == 2
    assert events[-1]["error_type"] == "ValueError"
    assert all("patient_id" not in event for event in events)


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_training_transfer_intent_and_phase_progress_preserve_cpu_checkpoint(device, monkeypatch, capsys):
    import json

    from cardiotwin import neural

    fit = [dict(patient_id=f"p{i}", accepted=True, label=label) for i in range(2) for label in (0, 1)]
    stop = [dict(patient_id="stop", accepted=True, label=label) for label in (0, 1)]

    def provider(row):
        array = np.ones((4, 16), dtype=np.float32)
        array[0] = 0.7 + 0.2 * row["label"]
        return array

    loaders, transfers = [], []
    original_loader, original_to = torch.utils.data.DataLoader, torch.Tensor.to

    def loader(*args, **kwargs):
        loaders.append(kwargs.copy())
        return original_loader(*args, **dict(kwargs, pin_memory=False))

    def tensor_to(self, *args, **kwargs):
        if args and str(args[0]) in ("cpu", "mps", "cuda") and "non_blocking" in kwargs:
            transfers.append((str(args[0]), kwargs.get("non_blocking")))
            return self
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.utils.data, "DataLoader", loader)
    monkeypatch.setattr(torch.Tensor, "to", tensor_to)
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *args, **kwargs: self)
    model, normalizer, details = neural.fit_network(
        fit, stop, provider, max_epochs=1, batch_size=2, device=device
    )
    assert all(item.get("pin_memory") is (device == "cuda") for item in loaders)
    assert all(item.get("num_workers", 0) == 0 for item in loaders)
    assert transfers and all(nonblocking is (target == "cuda") for target, nonblocking in transfers)
    assert details["epochs"] == 1 and details["batch_size"] == 2
    assert details["optimizer"] == "AdamW" and details["loss"] == "patient-class weighted BCE"
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    for stage in ("neural.fit", "neural.normalization", "neural.train_epoch", "neural.validate_epoch"):
        assert any(e["stage"] == stage and e["status"] == "started" for e in events)
        assert any(e["stage"] == stage and e["status"] == "done" for e in events)
    assert (
        next(e for e in events if e["stage"] == "neural.train_epoch" and e["status"] == "done")["completed"]
        == 2
    )


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_scoring_transfer_intent_keeps_logits_order(device, monkeypatch, capsys):
    import json
    from types import SimpleNamespace

    from cardiotwin import neural

    rows = [dict(accepted=True, label=label) for label in (0, 1, 0)]

    def provider(row):
        array = np.ones((4, 8), dtype=np.float32)
        array[0] = row["label"] + 0.5
        return array

    normalizer = neural.SequenceNormalizer().fit([provider(row) for row in rows])
    real_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(32, 1), torch.nn.Flatten(0))

    class Model:
        def parameters(self):
            return iter([SimpleNamespace(device=torch.device(device))])

        def eval(self):
            real_model.eval()

        def __call__(self, values):
            return real_model(values)

    expected = (
        real_model(torch.from_numpy(np.stack([normalizer.transform(provider(row)) for row in rows])))
        .detach()
        .numpy()
    )
    loaders, transfers = [], []
    original_loader, original_to = torch.utils.data.DataLoader, torch.Tensor.to

    def loader(*args, **kwargs):
        loaders.append(kwargs.copy())
        return original_loader(*args, **dict(kwargs, pin_memory=False))

    def tensor_to(self, *args, **kwargs):
        if args and str(args[0]) in ("cpu", "mps", "cuda"):
            transfers.append((str(args[0]), kwargs.get("non_blocking")))
            return self
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.utils.data, "DataLoader", loader)
    monkeypatch.setattr(torch.Tensor, "to", tensor_to)
    actual = neural.network_scores(Model(), normalizer, rows, provider, batch_size=3)
    np.testing.assert_array_equal(actual, expected)
    assert loaders[0].get("pin_memory") is (device == "cuda")
    assert transfers == [(device, device == "cuda")]
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["stage"] == "neural.scores" and events[-1]["status"] == "done"
    assert events[-1]["completed"] == events[-1]["total"] == 1


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:1"])
def test_device_phase_metadata_queries_allocated_memory_only_for_cuda(device, monkeypatch):
    from cardiotwin import neural

    queried = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda supplied: queried.append(supplied) or 123)
    details = neural._device_details(torch.device(device))
    assert details["device"] == device
    if device.startswith("cuda"):
        assert queried == [torch.device(device)]
        assert details["cuda_allocated_bytes"] == 123
    else:
        assert queried == [] and "cuda_allocated_bytes" not in details
    assert "utilization" not in details
