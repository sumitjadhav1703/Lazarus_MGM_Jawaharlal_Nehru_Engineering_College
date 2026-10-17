import hashlib
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from cardiotwin import ecgfounder as ecg

SOURCE = Path(__file__).resolve().parents[1] / (
    "reports/ml-investigation/next-cycle/ecgfounder-cycle-20261013/vendor-source/net1d.py"
)


def native_wave():
    t = np.arange(2000) / 200
    return 0.7 * np.sin(2 * np.pi * 1.3 * t) + 0.1 * np.cos(2 * np.pi * 29 * t)


def test_preprocessing_matches_independent_chunk_local_release_recipe():
    native = native_wave()
    interpolated = np.interp(np.arange(5000) / 500, np.arange(2000) / 200, native)
    b, a = signal.iirnotch(50, 30, fs=500)
    filtered = signal.filtfilt(b, a, interpolated)
    b, a = signal.butter(4, [0.67, 40], btype="bandpass", fs=500)
    filtered = signal.filtfilt(b, a, filtered)
    filtered -= signal.medfilt(filtered, kernel_size=201)
    expected = ((filtered - filtered.mean()) / (filtered.std() + 1e-8)).astype(np.float32)
    actual = ecg.preprocess_chunk(native, sample_positions=np.arange(34000, 36000))
    assert actual.shape == (5000,) and actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-7)


@pytest.mark.parametrize("native,reason", [
    (np.zeros(2000), "native_constant"),
    (np.ones(1999), "native_support"),
    (np.ones((2000, 2)), "native_support"),
    (np.r_[np.nan, np.ones(1999)], "nonfinite_native"),
    (np.r_[np.inf, np.ones(1999)], "nonfinite_native"),
    (native_wave() * 1e-12, "filtered_degenerate"),
])
def test_invalid_native_chunks_are_rejected_without_imputation(native, reason):
    with pytest.raises(ecg.InvalidChunk) as error:
        ecg.preprocess_chunk(native)
    assert error.value.reason == reason


def test_masks_and_missing_sample_positions_reject_a_chunk():
    mask = np.ones(2000, dtype=bool)
    mask[999] = False
    with pytest.raises(ecg.InvalidChunk, match="calibration_or_missing"):
        ecg.preprocess_chunk(native_wave(), validity=mask)
    coordinates = np.arange(2000)
    coordinates[1000:] += 1
    with pytest.raises(ecg.InvalidChunk, match="noncontiguous"):
        ecg.preprocess_chunk(native_wave(), sample_positions=coordinates)


@pytest.mark.parametrize("timestamp,end", [(600, 120000), (630, 126000),
                                         (86430, 17286000), (630.005, 126001)])
def test_history_bounds_preserve_initial_daily_and_terminal_sample_coordinates(timestamp, end):
    bounds = ecg.chunk_bounds(timestamp)
    assert len(bounds) == 60
    assert bounds[0] == (end - 120000, end - 118000)
    assert bounds[-1] == (end - 2000, end)
    assert all(b - a == 2000 for a, b in bounds)


def test_coordinate_roundoff_is_distinguished_from_a_non_grid_decision():
    assert ecg.decision_sample(np.nextafter(630.005, 0), duration=630.005) == 126001
    for timestamp in (599.995, 630.001, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="sample|history"):
            ecg.chunk_bounds(timestamp)


def test_pooling_requires_sixty_finite_vectors_and_uses_float64_means():
    vectors = np.full((60, 1024), 1e8, dtype=np.float32)
    vectors[30] += 8
    vectors[31, 1] += 16
    expected = np.full(1024, 8 / 30, dtype=np.float32)
    expected[1] = 24 / 30
    result = ecg.pool_delta(vectors)
    assert result.dtype == np.float32
    # Subtracting two 1e8 float64 means adds <5e-8 cancellation error.
    np.testing.assert_allclose(result, expected, atol=5e-8, rtol=0)
    for broken in (vectors[:59], vectors[:, :1023], np.full((60, 1024), np.nan)):
        with pytest.raises(ValueError, match="60|finite|1024"):
            ecg.pool_delta(broken)


def test_pinned_source_rejects_substitution_before_executing(tmp_path):
    marker = tmp_path / "executed"
    fake = tmp_path / "net1d.py"
    fake.write_text(f"open({str(marker)!r}, 'w').write('bad')")
    with pytest.raises(ValueError, match="source.*SHA256"):
        ecg.build_encoder(fake)
    assert not marker.exists()


@pytest.mark.skipif(
    not SOURCE.exists(), reason="requires private research receipts not shipped in the public repo"
)
def test_exact_net1d_configuration_and_disabled_branches():
    torch = pytest.importorskip("torch")
    with torch.device("meta"):
        model = ecg.build_encoder(SOURCE)
    assert model.in_channels == 1 and model.return_features
    assert model.filter_list == [64, 160, 160, 400, 400, 1024, 1024]
    assert model.m_blocks_list == [2, 2, 2, 3, 3, 4, 4]
    assert model.dense.in_features == 1024
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert all(not m.use_bn and not m.use_do for m in model.modules() if hasattr(m, "use_bn"))


def tiny_encoder():
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(1, 1024)
            self.dense = torch.nn.Linear(1024, 1)

        def forward(self, values):
            features = self.backbone(values.mean(-1))
            return self.dense(features), features

    return Tiny()


def test_backbone_coverage_is_strict_with_only_a_checked_classifier_exception():
    torch = pytest.importorskip("torch")
    model = tiny_encoder()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["dense.weight"] = torch.ones(7, 1024)
    state["dense.bias"] = torch.ones(7)
    receipt = ecg.load_backbone_state(model, state)
    assert receipt["classifier_ignored"] == ["dense.bias", "dense.weight"]
    assert receipt["missing_keys"] == ["dense.weight", "dense.bias"]
    assert receipt["unexpected_keys"] == []
    np.testing.assert_array_equal(model.backbone.weight.detach(), state["backbone.weight"])
    assert ecg.load_backbone_state(model, {k: v for k, v in state.items() if not k.startswith("dense.")})


@pytest.mark.parametrize("kind", ["missing", "unexpected", "shape", "not_tensor", "classifier_extra",
                                 "classifier_shape", "classifier_partial", "nonfinite", "dtype"])
def test_backbone_validation_fails_closed(kind):
    torch = pytest.importorskip("torch")
    model = tiny_encoder()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    if kind == "missing":
        del state["backbone.bias"]
    elif kind == "unexpected":
        state["other.weight"] = torch.zeros(1)
    elif kind == "shape":
        state["backbone.weight"] = torch.zeros(1023, 1)
    elif kind == "not_tensor":
        state["backbone.bias"] = [0] * 1024
    elif kind == "classifier_extra":
        state["dense.extra"] = torch.zeros(1)
    elif kind == "classifier_shape":
        state["dense.weight"] = torch.zeros(2, 1023)
    elif kind == "classifier_partial":
        del state["dense.bias"]
    elif kind == "nonfinite":
        state["backbone.bias"][0] = float("inf")
    elif kind == "dtype":
        state["backbone.bias"] = state["backbone.bias"].long()
    with pytest.raises(ValueError):
        ecg.load_backbone_state(model, state)


def test_restricted_checkpoint_loading_freezes_model_and_records_byte_identity(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    model = tiny_encoder()
    path = tmp_path / "fixture.pth"
    torch.save({"state_dict": model.state_dict()}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(ecg, "CHECKPOINT_BYTES", path.stat().st_size)
    monkeypatch.setattr(ecg, "CHECKPOINT_SHA256", digest)
    monkeypatch.setattr(ecg, "build_encoder", lambda source: model)
    loaded, receipt = ecg.load_frozen_encoder(SOURCE, path)
    assert receipt["checkpoint_sha256"] == digest and receipt["weights_only"] is True
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    path.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="checkpoint"):
        ecg.load_frozen_encoder(SOURCE, path)


def test_restricted_loader_does_not_fall_back_to_unrestricted_pickle(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    path = tmp_path / "unsafe.pth"
    torch.save({"state_dict": tiny_encoder().state_dict(), "unsafe": Path("bad")}, path)
    monkeypatch.setattr(ecg, "CHECKPOINT_BYTES", path.stat().st_size)
    monkeypatch.setattr(ecg, "CHECKPOINT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(ecg, "build_encoder", lambda source: tiny_encoder())
    with pytest.raises(Exception, match="Weights only load failed|Unsupported global"):
        ecg.load_frozen_encoder(SOURCE, path)


def test_batched_inference_is_eval_only_finite_and_batch_consistent():
    torch = pytest.importorskip("torch")
    model = tiny_encoder().eval().requires_grad_(False)
    chunks = np.arange(9 * 5000, dtype=np.float32).reshape(9, 5000) / 5000
    before = {k: v.clone() for k, v in model.state_dict().items()}
    one = ecg.encode_chunks(model, chunks, batch_size=1)
    eight = ecg.encode_chunks(model, chunks, batch_size=8)
    assert eight.shape == (9, 1024) and eight.dtype == np.float32
    np.testing.assert_allclose(one, eight, atol=1e-5, rtol=1e-4)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    model.train()
    with pytest.raises(ValueError, match="frozen|eval"):
        ecg.encode_chunks(model, chunks)


def test_complete_history_is_chunk_local_causal_and_terminal_exact():
    t = np.arange(126002) / 200
    values = np.sin(2 * np.pi * 1.3 * t) + 0.05 * np.cos(2 * np.pi * 25 * t)
    original = ecg.preprocess_history(values, timestamp=630.005, first_sample=0)
    values[126001:] = np.nan
    np.testing.assert_array_equal(original, ecg.preprocess_history(values, timestamp=630.005))
    assert original.shape == (60, 5000)
    np.testing.assert_array_equal(original[0], ecg.preprocess_chunk(values[6001:8001]))
    np.testing.assert_array_equal(original[-1], ecg.preprocess_chunk(values[124001:126001]))


@pytest.mark.parametrize("daily_offset", [0, 17280000, 16812345])
def test_calibration_mask_uses_actual_daily_sample_offset(daily_offset):
    # Variable daily-file lengths must not be silently rounded to 86400 seconds.
    end = max(120000, daily_offset + 6000)
    start = end - 120000
    positions = np.arange(start, end)
    native = np.sin(positions / 200 * 2 * np.pi * 1.3)
    validity = ~((positions >= daily_offset) & (positions < daily_offset + 6000))
    with pytest.raises(ecg.InvalidChunk, match="calibration_or_missing"):
        ecg.preprocess_history(native, timestamp=end / 200, first_sample=start, validity=validity)
    if daily_offset == 0:
        next_native = np.sin(np.arange(6000, 126000) / 200 * 2 * np.pi * 1.3)
        assert ecg.preprocess_history(next_native, timestamp=630, first_sample=6000).shape == (60, 5000)


def test_history_rejects_incomplete_support_and_malformed_masks():
    with pytest.raises(ecg.InvalidChunk, match="native_support"):
        ecg.preprocess_history(np.zeros(119999), timestamp=600)
    with pytest.raises(ecg.InvalidChunk, match="calibration_or_missing"):
        ecg.preprocess_history(np.zeros(120000), timestamp=600, validity=np.ones(119999, dtype=bool))


def legacy_numpy_fixture(path, metadata):
    torch = pytest.importorskip("torch")
    torch.save({"state_dict": tiny_encoder().state_dict(), "val_auroc": metadata}, path)
    # The pinned 2024 pickle uses NumPy's old module path; NumPy2 writes _core.
    with zipfile.ZipFile(path) as source:
        entries = [(item, source.read(item.filename)) for item in source.infolist()]
    with zipfile.ZipFile(path, "w") as destination:
        for item, data in entries:
            if item.filename.endswith("/data.pkl"):
                data = data.replace(b"cnumpy._core.multiarray\nscalar\n", b"cnumpy.core.multiarray\nscalar\n")
            destination.writestr(item, data)


def fixture_identity(path, monkeypatch):
    monkeypatch.setattr(ecg, "CHECKPOINT_BYTES", path.stat().st_size)
    monkeypatch.setattr(ecg, "CHECKPOINT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(ecg, "build_encoder", lambda source: tiny_encoder())


def test_reviewed_numeric_numpy_metadata_loads_restricted_with_contextual_cleanup(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    path = tmp_path / "legacy.pth"
    legacy_numpy_fixture(path, np.float64(0.5))
    fixture_identity(path, monkeypatch)
    before = torch.serialization.get_safe_globals().copy()
    model, receipt = ecg.load_frozen_encoder(SOURCE, path)
    assert receipt["weights_only"] and receipt["numpy_metadata_allowlist"] == [
        "numpy.core.multiarray.scalar", "numpy.dtype", "numpy.dtypes.Float64DType"
    ]
    assert not model.training
    assert torch.serialization.get_safe_globals() == before


@pytest.mark.parametrize("metadata", [np.dtype("O"), Path("unsupported"), np.float32(0.5)])
def test_numpy_allowlist_rejects_object_dtype_unknown_globals_and_unobserved_numeric_types(
        metadata, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    path = tmp_path / "unsupported.pth"
    legacy_numpy_fixture(path, metadata)
    fixture_identity(path, monkeypatch)
    before = torch.serialization.get_safe_globals().copy()
    with pytest.raises(pickle.UnpicklingError):
        ecg.load_frozen_encoder(SOURCE, path)
    assert torch.serialization.get_safe_globals() == before


def test_optional_stage_timing_preserves_preprocessing_bits_and_accumulates_nonnegative_stages():
    native = native_wave()
    original = ecg.preprocess_chunk(native)
    timings = {}
    measured = ecg.preprocess_chunk(native, timings=timings)
    np.testing.assert_array_equal(measured, original)
    assert set(timings) == {"resample_seconds", "filter_seconds", "normalize_seconds"}
    assert all(np.isfinite(value) and value >= 0 for value in timings.values())
    first = timings.copy()
    np.testing.assert_array_equal(ecg.preprocess_chunk(native, timings=timings), original)
    assert all(timings[key] >= first[key] for key in timings)
