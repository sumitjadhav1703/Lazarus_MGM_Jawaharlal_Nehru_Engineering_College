"""Offline frozen ECGFounder adapter; no outcome metadata enters these functions."""

import hashlib
import math
import time
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy import signal

SOURCE_SHA256 = "832bf90e05e20e62db6f9c7007acf7f0b1f739938ab80fa9b2c1ad438e8da7f7"
LICENSE_SHA256 = "b872b97f0675d67a350880e22e21ffd9fb76dfcb8989f89d950bd3f49eca959e"
CHECKPOINT_SHA256 = "f863a38897fb49a27fec7e44008ea3c7bdbd29c77fa4a02ecbb8c56df4f37603"
CHECKPOINT_BYTES = 369807481


class InvalidChunk(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def decision_sample(timestamp, duration=None):
    """Accept sample-grid coordinates plus the existing duration-roundoff allowance."""
    if not math.isfinite(timestamp) or timestamp < 600:
        raise ValueError("A complete history and finite sample-grid decision are required")
    if duration is None:
        duration = timestamp
    if not math.isfinite(duration) or duration < timestamp:
        raise ValueError("Decision sample must be within the record duration")
    scaled = timestamp * 200
    nearest = round(scaled)
    tolerance = 200 * math.ulp(duration) + math.ulp(scaled)
    if abs(scaled - nearest) > tolerance:
        raise ValueError("Decision must lie on the native sample grid")
    return nearest


def chunk_bounds(timestamp, duration=None):
    end = decision_sample(timestamp, duration)
    return [(start, start + 2000) for start in range(end - 120000, end, 2000)]


def preprocess_chunk(native, *, validity=None, sample_positions=None, timings=None):
    """Process only one ended ten-second chunk; invalid support is never imputed."""
    native = np.asarray(native, dtype=np.float64)
    if native.shape != (2000,):
        raise InvalidChunk("native_support")
    if sample_positions is not None:
        positions = np.asarray(sample_positions)
        if (positions.shape != (2000,) or not np.issubdtype(positions.dtype, np.integer)
                or not np.all(np.diff(positions) == 1)):
            raise InvalidChunk("noncontiguous_native_support")
    if validity is not None:
        mask = np.asarray(validity)
        if mask.shape != native.shape or mask.dtype != np.bool_ or not mask.all():
            raise InvalidChunk("calibration_or_missing")
    if not np.isfinite(native).all():
        raise InvalidChunk("nonfinite_native")
    if native.max() == native.min():
        raise InvalidChunk("native_constant")
    started = time.monotonic()
    values = np.interp(np.arange(5000) / 500, np.arange(2000) / 200, native)
    if timings is not None:
        timings["resample_seconds"] = timings.get("resample_seconds", 0.) + time.monotonic() - started
    started = time.monotonic()
    b, a = signal.iirnotch(50, 30, fs=500)
    values = signal.filtfilt(b, a, values)
    b, a = signal.butter(4, [0.67, 40], btype="bandpass", fs=500)
    values = signal.filtfilt(b, a, values)
    values -= signal.medfilt(values, kernel_size=201)
    if timings is not None:
        timings["filter_seconds"] = timings.get("filter_seconds", 0.) + time.monotonic() - started
    started = time.monotonic()
    std = values.std()
    if not np.isfinite(values).all() or not np.isfinite(std):
        raise InvalidChunk("nonfinite_filtered")
    if std <= 1e-8:
        raise InvalidChunk("filtered_degenerate")
    values = ((values - values.mean()) / (std + 1e-8)).astype(np.float32)
    if not np.isfinite(values).all():
        raise InvalidChunk("nonfinite_normalized")
    if timings is not None:
        timings["normalize_seconds"] = timings.get("normalize_seconds", 0.) + time.monotonic() - started
    return values


def preprocess_history(native, *, timestamp, duration=None, first_sample=0, validity=None):
    """Direct parity reference using exact coordinates and at most 60 prepared chunks."""
    native = np.asarray(native)
    if (native.ndim != 1 or not isinstance(first_sample, (int, np.integer))
            or first_sample < 0):
        raise InvalidChunk("native_support")
    bounds = chunk_bounds(timestamp, duration)
    if bounds[0][0] < first_sample or bounds[-1][1] > first_sample + len(native):
        raise InvalidChunk("native_support")
    if validity is not None:
        validity = np.asarray(validity)
        if validity.shape != native.shape or validity.dtype != np.bool_:
            raise InvalidChunk("calibration_or_missing")
    prepared = np.empty((60, 5000), dtype=np.float32)
    for index, (start, end) in enumerate(bounds):
        section = slice(start - first_sample, end - first_sample)
        prepared[index] = preprocess_chunk(native[section],
                                           validity=None if validity is None else validity[section])
    return prepared


def pool_delta(embeddings):
    embeddings = np.asarray(embeddings)
    if embeddings.shape != (60, 1024) or embeddings.dtype != np.float32:
        raise ValueError("Expected 60 float32 vectors of width 1024")
    if not np.isfinite(embeddings).all():
        raise ValueError("All 60 vectors must be finite")
    delta = (embeddings[30:].mean(axis=0, dtype=np.float64)
             - embeddings[:30].mean(axis=0, dtype=np.float64)).astype(np.float32)
    if not np.isfinite(delta).all():
        raise ValueError("Pooled delta must be finite")
    return delta


def build_encoder(source_path):
    """Execute exactly the hash-verified standalone upstream source, avoiding pyc reuse."""
    source_path = Path(source_path)
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("ECGFounder source SHA256 mismatch")
    if hashlib.sha256(source_path.with_name("LICENSE").read_bytes()).hexdigest() != LICENSE_SHA256:
        raise ValueError("ECGFounder LICENSE SHA256 mismatch")
    module = ModuleType("_cardiotwin_pinned_ecgfounder_net1d")
    module.__file__ = str(source_path)
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    model = module.Net1D(
        in_channels=1, base_filters=64, ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2,
        groups_width=16, n_classes=1, use_bn=False, use_do=False,
        return_features=True, verbose=False,
    )
    return model.eval().requires_grad_(False)


def load_backbone_state(model, state):
    """Require every backbone tensor; only the original classifier can be omitted."""
    import torch

    if not isinstance(state, Mapping) or any(not isinstance(key, str) for key in state):
        raise ValueError("Checkpoint state_dict must be a string-keyed mapping")
    expected = model.state_dict()
    classifier = {"dense.weight", "dense.bias"}
    backbone = set(expected) - classifier
    supplied_classifier = set(state) & classifier
    if set(state) - classifier != backbone or supplied_classifier not in (set(), classifier):
        raise ValueError("Backbone coverage or classifier keys mismatch")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Checkpoint value is not a tensor: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {key}")
        if key in backbone and (value.shape != expected[key].shape or value.dtype != expected[key].dtype):
            raise ValueError(f"Backbone shape/dtype mismatch: {key}")
    if supplied_classifier:
        weight, bias = state["dense.weight"], state["dense.bias"]
        if (weight.ndim != 2 or weight.shape[0] < 1 or weight.shape[1] != 1024
                or bias.shape != (weight.shape[0],) or not weight.is_floating_point()
                or not bias.is_floating_point()):
            raise ValueError("Classifier shape/dtype mismatch")
    result = model.load_state_dict({key: state[key] for key in state if key in backbone}, strict=False)
    if set(result.missing_keys) != classifier or result.unexpected_keys:
        raise ValueError("Unexpected backbone loading omissions")
    return dict(backbone_tensors=len(backbone), classifier_ignored=sorted(supplied_classifier),
                missing_keys=result.missing_keys, unexpected_keys=result.unexpected_keys)


def load_frozen_encoder(source_path, checkpoint_path, *, device="cpu"):
    import torch

    checkpoint_path = Path(checkpoint_path)
    # Hash and deserialize the same open file to avoid path-replacement races.
    with checkpoint_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        size = handle.tell()
        if size != CHECKPOINT_BYTES or digest != CHECKPOINT_SHA256:
            raise ValueError("ECGFounder checkpoint size/SHA256 mismatch")
        handle.seek(0)
        # The hash-bound checkpoint has one legacy NumPy float64 val_auroc scalar.
        allowlist = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
                     np.dtype, np.dtypes.Float64DType]
        with torch.serialization.safe_globals(allowlist):
            checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain state_dict")
    model = build_encoder(source_path)
    if model.dense.in_features != 1024:
        raise ValueError("Encoder preclassifier width must be 1024")
    loading = load_backbone_state(model, checkpoint["state_dict"])
    model = model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    return model, dict(checkpoint_sha256=digest, checkpoint_bytes=size, source_sha256=SOURCE_SHA256,
                       license_sha256=LICENSE_SHA256, weights_only=True, device=str(device),
                       dtype="float32", numpy_metadata_allowlist=["numpy.core.multiarray.scalar",
                       "numpy.dtype", "numpy.dtypes.Float64DType"], **loading)


def encode_chunks(model, chunks, *, batch_size=8):
    import torch

    chunks = np.asarray(chunks)
    if (chunks.ndim != 2 or chunks.shape[1] != 5000 or chunks.dtype != np.float32
            or not np.isfinite(chunks).all()):
        raise ValueError("Network chunks must be finite float32 [n,5000]")
    if batch_size not in (1, 8):
        raise ValueError("Only batch8 execution or batch1 numerical checking is declared")
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Encoder must be frozen in eval mode")
    parameter = next(model.parameters())
    if parameter.dtype != torch.float32:
        raise ValueError("Encoder must use float32")
    result = np.empty((len(chunks), 1024), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(chunks), batch_size):
            batch = torch.from_numpy(chunks[start:start + batch_size].copy()).unsqueeze(1)
            _, vectors = model(batch.to(parameter.device))
            if (vectors.shape != (len(batch), 1024) or vectors.dtype != torch.float32
                    or not torch.isfinite(vectors).all()):
                raise ValueError("Encoder must return finite float32 width1024 vectors")
            result[start:start + len(batch)] = vectors.cpu().numpy()
    return result
