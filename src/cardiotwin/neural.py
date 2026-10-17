"""Optional small causal sequence challengers; PyTorch is not a serving dependency.

The network receives ONLY a completed trailing history. Masks retain missing
intervals. Normalization is fitted on fitting patients, never calibration/test.
"""

import functools
import hashlib
import random
import sqlite3
import threading
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np

from .progress import Progress

try:
    import torch
    from torch import nn
except ImportError as error:
    raise ImportError("Neural experiments require the optional cardiotwin[neural] dependencies") from error


class CausalConv1d(nn.Conv1d):
    def forward(self, x):
        padding = (self.kernel_size[0] - 1) * self.dilation[0]
        return super().forward(torch.nn.functional.pad(x, (padding, 0)))


class CausalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            CausalConv1d(width, width, 3, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(0.1),
            CausalConv1d(width, width, 3, dilation=dilation),
            nn.ReLU(),
        )

    def forward(self, values):
        return values + self.layers(values)


class RRTCN(nn.Module):
    """RR, adjacent delta, elapsed seconds, validity; receptive field >=2400 beats."""

    def __init__(self, width=16):
        super().__init__()
        if width not in (16, 32):
            raise ValueError("Prespecified RR widths are 16 and 32")
        self.input = nn.Conv1d(4, width, 1)
        self.blocks = nn.Sequential(*(CausalBlock(width, 2**i) for i in range(10)))
        self.output = nn.Linear(width, 1)

    def forward(self, x):
        hidden = self.blocks(self.input(x))
        return self.output(hidden[:, :, -1]).squeeze(-1)


class WaveformCNN(nn.Module):
    """Shared five-second two-lead/mask encoder followed by a causal temporal head.

    No sample after the decision time is read. Each chunk is complete before its
    temporal token is emitted; the CNN need not be sample-causal inside the chunk.
    """

    def __init__(self, width=16, sample_rate=100, chunk_seconds=5):
        super().__init__()
        self.chunk_samples = sample_rate * chunk_seconds
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 8, 9, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(8, width, 7, stride=4, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.temporal = nn.Sequential(*(CausalBlock(width, 2**i) for i in range(6)))
        self.output = nn.Linear(width, 1)

    def forward(self, x):
        batch, channels, samples = x.shape
        if channels != 3 or samples % self.chunk_samples:
            raise ValueError("Waveform input requires two leads, mask, and complete five-second chunks")
        chunks = samples // self.chunk_samples
        values = x.reshape(batch, channels, chunks, self.chunk_samples).permute(0, 2, 1, 3)
        encoded = self.encoder(values.reshape(batch * chunks, channels, self.chunk_samples))
        tokens = encoded.reshape(batch, chunks, -1).transpose(1, 2)
        return self.output(self.temporal(tokens)[:, :, -1]).squeeze(-1)


class WaveformResNet(WaveformCNN):
    """Residual morphology encoder with the unchanged completed-chunk causal head."""

    def __init__(self, width=16):
        from .neural_architectures import ResidualChunkEncoder

        super().__init__(width=width)
        self.encoder = ResidualChunkEncoder(width)


def rr_sequence(signal, fs, history_seconds=600, max_intervals=2400):
    from .features import _peaks

    lead = np.asarray(signal)[:, 0]
    peaks = _peaks(lead, fs)
    rr = np.diff(peaks)
    indices = np.minimum(np.rint(peaks * fs).astype(int), len(lead) - 1)
    missing = np.r_[0, np.cumsum(~np.isfinite(lead))]
    valid = (rr >= 0.25) & (rr <= 2.5)
    valid &= missing[indices[1:] + 1] == missing[indices[:-1]]
    delta = np.zeros_like(rr)
    if len(rr) > 1:
        pairs = valid[:-1] & valid[1:]
        delta[1:] = np.where(pairs, np.diff(rr), 0)
    values = np.vstack((np.where(valid, rr, 0), delta, peaks[1:] / history_seconds, valid.astype(float)))[
        :, -max_intervals:
    ]
    result = np.zeros((4, max_intervals), dtype=np.float32)
    if values.shape[1]:
        result[:, -values.shape[1] :] = values
    return result


def waveform_sequence(signal, fs, history_seconds=600, sample_rate=100):
    """Causal anti-alias lowpass and sample inside the completed trailing history."""
    signal = np.asarray(signal)
    if signal.ndim != 2 or signal.shape[1] < 2:
        raise ValueError("Waveform challenger requires two physical leads")
    if fs > sample_rate:
        from scipy.signal import butter, sosfilt

        finite = np.isfinite(signal[:, :2])
        for lead in range(2):
            present = finite[:, lead]
            prior = np.maximum.accumulate(np.where(present, np.arange(len(signal)), 0))
            clean = np.where(present[prior], signal[prior, lead], 0.0)
            if lead == 0:
                filtered = np.empty_like(signal[:, :2], dtype=float)
            filtered[:, lead] = sosfilt(butter(4, 0.45 * sample_rate, fs=fs, output="sos"), clean)
        signal = np.where(finite, filtered, np.nan)
    count = int(history_seconds * sample_rate)
    indices = np.floor(np.arange(count) * fs / sample_rate).astype(int)
    if not len(signal) or indices[-1] >= len(signal):
        raise ValueError("Incomplete trailing history")
    values = signal[indices, :2].T.astype(np.float32)
    valid = np.isfinite(values).all(axis=0)
    # Missing amplitudes remain explicit zeros with a shared validity channel.
    values[:, ~valid] = 0
    return np.vstack((values, valid.astype(np.float32)))


class SequenceNormalizer:
    def fit(self, arrays, deadline=None, total=None):
        sums = squares = counts = None
        with Progress("neural.normalization", total=total, device="cpu") as progress:
            for completed, array in enumerate(arrays, 1):
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError("Sequence normalization budget exhausted")
                array = np.asarray(array, dtype=np.float64)
                if not np.isfinite(array).all() or array.ndim != 2:
                    raise ValueError("Sequence tensors must be finite channel-by-time arrays")
                values = array[:-1, array[-1] > 0]
                if sums is None:
                    sums, squares, counts = np.zeros(len(array) - 1), np.zeros(len(array) - 1), 0
                sums += values.sum(axis=1)
                squares += (values**2).sum(axis=1)
                counts += values.shape[1]
                progress.update(completed)
            if counts is None or counts == 0:
                raise ValueError("No valid fitting-history samples for normalization")
            self.mean = sums / counts
            self.scale = np.sqrt(np.maximum(squares / counts - self.mean**2, 1e-8))
        return self

    def transform(self, array):
        values = np.asarray(array, dtype=np.float32).copy()
        values[:-1] = (values[:-1] - self.mean[:, None]) / self.scale[:, None]
        values[:-1, values[-1] <= 0] = 0
        return values.astype(np.float32)


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, rows, provider, normalizer, weights=None):
        self.rows, self.provider, self.normalizer = rows, provider, normalizer
        self.weights = (
            np.ones(len(rows), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self._sample(index, self.provider(row))

    def _sample(self, index, sequence):
        row = self.rows[index]
        return (
            torch.from_numpy(self.normalizer.transform(sequence)),
            torch.tensor(float(row.get("label") or 0)),
            torch.tensor(self.weights[index]),
        )

    def __getitems__(self, indices):
        batch = getattr(self.provider, "batch", None)
        if batch is None:
            return [self[index] for index in indices]
        arrays = batch([self.rows[index] for index in indices])
        return [self._sample(index, array) for index, array in zip(indices, arrays, strict=True)]


def record_sequence_provider(
    records,
    config,
    representation="rr",
    cache_path=None,
    source_contract=None,
    max_cache_bytes=None,
    deadline=None,
):
    """On-demand reads avoid duplicating all overlapping waveforms on disk."""
    from .data import cached_signal_reader

    indexed = {(r.patient_id, r.record_id): r for r in records}
    history = config["target"]["history_seconds"]
    database = None
    path = None
    diagnostics = dict(
        limit_bytes=max_cache_bytes,
        bytes=0,
        pending_bytes=0,
        cap_hit=False,
        writes=0,
        errors=[],
        closed=False,
    )

    def close_database():
        try:
            database.close()
        except sqlite3.Error as error:
            diagnostics["errors"].append(str(error))

    if max_cache_bytes is not None and max_cache_bytes < 0:
        raise ValueError("Sequence cache byte limit must be nonnegative")
    if cache_path is not None and representation == "rr":
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            database = sqlite3.connect(path)
            database.execute("PRAGMA journal_mode=WAL")
            database.execute("PRAGMA synchronous=NORMAL")
            database.execute(
                "CREATE TABLE IF NOT EXISTS sequences (key TEXT PRIMARY KEY, data BLOB NOT NULL)"
            )
        except sqlite3.Error as error:
            diagnostics["errors"].append(str(error))
            if database is not None:
                close_database()
            database = None
    import json

    from .features import FEATURE_VERSION

    if source_contract is None:
        from .investigation import _record_contract

        source_contract = [_record_contract(r) for r in records]
    content = {
        "config": config,
        "feature_version": FEATURE_VERSION,
        "representation": representation + "-sequence-v1",
        "source": source_contract,
    }
    contract = hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if database is not None:
        try:
            database.execute("CREATE TABLE IF NOT EXISTS contract (identity TEXT NOT NULL)")
            prior = database.execute("SELECT identity FROM contract").fetchone()
            if prior is not None and prior[0] != contract:
                close_database()
                raise ValueError("Sequence cache source/representation contract differs")
            if prior is None:
                database.execute("INSERT INTO contract VALUES (?)", (contract,))
                database.commit()
        except sqlite3.Error as error:
            diagnostics["errors"].append(str(error))
            close_database()
            database = None

    def cache_bytes():
        total = 0
        if path is not None:
            for suffix in ("", "-wal", "-shm"):
                try:
                    total += Path(str(path) + suffix).stat().st_size
                except FileNotFoundError:
                    pass
        return total

    diagnostics["bytes"] = cache_bytes()
    read_enabled = database is not None
    write_enabled = database is not None
    page_size = database.execute("PRAGMA page_size").fetchone()[0] if database is not None else 4096
    if max_cache_bytes is not None and diagnostics["bytes"] >= max_cache_bytes:
        diagnostics["cap_hit"] = True
        write_enabled = False

    def cache_error(error):
        nonlocal write_enabled, read_enabled
        diagnostics["errors"].append(str(error))
        write_enabled = False
        try:
            database.rollback()
        except sqlite3.Error as rollback_error:
            diagnostics["errors"].append(str(rollback_error))
            read_enabled = False
        diagnostics["pending_bytes"] = 0

    def commit():
        try:
            database.commit()
            diagnostics["pending_bytes"] = 0
            diagnostics["bytes"] = cache_bytes()
        except sqlite3.Error as error:
            cache_error(error)

    reader_session = cached_signal_reader()
    read_signal = reader_session.__enter__()
    owner = threading.get_ident()
    executor = (
        ThreadPoolExecutor(max_workers=4, thread_name_prefix="cardiotwin-waveform")
        if representation == "waveform"
        else None
    )

    @functools.lru_cache(maxsize=4096 if representation == "rr" else 64)
    def extract(patient, identifier, t):
        nonlocal read_enabled, write_enabled
        key = hashlib.sha256(f"{contract}:{patient}:{identifier}:{t}".encode()).hexdigest()
        cached = None
        if read_enabled:
            try:
                cached = database.execute("SELECT data FROM sequences WHERE key=?", (key,)).fetchone()
            except sqlite3.Error as error:
                read_enabled = False
                cache_error(error)
            if cached:
                return np.frombuffer(zlib.decompress(cached[0]), dtype=np.float32).reshape(4, 2400).copy()
        record = indexed[(patient, identifier)]
        signal = read_signal(record, t - history, t)
        values = (
            rr_sequence(signal, record.fs, history)
            if representation == "rr"
            else waveform_sequence(signal, record.fs, history)
        )
        if write_enabled:
            compressed = zlib.compress(values.tobytes(), 1)
            # Reserve B-tree/overflow pages and both DB/WAL copies until the next measured commit.
            growth = 2 * ((len(compressed) + 192 + page_size - 1) // page_size + 1) * page_size
            if max_cache_bytes is not None and (
                diagnostics["bytes"] + diagnostics["pending_bytes"] + growth > max_cache_bytes
            ):
                diagnostics["cap_hit"] = True
                write_enabled = False
                commit()
            else:
                try:
                    database.execute("INSERT OR REPLACE INTO sequences VALUES (?, ?)", (key, compressed))
                    diagnostics["writes"] += 1
                    diagnostics["pending_bytes"] += growth
                    if diagnostics["writes"] % 256 == 0:
                        commit()
                except sqlite3.Error as error:
                    cache_error(error)
        return values

    def checked_time(row):
        if diagnostics["closed"]:
            raise RuntimeError("Sequence provider is closed")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Sequence extraction budget exhausted")
        t = float(row["timestamp"])
        if t < history:
            raise ValueError("Sequence extraction requires complete trailing history")
        return t

    def provide(row):
        t = checked_time(row)
        return extract(row["patient_id"], row["record_id"], t)

    def batch(rows):
        if diagnostics["closed"]:
            raise RuntimeError("Sequence provider is closed")
        if threading.get_ident() != owner:
            raise ValueError("Waveform batch reads require the provider's owning thread")
        if len(rows) > 64:
            raise ValueError("Waveform extraction batches are limited to 64 histories")
        histories = []
        for row in rows:
            t = checked_time(row)
            record = indexed[(row["patient_id"], row["record_id"])]
            histories.append((read_signal(record, t - history, t), record.fs))
        return list(executor.map(lambda item: waveform_sequence(item[0], item[1], history), histories))

    def close():
        if diagnostics["closed"]:
            return
        try:
            if database is not None:
                commit()
                close_database()
        finally:
            diagnostics["closed"] = True
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            extract.cache_clear()
            reader_session.__exit__(None, None, None)

    def cache_diagnostics():
        return dict(diagnostics, bytes=cache_bytes(), errors=list(diagnostics["errors"]))

    provide.close = close
    provide.cache_diagnostics = cache_diagnostics
    if executor is not None:
        provide.batch = batch
    return provide


def _device_details(device):
    details = {"device": str(device)}
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        details["cuda_allocated_bytes"] = torch.cuda.memory_allocated(device)
    return details


def _fit_network(
    fit_rows,
    early_stop_rows,
    provider,
    representation="rr",
    width=16,
    seed=42,
    max_epochs=30,
    patience=5,
    batch_size=64,
    max_seconds=7200,
    device=None,
    early_stop_weighting="patient_only",
):
    """Early-stop rows come from fitting-pool groups; calibration is never reused."""
    from sklearn.metrics import log_loss

    from .train import patient_macro_log_loss, training_weights

    if early_stop_weighting not in ("patient_only", "patient_class"):
        raise ValueError("Early-stop weighting must be patient_only or patient_class")

    models = {"rr": RRTCN, "waveform": WaveformCNN, "waveform_resnet": WaveformResNet}
    if representation not in models:
        raise ValueError(f"Unknown neural representation: {representation}")

    fit_rows = [r for r in fit_rows if r["accepted"] and r["label"] in (0, 1)]
    stop_rows = [r for r in early_stop_rows if r["accepted"] and r["label"] in (0, 1)]

    def groups(rows):
        return {r.get("group_id", r["patient_id"]) for r in rows}

    if groups(fit_rows) & groups(stop_rows) or not fit_rows or not stop_rows:
        raise ValueError("Neural fitting and early-stopping groups must be nonempty and disjoint")
    if {r["label"] for r in fit_rows} != {0, 1}:
        raise ValueError("Neural fitting requires both classes")
    if {r["label"] for r in stop_rows} != {0, 1}:
        raise ValueError("Neural early stopping requires both classes")
    stop_labels = [r["label"] for r in stop_rows]
    stop_groups = [r.get("group_id", r["patient_id"]) for r in stop_rows]
    stop_weights = training_weights(stop_labels, stop_groups, early_stop_weighting)
    started = time.monotonic()
    deadline = started + max_seconds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = device or (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    normalizer = SequenceNormalizer().fit((provider(r) for r in fit_rows), deadline, total=len(fit_rows))
    weights = training_weights(
        [r["label"] for r in fit_rows],
        [r.get("group_id", r["patient_id"]) for r in fit_rows],
        "patient_class",
    )
    fitting = SequenceDataset(fit_rows, provider, normalizer, weights)
    stopping = SequenceDataset(stop_rows, provider, normalizer)
    generator = torch.Generator().manual_seed(seed)
    cuda_transfer = torch.device(device).type == "cuda"
    # SQLite/HDF5 providers retain their owner thread; CUDA pinning is CPU-side.
    fit_loader = torch.utils.data.DataLoader(
        fitting,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=cuda_transfer,
        num_workers=0,
    )
    stop_loader = torch.utils.data.DataLoader(
        stopping, batch_size=batch_size, pin_memory=cuda_transfer, num_workers=0
    )
    model = models[representation](width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss, best_state, bad_epochs, history = float("inf"), None, 0, []
    best_epoch = None
    normalization_seconds = time.monotonic() - started
    for epoch in range(min(30, max_epochs)):
        if time.monotonic() - started > max_seconds:
            break
        model.train()
        training_sum, mass = 0.0, 0.0
        with Progress(
            "neural.train_epoch",
            total=len(fit_loader),
            epoch=epoch + 1,
            **_device_details(next(model.parameters()).device),
        ) as progress:
            for batch, (x, y, weight) in enumerate(fit_loader, 1):
                x, y, weight = (value.to(device, non_blocking=cuda_transfer) for value in (x, y, weight))
                optimizer.zero_grad(set_to_none=True)
                loss = (
                    nn.functional.binary_cross_entropy_with_logits(model(x), y, reduction="none") * weight
                ).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                training_sum += float(loss.detach()) * len(y)
                mass += len(y)
                progress.update(batch, windows=int(mass))
                if time.monotonic() - started > max_seconds:
                    break
            progress.finish(
                training_loss=training_sum / max(1, mass),
                **_device_details(next(model.parameters()).device),
            )
        model.eval()
        validation_sum, count = 0.0, 0
        validation_risks = []
        with (
            Progress(
                "neural.validate_epoch",
                total=len(stop_loader),
                epoch=epoch + 1,
                **_device_details(next(model.parameters()).device),
            ) as progress,
            torch.no_grad(),
        ):
            for batch, (x, y, _) in enumerate(stop_loader, 1):
                if time.monotonic() > deadline:
                    raise TimeoutError("Neural early-stopping evaluation budget exhausted")
                logits = model(x.to(device, non_blocking=cuda_transfer))
                validation_sum += float(
                    nn.functional.binary_cross_entropy_with_logits(
                        logits, y.to(device, non_blocking=cuda_transfer), reduction="sum"
                    )
                )
                count += len(y)
                validation_risks.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                progress.update(batch, windows=count)
            natural_loss = patient_macro_log_loss(stop_labels, validation_risks, stop_groups)
            validation_loss = (
                natural_loss
                if early_stop_weighting == "patient_only"
                else float(log_loss(stop_labels, validation_risks, sample_weight=stop_weights, labels=[0, 1]))
            )
            progress.finish(
                early_stop_loss=validation_loss,
                early_stop_group_macro_natural_loss=natural_loss,
                early_stop_weighting=early_stop_weighting,
                **_device_details(next(model.parameters()).device),
            )
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": training_sum / max(1, mass),
                "early_stop_loss": validation_loss,
                "early_stop_group_macro_natural_loss": natural_loss,
                "early_stop_window_mean_loss": validation_sum / max(1, count),
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss, bad_epochs = validation_loss, 0
            best_epoch = epoch + 1
            best_state = deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    if best_state is None:
        raise TimeoutError("No neural epoch completed within budget")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return (
        model,
        normalizer,
        {
            "representation": representation,
            "width": width,
            "seed": seed,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "loss": "patient-class weighted BCE",
            "scheduler": "none; early stopping",
            "batch_size": batch_size,
            "epochs": len(history),
            "best_epoch": best_epoch,
            "early_stop_weighting": early_stop_weighting,
            "early_stop_metric": (
                "independent-group macro log loss"
                if early_stop_weighting == "patient_only"
                else "independent-group class-balanced log loss"
            ),
            "fit_positive_windows": sum(r["label"] for r in fit_rows),
            "early_stop_positive_windows": sum(stop_labels),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "wall_seconds": time.monotonic() - started,
            "device": device,
            "torch_version": torch.__version__,
            "normalization_seconds": normalization_seconds,
            "epoch_history": history,
            "fit_groups": sorted(groups(fit_rows)),
            "early_stop_groups": sorted(groups(stop_rows)),
            "normalization": "fitting-history valid samples only",
        },
    )


def fit_network(
    fit_rows,
    early_stop_rows,
    provider,
    representation="rr",
    width=16,
    seed=42,
    max_epochs=30,
    patience=5,
    batch_size=64,
    max_seconds=7200,
    device=None,
    early_stop_weighting="patient_only",
):
    """Retry only measured GPU/CPU out-of-memory, recording each reduced batch."""
    import gc

    started, retries = time.monotonic(), []
    with Progress(
        "neural.fit", representation=representation, width=width, requested_device=str(device or "auto")
    ) as progress:
        while True:
            remaining = max_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Neural budget exhausted during OOM retries")
            try:
                model, normalizer, report = _fit_network(
                    fit_rows,
                    early_stop_rows,
                    provider,
                    representation,
                    width,
                    seed,
                    max_epochs,
                    patience,
                    batch_size,
                    remaining,
                    device,
                    early_stop_weighting,
                )
                report["oom_retries"] = retries
                report["wall_seconds_including_retries"] = time.monotonic() - started
                progress.finish(epochs=report["epochs"], **_device_details(next(model.parameters()).device))
                return model, normalizer, report
            except RuntimeError as error:
                if "out of memory" not in str(error).lower() or batch_size <= 1:
                    raise
                retries.append({"failed_batch_size": batch_size, "reason": str(error)})
                error.__traceback__ = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                batch_size = max(1, batch_size // 2)
                progress.update(force=True, oom_retries=len(retries), batch_size=batch_size)


def network_scores(model, normalizer, rows, provider, batch_size=64, deadline=None):
    available = [r for r in rows if r["accepted"]]
    device = next(model.parameters()).device
    cuda_transfer = device.type == "cuda"
    loader = torch.utils.data.DataLoader(
        SequenceDataset(available, provider, normalizer),
        batch_size=batch_size,
        pin_memory=cuda_transfer,
        num_workers=0,
    )
    scores = []
    model.eval()
    with (
        Progress(
            "neural.scores", total=len(loader), windows_total=len(available), **_device_details(device)
        ) as progress,
        torch.no_grad(),
    ):
        for batch, (x, _, _) in enumerate(loader, 1):
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("Neural prediction budget exhausted")
            scores.extend(model(x.to(device, non_blocking=cuda_transfer)).cpu().numpy().tolist())
            progress.update(batch, windows=len(scores))
        progress.finish(**_device_details(device))
    return np.asarray(scores, dtype=float)


def run_neural_ladder(
    rows,
    events,
    records,
    config,
    output,
    max_hours=6,
    seed=42,
    max_training_windows=10000,
    screening_epochs=8,
    specs=None,
    source_contract=None,
    cache_contract=None,
    max_sequence_cache_bytes=4_000_000_000,
    sequence_cache_path=None,
    early_stop_weighting="patient_only",
):
    """Bounded optional neural screening on the SAME three outer patient folds.

    Training subsampling is an explicit compute tradeoff. Policy/calibration and
    outer monitoring remain complete. A timeout preserves failed-run evidence;
    incomplete outer coverage cannot win. Full 30-epoch repeats follow only after
    a complete screening advantage is established, using a fresh output directory.
    """
    from itertools import chain

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .evaluation import _rates, evaluate_stream, probability_metrics
    from .investigation import (
        _compact_prediction,
        _digest,
        _events,
        _extraction_contract,
        _fresh,
        _group,
        _log_writer,
        _partitions,
        _pool_fold,
        _subset,
        _usable,
        _write,
        _write_predictions,
        select_policy,
    )

    if early_stop_weighting not in ("patient_only", "patient_class"):
        raise ValueError("Early-stop weighting must be patient_only or patient_class")
    specs = specs or [("rr", 16), ("waveform", 16), ("rr", 32)]
    names = {"rr": "rr_tcn", "waveform": "waveform_cnn", "waveform_resnet": "waveform_resnet"}
    if any(representation not in names or width not in (16, 32) for representation, width in specs):
        raise ValueError("Unknown neural candidate representation or width")
    if len(set(specs)) != len(specs):
        raise ValueError("Neural candidate specs must be unique to preserve fold artifacts")
    output = _fresh(output)
    rr_cache_path = (
        Path(sequence_cache_path) if sequence_cache_path is not None else output / "rr-sequences.sqlite"
    )
    log = _log_writer(output)
    started, deadline = time.monotonic(), time.monotonic() + max_hours * 3600
    all_groups = np.array(sorted({_group(row) for row in rows}))
    event_groups = {_group(event) for event in events}
    statuses = np.array([int(group in event_groups) for group in all_groups])
    if len(all_groups) < 12:
        raise ValueError("Neural outer validation requires twelve development groups")
    folds = list(
        StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(
            np.zeros(len(all_groups)), statuses, all_groups
        )
    )
    results, predictions, sequence_caches = [], {}, []

    def screen(sample_rows, maximum, random_seed):
        usable = _usable(sample_rows)
        if len(usable) <= maximum:
            return usable
        # Equal independent-group allocation preserves every fitting group.
        rng = np.random.default_rng(random_seed)
        groups = sorted({_group(r) for r in usable})
        limit = max(2, maximum // len(groups))
        retained = []
        for group in groups:
            group_rows = [r for r in usable if _group(r) == group]
            for label in (0, 1):
                cell = [r for r in group_rows if r["label"] == label]
                count = min(len(cell), max(1, limit // 2))
                retained.extend(cell[i] for i in rng.choice(len(cell), count, replace=False))
        return retained

    exhausted = False
    for representation, width in specs:
        name = f"{names[representation]}_{width}"
        with Progress("neural.candidate", total=len(folds), model=name) as candidate_progress:
            provider = record_sequence_provider(
                records,
                config,
                "rr" if representation == "rr" else "waveform",
                rr_cache_path if representation == "rr" else None,
                source_contract,
                max_cache_bytes=max_sequence_cache_bytes,
                deadline=deadline,
            )
            try:
                for fold, (training, held) in enumerate(folds):
                    if time.monotonic() >= deadline:
                        exhausted = True
                        break
                    fit, cal, policy = _partitions(_subset(rows, set(all_groups[training])), seed + fold)
                    # An independent group inside the fitting pool supplies early stopping.
                    fitting_rows = _subset(rows, fit)
                    inner_fit, inner_stop, _ = _partitions(fitting_rows, seed + fold + 100)
                    # Merge two fitting-pool roles, preserving holdout stop groups.
                    inner_training = set(fit) - set(inner_stop)
                    fit_rows = screen(_subset(rows, inner_training), max_training_windows, seed + fold)
                    stop_rows = _usable(_subset(rows, inner_stop))
                    cal_rows, policy_rows, outer_rows = [
                        _subset(rows, group) for group in (cal, policy, all_groups[held])
                    ]
                    trial = {
                        "experiment_id": f"seed-{seed}/fold-{fold}/{name}",
                        "model": name,
                        "fold": fold,
                        "seed": seed,
                        "features": representation,
                        "preprocessing": "fit-only valid-channel normalization; completed trailing history",
                        "augmentation": "none",
                        "validation": "three_outer_grouped_folds",
                        "screening_max_training_windows": max_training_windows,
                        "fit_windows": len(fit_rows),
                        "early_stop_windows": len(stop_rows),
                        "early_stop_weighting": early_stop_weighting,
                        "calibration_groups": cal,
                        "policy_groups": policy,
                        "outer_groups": all_groups[held].tolist(),
                    }
                    began = time.monotonic()
                    fold_progress = Progress(
                        "neural.fold",
                        total=1,
                        model=name,
                        fold=fold,
                        fit_windows=len(fit_rows),
                        early_stop_windows=len(stop_rows),
                        fit_positive_windows=sum(r["label"] for r in fit_rows),
                        early_stop_positive_windows=sum(r["label"] for r in stop_rows),
                    )
                    try:
                        model, normalizer, details = fit_network(
                            fit_rows,
                            stop_rows,
                            provider,
                            representation,
                            width,
                            seed + fold,
                            max_epochs=screening_epochs,
                            max_seconds=max(1, deadline - time.monotonic()),
                            early_stop_weighting=early_stop_weighting,
                        )
                        cal_usable = _usable(cal_rows)
                        with Progress(
                            "neural.calibration_scoring",
                            model=name,
                            fold=fold,
                            **_device_details(next(model.parameters()).device),
                        ):
                            scores = network_scores(
                                model, normalizer, cal_usable, provider, deadline=deadline
                            )
                        with Progress("neural.calibration", model=name, fold=fold, device="cpu") as progress:
                            labels = np.array([r["label"] for r in cal_usable])
                            if set(labels) != {0, 1}:
                                raise ValueError("Neural calibration requires both classes")
                            calibrator = make_pipeline(
                                StandardScaler(),
                                LogisticRegression(
                                    C=config["training"]["calibration_c"], random_state=seed, max_iter=1000
                                ),
                            )
                            calibrator.fit(scores[:, None], labels)
                            calibration_diagnostics = {
                                "positive_windows": int(labels.sum()),
                                "windows": len(labels),
                                "standardized_score_slope": float(calibrator[1].coef_[0, 0]),
                                "raw_score_slope": float(calibrator[1].coef_[0, 0] / calibrator[0].scale_[0]),
                                "ranking_reversed": bool(calibrator[1].coef_[0, 0] < 0),
                                "raw_score_min": float(scores.min()),
                                "raw_score_max": float(scores.max()),
                            }
                            progress.finish(**calibration_diagnostics)

                        def predicted(whole_rows):
                            scores = network_scores(
                                model, normalizer, whole_rows, provider, deadline=deadline
                            )
                            risks = iter(calibrator.predict_proba(scores[:, None])[:, 1])
                            return [
                                dict(
                                    _compact_prediction(row),
                                    risk=float(np.clip(next(risks), 0, np.nextafter(1.0, 0.0)))
                                    if row["accepted"]
                                    else None,
                                )
                                for row in whole_rows
                            ]

                        with Progress(
                            "neural.policy_scoring",
                            model=name,
                            fold=fold,
                            **_device_details(next(model.parameters()).device),
                        ):
                            policy_predictions = predicted(policy_rows)
                        with Progress("neural.policy_search", model=name, fold=fold, device="cpu"):
                            selected, searches = select_policy(
                                policy_predictions, _events(events, policy_rows), config, deadline=deadline
                            )
                            if selected is None:
                                raise ValueError("No neural development policy satisfies budget")
                        infer_started = time.monotonic()
                        with Progress(
                            "neural.outer_scoring",
                            model=name,
                            fold=fold,
                            **_device_details(next(model.parameters()).device),
                        ):
                            outer = predicted(outer_rows)
                        infer_seconds = time.monotonic() - infer_started
                        with Progress("neural.outer_evaluation", model=name, fold=fold, device="cpu"):
                            report = evaluate_stream(
                                outer, _events(events, outer_rows), selected, bootstrap_samples=0
                            )
                            frontier = {}
                            for budget in (0.1, 0.25, 1):
                                policy_config, _ = select_policy(
                                    policy_predictions,
                                    _events(events, policy_rows),
                                    config,
                                    budget,
                                    search=searches,
                                    deadline=deadline,
                                )
                                frontier[str(budget)] = (
                                    evaluate_stream(
                                        outer, _events(events, outer_rows), policy_config, bootstrap_samples=0
                                    )
                                    if policy_config
                                    else None
                                )
                        trial.update(
                            details,
                            status="complete",
                            report=report,
                            frontier=frontier,
                            inference_seconds=infer_seconds,
                            policy_search=searches,
                            selected_policy=selected["alert"],
                            calibration_diagnostics=calibration_diagnostics,
                            deployment_ready=False,
                        )
                        with Progress("neural.artifacts", model=name, fold=fold, device="cpu"):
                            torch.save(
                                {
                                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                                    "normalizer_mean": normalizer.mean.tolist(),
                                    "normalizer_scale": normalizer.scale.tolist(),
                                    "representation": representation,
                                    "width": width,
                                    "early_stop_weighting": early_stop_weighting,
                                    "config": selected,
                                    "feature_version": config["feature_version"],
                                    "source_contract": source_contract,
                                    "deployment_ready": False,
                                },
                                output / f"fold-{fold}-{name}.pt",
                            )
                            import joblib

                            joblib.dump(calibrator, output / f"fold-{fold}-{name}-calibrator.joblib")
                            trial["predictions_artifact"] = _write_predictions(
                                output / f"fold-{fold}-{name}-predictions.jsonl.gz", outer
                            )
                            predictions.setdefault(name, []).append(
                                _pool_fold(output, name, fold, outer, report)
                            )
                    except (ValueError, TimeoutError, RuntimeError, OSError) as error:
                        trial.update(status="failed", reason=str(error))
                        fold_progress.finish("failed", error_type=type(error).__name__)
                        if isinstance(error, TimeoutError):
                            exhausted = True
                    if trial["status"] == "complete":
                        fold_progress.update(1)
                        fold_progress.finish()
                    candidate_progress.update(len(predictions.get(name, [])), attempted_folds=fold + 1)
                    trial["wall_seconds"] = time.monotonic() - began
                    trial["sequence_cache"] = getattr(provider, "cache_diagnostics", lambda: {})()
                    log(trial)
                    results.append(trial)
                    if exhausted:
                        break
            finally:
                provider.close()
                sequence_caches.append(getattr(provider, "cache_diagnostics", lambda: {})())
            candidate_progress.finish(
                "failed"
                if exhausted or any(t["status"] == "failed" for t in results if t["model"] == name)
                else "done",
                complete_outer_folds=len(predictions.get(name, [])),
            )
        if exhausted:
            break
    combined = {}
    for name, parts in predictions.items():
        if len(parts) != 3:
            continue
        summed = {
            key: sum(part["report"][key] for part in parts)
            for key in (
                "eligible_events",
                "matched_events",
                "missed_events",
                "events_missed_during_abstention",
                "false_warnings",
                "warnings",
                "unassessable_warnings",
                "known_excluded_warnings",
                "unknown_outcome_warnings",
                "cooldown_suppressed_updates",
                "monitored_seconds",
                "eligible_seconds",
                "available_seconds",
                "rejected_seconds",
                "warning_seconds",
                "monitored_hours",
                "eligible_hours",
                "rejected_hours",
                "warning_hours",
            )
        }
        group_errors = {}
        for part in parts:
            for group, (total, count) in part["group_errors"].items():
                prior_total, prior_count = group_errors.get(group, (0.0, 0))
                group_errors[group] = (prior_total + total, prior_count + count)
        for key in (
            "exclusion_hours",
            "rejection_hours",
            "unassessable_warning_reasons",
            "known_excluded_warning_reasons",
            "unknown_outcome_warning_reasons",
        ):
            reasons = Counter()
            for part in parts:
                reasons.update(part["report"][key])
            summed[key] = dict(reasons)
        summed.update(
            **_rates(summed),
            per_group_brier={group: total / count for group, (total, count) in group_errors.items()},
            probability=probability_metrics(chain.from_iterable(part["probabilities"] for part in parts)),
            per_patient=[patient for part in parts for patient in part["report"]["per_patient"]],
            complete_outer_folds=3,
            cost_seconds=sum(trial["wall_seconds"] for trial in results if trial["model"] == name),
        )
        combined[name] = summed
    summary = {
        "status": "budget_exhausted" if exhausted else "complete",
        "combined": combined,
        "seed": seed,
        "extraction_contract": _extraction_contract(config),
        "development_cache_contract": cache_contract,
        "source_contract": source_contract,
        "specs": specs,
        "screening_max_training_windows": max_training_windows,
        "screening_epochs": screening_epochs,
        "early_stop_weighting": early_stop_weighting,
        "sequence_caches": sequence_caches,
        "wall_seconds": time.monotonic() - started,
        "failures": [
            {k: t.get(k) for k in ("model", "fold", "reason")} for t in results if t["status"] == "failed"
        ],
        "scope": "Development screening only; primary/external cohorts untouched; incomplete runs cannot win.",
        "training_subsampling": "Explicit group/class-balanced screening cap; full calibration/policy/outer timelines.",
        "deployment_ready": False,
    }
    summary["sha256"] = _digest(summary)
    _write(output / "summary.json", summary)
    return summary
