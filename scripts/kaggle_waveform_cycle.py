#!/usr/bin/env python3
"""Development-only waveform challenger with a measured hardware/cost gate."""

import argparse
import gc
import json
import math
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from kaggle_cycle import DATASET_BYTES, dataset_bytes, run_child

from cardiotwin.gpu_telemetry import GpuTelemetry
from cardiotwin.progress import Progress

SESSION_SECONDS = 41400
RESERVE_SECONDS = 300
PROBE_SECONDS = 600
EPOCHS = 4
LAUNCHED = time.monotonic()


def cost_gate(
    serial_seconds,
    random_seconds,
    chronological_seconds,
    train_seconds,
    stop_windows,
    score_windows,
    available,
):
    """Conservative screen estimate; charge each phase its measured access pattern."""
    values = (serial_seconds, random_seconds, chronological_seconds, train_seconds, available)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Probe timings and available time must be finite and positive")
    if stop_windows < 0 or score_windows < 0:
        raise ValueError("Workload counts must be nonnegative")
    fitting = 3 * 10000
    # Inference is cheaper than backward; charging backward cost is conservative.
    random_histories = fitting * EPOCHS
    chronological_histories = stop_windows * EPOCHS + score_windows
    batched = random_histories + chronological_histories
    estimate = 1.5 * (
        fitting * serial_seconds
        + random_histories * (random_seconds + train_seconds)
        + chronological_histories * (chronological_seconds + train_seconds)
    )
    return {
        "estimated_seconds": estimate,
        "available_seconds": available,
        "cost_gate_passed": estimate <= available,
        "safety_multiplier": 1.5,
        "maximum_fit_histories": fitting,
        "random_training_histories": random_histories,
        "chronological_evaluation_histories": chronological_histories,
        "epochs": EPOCHS,
        "maximum_batched_histories": batched,
    }


def probe(data, cache, split_path, deadline):
    import numpy as np
    import torch
    from sklearn.model_selection import StratifiedGroupKFold

    from cardiotwin.data import list_records
    from cardiotwin.investigation import _group, _partitions, _subset, _usable, read_cache
    from cardiotwin.neural import WaveformCNN, record_sequence_provider

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    rows, events, index = read_cache(cache)
    split = json.loads(Path(split_path).read_text())
    if split["seed"] != 42 or set(index["allowed_groups"]) != set(split["development_groups"]):
        raise ValueError("Frozen development membership/seed differs")
    records = [
        r for r in list_records(data) if split["group_ids"][r.patient_id] in set(split["development_groups"])
    ]
    usable = _usable(rows)
    if len(usable) < 128:
        raise ValueError("Insufficient eligible development histories for throughput probe")
    rng = np.random.default_rng(42)
    random_rows = [usable[i] for i in rng.choice(len(usable), 64, replace=False)]
    chronological = usable[:64]
    provider = record_sequence_provider(
        records, index["config"], "waveform", source_contract=index["input_sha256"], deadline=deadline
    )
    result = {
        "gpu_name": torch.cuda.get_device_name(0),
        "device": "cuda:0",
        "input_shape": [64, 3, 60000],
        "parameters": 10561,
        "probe_scope": "Development-only runtime measurements; no performance evaluation",
    }
    try:
        with Progress("waveform_probe.read", total=160) as progress:
            started = time.monotonic()
            for count, row in enumerate(random_rows[:32], 1):
                array = provider(row)
                if array.shape != (3, 60000) or not np.isfinite(array).all():
                    raise ValueError("Malformed waveform history")
                progress.update(count)
            serial = (time.monotonic() - started) / 32
            timings = []
            for count, batch in enumerate((random_rows, chronological), 1):
                started = time.monotonic()
                arrays = provider.batch(batch)
                timings.append((time.monotonic() - started) / 64)
                if any(a.shape != (3, 60000) or not np.isfinite(a).all() for a in arrays):
                    raise ValueError("Malformed batched histories")
                progress.update(32 + count * 64, force=True)
        # Runtime probe steps are discarded; full training starts from seed42 anew.
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True, warn_only=True)
        model = WaveformCNN(16).to("cuda:0")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        x = torch.from_numpy(np.stack(arrays)).to("cuda:0")
        y = torch.zeros(64, device="cuda:0")
        torch.cuda.reset_peak_memory_stats()
        timings_gpu = []
        with Progress("waveform_probe.cuda_backward", total=3, device="cuda:0") as progress:
            for step in range(3):
                torch.cuda.synchronize()
                started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite probe loss")
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                timings_gpu.append((time.monotonic() - started) / 64)
                progress.update(step + 1, force=True, loss=float(loss.detach().cpu()))
        result.update(
            serial_read_seconds_per_history=serial,
            random_batch_read_seconds_per_history=timings[0],
            chronological_read_seconds_per_history=timings[1],
            cuda_train_seconds_per_history=max(timings_gpu[1:]),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        )
        all_groups = np.array(sorted({_group(r) for r in rows}))
        event_groups = {_group(e) for e in events}
        statuses = np.array([int(g in event_groups) for g in all_groups])
        stop_count = score_count = 0
        for fold, (training, held) in enumerate(
            StratifiedGroupKFold(3, shuffle=True, random_state=42).split(
                np.zeros(len(all_groups)), statuses, all_groups
            )
        ):
            fit, cal, policy = _partitions(_subset(rows, set(all_groups[training])), 42 + fold)
            _, inner_stop, _ = _partitions(_subset(rows, fit), 142 + fold)
            stop_count += len(_usable(_subset(rows, inner_stop)))
            score_count += len(_usable(_subset(rows, cal)))
            score_count += sum(
                sum(bool(row["accepted"]) for row in _subset(rows, groups))
                for groups in (policy, all_groups[held])
            )
        result.update(
            cost_gate(
                serial,
                timings[0],
                timings[1],
                result["cuda_train_seconds_per_history"],
                stop_count,
                score_count,
                deadline - time.monotonic(),
            )
        )
        if result["cuda_peak_allocated_bytes"] > 0.85 * result["cuda_total_memory_bytes"]:
            result.update(cost_gate_passed=False, memory_gate_error="Less than 15% GPU memory headroom")
        return result
    finally:
        provider.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "cache", "split", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--session-seconds-remaining", required=True, type=float)
    args = parser.parse_args()
    remaining = args.session_seconds_remaining - (time.monotonic() - LAUNCHED)
    if not math.isfinite(remaining) or not RESERVE_SECONDS < remaining <= SESSION_SECONDS:
        raise ValueError("Invalid remaining session time")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "recipe": {
            "model": "waveform16",
            "epochs": EPOCHS,
            "training_windows": 10000,
            "seed": 42,
            "early_stop_weighting": "patient_only",
        },
        "automatic_retry": False,
    }
    deadline = time.monotonic() + remaining - RESERVE_SECONDS
    roots = (Path("/kaggle/input"), Path("/kaggle/working"))

    def save():
        receipt["updated_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["elapsed_seconds"] = time.monotonic() - LAUNCHED
        target = args.output / "receipt.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(target)

    def terminate(signum, frame):
        raise TimeoutError(f"Waveform supervisor interrupted by {signal.Signals(signum).name}")

    signal.signal(signal.SIGALRM, terminate)
    signal.signal(signal.SIGTERM, terminate)
    signal.setitimer(signal.ITIMER_REAL, deadline - time.monotonic())
    save()
    try:
        occupied = dataset_bytes(roots)
        if occupied + 100_000_000 > DATASET_BYTES:
            raise ValueError("Insufficient 25 GB storage headroom")
        receipt["initial_dataset_bytes"] = occupied
        with GpuTelemetry():
            signal.setitimer(signal.ITIMER_REAL, min(PROBE_SECONDS, deadline - time.monotonic()))
            receipt["probe"] = probe(args.data, args.cache, args.split, deadline)
            save()
            print(json.dumps({"stage": "waveform_probe", **receipt["probe"]}), flush=True)
            if not receipt["probe"]["cost_gate_passed"]:
                receipt["status"] = "blocked_by_runtime_gate"
                raise RuntimeError(
                    "Measured waveform cost/memory exceeds this session; full training not launched"
                )
            import torch

            gc.collect()
            torch.cuda.empty_cache()
            # Let child cleanup and its structured receipt finish before the supervisor alarm.
            cap = deadline - time.monotonic() - 10
            if cap <= 0:
                raise TimeoutError("No training budget remains after the probe")
            signal.setitimer(signal.ITIMER_REAL, cap + 10)
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).with_name("run_investigation.py")),
                "neural",
                "--data",
                str(args.data),
                "--cache",
                str(args.cache),
                "--split",
                str(args.split),
                "--output",
                str(args.output / "model"),
                "--models",
                "waveform16",
                "--epochs",
                str(EPOCHS),
                "--training-windows",
                "10000",
                "--early-stop-weighting",
                "patient_only",
                "--max-hours",
                str(cap / 3600),
            ]
            receipt["command"] = command
            save()
            receipt["child"] = run_child(command, args.output / "training.log", cap, storage_roots=roots)
            if receipt["child"]["status"] != "finished":
                raise RuntimeError("Waveform training failed or timed out; inspect child log and receipt")
            summary = json.loads((args.output / "model/summary.json").read_text())
            complete = (
                summary.get("status") == "complete"
                and not summary.get("failures")
                and summary.get("specs") == [["waveform", 16]]
                and summary.get("screening_epochs") == EPOCHS
                and summary.get("screening_max_training_windows") == 10000
                and summary.get("early_stop_weighting") == "patient_only"
                and summary.get("seed") == 42
                and summary.get("combined", {}).get("waveform_cnn_16", {}).get("complete_outer_folds") == 3
            )
            receipt["status"] = "complete" if complete else "incomplete"
            if not complete:
                raise RuntimeError("Incomplete or mismatched grouped waveform comparison")
    except BaseException as error:
        if receipt["status"] == "running":
            receipt["status"] = "failed"
        receipt.update(original_error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        save()
        print(
            json.dumps({"waveform_receipt": str(args.output / "receipt.json"), "status": receipt["status"]}),
            flush=True,
        )


if __name__ == "__main__":
    main()
