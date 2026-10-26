"""Short private Kaggle startup diagnostic; no forecasting experiment or downloads."""

import json
import signal
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def probe(inputs):
    import h5py
    import torch

    files = list(inputs.rglob("record_*_ecg_*.h5"))
    if not files:
        raise RuntimeError("Approved development ECG input is not mounted")
    with h5py.File(files[0], "r") as handle:
        datasets = []
        handle.visititems(lambda name, obj: datasets.append(name) if isinstance(obj, h5py.Dataset) else None)
        if not datasets or handle[datasets[0]].size == 0:
            raise RuntimeError("Mounted ECG file has no samples")
        dataset = handle[datasets[0]]
        if dataset.ndim:
            dataset[tuple(slice(0, 1) for _ in dataset.shape)]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in requested GPU session")
    device = torch.device("cuda:0")
    model = torch.nn.Linear(16, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    x = torch.randn(32, 16, device=device)
    target = torch.zeros(32, 1, device=device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), target)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    if not torch.isfinite(loss).item():
        raise RuntimeError("Synthetic CUDA training loss is nonfinite")
    return {"device": str(device), "device_name": torch.cuda.get_device_name(0),
            "visible_gpus": torch.cuda.device_count(), "torch_version": torch.__version__,
            "cuda_batch_verified": True, "synthetic_batch": True,
            "forecasting_performance_measured": False, "ecg_files_visible": len(files),
            "input_read_verified": True, "loss": loss.item()}


def run(working, inputs, probe=probe):
    working.mkdir(parents=True, exist_ok=True)
    receipt_path = working / "log-smoke-receipt.json"
    receipt_path.touch(exist_ok=False)
    started = time.monotonic()
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}

    def save():
        receipt["elapsed_seconds"] = time.monotonic() - started
        temporary = receipt_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(receipt_path)

    def emit(stage, **fields):
        event = {"stage": stage, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                 "elapsed_seconds": time.monotonic() - started, **fields}
        line = json.dumps(event, allow_nan=False)
        with (working / "log-smoke.jsonl").open("a") as output:
            output.write(line + "\n")
            output.flush()
        print(line, flush=True)

    save()
    emit("startup", status="started")
    try:
        emit("probe", status="started")
        receipt["probe"] = probe(inputs)
        receipt["status"] = "complete"
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__,
                       original_error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        save()
        emit("finished", **receipt)


def timed_out(signum, frame):
    raise TimeoutError("Startup diagnostic exceeded 150 seconds")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(150)
    try:
        run(Path("/kaggle/working"), Path("/kaggle/input"))
    finally:
        signal.alarm(0)
