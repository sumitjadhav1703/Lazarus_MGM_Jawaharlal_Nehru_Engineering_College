import importlib.util
import json
from pathlib import Path

import pytest


def load_smoke():
    path = Path(__file__).resolve().parents[1] / "scripts/kaggle_log_smoke.py"
    spec = importlib.util.spec_from_file_location("kaggle_log_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_persists_start_before_probe_and_records_success(tmp_path, capsys):
    smoke = load_smoke()

    def probe(inputs):
        receipt = json.loads((tmp_path / "log-smoke-receipt.json").read_text())
        assert receipt["status"] == "running"
        first_line = (tmp_path / "log-smoke.jsonl").read_text().splitlines()[0]
        assert json.loads(first_line)["stage"] == "startup"
        return {"device": "test-only", "cuda_batch_verified": False}

    smoke.run(tmp_path, tmp_path / "inputs", probe=probe)
    receipt = json.loads((tmp_path / "log-smoke-receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["probe"]["cuda_batch_verified"] is False
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["stage"] for event in events] == ["startup", "probe", "finished"]
    assert all(event["timestamp_utc"] for event in events)


def test_smoke_keeps_original_failure_in_receipt_and_logs(tmp_path, capsys):
    smoke = load_smoke()

    def failed_probe(inputs):
        raise RuntimeError("CUDA original failure")

    with pytest.raises(RuntimeError, match="CUDA original failure"):
        smoke.run(tmp_path, tmp_path / "inputs", probe=failed_probe)
    receipt = json.loads((tmp_path / "log-smoke-receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["original_error"] == "CUDA original failure"
    assert "RuntimeError: CUDA original failure" in receipt["traceback"]
    assert "CUDA original failure" in (tmp_path / "log-smoke.jsonl").read_text()
    assert "CUDA original failure" in capsys.readouterr().out
