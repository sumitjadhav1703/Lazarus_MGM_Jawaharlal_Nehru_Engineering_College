import subprocess
import threading
from types import SimpleNamespace

import pytest

from cardiotwin import gpu_telemetry


def test_gpu_metrics_preserve_each_device_and_unavailable_values(monkeypatch):
    def run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        assert kwargs["timeout"] == 5
        return SimpleNamespace(stdout="0, Tesla T4, 82, 41, 1234, 15360\n1, Tesla T4, 0, N/A, 0, 15360\n")

    monkeypatch.setattr(gpu_telemetry.subprocess, "run", run)
    metrics = gpu_telemetry.read_gpu_metrics()
    assert metrics[0]["gpu_utilization_percent"] == 82
    assert metrics[1]["gpu_utilization_percent"] == 0
    assert metrics[1]["memory_utilization_percent"] is None
    assert metrics[0]["memory_used_mib"] == 1234


def test_monitor_logs_missing_metrics_without_failing_training(monkeypatch, capsys):
    called = threading.Event()

    def unavailable():
        called.set()
        raise subprocess.TimeoutExpired("nvidia-smi", 5)

    monkeypatch.setattr(gpu_telemetry, "read_gpu_metrics", unavailable)
    with gpu_telemetry.GpuTelemetry():
        assert called.wait(1)
    output = capsys.readouterr().out
    assert "TimeoutExpired" in output
    assert '"status": "done"' in output


def test_monitor_preserves_training_exception_and_closes_thread(monkeypatch):
    monkeypatch.setattr(gpu_telemetry, "read_gpu_metrics", lambda: [])
    monitor = gpu_telemetry.GpuTelemetry()
    with pytest.raises(ValueError, match="training failed"):
        with monitor:
            raise ValueError("training failed")
    assert not monitor.thread.is_alive()
