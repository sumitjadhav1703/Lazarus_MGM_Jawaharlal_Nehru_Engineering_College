"""Bounded NVIDIA utilization samples; diagnostic failure never aborts training."""

import csv
import subprocess
import threading
import time

from .progress import Progress


def read_gpu_metrics():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    metrics = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 6:
            raise ValueError("Unexpected NVIDIA metric columns")
        values = [
            float(value) if value.strip() not in {"N/A", "[N/A]", "[Not Supported]"} else None
            for value in row[2:]
        ]
        metrics.append(
            dict(
                index=int(row[0]),
                name=row[1],
                **dict(
                    zip(
                        (
                            "gpu_utilization_percent",
                            "memory_utilization_percent",
                            "memory_used_mib",
                            "memory_total_mib",
                        ),
                        values,
                        strict=True,
                    )
                ),
            )
        )
    return metrics


class GpuTelemetry:
    def __init__(self, interval_seconds=30):
        self.interval_seconds = interval_seconds
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)

    def _run(self):
        previous_wall = previous_cpu = None
        while not self.stop.is_set():
            wall, cpu = time.monotonic(), time.process_time()
            # Multiple busy CPU cores can legitimately exceed 100 percent.
            cpu_percent = (
                100 * (cpu - previous_cpu) / max(wall - previous_wall, 1e-9)
                if previous_wall is not None
                else None
            )
            previous_wall, previous_cpu = wall, cpu
            try:
                self.progress.update(
                    force=True,
                    gpu=read_gpu_metrics(),
                    metrics_error=None,
                    process_cpu_core_percent=cpu_percent,
                )
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                self.progress.update(
                    force=True,
                    gpu=None,
                    metrics_error=type(error).__name__,
                    process_cpu_core_percent=cpu_percent,
                )
            self.stop.wait(self.interval_seconds)

    def __enter__(self):
        self.progress = Progress("gpu_resource_samples", sample_interval_seconds=self.interval_seconds)
        self.thread.start()
        return self

    def __exit__(self, error_type, error, traceback):
        self.stop.set()
        self.thread.join(timeout=6)
        self.progress.finish("failed" if error_type else "done")
