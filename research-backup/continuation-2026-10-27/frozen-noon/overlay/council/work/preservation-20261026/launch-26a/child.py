"""Guarded native cost child; source package and supervisor are supplied by outer entry."""

import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STOP = datetime(2026, 10, 27, 12, 30, tzinfo=timezone.utc)
WORKER = 'council/work/preservation-20261026/native-gap/worker.py'


def main(project, output, source_identity, session_started, started):
    project, output = Path(project).resolve(), Path(output).resolve()
    manifest = json.loads((project / 'execution-source-manifest.json').read_text())
    if manifest['source_identity'] != source_identity or not output.is_relative_to(project):
        raise ValueError('Bound staged source/output identity required')
    for name, digest in manifest['files_sha256'].items():
        if hashlib.sha256((project / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Staged source differs: ' + name)
    if os.environ.get('OMP_NUM_THREADS') != '1':
        raise ValueError('One-thread native measurement required')
    if not 0 <= started - session_started < 41100:
        raise ValueError('Inherited stage clock differs')

    def guard():
        now = time.monotonic()
        if now - session_started >= 41100 or now - started >= 1800 or datetime.now(timezone.utc) >= STOP:
            raise TimeoutError('Original session/component/new calendar ceiling reached')

    sys.path.insert(0, str(project / 'src'))
    sys.path.insert(0, str(project / 'council/work/preservation-20261026/native-gap'))
    spec = importlib.util.spec_from_file_location('native_cost_worker', project / WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    from cardiotwin import ecgfounder_runner as runner
    cycle_spec = importlib.util.spec_from_file_location('native_cycle', project / 'scripts/kaggle_cycle.py')
    cycle = importlib.util.module_from_spec(cycle_spec)
    cycle_spec.loader.exec_module(cycle)

    def storage():
        usage = cycle.dataset_usage([Path('/kaggle/input'), Path('/kaggle/working')])
        return runner._storage(lambda: dict(allocated_bytes=usage['allocated'],
                                            logical_bytes=usage['apparent']))

    pins = json.loads((project / 'council/work/preservation-20261026/launch-26a/input-pins.json').read_text())
    guard()
    worker.measure(project, output, input_sha256=pins, guard=guard, storage_check=storage)


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], *map(float, sys.argv[4:6]))
