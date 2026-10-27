"""Private single-use native cost capsule entry; no fit, raw read, or dispatch API."""

import base64
import hashlib
import importlib.util
import json
import os
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

IDENTITY = 'sumit1703/cardiotwin-ecgfounder-native-cost-20261026-a'
STOP = datetime(2026, 10, 27, 12, 30, tzinfo=timezone.utc)
INPUT_ROOT = Path('/kaggle/input/datasets/sumit1703/cardiotwin-iridia-development-gpu-inputs/cardiotwin-gpu-bundle/data/iridia-development-cache-recovered')
HERE = 'council/work/preservation-20261026/launch-26a'
SUPERVISOR = 'scripts/kaggle_ecgfounder_engineering.py'
OLD_DEADLINE = b'DEADLINE = datetime(2026, 10, 22, 12, 30, tzinfo=timezone.utc)'
NEW_DEADLINE = b'DEADLINE = datetime(2026, 10, 27, 12, 30, tzinfo=timezone.utc)'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap(payload, *, session_started, working=Path('/kaggle/working')):
    if working != Path('/kaggle/working') or payload['identity_name'] != IDENTITY:
        raise ValueError('Fixed private native cost identity/root required')
    manifest = payload['files_sha256']
    if sha(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()) != payload['source_identity']:
        raise ValueError('Frozen source identity differs')
    if payload['dataset_sources'] != ['sumit1703/cardiotwin-iridia-development-gpu-inputs/1']:
        raise ValueError('Only original private development dataset permitted')
    project = working / IDENTITY.split('/')[1]
    if project.exists():
        raise FileExistsError('Existing native measurement identity; no retry')
    decoded = {}
    for name, encoded in payload['files'].items():
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts or name != str(relative)
                or relative.parts[0] in ('data', '.git', '.venv')):
            raise ValueError('Unsafe or prohibited staged source path')
        content = base64.b64decode(encoded, validate=True)
        if sha(content) != manifest[name]:
            raise ValueError('Staged source checksum differs: ' + name)
        decoded[name] = content
    if set(decoded) != set(manifest):
        raise ValueError('Staged source membership differs')
    if (decoded[SUPERVISOR].count(NEW_DEADLINE) != 1 or OLD_DEADLINE in decoded[SUPERVISOR]
            or b'ENGINEERING_SECONDS = 41400, 300, 1800' not in decoded[SUPERVISOR]):
        raise ValueError('New calendar supervisor patch differs')

    worker_started = None

    def guard():
        if time.monotonic() - session_started >= 41100 or datetime.now(timezone.utc) >= STOP:
            raise TimeoutError('Original session/new calendar work ceiling reached')
        if worker_started is not None and time.monotonic() - worker_started >= 1800:
            raise TimeoutError('Composite post-dependency ceiling reached')

    guard()
    project.mkdir()
    for name, content in decoded.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (project / 'execution-source-manifest.json').write_text(json.dumps(dict(
        files_sha256=manifest, source_identity=payload['source_identity']), sort_keys=True))
    sys.path.insert(0, str(project / 'src'))
    supervisor = load(project / SUPERVISOR, 'native_cost_supervisor')
    output = project / 'native-cost-results'
    output.mkdir()
    scratch = project / 'scratch'
    scratch.mkdir()
    receipt = dict(status='running', identity=IDENTITY, source_identity=payload['source_identity'],
                   dataset_sources=payload['dataset_sources'], outcome_fits=0,
                   full_run_eligible=False, retry=False, session_started=session_started)

    def save():
        receipt['elapsed_seconds'] = time.monotonic() - session_started
        (output / 'outer-receipt.json').write_text(json.dumps(receipt, sort_keys=True, allow_nan=False))

    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1', 'PYTHONPATH': str(project / 'src')}
    try:
        save()
        dependencies = tomllib.loads((project / 'pyproject.toml').read_text())['project']['dependencies']
        dependency_scratch = project / 'dependencies/.pip-tmp'
        dependency_scratch.mkdir(parents=True)
        receipt['dependency_bootstrap'] = supervisor.supervise(
            [sys.executable, '-m', 'pip', '--isolated', 'install', '--quiet', '--no-cache-dir',
             '--target', str(project / 'dependencies'), '--index-url', 'https://pypi.org/simple',
             *dependencies, 'numpy==2.5.4', 'scipy==1.18.1', 'scikit-learn==1.9.1', 'pytest==9.1.1'],
            project=project, output=output / 'dependencies-guard',
            roots=[Path('/kaggle/input'), working], session_started=session_started,
            engineering_started=None, phase='staging_identity',
            env={**env, 'TMPDIR': str(dependency_scratch)})
        guard()
        worker_started = time.monotonic()
        receipt['worker_started'] = worker_started
        save()
        sys.path.insert(0, str(project / 'dependencies'))
        env['PYTHONPATH'] += os.pathsep + str(project / 'dependencies')
        pins = json.loads((project / HERE / 'input-pins.json').read_text())
        if not INPUT_ROOT.is_dir():
            raise ValueError('Original mounted development cache absent')
        for name in ('index.json', 'compact.json', 'features.npy', 'metadata.npy'):
            guard()
            target = project / 'data/iridia-development-cache-recovered' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with (INPUT_ROOT / name).open('rb') as source, target.open('xb') as sink:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    guard()
                    sink.write(block)
                    digest.update(block)
                sink.flush()
                os.fsync(sink.fileno())
            if digest.hexdigest() != pins['data/iridia-development-cache-recovered/' + name]:
                raise ValueError('Pinned development cache transfer differs: ' + name)
        receipt['cache_staged'] = True
        save()
        receipt['worker_supervisor'] = supervisor.supervise(
            [sys.executable, str(project / HERE / 'child.py'), str(project),
             str(output / 'measurement'), payload['source_identity'], str(session_started), str(worker_started)],
            project=project, output=output / 'worker-guard',
            roots=[Path('/kaggle/input'), working], session_started=session_started,
            engineering_started=worker_started, phase='synthetic_cpu',
            env={**env, 'TMPDIR': str(scratch)})
        worker = json.loads((output / 'measurement/receipt.json').read_text())
        if (worker['status'] != 'complete_cost_measurement_only'
                or set(worker['phases']) != {'native_chronological', 'native_scattered', 'native_batch_controls',
                                             'table_scoring', 'metadata', 'pairs', 'control', 'frozen_replay',
                                             'replay_evaluations', 'packages', 'reports'}
                or worker['outcome_fits'] != 0 or worker['full_run_eligible'] is not False):
            raise ValueError('Complete source-bound cost receipt differs')
        receipt.update(status='complete_cost_measurement_only',
                       worker_receipt_sha256=sha((output / 'measurement/receipt.json').read_bytes()))
        return receipt
    except BaseException as error:
        receipt.update(status='failed', error_type=type(error).__name__,
                       original_error=supervisor.safe_text(error),
                       failed_supervisor_receipt=getattr(error, 'receipt', None))
        raise
    finally:
        save()
