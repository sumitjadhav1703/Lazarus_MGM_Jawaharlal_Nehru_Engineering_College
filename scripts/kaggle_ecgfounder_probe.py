"""Package and guard one private, source-bound ECGFounder engineering probe; never submit."""

import time

LAUNCHED = globals().get('LAUNCHED', time.monotonic())

import argparse  # noqa: E402
import base64  # noqa: E402
import hashlib  # noqa: E402
import importlib.metadata  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import tomllib  # noqa: E402
import traceback  # noqa: E402
import urllib.request  # noqa: E402
import zlib  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
CYCLE = 'reports/ml-investigation/next-cycle/ecgfounder-cycle-20261013'
DATASET = 'sumit1703/cardiotwin-iridia-development-gpu-inputs'
CHECKPOINT_BYTES = 369807481
CHECKPOINT_SHA = 'f863a38897fb49a27fec7e44008ea3c7bdbd29c77fa4a02ecbb8c56df4f37603'
REVISION = 'd9b1793951b2342f5f7e84f1ac03cd37f8a08724'
SESSION_SECONDS, CLEANUP_SECONDS, PROBE_SECONDS = 41400, 300, 1800
OUTPUT_BYTES, STORAGE_BYTES = 250_000_000, 25_000_000_000
PANEL_SPEC_SHA = '792ade5406095eefe2ee0a4cddc9d137b3947ab2a36f5fc22e2ac8ec6f4f9713'
PANEL_DESIGN_SHA = 'baf347e0144373baaa7c3de95375783902419f94b7dc177c846247fb68997e42'
PANEL_TIMING_SHA = '73d24e5f94c97ffd8768be56923a0236bf1d453e911c2c045c89a94730683fb9'
OLD_SELECTION_SHA = '01697f20250db697e2e4d9ff5b98a239ba4025aa4e258ac818a157701fa88eca'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sanitized(error):
    return re.sub(r'https?://[^\s\'\"<>]+', '<URL redacted>', str(error))


def manifest_identity(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def valid_path(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name:
        return False
    return (path.parent == Path('src/cardiotwin') and path.suffix == '.py'
            or path.parent == Path('scripts') and path.suffix == '.py'
            or path.parent == Path('configs') and path.suffix == '.json'
            or name == 'pyproject.toml' or name == 'reports/ml-investigation/monitor-storage.py'
            or name in {'data/iridia-development-cache-recovered/index.json',
                        'data/iridia-development-cache-recovered/compact.json',
                        'reports/ml-investigation/primary-split.json'}
            or name.startswith('reports/ml-investigation/next-cycle/')
            and path.suffix in {'.json', '.jsonl', '.md'} and 'weights' not in path.parts
            or name in {CYCLE + '/vendor-source/net1d.py', CYCLE + '/vendor-source/LICENSE'})


def build_payload(root, names, selection, selection_sha256):
    root, selection = Path(root).resolve(), Path(selection).resolve()
    if not selection.is_relative_to(root) or sha(selection) != selection_sha256:
        raise ValueError('Reviewed selection hash/path differs')
    selected = json.loads(selection.read_text())
    bindings = {**selected['inputs']['files_sha256'], **selected['inputs']['source_sha256']}
    names = set(names) | set(bindings) | {str(selection.relative_to(root))}
    files, manifest = {}, {}
    total = 0
    for name in sorted(names):
        if not valid_path(name) or not (root / name).resolve().is_relative_to(root):
            raise ValueError(f'Invalid package path: {name}')
        path = root / name
        if path.stat().st_size > 5_000_000:
            raise ValueError('Only small source/metadata files may enter the package')
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if name in bindings and actual != bindings[name]:
            raise ValueError(f'Selection source/input binding changed: {name}')
        total += len(content)
        if total > 10_000_000:
            raise ValueError('Source package exceeds 10MB bound')
        manifest[name] = actual
        files[name] = base64.b64encode(content).decode('ascii')
    if manifest[str(selection.relative_to(root))] != selection_sha256:
        raise ValueError('Selection changed during packaging')
    return dict(files=files, manifest=manifest, identity=manifest_identity(manifest),
                selection=str(selection.relative_to(root)), selection_sha256=selection_sha256)


def overlay(project, payload):
    project = Path(project).absolute()
    files, manifest = payload['files'], payload['manifest']
    if not files or set(files) != set(manifest) or manifest_identity(manifest) != payload['identity']:
        raise ValueError('Source manifest identity/membership differs')
    decoded = {}
    for name, encoded in files.items():
        if not valid_path(name):
            raise ValueError(f'Invalid source path: {name}')
        content = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(content).hexdigest() != manifest[name]:
            raise ValueError(f'Source checksum differs: {name}')
        target = project / name
        if any(parent.is_symlink() for parent in [project, *target.parents]
               if parent == project or parent.is_relative_to(project)):
            raise ValueError(f'Source parent is a symlink: {name}')
        if not target.parent.resolve().is_relative_to(project.resolve()):
            raise ValueError(f'Source path escapes project: {name}')
        decoded[name] = content
    for name, content in decoded.items():
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        target.write_bytes(content)
        if sha(target) != manifest[name]:
            raise ValueError('Written source checksum differs')
    write_json(project / 'execution-source-manifest.json',
               dict(files_sha256=manifest, source_identity=payload['identity']))


def kernel_metadata(identity):
    if not re.fullmatch(r'sumit1703/cardiotwin-ecgfounder-probe-[0-9]{8}-[a-z0-9-]+', identity):
        raise ValueError('A unique prospective ECGFounder probe identity is required')
    return dict(id=identity, title=identity.split('/')[1], code_file='train.py', language='python',
                kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True,
                machine_shape='NvidiaTeslaT4', dataset_sources=[DATASET], competition_sources=[],
                kernel_sources=[], model_sources=[])


def load_script(name):
    path = PROJECT / 'scripts' / name
    spec = importlib.util.spec_from_file_location('_ecgfounder_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def storage_usage(roots):
    cycle = load_script('kaggle_cycle.py')
    return {name: cycle.dataset_bytes(roots, allocated=allocated)
            for name, allocated in (('apparent_bytes', False), ('allocated_bytes', True))}


def reserve_storage(roots, *, checkpoint_bytes=0):
    usage = storage_usage(roots)
    if max(usage.values()) + checkpoint_bytes + OUTPUT_BYTES > STORAGE_BYTES:
        raise RuntimeError('Checkpoint/output reservation exceeds storage cap')
    return usage


def acquire_checkpoint(url, destination, size, expected_sha256, progress):
    destination = Path(destination)
    partial = destination.with_suffix(destination.suffix + '.partial')
    if destination.exists() or partial.exists():
        raise FileExistsError('Checkpoint or partial already exists; reconcile before any retry')
    destination.parent.mkdir(parents=True, exist_ok=True)
    started, completed, last = time.monotonic(), 0, 0.
    with partial.open('xb') as handle:
        with urllib.request.urlopen(url, timeout=60) as response:
            length = response.headers.get('Content-Length')
            if length is not None and int(length) != size:
                raise ValueError('Checkpoint response size differs')
            while block := response.read(1024 * 1024):
                if completed + len(block) > size:
                    raise ValueError('Checkpoint exceeds pinned size')
                handle.write(block)
                completed += len(block)
                if time.monotonic() - last >= 5:
                    handle.flush()
                    progress(dict(completed_bytes=completed, total_bytes=size,
                                  elapsed_seconds=time.monotonic() - started))
                    last = time.monotonic()
        handle.flush()
        os.fsync(handle.fileno())
    if completed != size:
        raise ValueError('Checkpoint transfer incomplete')
    actual = sha(partial)
    if actual != expected_sha256:
        raise ValueError('Checkpoint SHA256 differs')
    partial.rename(destination)
    progress(dict(completed_bytes=completed, total_bytes=size,
                  elapsed_seconds=time.monotonic() - started, verified=True))
    return dict(bytes=completed, sha256=actual, elapsed_seconds=time.monotonic() - started,
                retry=False, signed_url_recorded=False)


def probe_session_limit(started, *, now=None, compute_seconds=PROBE_SECONDS):
    now = time.monotonic() if now is None else now
    elapsed = now - started
    if not math.isfinite(elapsed) or elapsed < 0 or not 0 < compute_seconds <= PROBE_SECONDS:
        raise ValueError('Invalid session/probe allowance')
    if elapsed >= SESSION_SECONDS - CLEANUP_SECONDS:
        raise TimeoutError('Session allowance exhausted during bootstrap')
    return min(SESSION_SECONDS, elapsed + compute_seconds + CLEANUP_SECONDS)


def guard_probe(command, output, storage_roots, *, session_started, result_path,
                additional_budget=None, compute_seconds=PROBE_SECONDS, compute_started=None):
    guard = load_script('kaggle_precursor_cycle.py')
    allowance = probe_session_limit(session_started, now=compute_started, compute_seconds=compute_seconds)
    result = guard.run_guarded(command, output, storage_roots, session_started=session_started,
                               session_seconds=allowance, reserve_seconds=CLEANUP_SECONDS,
                               experiment_roots=[Path(result_path).parent])
    receipt, error = {}, result['original_error']
    try:
        receipt = json.loads(Path(result_path).read_text())
        if not isinstance(receipt, dict) or receipt.get('status') != 'passed':
            raise ValueError('Probe runner did not pass its engineering gates')
    except (OSError, ValueError) as failure:
        if not isinstance(receipt, dict):
            receipt = {}
        if error is None:
            error = sanitized(failure)
    state_path = Path(result_path).with_name('execution-state.json')
    if not receipt and state_path.is_file():
        try:
            partial = json.loads(state_path.read_text())
            receipt = {key: partial.get(key, 0) for key in (
                'raw_history_requests', 'successful_raw_history_reads', 'completed_unique_histories')}
        except (OSError, ValueError, AttributeError) as failure:
            error = error or sanitized(failure)
    passed = result['status'] == 'finished' and receipt.get('status') == 'passed' and error is None
    value = dict(status='passed' if passed else 'failed', guard=result, probe=receipt,
                 original_error=error, full_run_eligible=passed and bool(additional_budget)
                 and receipt.get('full_run_eligible') is True, retry=False)
    write_json(Path(output) / 'wrapper-result.json', value)
    return value


def package(root, cycle, selection, selection_sha256, output, identity, additional_budget=None, *, attempt=None):
    root, cycle = Path(root).resolve(), Path(cycle).resolve()
    if cycle != root / CYCLE:
        raise ValueError('Only the frozen ECGFounder cycle may be packaged')
    metadata = kernel_metadata(identity)
    names = {CYCLE + '/probe-spec.json', CYCLE + '/declaration-binding.json',
             'scripts/kaggle_ecgfounder_probe.py', 'scripts/kaggle_precursor_cycle.py',
             'scripts/kaggle_cycle.py', 'data/iridia-development-cache-recovered/compact.json'}
    payload = build_payload(root, names, selection, selection_sha256)
    selected = json.loads(Path(selection).read_text())
    selected_inputs = selected['inputs']
    payload['attempt'] = None
    if attempt or selected.get('attempt'):
        expected = cycle / 'repair-panel-20261014-b/spec.md'
        if (not attempt or Path(attempt).resolve() != expected
                or identity != 'sumit1703/cardiotwin-ecgfounder-probe-20261014-b'
                or additional_budget is not None):
            raise ValueError('Fixed panel-b attempt/identity with no additional budget is required')
        panel = selected.get('attempt', {})
        if any(panel.get(key) != value for key, value in (
                ('spec_sha256', PANEL_SPEC_SHA), ('design_sha256', PANEL_DESIGN_SHA),
                ('timing_addendum_sha256', PANEL_TIMING_SHA), ('old_selection_sha256', OLD_SELECTION_SHA))):
            raise ValueError('Fixed panel specification/design/addendum/old selection pins differ')
        declaration = expected.parent / 'declaration-binding.json'
        required = {str(expected.relative_to(root)): panel.get('spec_sha256'),
                    str((cycle / 'repair-design-20261014.md').relative_to(root)): panel.get('design_sha256'),
                    str((expected.parent / 'dense-dispatch-timing-addendum.md').relative_to(root)):
                    panel.get('timing_addendum_sha256'),
                    str((cycle / 'selection-20261014/selected.json').relative_to(root)):
                    panel.get('old_selection_sha256'),
                    str(declaration.relative_to(root)): panel.get('declaration_sha256')}
        if (panel.get('identity') != 'ecgfounder-metadata-supported-panel-20261014-b'
                or panel.get('spec') != str(expected.relative_to(root))
                or panel.get('declaration') != str(declaration.relative_to(root))
                or any(not value or selected_inputs['files_sha256'].get(name) != value
                       or payload['manifest'].get(name) != value for name, value in required.items())):
            raise ValueError('Reviewed fixed attempt bindings differ')
        declared = json.loads(declaration.read_text())['files_sha256']
        selected_bindings = {**selected_inputs['files_sha256'], **selected_inputs['source_sha256']}
        if any(selected_bindings.get(name) != value for name, value in declared.items()):
            raise ValueError('Attempt declaration input map differs')
        payload['attempt'] = str(expected.relative_to(root))
    if sha(cycle / 'declaration-binding.json') != selected_inputs['declaration_sha256']:
        raise ValueError('Reviewed declaration binding differs')
    pin = json.loads((cycle / 'probe-spec.json').read_text())['checkpoint']
    if (pin['bytes'], pin['sha256'], pin['revision'], pin['repository'], pin['file']) != (
            CHECKPOINT_BYTES, CHECKPOINT_SHA, REVISION, 'PKUDigitalHealth/ECGFounder', '1_lead_ECGFounder.pth'):
        raise ValueError('Frozen checkpoint pins differ')
    compact = selected_inputs['compact_binding']
    if payload['manifest']['data/iridia-development-cache-recovered/compact.json'] != compact['compact_sha256']:
        raise ValueError('Compact metadata binding differs')
    roles = [name for name in payload['manifest'] if name.endswith('/roles.json')]
    if len(roles) != 1:
        raise ValueError('Exactly the frozen original roles are required')
    payload.update(checkpoint=pin, audit=str(Path(roles[0]).parent), identity_name=identity,
                   additional_budget=json.loads(Path(additional_budget).read_text()) if additional_budget else None)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    packed = base64.b64encode(zlib.compress(json.dumps(payload, separators=(',', ':')).encode())).decode()
    source = (root / 'scripts/kaggle_ecgfounder_probe.py').read_text()
    prefix = 'import time\nLAUNCHED = time.monotonic()\nEMBEDDED_PAYLOAD = ' + repr(packed) + '\n'
    (output / 'train.py').write_text(prefix + source)
    write_json(output / 'kernel-metadata.json', metadata)
    write_json(output / 'package-receipt.json', dict(status='prepared_not_submitted',
               source_identity=payload['identity'], files_sha256=payload['manifest'],
               train_sha256=sha(output / 'train.py'), selection_sha256=selection_sha256,
               attempt=payload['attempt'],
               checkpoint_uploaded=False, raw_bytes_read=0, automatic_retry=False))
    return output


def runner_command(project, payload, cache, output, data_root, data_manifest):
    project = Path(project)
    command = [sys.executable, '-u', str(project / 'scripts/run_ecgfounder_probe.py'),
               '--cycle', str(project / CYCLE), '--cache', str(cache), '--audit', str(project / payload['audit']),
               '--output', str(output), '--mode', 'probe', '--selection', str(project / payload['selection']),
               '--selection-sha256', payload['selection_sha256'], '--device', 'cuda:0',
               '--data-root', str(data_root), '--data-manifest', str(data_manifest),
               '--data-manifest-sha256', sha(data_manifest)]
    if payload.get('attempt'):
        if payload['attempt'] != CYCLE + '/repair-panel-20261014-b/spec.md':
            raise ValueError('Invalid remote attempt path')
        command += ['--attempt', str(project / payload['attempt'])]
    return command


def validate_runtime(python_version, torch_version, device_name):
    version = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)(?:\+[a-zA-Z0-9.]+)?', torch_version)
    if (not (3, 12) <= tuple(python_version[:2]) < (3, 15) or version is None
            or not (2, 6) <= tuple(map(int, version.groups()[:2])) < (3, 0)
            or 'T4' not in device_name):
        raise RuntimeError('Declared Python/PyTorch/T4 runtime requirements are not satisfied')


def locate_inputs(inputs, compact_sha256):
    inputs = Path(inputs).resolve()
    mounted = list(inputs.rglob('cardiotwin-gpu-bundle/data/iridia-development-cache-recovered/compact.json'))
    if len(mounted) != 1 or sha(mounted[0]) != compact_sha256:
        raise ValueError('Attach exactly the approved private development dataset v1')
    source_cache = mounted[0].parent
    dataset_root = mounted[0].parents[3]
    if not dataset_root.resolve().is_relative_to(inputs):
        raise ValueError('Private dataset mount escapes input root')
    raw = sorted(dataset_root.glob('record_*'))
    if len(raw) != 718 or len({p.name[:10] for p in raw}) != 132 or any(
            not p.is_file() or not p.resolve().is_relative_to(dataset_root.resolve())
            or not re.fullmatch(r'record_\d{3}_(ecg|rr)_(\d{2}\.h5|labels\.csv)', p.name) for p in raw):
        raise ValueError('Original development raw membership differs')
    return source_cache, dataset_root


def remote_probe(payload, *, working=Path('/kaggle/working'), inputs=Path('/kaggle/input')):
    global PROJECT
    working, inputs = Path(working), Path(inputs)
    run = working / payload['identity_name'].split('/')[1]
    run.mkdir(exist_ok=False)
    scratch = Path('/tmp') / run.name
    scratch.mkdir(exist_ok=False)
    PROJECT = scratch / 'project'
    receipt = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(), stages=[],
                   source_identity=payload['identity'], selection_sha256=payload['selection_sha256'],
                   retry=False, full_run_eligible=False, raw_history_requests=0, provenance_files_read=0,
                   successful_raw_history_reads=0, completed_unique_histories=0,
                   session_seconds=SESSION_SECONDS, cleanup_reserve_seconds=CLEANUP_SECONDS,
                   scratch=str(scratch), checkpoint_uploaded=False)

    def save():
        receipt['elapsed_seconds'] = time.monotonic() - LAUNCHED
        write_json(run / 'bootstrap-receipt.json', receipt)

    def event(stage, **fields):
        row = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(),
                   session_elapsed_seconds=time.monotonic() - LAUNCHED, **fields)
        receipt['stages'].append(row)
        save()
        print(json.dumps(row, allow_nan=False), flush=True)

    def interrupt(signum, frame):
        raise TimeoutError(f'Probe wrapper received {signal.Signals(signum).name}')

    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupt)
    signal.setitimer(signal.ITIMER_REAL, max(.001, SESSION_SECONDS - CLEANUP_SECONDS
                                          - (time.monotonic() - LAUNCHED)))
    save()
    try:
        if not (3, 12) <= sys.version_info[:2] < (3, 15):
            raise RuntimeError('Declared Python runtime requires >=3.12,<3.15')
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['PYTHONUNBUFFERED'] = '1'
        temporary = scratch / 'tmp'
        temporary.mkdir()
        os.environ['TMPDIR'] = str(temporary)
        os.environ['MPLCONFIGDIR'] = str(scratch / 'matplotlib')
        overlay(PROJECT, payload)
        selected_path = PROJECT / payload['selection']
        if sha(selected_path) != payload['selection_sha256']:
            raise ValueError('Reviewed selection differs before bootstrap/raw processing')
        event('verified_source_overlay', files=len(payload['manifest']))
        (run / 'source-overlay.json.zlib').write_bytes(zlib.compress(json.dumps(payload).encode()))
        sys.path.insert(0, str(PROJECT / 'src'))
        os.chdir(PROJECT)
        cache = PROJECT / 'data/iridia-development-cache-recovered'
        source_cache, dataset_root = locate_inputs(inputs, sha(cache / 'compact.json'))
        for name in ('rows.jsonl', 'features.npy', 'metadata.npy'):
            (cache / name).symlink_to((source_cache / name).resolve())
        roots = [inputs, working, scratch]
        receipt['storage_before'] = reserve_storage(roots, checkpoint_bytes=CHECKPOINT_BYTES)
        event('immutable_input_layout', development_records=132, source_files=718)
        dependencies = tomllib.loads((PROJECT / 'pyproject.toml').read_text())['project']['dependencies']
        cycle = load_script('kaggle_cycle.py')
        dependency_run = cycle.run_child([sys.executable, '-m', 'pip', '--isolated', 'install', '--quiet',
            '--no-cache-dir', '--index-url', 'https://pypi.org/simple', *dependencies, 'h5py>=3.14,<4', 'matplotlib'],
            run / 'dependencies.log', min(300, SESSION_SECONDS - CLEANUP_SECONDS - (time.monotonic() - LAUNCHED)),
            storage_roots=roots, check_allocated=True, output_limit=([working], OUTPUT_BYTES))
        if dependency_run['status'] != 'finished':
            raise RuntimeError('Dependency bootstrap failed: ' + sanitized(dependency_run))
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('A real T4 CUDA device is required; no CPU fallback')
        validate_runtime(sys.version_info, torch.__version__, torch.cuda.get_device_name(0))
        receipt['runtime'] = {name: importlib.metadata.version(name) for name in
            ('numpy', 'scipy', 'pandas', 'wfdb', 'scikit-learn', 'h5py', 'torch', 'matplotlib')}
        event('runtime_ready', device='cuda:0', device_name=torch.cuda.get_device_name(0), runtime=receipt['runtime'])
        pin = payload['checkpoint']
        if (pin['bytes'], pin['sha256'], pin['revision']) != (CHECKPOINT_BYTES, CHECKPOINT_SHA, REVISION):
            raise ValueError('Checkpoint pins changed')
        destination = PROJECT / CYCLE / 'weights/1_lead_ECGFounder.pth'
        url = f'https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/{REVISION}/1_lead_ECGFounder.pth'
        event('checkpoint_acquisition_started', completed_bytes=0, total_bytes=CHECKPOINT_BYTES)
        receipt['storage_before_transfer'] = reserve_storage(roots, checkpoint_bytes=CHECKPOINT_BYTES)
        receipt['checkpoint'] = acquire_checkpoint(url, destination, CHECKPOINT_BYTES, CHECKPOINT_SHA,
                                                   lambda row: event('checkpoint_acquisition', **row))
        compute_started = time.monotonic()
        signal.setitimer(signal.ITIMER_REAL, min(PROBE_SECONDS, SESSION_SECONDS - CLEANUP_SECONDS
                                                - (compute_started - LAUNCHED)))
        reserve_storage(roots)
        selected = json.loads(selected_path.read_text())
        index = json.loads((cache / 'index.json').read_text())
        records = {entry['record']['record_id']: entry['record'] for entry in index['records']}
        names = set()
        for rid in sorted({r['record_id'] for r in selected['requests'] if r['status'] == 'selected'}):
            if not re.fullmatch(r'record_\d{3}', rid):
                raise ValueError('Invalid reviewed record identity')
            names.update(Path(part['path']).name for part in records[rid]['signal_parts'])
            names.add(rid + '_ecg_labels.csv')
        hashes = {}
        for count, name in enumerate(sorted(names), 1):
            if (not re.fullmatch(r'record_\d{3}_ecg_(\d{2}\.h5|labels\.csv)', name)
                    or not (dataset_root / name).resolve().is_relative_to(dataset_root.resolve())):
                raise ValueError('Invalid selected provenance filename')
            hashes[name] = sha(dataset_root / name)
            receipt['provenance_files_read'] = count
            event('selected_provenance_hashing', completed=count, total=len(names), raw_history_requests=0)
        data_manifest = run / 'selected-data-manifest.json'
        write_json(data_manifest, dict(files_sha256=hashes))
        command = runner_command(PROJECT, payload, cache, run / 'probe', dataset_root, data_manifest)
        budget_path = None
        if payload['additional_budget'] is not None:
            budget_path = run / 'additional-budget.json'
            write_json(budget_path, payload['additional_budget'])
            command += ['--additional-budget', str(budget_path)]
        os.environ['PYTHONPATH'] = str(PROJECT / 'src')
        event('probe_started', compute_cap_seconds=PROBE_SECONDS)
        result = guard_probe(command, run / 'guard', roots, session_started=LAUNCHED,
                             result_path=run / 'probe/receipt.json', additional_budget=budget_path,
                             compute_started=compute_started)
        receipt.update(status=result['status'], full_run_eligible=result['full_run_eligible'], result=result,
                       **{key: result.get('probe', {}).get(key, 0) for key in (
                           'raw_history_requests', 'successful_raw_history_reads', 'completed_unique_histories')})
        event('probe_finished', status=result['status'], raw_history_requests=receipt['raw_history_requests'])
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt.update(status='failed', full_run_eligible=False, error_type=type(error).__name__,
                       original_error=sanitized(error))
        (run / 'original-error.txt').write_text(sanitized(traceback.format_exc()))
        event('failed', error_type=type(error).__name__, original_error=sanitized(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        partial = PROJECT / CYCLE / 'weights/1_lead_ECGFounder.pth.partial'
        receipt['checkpoint_partial_bytes'] = partial.stat().st_size if partial.is_file() else 0
        probe_receipt = run / 'probe/receipt.json'
        if not probe_receipt.is_file():
            probe_receipt = run / 'probe/execution-state.json'
        if probe_receipt.is_file():
            try:
                probe_state = json.loads(probe_receipt.read_text())
                receipt.update({key: probe_state.get(key, 0) for key in (
                    'raw_history_requests', 'successful_raw_history_reads', 'completed_unique_histories')})
            except (OSError, ValueError, AttributeError) as error:
                receipt['partial_receipt_error'] = sanitized(error)
        receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    return 0 if receipt['status'] == 'passed' else 1


def main(argv=None):
    if 'EMBEDDED_PAYLOAD' in globals():
        payload = json.loads(zlib.decompress(base64.b64decode(globals()['EMBEDDED_PAYLOAD'], validate=True)))
        return remote_probe(payload)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cycle', 'selection', 'selection-sha256', 'output', 'identity'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--additional-budget')
    parser.add_argument('--attempt', help='Fixed repair-panel-20261014-b/spec.md declaration')
    args = parser.parse_args(argv)
    package(args.root, args.cycle, args.selection, args.selection_sha256, args.output, args.identity,
            args.additional_budget, attempt=args.attempt)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
