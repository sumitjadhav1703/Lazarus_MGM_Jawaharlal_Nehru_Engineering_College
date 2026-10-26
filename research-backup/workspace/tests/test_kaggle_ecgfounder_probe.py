import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest


def wrapper():
    path = Path(__file__).resolve().parents[1] / 'scripts/kaggle_ecgfounder_probe.py'
    assert path.is_file(), 'ECGFounder private probe wrapper is missing'
    spec = importlib.util.spec_from_file_location('ecgfounder_wrapper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(data):
    return hashlib.sha256(data).hexdigest()


def payload_fixture(tmp_path):
    root = tmp_path / 'source'
    runner = root / 'scripts/run_ecgfounder_probe.py'
    runner.parent.mkdir(parents=True)
    runner.write_text('print("current source")\n')
    selection = root / 'reports/ml-investigation/next-cycle/ecgfounder-cycle-20261013/selected.json'
    selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps({'inputs': {'source_sha256': {
        'scripts/run_ecgfounder_probe.py': digest(runner.read_bytes())}, 'files_sha256': {}},
        'requests': []}))
    return root, runner, selection


def test_payload_binds_current_source_and_reviewed_selection(tmp_path):
    module = wrapper()
    root, runner, selection = payload_fixture(tmp_path)
    payload = module.build_payload(root, ['scripts/run_ecgfounder_probe.py'], selection, digest(selection.read_bytes()))
    project = tmp_path / 'remote'
    module.overlay(project, payload)
    assert (project / runner.relative_to(root)).read_bytes() == runner.read_bytes()
    assert (project / selection.relative_to(root)).read_bytes() == selection.read_bytes()
    runner.write_text('old or changed source')
    with pytest.raises(ValueError, match='binding'):
        module.build_payload(root, ['scripts/run_ecgfounder_probe.py'], selection, digest(selection.read_bytes()))


def test_wrong_selection_hash_refused_before_package_writes(tmp_path):
    module = wrapper()
    root, _, selection = payload_fixture(tmp_path)
    with pytest.raises(ValueError, match='selection'):
        module.build_payload(root, [], selection, '0' * 64)
    assert not (tmp_path / 'remote').exists()


@pytest.mark.parametrize('name', ['../escape.py', '/tmp/escape.py', 'src/../escape.py',
                                  'reports/ml-investigation/next-cycle/ecgfounder-cycle-20261013/weights/x.pth'])
def test_overlay_rejects_escape_and_weight_artifacts_before_writing(tmp_path, name):
    module = wrapper()
    payload = {'files': {name: 'eA=='}, 'manifest': {name: digest(b'x')}}
    payload['identity'] = module.manifest_identity(payload['manifest'])
    with pytest.raises(ValueError, match='path'):
        module.overlay(tmp_path / 'remote', payload)
    assert not (tmp_path / 'remote').exists()


def test_overlay_checks_all_hashes_before_replacing_leaf_symlink(tmp_path):
    module = wrapper()
    root, _, selection = payload_fixture(tmp_path)
    payload = module.build_payload(root, ['scripts/run_ecgfounder_probe.py'], selection, digest(selection.read_bytes()))
    project = tmp_path / 'remote'
    (project / 'scripts').mkdir(parents=True)
    immutable = tmp_path / 'immutable.py'
    immutable.write_text('immutable old source')
    target = project / 'scripts/run_ecgfounder_probe.py'
    target.symlink_to(immutable)
    changed = json.loads(json.dumps(payload))
    changed['files'][str(selection.relative_to(root))] = 'eA=='
    with pytest.raises(ValueError, match='checksum'):
        module.overlay(project, changed)
    assert target.is_symlink()
    module.overlay(project, payload)
    assert not target.is_symlink()
    assert immutable.read_text() == 'immutable old source'


def test_overlay_refuses_symlink_parent_without_touching_destination(tmp_path):
    module = wrapper()
    root, _, selection = payload_fixture(tmp_path)
    payload = module.build_payload(root, ['scripts/run_ecgfounder_probe.py'], selection, digest(selection.read_bytes()))
    project, outside = tmp_path / 'remote', tmp_path / 'outside'
    project.mkdir()
    outside.mkdir()
    (project / 'scripts').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        module.overlay(project, payload)
    assert list(outside.iterdir()) == []


@pytest.fixture
def http_payload(monkeypatch):
    import io
    import urllib.request

    class Response(io.BytesIO):
        calls = 0
        payload = b'bounded checkpoint fixture'
        headers = {'Content-Length': str(len(payload))}

    def open_payload(url, timeout):
        assert timeout > 0
        Response.calls += 1
        return Response(Response.payload)

    monkeypatch.setattr(urllib.request, 'urlopen', open_payload)
    return Response, 'https://fixture.invalid/checkpoint?token=do-not-log'


def test_checkpoint_atomic_exact_transfer_and_no_implicit_retry(tmp_path, http_payload):
    module = wrapper()
    handler, url = http_payload
    path = tmp_path / 'checkpoint.pth'
    events = []
    result = module.acquire_checkpoint(url, path, len(handler.payload), digest(handler.payload), events.append)
    assert path.read_bytes() == handler.payload and not path.with_suffix('.pth.partial').exists()
    assert result['sha256'] == digest(handler.payload)
    assert events[-1]['completed_bytes'] == len(handler.payload)
    with pytest.raises(FileExistsError):
        module.acquire_checkpoint(url, path, len(handler.payload), digest(handler.payload), events.append)
    assert handler.calls == 1
    assert 'do-not-log' not in json.dumps(events + [result])


@pytest.mark.parametrize('size,expected_sha', [(5, digest(b'x')), (26, '0' * 64), (100, digest(b'x'))])
def test_bad_checkpoint_size_or_hash_fails_once_preserving_partial(tmp_path, http_payload, size, expected_sha):
    module = wrapper()
    handler, url = http_payload
    path = tmp_path / 'checkpoint.pth'
    with pytest.raises(ValueError):
        module.acquire_checkpoint(url, path, size, expected_sha, lambda event: None)
    assert not path.exists() and path.with_suffix('.pth.partial').exists()
    assert handler.calls == 1
    with pytest.raises(FileExistsError):
        module.acquire_checkpoint(url, path, size, expected_sha, lambda event: None)
    assert handler.calls == 1


def test_private_metadata_is_fresh_and_has_only_original_input_source():
    module = wrapper()
    metadata = module.kernel_metadata('sumit1703/cardiotwin-ecgfounder-probe-20261013-a')
    assert metadata['is_private'] is True and metadata['enable_internet'] is True
    assert metadata['dataset_sources'] == ['sumit1703/cardiotwin-iridia-development-gpu-inputs']
    with pytest.raises(ValueError, match='identity'):
        module.kernel_metadata('sumit1703/cardiotwin-waveform-screen-20261011')


def test_bootstrap_time_and_cleanup_reserve_bound_compute_allowance():
    module = wrapper()
    assert module.probe_session_limit(100., now=200.) == 2200.
    assert module.probe_session_limit(100., now=41000.) == 41400.
    with pytest.raises(TimeoutError):
        module.probe_session_limit(100., now=41201.)


def test_guard_requires_passed_runner_receipt_and_missing_budget_stays_ineligible(tmp_path):
    module = wrapper()
    target = tmp_path / 'probe'
    target.mkdir()
    receipt = target / 'receipt.json'
    code = f'from pathlib import Path;Path({str(receipt)!r}).write_text({json.dumps({"status": "passed", "full_run_eligible": True})!r})'
    result = module.guard_probe([sys.executable, '-c', code], tmp_path / 'guard', [tmp_path],
                                session_started=time.monotonic(), result_path=receipt)
    assert result['status'] == 'passed' and result['full_run_eligible'] is False
    receipt.write_text('{"status":"failed"}')
    result = module.guard_probe([sys.executable, '-c', 'pass'], tmp_path / 'failed-guard', [tmp_path],
                                session_started=time.monotonic(), result_path=receipt)
    assert result['status'] == 'failed' and result['full_run_eligible'] is False


def test_compute_timeout_preserves_child_output_and_stops_owned_child(tmp_path):
    module = wrapper()
    result = module.guard_probe([sys.executable, '-u', '-c', 'import time;print("started");time.sleep(10)'],
                                tmp_path / 'guard', [tmp_path], session_started=time.monotonic(),
                                result_path=tmp_path / 'missing.json', compute_seconds=.1)
    assert result['status'] == 'failed' and result['guard']['status'] == 'timed_out'
    assert 'started' in result['guard']['child_output_tail']
    assert result['guard']['output_reader_finished']


def test_guard_refuses_output_reservation_before_child(tmp_path, monkeypatch):
    module = wrapper()
    monkeypatch.setattr(module, 'storage_usage', lambda roots: {'apparent_bytes': 24_900_000_000,
                                                              'allocated_bytes': 24_900_000_000})
    with pytest.raises(RuntimeError, match='reservation'):
        module.reserve_storage([tmp_path], checkpoint_bytes=369807481)
    assert not (tmp_path / 'guard').exists()


def test_post_checkpoint_hashing_consumes_same_compute_window(tmp_path):
    module = wrapper()
    started = time.monotonic()
    with pytest.raises(TimeoutError, match='allowance'):
        module.guard_probe([sys.executable, '-c', 'raise AssertionError("must not start")'],
                           tmp_path / 'expired', [tmp_path], session_started=started - 2000,
                           compute_started=started - 1801, result_path=tmp_path / 'receipt.json')
    assert not (tmp_path / 'expired/child.log').exists()


@pytest.mark.parametrize('python_version,torch_version,device', [((3, 11), '2.6.0', 'Tesla T4'),
    ((3, 15), '2.6.0', 'Tesla T4'), ((3, 12), '2.5.1', 'Tesla T4'),
    ((3, 12), '3.0.0', 'Tesla T4'), ((3, 12), '2.6.0a0', 'Tesla T4'),
    ((3, 12), '2.6.0', 'Tesla V100')])
def test_unsupported_runtime_fails_before_checkpoint(python_version, torch_version, device):
    module = wrapper()
    with pytest.raises(RuntimeError, match='runtime'):
        module.validate_runtime(python_version, torch_version, device)


def test_supported_cuda_runtime_accepts_stable_torch_build():
    wrapper().validate_runtime((3, 12), '2.6.0+cu124', 'Tesla T4')


@pytest.mark.parametrize('body', ['not json', '[]', '{"status":"running"}', None])
def test_missing_or_malformed_runner_receipt_is_failed(tmp_path, body):
    module = wrapper()
    target = tmp_path / 'probe'
    target.mkdir()
    receipt = target / 'receipt.json'
    if body is not None:
        receipt.write_text(body)
    result = module.guard_probe([sys.executable, '-c', 'pass'], tmp_path / 'guard', [tmp_path],
                                session_started=time.monotonic(), result_path=receipt)
    assert result['status'] == 'failed' and result['original_error']
    assert json.loads((tmp_path / 'guard/wrapper-result.json').read_text())['status'] == 'failed'


def test_failed_child_preserves_partial_reader_attempt_counts(tmp_path):
    module = wrapper()
    target = tmp_path / 'probe'
    target.mkdir()
    receipt = target / 'receipt.json'
    receipt.write_text(json.dumps({'status': 'failed', 'raw_history_requests': 3,
                                  'successful_raw_history_reads': 2, 'completed_unique_histories': 1}))
    result = module.guard_probe([sys.executable, '-c', 'import sys;print("original reader failure");sys.exit(7)'],
                                tmp_path / 'guard', [tmp_path], session_started=time.monotonic(),
                                result_path=receipt)
    assert result['status'] == 'failed' and result['guard']['exit_code'] == 7
    assert 'original reader failure' in result['guard']['child_output_tail']
    assert result['probe']['raw_history_requests'] == 3
    assert result['probe']['successful_raw_history_reads'] == 2
    assert result['probe']['completed_unique_histories'] == 1


@pytest.mark.parametrize('layout', ['approved-v1', 'datasets/sumit1703/approved-v1'])
def test_v1_flat_layout_uses_preserved_746_member_inventory_without_raw_reads(tmp_path, monkeypatch, layout):
    module = wrapper()
    source = Path(__file__).resolve().parents[1] / 'reports/ml-investigation/kaggle-upload/remote-reconciliation-20261009.json'
    inventory = json.loads(source.read_text())['files']
    inputs = tmp_path / 'input'
    mount = inputs / layout
    for entry in inventory:
        target = mount / entry['name']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    compact = mount / 'cardiotwin-gpu-bundle/data/iridia-development-cache-recovered/compact.json'
    compact.write_text('{}')
    original_open = Path.open

    def no_raw_open(path, *args, **kwargs):
        assert not path.name.startswith('record_'), 'Raw content must not be read during layout validation'
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', no_raw_open)
    source_cache, raw_root = module.locate_inputs(inputs, digest(b'{}'))
    assert source_cache == compact.parent and raw_root == mount
    (mount / 'record_000_ecg_00.h5').unlink()
    with pytest.raises(ValueError, match='membership'):
        module.locate_inputs(inputs, digest(b'{}'))


def package_fixture(tmp_path):
    module = wrapper()
    root, _, selection = payload_fixture(tmp_path)
    cycle = root / module.CYCLE
    own_source = Path(module.__file__).read_bytes()
    for name, content in {'scripts/kaggle_ecgfounder_probe.py': own_source,
                          'scripts/kaggle_precursor_cycle.py': b'pass\n',
                          'scripts/kaggle_cycle.py': b'pass\n',
                          'data/iridia-development-cache-recovered/compact.json': b'{}',
                          module.CYCLE + '/spec.md': b'frozen synthetic spec\n',
                          'reports/ml-investigation/next-cycle/original-audit/roles.json': b'{}'}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    pin = dict(bytes=369807481, sha256='f863a38897fb49a27fec7e44008ea3c7bdbd29c77fa4a02ecbb8c56df4f37603',
               revision='d9b1793951b2342f5f7e84f1ac03cd37f8a08724', repository='PKUDigitalHealth/ECGFounder',
               file='1_lead_ECGFounder.pth')
    (cycle / 'probe-spec.json').write_text(json.dumps({'checkpoint': pin}))
    (cycle / 'declaration-binding.json').write_text('{"status":"frozen"}')
    selected = json.loads(selection.read_text())
    selected['inputs'].update(declaration_sha256=digest((cycle / 'declaration-binding.json').read_bytes()),
                              compact_binding={'compact_sha256': digest(b'{}')})
    selected['inputs']['files_sha256'] = {name: digest((root / name).read_bytes()) for name in
        (module.CYCLE + '/spec.md', module.CYCLE + '/probe-spec.json',
         'reports/ml-investigation/next-cycle/original-audit/roles.json')}
    selection.write_text(json.dumps(selected))
    return module, root, cycle, selection


def test_synthetic_package_binds_generated_code_and_private_metadata(tmp_path):
    module, root, cycle, selection = package_fixture(tmp_path)
    output = tmp_path / 'package'
    module.package(root, cycle, selection, digest(selection.read_bytes()), output,
                   'sumit1703/cardiotwin-ecgfounder-probe-20261014-a')
    receipt = json.loads((output / 'package-receipt.json').read_text())
    assert receipt['train_sha256'] == digest((output / 'train.py').read_bytes())
    assert receipt['raw_bytes_read'] == 0 and receipt['checkpoint_uploaded'] is False
    assert json.loads((output / 'kernel-metadata.json').read_text())['is_private'] is True
    compile((output / 'train.py').read_text(), 'train.py', 'exec')
    assert not any('weights' in name for name in receipt['files_sha256'])
    with pytest.raises(FileExistsError):
        module.package(root, cycle, selection, digest(selection.read_bytes()), output,
                       'sumit1703/cardiotwin-ecgfounder-probe-20261014-a')


def test_package_refuses_changed_declaration_before_any_output(tmp_path):
    module, root, cycle, selection = package_fixture(tmp_path)
    (cycle / 'declaration-binding.json').write_text('changed declaration')
    with pytest.raises(ValueError, match='declaration'):
        module.package(root, cycle, selection, digest(selection.read_bytes()), tmp_path / 'package',
                       'sumit1703/cardiotwin-ecgfounder-probe-20261014-a')
    assert not (tmp_path / 'package').exists()


def test_bootstrap_failure_has_atomic_sanitized_receipt_without_reading_raw(tmp_path):
    import shutil
    import subprocess
    import uuid

    module = wrapper()
    name = 'cardiotwin-ecgfounder-probe-20261014-test-' + uuid.uuid4().hex
    working, inputs = tmp_path / 'working', tmp_path / 'inputs'
    working.mkdir()
    inputs.mkdir()
    bad_name = 'https://fixture.invalid/source?token=secret-value'
    payload = {'files': {bad_name: 'eA=='}, 'manifest': {bad_name: digest(b'x')},
               'identity_name': 'sumit1703/' + name, 'selection_sha256': '0' * 64}
    payload['identity'] = module.manifest_identity(payload['manifest'])
    code = ('import importlib.util;from pathlib import Path;'
            f's=importlib.util.spec_from_file_location("wrapper",{module.__file__!r});'
            'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
            f'raise SystemExit(m.remote_probe({payload!r},working=Path({str(working)!r}),'
            f'inputs=Path({str(inputs)!r})))')
    scratch = Path('/tmp') / name
    assert not scratch.exists()
    try:
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=5)
        assert result.returncode == 1, result.stderr
        receipt = json.loads((working / name / 'bootstrap-receipt.json').read_text())
        assert receipt['status'] == 'failed' and receipt['checkpoint_partial_bytes'] == 0
        assert receipt['raw_history_requests'] == receipt['successful_raw_history_reads'] == 0
        assert receipt['completed_unique_histories'] == receipt['provenance_files_read'] == 0
        assert receipt['cleanup_reserve_seconds'] == 300
        all_output = result.stdout + result.stderr + ''.join(p.read_text() for p in (working / name).iterdir())
        assert 'secret-value' not in all_output and '<URL redacted>' in all_output
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def test_timeout_recovers_reader_counts_when_child_receipt_never_written(tmp_path):
    module = wrapper()
    probe = tmp_path / 'probe'
    probe.mkdir()
    state = probe / 'execution-state.json'
    code = ('from pathlib import Path;import time;'
            f'Path({str(state)!r}).write_text(\'{{"raw_history_requests":3,"successful_raw_history_reads":2,'
            '"completed_unique_histories":1}\');time.sleep(10)')
    result = module.guard_probe([sys.executable, '-u', '-c', code], tmp_path / 'guard', [tmp_path],
                                session_started=time.monotonic(), result_path=probe / 'receipt.json',
                                compute_seconds=.1)
    assert result['status'] == 'failed' and result['guard']['status'] == 'timed_out'
    assert result['probe']['raw_history_requests'] == 3
    assert result['probe']['successful_raw_history_reads'] == 2
    assert result['probe']['completed_unique_histories'] == 1


def panel_package_fixture(tmp_path):
    module, root, cycle, selection = package_fixture(tmp_path)
    attempt = cycle / 'repair-panel-20261014-b/spec.md'
    attempt.parent.mkdir()
    files = [attempt, cycle / 'repair-design-20261014.md',
             attempt.parent / 'dense-dispatch-timing-addendum.md',
             cycle / 'selection-20261014/selected.json']
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixed synthetic ' + path.name)
    for constant, path in zip(('PANEL_SPEC_SHA', 'PANEL_DESIGN_SHA', 'PANEL_TIMING_SHA', 'OLD_SELECTION_SHA'),
                              files, strict=True):
        setattr(module, constant, digest(path.read_bytes()))
    selected = json.loads(selection.read_text())
    bindings = selected['inputs']['files_sha256']
    bindings.update({str(p.relative_to(root)): digest(p.read_bytes()) for p in files})
    declaration = attempt.parent / 'declaration-binding.json'
    declaration.write_text(json.dumps(dict(files_sha256={**bindings, **selected['inputs']['source_sha256']},
        protected_files_sha256={'README.md': 'opaque-parent-attestation'})))
    bindings[str(declaration.relative_to(root))] = digest(declaration.read_bytes())
    selected['attempt'] = dict(identity='ecgfounder-metadata-supported-panel-20261014-b',
        spec=str(attempt.relative_to(root)), spec_sha256=digest(attempt.read_bytes()),
        design_sha256=digest(files[1].read_bytes()), timing_addendum_sha256=digest(files[2].read_bytes()),
        old_selection_sha256=digest(files[3].read_bytes()), declaration=str(declaration.relative_to(root)),
        declaration_sha256=digest(declaration.read_bytes()))
    selection.write_text(json.dumps(selected))
    return module, root, cycle, selection, attempt


def test_panel_package_requires_explicit_fixed_attempt_and_remote_propagation(tmp_path):
    module, root, cycle, selection, attempt = panel_package_fixture(tmp_path)
    output = tmp_path / 'package'
    module.package(root, cycle, selection, digest(selection.read_bytes()), output,
                   'sumit1703/cardiotwin-ecgfounder-probe-20261014-b', attempt=attempt)
    receipt = json.loads((output / 'package-receipt.json').read_text())
    assert receipt['attempt'] == str(attempt.relative_to(root))
    assert 'README.md' not in receipt['files_sha256']
    payload = dict(selection=str(selection.relative_to(root)), selection_sha256=digest(selection.read_bytes()),
                   audit='reports/ml-investigation/next-cycle/original-audit', attempt=str(attempt.relative_to(root)))
    data_manifest = tmp_path / 'data-manifest.json'
    data_manifest.write_text('{}')
    command = module.runner_command(root, payload, root / 'data/cache', tmp_path / 'probe',
                                    tmp_path / 'flat-raw', data_manifest)
    assert command[command.index('--attempt') + 1] == str(attempt)
    payload['attempt'] = None
    assert '--attempt' not in module.runner_command(root, payload, root / 'data/cache', tmp_path / 'probe',
                                                  tmp_path / 'flat-raw', data_manifest)


@pytest.mark.parametrize('fault', ['omitted', 'wrong_path', 'old_identity', 'changed_spec', 'missing_binding', 'budget'])
def test_panel_package_fails_before_writes_on_attempt_mismatch(tmp_path, fault):
    module, root, cycle, selection, attempt = panel_package_fixture(tmp_path)
    identity = 'sumit1703/cardiotwin-ecgfounder-probe-20261014-b'
    budget = None
    if fault == 'omitted':
        attempt = None
    elif fault == 'wrong_path':
        attempt = cycle / 'spec.md'
    elif fault == 'old_identity':
        identity = 'sumit1703/cardiotwin-ecgfounder-probe-20261014-a'
    elif fault == 'changed_spec':
        attempt.write_text('changed declaration')
    elif fault == 'missing_binding':
        selected = json.loads(selection.read_text())
        del selected['inputs']['files_sha256'][str(attempt.relative_to(root))]
        selection.write_text(json.dumps(selected))
    else:
        budget = tmp_path / 'budget.json'
        budget.write_text('{}')
    with pytest.raises(ValueError):
        module.package(root, cycle, selection, digest(selection.read_bytes()), tmp_path / 'package', identity,
                       budget, attempt=attempt)
    assert not (tmp_path / 'package').exists()


def test_panel_package_rejects_consistently_rebound_wrong_fixed_spec(tmp_path):
    module, root, cycle, selection, attempt = panel_package_fixture(tmp_path)
    module.PANEL_SPEC_SHA = digest(attempt.read_bytes())
    attempt.write_text('different scientific specification')
    selected = json.loads(selection.read_text())
    name = str(attempt.relative_to(root))
    selected['inputs']['files_sha256'][name] = digest(attempt.read_bytes())
    selected['attempt']['spec_sha256'] = digest(attempt.read_bytes())
    declaration = attempt.parent / 'declaration-binding.json'
    declared = json.loads(declaration.read_text())
    declared['files_sha256'][name] = digest(attempt.read_bytes())
    declaration.write_text(json.dumps(declared))
    selected['inputs']['files_sha256'][str(declaration.relative_to(root))] = digest(declaration.read_bytes())
    selected['attempt']['declaration_sha256'] = digest(declaration.read_bytes())
    selection.write_text(json.dumps(selected))
    with pytest.raises(ValueError, match='fixed|Fixed'):
        module.package(root, cycle, selection, digest(selection.read_bytes()), tmp_path / 'package',
                       'sumit1703/cardiotwin-ecgfounder-probe-20261014-b', attempt=attempt)
    assert not (tmp_path / 'package').exists()
