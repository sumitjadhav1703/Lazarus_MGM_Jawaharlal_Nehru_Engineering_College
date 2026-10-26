import importlib.util
import sys
import time
from pathlib import Path

import pytest


def wrapper():
    path = Path(__file__).resolve().parents[1] / "scripts/kaggle_precursor_cycle.py"
    assert path.is_file(), "precursor guarded execution is missing"
    spec = importlib.util.spec_from_file_location("kaggle_precursor_cycle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guard_refuses_reservation_before_starting_child(tmp_path, monkeypatch):
    module = wrapper()
    monkeypatch.setattr(module, "dataset_bytes", lambda roots, *, allocated=False: 24900000000)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="reservation"):
        module.run_guarded([sys.executable, "-c", "raise AssertionError('must not run')"], output, [tmp_path],
                           session_started=time.monotonic())
    assert not (output / "child.log").exists()


def test_guard_preserves_original_failure_and_refuses_implicit_retry(tmp_path):
    module = wrapper()
    output = tmp_path / "run"
    result = module.run_guarded([sys.executable, "-c", "import sys;sys.stderr.write('original failure');sys.exit(7)"],
                                output, [tmp_path], session_started=time.monotonic())
    assert result["exit_code"] == 7
    assert "original failure" in result["child_output_tail"]
    with pytest.raises(ValueError, match="already exists"):
        module.run_guarded([], output, [tmp_path], session_started=time.monotonic())


def test_bootstrap_elapsed_is_charged_and_output_allocation_guarded(tmp_path):
    module = wrapper()
    with pytest.raises(TimeoutError, match="allowance"):
        module.run_guarded([], tmp_path / "late", [tmp_path], session_started=time.monotonic() - 41101)
    result = module.run_guarded([sys.executable, "-c", "print('done')"], tmp_path / "tiny", [tmp_path],
                                session_started=time.monotonic(), output_reserve_bytes=1)
    assert result["status"] == "failed"
    assert "output reservation" in result["original_error"]


def test_guard_counts_child_artifacts_outside_its_log_directory(tmp_path):
    module = wrapper()
    target = tmp_path / 'experiment'
    target.mkdir()
    command = [sys.executable, '-c', f"from pathlib import Path;Path({str(target/'payload')!r}).write_bytes(b'x'*10000)"]
    result = module.run_guarded(command, tmp_path/'guard', [tmp_path], session_started=time.monotonic(),
                                experiment_roots=[target], output_reserve_bytes=9000)
    assert result['status'] == 'failed'
    assert 'output reservation' in result['original_error']


def test_spawn_failure_has_atomic_original_error_receipt(tmp_path, monkeypatch):
    import json
    module = wrapper()
    def fail(*args, **kwargs):
        raise OSError('original spawn failure')
    monkeypatch.setattr(module, 'run_child', fail)
    output = tmp_path / 'guard'
    with pytest.raises(OSError, match='original spawn failure'):
        module.run_guarded(['missing'], output, [tmp_path], session_started=time.monotonic())
    result = json.loads((output/'receipt.json').read_text())
    assert result['status'] == 'failed' and result['original_error'] == 'original spawn failure'


@pytest.mark.skipif(sys.platform == 'win32', reason='Owned process-group cleanup requires POSIX')
def test_wrapper_sigterm_closes_owned_child_group_and_preserves_error(tmp_path):
    import json
    import os
    import signal
    import subprocess

    ready = tmp_path/'child.pid'
    code = f"import os,time;open({str(ready)!r},'w').write(str(os.getpid()));time.sleep(30)"
    output = tmp_path/'guard'
    script = Path(__file__).resolve().parents[1]/'scripts/kaggle_precursor_cycle.py'
    supervisor = subprocess.Popen([sys.executable, str(script), '--output', str(output), '--storage-root', str(tmp_path),
                                   '--', sys.executable, '-c', code], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    child_pid = None
    try:
        deadline = time.monotonic()+5
        while not ready.exists() and supervisor.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists(), 'Synthetic child did not start'
        child_pid = int(ready.read_text())
        supervisor.send_signal(signal.SIGTERM)
        _, stderr = supervisor.communicate(timeout=5)
        assert supervisor.returncode != 0
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        receipt = json.loads((output/'receipt.json').read_text())
        assert receipt['status'] == 'failed'
        assert receipt['original_error'] == 'Supervisor received SIGTERM'
        assert b'Supervisor received SIGTERM' in stderr
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait(timeout=5)
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize('failure', ['time', 'reservation', 'measurement'])
def test_guard_preflight_failures_have_original_error_and_stage_receipt(tmp_path, monkeypatch, failure):
    import json
    module = wrapper()
    started = time.monotonic() - (41101 if failure == 'time' else 0)
    if failure == 'reservation':
        monkeypatch.setattr(module, 'dataset_bytes', lambda *a, **k: 24900000000)
    elif failure == 'measurement':
        def fail(*a, **k):
            if k.get('allocated'):
                raise OSError('original allocated measurement failure')
            return 17
        monkeypatch.setattr(module, 'dataset_bytes', fail)
    output = tmp_path/'guard'
    with pytest.raises((TimeoutError, RuntimeError, OSError)) as error:
        module.run_guarded([], output, [tmp_path], session_started=started)
    receipt = json.loads((output/'receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['retry'] is False
    assert receipt['original_error'] == str(error.value)
    assert receipt['stage'] == ('time_preflight' if failure == 'time' else 'storage_reservation' if failure == 'reservation' else 'storage_measurement')
    if failure == 'measurement':
        assert receipt['storage_before'] == {'apparent_bytes': 17}
    assert not (output/'child.log').exists()
