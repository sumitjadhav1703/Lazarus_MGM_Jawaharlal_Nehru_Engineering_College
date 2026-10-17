import json
import runpy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("wrong_membership", [False, True])
@pytest.mark.parametrize(
    "candidate,representation", [("rr16", "rr"), ("waveform_resnet16", "waveform_resnet")]
)
def test_kaggle_entrypoint_reports_phases_and_preserves_failure(
    tmp_path, monkeypatch, capsys, wrong_membership, candidate, representation
):
    from cardiotwin import data, gpu_telemetry, investigation, neural

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"seed": 42, "development_groups": ["g"], "group_ids": {"p": "g"}}))
    index = dict(
        allowed_groups=["wrong" if wrong_membership else "g"],
        records=[{}],
        config={},
        input_sha256="source",
        rows_sha256="rows",
        sha256="index",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "mock T4")
    monkeypatch.setattr(gpu_telemetry, "GpuTelemetry", nullcontext)
    monkeypatch.setattr(investigation, "read_cache", lambda _: ([], [], index))
    monkeypatch.setattr(data, "list_records", lambda _: [SimpleNamespace(patient_id="p")])
    verified = []
    monkeypatch.setattr(investigation, "validate_live_inputs", lambda *args: verified.append(True))
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "budget_exhausted", "combined": {}}

    monkeypatch.setattr(neural, "run_neural_ladder", run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "script",
            "--data",
            "data",
            "--cache",
            "cache",
            "--split",
            str(split),
            "--epochs",
            "4",
            "--models",
            candidate,
            "--early-stop-weighting",
            "patient_class",
        ],
    )
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/kaggle_investigation.py"))[
        "main"
    ]
    if wrong_membership:
        with pytest.raises(ValueError, match="development cohort"):
            main()
        assert not calls and not verified
    else:
        main()
        assert verified
        assert calls[0]["screening_epochs"] == 4
        assert calls[0]["max_training_windows"] == 10000
        assert calls[0]["specs"] == [(representation, 16)]
        assert calls[0]["early_stop_weighting"] == "patient_class"
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{"')]
    assert events[0]["stage"] == "cuda_check"
    assert (
        next(e for e in events if e["stage"] == "cuda_check" and e["status"] == "done")["scheduled_device"]
        == "cuda:0"
    )
    assert events[-1]["status"] == ("failed" if wrong_membership else "done")
    if not wrong_membership:
        assert events[-1]["result_status"] == "budget_exhausted"
