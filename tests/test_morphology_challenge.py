"""Shared-reference challenge must retain failed sentinels and source provenance."""

import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest


def runner():
    spec = importlib.util.spec_from_file_location("shared_audit", "scripts/run_morphology_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_challenge_cannot_average_away_a_failed_required_history(tmp_path, monkeypatch):
    module = runner()
    assert hasattr(module, "measure_shared_challenge"), "shared challenge missing"
    sentinels = [("iridia-af:patient_021", "record_021", 56520.),
                 ("iridia-af:patient_150", "record_165", 167910.),
                 ("iridia-af:patient_137", "record_151", 84660.)]
    selected = [dict(fold=f, patient_id=p, record_id=r, timestamp=t, row_index=f*12+j)
                for f in range(3) for j, (p, r, t) in enumerate(sentinels+[('other', 'r', 630.)]*9)]
    (tmp_path / "selected.json").write_text(json.dumps(selected))
    records = {r["record_id"]: SimpleNamespace(fs=200) for r in selected}
    calls = []

    def read(record, start, end):
        assert json.loads((tmp_path / "selected.json").read_text()) == selected
        assert end-start == 600
        return np.zeros((120000, 2))

    def extract(signal, fs, *, anchor_mode, overlays):
        assert anchor_mode == "shared-channel0-v1" and overlays
        calls.append(1)
        return dict(available=len(calls) != 1, features={}, channels=[])

    monkeypatch.setattr(module, "extract_morphology", extract)
    report = module.measure_shared_challenge(selected, records, read, tmp_path, deadline=float("inf"))
    assert report["coverage"]["status"] == "passed"
    assert report["status"] == "failed"
    assert len(calls) == 36 and len((tmp_path / "measurements.jsonl").read_text().splitlines()) == 36
    assert not report["sentinels"]["record_021"]["passed"]
    with pytest.raises(FileExistsError):
        module.measure_shared_challenge(selected, records, read, tmp_path, deadline=float("inf"))


def test_shared_challenge_refuses_failed_synthetic_receipt(tmp_path):
    module = runner()
    assert hasattr(module, "synthetic_binding"), "synthetic prerequisite validation missing"
    (tmp_path / "receipt.json").write_text(json.dumps(dict(status="failed", cases=135)))
    with pytest.raises(ValueError, match="passed"):
        module.synthetic_binding(tmp_path)


def test_shared_challenge_checks_actual_source_and_case_identity(tmp_path):
    module = runner()
    assert hasattr(module, "synthetic_binding"), "synthetic prerequisite validation missing"
    from cardiotwin.data import file_hash
    from cardiotwin.investigation import _digest

    source = {name: file_hash(name) for name in ["src/cardiotwin/morphology.py", "src/cardiotwin/features.py",
                                                "scripts/run_anchor_noise_validation.py"]}
    (tmp_path / "cases.jsonl").write_text("{}\n"*135)
    declaration = dict(mode="shared-channel0-v1", source=dict(files=source, sha256=_digest(source)))
    (tmp_path / "predeclaration.json").write_text(json.dumps(declaration))
    receipt = dict(status="passed", cases=135, failed_checks=[],
        cases_sha256=file_hash(tmp_path / "cases.jsonl"),
        predeclaration_sha256=file_hash(tmp_path / "predeclaration.json"))
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    assert module.synthetic_binding(tmp_path)
    (tmp_path / "cases.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="identity"):
        module.synthetic_binding(tmp_path)


def test_shared_challenge_rejects_changed_inherited_declaration(tmp_path):
    module = runner()
    from cardiotwin.data import file_hash
    from cardiotwin.investigation import _digest

    inherited = tmp_path / "inherited-spec.md"
    inherited.write_text("original fixed quality gates")
    source = {name: file_hash(name) for name in ["src/cardiotwin/morphology.py", "src/cardiotwin/features.py",
                                                "scripts/run_anchor_noise_validation.py", str(inherited)]}
    (tmp_path / "cases.jsonl").write_text("{}\n"*135)
    (tmp_path / "predeclaration.json").write_text(json.dumps(dict(mode="shared-channel0-v1",
        source=dict(files=source, sha256=_digest(source)))))
    (tmp_path / "receipt.json").write_text(json.dumps(dict(status="passed", cases=135, failed_checks=[],
        cases_sha256=file_hash(tmp_path / "cases.jsonl"),
        predeclaration_sha256=file_hash(tmp_path / "predeclaration.json"))))
    inherited.write_text("changed quality gates")
    with pytest.raises(ValueError, match="source|declaration"):
        module.synthetic_binding(tmp_path)


def test_runner_already_rejects_different_primary_spec_before_challenge_reads(tmp_path, monkeypatch):
    module = runner()
    from cardiotwin.data import file_hash
    from cardiotwin.investigation import _digest

    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "pairs.jsonl").write_text("")
    (audit / "roles.json").write_text(json.dumps(dict(cache_index_sha256="index", roles=[])))
    (audit / "receipt.json").write_text(json.dumps(dict(status="passed", metadata_status="passed",
        pair_support_status="passed", cache_index_sha256="index", roles_sha256=_digest([]),
        pairs_sha256=file_hash(audit / "pairs.jsonl"))))
    validation = tmp_path / "validation.json"
    binding = {"index_sha256": "index"}
    validation.write_text(json.dumps(dict(status="passed", binding=binding)))
    synthetic = tmp_path / "synthetic"
    synthetic.mkdir()
    (synthetic / "predeclaration.json").write_text(json.dumps(dict(spec_sha256="different")))
    spec = tmp_path / "changed-spec.md"
    spec.write_text("changed")
    monkeypatch.setattr(module, "compact_cache_binding", lambda *a, **kw: binding)
    monkeypatch.setattr(module, "read_cache", lambda *a: ([], [], {}))
    monkeypatch.setattr(module, "synthetic_binding", lambda *a: {})
    args = SimpleNamespace(output=tmp_path / "out", audit=audit, source_validation=validation,
                           cache="not-read", spec=spec, max_seconds=1200, synthetic=synthetic,
                           shared_challenge=tmp_path / "never-read")
    with pytest.raises(ValueError, match="validated scientific declaration"):
        module.run(args)
