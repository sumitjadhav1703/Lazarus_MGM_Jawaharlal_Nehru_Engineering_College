"""Independent known-anchor numerical reference and durable validation receipts."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def validator():
    path = Path("scripts/run_anchor_noise_validation.py")
    assert path.is_file(), "fixed numerical validator missing"
    spec = importlib.util.spec_from_file_location("anchor_noise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_known_reference_is_independent_and_constant_shape_has_zero_changes(monkeypatch):
    module = validator()

    def forbidden(*args, **kwargs):
        raise AssertionError("Oracle must not call production extractor")

    monkeypatch.setattr(module, "extract_morphology", forbidden)
    report = module.known_reference(module.synthetic_signal())
    assert len(report["clean_scales"]) == 2
    for key, value in report["features"].items():
        assert abs(value) < 1e-9, key


def test_anchor_matching_is_chronological_one_to_one():
    module = validator()
    duplicated = [200, 201, *range(400, 120000, 200)]
    report = module.anchor_accuracy(duplicated)
    assert report["recall"] == 1 and report["precision"] == pytest.approx(599/600)


def test_validator_freezes_source_before_cases_and_preserves_original_error(tmp_path, monkeypatch):
    module = validator()
    spec = tmp_path / "spec.md"
    spec.write_text("fixed test declaration")
    output = tmp_path / "out"

    def stop(handle):
        declaration = json.loads((output / "predeclaration.json").read_text())
        assert "scripts/run_anchor_noise_validation.py" in declaration["source"]["files"]
        assert declaration["spec_sha256"]
        raise RuntimeError("preserved sentinel")

    monkeypatch.setattr(module, "run_cases", stop)
    with pytest.raises(RuntimeError, match="preserved sentinel"):
        module.run(SimpleNamespace(output=output, spec=spec))
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "error" and receipt["error"] == "preserved sentinel"
    with pytest.raises(ValueError, match="output already exists"):
        module.run(SimpleNamespace(output=output, spec=spec))
