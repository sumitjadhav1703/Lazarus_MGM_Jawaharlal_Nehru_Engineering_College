import json

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from cardiotwin.api import create_app
from cardiotwin.cli import main
from cardiotwin.config import load_config
from cardiotwin.data import make_demo
from cardiotwin.modeling import Predictor, build_pipeline, matrix
from cardiotwin.service import TwinService
from cardiotwin.train import _save_predictor


def failed_gate_package(tmp_path, config):
    schema = ["ecg_hr", "ecg_rr_mean"]
    rows = [{"ecg_hr": 60, "ecg_rr_mean": 1}, {"ecg_hr": 90, "ecg_rr_mean": 0.66}]
    pipeline = build_pipeline("logistic", {}, 42)
    pipeline.fit(matrix(rows, schema), [0, 1])
    calibrator = LogisticRegression().fit([[-1], [1]], [0, 1])
    manifest = {"version": "gate-failed-test", "deployment_ready": False}
    package = tmp_path / "model"
    _save_predictor(Predictor(pipeline, calibrator, schema, config, manifest), package, config, manifest, {})
    saved = json.loads((package / "manifest.json").read_text())
    saved["deployment_ready"] = False
    (package / "manifest.json").write_text(json.dumps(saved))
    return package


def test_gate_failed_model_needs_research_flag_and_is_labelled(tmp_path):
    config = load_config()
    package = failed_gate_package(tmp_path, config)
    make_demo(tmp_path / "data", config)
    with pytest.raises(ValueError, match="research-only"):
        Predictor.load(package)
    with pytest.raises(ValueError, match="research-only"):
        TwinService(tmp_path / "data", tmp_path / "a.sqlite", config, package)
    with pytest.raises(ValueError, match="requires a model"):
        TwinService(tmp_path / "data", tmp_path / "b.sqlite", config, research_model=True)
    service = TwinService(tmp_path / "data", tmp_path / "c.sqlite", config, package, research_model=True)
    with TestClient(create_app(service, auto_tick=False)) as client:
        health = client.get("/api/health").json()
    assert health["research_model"] is True and health["deployment_ready"] is False


def test_default_health_is_not_research_model(tmp_path):
    config = load_config()
    make_demo(tmp_path / "data", config)
    service = TwinService(tmp_path / "data", tmp_path / "d.sqlite", config)
    with TestClient(create_app(service, auto_tick=False)) as client:
        health = client.get("/api/health").json()
    assert health["research_model"] is False and health["deployment_ready"] is None


def test_cli_research_flag_requires_model():
    assert main(["serve", "--research-model"]) == 2
