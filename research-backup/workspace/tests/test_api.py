from fastapi.testclient import TestClient

from cardiotwin.api import create_app
from cardiotwin.config import load_config
from cardiotwin.data import make_demo
from cardiotwin.service import TwinService


def test_api_replay_unavailable_and_unknown_patient(tmp_path):
    c = load_config()
    make_demo(tmp_path / "data", c)
    s = TwinService(tmp_path / "data", tmp_path / "db.sqlite", c)
    with TestClient(create_app(s, auto_tick=False)) as client:
        assert client.get("/api/health").status_code == 200
        pid = client.get("/api/patients").json()[0]["patient_id"]
        assert client.get("/api/patients/missing/state").status_code == 404
        assert (
            client.post(f"/api/patients/{pid}/replay", json={"action": "step", "steps": 0}).status_code == 422
        )
        response = client.post(f"/api/patients/{pid}/replay", json={"action": "step", "steps": 26})
        assert response.status_code == 200
        state = response.json()
        assert state["timestamp"] == 780
        assert not state["forecast"]["available"]
        assert state["forecast"]["risk"] is None
        assert state["mode"] == "historical_simulation"
        direct = client.get(f"/api/patients/{pid}/forecast").json()
        assert direct["quality"] == state["quality"]
        assert client.get("/api/metrics").json()["status"] == "not_evaluated"
        assert client.post(f"/api/patients/{pid}/replay", json={"action": "reset"}).json()["timestamp"] == 0
