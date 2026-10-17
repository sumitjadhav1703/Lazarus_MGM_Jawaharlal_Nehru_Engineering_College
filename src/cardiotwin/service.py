"""Persistent annotation-free historical replay. Processing occurs on explicit ticks."""

import hashlib
import json
import threading
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from .alerts import advance_alert
from .config import validate_config
from .data import file_hash, list_records, read_signal, signal_available, signal_paths
from .features import feature_row, update_baseline
from .store import Store


class TwinService:
    def __init__(self, data_dir, db_path, config, model_dir=None, metrics_path=None, research_model=False):
        if research_model and not model_dir:
            raise ValueError("research_model requires a model directory")
        validate_config(config)
        self.config = config
        self.store = Store(db_path)
        self.lock = threading.RLock()
        self.records = {}
        for record in list_records(Path(data_dir)):
            if signal_available(record):
                self.records.setdefault(record.patient_id, record)
        if not self.records:
            raise ValueError("No downloaded ECG signals available for replay")
        self.research_model = research_model
        self.predictor = None
        if model_dir:
            from .modeling import Predictor

            self.predictor = Predictor.load(Path(model_dir), research=research_model)
            if self.predictor.config != config:
                raise ValueError("Runtime config must equal frozen model config")
        self.metrics_path = Path(metrics_path) if metrics_path else None
        version = self.predictor.version if self.predictor else "unavailable"
        for pid, record in self.records.items():
            identity = {
                "record": record.record_id,
                "signal_hash": record.signal_hash,
                "path": str(record.path),
                "duration": record.duration_seconds,
                "fs": record.fs,
                "clinical": record.clinical,
                "config": config,
                "model_version": version,
                "model_hash": file_hash(Path(model_dir) / "model.joblib") if model_dir else None,
                "schema_hash": file_hash(Path(model_dir) / "feature_schema.json") if model_dir else None,
                "manifest_hash": file_hash(Path(model_dir) / "manifest.json") if model_dir else None,
            }
            path = Path(record.path)
            if record.signal_parts:
                identity.update(source=record.source, signal_parts=record.signal_parts)
                files = signal_paths(record)
            else:
                files = (path, path.with_suffix(".dat"), path.with_suffix(".npz"))
            for file in files:
                if file.is_file():
                    key = "file_" + (file.name if record.signal_parts else file.suffix)
                    identity[key] = {
                        "size": file.stat().st_size,
                        "mtime": file.stat().st_mtime_ns,
                    }
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            saved = self.store.read(pid)
            if saved and saved.get("_fingerprint") != fingerprint:
                raise ValueError(f"Inputs/config/model changed for {pid}; use a fresh state database")
            if not saved:
                self.store.save(self._initial(record, fingerprint))

    def _initial(self, record, fingerprint):
        synthetic = (
            record.patient_id.startswith("demo")
            or record.record_id.startswith("demo")
            or str(record.path).endswith(".npz")
        )
        return {
            "patient_id": record.patient_id,
            "record_id": record.record_id,
            "clinical": record.clinical,
            "timestamp": 0.0,
            "duration_seconds": record.duration_seconds,
            "replay_status": "paused",
            "baseline": {},
            "quality": {"accepted": False, "reason": "insufficient_history"},
            "features": {},
            "feature_history": [],
            "timeline": [],
            "_alert": {},
            "_fingerprint": fingerprint,
            "mode": "historical_simulation",
            "provenance": "synthetic" if synthetic else record.source,
            "forecast": self._forecast(0.0, "insufficient_history"),
        }

    def _forecast(self, t, reason=None):
        target = self.config["target"]
        return {
            "available": reason is None,
            "risk": None,
            "timestamp": t,
            "target_start": t + target["gap_seconds"],
            "target_end": t + target["horizon_seconds"],
            "model_version": self.predictor.version if self.predictor else "unavailable",
            "reason": reason,
            "quality": {"accepted": False, "reason": reason},
            "evidence": [],
            "uncertainty": "Research score; clinical usefulness and Indian generalization are unproven.",
        }

    def _get(self, pid):
        if pid not in self.records:
            raise KeyError(pid)
        return self.store.read(pid)

    def patients(self):
        return [
            {
                k: s.get(k)
                for k in (
                    "patient_id",
                    "record_id",
                    "clinical",
                    "timestamp",
                    "duration_seconds",
                    "replay_status",
                    "quality",
                    "provenance",
                )
            }
            for s in (self._get(pid) for pid in self.records)
        ]

    def state(self, pid):
        with self.lock:
            state = self._get(pid)
            state = {k: v for k, v in state.items() if not k.startswith("_")}
            record = self.records[pid]
            start = max(0.0, state["timestamp"] - 10.0)
            signal = (
                read_signal(record, start, state["timestamp"])
                if state["timestamp"] > start
                else np.empty((0, 2))
            )
            if signal.ndim == 1:
                signal = signal[:, None]
            stride = max(1, int(np.ceil(len(signal) / 1500)))
            signal = signal[::stride]
            state["ecg"] = {
                "times": (start + np.arange(len(signal)) * stride / record.fs).tolist(),
                "leads": [[float(v) if np.isfinite(v) else None for v in lead] for lead in signal.T],
                "fs": record.fs,
            }
            state["warnings"] = self.store.warnings(pid, 50)
            return state

    def step(self, pid, steps=1):
        if not isinstance(steps, int) or not 1 <= steps <= 120:
            raise ValueError("steps must be an integer between 1 and 120")
        with self.lock:
            for _ in range(steps):
                # Cursor read and write share one database transaction, even across processes.
                with self.store.transaction() as db:
                    row = db.execute("SELECT state FROM patients WHERE id=?", (pid,)).fetchone()
                    if pid not in self.records or not row:
                        raise KeyError(pid)
                    s = json.loads(row[0])
                    record = self.records[pid]
                    if s["timestamp"] >= record.duration_seconds:
                        break
                    target = self.config["target"]
                    t = min(s["timestamp"] + target["cadence_seconds"], record.duration_seconds)
                    signal = read_signal(record, max(0, t - target["history_seconds"]), t)
                    features, quality = feature_row(
                        signal, record.fs, record.clinical, s["baseline"], self.config
                    )
                    reason = "insufficient_history" if t < target["history_seconds"] else quality["reason"]
                    if (
                        reason is None
                        and s["baseline"].get("count", 0) < self.config["baseline"]["minimum_updates"]
                    ):
                        reason = "baseline_warmup"
                    if reason is None and self.predictor is None:
                        reason = "model_not_loaded"
                    forecast = self._forecast(t, reason)
                    forecast["quality"] = quality
                    forecast["baseline"] = {"means": dict(s["baseline"].get("means", {}))}
                    if reason is None:
                        forecast.update(self.predictor.predict(features))
                    alert, warning = advance_alert(
                        s["_alert"], t, forecast["risk"], forecast["available"], self.config
                    )
                    if warning:
                        warning["id"] = str(
                            uuid5(
                                NAMESPACE_URL,
                                f"{pid}/{record.record_id}/{warning['timestamp']}/{forecast['model_version']}",
                            )
                        )
                        if warning["end_time"] is None:
                            warning.update(
                                status="pending",
                                evidence=forecast["evidence"],
                                model_version=forecast["model_version"],
                                target_start=forecast["target_start"],
                                target_end=forecast["target_end"],
                                baseline=s["baseline"].get("means", {}),
                                features=features,
                            )
                        s["timeline"].append(
                            {
                                "timestamp": t,
                                "type": "warning_closed" if warning["end_time"] else "warning",
                                "message": "Warning ended"
                                if warning["end_time"]
                                else "Clinician review requested",
                                "warning_id": warning["id"],
                            }
                        )
                    previous_reason = s["forecast"].get("reason")
                    if reason != previous_reason:
                        s["timeline"].append(
                            {
                                "timestamp": t,
                                "type": "availability",
                                "message": reason or "Forecast available",
                            }
                        )
                    if quality["accepted"] and t >= target["history_seconds"]:
                        s["baseline"] = update_baseline(s["baseline"], features, self.config)
                        if s["baseline"]["count"] == 1 or s["baseline"]["count"] % 20 == 0:
                            s["timeline"].append(
                                {
                                    "timestamp": t,
                                    "type": "baseline",
                                    "message": f"Baseline updated ({s['baseline']['count']} reliable samples)",
                                }
                            )
                    s.update(timestamp=t, quality=quality, features=features, forecast=forecast, _alert=alert)
                    s["feature_history"] = (s["feature_history"] + [{"timestamp": t, **features}])[-60:]
                    s["timeline"] = s["timeline"][-200:]
                    if t >= record.duration_seconds:
                        s["replay_status"] = "complete"
                    self.store.save_in(db, s, warning)
            return self.state(pid)

    def control(self, pid, action):
        if action not in ("start", "pause", "reset"):
            raise ValueError("Unknown replay action")
        with self.lock:
            with self.store.transaction() as db:
                row = db.execute("SELECT state FROM patients WHERE id=?", (pid,)).fetchone()
                if pid not in self.records or not row:
                    raise KeyError(pid)
                s = json.loads(row[0])
                if action == "reset":
                    s = self._initial(self.records[pid], s["_fingerprint"])
                else:
                    s["replay_status"] = (
                        "running"
                        if action == "start" and s["timestamp"] < s["duration_seconds"]
                        else "paused"
                    )
                self.store.save_in(db, s, reset=action == "reset")
            return self.state(pid)

    def warnings(self, pid, limit=200, offset=0):
        self._get(pid)
        return self.store.warnings(pid, limit, offset)

    def review(self, pid, warning_id, status, note=""):
        if status not in ("pending", "reviewed", "dismissed") or len(note) > 2000:
            raise ValueError("Invalid review status or note")
        with self.lock, self.store.transaction() as db:
            row = db.execute(
                "SELECT body FROM warnings WHERE id=? AND patient=?", (warning_id, pid)
            ).fetchone()
            if not row:
                raise KeyError(warning_id)
            warning = json.loads(row[0])
            warning.update(status=status, note=note)
            db.execute("UPDATE warnings SET body=? WHERE id=?", (json.dumps(warning), warning_id))
            s = json.loads(db.execute("SELECT state FROM patients WHERE id=?", (pid,)).fetchone()[0])
            event = {
                "timestamp": s["timestamp"],
                "type": "review",
                "message": status,
                "warning_id": warning_id,
                "note": note,
            }
            s["timeline"] = (s["timeline"] + [event])[-200:]
            db.execute(
                "INSERT INTO reviews(patient,warning,body) VALUES (?,?,?)",
                (pid, warning_id, json.dumps(event)),
            )
            self.store.save_in(db, s)
        return warning

    def tick(self):
        for patient in self.patients():
            if patient["replay_status"] == "running":
                self.step(patient["patient_id"])

    def metrics(self):
        if self.metrics_path and self.metrics_path.exists():
            return json.loads(self.metrics_path.read_text())
        return {
            "status": "not_evaluated",
            "limitations": ["No held-out evaluation loaded. Synthetic replay is software evidence only."],
        }
