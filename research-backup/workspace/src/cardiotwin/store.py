"""SQLite snapshots, warning history and review events in atomic replay commits."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with self.transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS patients (id TEXT PRIMARY KEY, state TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS warnings (id TEXT PRIMARY KEY, patient TEXT NOT NULL, timestamp REAL NOT NULL, body TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS warnings_patient ON warnings(patient,timestamp);
                CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY, patient TEXT NOT NULL, warning TEXT NOT NULL, body TEXT NOT NULL);
            """)

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def read(self, pid):
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT state FROM patients WHERE id=?", (pid,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, state, warning=None, reset=False):
        with self.transaction() as db:
            self.save_in(db, state, warning, reset)

    def save_in(self, db, state, warning=None, reset=False):
        pid = state["patient_id"]
        if reset:
            db.execute("DELETE FROM warnings WHERE patient=?", (pid,))
            db.execute("DELETE FROM reviews WHERE patient=?", (pid,))
        db.execute("INSERT OR REPLACE INTO patients VALUES (?,?)", (pid, json.dumps(state, allow_nan=False)))
        if warning:
            old = db.execute("SELECT body FROM warnings WHERE id=?", (warning["id"],)).fetchone()
            if old:
                warning = {**json.loads(old[0]), **warning}
            db.execute(
                "INSERT OR REPLACE INTO warnings VALUES (?,?,?,?)",
                (warning["id"], pid, warning["timestamp"], json.dumps(warning, allow_nan=False)),
            )

    def warnings(self, pid, limit=200, offset=0):
        with sqlite3.connect(self.path) as db:
            rows = db.execute(
                "SELECT body FROM warnings WHERE patient=? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (pid, limit, offset),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]
