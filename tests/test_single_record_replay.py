import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_single_record_replay.py"
spec = importlib.util.spec_from_file_location("prepare_single_record_replay", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "src"
    for name in ("record_000", "record_001"):
        (root / "records" / name).mkdir(parents=True)
        (root / "records" / name / f"{name}_ecg_00.h5").write_bytes(b"x")
    with (root / "metadata.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["record_id", "patient_id", "patient_age"])
        writer.writerow(["record_000", "patient_000", "60"])
        writer.writerow(["record_001", "patient_001", "70"])
    return root


def test_prepares_one_record_and_refuses_existing_output(source, tmp_path, capsys):
    out = tmp_path / "out"
    assert script.main(["--source", str(source), "--record", "record_001", "--output", str(out)]) == 0
    rows = list(csv.DictReader((out / "metadata.csv").open()))
    assert [r["record_id"] for r in rows] == ["record_001"]
    assert (out / "records" / "record_001").is_symlink()
    assert not (out / "records" / "record_000").exists()
    assert (out / "records" / "record_001" / "record_001_ecg_00.h5").exists()
    assert f"--data-dir {out}" in capsys.readouterr().out
    assert script.main(["--source", str(source), "--record", "record_001", "--output", str(out)]) == 2


def test_refuses_test_group_and_unknown_record(source, tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"group_ids": {"iridia-af:patient_000": "g0"}, "test_groups": ["g0"]}))
    args = ["--source", str(source), "--output"]
    assert script.main([*args, str(tmp_path / "a"), "--record", "record_000", "--split-file", str(split)]) == 2
    assert not (tmp_path / "a").exists()
    assert script.main([*args, str(tmp_path / "b"), "--record", "record_001", "--split-file", str(split)]) == 0
    assert script.main([*args, str(tmp_path / "c"), "--record", "record_999"]) == 2
