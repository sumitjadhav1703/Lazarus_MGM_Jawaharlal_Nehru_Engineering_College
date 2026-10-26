import gzip
import hashlib
import json

import pytest

from cardiotwin import ecgfounder_study as study


def storage():
    return dict(allocated_bytes=0, logical_bytes=0)


def test_report_exact_canonical_roundtrip_and_stored_cap(tmp_path):
    value = dict(report=[None, True, -0.0, 1.2345678901234567, 1e-200, 1e200, "é"] * 100)
    canonical = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    compressed = gzip.compress(canonical, mtime=0)
    first = study._report_artifact(tmp_path, "summary.json.gz", value, len(compressed), storage)
    assert first["bytes"] == len(compressed) < len(canonical)
    assert first["sha256"] == hashlib.sha256(compressed).hexdigest()
    assert (tmp_path / "summary.json.gz").read_bytes() == compressed
    assert gzip.decompress(compressed) == canonical
    assert json.dumps(study._read_report(tmp_path / "summary.json.gz"), sort_keys=True) == canonical.decode()
    with pytest.raises(RuntimeError, match="byte cap"):
        study._report_artifact(tmp_path, "other.json.gz", value, len(compressed), storage)
    assert not (tmp_path / "other.json.gz").exists()
    assert not list(tmp_path.glob("*.partial"))


def test_report_nan_rejected_without_file(tmp_path):
    with pytest.raises(ValueError):
        study._report_artifact(tmp_path, "summary.json.gz", [float("nan")], 1000, storage)
    assert not list(tmp_path.iterdir())


def test_corrupt_report_rejected(tmp_path):
    path = tmp_path / "review.json.gz"
    path.write_bytes(gzip.compress(b'{"risk":0.25}', mtime=0)[:-4])
    with pytest.raises((EOFError, OSError)):
        study._read_report(path)
