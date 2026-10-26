import runpy
from pathlib import Path


def test_sequence_cache_capacity_counts_roots_once_and_reserves_heldout_space(tmp_path):
    capacity = runpy.run_path(str(Path(__file__).parents[1] / "scripts/run_investigation.py"))[
        "sequence_cache_capacity"
    ]
    data, reports = tmp_path / "data", tmp_path / "reports"
    data.mkdir()
    reports.mkdir()
    (data / "raw").write_bytes(b"x" * 40)
    (reports / "result").write_bytes(b"x" * 10)
    assert capacity(data / "cache", reports / "neural", limit=100, reserve=20) == 30
    assert capacity(data / "cache", data / "neural", limit=100, reserve=20) == 40
    assert capacity(data / "cache", reports / "neural", limit=50, reserve=20) == 0
    assert capacity(data / "cache", reports / "neural") == 4_000_000_000
