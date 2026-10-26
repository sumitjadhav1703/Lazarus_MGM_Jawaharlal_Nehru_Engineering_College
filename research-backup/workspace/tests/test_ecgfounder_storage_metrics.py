"""Generated filesystem parity checks for the paired engineering storage scan."""

import importlib.util
import os
from pathlib import Path


def cycle():
    path = Path(__file__).parents[1] / "scripts/kaggle_cycle.py"
    spec = importlib.util.spec_from_file_location("storage_cycle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_metrics_match_existing_semantics_and_walk_once(tmp_path, monkeypatch):
    m = cycle()
    root = tmp_path / "root"
    root.mkdir()
    (root / "data").write_bytes(b"generated")
    os.link(root / "data", root / "hardlink")
    (root / "file-link").symlink_to(root / "data")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "excluded").write_bytes(b"not traversed")
    (root / "directory-link").symlink_to(outside, target_is_directory=True)
    roots = [root, root / "nested", tmp_path / "missing"]
    info = (root / "data").stat()
    expected = {"apparent": 9, "allocated": info.st_blocks * 512}
    assert m.dataset_bytes(roots) == expected["apparent"]
    assert m.dataset_bytes(roots, allocated=True) == expected["allocated"]
    calls = []
    original = Path.rglob

    def tracked(path, pattern):
        calls.append(path)
        return original(path, pattern)

    monkeypatch.setattr(Path, "rglob", tracked)
    assert m.dataset_usage(roots) == expected
    assert len(calls) == 2  # normalized root plus missing root; each traversed only once
    assert m.dataset_usage([]) == {"apparent": 0, "allocated": 0}


def test_paired_metrics_tolerate_unlink_between_type_and_stat(tmp_path, monkeypatch):
    m = cycle()
    gone = tmp_path / "gone"
    gone.write_bytes(b"generated")
    original = Path.is_file

    def remove(path):
        result = original(path)
        if path == gone:
            path.unlink()
        return result

    monkeypatch.setattr(Path, "is_file", remove)
    assert m.dataset_usage([tmp_path]) == {"apparent": 0, "allocated": 0}


def test_engineering_storage_boundaries_preserve_failure_snapshots(tmp_path, monkeypatch):
    import json
    import types

    import pytest

    path = Path(__file__).parents[1] / "scripts/kaggle_ecgfounder_engineering.py"
    spec = importlib.util.spec_from_file_location("engineering_guard", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    project = tmp_path / "project"
    project.mkdir()
    table = project / "table"
    helper = types.SimpleNamespace(write_json=lambda p, v: Path(p).write_text(json.dumps(v)), sanitized=str)
    monkeypatch.setattr(m, "remaining_seconds", lambda **k: 10)
    monkeypatch.setattr(
        m.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("launch reached"))
    )
    for metric in ("apparent", "allocated"):
        for category, cap in [
            ("global", m.PROJECT_BYTES - m.DIAGNOSTIC_BYTES),
            ("diagnostic", m.DIAGNOSTIC_BYTES),
            ("table", m.TABLE_BYTES + m.TABLE_RECEIPT_BYTES + 65536),
        ]:
            for extra in (0, 1):

                def usage(roots):
                    value = 0
                    if roots == ["global"] and category == "global":
                        value = cap + extra
                    elif roots == [project] and category == "diagnostic":
                        value = cap + extra
                    elif roots == [table] and category == "table":
                        value = cap + extra
                    return {name: value if name == metric else 0 for name in ("apparent", "allocated")}

                fake = types.SimpleNamespace(dataset_usage=usage)
                monkeypatch.setattr(
                    m, "load_script", lambda project, name: fake if name == "kaggle_cycle" else helper
                )
                output = project / f"{metric}-{category}-{extra}"
                with pytest.raises(RuntimeError) as caught:
                    m.supervise(
                        ["generated"],
                        project=project,
                        output=output,
                        roots=["global"],
                        session_started=0,
                        engineering_started=None,
                        phase="dense_native",
                        table_root=table,
                    )
                assert ("launch reached" in str(caught.value)) == (extra == 0)
                receipt = json.loads((output / "receipt.json").read_text())
                assert receipt["status"] == "failed"
                assert metric in receipt["storage"]


def test_worker_guard_retains_offending_storage_snapshot():
    import ast
    import time
    import types

    import pytest

    source = (Path(__file__).parents[1] / "scripts/run_ecgfounder_engineering.py").read_text()
    main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    functions = [
        n for n in main.body if isinstance(n, ast.FunctionDef) and n.name in ("global_storage", "guard")
    ]
    receipt = {}
    scope = {
        "Path": Path,
        "time": time,
        "receipt": receipt,
        "cycle": types.SimpleNamespace(
            dataset_usage=lambda roots: {"apparent": 25000000000, "allocated": 25000000001}
        ),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<worker guard>", "exec"), scope)
    with pytest.raises(RuntimeError, match="Global storage"):
        scope["guard"]()
    assert receipt["storage"] == {"allocated_bytes": 25000000001, "logical_bytes": 25000000000}
    assert receipt["storage_scan_calls"] == 1
    assert receipt["storage_scan_seconds"] >= 0
