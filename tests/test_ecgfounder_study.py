import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cardiotwin import ecgfounder_compare as compare
from cardiotwin import ecgfounder_runner as runner
from cardiotwin import ecgfounder_study as study
from cardiotwin.config import load_config
from cardiotwin.ecgfounder_probe import OVERHEAD_COMPONENTS
from cardiotwin.ecgfounder_tables import identity_digest, write_tables
from cardiotwin.features import FEATURE_VERSION
from cardiotwin.investigation import CACHE_VERSION
from cardiotwin.precursor import LINEAR_SCHEMA, compact_cache_binding


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def fixture(tmp_path, monkeypatch, unavailable=()):
    root = tmp_path
    cycle = root / study.CYCLE
    cache = root / "data/cache"
    audit = root / "reports/original-audit"
    out = root / "reports/extraction"
    bridge = root / "reports/bridge"
    for path in (cycle, cache, audit, out, bridge):
        path.mkdir(parents=True)
    source = {}
    for name in study.REQUIRED_SOURCE:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic " + name)
        source[name] = sha(path)
    monkeypatch.setattr(study, "source_manifest", lambda: dict(files=source))
    config = load_config()
    config_path = root / "configs/study.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config))
    records = [dict(record_id=f"r{i}", patient_id=f"p{i}", group_id=f"g{i}") for i in range(8)]
    roles = []
    for f, outer in enumerate(([0, 1, 2], [3, 4, 5], [6, 7])):
        remaining = [i for i in range(8) if i not in outer]
        roles.append(
            dict(
                fold=f,
                fit=[f"g{i}" for i in remaining[:-4]],
                stop=[f"g{i}" for i in remaining[-4:-2]],
                calibration=[f"g{remaining[-2]}"],
                policy=[f"g{remaining[-1]}"],
                outer=[f"g{i}" for i in outer],
            )
        )
    values = []
    rows = []
    events = []
    pairs = []
    for i, record in enumerate(records):
        times = sorted(
            set(
                [600.0 + 30 * j for j in range(146)]
                + [900.0 + 5400 * j for j in range(5)]
                + [3240.0 + 5400 * j for j in range(5)]
            )
        )
        positions = {t: len(rows) + j for j, t in enumerate(times)}
        for t in times:
            label = int(t in {3240.0 + 5400 * j for j in range(5)})
            values.append((i, t, 27000.0, 1, label, 0, 0))
            rows.append(
                dict(
                    row_index=len(rows),
                    **record,
                    timestamp=t,
                    duration_seconds=27000.0,
                    accepted=True,
                    label=label,
                    features={k: 0.0 for k in LINEAR_SCHEMA},
                )
            )
        for j in range(5):
            onset = 3600.0 + 5400 * j
            events.append(dict(**record, onset=onset))
            for role in roles:
                name = next(k for k in study.ROLE_NAMES if record["group_id"] in role[k])
                pairs.append(
                    dict(
                        fold=role["fold"],
                        role=name,
                        **record,
                        onset=onset,
                        positive_index=positions[onset - 360.0],
                        negative_index=positions[onset - 2700.0],
                    )
                )
    metadata = np.array(
        values,
        dtype=[
            ("record", "i4"),
            ("timestamp", "f8"),
            ("duration_seconds", "f8"),
            ("accepted", "i1"),
            ("label", "i1"),
            ("reason", "i2"),
            ("label_reason", "i2"),
        ],
    )
    np.save(cache / "metadata.npy", metadata)
    np.save(cache / "features.npy", np.zeros((len(rows), len(LINEAR_SCHEMA)), np.float64))
    split = dict(
        seed=42,
        development_groups=[r["group_id"] for r in records],
        test_groups=["closed"],
        group_ids={r["patient_id"]: r["group_id"] for r in records},
    )
    split_path = root / "reports/split.json"
    split_path.write_text(json.dumps(split))
    index = dict(
        cache_version=CACHE_VERSION,
        feature_version=FEATURE_VERSION,
        config=config,
        config_sha256=canonical(config),
        row_count=len(rows),
        rows_sha256="a" * 64,
        feature_fields=LINEAR_SCHEMA,
        events=events,
        records=[
            dict(
                record=dict(
                    **r,
                    duration_seconds=27000.0,
                    fs=200.0,
                    signal_parts=[dict(samples=5400000, path=f"{r['record_id']}.h5")],
                )
            )
            for r in records
        ],
        allowed_groups=split["development_groups"],
        group_ids=split["group_ids"],
    )
    index["sha256"] = canonical(index)
    (cache / "index.json").write_text(json.dumps(index))
    compact = dict(
        rows_sha256=index["rows_sha256"],
        features=LINEAR_SCHEMA,
        records=records,
        categories=dict(reason=[None], label_reason=[None]),
        checksums={n: sha(cache / n) for n in ("features.npy", "metadata.npy")},
    )
    (cache / "compact.json").write_text(json.dumps(compact))
    binding = compact_cache_binding(cache)
    (audit / "roles.json").write_text(
        json.dumps(dict(cache_index_sha256=binding["index_sha256"], roles=roles))
    )
    (audit / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    support = dict(status="passed", failed_criteria=[], folds=[])
    for role in roles:
        cells = {}
        for name in study.ROLE_NAMES:
            chosen = [p for p in pairs if p["fold"] == role["fold"] and p["role"] == name]
            groups = set(p["group_id"] for p in chosen)
            cells[name] = dict(
                eligible_events=len(chosen),
                observable_events=len(chosen),
                positive_people=len(groups),
                paired_events=len(chosen),
                paired_people=len(groups),
                pair_counts_by_person={g: sum(p["group_id"] == g for p in chosen) for g in groups},
                observable_event_fraction=1.0,
                positive_person_fraction=1.0,
            )
        support["folds"].append(dict(fold=role["fold"], roles=cells))
    (audit / "pair-support.json").write_text(json.dumps(support))
    meta = root / "data/metadata.csv"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text("opaque identity only")
    (audit / "metadata-audit.json").write_text(
        json.dumps(dict(status="passed", metadata_sha256={"metadata.csv": sha(meta)}))
    )
    audit_receipt = dict(
        status="passed",
        metadata_status="passed",
        pair_support_status="passed",
        failed_criteria=[],
        cache_index_sha256=binding["index_sha256"],
        cache_identity=index["sha256"],
        config_sha256=canonical(config),
        split_sha256=sha(split_path),
        roles_sha256=canonical(roles),
        pairs_sha256=sha(audit / "pairs.jsonl"),
        rows=len(rows),
        events=len(events),
        people=len(records),
        source=dict(files=source),
    )
    (audit / "receipt.json").write_text(json.dumps(audit_receipt))
    spec = cycle / "spec.md"
    spec.write_text("synthetic frozen spec")
    original = dict(
        files_sha256={
            str(p.relative_to(root)): sha(p)
            for p in (
                spec,
                config_path,
                split_path,
                audit / "roles.json",
                audit / "pairs.jsonl",
                cache / "index.json",
            )
        },
        compact_binding=binding,
    )
    original_path = cycle / "declaration-binding.json"
    original_path.write_text(json.dumps(original))
    monkeypatch.setattr(study, "ORIGINAL_DECLARATION_SHA256", sha(original_path))
    monkeypatch.setattr(study, "SPEC_SHA256", sha(spec))
    monkeypatch.setattr(runner, "EXPECTED_ROWS", len(rows))
    for name, value in [
        ("EXPECTED_ROWS", len(rows)),
        ("EXPECTED_PEOPLE", 8),
        ("EXPECTED_RECORDS", 8),
        ("EXPECTED_EVENTS", 40),
    ]:
        monkeypatch.setattr(study, name, value)
    checkpoint = cycle / "weights/1_lead_ECGFounder.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"synthetic weights")
    for module in (study, compare):
        monkeypatch.setattr(module, "CHECKPOINT_SHA256", sha(checkpoint))
        monkeypatch.setattr(module, "CHECKPOINT_BYTES", checkpoint.stat().st_size)
    validation = bridge / "validation.json"
    validation.write_text(
        json.dumps(
            dict(
                status="passed",
                binding=binding,
                rows_checked=len(rows),
                feature_values_checked=len(rows) * len(LINEAR_SCHEMA),
                historical_fit_time_identity_verified=False,
            )
        )
    )
    prior = bridge / "receipt.json"
    prior.write_text(
        json.dumps(
            dict(
                status="complete",
                historical_fit_time_identity_verified=False,
                artifacts={
                    "validation.json": sha(validation),
                    **{"source-package/" + n: v for n, v in source.items()},
                },
                identity=dict(
                    diagnostic_sha256=sha(audit / "receipt.json"),
                    cache_index_sha256=binding["index_sha256"],
                    config_sha256=canonical(config),
                    split_sha256=sha(split_path),
                    roles_sha256=canonical(roles),
                    pairs_sha256=sha(audit / "pairs.jsonl"),
                ),
            )
        )
    )
    ruling = bridge / "ruling.md"
    ruling.write_text("explicit parent current-source bridge only")
    evidence = bridge / "evidence.json"
    evidence.write_text(
        json.dumps(
            dict(
                status="verified_current_source_identity_bridge_only",
                source_receipt_path=str(prior.relative_to(root)),
                source_receipt_sha256=sha(prior),
                validation_path=str(validation.relative_to(root)),
                validation_sha256=sha(validation),
                original_cycle_declaration_sha256=sha(original_path),
                compact_binding_exact_match=True,
                rows_checked_in_prior_validation=len(rows),
                feature_values_checked_in_prior_validation=len(rows) * len(LINEAR_SCHEMA),
                current_helper_hashes_match_prior_validation={n: source[n] for n in study.BRIDGE_HELPERS},
                historical_fit_time_identity_verified=False,
            )
        )
    )
    raw = cycle / "raw-manifest.json"
    raw.write_text(json.dumps(dict(files_sha256={"opaque.h5": "b" * 64})))
    extraction_files = {
        str(p.relative_to(root)): sha(p)
        for p in (
            original_path,
            spec,
            config_path,
            split_path,
            audit / "roles.json",
            audit / "pairs.jsonl",
            cache / "index.json",
            cache / "compact.json",
            cache / "features.npy",
            cache / "metadata.npy",
            checkpoint,
            raw,
        )
    }
    extraction = cycle / "extraction-declaration.json"
    extraction.write_text(
        json.dumps(
            dict(
                version="ecgfounder-full-extraction-v1",
                files_sha256=extraction_files,
                source_sha256=source,
                compact_binding=binding,
                cache=str(cache.relative_to(root)),
                roles=str((audit / "roles.json").relative_to(root)),
                raw_manifest=str(raw.relative_to(root)),
                metadata_identity_sha256=identity_digest(rows),
            )
        )
    )
    contract = dict(
        row_count=len(rows),
        identity_sha256=identity_digest(rows),
        records=records,
        files_sha256={**extraction_files, str(extraction.relative_to(root)): sha(extraction)},
        source_sha256=source,
    )
    available = np.ones(len(rows), bool)
    available[list(unavailable)] = False

    def results():
        for r in rows:
            ok = bool(available[r["row_index"]])
            yield dict(
                **r,
                available=ok,
                decision_sample=round(r["timestamp"] * 200),
                valid_chunks=60 if ok else 59,
                rejections={} if ok else {"calibration_or_missing": 1},
                delta=np.zeros(1024, np.float32) if ok else None,
                nuisance=np.zeros(6, np.float32) if ok else None,
                original_rejection_reference=None,
            )

    write_tables(
        out / "tables",
        results(),
        iter(rows),
        contract,
        root=root,
        byte_cap=10_000_000,
        storage_check=lambda: dict(allocated_bytes=0, logical_bytes=0),
    )
    coverage = dict(
        rows=len(rows),
        identity_sha256=identity_digest(rows),
        table_receipt_sha256=sha(out / "tables/receipt.json"),
        cells=[],
        passed=True,
    )
    for role in roles:
        for name in study.ROLE_NAMES:
            selected = [r for r in rows if r["group_id"] in role[name]]
            n = int(sum(available[r["row_index"]] for r in selected))
            a = len(selected)
            coverage["cells"].append(
                dict(
                    fold=role["fold"],
                    role=name,
                    total_rows=a,
                    original_accepted=a,
                    common_available=int(n),
                    rejected=0,
                    unavailable_accepted=a - int(n),
                    fraction=n / a,
                    passed=n / a >= 0.95,
                )
            )
    coverage["passed"] = all(c["passed"] for c in coverage["cells"])
    (out / "coverage.json").write_text(json.dumps(coverage))
    checkpoint_info = dict(
        path=study.CHECKPOINT_PATH, sha256=study.CHECKPOINT_SHA256, bytes=study.CHECKPOINT_BYTES
    )
    preflight = dict(
        status="preflight_passed_not_executed",
        declaration_sha256=sha(extraction),
        row_count=len(rows),
        records=8,
        people=8,
        identity_sha256=identity_digest(rows),
        checkpoint=checkpoint_info,
    )
    (out / "preflight.json").write_text(json.dumps(preflight))
    runner_receipt = dict(
        status="complete",
        coverage_passed=True,
        tables_complete=True,
        rows_completed=len(rows),
        table_receipt_sha256=sha(out / "tables/receipt.json"),
        coverage_sha256=sha(out / "coverage.json"),
        checkpoint=checkpoint_info,
        core=dict(status="complete", record_complete=True, rows_completed=len(rows)),
        preflight_sha256=sha(out / "preflight.json"),
        hashes_after=dict(files_sha256={**contract["files_sha256"], **source}, hash_errors={}),
        raw_hashes_after=dict(files_sha256={"opaque.h5": "b" * 64}, hash_errors={}),
    )
    (out / "receipt.json").write_text(json.dumps(runner_receipt))
    cost = cycle / "cost.json"
    cost.write_text(
        json.dumps(
            dict(
                version="ecgfounder-complete-cost-v1",
                cohort_rows=len(rows),
                encoding_upper_bound=1895675,
                device_name="Tesla T4",
                cold_aggregate_chunks_per_second=100.0,
                cold_tenth_chunks_per_second=100.0,
                dispatch_aggregate_chunks_per_second=100.0,
                dispatch_tenth_chunks_per_second=100.0,
                rate_scope="dense_causal_full_timeline",
                rate_evidence_files=[str(evidence.relative_to(root))],
                components={
                    n: dict(upper_seconds=10.0, evidence_files=[str(evidence.relative_to(root))], scope=n)
                    for n in OVERHEAD_COMPONENTS
                },
                work_accounting={
                    n: dict(
                        charged_to="chunk_rate",
                        evidence_files=[str(evidence.relative_to(root))],
                        reason="synthetic joint evidence",
                    )
                    for n in runner.WORK
                },
                calendar_contingency_seconds=60.0,
                extraction_stage_seconds=30000.0,
            )
        )
    )
    cost_review = cycle / "cost-review.json"
    cost_review.write_text(
        json.dumps(
            dict(
                status="passed",
                cost_sha256=sha(cost),
                source_sha256=source,
                compact_binding_sha256=study._canonical_sha(binding),
            )
        )
    )
    monkeypatch.setattr(
        study,
        "BRIDGE_PINS",
        {str(p.relative_to(root)): sha(p) for p in (prior, validation, ruling, evidence)},
        raising=False,
    )
    all_paths = [
        spec,
        original_path,
        config_path,
        split_path,
        audit / "receipt.json",
        audit / "metadata-audit.json",
        audit / "roles.json",
        audit / "pairs.jsonl",
        audit / "pair-support.json",
        cache / "index.json",
        cache / "compact.json",
        cache / "features.npy",
        cache / "metadata.npy",
        checkpoint,
        validation,
        prior,
        ruling,
        evidence,
        raw,
        extraction,
        out / "preflight.json",
        out / "receipt.json",
        out / "coverage.json",
        out / "tables/receipt.json",
        cost,
        cost_review,
        meta,
    ]
    payload = dict(
        version="ecgfounder-study-v1",
        files_sha256={str(p.relative_to(root)): sha(p) for p in all_paths},
        source_sha256=source,
        allowed_stages=["preflight", "support"],
        cache=str(cache.relative_to(root)),
        config=str(config_path.relative_to(root)),
        split=str(split_path.relative_to(root)),
        audit_receipt=str((audit / "receipt.json").relative_to(root)),
        metadata_audit=str((audit / "metadata-audit.json").relative_to(root)),
        metadata_root="data",
        roles=str((audit / "roles.json").relative_to(root)),
        pairs=str((audit / "pairs.jsonl").relative_to(root)),
        pair_support=str((audit / "pair-support.json").relative_to(root)),
        source_validation_receipt=str(prior.relative_to(root)),
        source_validation=str(validation.relative_to(root)),
        bridge_ruling=str(ruling.relative_to(root)),
        bridge_evidence=str(evidence.relative_to(root)),
        runner_receipt=str((out / "receipt.json").relative_to(root)),
        runner_preflight=str((out / "preflight.json").relative_to(root)),
        extraction_declaration=str(extraction.relative_to(root)),
        tables=str((out / "tables").relative_to(root)),
        coverage_receipt=str((out / "coverage.json").relative_to(root)),
        cost_receipt=str(cost.relative_to(root)),
        cost_review=str(cost_review.relative_to(root)),
        raw_manifest=str(raw.relative_to(root)),
    )
    monkeypatch.setattr(
        study,
        "ORIGINAL_AUDIT_PINS",
        {
            key: (payload[key], payload["files_sha256"][payload[key]])
            for key in ("metadata_audit", "pair_support")
        },
    )
    declaration = cycle / "study-declaration.json"
    declaration.write_text(json.dumps(payload))
    return SimpleNamespace(
        root=root,
        cycle=cycle,
        cache=cache,
        audit=audit,
        out=out,
        bridge=bridge,
        declaration=declaration,
        payload=payload,
        source=source,
        rows=rows,
        pairs=pairs,
        storage=lambda: dict(allocated_bytes=0, logical_bytes=0),
    )


def prepare(f):
    return study.prepare_study(
        f.declaration, sha(f.declaration), root=f.root, storage_check=f.storage, guard=lambda: None
    )


def rebind(f, path):
    f.payload["files_sha256"][str(path.relative_to(f.root))] = sha(path)
    f.declaration.write_text(json.dumps(f.payload))


def test_complete_preparation_uses_bridge_and_global_indices_without_source_json(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    with prepare(f) as p:
        assert p.rows[0]["row_index"] == 0 and p.rows[-1]["row_index"] == len(f.rows) - 1
        assert p.rows[0]["features"]["ecg_hr"] == 0.0
        assert p.receipt["historical_fit_time_identity_verified"] is False
        assert not (f.cache / "rows.jsonl").exists()
        support = study.audit_common_support(p, guard=lambda: None)
        assert support["status"] == "passed" and len(support["cells"]) == 15
        assert not support["fits_started"] and not support["full_run_eligible"]


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "source",
        "bridge",
        "validation",
        "checkpoint",
        "runner",
        "coverage",
        "cost",
        "audit",
        "roles",
        "pairs",
        "index",
        "missing_array",
        "storage",
    ],
)
def test_authority_or_identity_mismatch_stops_preparation(tmp_path, monkeypatch, fault):
    f = fixture(tmp_path, monkeypatch)
    expected = sha(f.declaration)
    if fault == "hash":
        f.declaration.write_text(f.declaration.read_text() + " ")
    elif fault == "source":
        (f.root / next(iter(f.source))).write_text("changed")
    elif fault == "missing_array":
        (f.cache / "features.npy").unlink()
    elif fault == "storage":
        f.storage = lambda: dict(allocated_bytes=24_900_000_000, logical_bytes=0)
    else:
        key = {
            "bridge": "bridge_evidence",
            "validation": "source_validation",
            "checkpoint": None,
            "runner": "runner_receipt",
            "coverage": "coverage_receipt",
            "cost": "cost_review",
            "audit": "audit_receipt",
            "roles": "roles",
            "pairs": "pairs",
            "index": None,
        }[fault]
        path = (
            (f.cycle / "weights/1_lead_ECGFounder.pth")
            if fault == "checkpoint"
            else (f.cache / "index.json")
            if fault == "index"
            else f.root / f.payload[key]
        )
        if fault == "pairs":
            path.write_text("{}\n")
        else:
            p = json.loads(path.read_text()) if fault != "checkpoint" else {}
            if fault in ("runner", "coverage", "validation", "audit"):
                p["status"] = "failed"
                p["passed"] = False
            elif fault == "bridge":
                p["compact_binding_exact_match"] = False
            elif fault == "cost":
                p["source_sha256"] = {}
            elif fault == "roles":
                p["roles"][0]["stop"] += p["roles"][0]["fit"]
            elif fault == "index":
                p["cache_version"] = "bad"
            path.write_text(json.dumps(p))
        rebind(f, path)
    with pytest.raises((ValueError, RuntimeError, OSError, KeyError)):
        if fault == "hash":
            study.prepare_study(
                f.declaration, expected, root=f.root, storage_check=f.storage, guard=lambda: None
            )
        else:
            prepare(f)


def test_support_filters_original_pairs_preserves_denominators_and_exact_eighty_percent(
    tmp_path, monkeypatch
):
    initial = fixture(tmp_path / "initial", monkeypatch)
    p = next(p for p in initial.pairs if p["fold"] == 0 and p["role"] == "fit")
    f = fixture(tmp_path / "masked", monkeypatch, unavailable=[p["positive_index"]])
    with prepare(f) as prepared:
        result = study.audit_common_support(prepared, guard=lambda: None)
    cell = next(c for c in result["cells"] if c["fold"] == 0 and c["role"] == "fit")
    assert cell["observable_events"] == 5 and cell["surviving_pairs"] == 4 and cell["event_fraction"] == 0.8
    assert result["status"] == "passed" and len(cell["excluded_pairs"]) == 1
    assert cell["excluded_pairs"][0]["positive_index"] == p["positive_index"]
    assert len(result["surviving_pairs"][0]["fit"]) == 4


@pytest.mark.parametrize(
    "name", ["source_validation_receipt", "source_validation", "bridge_ruling", "bridge_evidence"]
)
def test_bridge_inputs_cannot_be_replaced_by_new_self_asserted_pass_receipts(tmp_path, monkeypatch, name):
    f = fixture(tmp_path, monkeypatch)
    path = f.root / f.payload[name]
    path.write_text(path.read_text() + " ")
    rebind(f, path)
    with pytest.raises(ValueError, match="bridge|validation"):
        prepare(f)


def test_table_mutation_after_preparation_stops_support(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    with prepare(f) as p:
        table = f.out / "tables/rows.npy"
        with table.open("r+b") as handle:
            handle.seek(-1, 2)
            handle.write(b"\xff")
        with pytest.raises(ValueError, match="binding|table"):
            study.audit_common_support(p, guard=lambda: None)


def test_coverage_failure_prevents_label_feature_view_construction(tmp_path, monkeypatch):
    original = fixture(tmp_path / "source", monkeypatch)
    role = next(p for p in original.pairs if p["fold"] == 0 and p["role"] == "fit")["group_id"]
    unavailable = [r["row_index"] for r in original.rows if r["group_id"] == role][:10]
    f = fixture(tmp_path / "failed", monkeypatch, unavailable=unavailable)
    monkeypatch.setattr(study, "CompactRows", lambda *a: pytest.fail("labels exposed before coverage gate"))
    with pytest.raises(ValueError, match="coverage"):
        prepare(f)


@pytest.mark.parametrize("role", ["fit", "stop", "calibration", "policy"])
def test_original_support_failures_stop_without_fits_or_replacement(tmp_path, monkeypatch, role):
    initial = fixture(tmp_path / "source", monkeypatch)
    chosen = [p for p in initial.pairs if p["fold"] == 0 and p["role"] == role]
    group = chosen[0]["group_id"]
    chosen = [p for p in chosen if p["group_id"] == group]
    missing = [p["positive_index"] for p in (chosen[:2] if role == "fit" else chosen)]
    f = fixture(tmp_path / "masked", monkeypatch, unavailable=missing)
    with prepare(f) as prepared:
        result = study.audit_common_support(prepared, guard=lambda: None)
    assert result["status"] == "support_failed" and not result["fits_started"]
    assert any(f"fold_0_{role}" in k for k in result["failed_criteria"])
    cell = next(c for c in result["cells"] if c["fold"] == 0 and c["role"] == role)
    assert cell["observable_events"] == len(
        [p for p in initial.pairs if p["fold"] == 0 and p["role"] == role]
    )
    assert len(cell["excluded_pairs"]) == len(missing)


def test_support_requires_declared_conditional_stage(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    f.payload["allowed_stages"] = ["preflight"]
    f.declaration.write_text(json.dumps(f.payload))
    with prepare(f) as p, pytest.raises(ValueError, match="stage"):
        study.audit_common_support(p, guard=lambda: None)


@pytest.mark.parametrize(
    "fault",
    ["missing_raw_post", "changed_raw_post", "missing_source_post", "changed_source_post", "wrong_preflight"],
)
def test_complete_runner_without_exact_raw_and_source_postchecks_cannot_expose_rows(
    tmp_path, monkeypatch, fault
):
    f = fixture(tmp_path, monkeypatch)
    path = f.root / f.payload["runner_receipt"]
    receipt = json.loads(path.read_text())
    if fault == "missing_raw_post":
        receipt.pop("raw_hashes_after")
    elif fault == "missing_source_post":
        receipt.pop("hashes_after")
    elif fault == "changed_raw_post":
        receipt["raw_hashes_after"] = dict(files_sha256={"opaque.h5": "c" * 64}, hash_errors={})
    elif fault == "changed_source_post":
        receipt["hashes_after"] = dict(files_sha256={"wrong.py": "c" * 64}, hash_errors={})
    else:
        receipt["preflight_sha256"] = "c" * 64
    path.write_text(json.dumps(receipt))
    rebind(f, path)
    monkeypatch.setattr(study, "CompactRows", lambda *a: pytest.fail("rows exposed without postchecks"))
    with pytest.raises(ValueError, match="post|authority"):
        prepare(f)


@pytest.mark.parametrize("fault", ["wrong_label", "wrong_person", "wrong_time", "overlap", "duplicate"])
def test_saved_pairs_validate_semantics_without_regenerating_pairs(tmp_path, monkeypatch, fault):
    f = fixture(tmp_path, monkeypatch)
    with prepare(f) as p:
        pairs = list(f.pairs)
        pairs[0] = dict(pairs[0])
        if fault == "wrong_label":
            pairs[0]["positive_index"] = pairs[0]["negative_index"]
        elif fault == "wrong_person":
            pairs[0]["patient_id"] = "other"
        elif fault == "wrong_time":
            pairs[0]["negative_index"] += 50
        elif fault == "overlap":
            pairs[0]["negative_index"] = pairs[0]["positive_index"] - 1
        else:
            pairs.append(dict(pairs[0]))
        (f.audit / "pairs.jsonl").write_text("".join(json.dumps(v) + "\n" for v in pairs))
        membership, _ = runner._roles(p.roles, set(p.index["allowed_groups"]))
        events = {(e["group_id"], e["patient_id"], e["record_id"], e["onset"]): e for e in p.events}
        with pytest.raises(ValueError, match="Saved pair|denominator"):
            study._saved_pairs(
                f.payload, f.root, p.rows, p.roles, p.original_support, membership, events, lambda: None
            )


@pytest.mark.parametrize("name", ["metadata_audit", "pair_support"])
def test_original_audit_artifacts_cannot_be_replaced_by_prospectively_rehashed_files(
    tmp_path, monkeypatch, name
):
    f = fixture(tmp_path, monkeypatch)
    path = f.root / f.payload[name]
    path.write_text(path.read_text() + " ")
    rebind(f, path)
    with pytest.raises(ValueError, match="original|Original"):
        prepare(f)


@pytest.mark.parametrize("phase", ["prepare", "support"])
def test_study_resource_cap_stops_before_phase_reads(tmp_path, monkeypatch, phase):
    f = fixture(tmp_path, monkeypatch)
    if phase == "prepare":
        monkeypatch.setattr(study, "_rss_bytes", lambda: 10_000_000_001, raising=False)
        with pytest.raises(RuntimeError, match="resource"):
            prepare(f)
    else:
        with prepare(f) as p:
            monkeypatch.setattr(study, "_rss_bytes", lambda: 10_000_000_001, raising=False)
            with pytest.raises(RuntimeError, match="resource"):
                study.audit_common_support(p, guard=lambda: None)


def test_support_rechecks_storage_and_does_not_assume_preparation_measurement_is_current(
    tmp_path, monkeypatch
):
    f = fixture(tmp_path, monkeypatch)
    usage = dict(allocated_bytes=0, logical_bytes=0)
    f.storage = lambda: dict(usage)
    with prepare(f) as p:
        usage["logical_bytes"] = 24_900_000_000
        with pytest.raises(RuntimeError, match="storage"):
            study.audit_common_support(p, guard=lambda: None)
