"""Source-bound native cost leaves only; no CLI, fitting or dispatch authority."""
import ast
import hashlib
import inspect
import time
from collections import Counter
from types import SimpleNamespace


def source_block(function, predicate):
    """Compile the unique original block without changing its statements or guards."""
    source = inspect.getsource(function)
    matches = [node for node in ast.walk(ast.parse(source)) if predicate(node, ast.get_source_segment(source, node))]
    if len(matches) != 1:
        raise ValueError('Production block identity is ambiguous or missing')
    node = matches[0]
    text = ast.get_source_segment(source, node)
    tree = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(tree)
    return compile(tree, inspect.getsourcefile(function), 'exec'), hashlib.sha256(text.encode()).hexdigest()


def run_block(function, predicate, values):
    code, digest = source_block(function, predicate)
    environment = dict(function.__globals__)
    environment.update(values)
    started = time.monotonic()
    exec(code, environment)
    return {'seconds': time.monotonic() - started, 'source_block_sha256': digest}, environment


def metadata_leaves(*, rows, compact_rows, compact, index, roles, guard, np, study):
    """Exact production loops with declared synthetic availability; never real support authority."""
    available = np.empty(len(rows), dtype=[('available', '?')])
    available['available'] = True
    tables = SimpleNamespace(rows=available)
    results = {'prepare': [], 'support_class_scan': [], 'synthetic_availability': 'all_available'}
    for _ in range(2):
        guard()
        membership, _ = study._roles(roles, set(index['allowed_groups']))
        values = dict(rows=rows, original_rows=compact_rows, guard=guard, index=index, compact=compact,
                      tables=tables, membership=membership,
                      counts={(f, n): [0, 0, 0] for f in range(3) for n in study.ROLE_NAMES}, previous={})
        coverage, coverage_env = run_block(study.prepare_study,
            lambda node, text: isinstance(node, ast.With) and 'ecgfounder.study_identity_coverage' in text,
            values)
        counts = coverage_env['counts']
        matching = {'cells': [dict(fold=f, role=name, **study.coverage_cell(*counts[f, name]))
                              for f in range(3) for name in study.ROLE_NAMES]}
        comparison, _ = run_block(study.prepare_study,
            lambda node, text: isinstance(node, ast.For) and text.startswith('for cell in coverage["cells"]:'),
            dict(coverage=matching, counts=counts))
        started = time.monotonic()
        def guarded_rows():
            for row in rows:
                guard()
                yield row
        identity = study.identity_digest(guarded_rows())
        identity_seconds = time.monotonic() - started
        comparison_identity, _ = run_block(study.prepare_study,
            lambda node, text: isinstance(node, ast.If) and text.startswith('if identity_digest(guarded_rows())'),
            dict(identity_digest=lambda unused: identity, guarded_rows=guarded_rows,
                 contract={'identity_sha256': identity}))
        label, _ = run_block(study.prepare_study,
            lambda node, text: isinstance(node, ast.For) and text.startswith('for meta in original_rows.metadata:'),
            values)
        results['prepare'].append(dict(coverage=coverage, coverage_comparison=comparison,
                                       identity_seconds=identity_seconds,
                                       identity_equality=comparison_identity,
                                       identity_sha256=identity, label_code_validation=label))
    prepared = SimpleNamespace(rows=rows, tables=tables, roles=roles, index=index)
    for _ in range(3):
        guard()
        membership, _ = study._roles(roles, set(index['allowed_groups']))
        scan, _ = run_block(study.audit_common_support,
            lambda node, text: isinstance(node, ast.For) and text.startswith('for i, row in enumerate(prepared.rows):'),
            dict(prepared=prepared, guard=guard, membership=membership,
                 classes={(f, n): Counter() for f in range(3) for n in ('calibration', 'policy')}))
        results['support_class_scan'].append(scan)
    results['rows_per_scan'] = len(rows)
    results['real_support_authority'] = False
    return results


def native_matrix_pass(features, *, count, indices, guard, compare):
    """One complete working-set unit; every selected row is visited once."""
    if len(indices) != count or count != len(features):
        raise ValueError('Complete fixed working-set unit required')
    # Caller predeclares either range(N) or an independently verified N-row permutation.
    if not isinstance(indices, range) or indices != range(count):
        if sorted(map(int, indices)) != list(range(count)):
            raise ValueError('Scattered unit must be an exact complete permutation')
    started = time.monotonic()
    completed = batches = 0
    for start in range(0, count, compare.BATCH_ROWS):
        guard()
        selected = indices[start:start + compare.BATCH_ROWS]
        matrix = compare._timing_matrix([features[int(i)] for i in selected])
        if matrix.shape != (len(selected), 46):
            raise ValueError('Actual native timing matrix shape differs')
        completed += len(selected)
        batches += 1
    guard()
    if completed != count:
        raise ValueError('Incomplete native unit')
    return dict(seconds=time.monotonic() - started, rows=completed, batches=batches,
                upper_native_scalar_lookups=69 * completed, scope='native112_mapping_timing_matrix_only')


def fixed_pair_leaves(*, payload, root, rows, roles, support, membership, events, guard, np, study):
    """Validate only original saved pairs; synthetic availability never authorizes support."""
    validation = []
    for _ in range(2):
        guard()
        started = time.monotonic()
        pairs = study._saved_pairs(payload, root, rows, roles, support, membership, events, guard)
        validation.append(time.monotonic() - started)
    filtering = []
    original = {f['fold']: f['roles'] for f in support['folds']}
    for available in (True, False):
        table_rows = np.zeros(len(rows), dtype=[('available', '?'), ('reason_counts', 'u1', (len(study.REASONS),))])
        table_rows['available'] = available
        if not available:
            table_rows['reason_counts'][:] = 1
        prepared = SimpleNamespace(rows=rows, pairs=pairs, tables=SimpleNamespace(rows=table_rows))
        values = dict(prepared=prepared, guard=guard, original=original,
                      cells=[], failed=[], surviving={f: {n: [] for n in study.ROLE_NAMES} for f in range(3)},
                      classes={(f, n): Counter({0: 1, 1: 1}) for f in range(3) for n in ('calibration', 'policy')})
        measurement, filtered = run_block(study.audit_common_support,
            lambda node, text: isinstance(node, ast.For) and text.startswith('for f in range(3):')
            and 'for pair in prepared.pairs[f][name]:' in text, values)
        started = time.monotonic()
        serialized = {f: {n: [study.asdict(pair) for pair in filtered['surviving'][f][n]]
                          for n in study.ROLE_NAMES} for f in range(3)}
        filtering.append(dict(measurement, synthetic_all_available=available,
                              surviving_pair_serialization_seconds=time.monotonic() - started,
                              surviving_pair_count=sum(len(v) for fold in serialized.values()
                                                       for v in fold.values())))
    return dict(saved_pair_validation_seconds=validation, filtering=filtering,
                saved_pairs_function_sha256=hashlib.sha256(inspect.getsource(study._saved_pairs).encode()).hexdigest(),
                fixed_pair_count=sum(len(p) for fold in pairs.values() for p in fold.values()),
                real_support_authority=False, new_pairs=0, outcome_fits=0)


def report_layout(fold_rows, fold_records):
    """Source count ceilings for the six receipts and their two full copies."""
    if len(fold_rows) != 3 or len(fold_records) != 3 or sum(fold_rows) != 629444 or sum(fold_records) != 132:
        raise ValueError('Original complete outer cohort and record partition required')
    if any(type(n) is not int or n <= 0 for n in (*fold_rows, *fold_records)):
        raise ValueError('Positive integral fold sizes required')
    # Each record can reset the 600-second cooldown; 30-second cadence permits one warning per 20 rows.
    per_fold = [(n + 19) // 20 + records for n, records in zip(fold_rows, fold_records, strict=True)]
    if sum(per_fold) != 31605:
        raise ValueError('Bounded warning partition differs from fixed source ceiling')
    names = [f'fold-{fold}-{arm}-receipt.json.gz' for fold in range(3) for arm in ('timing', 'candidate')]
    return dict(arm_receipts=names, summary='summary.json.gz', replay_review='review.json.gz',
                warning_upper_per_fold_policy=per_fold,
                arm_receipt_warning_details=2 * 10 * sum(per_fold),
                all_eight_warning_details=3 * 2 * 10 * sum(per_fold),
                all_eight_missed_event_details=3 * 2 * 10 * 241,
                source_scope='count ceilings only; production field layout and codec still require measurement')


def full_report_leaves(output, *, guard, storage_check, primitives, study):
    """Legacy warning-growth helper only; never sufficient for full report acceptance."""
    from pathlib import Path
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    reports = []
    for number in range(6):
        guard()
        if time.monotonic() - started >= 1200:
            raise TimeoutError('Full generated report component ceiling reached')
        result = primitives.report_work(output / f'arm-copy-{number}', guard=guard,
                                        storage_check=storage_check, warning_count=31605, policies=10)
        reports.append(result)
        if study._artifact_bytes(output) > 250_000_000:
            raise RuntimeError('Full generated report diagnostic ceiling reached')
    warnings = sum(item['sample']['warnings'] for item in reports)
    missed = sum(item['sample']['missed_events'] for item in reports)
    if (warnings, missed) != (1_896_300, 14_460):
        raise ValueError('Full persisted report detail counts differ')
    return dict(seconds=time.monotonic() - started, arm_copy_reports=reports,
                persisted_warnings=warnings, persisted_missed=missed,
                artifact_bytes=study._artifact_bytes(output),
                helper_source_sha256=hashlib.sha256(inspect.getsource(primitives.report_work).encode()).hexdigest(),
                scope='legacy six generated growth templates; omits production summary/replay and per_group',
                report_layout_complete=False,
                outcome_fits=0, forecast_scores=0, full_run_eligible=False)
