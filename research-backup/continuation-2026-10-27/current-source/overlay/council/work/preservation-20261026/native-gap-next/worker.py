"""One bounded development-cache cost measurement; no forecasting fit or dispatch."""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import fixed_metadata
import frozen_replay
import leaf_measurement
import metadata_geometry
import numpy as np
import package_replay
import report_codec
import resource_reservation
import table_scoring
from artifact_limits import check_tree

from cardiotwin import ecgfounder_compare as compare
from cardiotwin import ecgfounder_study as study
from cardiotwin import evaluation
from cardiotwin.investigation import CompactRows

CACHE = 'data/iridia-development-cache-recovered'
ROLES = 'reports/ml-investigation/next-cycle/precursor-diagnostic-20261012-cadence-fix/roles.json'
SUPPORT = 'reports/ml-investigation/next-cycle/precursor-diagnostic-20261012-cadence-fix/pair-support.json'
PAIRS = 'reports/ml-investigation/next-cycle/precursor-diagnostic-20261012-cadence-fix/pairs.jsonl'
CONFIG = 'configs/precursor-cycle.json'
INPUTS = (f'{CACHE}/index.json', f'{CACHE}/compact.json', f'{CACHE}/features.npy',
          f'{CACHE}/metadata.npy', ROLES, SUPPORT, PAIRS, CONFIG)


def _sha(path, guard):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            guard()
            digest.update(block)
    return digest.hexdigest()


def measure(root, output, *, input_sha256, guard, storage_check):
    """Run only pinned source leaves with caller-owned session/calendar/resource guards."""
    root, output = Path(root).resolve(), Path(output).resolve()
    retained_root = output.parent
    if not output.is_relative_to(root) or output.exists():
        raise ValueError('Fresh output inside staged development root required')
    if set(input_sha256) != set(INPUTS):
        raise ValueError('Complete fixed development-only input pins required')
    if any(len(v) != 64 for v in input_sha256.values()):
        raise ValueError('SHA256 input pins required')
    started = time.monotonic()

    def work():
        guard()
        if time.monotonic() - started >= 1800:
            raise TimeoutError('Composite post-dependency worker ceiling reached')

    def component_guard():
        component_started = time.monotonic()

        def bounded():
            work()
            if time.monotonic() - component_started >= 1200:
                raise TimeoutError('Declared cost component ceiling reached')

        return bounded

    def emit(phase, **values):
        print(json.dumps(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), phase=phase,
                              elapsed_seconds=time.monotonic() - started, **values)), flush=True)

    for name, digest in input_sha256.items():
        work()
        if _sha(root / name, work) != digest:
            raise ValueError('Pinned development input differs: ' + name)
    index = json.loads((root / f'{CACHE}/index.json').read_text())
    compact = json.loads((root / f'{CACHE}/compact.json').read_text())
    roles = json.loads((root / ROLES).read_text())['roles']
    support = json.loads((root / SUPPORT).read_text())
    config = json.loads((root / CONFIG).read_text())
    counts = json.loads((Path(__file__).with_name('report-metadata-counts.json')).read_text())
    role_counts = json.loads((Path(__file__).parent.parent / 'native-metadata-counts.json').read_text())['cells']
    shape = metadata_geometry.geometry(index, roles)
    metadata_geometry.validate_expected(shape, counts['totals'], counts['folds'], role_counts)
    if (index['row_count'], len(index['records']), len(index['allowed_groups']), len(index['events'])) != (
            629444, 132, 121, 241):
        raise ValueError('Original complete development cache cardinality differs')
    if compact['checksums'] != {name: input_sha256[f'{CACHE}/{name}']
                               for name in ('features.npy', 'metadata.npy')}:
        raise ValueError('Pinned native array checksums differ from compact contract')
    output.mkdir(parents=True)
    receipt = dict(status='running', input_sha256=input_sha256, phases={}, outcome_fits=0,
                   real_forecasting_metrics=0, full_run_eligible=False)
    template=report_codec.policy_report_template(config,evaluation,study._person_rows)
    bounded=report_codec.bounded_policy_report(template,warnings=3,missed=3,people=2,fold=0,variant=0,
        person_rows=study._person_rows,reason_categories=compact['categories']['reason'],
        label_reason_categories=compact['categories']['label_reason'])
    receipt['source_resource_reservation']=resource_reservation.reservation(
        warning=bounded['warning_details'][0],missed=bounded['missed_event_details'][0],person=bounded['per_patient'][0])
    if not receipt['source_resource_reservation']['admissible']:
        (output/'receipt.json').write_text(json.dumps(dict(receipt,status='source_reservation_rejected'),sort_keys=True))
        raise RuntimeError('Prospective source RSS reservation exceeds unchanged10GB ceiling')

    def checkpoint(phase, result):
        work()
        storage_check()
        receipt['phases'][phase] = result
        (output / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True, allow_nan=False))
        check_tree(retained_root, study)
        emit(phase, completed=True)

    original = CompactRows(root / CACHE, compact)
    try:
        rows = study.IndexedRows(original)
        features = study.FeatureRows(rows)
        if len(features) != 629444 or original.features.shape != (629444, 112):
            raise ValueError('Complete native112 cache shape required')
        emit('native_chronological', completed=0, total=629444)
        checkpoint('native_chronological', leaf_measurement.native_matrix_pass(
            features, count=629444, indices=range(629444), guard=component_guard(), compare=compare))
        scattered = np.random.default_rng(42).permutation(629444)
        emit('native_scattered', completed=0, total=629444)
        checkpoint('native_scattered', leaf_measurement.native_matrix_pass(
            features, count=629444, indices=scattered, guard=component_guard(), compare=compare))
        before = time.monotonic()
        for _ in range(133):
            work()
            if compare._timing_matrix([features[0]]).shape != (1, 46):
                raise ValueError('Fixed native batch-control shape differs')
        checkpoint('native_batch_controls', dict(seconds=time.monotonic() - before,
            batches=133, rows=133, scope='conservative fixed-batch slack beyond12fullpass projection'))
        generated_root = root / 'data/native-cost-generated-tables'
        checkpoint('table_scoring', table_scoring.measure(generated_root,
            shape=shape, guard=component_guard(), storage_check=storage_check))
        for name in ('delta.npy', 'nuisance.npy', 'rows.npy', 'table-scoring.json'):
            (generated_root / name).unlink()
        generated_root.rmdir()
        geometry_root = root / 'data/native-cost-generated-metadata'
        checkpoint('generated_geometry', metadata_geometry.measure(shape, study=study,
            guard=component_guard(), output=geometry_root))
        for name in ('metadata.npy', 'rows.npy', 'features.npy'):
            (geometry_root / name).unlink()
        geometry_root.rmdir()
        checkpoint('metadata', leaf_measurement.metadata_leaves(
            rows=rows, compact_rows=original, compact=compact, index=index, roles=roles,
            guard=component_guard(), np=np, study=study))
        membership, _ = study._roles(roles, set(index['allowed_groups']))
        events = {(e['group_id'], e['patient_id'], e['record_id'], float(e['onset'])): e
                  for e in index['events']}
        checkpoint('pairs', leaf_measurement.fixed_pair_leaves(
            payload={'pairs': PAIRS}, root=root, rows=rows, roles=roles, support=support,
            membership=membership, events=events, guard=component_guard(), np=np, study=study))
        counts = counts['folds']
        fixed = fixed_metadata.generated_fixed(
            config=config, fold_people=[cell['people'] for cell in counts],
            fold_events=[cell['events'] for cell in counts], study=study, guard=component_guard())
        checkpoint('packages', package_replay.measure(output / 'packages',
            identity=report_codec._full_identity_envelope(fixed['entry']['identity']),
            study=study, compare=compare, guard=component_guard(), storage_check=storage_check,
            artifact_root=retained_root))
        checkpoint('control', fixed_metadata.measure_control_replay_leaves(
            study=study, guard=component_guard(), identity_rows=629444))
        checkpoint('frozen_replay', frozen_replay.measure(config=config, study=study,
            evaluation=evaluation, guard=component_guard(),policy_output=output/'policies',shape=shape,
            fixed=dict(fixed['entry'],identity=report_codec._full_identity_envelope(fixed['entry']['identity'])),
            storage_check=storage_check,artifact_root=retained_root))
        checkpoint('replay_evaluations', frozen_replay.numerical_evaluations(config=config,
            shape=shape, evaluation=evaluation, guard=component_guard()))
        checkpoint('reports', report_codec.write_eight_reports(
            output / 'reports', config=config, fixed=fixed,
            fold_warning_bounds=[cell['warning_upper'] for cell in counts],
            fold_rows=[cell['rows'] for cell in counts],
            fold_people=[cell['people'] for cell in counts],
            fold_events=[cell['events'] for cell in counts],
            reason_categories=compact['categories']['reason'],
            label_reason_categories=compact['categories']['label_reason'],
            guard=component_guard(), storage_check=storage_check, study=study, evaluation=evaluation,
            artifact_root=retained_root))
        receipt['status'] = 'complete_cost_measurement_only'
        receipt['elapsed_seconds'] = time.monotonic() - started
        (output / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True, allow_nan=False))
        return receipt
    except BaseException as error:
        receipt.update(status='failed', error_type=type(error).__name__, original_error=str(error),
                       elapsed_seconds=time.monotonic() - started)
        (output / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True, allow_nan=False))
        raise
    finally:
        for array in (original.features, original.metadata):
            array._mmap.close()
