"""Generated geometry from pinned development metadata; never real rows or real pairs."""
import hashlib
import inspect
import json
import time
from collections import Counter
from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np

from cardiotwin.ecgfounder_runner import _roles
from cardiotwin.ecgfounder_tables import ROW_DTYPE
from cardiotwin.investigation import CompactRows


def geometry(index, roles):
    """Discard original identities/onsets after deriving record sizes and event counts."""
    if len(index['allowed_groups']) != len(set(index['allowed_groups'])):
        raise ValueError('Unique original groups required')
    _roles(roles, set(index['allowed_groups']))
    group_names = {name: f'g{i:03d}' for i, name in enumerate(sorted(index['allowed_groups']))}
    patients = {name: f'p{i:03d}' for i, name in enumerate(sorted(index['group_ids']))}
    counts = Counter((e['patient_id'], e['record_id']) for e in index['events'])
    records, seen = [], set()
    for i, item in enumerate(index['records']):
        original = item['record']
        key = (original['patient_id'], original['record_id'])
        if key in seen or type(item['rows']) is not int or item['rows'] <= 0:
            raise ValueError('Unique positive record cardinalities required')
        seen.add(key)
        records.append(dict(record_id=f'r{i:03d}', patient_id=patients[key[0]],
                            group_id=group_names[index['group_ids'][key[0]]],
                            rows=item['rows'], events=counts.pop(key, 0)))
    if counts or sum(r['rows'] for r in records) != index['row_count']:
        raise ValueError('Exact record/event membership required')
    generated_roles = [dict(fold=cell['fold'], **{name: [group_names[g] for g in cell[name]]
                           for name in ('fit', 'stop', 'calibration', 'policy', 'outer')})
                       for cell in sorted(roles, key=lambda cell: cell['fold'])]
    result = dict(records=records, roles=generated_roles)
    validate_shape(result)
    return result


def validate_shape(shape):
    records, roles = shape['records'], shape['roles']
    if not records or [cell['fold'] for cell in roles] != [0, 1, 2]:
        raise ValueError('Generated three-fold record geometry required')
    if len({r['record_id'] for r in records}) != len(records):
        raise ValueError('Unique generated records required')
    for record in records:
        if (type(record['rows']) is not int or record['rows'] <= 0
                or type(record['events']) is not int or record['events'] < 0):
            raise ValueError('Integral positive rows and nonnegative event counts required')
    groups = {r['group_id'] for r in records}
    for cell in roles:
        for name in ('fit', 'stop', 'calibration', 'policy', 'outer'):
            members = cell.get(name, [])
            if len(members) != len(set(members)) or not set(members) <= groups:
                raise ValueError('Unique known generated role members required')


def scope_records(shape, fold, role):
    groups = set(shape['roles'][fold][role])
    return [record for record in shape['records'] if record['group_id'] in groups]


def validate_expected(shape, totals, outer_counts, role_counts):
    """Check the existing metadata receipts without reading arrays or real labels."""
    validate_shape(shape)
    summary = public_summary(shape)
    if any(summary[key] != totals[key] for key in ('records', 'rows', 'people', 'events')):
        raise ValueError('Generated record/event aggregate differs from pinned metadata')
    for fold in range(3):
        for name, expected in role_counts[fold]['rows'].items():
            if sum(r['rows'] for r in scope_records(shape, fold, name)) != expected:
                raise ValueError('Generated role row count differs from pinned metadata')
        records = scope_records(shape, fold, 'outer')
        observed = dict(records=len(records), rows=sum(r['rows'] for r in records),
                        people=len({r['group_id'] for r in records}), events=sum(r['events'] for r in records))
        if any(observed[key] != outer_counts[fold][key] for key in observed):
            raise ValueError('Generated outer record/event geometry differs from pinned metadata')


def public_summary(shape):
    records = shape['records']
    encoded = json.dumps(shape, sort_keys=True, separators=(',', ':')).encode()
    return dict(records=len(records), rows=sum(r['rows'] for r in records),
                people=len({r['group_id'] for r in records}), events=sum(r['events'] for r in records),
                all_missed_row_event_visits=sum(r['rows'] * r['events'] for r in records),
                shape_sha256=hashlib.sha256(encoded).hexdigest(),
                detail_policy='Runtime shapes are private evidence; publish aggregate counts/hash only')


def generated_prepared(shape, study, guard, output=None):
    """Use the unchanged native view chain over generated structured arrays only."""
    validate_shape(shape)
    count = sum(record['rows'] for record in shape['records'])
    if count > 17 and output is None:
        raise ValueError('Complete generated metadata requires its own mmap scratch')
    dtype = np.dtype([('record', '<i4'), ('timestamp', '<f8'),
        ('duration_seconds', '<f8'), ('accepted', 'i1'), ('label', 'i1'),
        ('reason', '<i2'), ('label_reason', '<i2')])
    if output is None:
        metadata, table_rows = np.zeros(count, dtype=dtype), np.zeros(count, dtype=ROW_DTYPE)
    else:
        from pathlib import Path
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        metadata = np.lib.format.open_memmap(output/'metadata.npy', mode='w+', dtype=dtype, shape=(count,))
        table_rows = np.lib.format.open_memmap(output/'rows.npy', mode='w+', dtype=ROW_DTYPE, shape=(count,))
        unused = np.lib.format.open_memmap(output/'features.npy', mode='w+', dtype='<f8', shape=(count,0))
        unused.flush()
        unused._mmap.close()
        metadata[:] = 0
        table_rows[:] = 0
    table_rows['available'], table_rows['accepted'] = True, True
    offset = 0
    for i, record in enumerate(shape['records']):
        guard()
        stop = offset + record['rows']
        metadata['record'][offset:stop] = table_rows['record_index'][offset:stop] = i
        metadata['timestamp'][offset:stop] = np.arange(record['rows']) * 30.
        metadata['duration_seconds'][offset:stop] = record['rows'] * 30.
        metadata['accepted'][offset:stop] = 1
        offset = stop
    contract = dict(records=shape['records'], features=[], categories={'reason':[None], 'label_reason':[None]})
    if output is None:
        compact = CompactRows.__new__(CompactRows)
        compact.metadata, compact.records = metadata, shape['records']
        compact.categories, compact.names, compact.positions = contract['categories'], [], {}
        compact.features = np.empty((count,0))
    else:
        metadata.flush()
        metadata._mmap.close()
        compact = CompactRows(output, contract)
    indexed = study.IndexedRows(compact)

    class GuardedRows(Sequence):
        def __len__(self):
            return len(indexed)

        def __getitem__(self, index):
            if index % 256 == 0:
                guard()
            return indexed[index]

    tables = SimpleNamespace(rows=table_rows, records=[tuple(r[k] for k in
        ('record_id', 'group_id', 'patient_id')) for r in shape['records']])
    return SimpleNamespace(rows=GuardedRows(), tables=tables, generated_compact=compact)


def measure(shape, *, study, guard, output=None):
    """Exactly48 full-index role calls and12 source-equivalent generated drift scans."""
    before = time.monotonic()
    prepared = generated_prepared(shape, study, guard, output)
    setup = time.monotonic() - before
    roles, drifts = [], []
    for owner in ('complete_timelines', 'artifact_verification_download'):
        for fold in range(3):
            # Source: fit+stop once, calibration+policy+outer once per each of two primary arms.
            for name in ('fit', 'stop', 'calibration', 'policy', 'outer', 'calibration', 'policy', 'outer'):
                guard()
                before = time.monotonic()
                selected = study._role_indices(prepared, shape['roles'][fold][name])
                expected = sum(r['rows'] for r in scope_records(shape, fold, name))
                if len(selected) != expected:
                    raise ValueError('Generated exact role-index cardinality differs')
                roles.append(dict(owner=owner, fold=fold, role=name, rows=expected,
                                  seconds=time.monotonic() - before))
            for name in ('fit', 'stop'):
                guard()
                before = time.monotonic()
                pairs = study.negative_drift_pairs(study._CommonRows(prepared), set(shape['roles'][fold][name]))
                # Complete30s zero-origin record geometry: later=120*k and earlier=later-78.
                expected = sum((r['rows'] - 1) // 120 for r in scope_records(shape, fold, name))
                if len(pairs) != expected:
                    raise ValueError('Generated drift maximum differs from fixed chronology geometry')
                drifts.append(dict(owner=owner, fold=fold, role=name, pairs=len(pairs),
                    rows_scanned=len(prepared.rows), seconds=time.monotonic() - before))
                del pairs
    if any(sum(cell['pairs'] for cell in drifts if cell['owner'] == owner) > 7533
           for owner in ('complete_timelines', 'artifact_verification_download')):
        raise ValueError('Generated drift geometry exceeds retained per-arm source bound')
    result = dict(setup_seconds=setup, role_index_calls=roles, drift_scan_calls=drifts,
        aggregate=public_summary(shape), real_pairs=0, real_support_authority=False,
        source_sha256={name: hashlib.sha256(inspect.getsource(getattr(study, name)).encode()).hexdigest()
                       for name in ('_role_indices', '_CommonRows', 'negative_drift_pairs')},
        synthetic_metadata_labels='all zero; common availability all true; no original labels/onsets')
    for value in (prepared.generated_compact.metadata, prepared.generated_compact.features, prepared.tables.rows):
        if getattr(value, '_mmap', None):
            value._mmap.close()
    return result
