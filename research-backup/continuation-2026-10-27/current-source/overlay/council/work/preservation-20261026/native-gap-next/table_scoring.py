"""Generated full-working-set table/scoring primitive; no fitting or real outcomes."""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cardiotwin import ecgfounder_compare as compare
from cardiotwin.ecgfounder_tables import ROW_DTYPE
from cardiotwin.precursor import LINEAR_SCHEMA


def numeric_heads():
    """Prospectively fixed finite sklearn states, assigned directly without fit calls."""
    def imputer(width):
        value = SimpleImputer()
        value.n_features_in_ = width
        value._fit_dtype = value._fill_dtype = np.dtype('float64')
        value.indicator_ = None
        value.statistics_ = np.linspace(-0.125, 0.125, width)
        return value

    def scaler(width):
        value = StandardScaler()
        value.n_features_in_, value.n_samples_seen_ = width, np.int64(2)
        value.mean_ = np.linspace(-0.25, 0.25, width)
        value.var_ = np.linspace(0.5, 1.5, width)
        value.scale_ = np.sqrt(value.var_)
        return value

    transforms = compare.FitTransforms(imputer(46), scaler(46), scaler(1024),
                                        np.array([0, 1]), np.array([0.5, 0.5]))
    heads = {}
    for kind, width in (('timing', 46), ('candidate', 1070), ('nuisance', 52)):
        model = LogisticRegression(C=0.1, solver='lbfgs', max_iter=3000,
                                   fit_intercept=False, random_state=42)
        model.coef_ = np.linspace(-0.003, 0.004, width).reshape(1, -1)
        model.intercept_, model.classes_ = np.array([0.]), np.array([0, 1])
        model.n_features_in_, model.n_iter_ = width, np.array([0])
        head = compare.PointwiseHead(model, transforms, kind, 'fit_stop_only', frozenset({'g0'}))
        if kind == 'nuisance':
            head.nuisance_imputer, head.nuisance_scaler = imputer(6), scaler(6)
        heads[kind] = head
    return heads


class GeneratedTiming:
    def __getitem__(self, index):
        return {name: (None if (index + j) % 31 == 0 else ((index * 17 + j * 13) % 997) / 997)
                for j, name in enumerate(LINEAR_SCHEMA)}


def measure(output, *, guard, storage_check, count=629444, shape=None):
    """Full size on worker only; count<=17 is reserved for generated software checks."""
    if count != 629444 and not 2 <= count <= 17:
        raise ValueError('Complete declared working set or tiny software fixture required')
    if shape is None:
        if count > 17:
            raise ValueError('Pinned generated record geometry required for full table scoring')
        shape = dict(records=[dict(record_id='r0', group_id='g0', patient_id='p0', rows=count)])
    records = shape['records']
    if sum(r['rows'] for r in records) != count or len({r['record_id'] for r in records}) != len(records):
        raise ValueError('Exact unique generated table record geometry required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    arrays, hashes = {}, {}
    specs = {'delta.npy': ((count, 1024), '<f4'), 'nuisance.npy': ((count, 6), '<f4'),
             'rows.npy': ((count,), ROW_DTYPE)}
    expected_data_bytes = count * (1024 * 4 + 6 * 4 + ROW_DTYPE.itemsize)
    if expected_data_bytes + 3 * 65536 > 2_619_117_060 + 250_000 + 65536:
        raise ValueError('Generated table allocation exceeds declared existing table allowance')
    try:
        for name, (shape, dtype) in specs.items():
            guard()
            storage_check()
            arrays[name] = np.lib.format.open_memmap(output / name, mode='w+', dtype=dtype, shape=shape)
        for start in range(0, count, 4096):
            guard()
            stop = min(count, start + 4096)
            ids = np.arange(start, stop, dtype=np.int64)
            arrays['delta.npy'][start:stop] = (((ids[:, None] * 17 + np.arange(1024) * 13) % 65521) / 65521 - .5)
            arrays['nuisance.npy'][start:stop] = (((ids[:, None] * 23 + np.arange(6) * 7) % 997) / 997)
            status = arrays['rows.npy'][start:stop]
            status['row_index'] = ids
            status['record_index'] = np.searchsorted(np.cumsum([r['rows'] for r in records]), ids, side='right')
            status['available'], status['accepted'] = ids % 97 != 0, True
            storage_check()
        for name, array in arrays.items():
            array.flush()
            with (output / name).open('rb') as handle:
                os.fsync(handle.fileno())
                digest = hashlib.sha256()
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    guard()
                    digest.update(block)
                hashes[name] = digest.hexdigest()
            array._mmap.close()
            arrays[name] = np.load(output / name, mmap_mode='r', allow_pickle=False)
        generation_seconds = time.monotonic() - started
        data = compare.VerifiedTables(arrays, records)
        heads, timing, results = numeric_heads(), GeneratedTiming(), {}
        for head in heads.values():
            head.allowed_groups = frozenset({'g0'} | {r['group_id'] for r in records})
        for order, indices in (('chronological', np.arange(count)),
                               ('scattered', np.random.default_rng(42).permutation(count))):
            for kind, head in heads.items():
                before, completed, checksum = time.monotonic(), 0, 0.0
                for offset in range(0, count, compare.BATCH_ROWS):
                    guard()
                    selected = indices[offset:offset + compare.BATCH_ROWS]
                    logits = head.logits(timing, data, selected)
                    if not np.array_equal(np.isfinite(logits), data.rows['available'][selected]):
                        raise ValueError('Generated availability/scoring parity differs')
                    checksum += float(np.nansum(logits))
                    completed += len(selected)
                results[order + '_' + kind] = dict(seconds=time.monotonic() - before,
                                                    rows=completed, checksum=checksum)
        branch = {}
        batch_controls = {}
        for kind, head in heads.items():
            before = time.monotonic()
            for _ in range(76):
                guard()
                head.logits(timing, data, [1])
            batch_controls[kind] = dict(seconds=time.monotonic() - before, calls=76, rows=76)
        for available in (False, True):
            fake = compare.VerifiedTables({'delta.npy': np.ones((2, 1024), dtype='<f4'),
                'nuisance.npy': np.ones((2, 6), dtype='<f4'),
                'rows.npy': np.zeros(2, dtype=ROW_DTYPE)},
                [dict(record_id='r0', group_id='g0', patient_id='p0')])
            fake.rows['available'] = available
            for kind, head in heads.items():
                scores = head.logits(timing, fake, [0, 1])
                if bool(np.isfinite(scores).all()) != available:
                    raise ValueError('Generated all-available/unavailable branch differs')
                branch[str(available) + '_' + kind] = len(scores)
        result = dict(generation_hash_fsync_seconds=generation_seconds, passes=results,
                      generated_branches=branch, fixed_batch_controls=batch_controls,
                      table_bytes=sum((output / n).stat().st_size for n in specs),
                      table_sha256=hashes, count=count, source='actual VerifiedTables.batch/PointwiseHead.logits',
                      generated_records=len(records), generated_groups=len({r['group_id'] for r in records}),
                      nested_generated_timing_mapping='measurement slack; native112 owner remains separate',
                      cold_claim=False, fits=0, full_run_eligible=False)
        (output / 'table-scoring.json').write_text(json.dumps(result, sort_keys=True, allow_nan=False))
        return result
    finally:
        for value in arrays.values():
            if getattr(value, '_mmap', None) and not value._mmap.closed:
                value._mmap.close()
