"""Full66 generated numeric packages at fixed identity/pair bounds, independent no-fit replay."""
import copy
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import table_scoring
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cardiotwin.precursor import EventPair


def measure(output, *, identity, study, compare, guard, storage_check, folds=3):
    if folds not in (1, 3):
        raise ValueError('Only tiny one-fold check or complete three-fold unit allowed')
    output = Path(output).resolve()
    output.mkdir(exist_ok=False, parents=True)
    started = time.monotonic()
    pairs = [EventPair('g0', 'p0', 'r0', 3600. + i * 600, i * 2, i * 2 + 1) for i in range(241)]
    heads = table_scoring.numeric_heads()
    indices, masses, _ = compare.pair_row_masses(pairs)
    heads['candidate'].transforms.fit_indices = indices
    heads['candidate'].transforms.row_masses = masses
    scale = copy.deepcopy(heads['timing'].transforms.timing_scaler)
    scale.n_features_in_, scale.mean_, scale.var_, scale.scale_ = 1, np.array([0.]), np.array([1.]), np.array([1.])
    sigmoid = copy.deepcopy(heads['timing'].model)
    sigmoid.set_params(C=100., max_iter=1000, fit_intercept=True)
    sigmoid.n_features_in_, sigmoid.coef_ = 1, np.array([[.1234567890123456]])
    calibrator = Pipeline([('scale', scale), ('sigmoid', sigmoid)])
    records, manifest = [], {'files_sha256': {}}
    for fold in range(folds):
        specs = [('timing', None), ('candidate', None)] + [('candidate', seed) for seed in range(101, 120)] + [('nuisance', 0)]
        for number, (kind, control) in enumerate(specs):
            guard()
            groups = None if control is None else frozenset({'g0'})
            head = replace(heads[kind], scope='primary' if groups is None else 'fit_stop_only', allowed_groups=groups)
            name = f'fold-{fold}-{number}.joblib'
            record = study._package_artifact(output, name, head, calibrator if groups is None else None,
                                              identity, 250_000_000, storage_check)
            manifest['files_sha256'][name] = record['sha256']
            manifest['files_sha256'][name + '.json'] = record['numeric_manifest_sha256']
            records.append((name, kind, groups))
    write_seconds = time.monotonic() - started

    def forbidden(*args, **kwargs):
        raise AssertionError('Independent package replay must never fit')

    before = time.monotonic()
    with patch.object(LogisticRegression, 'fit', forbidden), patch.object(SimpleImputer, 'fit', forbidden), \
            patch.object(StandardScaler, 'fit', forbidden):
        for name, kind, groups in records:
            study._replay_package(output, name, manifest, identity, pairs, kind, groups, guard)
    guard()
    return dict(packages=len(records), manifests=len(records), fit_pairs=241,
                write_seconds=write_seconds, independent_replay_seconds=time.monotonic() - before,
                bytes=study._artifact_bytes(output), fits=0, full_run_eligible=False)
