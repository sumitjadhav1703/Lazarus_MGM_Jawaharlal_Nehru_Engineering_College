"""Generated frozen-search reconstruction overhead; numerical evaluations owned separately."""
import copy
import hashlib
import inspect
import time
from unittest.mock import patch

import numpy as np


class GeneratedRows:
    def __init__(self, count, guard):
        self.count, self.guard = count, guard

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        if index % 256 == 0:
            self.guard()
        record = min(131, index * 132 // self.count)
        return dict(patient_id=f'p{record % 121}', group_id=f'g{record % 121}',
                    record_id=f'r{record}', timestamp=float(index * 30),
                    duration_seconds=float(self.count * 30), accepted=index % 97 != 0,
                    label=index % 2, risk=((index * 9973) % 1000003 + 1) / 1000004,
                    reason=None, label_reason=None)


def measure(*, count=629444, config, study, evaluation, guard):
    """Exercise actual replay equality/tie logic with explicitly substituted numerical kernels."""
    if count != 629444 and not 2 <= count <= 17:
        raise ValueError('Complete source working set or tiny generated fixture required')
    rows = GeneratedRows(count, guard)
    tiny = list(GeneratedRows(7, guard))
    numerical = evaluation.evaluate_stream(tiny, [], config, bootstrap_samples=0, include_probability=False)
    numerical.update(event_sensitivity=0., false_warnings_per_hour=0., warnings=0,
                     unassessable_warnings=0, warning_hours=0., eligible_events=241,
                     burden_occupancy=0., eligible_negative_burden_occupancy=0., burden_seconds=0.)
    risks = [r['risk'] for r in rows if r['accepted'] and r['label'] in (0, 1)]
    thresholds = sorted(set(np.r_[np.linspace(.01, .99, 25), .001, .005, .025, .995, .999, 1.,
                                 np.quantile(risks, [.5, .75, .9, .925, .95, .975, .99, .995, .999])]))
    freeze = dict(policies={name: {} for name in study.POLICY_NAMES},
                  searches={name: [dict(threshold=t) for t in thresholds] for name in study.POLICY_NAMES})
    calls = [0, 0]

    def stub(stream, events, policy, **kwargs):
        guard()
        calls[0] += 1
        calls[1] += len(stream)
        return numerical

    original = study._same_numeric_json

    def construct(expected, actual):
        if isinstance(expected, dict):
            actual.clear()
            actual.update(copy.deepcopy(expected))
            return True
        return original(expected, actual)

    before = time.monotonic()
    with patch.object(study, 'evaluate_stream', stub), patch.object(study, '_same_numeric_json', construct):
        study._replay_policies(rows, [], config, freeze, guard)
    setup_seconds = time.monotonic() - before
    calls[:] = [0, 0]
    before = time.monotonic()
    with patch.object(study, 'evaluate_stream', stub):
        study._replay_policies(rows, [], config, freeze, guard)
    return dict(reconstruction_seconds=time.monotonic() - before,
                engineering_freeze_construction_seconds=setup_seconds,
                numerical_evaluation_substituted=True, numerical_evaluation_calls=calls[0],
                numerical_evaluation_rows=calls[1], thresholds=len(thresholds), policies=10,
                numerical_owner='same-unit numerical_evaluations; old22H supplies counts only, no isolated rate',
                reconstruction_owner='artifact_verification_download; six policy streams target',
                source_sha256=hashlib.sha256(inspect.getsource(study._replay_policies).encode()).hexdigest(),
                fits=0, selectors=0, full_replay_latency=False, full_run_eligible=False)


def numerical_evaluations(*, count=629444, config, evaluation, guard):
    """Actual full-row replay evaluation kernels; no selector, outcomes or source substitutions."""
    if count != 629444 and not 2 <= count <= 17:
        raise ValueError('Complete source working set or tiny generated fixture required')

    class PatternRows(GeneratedRows):
        def __init__(self, pattern):
            super().__init__(count, guard)
            self.pattern = pattern

        def __getitem__(self, index):
            row = super().__getitem__(index)
            record = min(131, index * 132 // count)
            first = (record * count + 131) // 132
            end = min(count, ((record + 1) * count + 131) // 132)
            local = index - first
            row.update(timestamp=float(local * 30), duration_seconds=float((end - first) * 30))
            if self.pattern == 'warning_rich':
                row.update(accepted=True, risk=.9999999 if local % 20 < 3 else 1e-9)
            else:
                row.update(label=None if index % 101 == 0 else index % 2,
                           label_reason='unknown' if index % 101 == 0 else None,
                           reason='forecast_unavailable' if index % 97 == 0 else None)
            return row

    events = [dict(patient_id=f'p{record % 121}', record_id=f'r{record}', onset=3600. + 600 * i)
              for record in range(min(132, count)) for i in range(2 if record < 109 else 1)]
    results = {}
    for pattern in ('warning_rich', 'varied_masks'):
        for legacy in (False, True):
            guard()
            policy = copy.deepcopy(config)
            policy['alert'].update(threshold=.5, release_threshold=.25)
            if legacy:
                for key in ('policy_version', 'max_duration_seconds', 'rearm_seconds',
                            'max_burden_occupancy', 'max_eligible_negative_burden_occupancy'):
                    policy['alert'].pop(key, None)
            for probability in (False, True):
                before = time.monotonic()
                report = evaluation.evaluate_stream(PatternRows(pattern), events, policy,
                                                     bootstrap_samples=0, include_probability=probability)
                guard()
                results[pattern + ('_legacy' if legacy else '_finite') + ('_probability' if probability else '_search')] = dict(
                    seconds=time.monotonic() - before, rows=count,
                    warnings=len(report['warning_details']), missed=len(report['missed_event_details']),
                    groups=len(report['per_patient']))
    fixed_rows = list(GeneratedRows(min(count, 132), guard))
    # Cover repeated per-call groups/events/dictionary setup beyond full-row projections as explicit slack.
    for i, row in enumerate(fixed_rows):
        row.update(patient_id=f'p{i % 121}', group_id=f'g{i % 121}', record_id=f'r{i}',
                   timestamp=0., duration_seconds=30.)
    fixed_calls = 540 if count == 629444 else 3
    before = time.monotonic()
    for _ in range(fixed_calls):
        guard()
        evaluation.evaluate_stream(fixed_rows, events, config, bootstrap_samples=0)
    fixed = dict(seconds=time.monotonic() - before, calls=fixed_calls,
                 rows_per_call=len(fixed_rows), events_per_call=len(events), scope='fixed-call slack')
    guard()
    return dict(evaluations=results, source_sha256=hashlib.sha256(
        inspect.getsource(evaluation.evaluate_stream).encode()).hexdigest(),
        fixed_call_controls=fixed,
        numerical_substitution=False, fits=0, selectors=0, full_run_eligible=False,
        applicability='eight complete generated evaluation kernels; postmeasurement source/count/branch acceptance required')
