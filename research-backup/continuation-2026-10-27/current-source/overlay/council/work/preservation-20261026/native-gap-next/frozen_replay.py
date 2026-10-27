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


def measure(*, count=629444, config, study, evaluation, guard,
            policy_output=None, shape=None, fixed=None, storage_check=None, artifact_root=None):
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
    result = dict(reconstruction_seconds=time.monotonic() - before,
                engineering_freeze_construction_seconds=setup_seconds,
                numerical_evaluation_substituted=True, numerical_evaluation_calls=calls[0],
                numerical_evaluation_rows=calls[1], thresholds=len(thresholds), policies=10,
                numerical_owner='same-unit numerical_evaluations; old22H supplies counts only, no isolated rate',
                reconstruction_owner='artifact_verification_download; six policy streams target',
                source_sha256=hashlib.sha256(inspect.getsource(study._replay_policies).encode()).hexdigest(),
                fits=0, selectors=0, full_replay_latency=False, full_run_eligible=False)
    if policy_output is not None:
        import policy_codec
        result['policy_authority']=policy_codec.measure(policy_output,template=freeze,fixed=fixed,
            shape=shape,study=study,guard=guard,storage_check=storage_check,artifact_root=artifact_root)
    return result


def numerical_evaluations(*, count=629444, shape=None, config, evaluation, guard):
    """Exact per-role generated geometry, including all missed-event rescans and matching."""
    import bisect

    import metadata_geometry
    if shape is None:
        if not 2 <= count <= 17:
            raise ValueError('Pinned generated metadata shape required for complete measurement')
        shape = dict(records=[dict(record_id='r0', patient_id='p0', group_id='g0', rows=count, events=2)],
            roles=[dict(fold=f, policy=['g0'], outer=['g0']) for f in range(3)])
    metadata_geometry.validate_shape(shape)

    class PatternRows:
        def __init__(self, records, pattern):
            self.records, self.pattern = records, pattern
            self.ends = np.cumsum([record['rows'] for record in records]).tolist()

        def __len__(self):
            return self.ends[-1]

        def __getitem__(self, index):
            if not 0 <= index < len(self):
                raise IndexError(index)
            if index % 256 == 0:
                guard()
            record_index = bisect.bisect_right(self.ends, index)
            record = self.records[record_index]
            local = index - (self.ends[record_index - 1] if record_index else 0)
            high = local % 20 < 3
            unknown = self.pattern == 'unknown_warnings' and high
            risk = .91 + ((local * 9973) % 100003) / 2_000_060 if high else .01 + ((local * 9973) % 100003) / 2_000_060
            return dict(patient_id=record['patient_id'], group_id=record['group_id'], record_id=record['record_id'],
                timestamp=float(local * 30), duration_seconds=float(record['rows'] * 30),
                accepted=high or local % 97 != 0, risk=risk,
                label=None if unknown else local % 2,
                label_reason='unknown' if unknown else None,
                reason='forecast_unavailable' if not high and local % 97 == 0 else None)

    results = []
    for fold in range(3):
        for role, probability in (('policy', False), ('outer', True)):
            records = metadata_geometry.scope_records(shape, fold, role)
            if not records:
                raise ValueError('Each original role requires nonempty generated record geometry')
            # Events beyond every record force all source-supported n_record*events_record rescans.
            events = [dict(patient_id=record['patient_id'], record_id=record['record_id'],
                           onset=float(record['rows'] * 30 + 1200 + 600 * i))
                      for record in records for i in range(record['events'])]
            weighted = sum(record['rows'] * record['events'] for record in records)
            for pattern in ('known_warnings', 'unknown_warnings'):
                for legacy in (False, True):
                    guard()
                    policy = copy.deepcopy(config)
                    policy['alert'].update(threshold=.5, release_threshold=.25)
                    if legacy:
                        for key in ('policy_version', 'max_duration_seconds', 'rearm_seconds',
                                    'max_burden_occupancy', 'max_eligible_negative_burden_occupancy'):
                            policy['alert'].pop(key, None)
                    rows = PatternRows(records, pattern)
                    before = time.monotonic()
                    report = evaluation.evaluate_stream(rows, events, policy,
                        bootstrap_samples=0, include_probability=probability)
                    guard()
                    if len(report['missed_event_details']) != len(events):
                        raise ValueError('Generated all-missed event scan branch differs')
                    results.append(dict(fold=fold, role=role, pattern=pattern, legacy=legacy,
                        include_probability=probability, seconds=time.monotonic() - before,
                        rows=len(rows), records=len(records), events=len(events),
                        all_missed_row_event_visits=weighted, warnings=len(report['warning_details']),
                        unknown_warnings=report['unknown_outcome_warnings'] if not legacy else sum(
                            d['assessment_category']=='unknown_outcome' for d in report['warning_details']),
                        groups=len(report['per_patient'])))
    return dict(evaluations=results, source_sha256=hashlib.sha256(
        inspect.getsource(evaluation.evaluate_stream).encode()).hexdigest(),
        numerical_substitution=False, fits=0, selectors=0, full_run_eligible=False,
        aggregate_geometry=metadata_geometry.public_summary(shape),
        projection=dict(search='perfold2arms*40thresholds*(slowerfinite_search+slowerlegacy_search)',
                        outer='perfold2arms*(9*slowerfinite_probability+slowerlegacy_probability)',
                        calls=dict(search=480, outer=60), fixed_call_controls=0,
                        ownership='artifact_verification_download; per-role measured envelope only, no row ratio'),
        applicability='24 exact-role generated kernels; metadata shape details remain private; no real onsets/labels')
