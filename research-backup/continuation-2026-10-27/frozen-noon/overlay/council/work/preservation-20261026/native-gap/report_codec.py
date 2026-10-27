"""Generated eight-file report codec fixture; cost evidence only, never forecast output."""

import copy
import hashlib
import inspect
import json
import time
from pathlib import Path

ENTRY_FIXED = frozenset(('identity', 'calibration', 'policies', 'model', 'policy_freeze',
                         'predictions', 'predictions_sha256', 'outer_rows', 'availability',
                         'outer_events', 'outer_groups', 'primary_constraints'))
SUMMARY_FIXED = frozenset(('identity', 'comparisons', 'diagnostics', 'rows', 'people', 'events',
                           'availability', 'timing_comparator_availability',
                           'coverage_unchanged_between_matched_arms', 'decision',
                           'failed_advancement_criteria', 'promising_development_lead',
                           'supported_development_improvement', 'fold_sensitivity_deltas',
                           'direction_survives_person_and_preset_checks',
                           'independent_replay_required', 'clinical_claim', 'confirmation'))


def wide_numeric_fixture(value, salt=0):
    """Serialized-width fixture only: finite floats including nullable numeric branches."""
    if isinstance(value, dict):
        return {key: wide_numeric_fixture(item, salt + i) for i, (key, item) in enumerate(value.items())}
    if isinstance(value, list):
        return [wide_numeric_fixture(item, salt + i) for i, item in enumerate(value)]
    if value is None or type(value) is float:
        return -(1.1234567890123456 + salt % 31 / 997) * 1e100
    if type(value) is int:
        return 999_999_999
    return value


def policy_report_template(config, evaluation, person_rows):
    """Obtain the actual finite-policy schema and detail types from production code."""
    rows = [dict(patient_id=patient, group_id=patient, record_id=record,
                 timestamp=float(i * 30), duration_seconds=180.0, accepted=True,
                 risk=risk, label=i % 2, label_reason=None, reason=None)
            for patient, record, risk in [('p0', 'r0', 0.99), ('p1', 'r1', 0.01)] for i in range(5)]
    report = evaluation.evaluate_stream(rows, [dict(patient_id='p1', record_id='r1', onset=300.0)],
                                        config, bootstrap_samples=0)
    report['per_group'] = person_rows(report, rows)
    if not report['warning_details'] or not report['missed_event_details'] or not report['probability']:
        raise ValueError('Generated production report lacks detail/probability schema')
    return report


def bounded_policy_report(template, *, warnings, missed, people, fold, variant, person_rows,
                          reason_categories=(), label_reason_categories=()):
    """Retain every production field and type while growing source-bounded detail lists."""
    report = wide_numeric_fixture(copy.deepcopy(template), variant)
    if not all(type(n) is int and n >= 0 for n in (warnings, missed, people)):
        raise ValueError('Nonnegative integral report cardinalities required')
    reason_categories = tuple(dict.fromkeys((*reason_categories, 'forecast_unavailable')))
    label_reason_categories = tuple(dict.fromkeys((*label_reason_categories, 'unlabeled', 'unknown')))
    bin_template = report['probability']['calibration'][0]
    report['probability']['calibration'] = [dict(bin_template, lower=i / 10, upper=(i + 1) / 10,
                                                count=fold * 100_000 + i + 1,
                                                mean_probability=(i + 0.5) / 10,
                                                event_rate=(i + 1) / 11)
                                            for i in range(10)]
    patient_template = report['per_patient'][0]
    for key in ('exclusion_hours', 'rejection_hours'):
        categories = reason_categories if key == 'rejection_hours' else label_reason_categories
        report[key] = {name: float(600 + i / 997)
                       for i, name in enumerate(dict.fromkeys(('missing_cadence', *categories)))}
    report['unassessable_warning_reasons'] = {name: warnings for name in label_reason_categories}
    report['known_excluded_warning_reasons'] = {name: warnings for name in label_reason_categories
                                                if name in ('current_AF', 'washout', 'gap', 'ineligible_event')}
    report['unknown_outcome_warning_reasons'] = {name: warnings for name in label_reason_categories
                                                 if name not in ('current_AF', 'washout', 'gap', 'ineligible_event')}
    if 'burden_seconds_by_label' in report:
        report['burden_seconds_by_label'] = {name: float(600 + i / 997)
                                             for i, name in enumerate(('eligible_negative', 'eligible_positive',
                                                                       *label_reason_categories))}
        report.update({f'burden_{name}_seconds': float(999999999 + i / 997)
                       for i, name in enumerate(report['burden_seconds_by_label'])})
    report['per_patient'] = []
    for i in range(people):
        person = copy.deepcopy(patient_template)
        person['patient_id'] = f'patient-{fold:02d}-{i:021d}'
        person['eligible_seconds'] = float(600 + 30 * i + variant / 997)
        if 'burden_seconds_by_label' in report:
            person.update({f'burden_{name}_seconds': float(600 + j / 997)
                           for j, name in enumerate(report['burden_seconds_by_label'])})
        report['per_patient'].append(person)
    report['per_group'] = person_rows(report, [dict(patient_id=person['patient_id'],
                                                   group_id=person['patient_id'])
                                                   for person in report['per_patient']])
    warning_template = template['warning_details'][0]
    missed_template = template['missed_event_details'][0]
    report['warning_details'] = []
    for i in range(warnings):
        detail = dict(warning_template)
        timestamp = float(i * 600 + (variant * 31 + i * 17) % 997 / 997)
        detail.update(timestamp=timestamp, end_time=timestamp + 30 + (i % 19) / 23,
                      risk=float(((i * 9973 + variant * 37) % 1_000_003 + 1) / 1_000_004),
                      patient_id=f'patient-{fold:02d}-{i % max(people, 1):021d}',
                      record_id=f'record-{i % 132:09d}',
                      outcome=('matched', 'false', 'unassessable')[i % 3],
                      assessment_category=('assessable', 'known_excluded', 'unknown_outcome')[i % 3],
                      annotation_exclusion=(None, 'current_AF', 'unknown')[i % 3])
        report['warning_details'].append(detail)
    report['missed_event_details'] = []
    for i in range(missed):
        detail = dict(missed_template)
        detail.update(patient_id=f'patient-{fold:02d}-{i % max(people, 1):021d}',
                      record_id=f'record-{i % 132:09d}', onset=float(i * 600 + variant / 997),
                      all_warning_opportunities_unavailable=bool(i % 2))
        report['missed_event_details'].append(detail)
    report['lead_seconds'] = [float(120 + 30 * (i % 17)) for i in range(missed)]
    report['patients'], report['records'], report['warnings'], report['missed_events'] = (
        people, people, warnings, missed)
    return report


def _full_identity_envelope(identity):
    """Fill the declared JSON allowance using source-shaped generated file bindings."""
    identity = copy.deepcopy(identity)
    files = identity['files_sha256']
    while True:
        used = len(json.dumps(identity, sort_keys=True, allow_nan=False).encode())
        remaining = 131_072 - used
        if remaining == 0:
            return identity
        number = len(files)
        stem = f'generated-authority/member-{number:05d}-'
        files[stem] = 'a' * 64
        base_growth = len(json.dumps(identity, sort_keys=True, allow_nan=False).encode()) - used
        del files[stem]
        if remaining < base_growth:
            raise ValueError('Identity envelope member cannot fit')
        extra = 80 if remaining > base_growth + 200 else remaining - base_growth
        padding = ''.join(hashlib.sha256(f'authority:{number}:{i}'.encode()).hexdigest()
                          for i in range((extra + 63) // 64))[:extra]
        files[stem + padding] = hashlib.sha256(f'authority-digest:{number}'.encode()).hexdigest()


def write_eight_reports(output, *, config, fold_warning_bounds, fold_rows, fold_people, fold_events, fixed,
                        reason_categories, label_reason_categories, guard, storage_check, study, evaluation):
    """Use production gzip/write/read paths on the six receipts and their two embedded copies."""
    if any(len(values) != 3 or any(type(n) is not int or n < 0 for n in values)
           for values in (fold_warning_bounds, fold_rows, fold_people, fold_events)):
        raise ValueError('Three nonnegative integral fold cardinality arrays required')
    if (sum(fold_warning_bounds), sum(fold_rows), sum(fold_events)) != (31605, 629444, 241):
        raise ValueError('Complete original fold warning/event ceilings required')
    if sum(fold_people) != 121:
        raise ValueError('Complete original people partition required')
    if set(fixed) != {'entry', 'summary'} or set(fixed['entry']) != ENTRY_FIXED or set(fixed['summary']) != SUMMARY_FIXED:
        raise ValueError('Exact source enumerated fixed report fields required')
    if fixed['summary']['identity'] != fixed['entry']['identity']:
        raise ValueError('Prospective shared identity authority envelope differs')
    fixed = copy.deepcopy(fixed)
    identity = _full_identity_envelope(fixed['entry']['identity'])
    fixed['entry']['identity'] = fixed['summary']['identity'] = identity
    for diagnostic in fixed['summary']['diagnostics']:
        diagnostic['identity'] = identity
    fixed_bytes = len(json.dumps(fixed, sort_keys=True, allow_nan=False).encode())
    identity_bytes = len(json.dumps(identity, sort_keys=True, allow_nan=False).encode())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    template = policy_report_template(config, evaluation, study._person_rows)
    entries = []

    def write(name, value):
        guard()
        if time.monotonic() - started >= 1200:
            raise TimeoutError('Complete report codec component ceiling reached')
        artifact = study._report_artifact(output, name, value, 250_000_000, storage_check)
        if study._read_report(output / name) != value:
            raise ValueError('Production report codec round-trip differs')
        return artifact

    for fold in range(3):
        for arm in study.STUDY_ARMS:
            guard()
            reports = {name: bounded_policy_report(template, warnings=fold_warning_bounds[fold],
                       missed=fold_events[fold], people=fold_people[fold], fold=fold,
                       variant=fold * 20 + (0 if arm == study.STUDY_ARMS[0] else 10) + policy,
                       person_rows=study._person_rows, reason_categories=reason_categories,
                       label_reason_categories=label_reason_categories)
                       for policy, name in enumerate(study.POLICY_NAMES)}
            entry = dict(fixed['entry'], status='complete', fold=fold, arm=arm, reports=reports,
                         outer_rows=fold_rows[fold], outer_events=fold_events[fold],
                         outer_groups=[f'patient-{fold:02d}-{i:021d}' for i in range(fold_people[fold])],
                         availability=dict(original_accepted_rows=fold_rows[fold],
                                           common_available_rows=fold_rows[fold], newly_unavailable_rows=0))
            prefix = f'fold-{fold}-{arm}'
            entry['model'] = dict(entry['model'], path=prefix + '-model.joblib',
                                  numeric_manifest=prefix + '-model.joblib.json')
            entry['policy_freeze'] = dict(entry['policy_freeze'], path=prefix + '-policy-freeze.json')
            entry['predictions'] = prefix + '-predictions.jsonl.gz'
            write(f'fold-{fold}-{arm}-receipt.json.gz', entry)
            entries.append(entry)
    summary = dict(fixed['summary'], status='complete', folds=entries)
    write('summary.json.gz', summary)
    write('review.json.gz', dict(summary))
    warning_total = sum(len(report['warning_details']) for entry in entries for report in entry['reports'].values())
    missed_total = sum(len(report['missed_event_details']) for entry in entries for report in entry['reports'].values())
    if 3 * warning_total != 1_896_300 or 3 * missed_total != 14_460:
        raise ValueError('Eight-file duplicated detail ceiling differs')
    return dict(files=8, arm_warning_details=warning_total, all_warning_details=3 * warning_total,
                all_missed_details=3 * missed_total, artifact_bytes=study._artifact_bytes(output),
                source_sha256={name: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for name, fn in (
                    ('evaluate_stream', evaluation.evaluate_stream), ('person_rows', study._person_rows),
                    ('report_artifact', study._report_artifact), ('read_report', study._read_report),
                    ('execute_study', study.execute_study), ('replay_study', study.replay_study))},
                fixed_json_bytes=fixed_bytes,
                shared_identity_json_bytes=identity_bytes,
                serialized_identity_copies=26,
                numeric_fixture='finite serialized-width stress fields; generated values are not metrics',
                entropy='deterministic varied digests/IDs/scalars; measured compression needs post-run applicability review',
                outcome_fits=0, full_run_eligible=False)
