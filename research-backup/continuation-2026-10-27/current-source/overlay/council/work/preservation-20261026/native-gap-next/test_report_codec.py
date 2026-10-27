import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cardiotwin import ecgfounder_study as study
from cardiotwin import evaluation

spec = importlib.util.spec_from_file_location('report_codec', Path(__file__).with_name('report_codec.py'))
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)
ROOT = Path(__file__).resolve().parents[4]


class ReportCodecChecks(unittest.TestCase):
    def test_missing_source_fixed_metadata_rejects_before_output(self):
        with TemporaryDirectory() as temp:
            output = Path(temp) / 'reports'
            with self.assertRaisesRegex(ValueError, 'Exact source enumerated'):
                codec.write_eight_reports(output, config={}, fold_warning_bounds=[9832, 9876, 11897],
                    fold_rows=[195736, 196655, 237053],
                    fold_people=[40, 40, 41], fold_events=[80, 80, 81], fixed={'entry': {}, 'summary': {}},
                    reason_categories=['ok'], label_reason_categories=['ok'],
                    guard=lambda: None, storage_check=lambda: None, study=study, evaluation=evaluation)
            self.assertFalse(output.exists())

    def test_production_schema_and_generated_detail_cardinalities(self):
        config = json.loads((ROOT / 'configs/precursor-cycle.json').read_text())
        template = codec.policy_report_template(config, evaluation, study._person_rows)
        report = codec.bounded_policy_report(template, warnings=7, missed=3, people=4, fold=1, variant=3,
                                             person_rows=study._person_rows,
                                             reason_categories=['forecast_unavailable', 'low_quality'],
                                             label_reason_categories=['current_AF', 'unknown'])
        self.assertEqual(set(report), set(template) | {'burden_current_AF_seconds',
                                                     'burden_unlabeled_seconds', 'burden_unknown_seconds'})
        self.assertEqual((len(report['warning_details']), len(report['missed_event_details'])), (7, 3))
        self.assertEqual((len(report['per_patient']), len(report['per_group'])), (4, 4))
        self.assertEqual(set(report['warning_details'][0]), set(template['warning_details'][0]))
        self.assertEqual(set(report['missed_event_details'][0]), set(template['missed_event_details'][0]))
        self.assertTrue(set(template['per_group'][0]) <= set(report['per_group'][0]))
        self.assertTrue(all(f'burden_{category}_seconds' in report
                            and f'burden_{category}_seconds' in report['per_patient'][0]
                            for category in ('current_AF', 'unlabeled', 'unknown')))
        self.assertTrue(all(value is not None for value in report['lead_time_summary'].values()))
        self.assertEqual(len(report['warning_details'][0]['patient_id']), 32)
        self.assertEqual(len(report['warning_details'][0]['record_id']), 16)
        self.assertEqual({v['assessment_category'] for v in report['warning_details']},
                         {'assessable', 'known_excluded', 'unknown_outcome'})
        self.assertEqual(len(report['probability']['calibration']), 10)
        self.assertEqual(set(report['known_excluded_warning_reasons']), {'current_AF'})
        self.assertEqual(set(report['unknown_outcome_warning_reasons']), {'unknown', 'unlabeled'})
        self.assertEqual(set(report['rejection_hours']),
                         {'missing_cadence', 'forecast_unavailable', 'low_quality'})
        with self.assertRaises(ValueError):
            codec.bounded_policy_report(template, warnings=-1, missed=3, people=4, fold=1, variant=3,
                                        person_rows=study._person_rows)

    def test_identity_envelope_is_exact_and_does_not_mutate_input(self):
        original = {'files_sha256': {'source': 'a' * 64}}
        filled = codec._full_identity_envelope(original)
        self.assertEqual(len(json.dumps(filled, sort_keys=True, allow_nan=False).encode()), 131072)
        self.assertEqual(len(original['files_sha256']), 1)


if __name__ == '__main__':
    unittest.main()
