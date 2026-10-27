"""Small checks for generated fixed report fields and source-bound controls."""

import importlib.util
import json
import unittest
from pathlib import Path

from cardiotwin import ecgfounder_study as study

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location("fixed_metadata", HERE / "fixed_metadata.py")
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)
codec_spec = importlib.util.spec_from_file_location("report_codec", HERE / "report_codec.py")
codec = importlib.util.module_from_spec(codec_spec)
codec_spec.loader.exec_module(codec)


class FixedMetadataChecks(unittest.TestCase):
    def test_complete_generated_fixed_shapes(self):
        config = json.loads((ROOT / "configs/precursor-cycle.json").read_text())
        result = fixed.generated_fixed(config=config, fold_people=[40, 40, 41],
                                       fold_events=[80, 80, 81], study=study, guard=lambda: None)
        self.assertEqual(set(result["entry"]), codec.ENTRY_FIXED)
        self.assertEqual(set(result["summary"]), codec.SUMMARY_FIXED)
        self.assertEqual(result["entry"]["identity"], result["summary"]["identity"])
        self.assertLessEqual(len(json.dumps(result["entry"]["identity"], sort_keys=True).encode()), 131_072)
        self.assertEqual(set(result["entry"]["policies"]), set(study.POLICY_NAMES))
        self.assertEqual(len(result["summary"]["comparisons"]), 10)
        self.assertEqual(len(result["summary"]["diagnostics"]), 3)
        self.assertTrue(all(len(diag["nulls"]) == 19 for diag in result["summary"]["diagnostics"]))
        self.assertEqual(
            [diag["roles"][role][study.STUDY_ARMS[0]]["negative_drift"]["pairs"]
             for diag in result["summary"]["diagnostics"] for role in ("fit", "stop")],
            [2453, 332, 2067, 331, 2128, 222],
        )
        self.assertTrue(all(diag["roles"][role][study.STUDY_ARMS[0]]["event_pairs"]["pairs"] == 241
                            for diag in result["summary"]["diagnostics"] for role in ("fit", "stop")))
        self.assertTrue(all(item["event_denominator"] == 241 and item["people"] == 121
                            and item["samples"] == 10_000
                            for item in result["summary"]["comparisons"].values()))
        self.assertFalse(result["summary"]["supported_development_improvement"])
        self.assertEqual(result["summary"]["failed_advancement_criteria"],
                         ["nonpositive_pooled_sensitivity_delta", "fewer_than_two_improved_folds"])
        self.assertIn("generated_measurement_only", result["entry"]["identity"]["compact_binding"])

    def test_partial_fold_counts_and_excess_pair_count_reject(self):
        config = json.loads((ROOT / "configs/precursor-cycle.json").read_text())
        with self.assertRaisesRegex(ValueError, "cardinalities"):
            fixed.generated_fixed(config=config, fold_people=[40, 40, 40],
                                  fold_events=[80, 80, 81], study=study, guard=lambda: None)
        with self.assertRaisesRegex(ValueError, "ceiling"):
            fixed.measure_control_replay_leaves(study=study, guard=lambda: None, pair_count=724)

    def test_control_pair_arithmetic_uses_production_source(self):
        result = fixed.measure_control_replay_leaves(study=study, guard=lambda: None, pair_count=3)
        self.assertEqual((result["calls"], result["pairs_per_call"], result["fits"]), (63, 3, 0))
        self.assertEqual(result["identity_calls"], 12)
        self.assertEqual(result["source_sha256"], fixed._digest(study._pair_diagnostics))
        self.assertEqual(result["identity_source_sha256"], fixed._digest(study._identity_variance))


if __name__ == "__main__":
    unittest.main()
