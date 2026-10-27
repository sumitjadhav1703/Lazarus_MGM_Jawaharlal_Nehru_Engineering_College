import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('leaf_measurement', Path(__file__).with_name('leaf_measurement.py'))
leaf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(leaf)


values = []

def guard():
    pass


def example():
    total = 0
    for value in values:
        guard()
        total += value


class Checks(unittest.TestCase):
    def test_bound_block_keeps_guard_and_statement(self):
        calls = []
        stats, result = leaf.run_block(example, lambda node, text: isinstance(node, leaf.ast.For),
                                      dict(total=0, values=[1, 2, 3], guard=lambda: calls.append(1)))
        self.assertEqual(result['total'], 6)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(stats['source_block_sha256']), 64)

    def test_ambiguous_or_missing_block_fails(self):
        with self.assertRaises(ValueError):
            leaf.source_block(example, lambda node, text: False)

    def test_native_exact_rows_and_reject_duplicate_permutation(self):
        class Compare:
            BATCH_ROWS = 2
            @staticmethod
            def _timing_matrix(rows):
                return type('Matrix', (), {'shape': (len(rows), 46)})()
        result = leaf.native_matrix_pass([{}, {}, {}], count=3, indices=range(3), guard=lambda: None, compare=Compare)
        self.assertEqual((result['rows'], result['batches']), (3, 2))
        with self.assertRaises(ValueError):
            leaf.native_matrix_pass([{}, {}, {}], count=3, indices=[0, 0, 2], guard=lambda: None, compare=Compare)




class ProductionBlocks(unittest.TestCase):
    def test_source_bound_metadata_and_both_support_branches(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        import numpy as np

        from cardiotwin import ecgfounder_study as study
        roles = [dict(fold=f, **{name: ['g'] if name == 'fit' else [] for name in study.ROLE_NAMES})
                 for f in range(3)]
        rows = [dict(row_index=i, record_id='r', group_id='g', patient_id='p',
                     timestamp=float(i * 30), accepted=True, label=i % 2) for i in range(2)]
        metadata = np.zeros(2, dtype=[('reason', 'i2'), ('label_reason', 'i2'), ('accepted', 'i1'), ('label', 'i1')])
        metadata['accepted'] = 1
        compact = dict(categories=dict(reason=['ok'], label_reason=['ok']))
        result = leaf.metadata_leaves(rows=rows, compact_rows=SimpleNamespace(metadata=metadata),
                                     compact=compact, index=dict(allowed_groups=['g']), roles=roles,
                                     guard=lambda: None, np=np, study=study)
        self.assertEqual(len(result['prepare']), 2)
        self.assertEqual(len(result['support_class_scan']), 3)
        support = dict(folds=[dict(fold=f, roles={name: dict(observable_events=1, positive_people=1)
                                                for name in study.ROLE_NAMES}) for f in range(3)])
        pair = study.EventPair('g', 'p', 'r', 3600.0, 1, 0)
        pairs = {f: {name: [pair] if name == 'fit' else [] for name in study.ROLE_NAMES} for f in range(3)}
        def saved_pairs(*args):
            return pairs
        with patch.object(study, '_saved_pairs', saved_pairs):
            measured = leaf.fixed_pair_leaves(payload={}, root=None, rows=rows, roles=roles, support=support,
                                             membership={}, events={}, guard=lambda: None, np=np, study=study)
        self.assertEqual(measured['fixed_pair_count'], 3)
        self.assertEqual([r['synthetic_all_available'] for r in measured['filtering']], [True, False])
        self.assertFalse(measured['real_support_authority'])


class ReportComposition(unittest.TestCase):
    def test_eight_report_warning_and_missed_detail_ceiling(self):
        layout = leaf.report_layout([195736, 196655, 237053], [45, 43, 44])
        self.assertEqual(layout['warning_upper_per_fold_policy'], [9832, 9876, 11897])
        self.assertEqual(len(layout['arm_receipts']), 6)
        self.assertEqual(layout['arm_receipt_warning_details'], 632100)
        self.assertEqual(layout['all_eight_warning_details'], 1896300)
        self.assertEqual(layout['all_eight_missed_event_details'], 14460)
        with self.assertRaises(ValueError):
            leaf.report_layout([195736, 196655, 237052], [45, 43, 44])

    def test_six_complete_arm_copies_and_shared_cap(self):
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        calls = []
        def report_work(output, **kwargs):
            calls.append(kwargs)
            return {'sample': {'warnings': 316050, 'missed_events': 2410}}
        primitives = SimpleNamespace(report_work=report_work)
        with TemporaryDirectory() as root:
            result = leaf.full_report_leaves(Path(root) / 'reports', guard=lambda: None,
                storage_check=lambda: None, primitives=primitives,
                study=SimpleNamespace(_artifact_bytes=lambda p: 1))
            self.assertEqual(result['persisted_warnings'], 1896300)
            self.assertEqual(len(calls), 6)
            self.assertFalse(result['report_layout_complete'])
            with self.assertRaises(RuntimeError):
                leaf.full_report_leaves(Path(root) / 'over-cap', guard=lambda: None,
                    storage_check=lambda: None, primitives=primitives,
                    study=SimpleNamespace(_artifact_bytes=lambda p: 250000001))


if __name__ == '__main__':
    unittest.main()
