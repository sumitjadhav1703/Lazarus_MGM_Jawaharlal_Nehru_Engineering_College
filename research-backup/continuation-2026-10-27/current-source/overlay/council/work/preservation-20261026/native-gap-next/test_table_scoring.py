"""Tiny generated checks; never execute a full table locally."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest
import table_scoring
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def test_tiny_all_kinds_orders_branches_no_fit():
    def forbidden(*args, **kwargs):
        raise AssertionError('No fit permitted in generated numeric states')

    with TemporaryDirectory() as temp:
        with patch.object(StandardScaler, 'fit', forbidden), patch.object(SimpleImputer, 'fit', forbidden), \
                patch.object(LogisticRegression, 'fit', forbidden):
            result = table_scoring.measure(Path(temp) / 'tables', count=7,
                                           guard=lambda: None, storage_check=lambda: None)
        assert len(result['passes']) == 6
        assert all(v['rows'] == 7 for v in result['passes'].values())
        assert result['fits'] == 0
        assert result['passes']['chronological_candidate']['checksum'] == pytest.approx(
            result['passes']['scattered_candidate']['checksum'])
        assert len(result['generated_branches']) == 6
        delta = np.load(Path(temp) / 'tables/delta.npy')
        assert delta.shape == (7, 1024) and delta.dtype == np.float32
        assert np.std(delta) > 0


def test_partial_working_set_rejected():
    with pytest.raises(ValueError, match='Complete'):
        table_scoring.measure('/never-created', count=18, guard=lambda: None, storage_check=lambda: None)
