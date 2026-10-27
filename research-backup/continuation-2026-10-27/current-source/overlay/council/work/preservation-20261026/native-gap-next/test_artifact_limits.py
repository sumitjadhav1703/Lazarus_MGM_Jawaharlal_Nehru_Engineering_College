from types import SimpleNamespace

import pytest
from artifact_limits import check_tree, remaining_subtree_cap


def test_shared_tree_budget_includes_siblings_and_rejects_outside(tmp_path):
    root, child = tmp_path/'all', tmp_path/'all/reports'
    sizes = {root:249_999_999,child:1}
    study=SimpleNamespace(_artifact_bytes=lambda path:sizes[path])
    assert remaining_subtree_cap(child,root,study)==2
    sizes[root]=250_000_001
    with pytest.raises(RuntimeError,match='Composite'):
        check_tree(root,study)
    with pytest.raises(ValueError,match='inside'):
        remaining_subtree_cap(tmp_path/'other',root,study)
