import copy
import json
from pathlib import Path

import frozen_replay
import policy_codec
import pytest
import resource_reservation
from test_metadata_geometry import tiny_shape

from cardiotwin import ecgfounder_study as study
from cardiotwin import evaluation


def test_tiny_complete_policy_codec_and_schema_rejection(tmp_path):
    config=json.loads((Path(__file__).resolve().parents[4]/'configs/precursor-cycle.json').read_text())
    result=frozen_replay.measure(count=7,config=config,study=study,evaluation=evaluation,guard=lambda:None,
        policy_output=tmp_path/'policy',shape=tiny_shape(),storage_check=lambda:dict(allocated_bytes=0,logical_bytes=0),
        fixed=dict(identity={'generated':True},model={'path':'generated','sha256':'a'*64,'bytes':1},calibration={}))
    assert result['policy_authority']['files']==7
    assert all(v['json_bytes']<=v['json_bound'] for v in result['policy_authority']['freezes'])
    freeze=json.loads(next((tmp_path/'policy').glob('fold-*-policy-freeze.json')).read_text())
    bad=copy.deepcopy(freeze)
    bad['searches'][study.POLICY_NAMES[0]]*=2
    with pytest.raises(ValueError,match='400'):
        policy_codec.validate(bad,study=study,calibration_rows=7,policy_rows=10)
    bad=copy.deepcopy(freeze)
    bad['extra']=True
    with pytest.raises(ValueError,match='schema'):
        policy_codec.validate(bad,study=study,calibration_rows=7,policy_rows=10)
    reservation=resource_reservation.reservation(warning={},missed={},person={})
    assert reservation['admissible'] is True
    assert reservation['measured'] is False
    with pytest.raises(ValueError,match='exceeds'):
        resource_reservation.reservation(warning={'huge':'a'*5000},missed={},person={})
