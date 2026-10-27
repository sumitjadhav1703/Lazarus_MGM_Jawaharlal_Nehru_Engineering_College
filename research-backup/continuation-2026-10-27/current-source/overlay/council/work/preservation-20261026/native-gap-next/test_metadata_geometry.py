import metadata_geometry as geometry
import pytest

from cardiotwin import ecgfounder_study as study


def tiny_shape():
    return dict(records=[dict(record_id='r0',patient_id='p0',group_id='g0',rows=7,events=2),
                         dict(record_id='r1',patient_id='p1',group_id='g1',rows=10,events=1)],
                roles=[dict(fold=i,fit=['g0'],stop=['g1'],calibration=['g0'],policy=['g1'],outer=['g0','g1'])
                       for i in range(3)])


def test_exact_view_chain_role_calls_and_generated_drift_geometry():
    shape=tiny_shape()
    result=geometry.measure(shape,study=study,guard=lambda:None)
    assert len(result['role_index_calls'])==48 and len(result['drift_scan_calls'])==12
    assert [r['pairs'] for r in result['drift_scan_calls']]==[0,0]*6
    assert result['aggregate']['all_missed_row_event_visits']==24
    assert result['real_pairs']==0
    prepared=geometry.generated_prepared(shape,study,lambda:None)
    assert len(prepared.rows[0]['features'])==0
    prepared.generated_compact.metadata['timestamp'][0:2]=[1260.,3600.]
    pairs=study.negative_drift_pairs(study._CommonRows(prepared),{'g0'})
    assert len(pairs)==1 and pairs[0].positive_index==1 and pairs[0].negative_index==0


def test_derive_shape_discards_original_ids_onsets_and_rejects_missing_events():
    index=dict(allowed_groups=['secret-group'],group_ids={'secret-patient':'secret-group'},row_count=7,
       records=[dict(record=dict(patient_id='secret-patient',record_id='secret-record'),rows=7)],
       events=[dict(patient_id='secret-patient',record_id='secret-record',onset=1234.)])
    roles=[dict(fold=i, fit=['secret-group'], stop=[], calibration=[], policy=[], outer=[]) for i in range(3)]
    shape=geometry.geometry(index,roles)
    assert 'secret' not in str(shape) and 'onset' not in str(shape)
    assert shape['records'][0]['events']==1
    index['events'][0]['record_id']='wrong'
    with pytest.raises(ValueError,match='membership'):
        geometry.geometry(index,roles)


def test_duplicate_roles_records_and_changed_events_reject():
    import copy
    shape=tiny_shape()
    altered=copy.deepcopy(shape)
    altered['roles'][0]['fit'] *= 2
    with pytest.raises(ValueError,match='Unique known'):
        geometry.validate_shape(altered)
    altered=copy.deepcopy(shape)
    altered['records'][1]['record_id']='r0'
    with pytest.raises(ValueError,match='Unique generated'):
        geometry.validate_shape(altered)
    outer=[dict(rows=17,records=2,people=2,events=3) for _ in range(3)]
    roles=[dict(rows={name:sum(r['rows'] for r in geometry.scope_records(shape,f,name))
           for name in ('fit','stop','calibration','policy','outer')}) for f in range(3)]
    totals=dict(rows=17,records=2,people=2,events=3)
    geometry.validate_expected(shape,totals,outer,roles)
    altered=copy.deepcopy(shape)
    altered['records'][0]['events']+=1
    with pytest.raises(ValueError,match='aggregate'):
        geometry.validate_expected(altered,totals,outer,roles)


def test_generated_mmap_chain_is_tiny_and_closes(tmp_path):
    result=geometry.measure(tiny_shape(),study=study,guard=lambda:None,output=tmp_path/'generated')
    assert len(result['role_index_calls'])==48
    assert result['aggregate']['rows']==17
