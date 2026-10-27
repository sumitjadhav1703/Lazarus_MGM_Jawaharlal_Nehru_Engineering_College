"""Six complete generated policy-freeze schemas; no selector or outcome evaluation."""
import copy
import hashlib
import inspect
import json
import time
from pathlib import Path

from artifact_limits import remaining_subtree_cap

FIELDS = frozenset(('identity', 'fold', 'arm', 'roles', 'model', 'calibration', 'policies',
                    'searches', 'frozen_before_any_outer_scoring', 'calibration_indices', 'policy_indices'))
FIXED_FIELDS_BYTES = 2_000_000


def validate(freeze, *, study, calibration_rows, policy_rows):
    if set(freeze) != FIELDS or freeze['frozen_before_any_outer_scoring'] is not True:
        raise ValueError('Exact complete policy-freeze schema required')
    if set(freeze['policies']) != set(study.POLICY_NAMES) or set(freeze['searches']) != set(study.POLICY_NAMES):
        raise ValueError('Exact ten policy/search members required')
    if any(not 1 <= len(entries) <= 40 for entries in freeze['searches'].values()):
        raise ValueError('At most400 complete search entries required per policy freeze')
    for name, count in (('calibration_indices', calibration_rows), ('policy_indices', policy_rows)):
        indices = freeze[name]
        if len(indices) != count or any(type(i) is not int or not 0 <= i < 629444 for i in indices):
            raise ValueError('Bounded policy-freeze index schema differs')
        if indices != sorted(set(indices)):
            raise ValueError('Unique sorted policy-freeze indices required')
    identity_bytes = len(json.dumps(freeze['identity'], sort_keys=True, allow_nan=False).encode())
    if identity_bytes > 131072:
        raise ValueError('Named study identity schema exceeds131072-byte bound')
    # Every actual index has at most6 digits plus separator; other complete fields are serialized explicitly.
    fixed = {key:value for key,value in freeze.items() if key not in ('calibration_indices','policy_indices')}
    fixed_bytes=len(json.dumps(fixed,sort_keys=True,allow_nan=False).encode())
    if fixed_bytes>FIXED_FIELDS_BYTES:
        raise ValueError('Complete fixed policy schema exceeds independent2MB allowance')
    bound = FIXED_FIELDS_BYTES + 64 + 8*(calibration_rows+policy_rows)
    actual = len(json.dumps(freeze, sort_keys=True, allow_nan=False).encode())
    if actual > bound:
        raise ValueError('Complete policy-freeze schema exceeds declared byte bound')
    return dict(json_bytes=actual, json_bound=bound, fixed_json_bytes=fixed_bytes,
                fixed_json_bound=FIXED_FIELDS_BYTES, identity_json_bytes=identity_bytes,
                calibration_indices=calibration_rows, policy_indices=policy_rows,
                search_entries=sum(len(v) for v in freeze['searches'].values()))


def measure(output, *, template, fixed, shape, study, guard, storage_check, artifact_root=None):
    import metadata_geometry
    output=Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records, summaries, writes, parses = [], [], 0., 0.
    for fold in range(3):
        counts={name:sum(r['rows'] for r in metadata_geometry.scope_records(shape,fold,name))
                for name in ('calibration','policy')}
        for arm in study.STUDY_ARMS:
            guard()
            prefix=f'fold-{fold}-{arm}'
            freeze=dict(identity=fixed['identity'],fold=fold,arm=arm,roles=copy.deepcopy(shape['roles'][fold]),
                model=dict(fixed['model'],path=prefix+'-model.joblib',numeric_manifest=prefix+'-model.joblib.json'),
                calibration=fixed['calibration'],policies=template['policies'],searches=template['searches'],
                frozen_before_any_outer_scoring=True,
                calibration_indices=list(range(629444-counts['calibration'],629444)),
                policy_indices=list(range(629444-counts['policy'],629444)))
            summary=validate(freeze,study=study,calibration_rows=counts['calibration'],policy_rows=counts['policy'])
            before=time.monotonic()
            artifact=study._json_artifact(output,prefix+'-policy-freeze.json',freeze,
                remaining_subtree_cap(output,artifact_root,study),storage_check)
            writes+=time.monotonic()-before
            before=time.monotonic()
            decoded=json.loads((output/artifact['path']).read_text())
            validate(decoded,study=study,calibration_rows=counts['calibration'],policy_rows=counts['policy'])
            if not study._same_numeric_json(freeze,decoded):
                raise ValueError('Production policy JSON roundtrip/equality differs')
            parses+=time.monotonic()-before
            records.append(dict(fold=fold,arm=arm,**artifact))
            summaries.append(dict(fold=fold,arm=arm,**summary))
    combined=dict(identity=fixed['identity'],entries=records)
    before=time.monotonic()
    artifact=study._json_artifact(output,'all-policy-freezes.json',combined,
        remaining_subtree_cap(output,artifact_root,study),storage_check)
    writes+=time.monotonic()-before
    before=time.monotonic()
    if json.loads((output/artifact['path']).read_text())!=combined:
        raise ValueError('Combined policy reference schema roundtrip differs')
    parses+=time.monotonic()-before
    return dict(freezes=summaries,combined_json_bytes=artifact['bytes'],files=7,
        write_hash_seconds=writes,parse_validate_equality_seconds=parses,
        source_sha256={name:hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for name,fn in
            (('execute_study',study.execute_study),('replay_study',study.replay_study),
             ('json_artifact',study._json_artifact),('same_numeric_json',study._same_numeric_json))},
        bounds='Named complete policy schema; index width8bytes; actual future freeze must satisfy all bounds',
        fixture='Generated codec fields and complete reconstructed searches; never a fitted policy',
        fits=0,selectors=0,full_run_eligible=False)
