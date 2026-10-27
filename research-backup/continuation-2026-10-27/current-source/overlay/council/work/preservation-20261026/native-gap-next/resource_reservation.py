"""Prospective source reservations, not measured RSS, bytes or accepted cost evidence."""
import json
import struct
import sys


def deep_bytes(value):
    total=sys.getsizeof(value)
    if isinstance(value,dict):
        total+=sum(deep_bytes(k)+deep_bytes(v) for k,v in value.items())
    elif isinstance(value,(list,tuple)):
        total+=sum(deep_bytes(v) for v in value)
    return total


def reservation(*, warning, missed, person):
    if sys.implementation.name!='cpython' or struct.calcsize('P')!=8:
        raise ValueError('Reservation requires64bit CPython runtime')
    units=dict(warning=4096,missed=4096,person=16384)
    json_units=dict(warning=1024,missed=2048,person=8192)
    for name,value in (('warning',warning),('missed',missed),('person',person)):
        if deep_bytes(value)>units[name]:
            raise ValueError('Generated schema object exceeds named Python reservation: '+name)
        if len(json.dumps(value,sort_keys=True,allow_nan=False).encode())>json_units[name]:
            raise ValueError('Generated schema exceeds named serialized reservation: '+name)
    # Six live receipts share their children with summary/review; one independent decoded full summary coexists.
    report_objects=632100*units['warning']+4820*units['missed']+4840*units['person']+200_000_000
    serialized_summary=632100*1024+4820*2048+4840*8192+26*131072
    native_cache=580347368
    module_runtime=1_000_000_000
    table_peak=native_cache+module_runtime+2_619_117_060+400_000_000
    report_peak=native_cache+module_runtime+2*report_objects+3*serialized_summary
    return dict(runtime=sys.version.split()[0],units_bytes=units,
        table_phase_rss_reservation=table_peak,report_phase_rss_reservation=report_peak,
        admissible=max(table_peak,report_peak)<=10_000_000_000,
        dependencies_and_imports_reservation=module_runtime,
        storage_additional_to_mounted_input=dict(staged_cache=580347368,dependencies_and_scratch=3_000_000_000,
            source_overlay=10_000_000,generated_table=2_619_117_060,retained_artifacts=250_000_000,reserve=250_000_000),
        admission='Future mounted input plus phase-live additions must fit25GB; actual guard remains mandatory',
        caveat='Conservative declared reservations; postmeasurement RSS/bytes and scalar/entropy applicability still require review',
        measured=False,full_run_eligible=False)
