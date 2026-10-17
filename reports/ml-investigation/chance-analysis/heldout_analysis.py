"""Descriptive null/AUROC on SAVED held-out prediction streams (no refit, no ECG access, no policy change)."""
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from common import *

CASES = {
    "fusion_primary": (RI + "/frozen-cohorts/primary-evaluation", RI + "/frozen-challenge-fusion/config.json"),
    "fusion_ltaf": (RI + "/frozen-cohorts/external-evaluation", RI + "/frozen-challenge-fusion/config.json"),
}
N_NULL = 200


def task(name):
    d, cfgp = CASES[name]
    ev = json.load(open(d + "/evaluation.json"))
    rows = rows_of(d + "/predictions.jsonl.gz")
    events = events_from_report(ev)
    cfg = json.load(open(cfgp))
    obs = run(rows, events, cfg)
    out = dict(name=name, obs=summarize(obs), saved=dict(m=ev["matched_events"], fw=ev["false_warnings"], n=ev["eligible_events"]))
    out["matched_by_patient"] = {p["patient_id"]: p["matched_events"] for p in obs["per_patient"] if p["matched_events"]}
    out["events_by_patient"] = {p["patient_id"]: p["eligible_events"] for p in obs["per_patient"] if p["eligible_events"]}
    out["n_patients_with_events"] = len(out["events_by_patient"])
    out["auroc_pooled"] = auroc_rows(rows)
    macro, per = patient_macro_auroc(rows, boot=2000, seed=1)
    out["auroc_patient_macro"] = macro
    rng = np.random.default_rng(7)
    out["null"] = [summarize(run(shift_risk(rows, rng), events, cfg)) for _ in range(N_NULL)]
    # sens-vs-FAR curve (descriptive only; thresholds from score quantiles; frozen policy untouched)
    acc = np.array([r["risk"] for r in rows if r["accepted"] and r["risk"] is not None and r["label"] in (0, 1)])
    curve = {}
    for q in [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]:
        thr = float(np.quantile(acc, q))
        c = cfg_with(dict(threshold=thr, release_threshold=0.75 * thr), cfg)
        e = dict(thr=thr, obs=summarize(run(rows, events, c)), null=[])
        for _ in range(20):
            e["null"].append(summarize(run(shift_risk(rows, rng), events, c)))
        curve[str(q)] = e
    out["curve"] = curve
    print(name, "done", flush=True)
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(task, list(CASES)))
    json.dump(res, open(str(HERE / "heldout_results.json"), "w"))
