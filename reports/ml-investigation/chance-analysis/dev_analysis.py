"""Development-fold analysis from saved per-row predictions. Run from the development workspace layout: OMP_NUM_THREADS=1 .venv/bin/python reports/ml-investigation/chance-analysis/dev_analysis.py"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from common import *

NC = RI + "/next-cycle"
MODELS = {
    "fusion_finalist": dict(src=RI + "/classical-resumed", model="logistic_combined"),
    "ecg_reference": dict(src=RI + "/classical-resumed", model="logistic_current_recipe"),
    "logistic_core": dict(src=RI + "/classical-resumed", model="logistic_core"),
    "rr_tcn16_patient_only": dict(src=NC + "/recovery-review-20261011/outputs/cardiotwin-checkpoint-cycle/patient_only", model="rr_tcn_16"),
    "rr_tcn16_patient_class": dict(src=NC + "/recovery-review-20261011/outputs/cardiotwin-checkpoint-cycle/patient_class", model="rr_tcn_16"),
    "waveform_cnn16": dict(src=NC + "/waveform-screen-20261011/outputs/cardiotwin-waveform-screen/model", model="waveform_cnn_16"),
    "precursor_pointwise": dict(prec="pointwise_pair_control"),
    "precursor_rank": dict(prec="within_person_rank"),
}
PKEY = "far_0.25_occupancy_0.05"
QS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
QNULL = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
N_NULL, N_CURVE_NULL = 60, 20


def load(name, f):
    m = MODELS[name]
    if "prec" in m:
        base = NC + "/precursor-linear-20261012"
        r = json.load(open(f"{base}/fold-{f}-{m['prec']}-receipt.json"))
        rows = rows_of(f"{base}/fold-{f}-{m['prec']}-predictions.jsonl.gz")
        rep = r["reports"][PKEY]
        base_cfg = r["policies"][PKEY]
        return rows, events_from_report(rep), base_cfg, rep
    exp = [json.loads(l) for l in open(m["src"] + "/experiments.jsonl")]
    r = [x for x in exp if x.get("model") == m["model"] and x.get("fold") == f][0]
    rows = rows_of(f"{m['src']}/fold-{f}-{m['model']}-predictions.jsonl.gz")
    return rows, events_from_report(r["report"]), cfg_with(r["selected_policy"]), r["report"]


def task(args):
    name, f = args
    t0 = time.time()
    rows, events, cfg, saved = load(name, f)
    out = dict(name=name, fold=f, n_events=len(events))
    obs = run(rows, events, cfg)
    out["obs"] = summarize(obs)
    out["saved_check"] = dict(m=saved["matched_events"], fw=saved["false_warnings"])
    out["matched_patients"] = sorted({w["patient_id"] for w in obs["warning_details"] if w["matched"]})
    out["matched_by_patient"] = {p["patient_id"]: p["matched_events"] for p in obs["per_patient"] if p["matched_events"]}
    out["events_by_patient"] = {p["patient_id"]: p["eligible_events"] for p in obs["per_patient"] if p["eligible_events"]}
    out["leads"] = obs["lead_seconds"]
    out["auroc"] = auroc_rows(rows)
    macro, per = patient_macro_auroc(rows)
    out["patient_auroc"] = per
    out["thr"] = cfg["alert"]["threshold"]
    if name.startswith("precursor"):
        out["burden_occ"] = obs.get("burden_seconds", None)
    # null at saved policy
    rng = np.random.default_rng(1000 + f)
    nulls = []
    for _ in range(N_NULL):
        s = summarize(run(shift_risk(rows, rng), events, cfg))
        nulls.append(s)
    out["null"] = nulls
    if not name.startswith("precursor"):  # finite-policy precursor curve is the saved 9-policy grid instead
        acc = np.array([r["risk"] for r in rows if r["accepted"] and r["risk"] is not None and r["label"] in (0, 1)])
        curve = {}
        for q in QS:
            thr = float(np.quantile(acc, q))
            c = cfg_with(dict(threshold=thr, release_threshold=0.75 * thr), cfg)
            entry = dict(thr=thr, obs=summarize(run(rows, events, c)), null=[])
            if q in QNULL:
                for _ in range(N_CURVE_NULL):
                    entry["null"].append(summarize(run(shift_risk(rows, rng), events, c)))
            curve[str(q)] = entry
        out["curve"] = curve
    out["secs"] = time.time() - t0
    print(name, f, "done", round(out["secs"]), "s", out["obs"]["m"], "/", out["obs"]["n"], flush=True)
    return out


if __name__ == "__main__":
    jobs = [(n, f) for n in MODELS for f in range(3)]
    with ProcessPoolExecutor(7) as ex:
        res = list(ex.map(task, jobs))
    json.dump(res, open(str(HERE / "dev_results.json"), "w"))
