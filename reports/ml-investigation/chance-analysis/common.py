"""Shared loaders for analyst work. Reads SAVED artifacts only; evaluate_stream is the repo's pure function."""
import gzip, json, sys, copy
from pathlib import Path
from collections import defaultdict
import numpy as np
# Saved-result inputs (predictions, next-cycle receipts) are NOT shipped in this repository; see README.md here.
HERE = Path(__file__).resolve().parent
ROOT = str(HERE.parents[2])
RI = str(HERE.parent)
sys.path.insert(0, ROOT + "/src")
from cardiotwin.evaluation import evaluate_stream  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

BASE_CFG = json.load(open(RI + "/frozen-challenge-fusion/config.json"))


def rows_of(path):
    import re
    with gzip.open(path, "rt") as f:
        out = [json.loads(l) for l in f if l.strip()]
    for r in out:  # classical folds label records "fold-N:record_X"; reports use "record_X"
        r["record_id"] = re.sub(r"^fold-\d+:", "", r["record_id"])
    return out


def events_from_report(rep):
    """Exact onsets: missed onsets + (matched warning timestamp + lead)."""
    ev = [dict(patient_id=m["patient_id"], record_id=m["record_id"], onset=m["onset"]) for m in rep["missed_event_details"]]
    matched = [w for w in rep["warning_details"] if w.get("matched")]
    assert len(matched) == len(rep["lead_seconds"]), (len(matched), len(rep["lead_seconds"]))
    for w, lead in zip(matched, rep["lead_seconds"]):
        ev.append(dict(patient_id=w["patient_id"], record_id=w["record_id"], onset=w["timestamp"] + lead))
    return ev


def cfg_with(alert, base=None):
    c = copy.deepcopy(base or BASE_CFG)
    c["alert"] = {**c["alert"], **alert}
    return c


def run(rows, events, cfg):
    r = evaluate_stream(rows, events, cfg, bootstrap_samples=0, include_probability=False)
    return r


def summarize(r):
    return dict(m=r["matched_events"], n=r["eligible_events"], fw=r["false_warnings"], w=r["warnings"],
                far=r["false_warnings_per_eligible_hour"], warn_h=r["warning_hours"], mon_h=r["monitored_hours"],
                elig_h=r["eligible_hours"])


def shift_risk(rows, rng):
    """Circularly shift risk values among accepted rows within each record (keeps marginal, autocorrelation, quality mask)."""
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[(r["patient_id"], r["record_id"])].append(i)
    out = [dict(r) for r in rows]
    for idx in by.values():
        idx.sort(key=lambda i: rows[i]["timestamp"])
        acc = [i for i in idx if rows[i]["accepted"] and rows[i]["risk"] is not None]
        if len(acc) < 2:
            continue
        k = int(rng.integers(1, len(acc)))
        vals = [rows[i]["risk"] for i in acc]
        vals = vals[k:] + vals[:k]
        for i, v in zip(acc, vals):
            out[i]["risk"] = v
    return out


def auroc_rows(rows):
    y, s = [], []
    for r in rows:
        if r["accepted"] and r["risk"] is not None and r["label"] in (0, 1):
            y.append(r["label"]); s.append(r["risk"])
    y = np.array(y); s = np.array(s)
    return (float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None), int(y.sum()), len(y)


def patient_macro_auroc(rows, boot=0, seed=0):
    by = defaultdict(lambda: ([], []))
    for r in rows:
        if r["accepted"] and r["risk"] is not None and r["label"] in (0, 1):
            by[r["patient_id"]][0].append(r["label"]); by[r["patient_id"]][1].append(r["risk"])
    per = {}
    for p, (y, s) in by.items():
        y = np.array(y)
        if 0 < y.sum() < len(y):
            per[p] = float(roc_auc_score(y, s))
    vals = np.array(list(per.values()))
    out = dict(macro=float(vals.mean()) if len(vals) else None, n_patients=len(vals),
               frac_gt_half=float((vals > 0.5).mean()) if len(vals) else None)
    if boot and len(vals):
        rng = np.random.default_rng(seed)
        bs = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(boot)]
        out["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out, per
