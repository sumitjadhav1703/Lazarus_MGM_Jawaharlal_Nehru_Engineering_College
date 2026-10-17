import json, numpy as np
from collections import defaultdict
from pathlib import Path
A = str(Path(__file__).resolve().parent) + "/"


def pool(runs):
    return dict(m=sum(r["m"] for r in runs), n=sum(r["n"] for r in runs), fw=sum(r["fw"] for r in runs), w=sum(r["w"] for r in runs),
                warn_h=sum(r["warn_h"] for r in runs), mon_h=sum(r["mon_h"] for r in runs), elig_h=sum(r["elig_h"] for r in runs))


def rates(p):
    return dict(m=p["m"], n=p["n"], sens=p["m"] / p["n"] if p["n"] else None, fw=p["fw"], w=p["w"],
                far=p["fw"] / p["elig_h"] if p["elig_h"] else None, occ=p["warn_h"] / p["mon_h"] if p["mon_h"] else None)


def pooled_null(folds_null):
    k = min(len(x) for x in folds_null)
    return [pool([x[j] for x in folds_null]) for j in range(k)]


def dev_summary(res):
    out = {}
    names = []
    for r in res:
        if r["name"] not in names:
            names.append(r["name"])
    for nme in names:
        fr = sorted([r for r in res if r["name"] == nme], key=lambda r: r["fold"])
        o = pool([r["obs"] for r in fr])
        ent = dict(obs=rates(o), by_fold=[dict(fold=r["fold"], m=r["obs"]["m"], n=r["obs"]["n"], fw=r["obs"]["fw"], thr=r["thr"]) for r in fr])
        mb = defaultdict(int)
        eb = defaultdict(int)
        for r in fr:
            for p, c in r["matched_by_patient"].items():
                mb[p] += c
            for p, c in r["events_by_patient"].items():
                eb[p] += c
        tot = sum(mb.values())
        ent["conc"] = dict(catches=tot, patients_caught=len(mb), patients_with_events=len(eb), top_patient_share=(max(mb.values()) / tot if tot else None),
                           folds_with_catch=[r["fold"] for r in fr if r["obs"]["m"] > 0], matched_by_patient=dict(mb))
        # AUROC
        ent["auroc_folds"] = [r["auroc"][0] for r in fr]
        pa = {}
        for r in fr:
            pa.update(r["patient_auroc"])
        v = np.array(list(pa.values()))
        rng = np.random.default_rng(3)
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
        ent["auroc_patient_macro"] = dict(mean=float(v.mean()), median=float(np.median(v)), n=len(v), frac_gt_half=float((v > .5).mean()),
                                          ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        # null at saved policy
        nl = pooled_null([r["null"] for r in fr])
        ms = np.array([x["m"] for x in nl])
        ent["null"] = dict(n_draws=len(nl), mean_m=float(ms.mean()), p2_5=float(np.percentile(ms, 2.5)), p97_5=float(np.percentile(ms, 97.5)), max=int(ms.max()),
                           p_ge_obs=float((1 + (ms >= o["m"]).sum()) / (1 + len(ms))), mean_far=float(np.mean([x["fw"] / x["elig_h"] for x in nl])),
                           mean_occ=float(np.mean([x["warn_h"] / x["mon_h"] for x in nl])), mean_sens=float(ms.mean() / o["n"]))
        if "curve" in fr[0]:
            cur = []
            for q in fr[0]["curve"]:
                ob = pool([r["curve"][q]["obs"] for r in fr])
                e = dict(q=float(q), **rates(ob))
                if fr[0]["curve"][q]["null"]:
                    nn = pooled_null([r["curve"][q]["null"] for r in fr])
                    mm = np.array([x["m"] for x in nn]) / ob["n"]
                    e["null_sens_mean"] = float(mm.mean()); e["null_sens_lo"] = float(np.percentile(mm, 2.5)); e["null_sens_hi"] = float(np.percentile(mm, 97.5))
                    e["null_far"] = float(np.mean([x["fw"] / x["elig_h"] for x in nn]))
                cur.append(e)
            ent["curve"] = cur
        out[nme] = ent
    return out


if __name__ == "__main__":
    res = json.load(open(A + "dev_results.json"))
    s = dev_summary(res)
    json.dump(s, open(A + "dev_summary.json", "w"), indent=1)
    for k, e in s.items():
        o = e["obs"]; nl = e["null"]
        print(f"{k:26s} {o['m']}/{o['n']} sens={o['sens']*100:.2f}% FAR={o['far']:.4f} occ={o['occ']*100:.2f}% | null mean {nl['mean_m']:.1f} [{nl['p2_5']:.0f},{nl['p97_5']:.0f}] p>={nl['p_ge_obs']:.3f} | pts caught {e['conc']['patients_caught']}/{e['conc']['patients_with_events']} folds {e['conc']['folds_with_catch']} | AUROC folds {[round(a,3) if a else None for a in e['auroc_folds']]} macro {e['auroc_patient_macro']['mean']:.3f} {[round(x,3) for x in e['auroc_patient_macro']['ci95']]}")
