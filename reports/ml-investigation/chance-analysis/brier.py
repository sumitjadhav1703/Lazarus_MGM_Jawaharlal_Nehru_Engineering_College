import json, numpy as np
from common import *
from dev_analysis import MODELS, load, NC
res={}
for name in MODELS:
    ys=[];ss=[];per=[]
    for f in range(3):
        m=MODELS[name]
        if 'prec' in m: rows=rows_of(f"{NC}/precursor-linear-20261012/fold-{f}-{m['prec']}-predictions.jsonl.gz")
        else: rows=rows_of(f"{m['src']}/fold-{f}-{m['model']}-predictions.jsonl.gz")
        y=np.array([r['label'] for r in rows if r['accepted'] and r['risk'] is not None and r['label'] in (0,1)])
        s=np.array([r['risk'] for r in rows if r['accepted'] and r['risk'] is not None and r['label'] in (0,1)])
        ys.append(y);ss.append(s)
        per.append(dict(fold=f,n=len(y),pos=int(y.sum()),brier=float(((s-y)**2).mean()),const_brier=float(y.mean()*(1-y.mean())),auroc=float(roc_auc_score(y,s))))
    y=np.concatenate(ys);s=np.concatenate(ss)
    res[name]=dict(per_fold=per,pooled_brier=float(((s-y)**2).mean()),pooled_const_brier=float(y.mean()*(1-y.mean())),pooled_auroc=float(roc_auc_score(y,s)),mean_pred=float(s.mean()),prev=float(y.mean()),n=len(y),pos=int(y.sum()))
    print(name,{k:(round(v,6) if isinstance(v,float) else v) for k,v in res[name].items() if k!='per_fold'},[ (p['fold'],round(p['brier']-p['const_brier'],7),round(p['auroc'],3)) for p in per],flush=True)
json.dump(res,open(str(HERE/'brier_results.json'),'w'),indent=1)
