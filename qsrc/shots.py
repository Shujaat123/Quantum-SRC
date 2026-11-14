"""Matched finite-shot simulation; exact shared RNG stream from archived experiments."""
import numpy as np
from sklearn.svm import SVC
from . import base
from .solver import predict, psd_repair

def balanced_queries(y,count,seed):
    count=max(count,len(np.unique(y)))
    rng=np.random.default_rng(seed)
    queues=[list(rng.permutation(np.flatnonzero(y==c))) for c in rng.permutation(np.unique(y))]
    result=[]
    while len(result)<min(count,len(y)):
        for q in queues:
            if q and len(result)<min(count,len(y)):result.append(q.pop())
    return np.array(result,dtype=int)

def matched_shots(data,cfg,exact,final,seed):
    K,_,kt=base.matrices_from_selected(data,exact);yd,yt=data['yd'],data['yt']
    idx=balanced_queries(yt,cfg['followup_queries'],seed+91);kt=kt[:,idx];yt=yt[idx]
    i,j=np.triu_indices(len(yd),1);rows=[]
    parameters=final['parameters']
    ideals={}
    for model in ['quantum_src']:
        p=parameters[model];ideals[model]=predict(K,kt,yd,p['lam'],p['mu'])[0]
    C=parameters['quantum_svm']['C']
    ideals['quantum_svm']=SVC(kernel='precomputed',C=C).fit(K,yd).predict(kt.T)
    for rep in range(cfg['followup_repeats']):
        for gs in cfg['shared_gram_shots']:
            rng=np.random.default_rng(np.random.SeedSequence([seed,rep,gs,1100]))
            raw=np.eye(len(yd));raw[i,j]=raw[j,i]=rng.binomial(gs,K[i,j])/gs
            H=psd_repair(raw)
            for qs in cfg['query_shots']:
                h=rng.binomial(qs,kt)/qs
                for model in ['quantum_src','quantum_svm']:
                    if model=='quantum_svm':
                        pred=SVC(kernel='precomputed',C=C).fit(H,yd).predict(h.T)
                    else:
                        p=parameters[model];pred=predict(H,h,yd,p['lam'],p['mu'])[0]
                    rows.append(dict(model=model,repeat=rep,gram_shots_per_arm=gs,
                        query_shots_per_arm=qs,**base.metrics(yt,pred),
                        agreement=float(np.mean(pred==ideals[model])),
                        gram_shots=int(gs*len(i)),query_shots=int(qs*kt.size),
                        total_batch_shots=int(gs*len(i)+qs*kt.size),
                        amortized_shots_per_query=float(gs*len(i)/len(yt)+qs*len(yd)),
                        query_indices=idx.tolist(),predictions=pred.tolist()))
    return dict(rows=rows,n_queries=len(idx),resource_model='One cached Gram, shared query samples across classifiers')
