"""Exact historical protocol for the four small benchmarks."""
import itertools
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from . import base
from .solver import predict


def run_clean(data,cfg):
    best=None
    # The first-stage map search uses the ORIGINAL mu grid. The follow-up
    # retunes mu without reselecting the map. Combining the searches changes results.
    for depth,bandwidth in itertools.product(cfg['depths'],cfg['bandwidths']):
        K,kv,kt=base.quantum_matrices(data,cfg['qubits'],depth,bandwidth)
        p=base.tune_src(K,kv,data['yd'],data['yv'],cfg['lambdas'],cfg['mus'])
        p.update(depth=depth,bandwidth=bandwidth,qubits=cfg['qubits'])
        if best is None or p['validation_score']>best[0]['validation_score']:
            best=(p,K,kv,kt)
    selected,K,kv,kt=best;yd,yv,yt=data['yd'],data['yv'],data['yt']
    rows=[];parameters={}
    p=base.tune_src(K,kv,yd,yv,cfg['lambdas'],cfg['followup_mus'])
    pred,a,info=predict(K,kt,yd,p['lam'],p['mu']);parameters['quantum_src']=p
    rows.append(dict(model='quantum_src',**p,**base.metrics(yt,pred),predictions=pred.tolist(),truth=yt.tolist(),
                     support_size=float(np.mean(np.sum(np.abs(a)>1e-6,axis=0))),kkt_residual=info['kkt_residual']))
    p=base.tune_svm(K,kv,yd,yv,cfg['followup_Cs']);parameters['quantum_svm']=p
    pred=SVC(kernel='precomputed',C=p['C']).fit(K,yd).predict(kt.T)
    rows.append(dict(model='quantum_svm',**p,**base.metrics(yt,pred),predictions=pred.tolist(),truth=yt.tolist()))
    pred=yd[np.argmax(kt,axis=0)]
    rows.append(dict(model='quantum_nn',**base.metrics(yt,pred),predictions=pred.tolist(),truth=yt.tolist()))
    D,V,T=[normalize(data[key]) for key in ['D','V','T']]
    H,hv,ht=D@D.T,D@V.T,D@T.T
    p=base.tune_src(H,hv,yd,yv,cfg['lambdas'],cfg['mus'])
    pred,a,info=predict(H,ht,yd,p['lam'],p['mu'])
    rows.append(dict(model='original_linear_src',**p,**base.metrics(yt,pred),predictions=pred.tolist(),truth=yt.tolist(),
                     support_size=float(np.mean(np.sum(np.abs(a)>1e-6,axis=0))),kkt_residual=info['kkt_residual']))
    return dict(rows=rows,parameters=parameters,feature_map=selected)
