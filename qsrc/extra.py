"""Quantum-map search and clean/occluded evaluations on Faces and Grid."""
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from .quantum import states,fidelity
from .solver import solve,residuals
from .data_extra import prepare, transform_extra, occlude


def select(X,y,tr,di,vi,ti,cfg,name):
    """Original dimension/depth/bandwidth search, retaining the requested decoders."""
    prep=prepare(X,tr,di,vi,ti,max(cfg['dimensions']+[64 if name=='faces' else X.shape[1]]))
    yd,yv=y[di],y[vi];best={}
    dimensions=sorted({min(d,prep['ad'].shape[1]) for d in cfg['dimensions']})
    for d in dimensions:
        print('    validation search: PCA dimensions',d,flush=True)
        ad,av=prep['ad'][:,:d],prep['av'][:,:d]
        for depth in cfg['depths']:
            for bandwidth in cfg['bandwidths']:
                args=dict(qubits=cfg['qubits'],depth=depth,bandwidth=bandwidth)
                sd,sv=states(ad,**args),states(av,**args)
                p=tune_src(fidelity(sd),fidelity(sd,sv),yd,yv,cfg)
                if 'quantum_src' not in best or p['validation_score']>best['quantum_src']['validation_score']:
                    best['quantum_src']=dict(p,dimensions=d,**args)
    qp=best['quantum_src'];d=qp['dimensions']
    args={k:qp[k] for k in ['qubits','depth','bandwidth']}
    sd,sv=states(prep['ad'][:,:d],**args),states(prep['av'][:,:d],**args)
    best['quantum_svm']=dict(tune_svm(fidelity(sd),fidelity(sd,sv),yd,yv,cfg),dimensions=d,**args)
    best['quantum_nn']=dict(dimensions=d,**args)
    return prep,best


def evaluate(X,y,di,ti,prep,best,cfg,name,seed):
    yd,yt=y[di],y[ti];rows=[]
    for fraction in cfg['occlusions'] if name=='faces' else [0.]:
        test,actual=occlude(X[ti],fraction,seed+988) if name=='faces' else (X[ti],0.)
        _,at=transform_extra(prep,test)
        for model,p in best.items():
            d=p['dimensions'];args={k:p[k] for k in ['qubits','depth','bandwidth']}
            sd,st=states(prep['ad'][:,:d],**args),states(at[:,:d],**args)
            K,k=fidelity(sd),fidelity(sd,st)
            if model=='quantum_svm':
                clf=SVC(kernel='precomputed',C=p['C']).fit(K,yd)
                pred=clf.predict(k.T);score=clf.decision_function(k.T) if len(np.unique(yd))==2 else None;info={}
            elif model=='quantum_nn':
                pred=yd[np.argmax(k,axis=0)];info={}
                score=np.max(k[yd==1],axis=0)-np.max(k[yd==0],axis=0) if len(np.unique(yd))==2 else None
            else:pred,score,info=src_scores(K,k,yd,p['lam'],p['mu'])
            rows.append(dict(model=model,occlusion_requested=fraction,occlusion_actual=actual,
                **metrics(yt,pred,score),parameters=p,predictions=pred.tolist(),truth=yt.tolist(),
                binary_scores=score.tolist() if score is not None else None,
                solver_kkt_residual=info.get('kkt_residual')))
    return rows

def metrics(y,pred,score=None):
    result=dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)))
    if score is not None and len(np.unique(y))==2:result['roc_auc']=float(roc_auc_score(y,score))
    return result

def src_scores(K,k,labels,lam,mu):
    a,info=solve(K,k,lam,mu)
    classes,r=residuals(K,k,a,labels)
    p=classes[np.argmin(r,axis=1)]
    score=r[:,0]-r[:,1] if len(classes)==2 else None
    return p,score,info

def tune_src(K,k,yd,yv,cfg,lambdas=None):
    best=None
    for lam in cfg['lambdas'] if lambdas is None else lambdas:
        for mu in cfg['mus']:
            p,score,info=src_scores(K,k,yd,lam,mu)
            val=float(balanced_accuracy_score(yv,p))
            if best is None or val>best['validation_score']:
                best=dict(lam=lam,mu=mu,validation_score=val)
    return best

def tune_svm(K,k,yd,yv,cfg):
    best=None
    for C in cfg['Cs']:
        model=SVC(kernel='precomputed',C=C).fit(K,yd)
        val=float(balanced_accuracy_score(yv,model.predict(k.T)))
        if best is None or val>best['validation_score']:best=dict(C=C,validation_score=val)
    return best
