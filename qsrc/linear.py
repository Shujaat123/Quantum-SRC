"""Historical six-dimensional angle-scaled linear SRC on shared extra-data preprocessing."""
DIMENSIONS=6
LAMBDAS=[.001,.01,.05]
MUS=[.1,.5]
MODEL="original_linear_src"

def angle_features(X,prep,dimensions=DIMENSIONS):
    """Apply saved StandardScaler -> leading PCA -> clipped [0,pi] transform.

    A full-SVD PCA's leading six components are the same six-component map used
    by the historical implementation, apart from ordinary numerical roundoff.
    """
    import numpy as np
    d=min(dimensions,len(prep['pca_components']))
    standardized=(np.asarray(X,float)-prep['scaler_mean'])/prep['scaler_scale']
    z=(standardized-prep['pca_mean'])@prep['pca_components'][:d].T
    a=z*prep['angle_scale'][:d]+prep['angle_min'][:d]
    return np.clip(a,0,np.pi)

def validate_split(X,y,split):
    import numpy as np
    keys=['train','dictionary','validation','test','truth']
    if any(k not in split for k in keys):raise ValueError('Saved split is incomplete')
    for key in keys[:-1]:
        ids=split[key]
        if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0) or np.any(ids>=len(y)):
            raise ValueError('Invalid saved index array: '+key)
        if len(set(ids.tolist()))!=len(ids):raise ValueError('Duplicate indices: '+key)
    tr,di,vi,ti=[set(split[k].tolist()) for k in keys[:-1]]
    if not di<=tr or tr&vi or tr&ti or vi&ti:raise ValueError('Saved split overlap or dictionary outside training set')
    if not np.array_equal(y[split['test']],split['truth']):raise ValueError('Saved test labels do not match loaded dataset')
    if set(np.unique(y[split['dictionary']]))!=set(np.unique(y[split['test']])):
        raise ValueError('Dictionary and test class sets differ')

def score_src(K,k,labels,lam,mu):
    import numpy as np
    from .solver import solve,residuals
    a,info=solve(K,k,lam,mu)
    classes,r=residuals(K,k,a,labels)
    predicted=classes[np.argmin(r,axis=1)]
    binary=r[:,0]-r[:,1] if len(classes)==2 else None
    return predicted,binary,a,info

def tune(K,kv,yd,yv):
    from sklearn.metrics import balanced_accuracy_score
    best=None;search=[]
    for lam in LAMBDAS:
        for mu in MUS:
            pred,_,_,info=score_src(K,kv,yd,lam,mu)
            val=float(balanced_accuracy_score(yv,pred))
            record=dict(lam=lam,mu=mu,validation_balanced_accuracy=val,kkt_residual=info['kkt_residual'])
            search.append(record)
            if best is None or val>best['validation_balanced_accuracy']:best=record.copy()
    return best,search

def metrics(y,pred,binary):
    import numpy as np
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score
    result=dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)))
    if binary is not None and len(np.unique(y))==2:result['roc_auc']=float(roc_auc_score(y,binary))
    return result

def run_case(X,y,name,seed,split,prep,occlusions):
    import numpy as np
    from sklearn.preprocessing import normalize
    from .data_extra import occlude
    validate_split(X,y,split)
    di,vi,ti=[split[k] for k in ['dictionary','validation','test']]
    D=normalize(angle_features(X[di],prep));V=normalize(angle_features(X[vi],prep))
    K=D@D.T;kv=D@V.T;yd,yv,yt=y[di],y[vi],y[ti]
    selected,search=tune(K,kv,yd,yv)
    rows=[]
    for fraction in occlusions if name=='faces' else [0.]:
        if name=='faces':test,actual=occlude(X[ti],fraction,seed+988)
        else:test,actual=X[ti],0.
        T=normalize(angle_features(test,prep))
        pred,binary,a,info=score_src(K,D@T.T,yd,selected['lam'],selected['mu'])
        rows.append(dict(model=MODEL,occlusion_requested=fraction,occlusion_actual=actual,
            dimensions=D.shape[1],lam=selected['lam'],mu=selected['mu'],
            **metrics(yt,pred,binary),predictions=pred.tolist(),truth=yt.tolist(),
            binary_scores=binary.tolist() if binary is not None else None,
            support_size=float(np.mean(np.sum(np.abs(a)>1e-6,axis=0))),
            solver=info,dictionary_size=len(di),test_size=len(ti)))
    return dict(rows=rows,selection=selected,validation_search=search,
        definition='Historical StandardScaler -> PCA(6) -> train-fitted [0,pi] scaling/clipping -> per-sample L2 normalization -> linear kernel elastic-net SRC')
