"""Public dataset loaders, strict Grid schema, and training-only transforms."""
import csv,hashlib,io,urllib.request,zipfile
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA

GRID_FEATURES=[f'{prefix}{j}' for prefix in ['tau','p','g'] for j in range(1,5)]
URLS={
 'grid':'https://archive.ics.uci.edu/static/public/471/electrical%2Bgrid%2Bstability%2Bsimulated%2Bdata.zip'}
FILENAMES={'grid':'Data_for_UCI_named.csv'}




def parse_grid(path):
    records=list(csv.DictReader(Path(path).read_text(encoding='utf-8-sig').splitlines()))
    if not records or not set(GRID_FEATURES+['stabf'])<=set(records[0]):
        raise ValueError('Grid CSV needs tau1..4, p1..4, g1..4 and stabf columns')
    labels=[r['stabf'].strip() for r in records]
    if not set(labels)<=set(['stable','unstable']):raise ValueError('Unknown Grid class label')
    # Fixed whitelist excludes BOTH target columns, including stab if supplied.
    return np.array([[float(r[k]) for k in GRID_FEATURES] for r in records]),np.array([int(v=='stable') for v in labels])


def load(name,cache,local=None,subjects=40):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    if name=='faces':
        if local:
            with np.load(local,allow_pickle=False) as d:
                X=np.asarray(d['X'],float);y=np.asarray(d['y'],int)
            if X.ndim==3:X=X.reshape(len(X),-1)
            if X.shape[1]!=4096:raise ValueError('Faces NPZ must contain 64x64 images under X and labels under y')
        else:
            d=fetch_olivetti_faces(data_home=str(cache),shuffle=False)
            X,y=d.data.astype(float),d.target.astype(int)
        classes=np.unique(y)[:subjects];mask=np.isin(y,classes);X,y=X[mask],y[mask]
        if len(np.unique(y))!=subjects or any(np.sum(y==c)!=10 for c in classes):
            raise ValueError('Faces requires ten images per selected identity')
        source='scikit-learn Olivetti Faces' if not local else str(Path(local).resolve())
    else:
        path=Path(local) if local else cache/FILENAMES[name]
        if not path.exists():
            print('Downloading official UCI dataset '+name,flush=True)
            with urllib.request.urlopen(URLS[name],timeout=60) as response:payload=response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as z:
                members=[n for n in z.namelist() if Path(n).name==FILENAMES[name]]
                if len(members)!=1:raise ValueError('Unexpected UCI download archive schema')
                path.write_bytes(z.read(members[0]))
        X,y=parse_grid(path)
        source=str(path.resolve())
    if not np.all(np.isfinite(X)):raise ValueError('Nonfinite dataset features')
    digest=hashlib.sha256(np.ascontiguousarray(X,dtype='<f8').tobytes()+np.ascontiguousarray(y,dtype='<i8').tobytes()).hexdigest()
    return X,y,dict(source=source,dataset_sha256=digest,rows=len(y),features=X.shape[1],classes=len(np.unique(y)))


def splits(y,name,seed,tabular_dictionary=50,validation_cap=200,test_cap=500):
    rng=np.random.default_rng(seed)
    if name=='faces':
        train=[];val=[];test=[]
        for c in np.unique(y):
            ids=rng.permutation(np.flatnonzero(y==c))
            train.extend(ids[:6]);val.extend(ids[6:8]);test.extend(ids[8:])
        return tuple(np.array(v,dtype=int) for v in [train,val,test])
    ids=np.arange(len(y))
    tr,rest=train_test_split(ids,test_size=.4,stratify=y,random_state=seed)
    va,te=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=seed+1)
    def cap(indices,n):
        return np.concatenate([rng.choice(indices[y[indices]==c],min(n,np.sum(y[indices]==c)),replace=False) for c in np.unique(y)])
    # Fit preprocessing on all tr; classifier dictionary is a capped subset.
    return tr,cap(va,validation_cap),cap(te,test_cap)


def dictionary_ids(train,y,name,seed,per_class):
    if name=='faces':return train.copy()
    rng=np.random.default_rng(seed+54)
    return np.concatenate([rng.choice(train[y[train]==c],min(per_class,np.sum(y[train]==c)),replace=False) for c in np.unique(y)])


def prepare(X,train,dictionary,val,test,maximum_dim):
    scaler=StandardScaler().fit(X[train])
    dimension=min(maximum_dim,X.shape[1],len(train)-1)
    pca=PCA(n_components=dimension,svd_solver='full').fit(scaler.transform(X[train]))
    ztrain=pca.transform(scaler.transform(X[train]))
    angle=MinMaxScaler(feature_range=(0,np.pi),clip=True).fit(ztrain)
    def transform(ids):
        z=pca.transform(scaler.transform(X[ids]));return z,angle.transform(z)
    zd,ad=transform(dictionary);zv,av=transform(val);zt,at=transform(test)
    return dict(zd=zd,zv=zv,zt=zt,ad=ad,av=av,at=at,scaler=scaler,pca=pca,angle=angle)


def transform_extra(prep,X):
    z=prep['pca'].transform(prep['scaler'].transform(X))
    return z,prep['angle'].transform(z)


def occlude(images,fraction,seed):
    """A deterministic contiguous black rectangle, applied only to test pixels.

    Same normalized anchors across severities; masks need not be nested.
    Reported occluded area is the actual pixel fraction.
    """
    X=np.array(images,copy=True).reshape(-1,64,64)
    if fraction==0:return X.reshape(len(X),-1),0.
    side=max(1,int(round(64*np.sqrt(fraction))))
    if side>64:raise ValueError('Occlusion fraction exceeds image area')
    rng=np.random.default_rng(seed)
    anchors=rng.uniform(0,1,(len(X),2))
    for i in range(len(X)):
        top,left=np.floor(anchors[i]*(65-side)).astype(int)
        X[i,top:top+side,left:left+side]=0
    return X.reshape(len(X),-1),float(side*side/(64*64))
