#!/usr/bin/env python3
"""Run the retained Quantum SRC experiments sequentially, with resumable checkpoints."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(name,'1')
import argparse,hashlib,json,platform,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile',choices=['full','smoke'],default='full')
    ap.add_argument('--datasets',nargs='+',choices=['iris','wine','digits','subspaces','faces','grid'])
    ap.add_argument('--seeds',nargs='+',type=int)
    ap.add_argument('--stages',nargs='+',choices=['clean','shots'],default=['clean','shots'])
    ap.add_argument('--output',type=Path)
    ap.add_argument('--cache',type=Path,default=ROOT/'data_cache')
    ap.add_argument('--faces-npz',type=Path,help='Optional original-order Olivetti X/y NPZ (400 images, 40 identities)')
    ap.add_argument('--grid-csv',type=Path,help='Optional official Data_for_UCI_named.csv')
    args=ap.parse_args()
    if args.seeds is not None and (any(s<0 for s in args.seeds) or len(set(args.seeds))!=len(args.seeds)):
        ap.error('Seeds must be unique nonnegative integers')
    if args.datasets and len(set(args.datasets))!=len(args.datasets):ap.error('Datasets must be unique')
    cfg=json.loads((ROOT/'configs'/f'{args.profile}.json').read_text())
    if args.datasets is not None:cfg['datasets']=args.datasets
    if args.seeds is not None:cfg['seeds']=args.seeds
    cfg['stages']=sorted(set(args.stages));out=(args.output or ROOT/'results'/args.profile).resolve()
    if out==ROOT/'archived_results':ap.error('Choose an output folder other than archived_results')
    import numpy as np,scipy,sklearn
    from threadpoolctl import threadpool_limits
    from qsrc import data, data_extra, extra, linear
    from qsrc.experiments import run_clean
    from qsrc.shots import matched_shots
    versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'run_all.py',*sorted((ROOT/'qsrc').glob('*.py'))]}
    # Load and hash input bytes before deciding whether existing runs are reusable.
    loaded={};identities={}
    for name in cfg['datasets']:
        if name in ['faces','grid']:
            local=args.faces_npz if name=='faces' else args.grid_csv
            loaded[name]=data_extra.load(name,args.cache,local,cfg['extra']['subjects'])
            identities[name]=loaded[name][2]['dataset_sha256']
        else:
            h=hashlib.sha256()
            for seed in cfg['seeds']:
                X,y=data.dataset(name,seed);h.update(np.ascontiguousarray(X,dtype='<f8').tobytes());h.update(np.ascontiguousarray(y,dtype='<i8').tobytes())
            identities[name]=h.hexdigest()
    identity=dict(config=cfg,versions=versions,sources=sources,dataset_hashes=identities)
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['fingerprint']!=fingerprint:
        raise SystemExit('Existing output has a different configuration, data, dependency version, or source code. Use a new --output folder.')
    if not manifest.exists():atomic(manifest,dict(identity,fingerprint=fingerprint,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    with threadpool_limits(limits=1):
        for name in cfg['datasets']:
            for seed in cfg['seeds']:
                folder=out/'runs'/f'{name}_seed{seed}';path=folder/'result.json'
                if path.exists():print(name,seed,'checkpoint ready',flush=True);continue
                print(name,'seed',seed,flush=True);start=time.perf_counter();folder.mkdir(parents=True,exist_ok=True)
                if name not in ['faces','grid']:
                    c=cfg['small'];d=data.prepare(name,seed,c['qubits'],c['dictionary_per_class'],c['val_per_class'],c['test_per_class'])
                    clean=run_clean(d,c)
                    sampled=matched_shots(d,c,dict(quantum_params=clean['feature_map']),clean,seed) if 'shots' in cfg['stages'] else dict(rows=[])
                    for r in sampled['rows']:r['truth']=d['yt'][r['query_indices']].tolist()
                    np.savez_compressed(folder/'split.npz',**{k:v for k,v in d.items() if isinstance(v,np.ndarray)})
                    selected=dict(clean['feature_map'],dimensions=d['D'].shape[1])
                    protocol=dict(dictionary=len(d['yd']),validation=len(d['yv']),test=len(d['yt']),dimensions=d['D'].shape[1])
                else:
                    c=cfg['extra'];X,y,_=loaded[name]
                    tr,vi,ti=data_extra.splits(y,name,seed,c['tabular_dictionary'],c['validation_cap'],c['test_cap'])
                    di=data_extra.dictionary_ids(tr,y,name,seed,c['tabular_dictionary'])
                    prep,best=extra.select(X,y,tr,di,vi,ti,c,name)
                    split=dict(train=tr,dictionary=di,validation=vi,test=ti,truth=y[ti])
                    pre=dict(scaler_mean=prep['scaler'].mean_,scaler_scale=prep['scaler'].scale_,pca_mean=prep['pca'].mean_,pca_components=prep['pca'].components_,angle_scale=prep['angle'].scale_,angle_min=prep['angle'].min_)
                    rows=extra.evaluate(X,y,di,ti,prep,best,c,name,seed)
                    rows+=linear.run_case(X,y,name,seed,split,pre,c['occlusions'])['rows']
                    clean=dict(rows=rows);sampled=dict(rows=[]);selected=best['quantum_src']
                    np.savez_compressed(folder/'split.npz',**split);np.savez_compressed(folder/'preprocessing.npz',**pre)
                    protocol=dict(dictionary=len(di),validation=len(vi),test=len(ti),dimensions=selected['dimensions'])
                atomic(path,dict(dataset=name,seed=seed,exact=clean['rows'],sampled=sampled['rows'],selected=selected,protocol=protocol,seconds=time.perf_counter()-start))
    from qsrc.tables import reproduce
    reproduce(out,out/'tables')
    print('Completed. Tables:',out/'tables')


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('Interrupted. Rerun the identical command to reuse completed dataset/seed cases.',file=sys.stderr);sys.exit(130)
