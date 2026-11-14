"""Reproduce numeric CSV and Markdown tables from experiment checkpoints."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t
from .base import metrics

DATASETS=['iris','wine','digits','subspaces','faces','grid']
MODELS=['quantum_src','quantum_svm','quantum_nn','original_linear_src']
LABELS=dict(zip(MODELS,['Quantum SRC','Quantum SVM','Quantum nearest neighbor','Original Linear SRC']))


def read_results(path):
    path=Path(path)
    if not (path/'manifest.json').is_file():raise ValueError('Results directory must contain manifest.json')
    exact=[];shots=[];selected=[];protocol=[];diagnostics=[]
    for f in sorted((path/'runs').glob('*/result.json')):
        case=json.loads(f.read_text());d,s=case['dataset'],case['seed']
        if d not in DATASETS:raise ValueError('Unsupported dataset: '+d)
        for row in case['exact']:
            if row['model'] not in MODELS:raise ValueError('Unsupported model: '+row['model'])
            if 'truth' not in row:raise ValueError('Missing labels for metric verification: '+str(f))
            actual=metrics(row['truth'],row['predictions'])
            for k,v in actual.items():
                if not np.isclose(v,row[k],atol=1e-10,rtol=0):raise ValueError(f'{f}: stored {k} disagrees with predictions')
            exact.append(dict(dataset=d,seed=s,model=row['model'],occlusion=row.get('occlusion_requested',0.),occlusion_actual=row.get('occlusion_actual',0.),**actual))
            kkt=row.get('kkt_residual',row.get('solver_kkt_residual',row.get('solver',{}).get('kkt_residual')))
            if kkt is not None:diagnostics.append(dict(dataset=d,seed=s,model=row['model'],occlusion=row.get('occlusion_requested',0.),kkt_residual=kkt))
        for row in case['sampled']:
            actual=metrics(row['truth'],row['predictions'])
            for k,v in actual.items():
                if not np.isclose(v,row[k],atol=1e-10,rtol=0):raise ValueError(f'{f}: sampled {k} disagrees with predictions')
            ideal=next(r for r in case['exact'] if r['model']==row['model'] and r.get('occlusion_requested',0)==0)
            agreement=np.mean(np.array(row['predictions'])==np.array(ideal['predictions'])[row['query_indices']])
            if not np.isclose(agreement,row['agreement'],atol=1e-10,rtol=0):raise ValueError(f'{f}: incorrect ideal-model agreement')
            shots.append(dict(dataset=d,seed=s,**{k:v for k,v in row.items() if k not in ['truth','predictions','query_indices']}))
        selected.append(dict(dataset=d,seed=s,**case['selected']))
        protocol.append(dict(dataset=d,seed=s,**case['protocol']))
    if not exact:raise ValueError('No completed run records found')
    exact=pd.DataFrame(exact)
    if exact.duplicated(['dataset','seed','model','occlusion']).any():raise ValueError('Duplicate dataset/seed/model/condition records')
    for (d,s),g in exact[exact.occlusion==0].groupby(['dataset','seed']):
        if set(g.model)!=set(MODELS):raise ValueError(f'Incomplete model pairing: {d} seed {s}')
    return exact,pd.DataFrame(shots),pd.DataFrame(selected),pd.DataFrame(protocol),pd.DataFrame(diagnostics)


def reproduce(source,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    exact,shots,selected,protocol,diagnostics=read_results(source)
    def save(name,df):df.to_csv(out/(name+'.csv'),index=False,float_format='%.12g')
    save('per_seed_results',exact);save('selected_parameters',selected);save('protocol',protocol);save('solver_diagnostics',diagnostics)
    clean=exact[exact.occlusion==0]
    summary=clean.groupby(['dataset','model'],sort=False).agg(mean_balanced_accuracy=('balanced_accuracy','mean'),sd_balanced_accuracy=('balanced_accuracy','std'),n_seeds=('seed','nunique')).reset_index()
    summary['rank']=summary.groupby('dataset').mean_balanced_accuracy.rank(method='dense',ascending=False).astype(int)
    save('clean_results',summary)
    secondary=clean.groupby(['dataset','model'],sort=False)[['accuracy','macro_f1']].mean().reset_index();save('secondary_metrics',secondary)
    pairs=[];seed_pairs=[]
    for d in DATASETS:
        g=clean[clean.dataset==d]
        if g.empty:continue
        pivot=g.pivot(index='seed',columns='model',values='balanced_accuracy')
        if pivot.isna().any().any():raise ValueError('Unpaired seed records for '+d)
        for m in MODELS[1:]:
            v=100*(pivot.quantum_src-pivot[m]);n=len(v);sd=v.std(ddof=1) if n>1 else np.nan
            half=t.ppf(.975,n-1)*sd/np.sqrt(n) if n>1 else np.nan
            pairs.append(dict(dataset=d,comparison=m,n_seeds=n,mean_pp=v.mean(),sd_pp=sd,descriptive_low=v.mean()-half,descriptive_high=v.mean()+half,wins=int((v>1e-10).sum()),ties=int((v.abs()<=1e-10).sum()),losses=int((v< -1e-10).sum())))
            seed_pairs.extend(dict(dataset=d,seed=int(s),comparison=m,difference_pp=float(delta)) for s,delta in v.items())
    save('paired_results',pd.DataFrame(pairs));save('paired_seed_differences',pd.DataFrame(seed_pairs))
    face=exact[exact.dataset=='faces']
    if not face.empty:
        save('face_occlusion',face.groupby(['occlusion_actual','model']).agg(mean_balanced_accuracy=('balanced_accuracy','mean'),sd_balanced_accuracy=('balanced_accuracy','std'),n_seeds=('seed','nunique')).reset_index())
    depth=selected.groupby(['dataset','depth']).size().reset_index(name='count');save('selected_depth_counts',depth)
    if not shots.empty:
        save('finite_shot_results',shots)
        keys=['dataset','model','gram_shots_per_arm','query_shots_per_arm']
        # First average measurement repeats WITHIN seed. Seeds, not individual
        # measurement repeats or repeated test appearances, are the summary units.
        perseed=shots.groupby(keys+['seed'])[['accuracy','balanced_accuracy','macro_f1','agreement','total_batch_shots']].mean().reset_index()
        grid=perseed.groupby(keys).agg(n_seeds=('seed','nunique'),mean_agreement=('agreement','mean'),sd_agreement=('agreement','std'),mean_balanced_accuracy=('balanced_accuracy','mean'),sd_balanced_accuracy=('balanced_accuracy','std'),batch_shots=('total_batch_shots','mean')).reset_index()
        save('finite_shot_grid',grid)
        save('finite_shot_display',grid[(grid.gram_shots_per_arm==4096)&(grid.query_shots_per_arm==2048)])
    lines=['# Reproduced results','', 'Balanced accuracy (%), mean ± sample SD across split seeds.','', '| Dataset | '+' | '.join(LABELS[m] for m in MODELS)+' |','|---|'+'---:|'*4]
    for d in DATASETS:
        g=summary[summary.dataset==d]
        if g.empty:continue
        cells=[]
        for m in MODELS:
            r=g[g.model==m].iloc[0];mean=f'{100*r.mean_balanced_accuracy:.2f}';sd='n/a' if np.isnan(r.sd_balanced_accuracy) else f'{100*r.sd_balanced_accuracy:.2f}'
            mean='**'+mean+'**' if r['rank']==1 else ('<ins>'+mean+'</ins>' if r['rank']==2 else (mean+'*' if r['rank']==3 else mean))
            cells.append(mean+' ± '+sd)
        lines.append('| '+d.title()+' | '+' | '.join(cells)+' |')
    lines+=['','Best: bold; second: underline; third: asterisk. Ranks use unrounded means.','', 'Fractions in CSV files are in [0,1], except fields ending in `_pp`, which are percentage points.','Repeated splits overlap; Student-t bands are descriptive, not validated population intervals or adjusted significance tests.','Finite-shot agreement compares sampled predictions to each model’s own exact predictions, not the labels.','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary
