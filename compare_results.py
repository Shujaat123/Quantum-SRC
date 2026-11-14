#!/usr/bin/env python3
"""Compare a fresh run with matching archived dataset/seed cases."""
from pathlib import Path
import argparse,json,sys
import numpy as np


def key(r):
    return (r['model'],r.get('occlusion_requested',0),r.get('repeat',0),r.get('gram_shots_per_arm',0),r.get('query_shots_per_arm',0))


def compare(results,reference):
    mismatches=[];counts=dict(cases=0,exact_records=0,sampled_records=0)
    for path in sorted((Path(results)/'runs').glob('*/result.json')):
        expected=Path(reference)/'runs'/path.parent.name/'result.json'
        if not expected.is_file():raise ValueError('No reference case for '+path.parent.name)
        actual=json.loads(path.read_text());old=json.loads(expected.read_text());counts['cases']+=1
        for stage in ['exact','sampled']:
            known={key(r):r for r in old[stage]}
            if stage=='exact' and {key(r) for r in actual[stage]}!=set(known):
                mismatches.append(dict(case=path.parent.name,stage=stage,issue='exact model/condition set differs'))
            # A clean-only run deliberately has no sampled rows. For sampled runs,
            # report only like-for-like matching shot settings.
            for r in actual[stage]:
                counts[stage+'_records']+=1;k=key(r)
                if k not in known:
                    mismatches.append(dict(case=path.parent.name,stage=stage,key=k,issue='no matching reference setting'));continue
                ref=known[k]
                for metric in ['accuracy','balanced_accuracy','macro_f1','agreement']:
                    if metric in ref and not np.isclose(r[metric],ref[metric],atol=1e-10,rtol=0):
                        mismatches.append(dict(case=path.parent.name,stage=stage,key=k,metric=metric,reference=ref[metric],actual=r[metric]))
                if r['predictions']!=ref['predictions']:
                    mismatches.append(dict(case=path.parent.name,stage=stage,key=k,issue='prediction vectors differ'))
    if not counts['cases']:raise ValueError('No result.json files found')
    return dict(**counts,matches=not mismatches,mismatches=mismatches)


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--reference',type=Path,default=root/'archived_results')
    a=p.parse_args();report=compare(a.results,a.reference);print(json.dumps(report,indent=2));sys.exit(0 if report['matches'] else 1)
