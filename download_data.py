#!/usr/bin/env python3
"""Cache public Faces/Grid datasets; accept local files for offline operation."""
from pathlib import Path
import argparse,json
from qsrc.data_extra import load

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',choices=['faces','grid'],default=['faces','grid'])
    p.add_argument('--cache',type=Path,default=Path(__file__).resolve().parent/'data_cache')
    p.add_argument('--faces-npz',type=Path);p.add_argument('--grid-csv',type=Path)
    a=p.parse_args()
    for name in a.datasets:
        _,_,meta=load(name,a.cache,a.faces_npz if name=='faces' else a.grid_csv,40)
        print(name,json.dumps(meta,indent=2))
