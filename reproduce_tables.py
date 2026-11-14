#!/usr/bin/env python3
"""Rebuild all tables from archived results or freshly generated experiments."""
from pathlib import Path
import argparse
from qsrc.tables import reproduce

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=root/'archived_results')
    p.add_argument('--output',type=Path,default=root/'tables')
    a=p.parse_args();s=reproduce(a.results,a.output)
    print(f'Rebuilt tables from {s.n_seeds.sum()} clean model/seed records in {a.output.resolve()}')
