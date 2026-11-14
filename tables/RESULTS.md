# Reproduced results

Balanced accuracy (%), mean ± sample SD across split seeds.

| Dataset | Quantum SRC | Quantum SVM | Quantum nearest neighbor | Original Linear SRC |
|---|---:|---:|---:|---:|
| Iris | <ins>93.67</ins> ± 5.54 | **95.00** ± 3.24 | 90.00* ± 5.67 | 82.67 ± 7.83 |
| Wine | **96.33** ± 4.54 | <ins>96.06</ins> ± 3.24 | 93.96* ± 4.25 | 93.74 ± 5.41 |
| Digits | **79.33** ± 4.54 | <ins>78.92</ins> ± 3.79 | 75.17* ± 3.33 | 69.33 ± 4.95 |
| Subspaces | **93.89** ± 7.15 | 74.17 ± 10.96 | <ins>93.61</ins> ± 6.68 | 86.67* ± 7.15 |
| Faces | **79.75** ± 3.05 | <ins>78.25</ins> ± 2.90 | 76.88* ± 3.96 | 53.00 ± 5.57 |
| Grid | **74.01** ± 2.82 | <ins>73.50</ins> ± 2.37 | 65.22 ± 3.29 | 67.70* ± 3.88 |

Best: bold; second: underline; third: asterisk. Ranks use unrounded means.

Fractions in CSV files are in [0,1], except fields ending in `_pp`, which are percentage points.
Repeated splits overlap; Student-t bands are descriptive, not validated population intervals or adjusted significance tests.
Finite-shot agreement compares sampled predictions to each model’s own exact predictions, not the labels.
