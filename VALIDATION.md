# Validation

The package was checked on Python 3.12.14 with NumPy 2.3.5, SciPy 1.17.0,
scikit-learn 1.8.0, and pandas 2.2.3.

- Eight numerical, data-schema, and archived-result tests passed.
- Recomputed all 320 archived clean/occlusion records from predictions and
  labels, checking ordinary accuracy, balanced accuracy, and macro-F1.
- Recomputed all 1,920 archived finite-shot records, including agreement with
  each model's exact predictions on the same selected queries.
- All 24 main-table means and SDs matched the supplied reference within 1e-11.
- Fresh seed-0 experiments used the **full hyperparameter settings on all six
  datasets**. All 32 exact clean/occlusion prediction vectors and all 192
  finite-shot prediction vectors matched their archived counterparts. Metrics
  matched within 1e-10.
- Official Faces and Grid downloads succeeded; their numerical dataset hashes
  matched the hashes in the original experiment archives.
- The smoke run and a repeated identical command checked checkpoint reuse.
- Python compilation checks passed.

The full ten-seed training suite was not rerun while preparing this package.
Its complete archived outcomes are included, audited, and sufficient to rebuild
all tables. `validation.json` records the checks numerically.

Run the tests locally with:

```bash
python -m unittest discover -s tests -v
python run_all.py --profile smoke
```
