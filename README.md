# Quantum SRC: experiment and table reproduction

**Shujaat Khan et al.**

CPU-based reproducibility code for Quantum SRC, Quantum SVM, quantum nearest
neighbor, and Original Linear SRC on Iris, Wine, Digits, Subspaces, Olivetti
Faces, and Electrical Grid Stability Simulated Data. The package includes the
archived result records and a single sequential experiment runner.

## Quick start

Use Python 3.12. In an existing environment such as `Py3126`:

```bash
python -m pip install -r requirements.txt
python reproduce_tables.py
```

The second command verifies the stored metrics against predictions and writes
all tables to `tables/`. It uses the included records, needs no dataset downloads,
and does not retrain models. Read `tables/RESULTS.md` for the ranked main table.

To rerun **all experiments sequentially** with the full settings:

```bash
python run_all.py --profile full
```

This runs six datasets with seeds 0–9, evaluates four methods, applies the face
occlusion conditions, runs the finite-shot grid on the four small datasets, and
reconstructs tables from the new results in `results/full/tables/`.

The script downloads Faces and Grid on first use. Iris, Wine, and Digits are
bundled with scikit-learn; Subspaces is generated with the specified seed.
Training and measurement experiments take substantially longer than rebuilding
the archived tables. Completed dataset/seed checkpoints are reused when the
identical command is rerun. Changes to configuration, dependencies, source code,
or input data require a new output directory.

## Software check and selected runs

```bash
python -m unittest discover -s tests -v
python run_all.py --profile smoke
```

The smoke profile uses reduced searches and seed 0 on Iris and Subspaces. Its
numbers are software checks, not the full-study results. A GitHub Actions
workflow runs the tests, archived-table reconstruction, and smoke profile.

To run full settings on a subset:

```bash
python run_all.py --profile full --datasets iris wine digits subspaces --seeds 0 1 --output results/small_subset
python run_all.py --profile full --datasets faces grid --stages clean --output results/faces_grid
```

`--stages clean` includes all clean comparisons and face occlusion. `--stages
shots` also computes the clean models needed as exact references. Finite-shot
experiments exist only for Iris, Wine, Digits, and Subspaces; Faces and Grid have
no finite-shot results in the original study.

Compare a fresh full-profile run against matching archived cases:

```bash
python compare_results.py --results results/full
```

Comparison checks individual predictions and metrics, not just rounded means.
It exits with a nonzero status if comparable records differ. Reduced smoke
settings should not be compared to the full reference scores.

## Datasets and offline operation

Download and cache external datasets separately if preferred:

```bash
python download_data.py
```

Dataset sources:

- Iris: <https://archive.ics.uci.edu/dataset/53/iris>
- Wine: <https://archive.ics.uci.edu/dataset/109/wine>
- Digits: scikit-learn's 1,797-image `load_digits` subset, with 64 features.
- Subspaces: three classes, each with 120 samples from a Gaussian two-dimensional
  subspace in 12 dimensions, plus Gaussian noise with standard deviation 0.04.
- Faces: scikit-learn's Olivetti dataset, 400 images of 40 identities, 64×64 pixels.
- Grid: <https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data>

No raw Faces or Grid data are committed. For an offline run, provide the original
Faces images in an NPZ with `X` (400×4096 or 400×64×64) and `y` (400 labels) in
the original scikit-learn ordering, and the official Grid CSV:

```bash
python run_all.py --profile full --faces-npz /path/to/faces.npz --grid-csv /path/to/Data_for_UCI_named.csv
```

On Windows, quote paths containing spaces. Dataset arrays are hashed and the
hashes are included in each run's metadata. Grid uses only `tau1..4`, `p1..4`,
and `g1..4`; both `stab` and `stabf` are excluded from predictors, and `stabf`
supplies the class label.

## Experiments reproduced

| Experiment | Datasets | Full settings |
|---|---|---|
| Clean comparisons | All six | Four methods, ten seeds |
| Paired method differences | All six | Identical dictionary/test indices within seed |
| Secondary outcomes | All six | Ordinary accuracy and macro-F1 |
| Face occlusion | Faces | Clean, 20×20 black square, 32×32 black square |
| Finite-shot kernel estimation | Iris, Wine, Digits, Subspaces | Three Gram budgets × four query budgets × two measurement repeats |
| Selected-map depth counts | All six | Validation-selected maps across ten seeds |
| Numerical solver diagnostics | All six | Stored KKT residuals |

The full archives contain 240 clean model evaluations, 80 additional occluded
face evaluations, and 1,920 finite-shot evaluations for the two retained
finite-shot decoders. Checkpoints include predictions and labels so table
reproduction can independently verify the reported classification metrics.

The two quantum comparators share the Quantum SRC-selected map. The small-data
map is first selected using `mu=[0.1, 0.5]`, then frozen while the follow-up
regularization search includes `mu=1`. These stages are deliberately separate.
The Original Linear SRC baseline retains its historical angle-scaled PCA
features and positive L2 penalty; it is not a raw-pixel SRC implementation.
Faces/Grid use the original extra-data search, while their linear baseline uses
the first six components of the same training-fitted preprocessing.

All original hyperparameter grids are in `configs/full.json`. Preprocessing is
fitted on the training partition, before dictionary subsampling. No validation
or test examples are used to fit it. Classifier parameters use validation
balanced accuracy only. Face occlusion changes test images only and keeps
clean-data parameters fixed.

Quantum kernels are evaluated by a six-qubit NumPy statevector simulator; no
QPU, Qiskit, or PennyLane installation is required. Finite shots are binomial
samples of ideal overlaps. They do not simulate gate or readout errors. The
same measured Gram/query matrices are shared by the finite-shot decoders;
the Gram is symmetrized and repaired by clipping negative eigenvalues.

## Tables and units

```bash
python reproduce_tables.py --results results/full --output results/full/tables
```

| Output | Contents |
|---|---|
| `RESULTS.md` | Ranked main table: best in bold, second underlined, third with an asterisk |
| `clean_results.csv` | Mean, sample SD, number of seeds, and rank |
| `per_seed_results.csv` | Clean and occlusion outcomes per seed |
| `paired_results.csv` | Paired Quantum SRC differences, descriptive intervals, win/tie/loss counts |
| `paired_seed_differences.csv` | Individual paired differences |
| `secondary_metrics.csv` | Ordinary accuracy and macro-F1 |
| `face_occlusion.csv` | Mean and SD at each actual occluded area |
| `finite_shot_grid.csv` | Full Gram/query budget grid, averaging repeats within seed first |
| `finite_shot_display.csv` | Summary at Gram budget 4,096 and query budget 2,048 |
| `finite_shot_results.csv` | Individual finite-shot outcomes and resource counts |
| `selected_depth_counts.csv` | Counts of selected circuit depths |
| `selected_parameters.csv` | Selected map parameters per dataset/seed |
| `protocol.csv` | Actual dictionary/validation/test sizes and selected dimension |
| `solver_diagnostics.csv` | KKT residuals for reconstruction models |

CSV accuracies are fractions in [0,1]; fields ending in `_pp` are percentage
points. The main metric is **balanced accuracy**, which differs slightly from
ordinary accuracy for Wine. Markdown percentages are rounded only for display.
Finite-shot agreement measures agreement with the same model's exact
predictions; it is distinct from accuracy against the labels. Across-seed SDs
and Student-t bands describe overlapping split variability and should not be
treated as independent-sample significance results. This is the retained
four-method comparison, not an exhaustive classical-baseline benchmark.

## Reproducibility environment

`requirements.txt` supports current compatible environments. The original full
result archives record Python 3.12.12, NumPy 1.26.4, SciPy 1.17.1, and
scikit-learn 1.9.1. Those recorded package versions are listed in
`requirements-reference.txt`; use a separate environment when installing them.
Pandas is used only for tabulation and threadpoolctl limits numerical-library
threads. Floating-point libraries, SVD implementations, and dependency changes
can affect tied validation selections; use `compare_results.py` to check a run
instead of assuming exact agreement.

`VALIDATION.md` states which fresh runs were actually checked while preparing
this package. The ten-seed result archives are included so the full tables are
reproducible even when a fresh experimental rerun is not performed.

## Repository layout

```text
run_all.py               Sequential experiment runner
download_data.py         External dataset cache helper
reproduce_tables.py      Archived/fresh result table reconstruction
compare_results.py       Prediction/metric comparison against reference runs
qsrc/                    Encoding, solver, preprocessing, searches, simulations
configs/                 Full and smoke protocols
archived_results/        Retained full-study per-seed records
tables/                  Reproduced CSV and Markdown tables
tests/                   Numerical, data-schema, and result-audit checks
```

## Attribution

**Shujaat Khan et al.** — Quantum SRC experiment reproducibility code.

Use the repository's `CITATION.cff` metadata when citing the software. No
unverified publication DOI or repository URL is assigned. Code is provided
under the included MIT license; external datasets retain their own terms.
