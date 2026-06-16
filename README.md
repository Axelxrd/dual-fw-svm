# dual-fw-svm

[![PyPI](https://img.shields.io/pypi/v/dual-fw-svm.svg)](https://pypi.org/project/dual-fw-svm/)
[![Python](https://img.shields.io/pypi/pyversions/dual-fw-svm.svg)](https://pypi.org/project/dual-fw-svm/)
[![Source](https://img.shields.io/badge/source-GitHub-24292f.svg)](https://github.com/1want2beaQuant/dual-fw-svm)

Fast linear SVM solvers that reproduce the dual proximal and matrix-wise
Frank-Wolfe algorithms from *New Optimization Methods for Large Scale SVMs*.
The implementation is designed for direct experimentation with the paper's
optimization ideas while keeping memory use low on linear datasets.

## What Is Included

- `BinaryL2DualSVM`: Algorithm 1, a proximal-gradient solver for binary L2-SVM
  dual variables with the paper's second-dual projection step.
- `MulticlassFrankWolfeSVM`: Algorithm 2, matrix-wise Frank-Wolfe for
  Crammer-Singer (`formulation="cs"`) and Weston-Watkins (`formulation="ww"`)
  multiclass SVMs.
- `BlockCoordinateFrankWolfeSVM`: Algorithm 3 style stochastic row-wise FW
  baseline for comparison with the paper's matrix-wise update.
- `benchmarks/compare_svm.py`: compares the reproduction with common sklearn
  baselines: `LinearSVC`, `LinearSVC(multi_class="crammer_singer")`,
  one-vs-rest `LinearSVC`, and `SGDClassifier`.

## Why It Is Fast

The default linear solvers avoid materializing the large Gram matrix. Binary
training uses `X.T @ (y * alpha)` and multiclass training keeps
`W = X.T @ alpha`, which is the main speed and memory choice for large
datasets. For small custom-kernel experiments, both main solvers also accept
`kernel="precomputed"` with a train Gram matrix during `fit` and a
test-by-train kernel matrix during prediction.

## Install

```powershell
pip install dual-fw-svm
```

For benchmark and test dependencies:

```powershell
pip install "dual-fw-svm[benchmark,test]"
```

## Quick Start

```python
from dual_fw_svm import BinaryL2DualSVM, MulticlassFrankWolfeSVM

binary = BinaryL2DualSVM(C=1.0, max_iter=1000, tol=1e-5)
binary.fit(X_train, y_train)
binary_pred = binary.predict(X_test)

multi = MulticlassFrankWolfeSVM(C=1.0, formulation="cs", max_iter=500)
multi.fit(X_train, y_train)
multi_pred = multi.predict(X_test)
```

## Precomputed Kernels

```python
from dual_fw_svm import BinaryL2DualSVM

K_train = X_train @ X_train.T
K_test = X_test @ X_train.T

model = BinaryL2DualSVM(C=1.0, kernel="precomputed")
model.fit(K_train, y_train)
pred = model.predict(K_test)
```

## Benchmark Snapshot

Current local benchmark results on synthetic binary data and sklearn digits:

| Task | Method | Fit time | Test accuracy |
| --- | --- | ---: | ---: |
| binary | sklearn LinearSVC | 0.0116s | 0.8617 |
| binary | paper L2 dual prox | 0.2297s | 0.8633 |
| multiclass | paper CS matrix-FW | 0.0762s | 0.9593 |
| multiclass | paper WW matrix-FW | 0.0796s | 0.9537 |
| multiclass | sklearn LinearSVC CS | 0.1349s | 0.9537 |
| multiclass | sklearn SGD hinge | 0.4333s | 0.9537 |

The binary solver is a transparent Python reproduction and is not expected to
beat LIBLINEAR on small dense problems. The matrix-wise multiclass solver is the
main speed-oriented implementation.

## Development

Run tests:

```powershell
python -m unittest discover -s tests
```

Run the benchmark:

```powershell
python benchmarks/compare_svm.py
```

The benchmark writes:

```text
benchmarks/results_latest.csv
```

## Notes

- `BinaryL2DualSVM.C` follows the paper's scaling:
  `0.5 ||w||^2 + C/2 * sum_i xi_i^2`.
- The multiclass implementations follow the no-bias formulation in the paper.
  Standardizing dense features before fitting is recommended for faster
  convergence and fair comparison.
- The benchmark uses `LinearSVC(C=C/2, loss="squared_hinge")` for the binary
  sklearn baseline because sklearn's squared-hinge objective uses a slightly
  different constant factor.
