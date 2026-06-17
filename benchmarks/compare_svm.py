"""Benchmark the package optimizers against common sklearn baselines."""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.datasets import load_digits, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from dual_fw_svm import (
    BinaryL2DualSVM,
    BlockCoordinateFrankWolfeSVM,
    MulticlassFrankWolfeSVM,
)


@dataclass
class Result:
    task: str
    method: str
    fit_time_s: float
    train_acc: float
    test_acc: float
    n_iter: int | str
    objective: float | str
    notes: str = ""


def _scale_split(X: np.ndarray, y: np.ndarray, *, test_size: float, random_state: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def _fit_result(
    task: str,
    method: str,
    factory: Callable[[], object],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    notes: str = "",
) -> Result:
    model = factory()
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        model.fit(X_train, y_train)
    elapsed = time.perf_counter() - started
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    n_iter = getattr(model, "n_iter_", "")
    if isinstance(n_iter, np.ndarray):
        n_iter = int(np.max(n_iter))
    if isinstance(n_iter, (int, np.integer)) and (n_iter < 0 or n_iter > 10_000_000):
        n_iter = ""
    obj = getattr(model, "dual_objective_", "")
    if isinstance(obj, (float, np.floating)):
        obj = float(obj)
    return Result(task, method, elapsed, train_acc, test_acc, n_iter, obj, notes)


def run_binary(args: argparse.Namespace) -> list[Result]:
    X, y = make_classification(
        n_samples=args.binary_samples,
        n_features=args.binary_features,
        n_informative=max(2, int(args.binary_features * 0.55)),
        n_redundant=max(0, int(args.binary_features * 0.15)),
        n_classes=2,
        class_sep=args.class_sep,
        random_state=args.random_state,
    )
    y = np.where(y == 0, -1, 1)
    X_train, X_test, y_train, y_test = _scale_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    results = [
        _fit_result(
            "binary",
            "l2-dual-prox",
            lambda: BinaryL2DualSVM(
                C=args.C,
                max_iter=args.binary_iter,
                tol=args.tol,
                accelerated=True,
                random_state=args.random_state,
            ),
            X_train,
            X_test,
            y_train,
            y_test,
            notes="dual proximal update, no Gram matrix",
        ),
        _fit_result(
            "binary",
            "sklearn-LinearSVC",
            lambda: LinearSVC(
                C=args.C / 2.0,
                loss="squared_hinge",
                dual="auto",
                fit_intercept=True,
                tol=args.tol,
                max_iter=args.sklearn_iter,
                random_state=args.random_state,
            ),
            X_train,
            X_test,
            y_train,
            y_test,
            notes="liblinear; C/2 matches this package's squared-slack scaling approximately",
        ),
        _fit_result(
            "binary",
            "sklearn-SGD-squared-hinge",
            lambda: SGDClassifier(
                loss="squared_hinge",
                alpha=1.0 / max(1.0, args.C * X_train.shape[0]),
                max_iter=args.sgd_iter,
                tol=args.tol,
                random_state=args.random_state,
            ),
            X_train,
            X_test,
            y_train,
            y_test,
            notes="first-order stochastic baseline",
        ),
    ]
    return results


def run_multiclass(args: argparse.Namespace) -> list[Result]:
    if args.multiclass_dataset == "digits":
        data = load_digits()
        X, y = data.data, data.target
    else:
        X, y = make_classification(
            n_samples=args.multi_samples,
            n_features=args.multi_features,
            n_informative=max(3, int(args.multi_features * 0.6)),
            n_redundant=max(0, int(args.multi_features * 0.1)),
            n_classes=args.multi_classes,
            n_clusters_per_class=1,
            class_sep=args.class_sep,
            random_state=args.random_state,
        )

    X_train, X_test, y_train, y_test = _scale_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    results = [
        _fit_result(
            "multiclass",
            "cs-matrix-FW",
            lambda: MulticlassFrankWolfeSVM(
                C=args.C,
                formulation="cs",
                max_iter=args.multi_iter,
                tol=args.fw_tol,
                random_state=args.random_state,
            ),
            X_train,
            X_test,
            y_train,
            y_test,
            notes="matrix-wise Crammer-Singer",
        ),
        _fit_result(
            "multiclass",
            "ww-matrix-FW",
            lambda: MulticlassFrankWolfeSVM(
                C=args.C,
                formulation="ww",
                max_iter=args.multi_iter,
                tol=args.fw_tol,
                random_state=args.random_state,
            ),
            X_train,
            X_test,
            y_train,
            y_test,
            notes="matrix-wise Weston-Watkins",
        ),
        _fit_result(
            "multiclass",
            "cs-block-FW",
            lambda: BlockCoordinateFrankWolfeSVM(
                C=args.C,
                formulation="cs",
                max_iter=args.block_iter,
                stepsize="optimal",
                random_state=args.random_state,
                record_every=max(100, args.block_iter // 20),
            ),
            X_train,
            X_test,
            y_train,
            y_test,
            notes="stochastic row-wise Frank-Wolfe",
        ),
        _fit_result(
            "multiclass",
            "sklearn-LinearSVC-CS",
            lambda: LinearSVC(
                C=args.C,
                multi_class="crammer_singer",
                fit_intercept=False,
                tol=args.tol,
                max_iter=args.sklearn_iter,
                random_state=args.random_state,
            ),
            X_train,
            X_test,
            y_train,
            y_test,
            notes="liblinear direct multiclass baseline",
        ),
        _fit_result(
            "multiclass",
            "sklearn-OVR-LinearSVC",
            lambda: OneVsRestClassifier(
                LinearSVC(
                    C=args.C / 2.0,
                    loss="squared_hinge",
                    dual="auto",
                    fit_intercept=False,
                    tol=args.tol,
                    max_iter=args.sklearn_iter,
                    random_state=args.random_state,
                )
            ),
            X_train,
            X_test,
            y_train,
            y_test,
            notes="one-vs-rest common production baseline",
        ),
        _fit_result(
            "multiclass",
            "sklearn-SGD-hinge",
            lambda: SGDClassifier(
                loss="hinge",
                alpha=1.0 / max(1.0, args.C * X_train.shape[0]),
                max_iter=args.sgd_iter,
                tol=args.tol,
                random_state=args.random_state,
            ),
            X_train,
            X_test,
            y_train,
            y_test,
            notes="stochastic large-scale baseline",
        ),
    ]
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--fw-tol", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--class-sep", type=float, default=1.2)
    parser.add_argument("--binary-samples", type=int, default=4000)
    parser.add_argument("--binary-features", type=int, default=80)
    parser.add_argument("--binary-iter", type=int, default=400)
    parser.add_argument("--multiclass-dataset", choices=["digits", "synthetic"], default="digits")
    parser.add_argument("--multi-samples", type=int, default=5000)
    parser.add_argument("--multi-features", type=int, default=80)
    parser.add_argument("--multi-classes", type=int, default=10)
    parser.add_argument("--multi-iter", type=int, default=300)
    parser.add_argument("--block-iter", type=int, default=5000)
    parser.add_argument("--sklearn-iter", type=int, default=5000)
    parser.add_argument("--sgd-iter", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results_latest.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_binary(args) + run_multiclass(args)
    df = pd.DataFrame([r.__dict__ for r in results])
    df = df.sort_values(["task", "fit_time_s"], kind="stable").reset_index(drop=True)
    df["fit_time_s"] = df["fit_time_s"].map(lambda x: round(float(x), 4))
    df["train_acc"] = df["train_acc"].map(lambda x: round(float(x), 4))
    df["test_acc"] = df["test_acc"].map(lambda x: round(float(x), 4))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
