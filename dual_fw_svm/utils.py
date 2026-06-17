"""Shared numerical utilities for the SVM solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # SciPy is present in the target environment, but keep imports defensive.
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None


def is_sparse_matrix(x: Any) -> bool:
    return sp is not None and sp.issparse(x)


def as_float64_matrix(x: Any, *, copy: bool = False):
    """Return a dense ndarray or CSR sparse matrix with float64 dtype."""

    if is_sparse_matrix(x):
        out = x.tocsr(copy=copy)
        if out.dtype != np.float64:
            out = out.astype(np.float64)
        return out

    return np.array(x, dtype=np.float64, copy=copy)


def as_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def matmul(x: Any, y: Any) -> np.ndarray:
    return np.asarray(x @ y, dtype=np.float64)


def matvec(x: Any, y: Any) -> np.ndarray:
    return as_1d(x @ y)


def squared_norm(x: Any) -> float:
    arr = np.asarray(x, dtype=np.float64)
    return float(np.sum(arr * arr))


def spectral_norm_squared(
    x: Any,
    *,
    max_iter: int = 80,
    tol: float = 1e-6,
    random_state: int | None = 0,
) -> float:
    """Estimate ||X||_2^2 with power iteration.

    The binary L2 solver needs sigma_1(X X^T). For a linear kernel this is
    exactly ||X||_2^2, so we estimate it without materializing the Gram matrix.
    """

    n_features = x.shape[1]
    if n_features == 0 or x.shape[0] == 0:
        return 0.0

    rng = np.random.default_rng(random_state)
    v = rng.normal(size=n_features)
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return 0.0
    v /= norm

    last = 0.0
    estimate = 0.0
    for _ in range(max(1, int(max_iter))):
        xv = matvec(x, v)
        estimate = float(np.dot(xv, xv))
        xtxv = matvec(x.T, xv)
        norm = np.linalg.norm(xtxv)
        if norm == 0.0:
            return 0.0
        v = xtxv / norm
        if abs(estimate - last) <= tol * max(1.0, abs(estimate)):
            break
        last = estimate

    return max(0.0, estimate)


def symmetric_operator_norm(
    x: Any,
    *,
    max_iter: int = 80,
    tol: float = 1e-6,
    random_state: int | None = 0,
) -> float:
    """Estimate the spectral norm of a symmetric matrix-like operator."""

    n = x.shape[0]
    if n == 0:
        return 0.0

    rng = np.random.default_rng(random_state)
    v = rng.normal(size=n)
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return 0.0
    v /= norm

    last = 0.0
    estimate = 0.0
    for _ in range(max(1, int(max_iter))):
        av = matvec(x, v)
        estimate = float(np.linalg.norm(av))
        if estimate == 0.0:
            return 0.0
        v = av / estimate
        if abs(estimate - last) <= tol * max(1.0, abs(estimate)):
            break
        last = estimate

    return max(0.0, estimate)


def encode_binary_labels(y: Any) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y)
    classes = np.unique(labels)
    if classes.size != 2:
        raise ValueError(f"Binary L2-SVM needs exactly 2 classes, got {classes.size}.")

    if set(classes.tolist()) == {-1, 1}:
        signed = labels.astype(np.float64)
        ordered = np.array([-1, 1], dtype=classes.dtype)
        return signed, ordered

    signed = np.where(labels == classes[0], -1.0, 1.0)
    return signed.astype(np.float64), classes


def decode_binary_scores(scores: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return np.where(scores >= 0.0, classes[1], classes[0])


@dataclass(frozen=True)
class MulticlassLabels:
    encoded: np.ndarray
    classes: np.ndarray


def encode_multiclass_labels(y: Any) -> MulticlassLabels:
    labels = np.asarray(y)
    classes, encoded = np.unique(labels, return_inverse=True)
    if classes.size < 2:
        raise ValueError("Multiclass SVM needs at least 2 classes.")
    return MulticlassLabels(encoded=encoded.astype(np.int64), classes=classes)


def project_nonnegative_balanced(
    beta: np.ndarray,
    y_signed: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-12,
) -> np.ndarray:
    """Project beta onto {alpha >= 0, <alpha, y> = 0}.

    The projection has the form alpha = [beta - lambda y]_+. The scalar lambda
    is found by bisection because the balance equation is monotone decreasing.
    """

    beta = np.asarray(beta, dtype=np.float64)
    y_signed = np.asarray(y_signed, dtype=np.float64)
    pos = y_signed > 0.0
    neg = y_signed < 0.0
    if not np.any(pos) or not np.any(neg):
        raise ValueError("Both positive and negative labels are required.")

    max_abs = float(np.max(np.abs(beta))) if beta.size else 0.0
    margin = max(1.0, max_abs)
    lo = min(float(np.min(beta[pos])), float(-np.max(beta[neg]))) - margin
    hi = max(float(np.max(beta[pos])), float(-np.min(beta[neg]))) + margin

    def balance(lam: float) -> float:
        alpha = np.maximum(beta - lam * y_signed, 0.0)
        return float(np.dot(y_signed, alpha))

    # Expand defensively for extreme floating-point input.
    for _ in range(20):
        if balance(lo) >= 0.0:
            break
        lo -= max(1.0, hi - lo)
    for _ in range(20):
        if balance(hi) <= 0.0:
            break
        hi += max(1.0, hi - lo)

    lam = 0.0
    for _ in range(max(1, int(max_iter))):
        lam = 0.5 * (lo + hi)
        val = balance(lam)
        if abs(val) <= tol:
            break
        if val > 0.0:
            lo = lam
        else:
            hi = lam

    return np.maximum(beta - lam * y_signed, 0.0)


def row_norm_squared(x: Any, i: int) -> float:
    if is_sparse_matrix(x):
        row = x.getrow(i)
        return float(row.multiply(row).sum())
    row = np.asarray(x[i], dtype=np.float64)
    return float(np.dot(row, row))


def rank_one_add_to_w(w: np.ndarray, x: Any, i: int, d: np.ndarray, scale: float) -> None:
    """Apply W += scale * outer(X[i], d), including sparse rows."""

    if scale == 0.0:
        return
    if is_sparse_matrix(x):
        row = x.getrow(i)
        if row.nnz:
            w[row.indices, :] += (scale * row.data)[:, None] * d[None, :]
        return
    w += scale * np.outer(np.asarray(x[i], dtype=np.float64), d)
