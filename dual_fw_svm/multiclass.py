"""Matrix-wise and block-wise Frank-Wolfe solvers for multiclass SVMs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .utils import (
    as_float64_matrix,
    encode_multiclass_labels,
    matmul,
    rank_one_add_to_w,
    row_norm_squared,
    squared_norm,
)

Formulation = Literal["cs", "ww"]


@dataclass
class MulticlassFrankWolfeSVM:
    """Matrix-wise Frank-Wolfe solver for Crammer-Singer or Weston-Watkins SVM.

    This is Algorithm 2 from the paper. For linear features it stores only
    alpha (n x K) and W = X.T @ alpha, avoiding the n-by-n kernel matrix and the
    huge Kronecker QP matrix.
    """

    C: float = 1.0
    formulation: Formulation = "cs"
    kernel: Literal["linear", "precomputed"] = "linear"
    max_iter: int = 500
    tol: float = 1e-4
    random_state: int | None = 0
    record_every: int = 1
    verbose: bool = False
    history_: list[dict[str, float]] = field(default_factory=list, init=False)

    def fit(self, X: Any, y: Any) -> "MulticlassFrankWolfeSVM":
        if self.C <= 0:
            raise ValueError("C must be positive.")
        if self.formulation not in {"cs", "ww"}:
            raise ValueError("formulation must be 'cs' or 'ww'.")
        if self.kernel not in {"linear", "precomputed"}:
            raise ValueError("kernel must be 'linear' or 'precomputed'.")

        X = as_float64_matrix(X)
        labels = encode_multiclass_labels(y)
        y_enc = labels.encoded
        n_samples, n_features = X.shape
        if self.kernel == "precomputed" and X.shape[0] != X.shape[1]:
            raise ValueError("For kernel='precomputed', X must be a square train Gram matrix.")
        n_classes = labels.classes.size
        rows = np.arange(n_samples)

        alpha = np.zeros((n_samples, n_classes), dtype=np.float64)
        W = np.zeros((n_features, n_classes), dtype=np.float64) if self.kernel == "linear" else None
        K_alpha = np.zeros_like(alpha) if self.kernel == "precomputed" else None
        self.history_ = []

        obj = self._objective_linear(W, alpha, y_enc) if self.kernel == "linear" else self._objective_kernel(K_alpha, alpha, y_enc)
        gap = np.inf
        for it in range(int(self.max_iter) + 1):
            grad = self._gradient(X, W, K_alpha, y_enc)
            s = self._linear_minimizer(grad, y_enc)
            d = s - alpha
            gap = float(-np.sum(d * grad))
            rel_gap = gap / max(1.0, abs(obj))

            if it % max(1, self.record_every) == 0:
                self.history_.append(
                    {
                        "iter": float(it),
                        "objective": obj,
                        "fw_gap": gap,
                        "relative_gap": rel_gap,
                    }
                )
            if self.verbose and (it == 0 or it % 25 == 0):
                print(f"iter={it} objective={obj:.6g} gap={gap:.3g}")
            if gap <= self.tol * max(1.0, abs(obj)) or it == int(self.max_iter):
                break

            if self.kernel == "linear":
                dW = matmul(X.T, d)
                K_d = None
                denom = squared_norm(dW)
            else:
                dW = None
                K_d = matmul(X, d)
                denom = float(np.sum(d * K_d))
            if gap <= 0.0:
                gamma = 0.0
            elif denom <= 1e-18:
                gamma = 1.0
            else:
                gamma = min(1.0, gap / denom)

            alpha += gamma * d
            if self.kernel == "linear":
                W += gamma * dW
                obj = self._objective_linear(W, alpha, y_enc)
            else:
                K_alpha += gamma * K_d
                obj = self._objective_kernel(K_alpha, alpha, y_enc)

        self.alpha_ = alpha
        self.W_ = W
        self.K_alpha_ = K_alpha
        self.coef_ = W.T if self.kernel == "linear" else None
        self.intercept_ = np.zeros(n_classes, dtype=np.float64)
        self.classes_ = labels.classes
        self.n_iter_ = it
        self.dual_objective_ = obj
        self.fw_gap_ = gap
        self.row_balance_max_ = float(np.max(np.abs(alpha.sum(axis=1))))
        self._rows_ = rows
        return self

    def _gradient(
        self,
        X: Any,
        W: np.ndarray | None,
        K_alpha: np.ndarray | None,
        y_enc: np.ndarray,
    ) -> np.ndarray:
        grad = matmul(X, W) if self.kernel == "linear" else np.array(K_alpha, copy=True)
        grad[np.arange(y_enc.size), y_enc] -= 1.0
        return grad

    def _linear_minimizer(self, grad: np.ndarray, y_enc: np.ndarray) -> np.ndarray:
        if self.formulation == "cs":
            return _lmo_crammer_singer(grad, y_enc, self.C)
        return _lmo_weston_watkins(grad, y_enc, self.C)

    @staticmethod
    def _objective_linear(W: np.ndarray, alpha: np.ndarray, y_enc: np.ndarray) -> float:
        return 0.5 * squared_norm(W) - float(np.sum(alpha[np.arange(y_enc.size), y_enc]))

    @staticmethod
    def _objective_kernel(K_alpha: np.ndarray, alpha: np.ndarray, y_enc: np.ndarray) -> float:
        return 0.5 * float(np.sum(alpha * K_alpha)) - float(
            np.sum(alpha[np.arange(y_enc.size), y_enc])
        )

    def decision_function(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        X = as_float64_matrix(X)
        if self.kernel == "linear":
            return matmul(X, self.W_)
        if X.shape[1] != self.alpha_.shape[0]:
            raise ValueError(
                "For kernel='precomputed', prediction input must have shape "
                "(n_test, n_train)."
            )
        return matmul(X, self.alpha_)

    def predict(self, X: Any) -> np.ndarray:
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X: Any, y: Any) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "alpha_"):
            raise RuntimeError("Call fit before prediction.")


@dataclass
class BlockCoordinateFrankWolfeSVM:
    """Stochastic row-wise FW baseline corresponding to Algorithm 3."""

    C: float = 1.0
    formulation: Formulation = "cs"
    kernel: Literal["linear", "precomputed"] = "linear"
    max_iter: int = 5000
    tol: float = 0.0
    stepsize: Literal["optimal", "oblivious"] = "optimal"
    random_state: int | None = 0
    record_every: int = 100
    verbose: bool = False
    history_: list[dict[str, float]] = field(default_factory=list, init=False)

    def fit(self, X: Any, y: Any) -> "BlockCoordinateFrankWolfeSVM":
        if self.C <= 0:
            raise ValueError("C must be positive.")
        if self.formulation not in {"cs", "ww"}:
            raise ValueError("formulation must be 'cs' or 'ww'.")
        if self.kernel not in {"linear", "precomputed"}:
            raise ValueError("kernel must be 'linear' or 'precomputed'.")
        if self.stepsize not in {"optimal", "oblivious"}:
            raise ValueError("stepsize must be 'optimal' or 'oblivious'.")

        X = as_float64_matrix(X)
        labels = encode_multiclass_labels(y)
        y_enc = labels.encoded
        n_samples, n_features = X.shape
        if self.kernel == "precomputed" and X.shape[0] != X.shape[1]:
            raise ValueError("For kernel='precomputed', X must be a square train Gram matrix.")
        n_classes = labels.classes.size
        rng = np.random.default_rng(self.random_state)

        alpha = np.zeros((n_samples, n_classes), dtype=np.float64)
        W = np.zeros((n_features, n_classes), dtype=np.float64) if self.kernel == "linear" else None
        K_alpha = np.zeros_like(alpha) if self.kernel == "precomputed" else None
        self.history_ = []

        for it in range(int(self.max_iter) + 1):
            if it % max(1, self.record_every) == 0:
                obj = (
                    MulticlassFrankWolfeSVM._objective_linear(W, alpha, y_enc)
                    if self.kernel == "linear"
                    else MulticlassFrankWolfeSVM._objective_kernel(K_alpha, alpha, y_enc)
                )
                self.history_.append({"iter": float(it), "objective": obj})
                if self.verbose:
                    print(f"iter={it} objective={obj:.6g}")

            if it == int(self.max_iter):
                break

            i = int(rng.integers(0, n_samples))
            if self.kernel == "linear":
                q = np.asarray(X[i] @ W, dtype=np.float64).reshape(-1)
            else:
                q = np.asarray(K_alpha[i], dtype=np.float64).reshape(-1)
            q[y_enc[i]] -= 1.0
            s_i = (
                _lmo_crammer_singer_row(q, int(y_enc[i]), self.C)
                if self.formulation == "cs"
                else _lmo_weston_watkins_row(q, int(y_enc[i]), self.C)
            )
            d_i = s_i - alpha[i]
            gap_i = float(-np.dot(d_i, q))
            if gap_i <= self.tol:
                continue

            if self.stepsize == "oblivious":
                gamma = min(1.0, 2.0 * n_samples / (it + 2.0 * n_samples))
            else:
                if self.kernel == "linear":
                    denom = row_norm_squared(X, i) * float(np.dot(d_i, d_i))
                else:
                    denom = float(X[i, i]) * float(np.dot(d_i, d_i))
                if denom <= 1e-18:
                    gamma = 1.0
                else:
                    gamma = min(1.0, gap_i / denom)

            alpha[i] += gamma * d_i
            if self.kernel == "linear":
                rank_one_add_to_w(W, X, i, d_i, gamma)
            else:
                K_alpha += gamma * matmul(X[:, [i]], d_i[None, :])

        self.alpha_ = alpha
        self.W_ = W
        self.K_alpha_ = K_alpha
        self.coef_ = W.T if self.kernel == "linear" else None
        self.intercept_ = np.zeros(n_classes, dtype=np.float64)
        self.classes_ = labels.classes
        self.n_iter_ = it
        self.dual_objective_ = (
            MulticlassFrankWolfeSVM._objective_linear(W, alpha, y_enc)
            if self.kernel == "linear"
            else MulticlassFrankWolfeSVM._objective_kernel(K_alpha, alpha, y_enc)
        )
        self.row_balance_max_ = float(np.max(np.abs(alpha.sum(axis=1))))
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        X = as_float64_matrix(X)
        if self.kernel == "linear":
            return matmul(X, self.W_)
        if X.shape[1] != self.alpha_.shape[0]:
            raise ValueError(
                "For kernel='precomputed', prediction input must have shape "
                "(n_test, n_train)."
            )
        return matmul(X, self.alpha_)

    def predict(self, X: Any) -> np.ndarray:
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X: Any, y: Any) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "alpha_"):
            raise RuntimeError("Call fit before prediction.")


def _lmo_crammer_singer(grad: np.ndarray, y_enc: np.ndarray, C: float) -> np.ndarray:
    n_samples, n_classes = grad.shape
    rows = np.arange(n_samples)
    q_y = grad[rows, y_enc]
    j_max = np.argmax(grad, axis=1)
    q_max = grad[rows, j_max]
    improve = q_max > q_y

    s = np.zeros((n_samples, n_classes), dtype=np.float64)
    active_rows = rows[improve]
    s[active_rows, y_enc[improve]] = C
    s[active_rows, j_max[improve]] -= C
    return s


def _lmo_weston_watkins(grad: np.ndarray, y_enc: np.ndarray, C: float) -> np.ndarray:
    n_samples, n_classes = grad.shape
    rows = np.arange(n_samples)
    q_y = grad[rows, y_enc]
    take = grad > q_y[:, None]
    take[rows, y_enc] = False

    s = np.zeros((n_samples, n_classes), dtype=np.float64)
    s[take] = -C
    s[rows, y_enc] = -np.sum(s, axis=1)
    return s


def _lmo_crammer_singer_row(q: np.ndarray, y_i: int, C: float) -> np.ndarray:
    j_max = int(np.argmax(q))
    s = np.zeros_like(q, dtype=np.float64)
    if q[j_max] > q[y_i]:
        s[y_i] = C
        s[j_max] -= C
    return s


def _lmo_weston_watkins_row(q: np.ndarray, y_i: int, C: float) -> np.ndarray:
    s = np.zeros_like(q, dtype=np.float64)
    take = q > q[y_i]
    take[y_i] = False
    s[take] = -C
    s[y_i] = -float(np.sum(s))
    return s
