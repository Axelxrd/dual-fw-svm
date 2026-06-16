"""Binary L2-SVM via the paper's dual-of-dual proximal method."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import Literal

import numpy as np

from .utils import (
    as_float64_matrix,
    decode_binary_scores,
    encode_binary_labels,
    matvec,
    project_nonnegative_balanced,
    spectral_norm_squared,
    squared_norm,
    symmetric_operator_norm,
)


@dataclass
class BinaryL2DualSVM:
    """Binary L2-loss SVM solver based on Algorithm 1 in the paper.

    The implementation targets large linear problems and never forms the
    n-by-n Gram matrix. It solves

        min 0.5 alpha^T K alpha + 1/(2C)||alpha||^2 - 1^T alpha
        s.t. alpha >= 0, y^T alpha = 0.

    Parameters follow the paper's primal scaling
    0.5||w||^2 + C/2 * sum_i xi_i^2.
    """

    C: float = 1.0
    kernel: Literal["linear", "precomputed"] = "linear"
    max_iter: int = 1000
    tol: float = 1e-5
    lipschitz: float | None = None
    spectral_iter: int = 80
    projection_iter: int = 50
    accelerated: bool = True
    monotone: bool = True
    backtracking: bool = True
    stationarity_every: int = 10
    random_state: int | None = 0
    verbose: bool = False
    history_: list[dict[str, float]] = field(default_factory=list, init=False)

    def fit(self, X: Any, y: Any) -> "BinaryL2DualSVM":
        if self.C <= 0:
            raise ValueError("C must be positive.")
        if self.kernel not in {"linear", "precomputed"}:
            raise ValueError("kernel must be 'linear' or 'precomputed'.")

        X = as_float64_matrix(X)
        y_signed, classes = encode_binary_labels(y)
        n_samples, n_features = X.shape
        if n_samples != y_signed.size:
            raise ValueError("X and y have incompatible lengths.")
        if self.kernel == "precomputed" and X.shape[0] != X.shape[1]:
            raise ValueError("For kernel='precomputed', X must be a square train Gram matrix.")

        if self.lipschitz is None:
            if self.kernel == "linear":
                kernel_lipschitz = spectral_norm_squared(
                    X,
                    max_iter=self.spectral_iter,
                    random_state=self.random_state,
                )
            else:
                kernel_lipschitz = symmetric_operator_norm(
                    X,
                    max_iter=self.spectral_iter,
                    random_state=self.random_state,
                )
            L = kernel_lipschitz + 1.0 / self.C
        else:
            L = float(self.lipschitz)
        if not np.isfinite(L) or L <= 0.0:
            L = 1.0 / self.C

        alpha = np.zeros(n_samples, dtype=np.float64)
        momentum = alpha.copy()
        t = 1.0
        obj, w = self._objective_and_weight(X, y_signed, alpha)
        self.history_ = [{"iter": 0.0, "objective": obj, "step": np.inf, "L": L}]

        for it in range(1, int(self.max_iter) + 1):
            old_alpha = alpha
            old_obj = obj
            base = momentum if self.accelerated else old_alpha

            candidate, cand_obj, cand_w, L = self._prox_step(
                X, y_signed, base, L, old_obj if self.monotone else None
            )

            if self.monotone and cand_obj > old_obj + 1e-12 * max(1.0, abs(old_obj)):
                # Momentum can raise the objective; retry from the last accepted alpha.
                candidate, cand_obj, cand_w, L = self._prox_step(
                    X, y_signed, old_alpha, L, old_obj
                )
                t = 1.0

            step = np.linalg.norm(candidate - old_alpha) / max(1.0, np.linalg.norm(old_alpha))
            alpha = candidate
            obj = cand_obj
            w = cand_w

            if self.accelerated:
                t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
                momentum = alpha + ((t - 1.0) / t_next) * (alpha - old_alpha)
                # Gradient restart for stability on ill-conditioned data.
                if np.dot(alpha - old_alpha, momentum - alpha) > 0.0:
                    momentum = alpha.copy()
                    t_next = 1.0
                t = t_next
            else:
                momentum = alpha

            self.history_.append(
                {
                    "iter": float(it),
                    "objective": obj,
                    "step": step,
                    "pg_step": np.nan,
                    "L": L,
                }
            )

            pg_step = np.nan
            check_stationarity = (
                it == 1
                or it % max(1, self.stationarity_every) == 0
                or step <= self.tol
            )
            if check_stationarity:
                # Measure stationarity on the accepted iterate, not the
                # extrapolated point, to avoid FISTA momentum masking progress.
                current_grad = self._gradient(X, y_signed, alpha)
                current_beta = alpha - current_grad / L
                current_proj = project_nonnegative_balanced(
                    current_beta,
                    y_signed,
                    max_iter=self.projection_iter,
                )
                pg_step = np.linalg.norm(current_proj - alpha) / max(1.0, np.linalg.norm(alpha))
                self.history_[-1]["pg_step"] = float(pg_step)
            if self.verbose and (it == 1 or it % 25 == 0):
                print(
                    f"iter={it} objective={obj:.6g} step={step:.3g} "
                    f"pg={pg_step:.3g} L={L:.6g}"
                )
            if step <= self.tol or (check_stationarity and pg_step <= self.tol):
                break

        scores_no_bias = matvec(X, w)
        active_tol = max(1e-10, 1e-7 * max(1.0, float(np.max(alpha))))
        active = alpha > active_tol
        if np.any(active):
            b_values = y_signed[active] * (1.0 - alpha[active] / self.C) - scores_no_bias[active]
            intercept = float(np.median(b_values))
        else:
            intercept = 0.0

        self.alpha_ = alpha
        self.dual_coef_ = (y_signed * alpha).reshape(1, n_samples)
        self.coef_ = w.reshape(1, n_features) if self.kernel == "linear" else None
        self.intercept_ = np.array([intercept], dtype=np.float64)
        self.classes_ = classes
        self.n_features_in_ = n_features
        self.n_iter_ = len(self.history_) - 1
        self.lipschitz_ = L
        self.dual_objective_ = obj
        self.balance_ = float(np.dot(y_signed, alpha))
        return self

    def _prox_step(
        self,
        X: Any,
        y_signed: np.ndarray,
        base: np.ndarray,
        L: float,
        max_objective: float | None,
    ) -> tuple[np.ndarray, float, np.ndarray, float]:
        trials = 30 if self.backtracking else 1
        for _ in range(trials):
            grad = self._gradient(X, y_signed, base)
            beta = base - grad / L
            candidate = project_nonnegative_balanced(
                beta,
                y_signed,
                max_iter=self.projection_iter,
            )
            obj, w = self._objective_and_weight(X, y_signed, candidate)
            if max_objective is None or obj <= max_objective + 1e-12 * max(1.0, abs(max_objective)):
                return candidate, obj, w, L
            L *= 2.0

        return candidate, obj, w, L

    def _gradient(self, X: Any, y_signed: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        if self.kernel == "linear":
            w = matvec(X.T, y_signed * alpha)
            return y_signed * matvec(X, w) + alpha / self.C - 1.0

        return y_signed * matvec(X, y_signed * alpha) + alpha / self.C - 1.0

    def _objective_and_weight(
        self, X: Any, y_signed: np.ndarray, alpha: np.ndarray
    ) -> tuple[float, np.ndarray]:
        if self.kernel == "linear":
            w = matvec(X.T, y_signed * alpha)
            obj = 0.5 * squared_norm(w) + 0.5 / self.C * squared_norm(alpha) - float(
                np.sum(alpha)
            )
            return obj, w

        dual_coef = y_signed * alpha
        obj = (
            0.5 * float(np.dot(dual_coef, matvec(X, dual_coef)))
            + 0.5 / self.C * squared_norm(alpha)
            - float(np.sum(alpha))
        )
        return obj, dual_coef

    def decision_function(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        X = as_float64_matrix(X)
        if self.kernel == "linear":
            return matvec(X, self.coef_.reshape(-1)) + float(self.intercept_[0])
        if X.shape[1] != self.dual_coef_.shape[1]:
            raise ValueError(
                "For kernel='precomputed', prediction input must have shape "
                "(n_test, n_train)."
            )
        return matvec(X, self.dual_coef_.reshape(-1)) + float(self.intercept_[0])

    def predict(self, X: Any) -> np.ndarray:
        return decode_binary_scores(self.decision_function(X), self.classes_)

    def score(self, X: Any, y: Any) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "alpha_"):
            raise RuntimeError("Call fit before prediction.")
