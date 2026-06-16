from __future__ import annotations

import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from dual_fw_svm import BinaryL2DualSVM, MulticlassFrankWolfeSVM
from dual_fw_svm.utils import project_nonnegative_balanced


class NewOptimizationSVMTests(unittest.TestCase):
    def test_projection_is_feasible(self):
        rng = np.random.default_rng(0)
        beta = rng.normal(size=100)
        y = np.r_[np.ones(45), -np.ones(55)]
        alpha = project_nonnegative_balanced(beta, y)
        self.assertGreaterEqual(float(alpha.min()), -1e-12)
        self.assertLess(abs(float(np.dot(alpha, y))), 1e-8)

    def test_binary_l2_dual_solver_fits_linear_data(self):
        X, y01 = make_classification(
            n_samples=220,
            n_features=12,
            n_informative=10,
            n_redundant=0,
            n_classes=2,
            class_sep=2.2,
            random_state=1,
        )
        y = np.where(y01 == 0, -1, 1)
        X = StandardScaler().fit_transform(X)
        model = BinaryL2DualSVM(C=1.0, max_iter=500, tol=1e-6, random_state=0).fit(X, y)
        self.assertLess(abs(model.balance_), 1e-6)
        self.assertGreater(model.score(X, y), 0.93)
        self.assertLess(model.dual_objective_, model.history_[0]["objective"])

    def test_binary_precomputed_kernel_path(self):
        X, y01 = make_classification(
            n_samples=120,
            n_features=8,
            n_informative=6,
            n_redundant=0,
            n_classes=2,
            class_sep=2.0,
            random_state=3,
        )
        y = np.where(y01 == 0, -1, 1)
        X = StandardScaler().fit_transform(X)
        K = X @ X.T
        model = BinaryL2DualSVM(
            C=1.0, kernel="precomputed", max_iter=250, tol=1e-5, random_state=0
        ).fit(K, y)
        self.assertGreater(model.score(K, y), 0.9)

    def test_multiclass_fw_solver_fits_and_respects_constraints(self):
        X, y = make_classification(
            n_samples=240,
            n_features=16,
            n_informative=12,
            n_redundant=0,
            n_classes=4,
            n_clusters_per_class=1,
            class_sep=2.0,
            random_state=2,
        )
        X = StandardScaler().fit_transform(X)
        model = MulticlassFrankWolfeSVM(
            C=0.5, formulation="cs", max_iter=250, tol=1e-4, random_state=0
        ).fit(X, y)
        self.assertLess(model.row_balance_max_, 1e-9)
        self.assertGreater(model.score(X, y), 0.78)

        alpha = model.alpha_
        rows = np.arange(y.size)
        self.assertTrue(np.all(alpha[rows, y] >= -1e-12))
        off = alpha.copy()
        off[rows, y] = 0.0
        self.assertTrue(np.all(off <= 1e-12))

    def test_multiclass_precomputed_kernel_path(self):
        X, y = make_classification(
            n_samples=120,
            n_features=10,
            n_informative=8,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            class_sep=2.0,
            random_state=4,
        )
        X = StandardScaler().fit_transform(X)
        K = X @ X.T
        model = MulticlassFrankWolfeSVM(
            C=0.5,
            formulation="cs",
            kernel="precomputed",
            max_iter=150,
            tol=1e-4,
            random_state=0,
        ).fit(K, y)
        self.assertGreater(model.score(K, y), 0.75)


if __name__ == "__main__":
    unittest.main()
