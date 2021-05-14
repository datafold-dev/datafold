""" Unit test for the jsf module.

"""
import unittest

import numpy as np
import numpy.testing as nptest

from datafold.dynfold.jsf import ColumnSplitter, JointlySmoothFunctions, JsfDataset
from datafold.dynfold.tests.helper import make_strip
from datafold.pcfold.kernels import GaussianKernel
from datafold.utils.general import random_subsample


class ColumnSplitterTest(unittest.TestCase):
    def test_fit_transform(self):
        observations = [np.random.rand(1000, i + 1) for i in range(3)]

        columns_splitter = ColumnSplitter(
            [
                JsfDataset("observation0", slice(0, 1)),
                JsfDataset("observation1", slice(1, 3)),
                JsfDataset("observation2", slice(3, 6)),
            ]
        )

        X = np.column_stack(observations)

        split_X = columns_splitter.fit_transform(X)

        for expected_observation, actual_observation in zip(observations, split_X):
            nptest.assert_array_equal(expected_observation, actual_observation)


class JointlySmoothFunctionsTest(unittest.TestCase):
    def setUp(self):
        self.xmin = 0.0
        self.ymin = 0.0
        self.width = 1.0
        self.height = 1e-1
        self.num_samples = 50000
        self.data = make_strip(
            self.xmin, self.ymin, self.width, self.height, self.num_samples
        )

    @staticmethod
    def _compute_rayleigh_quotients(matrix, eigenvectors):
        """Compute Rayleigh quotients."""
        n = eigenvectors.shape[1]
        rayleigh_quotients = np.zeros(n)
        for i in range(n):
            v = eigenvectors[:, i]
            rayleigh_quotients[i] = np.dot(v, matrix @ v) / np.dot(v, v)
        rayleigh_quotients = np.sort(np.abs(rayleigh_quotients))
        return rayleigh_quotients[::-1]

    def test_accuracy(self):
        n_observations = 2
        n_samples = 200
        n_kernel_eigenvectors = 100
        n_jointly_smooth_functions = 10
        epsilon = 5e-1

        downsampled_data = [
            random_subsample(self.data, n_samples)[0] for _ in range(n_observations)
        ]

        X = np.column_stack(downsampled_data)

        column_splitter = ColumnSplitter(
            [
                JsfDataset(
                    f"observation{i}",
                    slice(i * 2, (i + 1) * 2),
                    kernel=GaussianKernel(epsilon=epsilon),
                )
                for i in range(n_observations)
            ]
        )

        jsf = JointlySmoothFunctions(
            n_kernel_eigenvectors=n_kernel_eigenvectors,
            n_jointly_smooth_functions=n_jointly_smooth_functions,
            datasets=column_splitter,
        ).fit(X)

        actual_kernel_eigvals = jsf.kernel_eigenvalues_
        expected_kernel_eigvals = [
            self._compute_rayleigh_quotients(kernel_matrix, eigenvectors)
            for kernel_matrix, eigenvectors in zip(
                jsf.kernel_matrices_, jsf.kernel_eigenvectors_
            )
        ]

        for a, e in zip(actual_kernel_eigvals, expected_kernel_eigvals):
            nptest.assert_allclose(np.abs(a), np.abs(e), atol=1e-16)

    def test_set_param(self):
        jsf = JointlySmoothFunctions()
        jsf.set_params(**dict(n_jointly_smooth_functions=42))

        self.assertEqual(jsf.n_jointly_smooth_functions, 42)

    def test_is_valid_sklearn_estimator(self):
        from sklearn.utils.estimator_checks import check_estimator

        for estimator, check in check_estimator(
            JointlySmoothFunctions(
                n_kernel_eigenvectors=10, n_jointly_smooth_functions=3
            ),
            generate_only=True,
        ):
            check(estimator)


if __name__ == "__main__":
    unittest.main()
