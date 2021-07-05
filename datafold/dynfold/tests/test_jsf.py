""" Unit test for the jsf module.

"""
import unittest
from typing import List, Optional

import numpy as np
import numpy.testing as nptest
import pandas as pd

from datafold.dynfold.base import TransformType
from datafold.dynfold.jsf import JointlySmoothFunctions, JsfDataset, _ColumnSplitter
from datafold.pcfold import TSCDataFrame
from datafold.pcfold.kernels import GaussianKernel


def generate_parameters(_x, _y):
    return np.column_stack(
        [
            _x,
            _y,
        ]
    )


def generate_observations(_x, _z, div=5, mult=6):
    return np.column_stack(
        [
            (div / 2 * _z + _x / 2 + 2 / 3) * np.cos(mult * np.pi * _z) / 2,
            (div / 2 * _z + _x / 2 + 2 / 3) * np.sin(mult * np.pi * _z) / 2,
        ]
    )


def generate_points(n_samples):
    rng = np.random.default_rng(42)
    xyz = rng.uniform(low=-0.5, high=0.5, size=(n_samples, 3))
    x, y, z = (
        xyz[:, 0].reshape(-1, 1),
        xyz[:, 1].reshape(-1, 1),
        xyz[:, 2].reshape(-1, 1),
    )

    parameters = generate_parameters(x, y)
    effective_parameter = parameters[:, 0] + parameters[:, 1] ** 2
    observations = generate_observations(effective_parameter, z[:, 0], 2, 2)

    return parameters, observations, effective_parameter


class ColumnSplittingTest(unittest.TestCase):
    def test_splitting(self):
        observations = [np.random.rand(1000, i + 1) for i in range(3)]

        columns_splitter = _ColumnSplitter(
            [
                JsfDataset("observation0", slice(0, 1)),
                JsfDataset("observation1", slice(1, 3)),
                JsfDataset("observation2", slice(3, 6)),
            ]
        )

        X = np.column_stack(observations)

        split_X = columns_splitter.split(X)

        for expected_observation, actual_observation in zip(observations, split_X):
            nptest.assert_array_equal(expected_observation, actual_observation)


class JointlySmoothFunctionsTest(unittest.TestCase):
    def setUp(self):
        self.parameters, self.observations, self.effective_parameter = generate_points(
            1000
        )

        self.X = np.column_stack([self.parameters, self.observations])

        self.datasets = [
            JsfDataset("parameters", slice(0, 2)),
            JsfDataset("observations", slice(2, 4)),
        ]

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

    def _test_accuracy(
        self,
        datasets: List[JsfDataset],
        X: TransformType,
        n_kernel_eigenvectors=100,
        n_jointly_smooth_functions=10,
    ):
        jsf = JointlySmoothFunctions(
            n_kernel_eigenvectors=n_kernel_eigenvectors,
            n_jointly_smooth_functions=n_jointly_smooth_functions,
            datasets=datasets,
            eigenvector_tolerance=1e-10,
        ).fit(X)

        actual_kernel_eigvals = jsf.kernel_eigenvalues_
        expected_kernel_eigvals = [
            self._compute_rayleigh_quotients(kernel_matrix, eigenvectors)
            for kernel_matrix, eigenvectors in zip(
                jsf.kernel_matrices_, jsf.kernel_eigenvectors_
            )
        ]

        for a, e in zip(actual_kernel_eigvals, expected_kernel_eigvals):
            nptest.assert_allclose(np.abs(a), np.abs(e), atol=1e-9)

    def test_accuracy(self):
        self._test_accuracy(self.datasets, self.X)

    def test_set_param(self):
        jsf = JointlySmoothFunctions([])
        jsf.set_params(**dict(n_jointly_smooth_functions=42))

        self.assertEqual(jsf.n_jointly_smooth_functions, 42)

    def test_more_than_two_datasets(self):
        X = np.column_stack(
            [self.parameters, self.observations, self.effective_parameter]
        )
        datasets = [
            JsfDataset("parameters", slice(0, 2)),
            JsfDataset("observations", slice(2, 4)),
            JsfDataset("effective_parameter", slice(4, 5)),
        ]

        self._test_accuracy(datasets, X)

    def test_is_valid_sklearn_estimator(self):
        from sklearn.utils.estimator_checks import check_estimator

        for estimator, check in check_estimator(
            JointlySmoothFunctions(
                n_kernel_eigenvectors=5,
                n_jointly_smooth_functions=3,
                datasets=[JsfDataset(kernel=GaussianKernel(epsilon=1.0))],
            ),
            generate_only=True,
        ):
            check(estimator)

    def _test_tsc_data(
        self, tsc_data: TSCDataFrame, datasets: Optional[List[JsfDataset]] = None
    ):
        jsf = JointlySmoothFunctions(
            n_kernel_eigenvectors=10, n_jointly_smooth_functions=2, datasets=datasets
        ).fit(tsc_data)

        for kernel_eigenvectors in jsf.kernel_eigenvectors_:
            self.assertIsInstance(kernel_eigenvectors, TSCDataFrame)

        self.assertIsInstance(jsf.jointly_smooth_functions, TSCDataFrame)
        self.assertIsInstance(jsf.transform(tsc_data), TSCDataFrame)

    def test_tsc_data_no_datasets(self):
        _x = np.linspace(0, 2 * np.pi, 200)
        df = pd.DataFrame(
            np.column_stack([np.sin(_x), np.cos(_x)]), columns=["sin", "cos"]
        )
        tsc_data = TSCDataFrame.from_single_timeseries(df=df)

        self._test_tsc_data(tsc_data)

    def test_tsc_data_two_datasets(self):
        _x = np.linspace(0, 2 * np.pi, 200)
        df = pd.DataFrame(
            np.column_stack([np.sin(_x), np.cos(_x)]), columns=["sin", "cos"]
        )
        tsc_data = TSCDataFrame.from_single_timeseries(df=df)
        dataset1 = JsfDataset(columns=slice(0, 1), kernel=GaussianKernel())
        dataset2 = JsfDataset(columns=slice(1, 2), kernel=GaussianKernel())

        self._test_tsc_data(tsc_data, [dataset1, dataset2])

    def test_tsc_data_more_than_two_datasets(self):
        _x = np.linspace(0, 2 * np.pi, 200)
        df = pd.DataFrame(
            np.column_stack([np.sin(_x), np.cos(_x), np.tan(_x)]),
            columns=["sin", "cos", "tan"],
        )
        tsc_data = TSCDataFrame.from_single_timeseries(df=df)
        dataset1 = JsfDataset(columns=slice(0, 1), kernel=GaussianKernel())
        dataset2 = JsfDataset(columns=slice(1, 2), kernel=GaussianKernel())
        dataset3 = JsfDataset(columns=slice(2, 3), kernel=GaussianKernel())

        self._test_tsc_data(tsc_data, [dataset1, dataset2, dataset3])

    def test_tsc_data_multiple_time_series(self):
        _x_1 = np.linspace(0, 2 * np.pi, 200)
        _x_2 = np.linspace(2 * np.pi, 4 * np.pi, 200)
        df1 = pd.DataFrame(np.column_stack([np.sin(_x_1), np.cos(_x_1)]))
        df2 = pd.DataFrame(np.column_stack([np.sin(_x_2), np.cos(_x_2)]))

        tsc_data = TSCDataFrame.from_frame_list([df1, df2])

        self._test_tsc_data(tsc_data)


if __name__ == "__main__":
    unittest.main()
