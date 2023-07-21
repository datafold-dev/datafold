"""Unit test for the jsf module."""
import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest

from datafold.dynfold.base import TransformType
from datafold.dynfold.jsf import JointlySmoothFunctions
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


class JointlySmoothFunctionsTest(unittest.TestCase):
    def setUp(self):
        self.parameters, self.observations, self.effective_parameter = generate_points(
            1000
        )

        self.X = np.column_stack([self.parameters, self.observations])
        self.data_splits = [
            ("parameters", GaussianKernel(), slice(0, 2)),
            ("observations", GaussianKernel(), slice(2, 4)),
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
        X: TransformType,
        data_splits,
        n_kernel_eigenvectors=100,
        n_jointly_smooth_functions=10,
    ):
        jsf = JointlySmoothFunctions(
            data_splits=data_splits,
            n_kernel_eigenvectors=n_kernel_eigenvectors,
            n_jointly_smooth_functions=n_jointly_smooth_functions,
            eigenvector_tolerance=1e-10,
        ).fit(X, store_kernel_matrix=True)

        for name in jsf.kernel_content_:
            actual_kernel_eigvals = jsf.kernel_eigenvalues_[name]

            kernel_matrix = jsf.kernel_content_[name]["kernel_matrix"]
            evec = jsf.kernel_eigenvectors_[name]
            expected_kernel_eigvals = self._compute_rayleigh_quotients(
                kernel_matrix, evec
            )
            nptest.assert_allclose(
                np.abs(actual_kernel_eigvals),
                np.abs(expected_kernel_eigvals),
                atol=1e-9,
            )

        # nptest.assert_allclose(
        #     jsf.transform(X), jsf.jointly_smooth_vectors_, atol=1e-2, rtol=1e-3
        # )

    def test_accuracy(self):
        self._test_accuracy(self.X, self.data_splits)

    def test_set_param(self):
        jsf = JointlySmoothFunctions([])
        jsf.set_params(**dict(n_jointly_smooth_functions=42))

        self.assertEqual(jsf.n_jointly_smooth_functions, 42)

    def test_more_than_two_datasets(self):
        X = np.column_stack(
            [self.parameters, self.observations, self.effective_parameter]
        )

        data_splits = [
            ("parameters", GaussianKernel(), slice(0, 2)),
            ("observations", GaussianKernel(), slice(2, 4)),
            ("effective_parameter", GaussianKernel(), slice(2, 4)),
        ]

        self._test_accuracy(X, data_splits)

    def test_is_valid_sklearn_estimator(self):
        from sklearn.utils.estimator_checks import check_estimator

        estimator = JointlySmoothFunctions(
            data_splits=[
                ("one", GaussianKernel(), slice(0, 2)),
                ("two", GaussianKernel(), slice(0, 3)),
            ],
            n_kernel_eigenvectors=5,
            n_jointly_smooth_functions=3,
        )

        for e, check in check_estimator(estimator, generate_only=True):
            check(e)

    def _test_tsc_data(self, tsc_data: TSCDataFrame, data_splits):
        jsf = JointlySmoothFunctions(
            data_splits=data_splits,
            n_kernel_eigenvectors=10,
            n_jointly_smooth_functions=2,
        ).fit(tsc_data)

        for name in jsf.kernel_eigenvectors_:
            self.assertIsInstance(jsf.kernel_eigenvectors_[name], TSCDataFrame)

        self.assertIsInstance(jsf.jointly_smooth_vectors_, TSCDataFrame)
        self.assertIsInstance(jsf.transform(tsc_data), TSCDataFrame)
        self.assertIsInstance(jsf.fit_transform(tsc_data), TSCDataFrame)

        pdtest.assert_frame_equal(
            jsf.fit(tsc_data).transform(tsc_data),
            jsf.fit_transform(tsc_data),
            atol=1e-2,
        )

    def test_tsc_data_two_datasets(self):
        _x = np.linspace(0, 2 * np.pi, 200)
        df = pd.DataFrame(
            np.column_stack([np.sin(_x), np.cos(_x)]), columns=["sin", "cos"]
        )
        tsc_data = TSCDataFrame.from_single_timeseries(df=df)

        data_splits = [
            ("one", GaussianKernel(epsilon=0.1), slice(0, 1)),
            ("two", GaussianKernel(epsilon=0.1), slice(1, 2)),
        ]

        self._test_tsc_data(tsc_data, data_splits=data_splits)

    def test_data_splits(self):
        _x = np.linspace(0, 2 * np.pi, 200)
        df = pd.DataFrame(
            np.column_stack([np.sin(_x), np.cos(_x)]), columns=["sin", "cos"]
        )
        tsc_data = TSCDataFrame.from_single_timeseries(df=df)
        data = tsc_data.copy().to_numpy()

        data_split1 = [
            ("one", GaussianKernel(epsilon=0.1), slice(0, 1)),
            ("two", GaussianKernel(epsilon=0.1), slice(1, 2)),
        ]

        data_split2 = [
            ("one", GaussianKernel(epsilon=0.1), np.array([0])),
            ("two", GaussianKernel(epsilon=0.1), np.array([1])),
        ]

        data_split3 = [
            ("one", GaussianKernel(epsilon=0.1), [0]),
            ("two", GaussianKernel(epsilon=0.1), [1]),
        ]

        data_split4 = [
            ("one", GaussianKernel(epsilon=0.1), np.array([True, False])),
            ("two", GaussianKernel(epsilon=0.1), np.array([False, True])),
        ]

        data_split5 = [
            ("one", GaussianKernel(epsilon=0.1), np.array([0, 100])),
            ("two", GaussianKernel(epsilon=0.1), np.array([1, 200])),
        ]

        self._test_tsc_data(tsc_data, data_splits=data_split1)
        self._test_tsc_data(tsc_data, data_splits=data_split2)
        self._test_tsc_data(tsc_data, data_splits=data_split3)
        self._test_tsc_data(tsc_data, data_splits=data_split4)

        with self.assertRaises(IndexError):
            self._test_tsc_data(tsc_data, data_splits=data_split5)

        self._test_accuracy(data, data_split1)
        self._test_accuracy(data, data_split2)
        self._test_accuracy(data, data_split3)
        self._test_accuracy(data, data_split4)

        with self.assertRaises(IndexError):
            self._test_accuracy(tsc_data, data_splits=data_split5)

    def test_tsc_data_more_than_two_datasets(self):
        _x = np.linspace(0, 2 * np.pi, 200)
        df = pd.DataFrame(
            np.column_stack([np.sin(_x), np.cos(_x), np.tan(_x)]),
            columns=["sin", "cos", "tan"],
        )
        tsc_data = TSCDataFrame.from_single_timeseries(df=df)

        data_splits = [
            ("one", GaussianKernel(), slice(0, 1)),
            ("two", GaussianKernel(), slice(1, 2)),
            ("three", GaussianKernel(), slice(2, 3)),
        ]
        self._test_tsc_data(tsc_data, data_splits)

    def test_tsc_data_multiple_time_series(self):
        _x_1 = np.linspace(0, 2 * np.pi, 200)
        _x_2 = np.linspace(2 * np.pi, 4 * np.pi, 200)
        df1 = pd.DataFrame(np.column_stack([np.sin(_x_1), np.cos(_x_1)]))
        df2 = pd.DataFrame(np.column_stack([np.sin(_x_2), np.cos(_x_2)]))

        tsc_data = TSCDataFrame.from_frame_list([df1, df2])
        tsc_data.columns = tsc_data.columns.astype(str)

        kernel_split = [
            ("one", GaussianKernel(), slice(0, 1)),
            ("two", GaussianKernel(), slice(1, 2)),
        ]

        self._test_tsc_data(tsc_data, data_splits=kernel_split)


if __name__ == "__main__":
    unittest.main()
