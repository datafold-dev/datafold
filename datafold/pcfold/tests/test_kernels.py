#!/usr/bin/env python3
import unittest
import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest
import scipy.sparse
from scipy.spatial.distance import pdist, squareform

from datafold.pcfold.distance import SklearnKNN, compute_distance_matrix
from datafold.pcfold.kernels import (
    ConeKernel,
    ContinuousNNKernel,
    CubicKernel,
    DmapKernelFixed,
    GaussianKernel,
    InverseMultiquadricKernel,
    MultiquadricKernel,
    QuinticKernel,
    _kth_nearest_neighbor_dist,
    _symmetric_matrix_division,
)
from datafold.pcfold.timeseries.collection import TSCDataFrame, TSCException
from datafold.utils.general import is_symmetric_matrix


def generate_box_data(n_left, n_middle, n_right, seed):
    rng = np.random.default_rng(seed)

    def _sample_box(low, high, n_samples):
        x = rng.uniform(low, high, size=n_samples)
        y = rng.uniform(size=n_samples)
        return np.column_stack([x, y])

    left_box = _sample_box(0, 0.3, n_left)
    middle_box = _sample_box(0.4, 0.7, n_middle)
    right_box = _sample_box(0.8, 1.3, n_right)

    return np.vstack([left_box, middle_box, right_box])


def generate_circle_data(n_large, n_small, seed):
    rng = np.random.default_rng(seed)

    radius_large = 1
    radius_small = 0.2
    center_large = (0, 0)
    center_small = (radius_large + radius_small, 0)
    radius_noise_large = 0.1
    radius_noise_small = 0.02

    def _sample_circle(n_sample, radius, radius_noise, center):
        angle_values = rng.uniform(0, 2 * np.pi, n_sample)
        cos_evals = np.cos(angle_values)
        sin_evals = np.sin(angle_values)

        x = cos_evals * radius + rng.normal(loc=0, scale=radius_noise, size=n_sample)
        y = sin_evals * radius + rng.normal(loc=0, scale=radius_noise, size=n_sample)
        x += center[0]
        y += center[1]

        return np.column_stack([x, y])

    large = _sample_circle(n_large, radius_large, radius_noise_large, center_large)
    small = _sample_circle(n_small, radius_small, radius_noise_small, center_small)

    return np.vstack([large, small])


class TestKernelUtils(unittest.TestCase):
    def test_symmetric_division_raise_zero_division(self):
        data = generate_circle_data(100, 100, 100)
        distance_matrix = compute_distance_matrix(data)

        vec_left = np.arange(0, distance_matrix.shape[0])
        vec_right = np.arange(distance_matrix.shape[0] - 1, -1, -1)

        with self.assertRaises(ZeroDivisionError):
            _symmetric_matrix_division(
                matrix=distance_matrix,
                vec=vec_left,
                is_symmetric=True,
                value_zero_division="raise",
            )

        with self.assertRaises(ZeroDivisionError):
            _symmetric_matrix_division(
                matrix=distance_matrix,
                vec=np.arange(1, distance_matrix.shape[0] + 1),
                is_symmetric=True,
                vec_right=vec_right,
                value_zero_division="raise",
            )

    def test_symmetric_division_fill_value(self):
        data = generate_circle_data(100, 100, 100)
        distance_matrix = compute_distance_matrix(data)

        vec_left = np.arange(0, distance_matrix.shape[0]).astype(float)

        value_zero_division = -1

        actual_dense = _symmetric_matrix_division(
            matrix=np.copy(distance_matrix),
            vec=np.copy(vec_left),
            is_symmetric=True,
            value_zero_division=value_zero_division,
        )

        # dense case
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            expected = (
                np.diag(np.reciprocal(vec_left))
                @ distance_matrix
                @ np.diag(np.reciprocal(vec_left))
            )
            expected[~np.isfinite(expected)] = value_zero_division

        self.assertIsInstance(actual_dense, np.ndarray)
        nptest.assert_allclose(actual_dense, expected, atol=1e-16, rtol=1e-17)

        # sparse
        distance_matrix[distance_matrix == 0] = np.nan
        distance_matrix = scipy.sparse.csr_matrix(distance_matrix)
        distance_matrix.data[np.isnan(distance_matrix.data)] = 0
        actual_sparse = _symmetric_matrix_division(
            matrix=distance_matrix,
            vec=np.copy(vec_left),
            is_symmetric=True,
            value_zero_division=value_zero_division,
        )

        self.assertTrue(scipy.sparse.isspmatrix_csr(actual_sparse))
        nptest.assert_allclose(
            actual_sparse.toarray(), expected, atol=1e-16, rtol=1e-17
        )

    def test_symmetric_division_sparse_dense01(self):
        data = generate_circle_data(100, 100, 100)
        distance_matrix = compute_distance_matrix(data)

        actual = _symmetric_matrix_division(
            matrix=scipy.sparse.csr_matrix(distance_matrix),
            vec=np.arange(1, distance_matrix.shape[0] + 1),
            is_symmetric=True,
        )

        expected = _symmetric_matrix_division(
            matrix=distance_matrix,
            vec=np.arange(1, distance_matrix.shape[0] + 1),
            is_symmetric=True,
        )

        nptest.assert_array_equal(actual.toarray(), expected)

    def test_symmetric_division_sparse_dense02(self):
        data = generate_circle_data(100, 100, 100)
        distance_matrix = compute_distance_matrix(data)

        actual = _symmetric_matrix_division(
            matrix=scipy.sparse.csr_matrix(distance_matrix),
            vec=np.arange(1, distance_matrix.shape[0] + 1),
            is_symmetric=False,
            vec_right=np.arange(1, distance_matrix.shape[1] + 1)[::-1],
        )

        expected = _symmetric_matrix_division(
            matrix=distance_matrix,
            vec=np.arange(1, distance_matrix.shape[0] + 1),
            is_symmetric=False,
            vec_right=np.arange(1, distance_matrix.shape[1] + 1)[::-1],
        )
        nptest.assert_allclose(actual.toarray(), expected, rtol=1e-16, atol=1e-17)

    def test_symmetric_division_sparse_dense03(self):
        data_ref = generate_circle_data(100, 100, 100)
        data_query = generate_circle_data(50, 50, 100)
        distance_matrix = compute_distance_matrix(data_ref, data_query)

        actual = _symmetric_matrix_division(
            matrix=scipy.sparse.csr_matrix(distance_matrix),
            vec=np.arange(1, distance_matrix.shape[0] + 1),
            is_symmetric=False,
            vec_right=np.arange(1, distance_matrix.shape[1] + 1)[::-1],
        )

        expected = _symmetric_matrix_division(
            matrix=distance_matrix,
            vec=np.arange(1, distance_matrix.shape[0] + 1),
            is_symmetric=False,
            vec_right=np.arange(1, distance_matrix.shape[1] + 1)[::-1],
        )

        nptest.assert_allclose(actual.toarray(), expected, rtol=1e-16, atol=1e-17)

    def test_sparse_kth_dist01(self):
        data = generate_circle_data(100, 100, 1)
        distance_matrix = compute_distance_matrix(data)

        for k in np.linspace(2, 90, 20).astype(int):
            expected = _kth_nearest_neighbor_dist(distance_matrix, k)

            distance_matrix = scipy.sparse.csr_matrix(distance_matrix)
            distance_matrix.setdiag(0)
            actual = _kth_nearest_neighbor_dist(distance_matrix, k)

            nptest.assert_array_equal(actual, expected)
            self.assertEqual(len(actual), data.shape[0])
            self.assertEqual(actual.ndim, 1)

    def test_sparse_kth_dist02(self):
        data_ref = generate_circle_data(100, 100, 0)
        data_query = generate_circle_data(50, 50, 1)

        distance_matrix = compute_distance_matrix(data_ref, data_query)

        for k in np.linspace(2, 90, 20).astype(int):
            actual = _kth_nearest_neighbor_dist(
                scipy.sparse.csr_matrix(distance_matrix), k
            )
            expected = _kth_nearest_neighbor_dist(distance_matrix, k)

            nptest.assert_array_equal(actual, expected)
            self.assertEqual(len(actual), data_query.shape[0])
            self.assertEqual(actual.ndim, 1)

    def test_pdist_kth_dist(self):
        # sanity test for pdist: k=0 should always be 0

        data = generate_box_data(100, 100, 100, 1)

        dense_dist_mat = compute_distance_matrix(data)
        sparse_dist_mat = compute_distance_matrix(data, cut_off=1e100)

        actual_dense = _kth_nearest_neighbor_dist(dense_dist_mat, 1)
        actual_sparse = _kth_nearest_neighbor_dist(sparse_dist_mat, 1)

        expected = np.min(dense_dist_mat, axis=1)  # self is always zero

        nptest.assert_array_equal(expected, np.zeros(len(expected)))

        nptest.assert_array_equal(expected, actual_dense)
        nptest.assert_array_equal(expected, actual_sparse)

    def test_cdist_kth_dist(self):
        # sanity test for pdist: k=0 should always be 0

        data_X = generate_box_data(100, 100, 100, 1)
        data_Y = generate_box_data(100, 100, 100, 1)

        dense_dist_mat = compute_distance_matrix(data_Y, data_X)
        sparse_dist_mat = compute_distance_matrix(data_Y, data_X, cut_off=1e100)

        actual_dense = _kth_nearest_neighbor_dist(dense_dist_mat, 1)
        actual_sparse = _kth_nearest_neighbor_dist(sparse_dist_mat, 1)

        # because data_X and data_Y are duplicates, on the diagonal must be zeros
        # like in the pdist case
        expected = np.min(dense_dist_mat, axis=1)

        nptest.assert_array_equal(expected, np.zeros(len(expected)))

        nptest.assert_array_equal(expected, actual_dense)
        nptest.assert_array_equal(expected, actual_sparse)


class TestPCManifoldKernel(unittest.TestCase):
    rng = np.random.default_rng(2)

    def test_gaussian_kernel_print(self):
        kernel = GaussianKernel(epsilon=1)

        r = (
            "GaussianKernel(\n\tdistance=BruteForceDist(metric='sqeuclidean', "
            "is_symmetric=True, is_sparse=True, cut_off=inf, kmin=None)\n\tepsilon=1\n)"
        )
        self.assertEqual(kernel.__repr__(), r)

    def test_kernels_symmetry(self):
        data = self.rng.random(size=[100, 2])
        data_tsc = TSCDataFrame.from_single_timeseries(pd.DataFrame(data))

        kernels = [
            MultiquadricKernel(),
            QuinticKernel(),
            InverseMultiquadricKernel(),
            CubicKernel(),
        ]
        kernels_tsc = [ConeKernel(zeta=0), ConeKernel(zeta=0.5)]

        for k in kernels:
            kernel_matrix = k(data)
            self.assertTrue(is_symmetric_matrix(kernel_matrix))

        for k in kernels_tsc:
            kernel_matrix = k(data_tsc)
            self.assertTrue(is_symmetric_matrix(kernel_matrix.to_numpy()))

    def test_gaussian_kernel_callable(self):
        data = self.rng.random((10, 10))

        kernel_ufunc = GaussianKernel(epsilon=np.median)
        kernel_ufunc(data)

        self.assertEqual(
            kernel_ufunc.epsilon,
            np.median(squareform(pdist(data, metric="sqeuclidean"))),
        )

        kernel_lambda = GaussianKernel(epsilon=lambda x: np.median(x))
        kernel_lambda(data)

        self.assertEqual(
            kernel_lambda.epsilon,
            np.median(squareform(pdist(data, metric="sqeuclidean"))),
        )


class TestDiffusionMapsKernelTest(unittest.TestCase):
    rng = np.random.default_rng(5)

    def test_is_symmetric01(self):
        # stochastic False

        # Note: in this case the alpha value is ignored
        k1 = DmapKernelFixed(
            GaussianKernel(), is_stochastic=False, symmetrize_kernel=True
        )
        self.assertTrue(k1._is_symmetric_kernel)

        # No transformation to symmetrize the kernel is required
        self.assertFalse(k1.is_conjugate)

        # Because the kernel is not stochastic, the kernel remains symmetric
        k2 = DmapKernelFixed(
            GaussianKernel(), is_stochastic=False, symmetrize_kernel=False
        )
        self.assertTrue(k2._is_symmetric_kernel)

        # No transformation is required
        self.assertFalse(k1.is_conjugate)

    def test_is_symmetric02(self):
        # symmetric_kernel and alpha == 0
        k1 = DmapKernelFixed(
            GaussianKernel(), is_stochastic=True, alpha=0, symmetrize_kernel=False
        )
        self.assertFalse(k1._is_symmetric_kernel)
        self.assertFalse(k1.is_conjugate)

        k2 = DmapKernelFixed(
            GaussianKernel(), is_stochastic=True, alpha=0, symmetrize_kernel=True
        )
        self.assertTrue(k2._is_symmetric_kernel)
        self.assertTrue(k2.is_conjugate)

    def test_is_symmetric03(self):
        # symmetric_kernel and alpha > 0
        k1 = DmapKernelFixed(
            GaussianKernel(), is_stochastic=True, alpha=1, symmetrize_kernel=False
        )
        self.assertFalse(k1._is_symmetric_kernel)
        self.assertFalse(k1.is_conjugate)

        k2 = DmapKernelFixed(
            GaussianKernel(), is_stochastic=True, alpha=1, symmetrize_kernel=True
        )
        self.assertTrue(k2._is_symmetric_kernel)
        self.assertTrue(k2.is_conjugate)

    def test_missing_row_alpha_fit(self):
        data_X = self.rng.random((100, 5))
        data_Y = self.rng.random((5, 5))

        kernel = DmapKernelFixed(
            GaussianKernel(epsilon=1),
            is_stochastic=True,
            alpha=1,
            symmetrize_kernel=False,
        )

        _ = kernel(X=data_X)
        self.assertIsInstance(kernel.row_sums_alpha_, np.ndarray)

        # No error:
        kernel(X=data_X, Y=data_Y)


class TestContinuousNNKernel(unittest.TestCase):
    @staticmethod
    def plot_data(train_data, train_graph, test_data=None, test_graph=None, ax=None):
        if ax is None:
            if train_data.shape[1] == 2:
                fig, ax = plt.subplots()
            elif train_data.shape[1] == 3:
                from mpl_toolkits.mplot3d import Axes3D  # noqa

                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
            else:
                raise RuntimeError("only 2d or 3d")

        def _plot_data(query_data, ref_data, graph, color):
            ax.scatter(*query_data.T, c=color)

            if ref_data is None:
                ref_data = query_data.view()

            source, target = graph.nonzero()
            for s, t in zip(source, target):
                start_point = query_data[s, :]
                end_point = ref_data[t, :]
                points = np.row_stack([start_point, end_point])

                ax.plot(*points.T, linewidth=0.5, color=color)

        _plot_data(train_data, None, train_graph, color="black")

        if test_data is not None:
            _plot_data(test_data, train_data, test_graph, color="red")

        if train_data.shape[1] == 2:
            ax.axis("equal")
        return ax

    def test_box_example(self, plot=False):
        train_data = generate_box_data(300, 300, 20, 1)
        test_data = generate_box_data(100, 100, 100, 2)

        for dist_cut_off in [None, 1e100]:
            # test if a sparse distance matrix gets the same result
            cknn = ContinuousNNKernel(
                k_neighbor=5, delta=1.5, distance=dict(cut_off=dist_cut_off)
            )

            graph_train = cknn(train_data)
            graph_test = cknn(train_data, test_data)

            self.assertIsInstance(graph_train, scipy.sparse.csr_matrix)
            self.assertIsInstance(graph_test, scipy.sparse.csr_matrix)

            self.assertTrue(graph_train.dtype == bool)
            self.assertTrue(graph_test.dtype == bool)

            # Only reference testing for the examples possible.
            # This test fails if there are changes in the implementation and need to be
            # adapted
            print(graph_train.getnnz(axis=1).mean())
            print(graph_test.getnnz(axis=1).mean())
            self.assertEqual(graph_train.getnnz(axis=1).mean(), 8.993548387096775)
            self.assertEqual(graph_test.getnnz(axis=1).mean(), 8.15)

        if plot:
            self.plot_data(train_data, graph_train, test_data, graph_test)
            plt.show()

    def test_knn_kernel(self, plot=False):
        train_data = generate_circle_data(40, 40, 1)

        k = 30
        distance = SklearnKNN(metric="sqeuclidean", k=k)
        cknn = ContinuousNNKernel(k_neighbor=5, delta=2.3, distance=distance)

        def distance_with_additional_checks(X, Y):
            distance_matrix = distance(X, Y)
            self.assertFalse(is_symmetric_matrix(distance_matrix))
            self.assertEqual(distance_matrix.nnz, k * train_data.shape[0])
            return distance_matrix

        cknn.distance = distance_with_additional_checks
        cknn.distance.is_symmetric = False  # mock - k-nn is not symmetric in general

        graph_train = cknn(train_data)

        self.assertIsInstance(graph_train, scipy.sparse.csr_matrix)
        self.assertLessEqual(graph_train.nnz, k * train_data.shape[0])
        self.assertEqual(graph_train.dtype, bool)

        if plot:
            self.plot_data(train_data, graph_train)
            plt.show()

    def test_circle_example(self, plot=False):
        train_data = generate_circle_data(40, 40, 1)
        test_data = generate_circle_data(40, 40, 2)

        for dist_cut_off in [None, 1e100]:
            cknn = ContinuousNNKernel(
                k_neighbor=5, delta=2.3, distance=dict(cut_off=dist_cut_off)
            )
            graph_train = cknn(train_data)
            graph_test = cknn(
                train_data,
                test_data,
            )

            self.assertIsInstance(graph_train, scipy.sparse.csr_matrix)
            self.assertIsInstance(graph_test, scipy.sparse.csr_matrix)

            self.assertTrue(graph_train.dtype == bool)
            self.assertTrue(graph_test.dtype == bool)

            # Only reference testing for the examples possible.
            # This test fails if there are changes in the implementation and need to be
            # adapted.
            # print(graph_train.getnnz(axis=1).mean())
            # print(graph_test.getnnz(axis=1).mean())
            self.assertEqual(graph_train.getnnz(axis=1).mean(), 11.65)
            self.assertEqual(graph_test.getnnz(axis=1).mean(), 11.525)

        if plot:
            self.plot_data(train_data, graph_train, test_data, graph_test)
            plt.show()

    def test_error_insufficient_neighbors(self):
        data = generate_circle_data(100, 20, 0)
        cut_off = np.quantile(pdist(data), 0.1)
        sparse_distance_matrix = compute_distance_matrix(data, cut_off=cut_off)

        cknn = ContinuousNNKernel(k_neighbor=10, delta=2)

        self.assertTrue((sparse_distance_matrix.getnnz(axis=1) < 10).any())

        with self.assertRaises(ValueError):
            cknn.evaluate(sparse_distance_matrix, is_pdist=True)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            ContinuousNNKernel(k_neighbor=0, delta=1)

        with self.assertRaises(ValueError):
            ContinuousNNKernel(k_neighbor=-1, delta=1)

        with self.assertRaises(ValueError):
            ContinuousNNKernel(k_neighbor=1, delta=0)

        with self.assertRaises(ValueError):
            ContinuousNNKernel(k_neighbor=1, delta=-1)

        # 20 + 20 = 40 points
        distance_matrix = compute_distance_matrix(generate_circle_data(20, 20, 1))

        with self.assertRaises(ValueError):
            # k_neighbor larger than the distance matrix
            ContinuousNNKernel(k_neighbor=41, delta=1).evaluate(distance_matrix)


class TestConeKernel(unittest.TestCase):
    def setUp(self) -> None:
        data_X = np.random.default_rng(2).uniform(size=(100, 2))
        data_Y = np.random.default_rng(2).uniform(size=(50, 2))

        self.X_tsc = TSCDataFrame.from_single_timeseries(
            pd.DataFrame(data_X, columns=["A", "B"])
        )

        self.Y_tsc = TSCDataFrame.from_single_timeseries(
            pd.DataFrame(data_Y, columns=["A", "B"])
        )

    def test_return_type(self):
        cone_kernel = ConeKernel(zeta=0.5)
        actual = cone_kernel(self.X_tsc)

        self.assertIsInstance(actual, TSCDataFrame)

        self.assertIsInstance(cone_kernel.timederiv_X_, TSCDataFrame)
        self.assertIsInstance(cone_kernel.norm_timederiv_X_, TSCDataFrame)

        actual2 = cone_kernel(self.X_tsc, self.Y_tsc)
        self.assertIsInstance(actual2, TSCDataFrame)

    def test_zeta_approx_zero(self):
        cone_one = ConeKernel(zeta=1e-15)
        cone_two = ConeKernel(zeta=0)

        actual = cone_one(self.X_tsc)
        expected = cone_two(self.X_tsc)

        nptest.assert_allclose(actual, expected, rtol=0, atol=1e-15)

        actual = cone_one(self.X_tsc, self.Y_tsc)
        expected = cone_two(self.X_tsc, self.Y_tsc)
        nptest.assert_allclose(actual, expected, rtol=0, atol=1e-15)

    def test_cdist_evaluation_no_error(self):
        cone_kernel = ConeKernel(0.5)
        kernel_pdist = cone_kernel(self.X_tsc)
        kernel_cdist = cone_kernel(self.X_tsc, self.Y_tsc)

        self.assertTrue(np.isfinite(kernel_pdist).all().all())
        self.assertTrue(np.isfinite(kernel_cdist).all().all())

    def test_cdist(self):
        kernel = ConeKernel(zeta=0.5)
        expected_kernel = kernel(self.X_tsc)
        actual_kernel = kernel(self.X_tsc, self.X_tsc)
        pdtest.assert_frame_equal(expected_kernel, actual_kernel)

        # zeta=0 is a special case
        kernel = ConeKernel(zeta=0.0)
        expected_kernel = kernel(self.X_tsc)
        actual_kernel = kernel(self.X_tsc, self.X_tsc)
        pdtest.assert_frame_equal(expected_kernel, actual_kernel)

        # test pdist followed by cdist versus direct cdist versus
        kernel1 = ConeKernel(zeta=0.5)
        kernel2 = ConeKernel(zeta=0.5)

        _ = kernel1(self.X_tsc)
        actual = kernel1(self.X_tsc, self.Y_tsc)
        expected = kernel2(self.X_tsc, self.Y_tsc)

        pdtest.assert_frame_equal(actual, expected)

    def test_duplicate_samples(self):
        X = self.X_tsc.copy()
        Y = self.X_tsc.copy()

        X.iloc[0, :] = X.iloc[1, :]
        Y.iloc[0, :] = X.iloc[0, :]

        cone_kernel = ConeKernel(0.5)
        kernel_pdist = cone_kernel(self.X_tsc)
        kernel_cdist = cone_kernel(self.X_tsc, self.Y_tsc)

        self.assertTrue(np.isfinite(kernel_pdist).all().all())
        self.assertTrue(np.isfinite(kernel_cdist).all().all())

    def test_invalid_setting(self):
        with self.assertRaises(ValueError):
            ConeKernel(zeta=1)(self.X_tsc)

        with self.assertRaises(ValueError):
            ConeKernel(zeta=-0.1)(self.X_tsc)

        with self.assertRaises(ValueError):
            ConeKernel(epsilon=-0.1)(self.X_tsc, self.Y_tsc)

        # different sampling frequency in X than in Y
        Y_tsc = self.Y_tsc.copy().set_index(
            pd.MultiIndex.from_arrays(
                [np.ones(self.Y_tsc.shape[0]), np.arange(0, 2 * self.Y_tsc.shape[0], 2)]
            )
        )
        kernel = ConeKernel(zeta=0.5)
        kernel(self.X_tsc)

        with self.assertRaises(TSCException):
            ConeKernel(0.5)(self.X_tsc, Y_tsc)

        # non constant time sampling:
        X_tsc = self.X_tsc.copy().drop(5, level=1)
        with self.assertRaises(TSCException):
            ConeKernel(0.5)(X_tsc)
