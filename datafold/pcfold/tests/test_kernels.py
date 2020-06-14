#!/usr/bin/env python3
import unittest

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nptest
import scipy.sparse
from scipy.spatial.distance import pdist

from datafold.pcfold.distance import compute_distance_matrix
from datafold.pcfold.kernels import (
    ContinuousNNKernel,
    DmapKernelFixed,
    GaussianKernel,
    _kth_nearest_neighbor_dist,
)


def generate_box_data(n_left, n_middle, n_right, seed):
    rng = np.random.default_rng(seed)

    def _sample_box(low, high, n_samples):
        x = rng.uniform(low, high, size=n_samples)
        y = rng.uniform(0, 1, size=n_samples)
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
    def test_symmetric_division_sparse_dense01(self):
        from datafold.pcfold.kernels import _symmetric_matrix_division

        data = generate_circle_data(100, 100, 100)
        distance_matrix = compute_distance_matrix(data)

        actual = _symmetric_matrix_division(
            matrix=scipy.sparse.csr_matrix(distance_matrix),
            vec=np.arange(1, distance_matrix.shape[0] + 1),
        )

        expected = _symmetric_matrix_division(
            matrix=distance_matrix, vec=np.arange(1, distance_matrix.shape[0] + 1),
        )

        nptest.assert_array_equal(actual.toarray(), expected)

    def test_symmetric_division_sparse_dense02(self):
        from datafold.pcfold.kernels import _symmetric_matrix_division

        data = generate_circle_data(100, 100, 100)
        distance_matrix = compute_distance_matrix(data)

        actual = _symmetric_matrix_division(
            matrix=scipy.sparse.csr_matrix(distance_matrix),
            vec=np.arange(1, distance_matrix.shape[0] + 1),
            vec_right=np.arange(1, distance_matrix.shape[1] + 1)[::-1],
        )

        expected = _symmetric_matrix_division(
            matrix=distance_matrix,
            vec=np.arange(1, distance_matrix.shape[0] + 1),
            vec_right=np.arange(1, distance_matrix.shape[1] + 1)[::-1],
        )

        nptest.assert_array_equal(actual.toarray(), expected)

    def test_symmetric_division_sparse_dense03(self):
        from datafold.pcfold.kernels import _symmetric_matrix_division

        data_ref = generate_circle_data(100, 100, 100)
        data_query = generate_circle_data(50, 50, 100)
        distance_matrix = compute_distance_matrix(data_ref, data_query)

        actual = _symmetric_matrix_division(
            matrix=scipy.sparse.csr_matrix(distance_matrix),
            vec=np.arange(1, distance_matrix.shape[0] + 1),
            vec_right=np.arange(1, distance_matrix.shape[1] + 1)[::-1],
        )

        expected = _symmetric_matrix_division(
            matrix=distance_matrix,
            vec=np.arange(1, distance_matrix.shape[0] + 1),
            vec_right=np.arange(1, distance_matrix.shape[1] + 1)[::-1],
        )

        nptest.assert_array_equal(actual.toarray(), expected)

    def test_sparse_kth_dist01(self):
        data = generate_circle_data(100, 100, 1)
        distance_matrix = compute_distance_matrix(data)

        for k in np.linspace(2, 90, 20).astype(np.int):
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

        for k in np.linspace(2, 90, 20).astype(np.int):

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
    def test_gaussian_kernel_print(self):
        kernel = GaussianKernel(epsilon=1)
        self.assertEqual(kernel.__repr__(), "GaussianKernel(epsilon=1)")


class TestDiffusionMapsKernelTest(unittest.TestCase):
    def test_is_symmetric01(self):
        # stochastic False

        # Note: in this case the alpha value is ignored
        k1 = DmapKernelFixed(is_stochastic=False, symmetrize_kernel=True)
        self.assertTrue(k1.is_symmetric)

        # No transformation to symmetrize the kernel is required
        self.assertFalse(k1.is_symmetric_transform(is_pdist=True))

        # Because the kernel is not stochastic, the kernel remains symmetric
        k2 = DmapKernelFixed(is_stochastic=False, symmetrize_kernel=False)
        self.assertTrue(k2.is_symmetric)

        # No transformation is required
        self.assertFalse(k1.is_symmetric_transform(is_pdist=True))

    def test_is_symmetric02(self):
        # symmetric_kernel and alpha == 0
        k1 = DmapKernelFixed(is_stochastic=True, alpha=0, symmetrize_kernel=False)
        self.assertFalse(k1.is_symmetric)
        self.assertFalse(k1.is_symmetric_transform(is_pdist=True))

        k2 = DmapKernelFixed(is_stochastic=True, alpha=0, symmetrize_kernel=True)
        self.assertTrue(k2.is_symmetric)
        self.assertTrue(k2.is_symmetric_transform(is_pdist=True))

    def test_is_symmetric03(self):
        # symmetric_kernel and alpha > 0
        k1 = DmapKernelFixed(is_stochastic=True, alpha=1, symmetrize_kernel=False)
        self.assertFalse(k1.is_symmetric)
        self.assertFalse(k1.is_symmetric_transform(is_pdist=True))

        k2 = DmapKernelFixed(is_stochastic=True, alpha=1, symmetrize_kernel=True)
        self.assertTrue(k2.is_symmetric)
        self.assertTrue(k2.is_symmetric_transform(is_pdist=True))

    def test_is_symmetric04(self):
        # when is_pdist is False
        k1 = DmapKernelFixed(is_stochastic=True, alpha=1, symmetrize_kernel=True)
        self.assertFalse(k1.is_symmetric_transform(is_pdist=False))

        k2 = DmapKernelFixed(is_stochastic=False, alpha=1, symmetrize_kernel=True)
        self.assertFalse(k2.is_symmetric_transform(is_pdist=False))

    def test_missing_row_alpha_fit(self):
        data_X = np.random.rand(100, 5)
        data_Y = np.random.rand(5, 5)

        kernel = DmapKernelFixed(
            GaussianKernel(epsilon=1),
            is_stochastic=True,
            alpha=1,
            symmetrize_kernel=False,
        )

        _, cdist_kwargs, _ = kernel(X=data_X)

        with self.assertRaises(ValueError):
            kernel(X=data_X, Y=data_Y)

        # No error:
        kernel(X=data_X, Y=data_Y, **cdist_kwargs)


class TestContinuousNNKernel(unittest.TestCase):
    @staticmethod
    def plot_data(train_data, train_graph, test_data=None, test_graph=None):

        fig, ax = plt.subplots()

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

        ax.axis("equal")

    def test_box_example(self, plot=False):
        train_data = generate_box_data(300, 300, 20, 1)
        test_data = generate_box_data(100, 100, 100, 2)

        for dist_cut_off in [None, 1e100]:
            # test if a sparse distance matrix gets the same result
            cknn = ContinuousNNKernel(k_neighbor=5, delta=1.5)

            graph_train, cdist_kwargs = cknn(
                train_data, dist_kwargs=dict(cut_off=dist_cut_off)
            )
            graph_test, _ = cknn(
                train_data,
                test_data,
                dist_kwargs=dict(cut_off=dist_cut_off),
                **cdist_kwargs,
            )

            self.assertIsInstance(graph_train, scipy.sparse.csr_matrix)
            self.assertIsInstance(graph_test, scipy.sparse.csr_matrix)

            self.assertTrue(graph_train.dtype == np.bool)
            self.assertTrue(graph_test.dtype == np.bool)

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

    def test_circle_example(self, plot=False):
        train_data = generate_circle_data(40, 40, 1)
        test_data = generate_circle_data(40, 40, 2)

        for dist_cut_off in [None, 1e100]:

            cknn = ContinuousNNKernel(k_neighbor=5, delta=2.3)
            graph_train, cdist_kwargs = cknn(train_data)
            graph_test, _ = cknn(
                train_data,
                test_data,
                dist_kwargs=dict(cut_off=dist_cut_off),
                **cdist_kwargs,
            )

            self.assertIsInstance(graph_train, scipy.sparse.csr_matrix)
            self.assertIsInstance(graph_test, scipy.sparse.csr_matrix)

            self.assertTrue(graph_train.dtype == np.bool)
            self.assertTrue(graph_test.dtype == np.bool)

            # Only reference testing for the examples possible.
            # This test fails if there are changes in the implementation and need to be
            # adapted.
            print(graph_train.getnnz(axis=1).mean())
            print(graph_test.getnnz(axis=1).mean())
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
            cknn.eval(sparse_distance_matrix, is_pdist=True)

    def test_wrong_setups(self):

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
            ContinuousNNKernel(k_neighbor=41, delta=1).eval(distance_matrix)


if __name__ == "__main__":
    TestContinuousNNKernel().test_circle_example(True)
