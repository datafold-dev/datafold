#!/usr/bin/env python

import unittest

import numpy.testing as nptest
from scipy.spatial.distance import cdist, pdist, squareform

from datafold.pcfold.distance import *


class TestContinuousDistance(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(1)
        example_matrix = np.random.permutation(np.arange(1, 101, 1))
        self.example_matrix = example_matrix.reshape(10, 10)

    def test_k_smallest01(self):
        # compare between dense and sparse

        dense_dist_mat = compute_distance_matrix(self.example_matrix)
        sparse_dist_mat = compute_distance_matrix(self.example_matrix, cut_off=150)

        dense_actual = get_k_smallest_element_value(dense_dist_mat, 3)
        sparse_actual = get_k_smallest_element_value(sparse_dist_mat, 3)

        nptest.assert_array_equal(dense_actual, sparse_actual)

    def test_k_smallest02(self):
        # test that distance matrix is unchanged (in get_k_smallest_element_value, values are adapted

        dense_dist_mat = compute_distance_matrix(self.example_matrix)
        sparse_dist_mat = compute_distance_matrix(self.example_matrix, cut_off=150)

        get_k_smallest_element_value(dense_dist_mat, 3)
        get_k_smallest_element_value(sparse_dist_mat, 3)

        nptest.assert_array_equal(
            dense_dist_mat, compute_distance_matrix(self.example_matrix)
        )
        nptest.assert_array_equal(
            sparse_dist_mat.toarray(),
            compute_distance_matrix(self.example_matrix, cut_off=150).toarray(),
        )

    def test_k_smallest03(self):
        # sanity test for pdist and ignore zeros and k=0 should always be 0

        dense_dist_mat = compute_distance_matrix(self.example_matrix)
        sparse_dist_mat = compute_distance_matrix(self.example_matrix, cut_off=150)

        dense_actual = get_k_smallest_element_value(
            dense_dist_mat, 0, ignore_zeros=False
        )
        sparse_actual = get_k_smallest_element_value(
            sparse_dist_mat, 0, ignore_zeros=False
        )

        expected = dense_dist_mat.min(axis=1)  # self is always zero

        nptest.assert_array_equal(dense_actual, expected)
        nptest.assert_array_equal(sparse_actual, expected)

    def test_k_smallest04(self):
        # Test from legacy code in estimators.py
        n_points = self.example_matrix.shape[0]
        perm_indices_all = np.random.permutation(np.arange(n_points))

        # only dense case
        distance_matrix = compute_distance_matrix(
            self.example_matrix[perm_indices_all[:3], :],
            self.example_matrix,
            metric="euclidean",
        )

        sorted_distance_matrix = np.sort(distance_matrix, axis=1)
        kmin = 2

        expected = sorted_distance_matrix[:, kmin]
        actual = get_k_smallest_element_value(distance_matrix, kmin, ignore_zeros=False)

        nptest.assert_array_equal(expected, actual)

        self.assertEqual(np.median(expected), np.median(actual))

    def test_k_smallest_05(self):
        # Test against legacy code in the sparse case in estimators.py
        distance_matrix = compute_distance_matrix(self.example_matrix, cut_off=150)

        def get_kth_smallest_in_row_legacy(_mat_sp, k, default=0):
            """Returns kth largest element of every row, k is zero based, and zeros in the matrix are ignored."""
            res = np.zeros((_mat_sp.shape[0],))
            for i in range(_mat_sp.shape[0]):
                r = _mat_sp.getrow(i).data
                r = r[
                    r != 0
                ]  # added to legacy code to make sure zeros are completely ignored

                if len(r) <= k:
                    res[i] = default
                else:
                    res[i] = np.sort(r)[k]
            return res

        expected = get_k_smallest_element_value(distance_matrix, k=1, ignore_zeros=True)
        actual = get_kth_smallest_in_row_legacy(distance_matrix, k=1)

        nptest.assert_array_equal(expected, actual)

    def test_k_smallest_06(self):
        # Test errors
        distance_matrix = compute_distance_matrix(self.example_matrix)

        with self.assertRaises(ValueError):
            get_k_smallest_element_value(distance_matrix, 11)

        with self.assertRaises(ValueError):
            get_k_smallest_element_value(distance_matrix, -1)

    def test_continuous_nn(self):
        kmin = 3  # magic number  # TODO: make as a parameter?
        tol = 1
        distance_matrix = compute_distance_matrix(
            X=self.example_matrix, metric="euclidean", cut_off=150
        )

        def legacy_code_continuous_nn(dmat, k, tolerance):
            def get_kth_largest_in_row(_mat_sp, k, default=0):
                """Returns kth largest element of every row, k is zero based, and zeros in the matrix are ignored """
                res = np.zeros((_mat_sp.shape[0],))
                for i in range(_mat_sp.shape[0]):
                    r = _mat_sp.getrow(i).data
                    r = r[
                        r != 0
                    ]  # added to legacy code, to make sure zeros are ignored
                    if len(r) <= k:
                        res[i] = default
                    else:
                        res[i] = np.sort(r)[k]
                return res

            xk = 1.0 / np.array(get_kth_largest_in_row(dmat, k, 1 / tolerance))
            xk_inv_sp = scipy.sparse.dia_matrix((xk, 0), (xk.shape[0], xk.shape[0]))

            dists_sp = dmat.copy()
            dists_sp.data = np.power(dists_sp.data, 2)
            dists_sp = xk_inv_sp @ dists_sp @ xk_inv_sp
            dists_sp.data = np.sqrt(dists_sp.data)

            epsilon = 0.25
            dists_sp.data[dists_sp.data > 4 * epsilon] = 0
            dists_sp.eliminate_zeros()
            return dists_sp

        distance_matrix_expected = legacy_code_continuous_nn(distance_matrix, kmin, tol)
        distance_matrix_actual = apply_continuous_nearest_neighbor(
            distance_matrix, kmin=kmin, tol=1
        )
        nptest.assert_array_equal(
            distance_matrix_expected.toarray(), distance_matrix_actual.toarray()
        )


class TestDistAlgorithms(unittest.TestCase):
    def setUp(self) -> None:
        self.data_X = np.random.rand(500, 100)
        self.data_Y = np.random.rand(300, 100)

        self.algos = all_available_distance_algorithm()

    def test_pdist_dense(self):
        backend_options = {}
        expected = squareform(pdist(self.data_X))

        for metric in ["euclidean", "sqeuclidean"]:

            if metric == "sqeuclidean":
                expected = np.square(expected)

            for algo in self.algos:

                actual = compute_distance_matrix(
                    X=self.data_X,
                    metric=metric,
                    cut_off=None,
                    kmin=0,
                    tol=1,
                    backend=algo.NAME,
                    **backend_options,
                )

                try:
                    self.assertIsInstance(actual, np.ndarray)
                    nptest.assert_allclose(actual, expected, atol=1e-15, rtol=1e-14)
                except AssertionError as e:
                    print(f"{algo.NAME} failed for metric {metric}")
                    raise e

    def test_cdist_dense(self):
        backend_options = {}

        # NOTE: first Y and then X because, the Y (query points) should be in rows, the X (reference points) in columns
        # This turned out to be a better handling for equations (e.g. in geometric harmonics).
        expected = cdist(self.data_Y, self.data_X)

        for metric in ["euclidean", "sqeuclidean"]:

            if metric == "sqeuclidean":
                expected = np.square(expected)

            for algo in self.algos:

                actual = compute_distance_matrix(
                    X=self.data_X,
                    Y=self.data_Y,
                    metric=metric,
                    cut_off=None,
                    kmin=0,
                    tol=1,
                    backend=algo.NAME,
                    **backend_options,
                )
                try:
                    self.assertIsInstance(actual, np.ndarray)
                    nptest.assert_allclose(actual, expected, atol=1e-15, rtol=1e-14)
                except AssertionError as e:
                    print(f"{algo.NAME} failed for metric {metric}")
                    raise e

    def test_pdist_sparse(self):
        backend_options = {}
        expected = squareform(pdist(self.data_X))
        cut_off = np.median(expected)

        expected[expected > cut_off] = 0

        for metric in ["euclidean", "sqeuclidean"]:

            if metric == "sqeuclidean":
                expected = np.square(expected)

            for algo in self.algos:

                actual = compute_distance_matrix(
                    X=self.data_X,
                    metric=metric,
                    cut_off=cut_off,
                    kmin=0,
                    tol=1,
                    backend=algo.NAME,
                    **backend_options,
                )
                try:
                    self.assertIsInstance(actual, scipy.sparse.csr_matrix)
                    nptest.assert_allclose(
                        actual.toarray(), expected, atol=1e-15, rtol=1e-14
                    )
                except AssertionError as e:
                    print(f"{algo.NAME} failed for metric {metric}")
                    raise e

    def test_cdist_sparse(self):
        backend_options = {}
        expected = cdist(
            self.data_Y, self.data_X
        )  # See also comment in 'test_cdist_dense'
        cut_off = np.median(expected)

        expected[expected > cut_off] = 0

        for metric in ["euclidean", "sqeuclidean"]:

            if metric == "sqeuclidean":
                expected = np.square(expected)

            for algo in self.algos:
                actual = compute_distance_matrix(
                    X=self.data_X,
                    Y=self.data_Y,
                    metric=metric,
                    cut_off=cut_off,
                    kmin=0,
                    tol=1,
                    backend=algo.NAME,
                    **backend_options,
                )
                try:
                    self.assertIsInstance(actual, scipy.sparse.csr_matrix)
                    nptest.assert_allclose(
                        actual.toarray(), expected, atol=1e-15, rtol=1e-14
                    )
                except AssertionError as e:
                    print(f"{algo.NAME} failed")
                    raise e

    def test_pdist_sparse_zeros(self):
        backend_options = {}
        expected = squareform(pdist(self.data_X))
        cut_off = np.median(expected)

        expected[expected > cut_off] = 0
        expected = scipy.sparse.csr_matrix(expected)
        expected.eliminate_zeros()
        expected.setdiag(0)
        expected.sort_indices()

        for metric in ["euclidean", "sqeuclidean"]:

            if metric == "sqeuclidean":
                expected.data = np.square(expected.data)

            for algo in self.algos:

                actual = compute_distance_matrix(
                    X=self.data_X,
                    metric=metric,
                    cut_off=cut_off,
                    kmin=0,
                    tol=1,
                    backend=algo.NAME,
                    **backend_options,
                )
                try:
                    self.assertIsInstance(actual, scipy.sparse.csr_matrix)
                    nptest.assert_allclose(
                        expected.data, actual.data, atol=1e-15, rtol=1e-14
                    )
                except AssertionError as e:
                    print(f"{algo.NAME} failed for metric {metric}")
                    raise e

    def test_cdist_sparse_zeros(self):
        backend_options = {}

        data_Y = self.data_Y.copy()  # make copy to manipulate values
        data_Y[0:3, :] = self.data_X[0:3, :]  # make duplicate values
        expected = cdist(data_Y, self.data_X)
        cut_off = np.median(expected)
        expected[expected > cut_off] = 0

        expected = scipy.sparse.csr_matrix(expected)
        expected[0, 0] = 0
        expected[1, 1] = 0
        expected[2, 2] = 0
        expected.sort_indices()

        for metric in ["euclidean", "sqeuclidean"]:

            if metric == "sqeuclidean":
                expected.data = np.square(expected.data)

            for algo in self.algos:

                actual = compute_distance_matrix(
                    X=self.data_X,
                    Y=data_Y,
                    metric=metric,
                    cut_off=cut_off,
                    kmin=0,
                    tol=1,
                    backend=algo.NAME,
                    **backend_options,
                )
                try:
                    self.assertIsInstance(actual, scipy.sparse.csr_matrix)
                    nptest.assert_allclose(
                        actual.data, expected.data, atol=1e-15, rtol=1e-14
                    )
                except AssertionError as e:
                    print(f"{algo.NAME} failed for metric {metric}")
                    raise e


if __name__ == "__main__":
    td = TestDistAlgorithms()
    td.setUp()
    td.test_pdist_sparse_zeros()
