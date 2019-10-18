#!/usr/bin/env python

import unittest
import numpy.testing as nptest

from datafold.pcfold.distance import *


class TestDistance(unittest.TestCase):

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

        nptest.assert_array_equal(dense_dist_mat, compute_distance_matrix(self.example_matrix))
        nptest.assert_array_equal(sparse_dist_mat.toarray(), compute_distance_matrix(self.example_matrix, cut_off=150).toarray())

    def test_k_smallest03(self):
        # sanity test for pdist and ignore zeros and k=0 should always be 0

        dense_dist_mat = compute_distance_matrix(self.example_matrix)
        sparse_dist_mat = compute_distance_matrix(self.example_matrix, cut_off=150)

        dense_actual = get_k_smallest_element_value(dense_dist_mat, 0, ignore_zeros=False)
        sparse_actual = get_k_smallest_element_value(sparse_dist_mat, 0, ignore_zeros=False)

        expected = dense_dist_mat.min(axis=1)  # self is always zero

        nptest.assert_array_equal(dense_actual, expected)
        nptest.assert_array_equal(sparse_actual, expected)

    def test_k_smallest04(self):
        # Test from legacy code in estimators.py
        n_points = self.example_matrix.shape[0]
        perm_indices_all = np.random.permutation(np.arange(n_points))

        # only dense case
        distance_matrix = compute_distance_matrix(self.example_matrix[perm_indices_all[:3], :], self.example_matrix,
                                                  metric="euclidean")

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
                r = r[r != 0]  # added to legacy code to make sure zeros are completely ignored

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
        distance_matrix = compute_distance_matrix(X=self.example_matrix, metric="euclidean", cut_off=150)

        def legacy_code_continuous_nn(dmat, k, tolerance):

            def get_kth_largest_in_row(_mat_sp, k, default=0):
                """Returns kth largest element of every row, k is zero based, and zeros in the matrix are ignored """
                res = np.zeros((_mat_sp.shape[0],))
                for i in range(_mat_sp.shape[0]):
                    r = _mat_sp.getrow(i).data
                    r = r[r != 0]  # added to legacy code, to make sure zeros are ignored
                    if len(r) <= k:
                        res[i] = default
                    else:
                        res[i] = np.sort(r)[k]
                return res

            xk = (1. / np.array(get_kth_largest_in_row(dmat, k, 1 / tolerance)))
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
        distance_matrix_actual = apply_continuous_nearest_neighbor(distance_matrix, kmin=kmin, tol=1)
        nptest.assert_array_equal(distance_matrix_expected.toarray(), distance_matrix_actual.toarray())


if __name__ == "__main__":
    td = TestDistance()
    td.setUp()
    td.test_continuous_nn()