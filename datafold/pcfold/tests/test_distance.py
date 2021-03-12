#!/usr/bin/env python

import unittest
import warnings

import numpy as np
import numpy.testing as nptest
import scipy
import scipy.sparse
from scipy.sparse.base import SparseEfficiencyWarning
from scipy.spatial.distance import cdist, pdist, squareform

from datafold.pcfold.distance import (
    _all_available_distance_algorithm,
    _ensure_kmin_nearest_neighbor,
    compute_distance_matrix,
)
from datafold.utils.general import is_symmetric_matrix


class TestDistAlgorithms(unittest.TestCase):
    def setUp(self) -> None:
        self.data_X = np.random.rand(500, 100)
        self.data_Y = np.random.rand(300, 100)

        self.algos = _all_available_distance_algorithm()

    def test_pdist_dense(self):
        backend_options = {}
        expected = squareform(pdist(self.data_X))

        for metric in ["euclidean", "sqeuclidean"]:

            if metric == "sqeuclidean":
                expected = np.square(expected)

            for algo in self.algos:

                try:
                    actual = compute_distance_matrix(
                        X=self.data_X,
                        metric=metric,
                        cut_off=None,
                        kmin=0,
                        backend=algo.backend_name,
                        **backend_options,
                    )

                    self.assertIsInstance(actual, np.ndarray)
                    nptest.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
                except AssertionError as e:
                    print(f"{algo.backend_name} failed for metric {metric}")
                    raise e

    def test_cdist_dense(self):
        backend_options = {}

        # NOTE: first Y and then X because, the Y (query points) should be in rows, the X
        # (reference points) in columns This turned out to be a better handling for
        # equations (e.g. in geometric harmonics).
        expected = cdist(self.data_Y, self.data_X)

        for metric in ["euclidean", "sqeuclidean"]:

            if metric == "sqeuclidean":
                expected = np.square(expected)

            for algo in self.algos:
                try:
                    actual = compute_distance_matrix(
                        X=self.data_X,
                        Y=self.data_Y,
                        metric=metric,
                        cut_off=None,
                        kmin=0,
                        backend=algo.backend_name,
                        **backend_options,
                    )

                    self.assertIsInstance(actual, np.ndarray)
                    nptest.assert_allclose(actual, expected, atol=1e-15, rtol=1e-14)
                except Exception as e:
                    print(f"{algo.backend_name} failed for metric {metric}")
                    raise e

    def test_pdist_sparse(self):
        backend_options = {}
        expected = squareform(pdist(self.data_X))
        cut_off = float(np.median(expected))

        expected[expected > cut_off] = 0

        for metric in ["euclidean", "sqeuclidean"]:

            if metric == "sqeuclidean":
                expected = np.square(expected)

            for algo in self.algos:
                try:
                    actual = compute_distance_matrix(
                        X=self.data_X,
                        metric=metric,
                        cut_off=cut_off,
                        kmin=0,
                        backend=algo.backend_name,
                        **backend_options,
                    )

                    self.assertIsInstance(actual, scipy.sparse.csr_matrix)
                    nptest.assert_allclose(
                        actual.toarray(), expected, atol=1e-14, rtol=1e-14
                    )

                    self.assertTrue(is_symmetric_matrix(actual, tol=0))

                except Exception as e:
                    print(f"{algo.backend_name} failed for metric {metric}")
                    raise e

    def test_cdist_sparse(self):
        backend_options = {}

        # See also comment in 'test_cdist_dense'
        expected = cdist(self.data_Y, self.data_X)
        cut_off = float(np.median(expected))

        expected[expected > cut_off] = 0

        for metric in ["euclidean", "sqeuclidean"]:

            if metric == "sqeuclidean":
                expected = np.square(expected)

            for algo in self.algos:
                try:
                    actual = compute_distance_matrix(
                        X=self.data_X,
                        Y=self.data_Y,
                        metric=metric,
                        cut_off=cut_off,
                        kmin=0,
                        backend=algo.backend_name,
                        **backend_options,
                    )

                    self.assertIsInstance(actual, scipy.sparse.csr_matrix)
                    nptest.assert_allclose(
                        actual.toarray(), expected, atol=1e-15, rtol=1e-14
                    )
                except Exception as e:
                    print(f"{algo.backend_name} failed with metric {metric}")
                    raise e

    def test_pdist_sparse_zeros(self):
        backend_options = {}
        expected = squareform(pdist(self.data_X))
        cut_off = float(np.median(expected))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseEfficiencyWarning)
            expected[expected > cut_off] = 0
            expected = scipy.sparse.csr_matrix(expected)
            expected.eliminate_zeros()
            expected.setdiag(0)
            expected.sort_indices()

        for metric in ["euclidean", "sqeuclidean"]:

            if metric == "sqeuclidean":
                expected.data = np.square(expected.data)

            for algo in self.algos:

                try:
                    actual = compute_distance_matrix(
                        X=self.data_X,
                        metric=metric,
                        cut_off=cut_off,
                        kmin=0,
                        backend=algo.backend_name,
                        **backend_options,
                    )

                    self.assertTrue(is_symmetric_matrix(actual, tol=0))
                    self.assertIsInstance(actual, scipy.sparse.csr_matrix)
                    nptest.assert_allclose(
                        expected.data, actual.data, atol=1e-14, rtol=1e-14
                    )
                except Exception as e:
                    print(f"{algo.backend_name} failed for metric {metric}")
                    raise e

    def test_cdist_sparse_duplicate_zeros(self):
        backend_options = {}

        data_Y = self.data_Y.copy()  # make copy to manipulate values
        data_Y[0:3, :] = self.data_X[0:3, :]  # make duplicate values
        expected = cdist(data_Y, self.data_X)
        cut_off = float(np.median(expected))
        expected[expected > cut_off] = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseEfficiencyWarning)

            expected = scipy.sparse.csr_matrix(expected)
            expected[0, 0] = 0
            expected[1, 1] = 0
            expected[2, 2] = 0
            expected.sort_indices()

        for metric in ["euclidean", "sqeuclidean"]:

            if metric == "sqeuclidean":
                expected.data = np.square(expected.data)

            for algo in self.algos:

                try:
                    actual = compute_distance_matrix(
                        X=self.data_X,
                        Y=data_Y,
                        metric=metric,
                        cut_off=cut_off,
                        backend=algo.backend_name,
                        **backend_options,
                    )

                    self.assertIsInstance(actual, scipy.sparse.csr_matrix)
                    nptest.assert_allclose(
                        actual.data, expected.data, atol=1e-15, rtol=1e-14
                    )

                except Exception as e:
                    print(f"{algo.backend_name} failed for metric {metric}")
                    raise e

    def test_ensure_kmin_nearest_neighbours_pdist(self):

        print("SUPRESSED SPARSITY WARNINGS. TODO: See #93")
        warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

        for quantile in [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]:

            for kmin in np.linspace(1, self.data_X.shape[1], 5).astype(int):

                cut_off = np.quantile(pdist(self.data_X), q=quantile)
                # The matrix is essentially zero, with only the diagonal saved zeros
                pdist_distance_matrix = compute_distance_matrix(
                    self.data_X, cut_off=cut_off
                )

                distance_matrix = _ensure_kmin_nearest_neighbor(
                    self.data_X,
                    Y=None,
                    metric="euclidean",
                    kmin=kmin,
                    distance_matrix=pdist_distance_matrix,
                )

                try:
                    self.assertTrue((distance_matrix.getnnz(axis=1) >= kmin).all())
                    self.assertTrue(is_symmetric_matrix(distance_matrix))

                    rows, columns = distance_matrix.nonzero()
                    actual = scipy.sparse.csr_matrix(
                        (
                            pdist_distance_matrix[rows, columns].A1,
                            (rows, columns),
                        ),
                        shape=distance_matrix.shape,
                    )
                    self.assertTrue(is_symmetric_matrix(actual))
                    nptest.assert_array_equal(
                        actual.toarray(),
                        distance_matrix.toarray(),
                    )
                except AssertionError as e:
                    print(f"Failed for quantile={quantile} and kmin={kmin}")
                    raise e

    def test_ensure_kmin_nearest_neighbours_cdist(self):

        print("SUPRESSED SPARSITY WARNINGS. TODO: See #93")
        warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

        for quantile in [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]:

            for kmin in np.linspace(1, self.data_X.shape[1], 5).astype(int):

                cut_off = np.quantile(pdist(self.data_X), q=quantile)
                # The matrix is essentially zero, with only the diagonal saved zeros
                cdist_distance_matrix = compute_distance_matrix(
                    self.data_X, Y=self.data_Y, cut_off=cut_off
                )

                # TODO: resolve SparsityWarning, see issue #93
                distance_matrix = _ensure_kmin_nearest_neighbor(
                    self.data_X,
                    Y=self.data_Y,
                    metric="euclidean",
                    kmin=kmin,
                    distance_matrix=cdist_distance_matrix,
                )

                try:

                    rows, columns = distance_matrix.nonzero()
                    actual = scipy.sparse.csr_matrix(
                        (
                            cdist_distance_matrix[rows, columns].A1,
                            (rows, columns),
                        ),
                        shape=distance_matrix.shape,
                    )
                    nptest.assert_array_equal(
                        actual.toarray(), distance_matrix.toarray()
                    )
                    self.assertTrue((distance_matrix.getnnz(axis=1) >= kmin).all())
                except AssertionError as e:
                    print(f"Failed for quantile={quantile} and kmin={kmin}")
                    raise e
