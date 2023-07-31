import unittest

import numpy as np
import numpy.testing as nptest
import scipy

from datafold.utils.general import (
    diagmat_dot_mat,
    is_matrix,
    is_stochastic_matrix,
    is_symmetric_matrix,
    is_vector,
    mat_dot_diagmat,
    sort_eigenpairs,
)


class TestMathUtils(unittest.TestCase):
    def test_diagmat_dot_mat_dense(self):
        diag_elements = np.random.default_rng(1).random(size=100)
        full_matrix = np.random.default_rng(2).random(size=[100, 100])

        actual = diagmat_dot_mat(diag_elements, full_matrix)
        expected = np.diag(diag_elements) @ full_matrix

        nptest.assert_equal(actual, expected)

    def test_diagmat_dot_mat_sparse(self):
        diag_elements = np.random.default_rng(1).random(size=100)
        full_matrix = np.random.default_rng(2).random(size=[100, 100])
        full_matrix[full_matrix < 0.5] = 0
        full_matrix = scipy.sparse.csr_matrix(full_matrix)

        actual = mat_dot_diagmat(full_matrix, diag_elements)
        self.assertIsInstance(actual, scipy.sparse.csr_matrix)

        expected = full_matrix.toarray() @ np.diag(diag_elements)
        nptest.assert_equal(actual.toarray(), expected)

    def test_mat_dot_diagmat_dense(self):
        diag_elements = np.random.default_rng(3).random(size=100)
        full_matrix = np.random.default_rng(4).random(size=[100, 100])

        actual = mat_dot_diagmat(full_matrix, diag_elements)
        expected = full_matrix @ np.diag(diag_elements)

        nptest.assert_equal(actual, expected)

    def test_mat_dot_diagmat_sparse(self):
        diag_elements = np.random.default_rng(3).random(size=100)
        full_matrix = np.random.default_rng(4).random(size=[100, 100])
        full_matrix[full_matrix < 0.5] = 0
        full_matrix = scipy.sparse.csr_matrix(full_matrix)

        actual = mat_dot_diagmat(full_matrix, diag_elements)
        self.assertIsInstance(actual, scipy.sparse.csr_matrix)

        expected = full_matrix.toarray() @ np.diag(diag_elements)
        nptest.assert_equal(actual.toarray(), expected)

    def test_sort_eigenpairs1(self):
        matrix = np.random.default_rng(1).random(size=[10, 10])

        eigval, eigvec = np.linalg.eig(matrix)

        # make sure the eigenvalues are not already sorted
        self.assertFalse((eigval == np.sort(eigval)[::-1]).all())

        sorted_eigval, sorted_eigvec = sort_eigenpairs(eigval, eigvec)

        self.assertEqual(np.max(eigval), sorted_eigval[0])
        self.assertEqual(np.min(eigval), sorted_eigval[-1])
        self.assertTrue((sorted_eigval == np.sort(sorted_eigval)[::-1]).all())

        reconstruct_vectors_actual = sorted_eigvec @ np.diag(sorted_eigval)
        reconstruct_vectors_expected = eigvec @ np.diag(eigval)

        # need vectors row-wise to better iterate:
        reconstruct_vectors_actual = reconstruct_vectors_actual.T
        reconstruct_vectors_expected = reconstruct_vectors_expected.T

        for col_actual in reconstruct_vectors_actual:
            self.assertIn(col_actual, reconstruct_vectors_expected)

    def test_sort_eigenpairs2(self):
        # same as test_sort_eigenpairs1, but now in reverse order and different seed

        matrix = np.random.default_rng(2).random(size=[10, 10])

        eigval, eigvec = np.linalg.eig(matrix)

        # make sure the eigenvalues are not already sorted
        self.assertFalse((eigval == np.sort(eigval)[::-1]).all())

        sorted_eigval, sorted_eigvec = sort_eigenpairs(eigval, eigvec, ascending=True)

        self.assertEqual(np.max(eigval), sorted_eigval[-1])
        self.assertEqual(np.min(eigval), sorted_eigval[0])
        self.assertTrue((sorted_eigval == np.sort(sorted_eigval)).all())

        reconstruct_vectors_actual = sorted_eigvec @ np.diag(sorted_eigval)
        reconstruct_vectors_expected = eigvec @ np.diag(eigval)

        # need vectors row-wise to better iterate:
        reconstruct_vectors_actual = reconstruct_vectors_actual.T
        reconstruct_vectors_expected = reconstruct_vectors_expected.T

        for col_actual in reconstruct_vectors_actual:
            self.assertIn(col_actual, reconstruct_vectors_expected)

    def test_sort_eigenpairs3(self):
        matrix = np.random.default_rng(3).random(size=[10, 10])

        # NOTE: svd already sorts in descending order
        expected_U, expected_E, expected_V = np.linalg.svd(matrix)

        actual_E, actual_U, actual_V = sort_eigenpairs(
            expected_E, expected_U, left_eigenvectors=expected_V, ascending=False
        )

        nptest.assert_array_equal(actual_U, expected_U)
        nptest.assert_array_equal(actual_E, expected_E)
        nptest.assert_array_equal(actual_V, expected_V)

    def test_is_symmetric_matrix_dense(self):
        nonsymmetric_matrix = np.random.default_rng(6).random(size=[10, 10])
        symmetric_matrix = (nonsymmetric_matrix + nonsymmetric_matrix.T) / 2.0

        self.assertFalse(is_symmetric_matrix(nonsymmetric_matrix))
        self.assertTrue(is_symmetric_matrix(symmetric_matrix, tol=0))

    def test_is_symmetric_matrix_sparse(self):
        nonsymmetric_matrix = np.random.default_rng(5).random(size=[10, 10])
        nonsymmetric_matrix[nonsymmetric_matrix < 0.5] = 0
        symmetric_matrix = (nonsymmetric_matrix + nonsymmetric_matrix.T) / 2.0

        nonsymmetric_matrix = scipy.sparse.csr_matrix(nonsymmetric_matrix)
        symmetric_matrix = scipy.sparse.csr_matrix(symmetric_matrix)

        self.assertFalse(is_symmetric_matrix(nonsymmetric_matrix))
        self.assertTrue(is_symmetric_matrix(symmetric_matrix, tol=0))

    def test_is_stochastic_matrix_dense(self):
        matrix = np.random.default_rng(6).random(size=[10, 9])
        stochastic_matrix_col = matrix / matrix.sum(axis=1)[:, np.newaxis]
        stochastic_matrix_row = matrix / matrix.sum(axis=0)

        self.assertFalse(is_stochastic_matrix(matrix, axis=0))
        self.assertFalse(is_stochastic_matrix(matrix, axis=1))

        self.assertFalse(is_stochastic_matrix(stochastic_matrix_col, axis=0))
        self.assertTrue(is_stochastic_matrix(stochastic_matrix_col, axis=1))

        self.assertTrue(is_stochastic_matrix(stochastic_matrix_row, axis=0))
        self.assertFalse(is_stochastic_matrix(stochastic_matrix_row, axis=1))

    def test_is_stochastic_matrix_sparse(self):
        matrix = np.random.default_rng(6).random(size=[10, 9])
        matrix[matrix < 0.5] = 0
        matrix = scipy.sparse.csr_matrix(matrix)

        stochastic_matrix_col = matrix / matrix.sum(axis=1).A1[:, np.newaxis]
        stochastic_matrix_row = matrix / matrix.sum(axis=0).A1

        self.assertFalse(is_stochastic_matrix(matrix, axis=0))
        self.assertFalse(is_stochastic_matrix(matrix, axis=1))

        self.assertFalse(is_stochastic_matrix(stochastic_matrix_col, axis=0))
        self.assertTrue(is_stochastic_matrix(stochastic_matrix_col, axis=1))

        self.assertTrue(is_stochastic_matrix(stochastic_matrix_row, axis=0))
        self.assertFalse(is_stochastic_matrix(stochastic_matrix_row, axis=1))

    def test_is_matrix(self):
        square_dense = np.random.default_rng(1).random(size=[10, 10])
        rect_dense = np.random.default_rng(1).random(size=[5, 10])

        square_sparse = square_dense.copy()
        square_sparse[square_sparse < 0.5] = 0
        square_sparse = scipy.sparse.csr_matrix(square_sparse)

        rect_sparse = rect_dense.copy()
        rect_sparse[rect_sparse < 0.5] = 0
        rect_sparse = scipy.sparse.csr_matrix(rect_sparse)

        self.assertTrue(is_matrix(square_dense, "m", square=True))
        self.assertTrue(is_matrix(rect_dense, "m"))
        self.assertTrue(is_matrix(square_sparse, "m", allow_sparse=True))
        self.assertTrue(is_matrix(rect_sparse, "m", allow_sparse=True))

        with self.assertRaises(ValueError):
            is_matrix(rect_dense, "m", square=True)

        self.assertFalse(is_matrix(rect_dense, "m", square=True, handle=None))

        with self.assertRaises(TypeError):
            is_matrix(square_sparse, "m", allow_sparse=False)

        self.assertFalse(is_matrix(square_sparse, "m", allow_sparse=False, handle=None))

        with self.assertRaises(TypeError):
            is_matrix(rect_sparse, "m", allow_sparse=False)

        self.assertFalse(is_matrix(rect_sparse, "m", allow_sparse=False, handle=None))

        with self.assertRaises(ValueError):
            is_matrix(rect_sparse, "m", square=True, allow_sparse=True)

        self.assertFalse(
            is_matrix(rect_sparse, "m", square=True, allow_sparse=True, handle=None)
        )

    def test_is_vector(self):
        valid_vector = np.random.default_rng(1).random(size=[5])
        self.assertTrue(is_vector(valid_vector))

        invalid_vector = np.random.default_rng(1).random(size=[5, 5])

        with self.assertRaises(ValueError):
            is_vector(invalid_vector)

        self.assertFalse(is_vector(invalid_vector, handle=None))

        invalid_vector = np.random.default_rng(1).random(size=[0])

        with self.assertRaises(ValueError):
            is_vector(invalid_vector)

        self.assertFalse(is_vector(invalid_vector, handle=None))


if __name__ == "__main__":
    unittest.main()
