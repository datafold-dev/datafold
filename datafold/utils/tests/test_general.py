import unittest

from datafold.utils.general import *


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


if __name__ == "__main__":
    unittest.main()
