import unittest

import numpy as np
import numpy.testing as nptest

from datafold.utils.general import *


class TestMathUtils(unittest.TestCase):
    def test_diagmat_dot_mat(self):
        diag_elements = np.random.rand(100)
        full_matrix = np.random.rand(100, 100)

        actual = diagmat_dot_mat(diag_elements, full_matrix)
        expected = np.diag(diag_elements) @ full_matrix

        nptest.assert_equal(actual, expected)

    def test_mat_dot_diagmat(self):
        diag_elements = np.random.rand(100)
        full_matrix = np.random.rand(100, 100)

        actual = mat_dot_diagmat(full_matrix, diag_elements)
        expected = full_matrix @ np.diag(diag_elements)

        nptest.assert_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
