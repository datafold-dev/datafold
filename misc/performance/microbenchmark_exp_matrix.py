"""
    This is a micro-benchmark used to verify the improvement of
    exp(large_matrix) using numexpr
"""

import functools
import timeit

import numexpr as ne
import numpy as np
import numpy.testing as nptest

NUMBER_OF_RUNS = 1
N = 5000

matrix = np.random.rand(N, N)

# symmetric case, for methods that try to exploit this
matrix = (matrix + matrix.T) / 2


def normal(m):
    m = m.copy()
    # using
    return np.exp(-0.5 * m)


def numexpr(m):
    m = m.copy()
    # using numpy
    return ne.evaluate("exp(-0.5*m)")


nptest.assert_array_equal(normal(matrix.copy()), numexpr(matrix.copy()))

print(
    f"normal "
    f"{timeit.timeit(functools.partial(normal, matrix.copy()), number=NUMBER_OF_RUNS)}"
)

print(
    f"numexpr"
    f" {timeit.timeit(functools.partial(numexpr, matrix.copy()), number=NUMBER_OF_RUNS)}"
)
