"""
    This is a micro-benchmark used to verify the improvement of
    exp(large_matrix)
"""

import functools
import timeit

import numexpr as ne
import numpy as np
import numpy.testing as nptest

NUMBER_OF_RUNS = 1
N = 7000

diag = np.random.rand(N)
matrix = np.random.rand(N, N)


def normal(d, m):
    return np.multiply(m, d[:, np.newaxis])


def numexpr(d, m):
    d = d[:, np.newaxis]
    return ne.evaluate("m * d")


nptest.assert_array_equal(normal(diag, matrix), numexpr(diag, matrix))

t = timeit.timeit(functools.partial(normal, diag, matrix), number=NUMBER_OF_RUNS)
print(f"normal {t}")

t = timeit.timeit(functools.partial(numexpr, diag, matrix), number=NUMBER_OF_RUNS)
print(f"numexpr {t}")
