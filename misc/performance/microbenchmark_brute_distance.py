"""
    This is a micro-benchmark used to verify which is the better dense distance matrix
    algorithm.

    Result: The numexpr based brute distance outperforms pdist from
"""

import functools
import timeit

import numpy as np
from sklearn.datasets import make_swiss_roll

from datafold.pcfold.distance import compute_distance_matrix

NUMBER_OF_RUNS = 1
N = 7000

rand_matrix = np.random.rand(N, 1024)
swiss_matrix = make_swiss_roll(N)[0]

used_matrix = rand_matrix


# Calling brute_numexpr results in compiling it already!

cut_off = None


def with_backend(m, backend, **kwargs):
    compute_distance_matrix(
        X=m, metric="euclidean", backend=backend, cut_off=cut_off, **kwargs
    )


t = timeit.timeit(
    functools.partial(with_backend, used_matrix, "brute"), number=NUMBER_OF_RUNS,
)
print(f"brute pdist {t} s")


t = timeit.timeit(
    functools.partial(with_backend, used_matrix, "brute", exact_numeric=False),
    number=NUMBER_OF_RUNS,
)
print(f"brute numexpr {t} s")
