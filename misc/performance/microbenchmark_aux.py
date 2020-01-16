"""
    This is a micro-benchmark used to verify the improvement
    in the speed of the computation of the aux-vector as used in geometrics-harmonics
"""

import functools
import timeit
import numpy as np

try:
    import numexpr as ne
except ImportError:
    numexpr_bm = False
else:
    numexpr_bm = True

NUMBER_OF_RUNS = 1000
N, M = 10000, 5000

a = np.random.rand(M, N)
b = np.random.rand(M)
c = np.random.rand(N)


def n2(a, b, c):
    # Current implemented version
    return (a.T * (1.0 / b)) @ (a @ c)


def n2_alternative(a, b, c):
    # Current implemented version
    return np.linalg.multi_dot([(a.T * (1.0 / b)), a, c])


def n2_numexpr(a, b, c):
    # Mainly for testing the numexpr library, which is particular good for elementwise
    # computation. Even though it is faster, it is not implemented because it
    # introduces another package dependency.
    # About 1,3 - 1.4 faster for (N, M, NUMBER_OF_RUNS = 100000, 500, 300)
    # For smaller problems it is often slower (which can be neglected then anyway).

    d = a.T  # transpose is not supported in numexpr.evaluate

    return ne.evaluate("d * (1. / b)") @ (a @ c)


def n3_alternative(a, b, c):
    return ((a.T * (1.0 / b)) @ a) @ c


def n3(a, b, c):
    return a.T @ np.diag(1.0 / b) @ a @ c  # Original version


print(f"n2 {timeit.timeit(functools.partial(n2, a,b,c), number=NUMBER_OF_RUNS)}")

print(
    f"n2_alternative"
    f" {timeit.timeit(functools.partial(n2_alternative, a,b,c), number=NUMBER_OF_RUNS)}"
)

if numexpr_bm:
    print(
        f"n2 {timeit.timeit(functools.partial(n2_numexpr, a, b, c), number=NUMBER_OF_RUNS)}"
    )

print(f"n3 {timeit.timeit(functools.partial(n3, a,b,c), number=NUMBER_OF_RUNS)}")
print(
    f"n3_alternative {timeit.timeit(functools.partial(n3_alternative, a,b,c), number=NUMBER_OF_RUNS)}"
)
