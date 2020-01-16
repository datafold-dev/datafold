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

NUMBER_OF_RUNS = 1
N = 5000

matrix = np.random.rand(N, N)

# symmetric
matrix = (matrix + matrix.T) / 2


def normal(m):
    m = m.copy()
    # Current implemented version
    return np.exp(m)


def numexpr(m):
    m = m.copy()
    # Current implemented version
    return ne.evaluate("exp(m)")


def exploit_symmetry(m):
    m_new = np.zeros_like(m)
    idx = np.triu_indices(m_new.shape[0])
    m_new[idx] = np.exp(m_new[idx])
    m_new + m_new.T - np.diag(m_new.diagonal())
    return m


print(
    f"normal "
    f"{timeit.timeit(functools.partial(normal, matrix.copy()), number=NUMBER_OF_RUNS)}"
)

print(
    f"numexpr"
    f" {timeit.timeit(functools.partial(numexpr, matrix.copy()), number=NUMBER_OF_RUNS)}"
)


print(
    f"symmetry "
    f"{timeit.timeit(functools.partial(exploit_symmetry, matrix.copy()), number=NUMBER_OF_RUNS)}"
)
exit()

if numexpr_bm:
    print(
        f"n2 {timeit.timeit(functools.partial(n2_numexpr, a, b, c), number=NUMBER_OF_RUNS)}"
    )

print(f"n3 {timeit.timeit(functools.partial(n3, a,b,c), number=NUMBER_OF_RUNS)}")
print(
    f"n3_alternative {timeit.timeit(functools.partial(n3_alternative, a,b,c), number=NUMBER_OF_RUNS)}"
)
