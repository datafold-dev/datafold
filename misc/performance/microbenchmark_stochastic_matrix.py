#!/usr/bin/env python3

import timeit

import numexpr
import numpy as np
import scipy.sparse
from sklearn.preprocessing import normalize

N = 10000
matrix_dense = np.random.rand(N, N)

density = 0.05
matrix_sparse = scipy.sparse.random(N, N, density).tocsr()


def np_dense():
    inv_diag = np.reciprocal(np.sum(matrix_dense, axis=1))
    return np.multiply(inv_diag[:, np.newaxis], matrix_dense)


def np_dense_numexpr():
    row_sum = numexpr.evaluate("sum(matrix_dense, 1)")
    inv_diag = numexpr.evaluate("1.0 / row_sum", out=row_sum)
    return numexpr.evaluate("inv_diag * matrix_dense")


def scikit_learn_dense():
    return normalize(matrix_dense, copy=False, norm="l1")


def csr_sparse():
    data = matrix_sparse.data
    indptr = matrix_sparse.indptr
    for i in range(matrix_sparse.shape[0]):
        a, b = indptr[i : i + 2]
        norm1 = np.sum(data[a:b])
        data[a:b] /= norm1

    return matrix_sparse


def csr_sparse2():
    row_sum = matrix_sparse.sum(axis=1)
    row_sum = np.reciprocal(row_sum, out=row_sum)
    diag = scipy.sparse.spdiags(np.ravel(row_sum), 0, *matrix_sparse.shape)
    return matrix_sparse @ diag


def scikit_learn_sparse():
    return normalize(matrix_sparse, copy=False, norm="l1")


NUMBER_OF_RUNS = 10
testcase = ["dense", "sparse"][0]

if testcase == "dense":
    time_np_dense = timeit.timeit(np_dense, number=NUMBER_OF_RUNS)
    print(f"time_np_dense {time_np_dense}")

    time_np_dense_numexpr = timeit.timeit(np_dense_numexpr, number=NUMBER_OF_RUNS)
    print(f"time_np_dense_numexpr {time_np_dense_numexpr}")

    time_scikit_learn_dense = timeit.timeit(scikit_learn_dense, number=NUMBER_OF_RUNS)
    print(f"time_scikit_learn_dense {time_scikit_learn_dense}")

    assert np.allclose(np_dense(), np_dense_numexpr(), rtol=1e-15, atol=1e-15)
    assert np.allclose(np_dense(), scikit_learn_dense(), rtol=1e-15, atol=1e-15)

else:  # sparse
    time_csr_sparse = timeit.timeit(csr_sparse, number=NUMBER_OF_RUNS)
    print(f"time_csr_sparse {time_csr_sparse}")

    time_csr_sparse2 = timeit.timeit(csr_sparse2, number=NUMBER_OF_RUNS)
    print(f"time_csr_sparse2 {time_csr_sparse2}")

    time_scikit_learn_sparse = timeit.timeit(scikit_learn_sparse, number=NUMBER_OF_RUNS)
    print(f"time_scikit_learn_sparse {time_scikit_learn_sparse}")

    assert np.allclose(
        scikit_learn_sparse().toarray(), csr_sparse().toarray(), rtol=1e-15, atol=1e-15
    )
    assert np.allclose(
        scikit_learn_sparse().toarray(), csr_sparse2().toarray(), rtol=1e-15, atol=1e-15
    )

print("tests successful")
