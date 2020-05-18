#!/usr/bin/env python3

from time import time

import numpy as np
import scipy.sparse
from numba import jit, njit, prange


def _kth_dist_sparse_own_order(distance_matrix: scipy.sparse.csr_matrix, k_neighbor):
    @njit(fastmath=True)
    def _get_kth_largest_elements_sparse(
        data: np.ndarray, indptr: np.ndarray, row_nnz, k: int,
    ):
        dist_knn = np.zeros(len(row_nnz))

        for i in range(len(row_nnz)):

            start_row = indptr[i]
            current_array = np.sort(data[start_row : start_row + k])

            for j in range(k, row_nnz[i]):
                if data[start_row + j] < current_array[k - 1]:
                    ctr = 0
                    while data[start_row + j] > current_array[ctr]:
                        ctr += 1

                    current_array[ctr + 1 :] = current_array[ctr : k - 1]
                    current_array[ctr] = data[start_row + j]

            dist_knn[i] = current_array[k - 1]

        return dist_knn

    row_nnz = distance_matrix.getnnz(axis=1)

    if (row_nnz < k_neighbor).any():
        raise ValueError("")

    return _get_kth_largest_elements_sparse(
        distance_matrix.data, distance_matrix.indptr, row_nnz, k_neighbor,
    )


def _kth_dist_sparse_partition(distance_matrix: scipy.sparse.csr_matrix, k_neighbor):
    # @jit(nopython=True)
    def _get_kth_largest_elements_sparse(
        data: np.ndarray, indptr: np.ndarray, row_nnz, k: int,
    ):
        dist_knn = np.zeros(len(row_nnz))
        for i in prange(len(row_nnz)):
            start_row = indptr[i]
            dist_knn[i] = np.partition(data[start_row : start_row + row_nnz[i]], k - 1)[
                k - 1
            ]

        return dist_knn

    row_nnz = distance_matrix.getnnz(axis=1)

    if (row_nnz < k_neighbor).any():
        raise ValueError("")

    return _get_kth_largest_elements_sparse(
        distance_matrix.data, distance_matrix.indptr, row_nnz, k_neighbor,
    )


A = np.random.rand(3000, 30000)
# A[A < 0.8] = 0

k = 2
start = time()
print(f"partition {np.partition(A, kth=k-1, axis=1)[:, k-1]}")
print(f"this took {time()- start}")

A_sparse = scipy.sparse.csr_matrix(A)
start = time()

dist_knn = np.zeros(A.shape[0])
print(f"own COMPILED " f"{_kth_dist_sparse_own_order(A_sparse, k_neighbor=k)}")
print(f"this took {time() - start}")

start = time()
print(f"own AFTER COMPILE " f"{_kth_dist_sparse_own_order(A_sparse, k_neighbor=k)}")
print(f"this took {time() - start}")

dist_knn = np.zeros(A.shape[0])
start = time()
print(f"partition " f"{_kth_dist_sparse_partition(A_sparse, k_neighbor=k)}")
print(f"this took {time() - start}")
