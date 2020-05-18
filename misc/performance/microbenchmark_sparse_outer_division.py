#!/usr/bin/env python3

from time import time

import numpy as np
import scipy
import scipy.sparse
from numba import njit

from datafold.pcfold.kernels import _symmetric_matrix_division


def _outer_division_sparse(
    distance_matrix: scipy.sparse.csr_matrix, dist_knn, reference_dist_knn
):

    if reference_dist_knn is None:
        reference_dist_knn = dist_knn.view()

    @njit(parallel=False)  # TODO: check if it can be parallelized?
    def sparse_division(data, indptr, indices, dist_knn, reference_dist_knn):
        for row in range(len(indptr) - 1):
            n_elements_row = indptr[row + 1] - indptr[row]
            row_start = indptr[row]

            for column_idx in range(n_elements_row):
                idx = row_start + column_idx
                data[idx] /= np.sqrt(dist_knn[row] * reference_dist_knn[indices[idx]])

        return data

    distance_matrix.data = sparse_division(
        distance_matrix.data,
        distance_matrix.indptr,
        distance_matrix.indices,
        dist_knn=dist_knn,
        reference_dist_knn=reference_dist_knn,
    )
    return distance_matrix


def _outer_division_dia(
    distance_matrix: scipy.sparse.csr_matrix, dist_knn, reference_dist_knn
):
    left = scipy.sparse.dia_matrix(
        (np.reciprocal(np.sqrt(dist_knn)), np.array([0])),
        shape=(dist_knn.shape[0], dist_knn.shape[0]),
    )

    right = scipy.sparse.dia_matrix(
        (np.reciprocal(np.sqrt(reference_dist_knn)), np.array([0])),
        shape=(reference_dist_knn.shape[0], reference_dist_knn.shape[0]),
    )

    return left @ distance_matrix @ right


n_samples_X = 20
n_samples_Y = 20

distance_matrix_orig = np.random.rand(n_samples_Y, n_samples_X)
distance_matrix_orig[distance_matrix_orig < 0.9] = 0
division_X = np.random.rand(n_samples_X)
division_Y = np.random.rand(n_samples_Y)

print_result = False

print("----")

distance_matrix = distance_matrix_orig.copy()
start = time()
result = distance_matrix / np.sqrt(np.outer(division_Y, division_X))
print(f"runtime numpy.ndarray = {time() - start}")
if print_result:
    print(result)

print("----")

distance_matrix = scipy.sparse.csr_matrix(distance_matrix_orig, copy=True)
start = time()
result = _outer_division_sparse(distance_matrix.copy(), division_Y, division_X)
print(f"runtime _outer_division_sparse = {time() - start}")
if print_result:
    print(result.toarray())

print("----")

distance_matrix = distance_matrix_orig.copy()
start = time()
result = _symmetric_matrix_division(
    distance_matrix, np.sqrt(division_Y), np.sqrt(division_X)
)
print(f"runtime _symmetric_division dense = {time() - start}")

if print_result:
    print(result)

print("----")

distance_matrix = scipy.sparse.csr_matrix(distance_matrix_orig, copy=True)
start = time()
result = _symmetric_matrix_division(
    distance_matrix, np.sqrt(division_Y), np.sqrt(division_X)
)
print(f"runtime _symmetric_matrix_division sparse = {time() - start}")
if print_result:
    print(result.toarray())
