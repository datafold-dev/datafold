import functools
import timeit

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import eigs, eigsh
from sklearn.datasets import make_swiss_roll

from datafold.pcfold.distance import compute_distance_matrix
from datafold.pcfold.kernels import DmapKernelFixed, DmapKernelVariable
from datafold.utils.general import sort_eigenpairs

data = make_swiss_roll(3000)[0]
distance_matrix = compute_distance_matrix(X=data, metric="sqeuclidean")

dist_mat_sparsity_quantile = 1
kernel_sparsity_cut_off = 1e-17
add_numeric_break = 1e-13
nr_evec = 100

init_vec_mode = ["ones", "random"][0]
if init_vec_mode == "ones":
    initial_vec = np.ones(data.shape[0])
else:
    initial_vec = np.random.rand(data.shape[0])


NUMBER_OF_RUNS = 2

compute_exact = False

print("SUMMARY OF BENCHMARK")
print(f"NUMER OF RUNS = {NUMBER_OF_RUNS}")
print(f"initial_vec is in mode '{init_vec_mode}'")
print(f"kernel_sparsity_cut_off={kernel_sparsity_cut_off}")
print(f"number_break_exact_sigma={add_numeric_break}")
print(f"compute_exact={compute_exact}")
print(f"number_evec={nr_evec}")


if dist_mat_sparsity_quantile < 1:
    print(f"use sparsity level {0.5}")
    assert 0 < dist_mat_sparsity_quantile < 1
    cut_off = np.quantile(distance_matrix, dist_mat_sparsity_quantile)
    distance_matrix_sparse = distance_matrix.copy()
    distance_matrix_sparse[distance_matrix_sparse < cut_off] = 0
    distance_matrix_sparse = csr_matrix(distance_matrix_sparse)
else:
    distance_matrix_sparse = distance_matrix

dmap_fixed_symmetric_stoch, _, _ = DmapKernelFixed(
    epsilon=1, symmetrize_kernel=True
).eval(distance_matrix_sparse, is_pdist=True)

dmap_fixed_symmetric_nonstoch, _, _ = DmapKernelFixed(
    epsilon=1, is_stochastic=False, symmetrize_kernel=True
).eval(distance_matrix_sparse, is_pdist=True)


dmap_fixed_nonsymmetric_stoch, _, _ = DmapKernelFixed(
    epsilon=1, symmetrize_kernel=False
).eval(distance_matrix_sparse, is_pdist=True)

dmap_fixed_nonsymmetric_nonstoch, _, _ = DmapKernelFixed(
    epsilon=1, is_stochastic=False, symmetrize_kernel=False
).eval(distance_matrix_sparse, is_pdist=True)

dmap_variable_symmetric = DmapKernelVariable(
    epsilon=1.25, k=200, expected_dim=2, beta=-1 / 2, symmetrize_kernel=True
).eval(distance_matrix)[0]


def print_first_eigval(eigval):
    NUMBER = 3
    print(
        f"fist {NUMBER} eigval (only real part): {np.sort(eigval)[::-1][:NUMBER].real}"
    )


def print_true_solution(kernel_matrix):
    eival, eivec = np.linalg.eig(kernel_matrix)
    eival, eivec = sort_eigenpairs(eival, eivec)
    print("TRUE VALUES:")
    print_first_eigval(eival)
    print(str_sep)


def case_eigd(m):
    eigval, eigvec = np.linalg.eig(m)
    print_first_eigval(eigval)
    eigval = eigval[:nr_evec]
    eigvec = eigvec[:, :nr_evec]


def case_eigdh(m):
    eigval, eigvec = np.linalg.eigh(m)
    print_first_eigval(eigval)
    eigval = eigval[:nr_evec]
    eigvec = eigvec[:, :nr_evec]


def case_eigs_sigma(m, s):
    eigval, _ = eigs(m, k=nr_evec, sigma=s, which="LM", v0=initial_vec)
    print_first_eigval(eigval)


def case_eigs_sigma_sparsify(m, sigma):
    m[np.abs(m) < kernel_sparsity_cut_off] = 0
    m = csr_matrix(m)
    eigval, eigvec = eigs(m, k=nr_evec, sigma=sigma, which="LM", v0=initial_vec)
    print_first_eigval(eigval)


def case_eigsh_sigma(m, s):
    eigval, eigvec = eigsh(
        m, k=nr_evec, sigma=s, which="LM", mode="cayley", v0=initial_vec
    )
    print_first_eigval(eigval)


def case_eigsh_sigma_sparsify(m, sigma):
    m[np.abs(m) < kernel_sparsity_cut_off] = 0
    m = csr_matrix(m)
    eigval, eigvec = eigsh(m, k=nr_evec, sigma=sigma, which="LM", v0=initial_vec)
    print_first_eigval(eigval)


str_sep = "-----------------------------------------------------"

print("\n\nSTOCHASTIC KERNEL MATRIX")

if compute_exact:
    print_true_solution(dmap_fixed_nonsymmetric_stoch)

    t = timeit.timeit(
        functools.partial(case_eigd, dmap_fixed_nonsymmetric_stoch),
        number=NUMBER_OF_RUNS,
    )
    print(f"case_eigd {t}")
    print(str_sep)

sigma = 1
t = timeit.timeit(
    functools.partial(case_eigs_sigma, dmap_fixed_nonsymmetric_stoch, sigma),
    number=NUMBER_OF_RUNS,
)
print(f"case_eigs_sigma_{sigma} {t}")
print(str_sep)

sigma = None
t = timeit.timeit(
    functools.partial(case_eigs_sigma, dmap_fixed_nonsymmetric_stoch, sigma),
    number=NUMBER_OF_RUNS,
)
print(f"case_eigs_sigma_{sigma} {t}")
print(str_sep)

t = timeit.timeit(
    functools.partial(case_eigs_sigma_sparsify, dmap_fixed_nonsymmetric_stoch, sigma),
    number=NUMBER_OF_RUNS,
)
print(f"case_eigs_sigma_sparsify_sigma_{sigma} {t}")
print(str_sep)

if compute_exact:
    t = timeit.timeit(
        functools.partial(case_eigdh, dmap_fixed_symmetric_stoch),
        number=NUMBER_OF_RUNS,
    )
    print(f"case_eigdh {t}")
    print(str_sep)


sigma = 1 + add_numeric_break
t = timeit.timeit(
    functools.partial(case_eigsh_sigma, dmap_fixed_symmetric_stoch, sigma),
    number=NUMBER_OF_RUNS,
)
print(f"case_eigsh_sigma_{sigma} {t}")
print(str_sep)

sigma = None
t = timeit.timeit(
    functools.partial(case_eigsh_sigma, dmap_fixed_symmetric_stoch, sigma),
    number=NUMBER_OF_RUNS,
)
print(f"case_eigsh_sigma_{sigma} {t}")
print(str_sep)

t = timeit.timeit(
    functools.partial(case_eigs_sigma_sparsify, dmap_fixed_symmetric_stoch, sigma),
    number=NUMBER_OF_RUNS,
)
print(f"case_eigsh_sigma_sparsify_sigma_{sigma} {t}")
print(str_sep)

print("\n\nNON-STOCHASTIC KERNEL MATRIX")

if compute_exact:
    print_true_solution(dmap_fixed_nonsymmetric_nonstoch)

    t = timeit.timeit(
        functools.partial(case_eigd, dmap_fixed_nonsymmetric_nonstoch),
        number=NUMBER_OF_RUNS,
    )
    print(f"case_eigd {t}")
    print(str_sep)

sigma = 1
t = timeit.timeit(
    functools.partial(case_eigs_sigma, dmap_fixed_nonsymmetric_nonstoch, sigma),
    number=NUMBER_OF_RUNS,
)
print(f"case_eigs_sigma_{sigma} {t}")
print(str_sep)

sigma = None
t = timeit.timeit(
    functools.partial(case_eigs_sigma, dmap_fixed_nonsymmetric_nonstoch, sigma),
    number=NUMBER_OF_RUNS,
)
print(f"case_eigs_sigma_{sigma} {t}")
print(str_sep)

t = timeit.timeit(
    functools.partial(
        case_eigs_sigma_sparsify, dmap_fixed_nonsymmetric_nonstoch, sigma
    ),
    number=NUMBER_OF_RUNS,
)
print(f"case_eigs_sigma_sparsify_sigma_{sigma} {t}")
print(str_sep)

if compute_exact:
    t = timeit.timeit(
        functools.partial(case_eigdh, dmap_fixed_symmetric_nonstoch),
        number=NUMBER_OF_RUNS,
    )
    print(f"case_eigdh {t}")
    print(str_sep)

sigma = 1
t = timeit.timeit(
    functools.partial(case_eigsh_sigma, dmap_fixed_symmetric_nonstoch, sigma),
    number=NUMBER_OF_RUNS,
)
print(f"case_eigsh_sigma_{sigma} {t}")
print(str_sep)

sigma = None
t = timeit.timeit(
    functools.partial(case_eigsh_sigma, dmap_fixed_symmetric_nonstoch, sigma),
    number=NUMBER_OF_RUNS,
)
print(f"case_eigsh_sigma_{sigma} {t}")
print(str_sep)

t = timeit.timeit(
    functools.partial(case_eigs_sigma_sparsify, dmap_fixed_symmetric_nonstoch, sigma),
    number=NUMBER_OF_RUNS,
)
print(f"case_eigsh_sigma_sparsify_sigma_{sigma} {t}")
print(str_sep)


print("\n\nSYMMETRIC-STOCHASTIC VARIABLE KERNEL MATRIX")
if compute_exact:
    print_true_solution(dmap_variable_symmetric)

    t = timeit.timeit(
        functools.partial(case_eigdh, dmap_variable_symmetric), number=NUMBER_OF_RUNS,
    )
    print(f"case_eigdh {t}")
    print(str_sep)

sigma = 1 + add_numeric_break
t = timeit.timeit(
    functools.partial(case_eigsh_sigma, dmap_variable_symmetric, sigma),
    number=NUMBER_OF_RUNS,
)
print(f"case_eigsh_sigma_{sigma} {t} s"),
print(str_sep)

sigma = 0
t = timeit.timeit(
    functools.partial(case_eigsh_sigma, dmap_variable_symmetric, sigma),
    number=NUMBER_OF_RUNS,
)
print(f"case_eigsh_sigma_{sigma} {t} s"),
print(str_sep)

sigma = None
t = timeit.timeit(
    functools.partial(case_eigsh_sigma, dmap_variable_symmetric, sigma),
    number=NUMBER_OF_RUNS,
)
print(f"case_eigsh_sigma_{sigma} {t} s")
print(str_sep)
sigma = 1 + 1e-13
t = timeit.timeit(
    functools.partial(case_eigs_sigma_sparsify, dmap_variable_symmetric, sigma),
    number=NUMBER_OF_RUNS,
)
print(f"case_eigs_sigma_sparsify_sigma_{sigma} {t}")
print(str_sep)

t = timeit.timeit(
    functools.partial(case_eigsh_sigma_sparsify, dmap_variable_symmetric, sigma),
    number=NUMBER_OF_RUNS,
)
print(f"case_eigsh_sigma_sparsify_sigma_{sigma} {t}")
