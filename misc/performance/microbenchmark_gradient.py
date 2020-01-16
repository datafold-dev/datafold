import numpy as np
from numba import jit, float64, int64, prange, njit
from math import pi
import timeit
import functools
import matplotlib.pyplot as plt
import numexpr

N = 10000
x = np.random.rand(N) * 2 * pi
y = np.tan(x) + np.random.normal(0, 0.05, N)
data_set = np.stack([x, y]).T

"""This whole benchmark needs a review, however, first solve the gradient bug 
gitlab issue #16"""


# Define random matrix, and use it as kernel matrix
from datafold.pcfold import PCManifold


ps = PCManifold(np.random.rand(1000, 1000))
values = np.sin(np.linspace(0, 2 * np.pi, 1000))

kernel_matrix = ps.compute_kernel_matrix()

# Auxiliary
number_of_points = data_set.shape[0]
size_of_sample = ps.shape[0]
points_dimension = ps.shape[1]


# Old
def old_grad():
    V = np.zeros((points_dimension, number_of_points, size_of_sample))
    for i in range(number_of_points):
        for j in range(size_of_sample):
            V[:, i, j] = data_set[i, :] - ps[j, :]

    grad = np.zeros([number_of_points, points_dimension])
    for i in range(number_of_points):
        ri = kernel_matrix[i, :]
        ki_psi = ri * np.squeeze(values)
        grad[i, :] = V[:, i, :] @ ki_psi
    return grad


# New improved
def new_vectorized():
    ki_psis = kernel_matrix * np.squeeze(values)
    grad = np.zeros_like(data_set)
    v = np.empty_like(ps)
    for p in range(number_of_points):
        np.subtract(data_set[p, :], ps, out=v)
        np.matmul(v.T, ki_psis[p, :], out=grad[p, :])
    return grad


def new_grad_numexpr():
    # especially good for large N -- currently numexpr is not a dependency and should not be used in the main code
    # speedup for N=10000 --> x1.75 compared to new_grad()
    # ki_psis = kernel_matrix * np.squeeze(values)

    numexpr.set_num_threads(8)

    values_squeeze = np.squeeze(values)
    ki_psis = numexpr.evaluate("kernel_matrix * values_squeeze")

    v = np.empty_like(ps)
    grad = np.zeros_like(data_set)
    for p in range(number_of_points):
        np.subtract(data_set[p, :], ps, out=v)
        np.matmul(v.T, ki_psis[p, :], out=grad[p, :])
    return grad


@jit(
    float64[:, :](float64[:, :], float64[:, :], float64[:, :], float64[:, :], int64),
    nopython=True,
    nogil=True,
    parallel=True,
    cache=True,
)
def numba_grad(
    kernel_matrix_arg, values_arg, data_set_arg, ps_arg, number_of_points_arg
):
    ki_psis = kernel_matrix_arg * values_arg[:, 0]
    grad = np.zeros_like(data_set_arg)
    for p in prange(
        number_of_points_arg
    ):  # use parallel range from numba to parallelize this loop
        grad[p, :] = (data_set_arg[p, :] - ps_arg).T @ ki_psis[p, :]
    return grad


def s1_numexpr(kernel_matrix_arg, values_arg):
    numexpr.set_num_threads(8)
    values_arg = np.squeeze(values_arg)
    ki_psis = numexpr.evaluate("kernel_matrix_arg * values_arg")
    return ki_psis


@jit(
    float64[:, :](float64[:, :], float64[:, :], float64[:, :], int64),
    nopython=True,
    nogil=True,
    parallel=True,
)
def s2_numba(ki_psis, data_set_arg, ps_arg, number_of_points_arg):
    grad = np.zeros_like(data_set_arg)
    for p in prange(number_of_points_arg):
        v = data_set[p, :] - ps_arg
        grad[p, :] = v.T @ ki_psis[p, :]
    return grad


def new_grad_mixed(
    kernel_matrix_arg, values_arg, data_set_arg, ps_arg, number_of_points_arg
):
    ki_psis = s1_numexpr(kernel_matrix_arg, values_arg)
    return s2_numba(ki_psis, data_set_arg, ps_arg, number_of_points)


def list_comprehension_grad():
    ki_psis = kernel_matrix * np.squeeze(values)
    grad = [
        (data_set[p : p + 1, :] - ps).T @ ki_psis[p, :] for p in range(number_of_points)
    ]
    return grad


# Another variant
def other_variant():
    # requires a lot more memory than the other methods and seems not to gain any speed up
    ps_transposed = ps.transpose()
    V = np.zeros((points_dimension, number_of_points, size_of_sample))
    for d in range(points_dimension):
        V[d, :, :] = data_set[:, d : d + 1] - ps_transposed[d : d + 1, :]
    # DOES NOT WORK V = (np.expand_dims(data_set, 1) - np.expand_dims(ps, 0)).reshape(points_dimension, number_of_points, size_of_sample)
    ki_psis = kernel_matrix * np.squeeze(values)
    grad = np.zeros_like(data_set)
    for i in range(number_of_points):
        grad[i, :] = V[:, i, :] @ ki_psis[i, :]
    return grad


is_test = False

if is_test:
    grad_old = old_grad()
    grad_vector = new_vectorized()
    grad_numexp = new_grad_numexpr()
    # grad_new2 = list_comprehension_grad()
    grad_numba = numba_grad(kernel_matrix, values, data_set, ps, number_of_points)
    grad_mixed = new_grad_mixed(kernel_matrix, values, data_set, ps, number_of_points)
    grad_other = other_variant()

    assert np.all(grad_old == grad_vector)
    assert np.all(grad_old == grad_other)
    assert np.all(grad_old == grad_numexp)
    assert np.allclose(grad_old, grad_numba, atol=1e-15)
    assert np.allclose(grad_old, grad_mixed, atol=1e-15)
    assert np.all(grad_old == grad_other)
    print("SUCCESS -- all variants are equal to the legacy code")
else:  # benchmark
    NUMBER_OF_RUNS = 5

    time_new_grad = timeit.timeit(new_vectorized, number=NUMBER_OF_RUNS)
    print(f"grad_vectorized {time_new_grad}")

    time_new_grad_numexpr = timeit.timeit(new_grad_numexpr, number=NUMBER_OF_RUNS)
    print(f"grad_numexpr {time_new_grad_numexpr}")

    time_numba_grad = timeit.timeit(
        functools.partial(
            numba_grad, kernel_matrix, values, data_set, ps, number_of_points
        ),
        number=NUMBER_OF_RUNS,
    )
    # time_numba_grad = timeit.timeit(numba_grad, number=NUMBER_OF_RUNS)
    print(f"grad_numba {time_numba_grad}")

    time_mixed = timeit.timeit(
        functools.partial(
            new_grad_mixed, kernel_matrix, values, data_set, ps, number_of_points
        ),
        number=NUMBER_OF_RUNS,
    )

    print(f"grad_mixed {time_mixed}")

    # time_other_variant_grad = timeit.timeit(other_variant, number=NUMBER_OF_RUNS)
    # print(f"grad_other {time_other_variant_grad}")

    include_original = False

    if include_original:
        time_old_grad = timeit.timeit(old_grad, number=NUMBER_OF_RUNS)
        print(f"grad_legacy {time_old_grad}")

        collect_speedup = np.array(
            [
                time_old_grad,
                time_new_grad,
                time_new_grad_numexpr,
                time_numba_grad,
                time_mixed,
            ]
        )
        names = [
            "grad_legacy",
            "grad_vectorized",
            "grad_numexpr",
            "grad_numba",
            "grad_mixed",
            "grad_other",
        ]
    else:
        collect_speedup = np.array(
            [time_new_grad, time_new_grad_numexpr, time_numba_grad, time_mixed]
        )
        names = ["grad_vectorized", "grad_numexpr", "grad_numba", "grad_mixed"]
    collect_speedup = collect_speedup.max() / collect_speedup

    plt.figure()
    plt.bar(x=np.arange(collect_speedup.shape[0]), height=collect_speedup)
    plt.xticks(np.arange(collect_speedup.shape[0]), names)

    plt.show()
