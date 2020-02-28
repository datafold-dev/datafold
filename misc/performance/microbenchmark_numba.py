"""This is a microbenchmark example which makes use of the Python package numba. It is not related to
Diffusion Maps. """

import time

import numpy as np
from numba import jit

x = np.arange(10000).reshape(100, 100)


def go_slow(a):  # Function is compiled to machine code when called the first time
    trace = 0
    for i in range(a.shape[0]):  # Numba likes loops
        trace += np.tanh(a[i, i])  # Numba likes NumPy functions
    return a + trace  # Numba likes NumPy broadcasting


# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
mean_time = 0
for i in range(100):

    @jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
    def go_fast(a):  # Function is compiled to machine code when called the first time
        trace = 0
        for i in range(a.shape[0]):  # Numba likes loops
            trace += np.tanh(a[i, i])  # Numba likes NumPy functions
        return a + trace  # Numba likes NumPy broadcasting

    start = time.time()
    go_fast(x)
    end = time.time()
    mean_time += end - start
mean_time /= 100
print("Elapsed (with compilation) = %s" % mean_time)

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
mean_time = 0
for i in range(100):
    start = time.time()
    go_fast(x)
    end = time.time()
    mean_time += end - start
mean_time /= 100
print("Elapsed (after compilation) = %s" % mean_time)

# COMPARE WITHOUT NUMBA
mean_time = 0
for i in range(100):
    start = time.time()
    go_slow(x)
    end = time.time()
    mean_time += end - start
mean_time /= 100
print("Elapsed (after compilation without numba) = %s" % mean_time)
