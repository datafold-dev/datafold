#!/usr/bin/env python3

import numpy as np

NUM_FLOAT_PREC = 1e-15


def generate_mushroom(n_points=500):
    ## generate test data
    NX = int(np.sqrt(n_points))
    space = np.linspace(0, 1, NX)

    x, y = np.meshgrid(space, 2 * space)

    data = np.vstack([x.flatten(), y.flatten()]).T
    data = np.random.rand(NX * NX, 2)
    data[:, 1] = data[:, 1] * 1.0

    def transform(x, y):
        return x + y ** 3, y - x ** 3

    xt, yt = transform(data[:, 0], data[:, 1])
    data_mushroom = np.vstack([xt.flatten(), yt.flatten()]).T
    data_rectangle = data

    return data_mushroom, data_rectangle


def _assert_eq_matrices_tol(a, b, tol=NUM_FLOAT_PREC):
    mat_diff = np.abs(a - b)

    if (mat_diff < tol).all():
        return
    else:
        assert False, f"Tolerance is {tol}, but difference is {mat_diff.max()}"


def _assert_eq_matrices_exact(a, b):
    return _assert_eq_matrices_tol(a, b, tol=1e-16)
