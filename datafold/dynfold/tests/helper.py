#!/usr/bin/env python3

"""Helper functions for testing. """

import logging
import os
import sys
from typing import Optional

import diffusion_maps as legacy_dmap
import numpy as np
import numpy.testing as nptest
from scipy.sparse import csr_matrix

from datafold.dynfold.dmap import DiffusionMaps


def make_strip(
    xmin: float, ymin: float, width: float, height: float, num_samples: int
) -> np.ndarray:
    """Draw samples from a 2D strip with uniform distribution.

    """
    x = width * np.random.rand(num_samples) - xmin
    y = height * np.random.rand(num_samples) - ymin

    return np.stack((x, y), axis=-1)


def make_points(
    num_points: int, x0: float, y0: float, x1: float, y1: float
) -> np.ndarray:
    xx, yy = np.meshgrid(
        np.linspace(x0, x1, num_points), np.linspace(y0, y1, num_points)
    )
    return np.stack((xx.ravel(), yy.ravel())).T


def swiss_roll(nt: int, ns: int, freq: Optional[float] = 2.0) -> np.ndarray:
    """Draw samples from the swiss roll manifold.

    """
    tt = np.linspace(0.0, 2.0 * np.pi, nt)
    ss = np.linspace(-0.5, 0.5, ns)
    t, s = np.meshgrid(tt, ss)

    x = t * np.cos(freq * t)
    y = t * np.sin(freq * t)
    z = s

    return np.stack((x.ravel(), y.ravel(), z.ravel())).T


def print_problem(a1, a2, cmp_str=""):

    if type(a1) != type(a2):
        print(f"Type mismatch, got {type(a1)} and {type(a2)}")
        return

    if isinstance(a1, csr_matrix):
        assert isinstance(a2, csr_matrix)
        a1 = a1.toarray()
        a2 = a2.toarray()

    if a1.shape != a2.shape:
        print(f"There is a shape mismatch in {cmp_str}: {a1.shape} vs. {a2.shape}")
    elif np.isnan(a1).sum() > 0 or np.isnan(a2).sum() > 0:
        print(f"There are nan values present in {cmp_str}.")
    else:
        print(f"Largest abs. difference in {cmp_str} = {np.max(np.abs(a1 - a2))}")


def cmp_eigenpairs(dmap1: DiffusionMaps, dmap2: legacy_dmap.BaseDiffusionMaps):
    def diagnosis():
        if isinstance(dmap1.kernel_matrix_, np.ndarray):
            assert isinstance(
                dmap2, legacy_dmap.DenseDiffusionMaps
            ), f"got instance {type(dmap2)}"
            logging.debug("DenseDiffusionMaps")
            kernel_matrix1 = dmap1.kernel_matrix_
            kernel_matrix2 = dmap2.kernel_matrix
        elif isinstance(dmap1.kernel_matrix_, csr_matrix):
            assert isinstance(
                dmap2, legacy_dmap.SparseDiffusionMaps
            ), f"got instance {type(dmap2)}"
            logging.debug("SparseDiffusionMaps")
            kernel_matrix1 = dmap1.kernel_matrix_.todense()
            kernel_matrix2 = dmap2.kernel_matrix.todense()
        else:
            raise RuntimeError("Shouldnt get here")

        if np.array_equal(kernel_matrix1, kernel_matrix2):
            logging.debug("Kernel matrices are equal.")
        else:
            logging.error(
                f"Kernel matrices are NOT equal. \n "
                f"{print_problem(kernel_matrix1, kernel_matrix2), 'kernel matrix'}"
            )

        logging.debug(
            f"Condition number of kernel matrix is {np.linalg.cond(kernel_matrix1)}"
        )

    try:
        nptest.assert_allclose(
            dmap1.eigenvalues_,
            dmap2.eigenvalues,
            rtol=1e-13,
            atol=1e-15,
            equal_nan=False,
        )
    except Exception as e:
        if not (dmap1.eigenvalues_ - 1 < 1e-14).all():
            raise e
        else:
            logging.debug(
                "All eigenvalues are very close to one, did not compare eigenvectors. "
            )


def assert_equal_eigenvectors(eigvec1, eigvec2, tol=1e-14):
    # Allows to also check orthogonality, but is not yet implemented
    norms1 = np.linalg.norm(eigvec1, axis=0)
    norms2 = np.linalg.norm(eigvec2, axis=0)
    eigvec_test = (eigvec1.T @ eigvec2) * np.reciprocal(np.outer(norms1, norms2))

    actual = np.abs(np.diag(eigvec_test))  # -1 is also allowed for same direction
    expected = np.ones(actual.shape[0])

    nptest.assert_allclose(expected, actual, atol=tol, rtol=0)


def cmp_kernel_matrix(
    actual: DiffusionMaps, expected: legacy_dmap.BaseDiffusionMaps, rtol=None, atol=None
):

    if rtol is None and atol is None:
        exact = True
    else:
        assert rtol is not None and atol is not None
        exact = False

    if isinstance(actual.kernel_matrix_, np.ndarray):
        assert isinstance(expected, legacy_dmap.DenseDiffusionMaps)

        if exact:
            nptest.assert_equal(actual.kernel_matrix_, expected.kernel_matrix)
        else:
            nptest.assert_allclose(
                actual.kernel_matrix_, expected.kernel_matrix, rtol=rtol, atol=atol
            )

    elif isinstance(actual.kernel_matrix_, csr_matrix):
        assert isinstance(expected, legacy_dmap.SparseDiffusionMaps)
        actual_csr = actual.kernel_matrix_
        expected_csr = expected.kernel_matrix

        actual_csr.sort_indices()
        expected_csr.sort_indices()

        if exact:
            nptest.assert_equal(actual_csr.indices, expected_csr.indices)
            nptest.assert_equal(actual_csr.data, expected_csr.data)
        else:
            nptest.assert_equal(actual_csr.indices.sort(), expected_csr.indices.sort())
            nptest.assert_allclose(
                actual_csr.data, expected_csr.data, rtol=rtol, atol=atol
            )


def cmp_dmap_legacy(
    actual_dmap: DiffusionMaps,
    expected_dmap: legacy_dmap.BaseDiffusionMaps,
    rtol=None,
    atol=None,
):
    cmp_eigenpairs(actual_dmap, expected_dmap)
    cmp_kernel_matrix(actual_dmap, expected_dmap, atol, rtol)


def cmp_dmap(dmap1: DiffusionMaps, dmap2: DiffusionMaps):

    nptest.assert_equal(dmap1.kernel.epsilon, dmap2.kernel.epsilon)

    nptest.assert_equal(dmap1.eigenvalues_, dmap2.eigenvalues_)
    nptest.assert_equal(dmap1.eigenvectors_, dmap2.eigenvectors_)

    from scipy.sparse import csr_matrix

    if isinstance(dmap1.kernel_matrix_, np.ndarray):
        assert isinstance(dmap2.kernel_matrix_, np.ndarray)
        nptest.assert_equal(dmap1.kernel_matrix_, dmap2.kernel_matrix_)

    elif isinstance(dmap1.kernel_matrix_, csr_matrix):
        assert isinstance(dmap2.kernel_matrix_, csr_matrix)
        nptest.assert_equal(
            dmap1.kernel_matrix_.toarray(), dmap1.kernel_matrix_.toarray()
        )


def circle_data(nsamples=100):
    data = np.vectorize(lambda w: np.exp(-1j * w))(
        np.linspace(0, 2 * np.pi, nsamples)[:-1][:, np.newaxis]
    )
    return (
        np.hstack([np.real(data), np.imag(data)]),
        1e-3,
    )  # epsilon used in jupyter notebook in method_examples
