#!/usr/bin/env python3

import warnings
from functools import partial
from typing import Optional

import numpy as np
import scipy.spatial

from datafold.pcfold import GaussianKernel
from datafold.pcfold.distance import _k_smallest_element_value, compute_distance_matrix


def _warn_if_not_gaussian_kernel(kernel):
    if not isinstance(kernel, GaussianKernel):
        warnings.warn(
            "There is no guarantee that the method works with a kernel other than the "
            "GaussianKernel"
        )


def estimate_cutoff(
    pcm,
    n_subsample: int = 1000,
    kmin: int = 10,
    random_state: Optional[int] = None,
    distance_matrix=None,
) -> float:
    """Estimates the cut off needed for a Gaussian radial basis kernel, given a certain
    tolerance below which the kernel values are considered zero.

    Parameters
    ----------
    pcm
        point cloud to compute pair wise kernel matrix with

    n_subsample
        Maximum subsample used for the estimation. Ignored if distance_matrix not None.

    kmin
        median of the kmin k-nearest neighbor distance is used

    random_state
        sets :code:`np.random.seed(random_state)`

    distance_matrix
        if set to sparse csr_matrix, used instead of the scipy spatial cdist method

    See Also
    --------

    :py:class:`datafold.pcfold.kernels.GaussianKernel`

    """

    n_points = pcm.shape[0]
    n_subsample = np.min([n_points, n_subsample])  # undersample the point set

    if n_points < 10:
        d = scipy.spatial.distance.pdist(pcm)
        # TODO: we could also return None or np.inf here (which is the equivalent to
        #  "dense case")
        return np.max(d)

    if random_state is not None:
        np.random.seed(random_state)

    if distance_matrix is None:
        perm_indices_all = np.random.permutation(np.arange(n_points))
        distance_matrix = compute_distance_matrix(
            pcm[perm_indices_all[:n_subsample], :],
            pcm,
            metric="euclidean",
            backend="brute",
            # for estimation it is okay to be not exact and compute faster
            **dict(exact_numeric=False)
        )

        kmin = np.min([kmin, distance_matrix.shape[1] - 1])
        k_smallest_values = _k_smallest_element_value(
            distance_matrix, kmin, ignore_zeros=False
        )
    else:
        k_smallest_values = _k_smallest_element_value(
            distance_matrix, kmin, ignore_zeros=False
        )

    est_cutoff = np.median(k_smallest_values)

    return float(est_cutoff)


def estimate_scale(
    pcm, tol=1e-8, cut_off: Optional[float] = None, **estimate_cutoff_params
) -> float:
    """Estimates the Gaussian kernel scale (epsilon) for a Gaussian kernel, given a
    certain tolerance below which the kernel values are considered zero.

    Parameters
    ----------
    pcm
        point cloud to estimate the kernel scale with

    tol
        Tolerance where the cut_off should be made.
        
    cut_off
        The `tol` parameter is ignored and the cut off is used directly

    **estimate_cutoff_params
        parameters to handle to method :py:meth:`estimate_cutoff` if cut_off is None
    """

    _warn_if_not_gaussian_kernel(pcm.kernel)

    if cut_off is None:
        cut_off = estimate_cutoff(pcm, **estimate_cutoff_params)

    # this formula is derived by solving for epsilon in
    # tol >= exp(-cut_off**2 / epsilon)
    eps0 = cut_off ** 2 / (-np.log(tol))
    return float(eps0)


def __estimate_dimension(pcm):
    """
    Estimates the intrinsic dimension of the given manifold.
    """
    raise NotImplementedError()
