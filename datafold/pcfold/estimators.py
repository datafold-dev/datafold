#!/usr/bin/env python3

import warnings
from functools import partial
from typing import Optional

import numpy as np
import scipy.spatial

from datafold.pcfold import GaussianKernel
from datafold.pcfold.distance import _k_smallest_element_value, compute_distance_matrix
from datafold.pcfold.kernels import _kth_nearest_neighbor_dist


def _warn_if_not_gaussian_kernel(kernel):
    if not isinstance(kernel, GaussianKernel):
        warnings.warn(
            "There is no guarantee that the method works with a kernel other than the "
            "GaussianKernel"
        )


def estimate_cutoff(
    pcm,
    n_subsample: int = 1000,
    k: int = 10,
    random_state: Optional[int] = None,
    distance_matrix=None,
) -> float:
    """Estimates a good choice of cut-off for a Gaussian radial basis kernel, given a
    certain tolerance below which the kernel values are considered zero.

    Parameters
    ----------
    pcm
        point cloud to compute pair-wise kernel matrix with

    n_subsample
        Maximum subsample used for the estimation. Ignored if :code:`distance_matrix is not
        None`.

    k
        Compute the `k`-th nearest neighbor distance to estimate the
        cut-off distance.

    random_state
        sets :code:`np.random.default_rng(random_state)`

    distance_matrix
        pre-computed distance matrix instead of using the internal `cdist` method

    See Also
    --------

    :py:class:`datafold.pcfold.kernels.GaussianKernel`

    """

    if k <= 1:
        raise ValueError("")

    n_points = pcm.shape[0]
    n_subsample = np.min([n_points, n_subsample])  # undersample the point set

    if n_points < 10:
        d = scipy.spatial.distance.pdist(pcm)
        # TODO: we could also return None or np.inf here (which is the equivalent to
        #  "dense case")
        return np.max(d)

    if distance_matrix is None:
        perm_indices_all = np.random.default_rng(random_state).permutation(n_points)

        distance_matrix = compute_distance_matrix(
            pcm[perm_indices_all[:n_subsample], :],
            pcm,
            metric="euclidean",
            backend="brute",
            kmin=k,
            # for estimation it is okay to be not exact and compute faster
            **dict(exact_numeric=False)
        )

        k = np.min([k, distance_matrix.shape[1]])
        # need to transpose the matrix here to correctly work with _kth_nearest_neighbor_dist
        k_smallest_values = _kth_nearest_neighbor_dist(distance_matrix.T, k)
    else:
        k_smallest_values = _kth_nearest_neighbor_dist(distance_matrix, k)

    est_cutoff = np.max(k_smallest_values)

    return float(est_cutoff)


def estimate_scale(
    pcm, tol=1e-8, cut_off: Optional[float] = None, **estimate_cutoff_params
) -> float:
    """Estimates the Gaussian kernel scale (epsilon) for a Gaussian kernel, given a
    certain tolerance below which the kernel values are considered zero.

    Parameters
    ----------
    pcm
        Point cloud to estimate the kernel scale with.

    tol
        Tolerance where the cut_off should be made.
        
    cut_off
        The `tol` parameter is ignored and the cut-off is used directly

    **estimate_cutoff_params
        Parameters to handle to method :py:meth:`estimate_cutoff` if ``cut_off is None``.
    """

    _warn_if_not_gaussian_kernel(pcm.kernel)

    if cut_off is None:
        cut_off = estimate_cutoff(pcm, **estimate_cutoff_params)

    # this formula is derived by solving for epsilon in
    # tol >= exp(-cut_off**2 / epsilon)
    eps0 = cut_off ** 2 / (-np.log(tol))
    return float(eps0)
