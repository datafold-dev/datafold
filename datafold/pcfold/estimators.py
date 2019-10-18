#!/usr/bin/env python3

import warnings
from functools import partial

import numpy as np
import scipy.spatial

from datafold.pcfold.distance import compute_distance_matrix, get_k_smallest_element_value
from datafold.pcfold.kernels import RadialBasisKernel


def _warn_if_not_rbf_kernel(kernel):
    if not isinstance(kernel, RadialBasisKernel):
        warnings.warn("There is no guarantee that the method works with a kernel other than RadialBasisKernel")


def estimate_cutoff(pcm, n_subsample=1000, kmin=10, random_state=None, distance_matrix=None):
    """
    Estimates the cutoff needed for a Gaussian Kernel exp(-r^2/epsilon),
    given a certain tolerance below which the kernel values are considered 'zero'.

    Parameters
    ==========

    pcm:          PCManifold
    tol:          1e-8       (tolerance where the cutoff should be made)
    n_subsample:  1000       (maximum of subsample used for the estimation, ignored if distance_matrix not None)
    kmin:         10         (the median of the kmin k-nearest neighbor distance is used)
    random_state: None       (if set, used to initialize the np.random.seed)
    distance_matrix: None    (if set to sparse csr_matrix, used instead of the scipy spatial cdist method)
    """

    n_points = pcm.shape[0]
    n_subsample = np.min([n_points, n_subsample])  # undersample the point set

    if n_points < 10:  # TODO: magic number
        d = scipy.spatial.distance.pdist(pcm)
        # TODO: we could also return None or np.inf here (which is the equivalent to "dense case")
        return np.max(d)

    if random_state is not None:
        np.random.seed(random_state)

    if distance_matrix is None:
        perm_indices_all = np.random.permutation(np.arange(n_points))
        distance_matrix = compute_distance_matrix(pcm[perm_indices_all[:n_subsample], :], pcm, metric="euclidean")

        kmin = np.min([kmin, distance_matrix.shape[1] - 1])
        k_smallest_values = get_k_smallest_element_value(distance_matrix, kmin, ignore_zeros=False)
    else:
        k_smallest_values = get_k_smallest_element_value(distance_matrix, kmin, ignore_zeros=False)

    est_cutoff = np.median(k_smallest_values)

    return est_cutoff


def estimate_scale(pcm, tol=1e-8, cut_off=None, **estimate_cut_off_params):
    """
    Estimates the kernel scale epsilon for a Gaussian Kernel exp(-r^2/epsilon),
    given a certain tolerance below which the kernel values are considered 'zero'.

    pcm:       PCManifold
    tol:       1e-8 (tolerance where the cut_off should be made)
    cut_off:   None (if given, the tol parameter is ignored and the cut_off is used directly)

    **estimate_cut_off_params: parameters to handle to method estimate_cutoff if cut_off is None
    """

    _warn_if_not_rbf_kernel(pcm.kernel)

    if cut_off is None:
        cut_off = estimate_cutoff(pcm, **estimate_cut_off_params)

    magic = 2  # doubling it since we want the kernel values to be BELOW the tolerance, not exactly on it
    eps0 = magic*np.sqrt(cut_off ** 2 / (-np.log(tol)))
    return eps0


def estimate_dimension(pcm):
    """
    Estimates the intrinsic dimension of the given manifold.
    """
    raise NotImplementedError()


if __name__ == "__main__":

    from pcmanifold.pcmanifold_new import PCManifold

    data = np.random.rand(10000, 5)
    pcm = PCManifold(data)

    dist = compute_distance_matrix(pcm)

    print(estimate_cutoff(pcm))
