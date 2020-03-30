#!/usr/bin/env python

import copy
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse

from datafold.pcfold.distance import (
    compute_distance_matrix,
    get_backend_distance_algorithm,
)
from datafold.pcfold.estimators import estimate_cutoff, estimate_scale
from datafold.pcfold.kernels import GaussianKernel, Kernel
from datafold.utils.maths import random_subsample

# TODO: Consider to have a separate methods section in documentation for the methods
#  that are only for PCManifold
#   source: https://numpydoc.readthedocs.io/en/latest/format.html#class-docstring
#   > In some cases, however, a class may have a great many methods, of which only a
#     few are relevant (e.g.,
#   > subclasses of ndarray). Then, it becomes useful to have an additional "Methods"
#   section.


class PCManifold(np.ndarray):
    """
    Point cloud on a manifold.

    ...

    Attributes
    ----------
    kernel : Kernel
        Kernel defined on data (manifold).
    cut_off : float
        Cut-off distance, with larger distance values be treated as zeros in the kernel.
    dist_backend : Union[str, DistanceAlgorithm]
        Algorithm to compute the distance matrix.

    Methods
    -------
    compute_kernel_matrix(self, Y=None, **kernel_kwargs)
        Compute the kernel matrix of point cloud hold by self or with respect to
        another point cloud (Y).

    compute_distance_matrix(self, Y=None, metric="euclidean")
        Compute the distance matrix of point cloud hold by self or with respect to
        another point cloud (Y).

    optimize_parameter()
    """

    # See https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html

    def __new__(
        cls,
        data: Union[np.ndarray, pd.DataFrame],
        kernel: Kernel = None,
        cut_off=None,
        dist_backend="guess_optimal",
        **dist_params,
    ):
        if kernel is None:
            # TODO: also allow kernel=None? The distance matrix can still be computed.
            kernel = GaussianKernel()

        # view casting --> the np.ndarray as a PCManifold object --> this calls
        # internally __array_finalize__
        obj = np.asarray(data).view(cls)

        if obj.dtype.kind not in "biufc":
            # See:
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.kind.html
            raise ValueError("Point cloud has to be numeric.")

        if obj.ndim != 2:
            raise ValueError("Point cloud has to be represented by a 2-dim array.")

        if not np.isfinite(obj).all():
            raise ValueError("Point cloud has illegal values (nan or inf).")

        # Set the kernel according to user input
        obj._kernel = kernel

        obj._cut_off = cut_off
        obj._dist_backend = get_backend_distance_algorithm(dist_backend)
        obj._dist_params = dist_params

        return obj

    def __array_finalize__(self, obj):
        # Gets called for all three ways of object creation
        # 1) explicit construction (PCManifold(...)) --> obj is None
        # 2) view casting -->  obj can be an instance of any subclass of ndarray,
        #    including our own
        # 3) new-from-template --> obj is another instance of our own subclass,
        #    that we might use to update the new self instance.

        # For details
        # see https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html

        # Because __array_finalize__ is the only method that always sees new instances
        # being created, it is the sensible place to fill in instance defaults for new
        # object attributes, among other tasks.

        # "self" is a new object resulting from ndarray.__new__(InfoArray, ...),
        # therefore it only has attributes that the ndarray.__new__ constructor gave it
        # - i.e. those of a standard ndarray.

        if obj is None:
            return obj

        # default parameters:
        self.kernel = getattr(obj, "kernel", GaussianKernel())

        self._cut_off = getattr(obj, "_cut_off", None)
        self._dist_backend = getattr(obj, "_dist_backend", "brute")
        self._dist_params = getattr(obj, "_dist_params", None)

    def __repr__(self):
        attributes_line = " | ".join(
            [
                f"kernel={self.kernel}",
                f"cut_off={str(self.cut_off)}",
                f"dist_backend={str(self.dist_backend.NAME)}",
                f"dist_params={str(self._dist_params)}",
            ]
        )

        repr = "\n".join([attributes_line, super(PCManifold, self).__repr__()])
        return repr

    def __reduce__(self):
        # __reduce__ and __setstate__ are required for pickling (which is e.g. required
        # if a model such as DiffusionMaps is stored)
        # The solution is from (answer from Mike McKerns):
        # https://stackoverflow.com/a/26599346

        # Get the parent's __reduce__ tuple
        pickled_state = super(PCManifold, self).__reduce__()

        # Create own tuple to pass to __setstate__

        new_state = pickled_state[2] + (
            self._kernel,  # -4
            self._cut_off,  # -3
            self._dist_backend,  # -2
            self._dist_params,  # -1
        )

        # Return a tuple that replaces the parent's __setstate__ tuple with own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state, *args, **kwargs):
        # Set own attributes attached to every np.ndarray
        self.kernel = state[-4]
        self._cut_off = state[-3]
        self._dist_backend = state[-2]
        self._dist_params = state[-1]

        # Call the parent's __setstate__ with the other tuple elements.
        super(PCManifold, self).__setstate__(state[0:-4])

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, new_kernel: Kernel):
        self._kernel = new_kernel

    @property
    def cut_off(self):
        return self._cut_off

    @cut_off.setter
    def cut_off(self, new_cut_off):
        self._cut_off = new_cut_off

    @property
    def dist_backend(self):
        if isinstance(self._dist_backend, str):
            self.dist_backend = self._dist_backend

        return self._dist_backend

    @dist_backend.setter
    def dist_backend(self, backend):
        self._dist_backend = get_backend_distance_algorithm(backend)

    def compute_kernel_matrix(self, Y=None, **kernel_kwargs):
        return self.kernel(
            X=self,
            Y=Y,
            dist_cut_off=self.cut_off,
            dist_backend=self.dist_backend,
            kernel_kwargs=kernel_kwargs,
            dist_backend_kwargs=self._dist_params,
        )

    def compute_distance_matrix(self, Y=None, metric="euclidean"):
        return compute_distance_matrix(
            X=self,
            Y=Y,
            metric=metric,
            cut_off=self.cut_off,
            backend=self.dist_backend,
            **self._dist_params,
        )

    def optimize_parameters(
        self,
        n_subsample=1000,
        tol=1e-8,
        kmin=25,
        random_state=None,
        result_scaling=1.0,
        inplace=True,
    ):
        """ optimize_parameters
        
        Estimates cut_off and kernel bandwidth epsilon, assuming a Gaussian kernel.
        
        Arguments
        ---------
        n_subsample: integer.
            Number of samples to use for estimations. Default: 1000
        tol: float.
            Tolerance below which the Gaussian kernel is assumed to be zero. Default: 1e-8
        kmin: integer.
            Number of nearest neighbors to use in the cut_off estimation. Default 25.
        random_state: integer [optional].
            The random state used in the selection of samples. Default: None.
        result_scaling: float.
            The estimated cut_off will be scaled by this number, and then epsilon will be computed accordingly. Default 1.0.
        inplace: boolean.
            If True, will set the cut_off and kernel.epsilon parameters of this instance.
            
        Returns
        -------
        cut_off: float.
        epsilon: float.
        """

        if not hasattr(self._kernel, "epsilon"):
            # fails if kernel has no epsilon parameter
            raise AttributeError(
                f"Kernel {type(self._kernel)} has no epsilon parameter to optimize."
            )

        cut_off = estimate_cutoff(
            self, n_subsample=n_subsample, kmin=kmin, random_state=random_state
        )

        if result_scaling != 1:
            cut_off *= result_scaling

        epsilon = estimate_scale(self, tol=tol, cut_off=cut_off, kmin=kmin)

        if inplace:
            self.cut_off = cut_off
            self.kernel.epsilon = epsilon

        return cut_off, epsilon


def pcm_subsample(
    pcm,
    min_distance=None,
    n_samples=100,
    min_added_per_iteration=1,
    randomized=False,
    random_state=None,
):
    """
    Returns a new PCManifold that has a converged subsampling of the given points.
    randomized: False (default, will subsample iteratively) True (will randomly pick
    indices, but not necessarily uniformly distributed over the manifold. Very fast)
    min_added_per_iteration: default (1), number of points that need to be added per
    iteration to keep going. Setting it to zero will search the entire dataset.
    """

    # if not randomized, then we need the kernel
    if not randomized and not isinstance(pcm, PCManifold):
        raise TypeError(f"type={type(pcm)} not valid")

    if min_distance is None and pcm.cut_off is not None:
        min_distance = pcm.cut_off / 2

    if min_distance is None:
        raise ValueError(
            "cut_off cannot be None. Either provide in function or PCManifold."
        )

    if random_state is not None:
        np.random.seed(random_state)

    n_samples_pcm = pcm.shape[0]

    if randomized:
        # for convenience, the function also allows "classic" randomized subsampling
        subsample_points, subsample_indices = random_subsample(pcm, n_samples=n_samples)
    else:
        all_indices = np.random.permutation(n_samples_pcm)
        indices_splits = np.array_split(all_indices, n_samples_pcm // n_samples + 1)

        # choose first block of random samples as a basis
        subsample_indices = indices_splits[0]
        subsample_points = pcm[subsample_indices, :]

        # block-wise iteration of other blocks
        for iteration_indices in indices_splits[1:]:
            iteration_points = pcm[iteration_indices, :]

            # TODO: "guess_optimal" may be justified here, or what is set in PCM?
            #  if iteration points is not too big, backend could also be brute force
            distances = compute_distance_matrix(
                X=subsample_points,
                Y=iteration_points,
                cut_off=min_distance,
                metric="euclidean",
                backend="scipy.kdtree",
            )

            cond_1 = distances.getnnz(axis=1) == 0
            cond_2 = distances.min(axis=1).toarray().ravel() >= min_distance
            bool_mask_select_indices = np.logical_or(cond_1, cond_2, out=cond_1)

            current_indices_selected = iteration_indices[bool_mask_select_indices]

            subsample_indices = np.append(subsample_indices, current_indices_selected)
            subsample_points = pcm[subsample_indices, :]

            if bool_mask_select_indices.sum() <= min_added_per_iteration:
                # prematurely end loop, if not enough points are added
                break

    return (
        PCManifold(subsample_points, kernel=copy.deepcopy(pcm.kernel)),
        subsample_indices,
    )


def remove_outliers(pcm, kmin, cut_off):
    """
    given the metric options in the constructor, remove all points
    that are not in range of cut_off of any neighbors.
    """

    assert kmin > 0

    distance_matrix = compute_distance_matrix(
        pcm, metric="sqeuclidean", cut_off=cut_off, backend="brute"
    )

    if scipy.sparse.issparse(distance_matrix):
        mask_non_outliers = distance_matrix.getnnz(axis=1) >= kmin
    else:
        mask_non_outliers = (distance_matrix > 0).sum(axis=1) >= kmin

    # TODO: make this as a parameter "return_removed_indices"?
    removed_indices = np.arange(pcm.shape[0])[np.logical_not(mask_non_outliers)]
    new_pcm = pcm[mask_non_outliers, :]

    return new_pcm, removed_indices


def plot_scales(pcm, scale_range=(1e-5, 1e3), scale_tests=20):
    """
    plots a scale plot for varying scales.
    n_subsample: subsample the distance matrix to plot faster
    """

    np.random.seed(1)

    scales = np.exp(
        np.linspace(np.log(scale_range[0]), np.log(scale_range[1]), scale_tests)
    )
    scale_sum = np.zeros_like(scales)

    distance_matrix = pcm.compute_kernel_matrix()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # n_samples = 100000
    # ind =
    #  np.random.permutation(np.arange(_d.shape[0]))[0:np.min([_d.shape[0], n_samples])]
    # _d = _d[ind,:]
    # _d = _d[:, ind]

    save_eps = pcm.kernel.epsilon  # TODO: error if not available

    for i, scale in enumerate(scales):

        pcm.kernel.epsilon = scale
        kernel_matrix_scale = pcm.kernel.eval(distance_matrix=distance_matrix)
        kernel_sum = kernel_matrix_scale.sum()

        scale_sum[i] = kernel_sum / (kernel_matrix_scale.shape[0] ** 2)

    # ax.loglog(scales, scale_sum, 'k-', label='points')
    pcm.kernel.epsilon = save_eps

    gradient = np.exp(
        np.gradient(np.log(scale_sum), np.log(scales)[1] - np.log(scales)[0])
    )
    ax.semilogx(scales, gradient, "k-", label="points")

    igmax = np.argmax(gradient)

    eps = scales[igmax]
    dimension = gradient[igmax] - 1 / 2

    ax.semilogx(
        [scales[igmax], scales[igmax]],
        [np.min(gradient), np.max(gradient)],
        "r-",
        label=r"max at $\epsilon=%.5f$" % (eps),
    )
    ax.semilogx(
        [np.min(scales), np.max(scales)],
        [gradient[igmax], gradient[igmax]],
        "b-",
        label=r"dimension $\approx %.1f$" % (dimension),
    )

    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$\mathbb{E}_\epsilon$")
    # ax.loglog(scales, 1*scales, 'r--', label='dim=1')
    # ax.loglog(scales, 2*scales, 'g--', label='dim=2')
    ax.legend()
    fig.tight_layout()


if __name__ == "__main__":
    # Small test cases

    import logging

    logging.basicConfig(level=logging.INFO)

    array = np.random.rand(5, 5)
    pcm = PCManifold(array, cut_off=10)

    print(pcm.compute_distance_matrix())

    exit()

    print("")
    cdist_cmp_array = np.random.rand(3, 5)
    print(pcm.compute_kernel_matrix(cdist_cmp_array))

    subsmaple_array = np.random.rand(5000, 5)
    pcm_subs = PCManifold(subsmaple_array)

    print(pcm_subsample(pcm_subs, min_distance=1.5))

    remove_outliers(pcm_subs, kmin=2, cut_off=1)

    # pdist case
    print(pcm_subs.compute_distance_matrix(metric="mahalanobis"))

    # cdist case
    print(pcm_subs.compute_distance_matrix(pcm_subs[0:3, :], metric="mahalanobis"))
