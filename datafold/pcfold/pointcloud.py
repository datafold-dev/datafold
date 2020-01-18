#!/usr/bin/env python

import copy

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

from datafold.pcfold.distance import (
    compute_distance_matrix,
    get_backend_distance_algorithm,
)
from datafold.pcfold.estimators import estimate_cutoff, estimate_scale
from datafold.pcfold.kernels import Kernel, RadialBasisKernel

# TODO: Consider to have a separate Methods section in documentation for the methods
#  that are only for PCManifold
#   source: https://numpydoc.readthedocs.io/en/latest/format.html#class-docstring
#   > In some cases, however, a class may have a great many methods, of which only a
#     few are relevant (e.g.,
#   > subclasses of ndarray). Then, it becomes useful to have an additional "Methods"
#   section.


class PCManifold(np.ndarray):
    # See https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html

    def __new__(
        cls,
        data: np.ndarray,
        kernel: Kernel = None,
        cut_off=None,
        dist_backend="guess_optimal",
        **dist_params,
    ):

        if kernel is None:
            # TODO: also allow kernel=None? The distance matrix can still be computed.
            kernel = RadialBasisKernel()

        # view casting --> the np.ndarray as a PCManifold object --> this calls
        # internally __array_finalize__
        obj = np.asarray(data).view(cls)

        if obj.ndim != 2:
            raise ValueError("Point cloud has to be represented by a 2-dim array.")

        if np.isnan(obj).any() or np.isinf(obj).any():
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
        self.kernel = getattr(obj, "kernel", RadialBasisKernel())

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

        if not hasattr(self._kernel, "_epsilon"):
            # fails if kernel has no epsilon parameter
            raise AttributeError(
                f"Kernel {type(self._kernel)} has no epsilon parameter to optimize."
            )

        cut_off = estimate_cutoff(
            self, n_subsample=n_subsample, kmin=kmin, random_state=random_state
        )
        epsilon = estimate_scale(self, tol=tol, cut_off=self.cut_off, kmin=kmin)

        if result_scaling != 1:
            cut_off *= result_scaling
            epsilon *= result_scaling

        if inplace:
            self.cut_off = cut_off
            self.kernel._epsilon = epsilon

        return cut_off, epsilon


def subsample(
    pcm, min_distance=None, tol=1e-4, n_samples=100, random_state=None, randomized=False, min_added_per_iteration=1
):
    """
    Returns a new PCManifold that has a converged subsampling of the given points.
    randomized: False (default, will subsample iteratively) True (will randomly pick indices, but not necessarily uniformly distributed over the manifold. Very fast)
    min_added_per_iteration: default (1), number of points that need to be added per iteration to keep going. Setting it to zero will search the entire dataset.
    """

    if not isinstance(pcm, PCManifold):
        raise TypeError(
            "point cloud not valid"
        )  # TODO: for now enforce that we deal only with a PCM

    if min_distance is None and not(pcm.cut_off is None):
        min_distance = pcm.cut_off / 2

    if min_distance is None:
        raise ValueError(
            "cut_off cannot be None. Either provide in function or PCManifold."
        )

    if random_state is not None:
        np.random.seed(random_state)

    orig_n_samples = pcm.shape[0]

    if randomized:
        subsample_indices = np.random.permutation(orig_n_samples)[:n_samples]
        subsample_points = pcm[subsample_indices, :]
    else:
        all_indices = np.random.permutation(pcm.shape[0])
        indices_splits = np.array_split(all_indices, pcm.shape[0] // n_samples + 1)

        # choose first block of random samples as a basis
        subsample_indices = indices_splits[0]
        subsample_points = pcm[subsample_indices, :]

        # block-wise iteration of other shuffled blocks
        for iteration_indices in indices_splits[1:]:
            iteration_points = pcm[iteration_indices, :]

            distances = compute_distance_matrix(
                X=subsample_points,
                Y=iteration_points,
                cut_off=min_distance,
                metric="euclidean",
                backend="scipy.kdtree",
            )

            cond_1 = distances.getnnz(axis=1) == 0
            cond_2 = distances.min(axis=1).toarray().ravel() >= min_distance
            bool_mask_select_indices = np.logical_or(cond_1, cond_2)

            current_indices_selected = iteration_indices[bool_mask_select_indices]

            if bool_mask_select_indices.sum() <= min_added_per_iteration: # + int(n_samples * tol) + 
                # TODO: not sure why we need this condition -- break out of look and we
                #  don't look at other chunks.
                # FD: this is needed to prematurely end the iteration, stopping if too few points are added.
                #  Original code:
                #  if len(new_indices_k) < int(n_samples * tol) + 1:
                #     break
                break

            subsample_indices = np.append(subsample_indices, current_indices_selected)
            subsample_points = pcm[subsample_indices, :]

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

    save_eps = pcm.kernel._epsilon  # TODO: error if not available

    for i, scale in enumerate(scales):

        pcm.kernel._epsilon = scale
        kernel_matrix_scale = pcm.kernel.eval(distance_matrix=distance_matrix)
        kernel_sum = kernel_matrix_scale.sum()

        scale_sum[i] = kernel_sum / (kernel_matrix_scale.shape[0] ** 2)

    # ax.loglog(scales, scale_sum, 'k-', label='points')
    pcm.kernel._epsilon = save_eps

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
    pcm_subsample = PCManifold(subsmaple_array)

    print(subsample(pcm_subsample, min_distance=1.5))

    remove_outliers(pcm_subsample, kmin=2, cut_off=1)

    # pdist case
    print(pcm_subsample.compute_distance_matrix(metric="mahalanobis"))

    # cdist case
    print(
        pcm_subsample.compute_distance_matrix(
            pcm_subsample[0:3, :], metric="mahalanobis"
        )
    )
