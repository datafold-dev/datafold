#!/usr/bin/env python

import copy
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse

from datafold.pcfold import GaussianKernel, PCManifoldKernel
from datafold.pcfold.distance import (
    DistanceAlgorithm,
    compute_distance_matrix,
    get_backend_distance_algorithm,
)
from datafold.pcfold.estimators import estimate_cutoff, estimate_scale


class PCManifold(np.ndarray):
    """Subclasses numpy.ndarray to represent point clouds on manifolds.

    The array is extended by a kernel and a backend to compute the distance matrix.
    Points are row-wise.

    ...

    Attributes
    ----------
    kernel
        Kernel defined on manifold.

    cut_off
        Cut-off distance. Larger distance values are set to zeros in the kernel.

    dist_backend : Union[str, DistanceAlgorithm]
        Algorithm to compute the distance matrix (must support the metric in the kernel).
    """

    def __new__(
        cls,
        data: Union[np.ndarray, pd.DataFrame],
        kernel: Optional[PCManifoldKernel] = None,
        cut_off: Optional[float] = None,
        dist_backend: Union[str, DistanceAlgorithm] = "guess_optimal",
        **dist_params,
    ):
        # See https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
        # to learn about subclassing numpy.ndarray

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
            raise ValueError("Point cloud must be in a 2 dim. array.")

        if not np.isfinite(obj).all():
            raise ValueError("Point cloud must be finite (no 'nan' or 'inf' values).")

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
        #    including own 'PCManifold'
        # 3) new-from-template --> obj is another instance of our own subclass,
        #    that we might use to update the new self instance.

        # For details
        # see https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html

        # Because __array_finalize__ is the only method that always sees new instances
        # being created, it is the sensible place to fill in instance defaults for new
        # object attributes, among other tasks.

        # "self" is a new object resulting from ndarray.__new__(InfoArray, ...),
        # therefore it only has attributes that the ndarray.__new__ constructor gave to
        # it - i.e. those of a standard ndarray.

        if obj is None:
            return obj

        # default parameters:
        self.kernel = getattr(obj, "kernel", GaussianKernel())

        self._cut_off = getattr(obj, "_cut_off", None)
        self._dist_backend = getattr(obj, "_dist_backend", "brute")
        self._dist_params = getattr(obj, "_dist_params", None)

    def __repr__(self):
        # att information about PCManifold kernels
        attributes_line = " | ".join(
            [
                f"kernel={self.kernel}",
                f"cut_off={str(self.cut_off)}",
                f"dist_backend={str(self.dist_backend.backend_name)}",
                f"dist_params={str(self._dist_params)}",
            ]
        )

        repr = "\n".join([attributes_line, super(PCManifold, self).__repr__()])
        return repr

    def __reduce__(self):
        # __reduce__ and __setstate__ are required for pickling (required if a model
        # such as DiffusionMaps that holds a PCManifold object is pickled)
        # The solution is from Mike McKerns' answer:
        # https://stackoverflow.com/a/26599346

        # Get the parent's __reduce__ tuple
        pickled_state = super(PCManifold, self).__reduce__()

        # Create own tuple to pass to __setstate__ (see below)
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
    def kernel(self) -> PCManifoldKernel:
        return self._kernel

    @kernel.setter
    def kernel(self, new_kernel: PCManifoldKernel):
        self._kernel = new_kernel

    @property
    def cut_off(self) -> Optional[float]:
        return self._cut_off

    @cut_off.setter
    def cut_off(self, new_cut_off):
        self._cut_off = new_cut_off

    @property
    def dist_backend(self) -> str:
        if isinstance(self._dist_backend, str):
            self.dist_backend = self._dist_backend

        return self._dist_backend

    @dist_backend.setter
    def dist_backend(self, backend):
        self._dist_backend = get_backend_distance_algorithm(backend)

    def compute_kernel_matrix(
        self, Y=None, **kernel_kwargs
    ) -> Union[np.ndarray, scipy.sparse.spmatrix]:
        return self.kernel(
            X=self,
            Y=Y,
            dist_cut_off=self.cut_off,
            dist_backend=self.dist_backend,
            kernel_kwargs=kernel_kwargs,
            dist_backend_kwargs=self._dist_params,
        )

    def compute_distance_matrix(
        self, Y: Optional[np.ndarray] = None, metric="euclidean"
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Compute distance matrix.

        Calls :py:meth:`datafold.pcfold.distance.compute_distance_matrix` with set
        `cut_off` and `backend`.

        Parameters
        ----------
        Y
            Reference dataset.

        metric
            distance metric (must be supported by backend)
            
        Returns
        -------
        Union[np.ndarray, scipy.sparse.csr_matrix]
            distance matrix
        """
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
        n_subsample: int = 1000,
        tol: float = 1e-8,
        kmin: int = 25,
        random_state: Optional[int] = None,
        result_scaling: float = 1.0,
        inplace: bool = True,
    ) -> Tuple[float, float]:
        """Estimates ``cut_off`` and kernel bandwidth ``epsilon`` of a Gaussian kernel.
        
        Parameters
        ----------

        n_subsample
            Number of samples to use for cut-off estimation.

        tol
            Tolerance below which the Gaussian kernel is assumed to be zero. Default: 1e-8

        kmin
            Number of nearest neighbors to use for :py:meth:`estimate_cutoff`.

        random_state
            Random state used in for random subsample.

        result_scaling
            The estimated `cut_off` will be scaled by this factor, and the scale (
            epsilon) will be computed accordingly.

        inplace
            If True, will set the `cut_off` and `kernel.epsilon` parameters of this
            instance.
            
        Returns
        -------
        Tuple[float, float]
            estimated cut off and epsilon
        """

        if not isinstance(self._kernel, GaussianKernel):
            raise TypeError("kernel must be a Gaussian kernel")

        if not hasattr(self._kernel, "epsilon"):
            # fails if kernel has no epsilon parameter
            raise AttributeError(
                f"Kernel {type(self._kernel)} has no epsilon parameter to optimize."
            )

        cut_off = estimate_cutoff(
            self, n_subsample=n_subsample, k=kmin, random_state=random_state
        )

        if result_scaling != 1:
            cut_off *= result_scaling

        epsilon = estimate_scale(self, tol=tol, cut_off=cut_off, kmin=kmin)

        if inplace:
            self.cut_off = cut_off
            self.kernel.epsilon = epsilon

        return cut_off, epsilon


def pcm_subsample(
    pcm: PCManifold,
    n_samples=100,
    min_distance: Optional[float] = None,
    min_added_per_iteration=1,
):
    """Subsamples a manifold point cloud with converged subsampling.

    Parameters
    ----------

    pcm
        point cloud to subsample

    n_samples
        block size for iteration

    min_distance
        cut off for distance matrix, should be lower than the kernel cut off

    min_added_per_iteration
         Loop terminates if less points points are added in current iteration. Setting
         it to zero searches the entire point cloud.

    Returns
    -------
    PCManifold
        subsampled dataset

    numpy.ndarray
        subsampled indices of the original dataset

    See Also
    --------
    :py:meth:`datafold.utils.math.random_subsample`
    """

    if min_distance is None and pcm.cut_off is not None:
        min_distance = pcm.cut_off / 2

    if min_distance is None:
        raise ValueError(
            "'cut_off' cannot be None. Either provide in 'min_diatnce' or 'pcm.cut_off'."
        )

    n_samples_pcm = pcm.shape[0]

    all_indices = np.random.permutation(n_samples_pcm)
    indices_splits = np.array_split(all_indices, n_samples_pcm // n_samples + 1)

    # choose first block of random samples as a basis
    subsample_indices = indices_splits[0]
    subsample_points = pcm[subsample_indices, :]

    # block-wise iteration of other blocks
    for iteration_indices in indices_splits[1:]:
        iteration_points = pcm[iteration_indices, :]

        # Currently, it uses brute backend with not exact numerics (justified by the
        # approximate sub-sampling) and because it assumes rather small junks
        distances = compute_distance_matrix(
            X=subsample_points,
            Y=iteration_points,
            cut_off=min_distance,
            metric="euclidean",
            backend="brute",
            **dict(exact_numeric=False),
        )

        # conditions to select
        # cond_1 = all samples that have no neighbor found
        # cond_2 = all samples where the smallest distance is larger than min_distance
        cond_1 = distances.getnnz(axis=1) == 0
        cond_2 = distances.min(axis=1).toarray().ravel() >= min_distance
        bool_mask_select_indices = np.logical_or(cond_1, cond_2, out=cond_1)

        # attach the samples where cond_1 or cond_2 is true to the subsample:
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


def pcm_remove_outlier(pcm: PCManifold, kmin: int, cut_off: float):
    """Remove all points that have not a minimum number of neighbors in cut-off range.

    Parameters
    ----------

    pcm
        point cloud

    kmin
        minimum number of minimum neighbors

    cut_off
        range in which to count the neighbors
    """

    if kmin <= 0 or not isinstance(kmin, int):
        raise ValueError("kmin must be a positive integer")

    distance_matrix = compute_distance_matrix(
        pcm, metric="sqeuclidean", cut_off=cut_off, backend="guess_optimal"
    )

    if scipy.sparse.issparse(distance_matrix):
        mask_non_outliers = distance_matrix.getnnz(axis=1) >= kmin
    else:
        mask_non_outliers = (distance_matrix > 0).sum(axis=1) >= kmin

    removed_indices = np.arange(pcm.shape[0])[np.logical_not(mask_non_outliers)]
    new_pcm = pcm[mask_non_outliers, :]

    return new_pcm, removed_indices
