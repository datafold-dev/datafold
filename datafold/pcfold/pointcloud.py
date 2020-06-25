#!/usr/bin/env python

import copy
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse

from datafold.pcfold import GaussianKernel, PCManifoldKernel
from datafold.pcfold.distance import (
    compute_distance_matrix,
    get_backend_distance_algorithm,
)
from datafold.pcfold.estimators import estimate_cutoff, estimate_scale


class PCManifold(np.ndarray):
    """Represent a point cloud lying near a manifold with a kernel.

    ``PCManifold`` is derived from NumPy's ``ndarray``. It attaches a kernel that is
    associated with the data. Furthermore, distance parameter are attached to the data to
    select a suitable distance matrix algorithms, which supports the kernel metric
    and/or promotes sparsity by defining a cut-off distance.

    The data must be two-dimensional with the points in the rows of the matrix.

    ...

    Attributes
    ----------
    kernel
        Kernel to describe local proximity between data points.

    dist_kwargs
        Keyword arguments passed to the internal distance matrix computation. See
        :py:meth:`datafold.pcfold.compute_distance_matrix` for parameter arguments.

    See Also
    --------

    :class:`numpy.ndarray`
    :class:`numpy.array`
    """

    def __new__(
        cls,
        data: Union[np.ndarray, pd.DataFrame],
        kernel: Optional[PCManifoldKernel] = None,
        dist_kwargs: Optional[dict] = None,
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
        obj.kernel = kernel
        obj.dist_kwargs = dist_kwargs or {}

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

        self.dist_kwargs = getattr(obj, "dist_kwargs", {})
        self.dist_kwargs.setdefault("cut_off", np.inf)
        self.dist_kwargs.setdefault("kmin", 0)
        self.dist_kwargs.setdefault("backend", "guess_optimal")

    def __repr__(self):
        # att information about PCManifold kernels
        attributes_line = " | ".join(
            [f"kernel={self.kernel}", f"dist_kwargs={str(self.dist_kwargs)}",]
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
        new_state = pickled_state[2] + (self.kernel, self.dist_kwargs,)  # -2  # -1

        # Return a tuple that replaces the parent's __setstate__ tuple with own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state, *args, **kwargs):
        # Set own attributes attached to every np.ndarray
        self.kernel = state[-2]
        self.dist_kwargs = state[-1]

        # Call the parent's __setstate__ with the other tuple elements.
        super(PCManifold, self).__setstate__(state[0:-2])

    @property
    def cut_off(self) -> float:
        """Larger distance values are not set in a distance matrix computation. The
        corresponding kernel value is then treated as zero.

        The cut-off value is part of the dist_kwargs.

        Returns
        -------
        """
        return self.dist_kwargs.get("cut_off", np.inf)

    @cut_off.setter
    def cut_off(self, cut_off: float) -> None:
        """Set new cut-off value.

        Parameters
        ----------
        cut_off
            Non-negative value.

        """
        if cut_off <= 0:
            raise ValueError("cut_off (={}) must be a positive float")

        self.dist_kwargs["cut_off"] = float(cut_off)

    def compute_kernel_matrix(self, Y=None, **kernel_kwargs):
        """Compute the kernel matrix on the point cloud.

        Parameters
        ----------
        Y
            Query point cloud of shape `(n_samples_Y, n_features)`. If provided, compute
            the kernel matrix component-wise, else `Y=self` (pair-wise).

        **kernel_kwargs
            Keyword arguments passed passed to the kernel.
            
        Returns
        -------
        Union[np.ndarray, scipy.sparse.csr_matrix]
            kernel matrix of shape `(n_samples_Y, n_samples_self)`

        Optional
            A kernel can return further values see :meth:`PCManifoldKernel.__call__`
            for details.
        """
        return self.kernel(X=self, Y=Y, dist_kwargs=self.dist_kwargs, **kernel_kwargs)

    def compute_distance_matrix(
        self, Y: Optional[np.ndarray] = None, metric="euclidean"
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Compute distance matrix on points cloud.

        Internally calls :py:meth:`datafold.pcfold.distance.compute_distance_matrix`.

        Parameters
        ----------
        Y
            Query point cloud of shape (n_samples_Y, n_features). If provided, compute
            the distance matrix component-wise, else `Y=self` (pair-wise). For further
            details see also :class:`.DistanceAlgorithm`.

        metric
            Distance metric. The backend algorithm must supported the metric.
            
        Returns
        -------
        Union[np.ndarray, scipy.sparse.csr_matrix]
            distance matrix
        """
        return compute_distance_matrix(X=self, Y=Y, metric=metric, **self.dist_kwargs,)

    def optimize_parameters(
        self,
        n_subsample: int = 1000,
        tol: float = 1e-8,
        k: int = 25,
        random_state: Optional[int] = None,
        result_scaling: float = 1.0,
        inplace: bool = True,
    ) -> Tuple[float, float]:
        """Estimates ``cut_off`` and kernel bandwidth ``epsilon`` for a Gaussian kernel.
        
        Parameters
        ----------

        n_subsample
            Number of samples to use for cut-off estimation.

        tol
            Tolerance below which the Gaussian kernel is assumed to be zero.

        k
            Compute the `k` nearest neighbors distance for the cut-off.

        random_state
            Random state used in for random subsample.

        result_scaling
            The estimated `cut_off` will be scaled by this factor, and the scale (
            epsilon) will be computed accordingly.

        inplace
            If True, the `cut_off` and `kernel.epsilon` parameters are set for this
            instance.
            
        Returns
        -------
        float
            cut-off
        float
            epsilon
        """

        if not isinstance(self.kernel, GaussianKernel):
            raise TypeError("kernel must be of type GaussianKernel")

        if not hasattr(self.kernel, "epsilon"):
            # fails if kernel has no epsilon parameter
            raise AttributeError(
                f"Kernel {type(self.kernel)} has no epsilon parameter to optimize."
            )

        cut_off = estimate_cutoff(
            self, n_subsample=n_subsample, k=k, random_state=random_state
        )

        if result_scaling != 1:
            cut_off *= result_scaling

        epsilon = estimate_scale(self, tol=tol, cut_off=cut_off)

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
    """Subsample a manifold point cloud with a uniform sample density.

    Parameters
    ----------

    pcm
        Point cloud to subsample.

    n_samples
        Block size for iteration.

    min_distance
        Cut-off for distance matrix, should be larger than the `pcm` cut-off.

    min_added_per_iteration
         Loop terminates if less subsample points are added in a iteration.
         Setting it to zero iterates the entire point cloud.

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
        min_distance = pcm.cut_off * 2

    if min_distance is None:
        raise ValueError(
            "'cut_off' cannot be None. Either provide in 'min_distance' or 'pcm.cut_off'."
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
    """Remove all points that have not a minimum number of neighbors insinde the
    distance range.

    Parameters
    ----------

    pcm
        Point cloud.

    kmin
        The minimum number of a point to be not treated as an outlier.

    cut_off
        The distance range (Euclidean) in which to count the neighbours for
        each point.
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
