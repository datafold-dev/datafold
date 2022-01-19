import sys
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_scalar

from datafold.dynfold.base import TransformType, TSCTransformerMixin
from datafold.pcfold import PCManifold, TSCDataFrame
from datafold.pcfold.eigsolver import NumericalMathError
from datafold.pcfold.kernels import (
    GaussianKernel,
    KernelType,
    PCManifoldKernel,
    RoselandKernel,
)
from datafold.utils.general import (
    diagmat_dot_mat,
    mat_dot_diagmat,
    random_subsample,
    sort_eigenpairs,
)


# TODO: make SVD backend for kernel matrices
class _RoselandKernelAlgorithms:
    """Collection of re-useable algorithms that appear in models that have a Roseland
    kernel.

    See Also
    --------

    :class:`.Roseland`
    """

    @staticmethod
    def solve_svdproblem(
        kernel_matrix: KernelType,
        n_svdtriplets: int,
        normalize_diagonal: np.ndarray,
        index_from: Optional[TSCDataFrame] = None,
    ) -> Tuple[np.ndarray, Union[np.ndarray, TSCDataFrame], np.ndarray]:

        svdvects, svdvals, right_svdvects = scipy.sparse.linalg.svds(
            kernel_matrix,
            k=n_svdtriplets,
            which="LM",
            v0=np.ones(min(kernel_matrix.shape)),
        )

        svdvals, svdvects, right_svdvects = sort_eigenpairs(
            svdvals, svdvects, left_eigenvectors=right_svdvects
        )

        # Return the maximal absolute imaginary part of the singular vectors
        # Note that the eigenvalues are guaranteed to be real-valued in SVD
        max_imag_svdvect = np.abs(np.imag(svdvects)).max()
        max_imag_right_svdvect = np.abs(np.imag(right_svdvects)).max()

        if max(max_imag_svdvect, max_imag_right_svdvect) > 1e2 * sys.float_info.epsilon:
            raise NumericalMathError(
                "SVD eigenvectors have non-negligible imaginary part (larger than "
                f"{1e2 * sys.float_info.epsilon})."
            )
        else:
            svdvects = np.real(svdvects)
            right_svdvects = np.real(right_svdvects)

        # Change coordinates of the left singular vectors by using the diagonal matrix
        svdvects = diagmat_dot_mat(normalize_diagonal, svdvects)

        if index_from is not None:
            svdvects = TSCDataFrame.from_same_indices_as(
                index_from,
                svdvects,
                except_columns=[f"sv{i}" for i in range(n_svdtriplets)],
            )

        # transpose such that the right eigenvectors are now column-wise
        right_svdvects = right_svdvects.T
        return svdvals, svdvects, right_svdvects


class Roseland(BaseEstimator, TSCTransformerMixin):
    """Define a diffusion process on a point cloud by using a landmark set to find meaningful
    geometric descriptions.

    The model can be used for

    * non-linear dimensionality reduction
    * spectral clustering

    Parameters
    ----------
    kernel
        The kernel to describe similarity between points. A landmark-set affinity (kernel)
        matrix is computed using it. The normalized kernel matrix is used to define the
        diffusion process.
        Defaults to :py:class:`.GaussianKernel` with bandwidth `epsilon=1`.

    n_svdpairs
        The number of singular value decomposition pairs (left singular vector, singular
        value) to be computed from the normalized landmark-set affinity matrix.
        The right singular vectors are also computed and their number is governed by
        ``n_svdpairs``.

    time_exponent
        The time of the diffusion process (exponent of the singular values in the
        embedding). The value can be changed after the model is fitted.

    Y
        The landmark set used for computing the landmark-set affinity matrix. Either
        subsampled from X or provided at the instance creation.

    gamma
        The relative landmark set size, given as a number in (0; 1]. If provided when Y is
        also given, it is ignored. When Y is not provided it is used for the subsampling
        of the landmarks. The size of the landmark set during fit will then be

         .. math::

            |Y| = gamma|X|.

    random_state
        Random seed for the selection of the landmark set. If provided when `Y` is also
        given, it is ignored. When `Y` is not provided it is used for the subsampling of
        the landmarks.

    dist_kwargs
        Keyword arguments passed to the point clouds of the two sets. See
        :py:meth:`datafold.pcfold.PCManifold` for parameter arguments.

    Attributes
    ----------

    X_fit_: PCManifold
        The training data during fit.
        ``np.asarray(X_fit_)`` casts the object to a standard numpy array.

    Y_fit_: PCManifold
        The landmark data during fit. It is required for both in-sample and out-of sample
        embeddings. ``np.asarray(Y_fit_)`` casts the object to a standard numpy array.

    svdvalues_ : numpy.ndarray
        The singular values of the diffusion kernel matrix in decreasing order.

    svdvectors_: numpy.ndarray
        The left singular vectors of the diffusion kernel matrix.

    right_svdvectors_: numpy.ndarray
        The right singular vectors of the diffusion kernel matrix

    target_coords_: numpy.ndarray
        The coordinate indices to map to when transforming the data. The target point
        dimension equals the number of indices included in `target_coords_`. Note that the
        attributes `svdvalues_`, `svdvectors_`, and `right_svdvectors` still contain *all*
        computed `n_svdpairs`.

    kernel_matrix_ : Union[numpy.ndarray, scipy.sparse.csr_matrix]
        The computed kernel matrix; the matrix is only stored if
        ``store_kernel_matrix=True`` is set during :py:meth:`.fit`.

    References
    ----------
    :cite:`shen2020scalability`
    """

    def __init__(
        self,
        kernel: Optional[PCManifoldKernel] = None,
        *,  # keyword-only
        n_svdpairs: int = 10,
        time_exponent: float = 0,
        landmarks: Optional[np.ndarray] = None,
        gamma: float = 0.25,
        random_state: Optional[int] = None,
        dist_kwargs=None,
    ) -> None:

        self.kernel = kernel
        self.n_svdpairs = n_svdpairs
        self.time_exponent = time_exponent
        self.landmarks = landmarks
        self.gamma = gamma
        self.random_state = random_state
        self.dist_kwargs = dist_kwargs

        self.svdvalues_: np.ndarray
        self.svdvectors_: np.ndarray
        self.right_svdvectors_: np.ndarray

    def _validate_settings(self):

        check_scalar(
            self.n_svdpairs, "n_svdpairs", target_type=(int, np.integer), min_val=1
        )

        check_scalar(
            self.time_exponent,
            "time_exponent",
            target_type=(float, int, np.floating, np.integer),
        )

    # returns a default (Gaussian) kernel
    def _get_default_kernel(self):
        return GaussianKernel(epsilon=1.0)

    # returns the computed kernel matrix
    # def _get_kernel_output(
    #     self,
    #     X: np.ndarray,
    #     Y: np.ndarray,
    #     **fit_params,
    # ):
    #
    #     self._validate_settings()
    #
    #     X = self._validate_datafold_data(
    #         X=X,
    #         array_kwargs=dict(ensure_min_samples=max(2, self.n_svdpairs)),
    #         tsc_kwargs=dict(ensure_min_samples=max(2, self.n_svdpairs)),
    #     )
    #
    #     Y = self._validate_datafold_data(
    #         X=Y,
    #         array_kwargs=dict(ensure_min_samples=2),
    #         tsc_kwargs=dict(ensure_min_samples=2),
    #     )
    #
    #     self._setup_feature_attrs_fit(X, features_out=self._feature_names())
    #
    #     self._setup_default_dist_kwargs()
    #
    #     internal_kernel = (
    #         self.kernel if self.kernel is not None else self._get_default_kernel()
    #     )
    #
    #     if isinstance(X, TSCDataFrame):
    #         self.X_fit_ = TSCDataFrame(
    #             X, kernel=internal_kernel, dist_kwargs=self.dist_kwargs_
    #         )
    #     elif isinstance(X, (np.ndarray, pd.DataFrame)):
    #         self.X_fit_ = PCManifold(
    #             X,
    #             kernel=internal_kernel,
    #             dist_kwargs=self.dist_kwargs_,
    #         )
    #
    #     if isinstance(Y, (np.ndarray, pd.DataFrame)):
    #         self.Y_fit_ = PCManifold(
    #             Y,
    #             kernel=internal_kernel,
    #             dist_kwargs=self.dist_kwargs_,
    #         )
    #
    #     kernel_output = self.Y_fit_.compute_kernel_matrix(self.X_fit_)
    #     return kernel_output

    def _subsample_landmarks(self, X: np.ndarray):
        """Subsamples landmarks from training data `X` when no `landmarks` are
        provided."""
        check_scalar(
            self.gamma,
            "gamma",
            target_type=(float, np.floating),
            min_val=0.0,
            max_val=1.0,
            include_boundaries="right",
        )

        n_landmarks = int(X.shape[0] * self.gamma)
        if n_landmarks <= 1:
            raise ValueError("The landmark set size must contain at least two samples.")

        if n_landmarks == X.shape[0]:
            landmarks = X
        else:
            landmarks, _ = random_subsample(
                X, n_landmarks, random_state=self.random_state
            )

        if isinstance(landmarks, (np.ndarray, pd.DataFrame)):
            landmarks = PCManifold(landmarks)

        return landmarks

    def fit(
        self,
        X: TransformType,
        y=None,
        **fit_params,
    ) -> "Roseland":
        """Compute the Roseland kernel matrix and its singular vectors and values.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Training data.

        y: None

        **fit_params: Dict[str, object]
            - store_kernel_matrix: ``bool``
                If True, store the kernel matrix in attribute ``kernel_matrix``.

        Returns
        -------
        Roseland
            self
        """

        self._validate_settings()
        self._setup_default_dist_kwargs()

        store_kernel_matrix = self._read_fit_params(
            attrs=[("store_kernel_matrix", False)],
            fit_params=fit_params,
        )

        X = self._validate_datafold_data(
            X=X,
            array_kwargs=dict(ensure_min_samples=max(2, self.n_svdpairs)),
            tsc_kwargs=dict(ensure_min_samples=max(2, self.n_svdpairs)),
        )

        self._setup_feature_attrs_fit(X, features_out=self._feature_names())

        if self.landmarks is None:
            self.landmarks_ = self._subsample_landmarks(X=X)
        else:
            # TODO: I don't like this but this follows the guidelines of scikit-learn
            #  -- maybe landmarks should be given in **fit_params ??
            self.landmarks_ = self.landmarks

        self.landmarks_ = self._validate_datafold_data(
            X=self.landmarks_, array_kwargs=dict(ensure_min_samples=2), ensure_np=True
        )

        if self.kernel is None:
            self.landmarks_.optimize_parameters()
            self.kernel = self._get_default_kernel()
            self.kernel.epsilon = self.landmarks_.kernel.epsilon
            self.dist_kwargs["cut_off"] = self.landmarks_.cut_off

        self.landmarks_ = PCManifold(
            self.landmarks_,
            kernel=RoselandKernel(internal_kernel=self.kernel),
            dist_kwargs=self.dist_kwargs_,
        )

        kernel_output = self.landmarks_.compute_kernel_matrix(X)

        (
            kernel_matrix,
            _,
            ret_extra,
        ) = PCManifoldKernel.read_kernel_output(kernel_output=kernel_output)

        # try and match timestamps for timeseries data
        if (
            isinstance(X, TSCDataFrame)
            # TODO: I think this second condition is not true for landmarks?
            and kernel_matrix.shape[0] == X.shape[0]
        ):
            # if kernel is numpy.ndarray or scipy.sparse.csr_matrix, but X_fit_ is a time
            # series, then take indices from X_fit_ -- this only works if no samples are
            # dropped in the kernel computation.
            index_from: Optional[Union[TSCDataFrame, None]] = X
        else:
            index_from = None

        (
            self.svdvalues_,
            self.svdvectors_,
            self.right_svdvectors_,
        ) = _RoselandKernelAlgorithms.solve_svdproblem(
            kernel_matrix=kernel_matrix,
            n_svdtriplets=self.n_svdpairs,
            normalize_diagonal=ret_extra["normalize_diagonal"],
            index_from=index_from,
        )

        if store_kernel_matrix:
            self.kernel_matrix_ = kernel_matrix

        return self

    def _perform_roseland_embedding(
        self, svdvectors: Union[np.ndarray, pd.DataFrame], svdvalues: np.ndarray
    ) -> Union[np.ndarray, pd.DataFrame]:

        check_scalar(
            self.time_exponent,
            "time_exponent",
            target_type=(float, np.floating, int, np.integer),
        )

        if self.time_exponent == 0:
            roseland_embedding = svdvectors
        else:
            svdvals_time = np.power(svdvalues, 2 * self.time_exponent)
            roseland_embedding = mat_dot_diagmat(np.asarray(svdvectors), svdvals_time)

        return roseland_embedding

    def fit_transform(self, X: TransformType, y=None, **fit_params) -> np.ndarray:
        """Compute Roseland fit from data and apply embedding on the same data.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Training data.

        y: None

        **fit_params: Dict[str, object]
            See `fit` method for additional parameter.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            the new coordinates of the points in X

        """

        self.fit(X=X, **fit_params)

        svdvec, svdvals, _ = self._select_svdpairs_target_coords()

        roseland_embedding = self._perform_roseland_embedding(svdvec, svdvals)

        if isinstance(X, TSCDataFrame):
            roseland_embedding = TSCDataFrame.from_same_indices_as(
                indices_from=X,
                values=roseland_embedding,
                except_columns=self.feature_names_out_,
            )

        return roseland_embedding

    def _select_svdpairs_target_coords(self):
        """Returns either
        * all svd-triplets, or
        * the ones that were selected during set_target_coords

        It is assumed that the model is already fit.
        """

        if hasattr(self, "target_coords_"):
            if isinstance(self.svdvectors_, np.ndarray):
                svdvec = self.svdvectors_[:, self.target_coords_]
                right_svdvec = self.right_svdvectors_[:, self.target_coords_]

            svdvals = self.svdvalues_[self.target_coords_]
        else:
            svdvec, svdvals, right_svdvec = (
                self.svdvectors_,
                self.svdvalues_,
                self.right_svdvectors_,
            )
        return svdvec, svdvals, right_svdvec

    def set_target_coords(self, indices: Union[np.ndarray, List[int]]) -> "Roseland":
        """Set singular vector coordinates for parsimonious mapping.

        Parameters
        ----------
        indices
            Index values of svdpairs (``svdvalues_`` and ``svdvectors_``) to map
            new points to.

        Returns
        -------
        Roseland
            self
        """

        indices = np.asarray(indices)
        indices = np.sort(indices)

        if indices.dtype != int:
            raise TypeError(f"The indices must be integers. Got type {indices.dtype}.")

        if indices[0] < 0 or indices[-1] >= self.n_svdpairs:
            raise ValueError(
                f"Indices {indices} are out of bound. Only integer values "
                f"in [0, {self.n_svdpairs}] are allowed."
            )

        self.target_coords_ = indices

        self.n_features_out_ = len(self.target_coords_)

        if hasattr(self, "feature_names_out_") and self.feature_names_out_ is not None:
            self.feature_names_out_ = self._feature_names()

        return self

    def _feature_names(self):
        if hasattr(self, "target_coords_"):
            feature_names = pd.Index(
                [f"rose{i}" for i in self.target_coords_],
                name=TSCDataFrame.tsc_feature_col_name,
            )

        else:
            feature_names = pd.Index(
                [f"rose{i}" for i in range(self.n_svdpairs)],
                name=TSCDataFrame.tsc_feature_col_name,
            )

        return feature_names

    def _setup_default_dist_kwargs(self):
        from copy import deepcopy

        self.dist_kwargs_ = deepcopy(self.dist_kwargs) or {}
        self.dist_kwargs_.setdefault("cut_off", np.inf)

    def transform(self, X: TransformType) -> TransformType:

        r"""Embed out-of-sample points with the Nyström extension:

        .. math::
            K(Y, X) V \Lambda^{-1} = U_{new}

        where :math:`K(Y, X)` is a component-wise evaluation of the new kernel matrix,
        :math:`V` the right singular vectors associated with the fitted model,
        :math:`\Lambda` the singular values of the fitted model, and
        :math:`U_{new}` the approximated left singular vectors of the (normalized) new kernel matrix.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data points to be embedded.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            the new coordinates of the points in X
        """
        check_is_fitted(
            self, ("landmarks_", "svdvalues_", "svdvectors_", "right_svdvectors_")
        )

        X = self._validate_datafold_data(X)
        self._validate_feature_input(X, direction="transform")

        kernel_output = self.landmarks_.compute_kernel_matrix(X)
        kernel_matrix_cdist, _, ret_extra = PCManifoldKernel.read_kernel_output(
            kernel_output=kernel_output
        )

        _, svdvals, right_svdvec = self._select_svdpairs_target_coords()

        svdvec_nystroem = self._nystrom(
            kernel_matrix_cdist,
            right_svdvec=right_svdvec,
            svdvals=svdvals,
        )

        svdvec_nystroem = diagmat_dot_mat(
            ret_extra["normalize_diagonal"], svdvec_nystroem
        )

        roseland_embedding = self._perform_roseland_embedding(svdvec_nystroem, svdvals)

        if isinstance(X, TSCDataFrame) and kernel_matrix_cdist.shape[0] == X.shape[0]:
            roseland_embedding = TSCDataFrame.from_same_indices_as(
                indices_from=X,
                values=roseland_embedding,
                except_columns=self._feature_names(),
            )

        return roseland_embedding

    def _nystrom(self, kernel_cdist, right_svdvec, svdvals):
        _kernel_cdist = kernel_cdist

        _magic_tol = 1e-14

        if (np.abs(svdvals) < _magic_tol).any():
            warnings.warn(
                "Roseland singular values are close to zero, which can cause "
                "numerical instabilities when applying the Nystroem extension."
            )

        approx_eigenvectors = _kernel_cdist @ mat_dot_diagmat(
            right_svdvec, np.reciprocal(svdvals)
        )

        return approx_eigenvectors
