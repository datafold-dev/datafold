import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_scalar

from datafold.dynfold.base import TransformType, TSCTransformerMixin
from datafold.pcfold import PCManifold, TSCDataFrame
from datafold.pcfold.eigsolver import compute_kernel_svd
from datafold.pcfold.kernels import (
    GaussianKernel,
    KernelType,
    PCManifoldKernel,
    RoselandKernel,
)
from datafold.utils.general import diagmat_dot_mat, mat_dot_diagmat, random_subsample


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

    n_svdtriplet
        The number of singular value decomposition pairs (left singular vector, singular
        value) to be computed from the normalized landmark-set affinity matrix.
        The right singular vectors are also computed and their number is governed by
        ``n_svdpairs``.

    time_exponent
        The time of the diffusion process (exponent of the singular values in the
        embedding). The value can be changed after the model is fitted.

    landmarks
        The landmark set used for computing the landmark-set affinity matrix.
        If `landmarks` is provided as

        * a float in (0,1], then it is proportion to randomly sample from `X` during `fit`
        * an integer > 1, then it is the number of random samples from `X` during `fit`
        * a `np.ndarray` containing the final landmark points

    alpha
        Normalization of the sampling density, indicated by a float in [0,1]. The
        parameter corresponds to `alpha` in :py:meth:`DiffusionMaps`.

        .. note::
            The parameter is not covered in the original Roseland paper and therefore
            should used with care.

    random_state
        Random seed for the selection of the landmark set. If provided when `Y` is also
        given, it is ignored. When `Y` is not provided it is used for the subsampling of
        the landmarks.

    dist_kwargs
        Keyword arguments passed to the point clouds of the two sets. See
        :py:meth:`datafold.pcfold.PCManifold` for parameter arguments.

    Attributes
    ----------

    landmarks_: PCManifold
        The final landmark data used. It is required for both in-sample and
        out-of-sample embeddings. ``np.asarray(landmarks_)`` casts the object to a
        standard numpy array.

    svdvalues_ : numpy.ndarray
        The singular values of the diffusion kernel matrix in decreasing order.

    svdvec_left_: numpy.ndarray
        The left singular vectors of the diffusion kernel matrix.

    svdvec_right_: numpy.ndarray
        The right singular vectors of the diffusion kernel matrix

    target_coords_: numpy.ndarray
        The coordinate indices to map to when transforming the data. The target point
        dimension equals the number of indices included in `target_coords_`. Note that the
        attributes `svdvalues_`, `svdvec_left_`, and `svdvec_right_` still contain *all*
        computed `n_svdtriplet`.

    kernel_matrix_ : Union[numpy.ndarray, scipy.sparse.csr_matrix]
        The computed kernel matrix; the matrix is only stored if
        ``store_kernel_matrix=True`` is set during :py:meth:`fit`.

    References
    ----------
    :cite:`shen2020scalability`
    """

    def __init__(
        self,
        kernel: Optional[PCManifoldKernel] = None,
        *,  # keyword-only
        n_svdtriplet: int = 10,
        time_exponent: float = 0,
        landmarks: Union[float, int, np.ndarray] = 0.25,
        alpha: float = 0,
        random_state: Optional[int] = None,
        dist_kwargs=None,
    ) -> None:

        self.kernel = kernel
        self.n_svdtriplet = n_svdtriplet
        self.time_exponent = time_exponent
        self.landmarks = landmarks
        self.alpha = alpha
        self.random_state = random_state
        self.dist_kwargs = dist_kwargs

        self.svdvalues_: np.ndarray
        self.svdvec_left_: np.ndarray
        self.svdvec_right_: np.ndarray

    def _validate_setting(self, n_samples):

        from datafold.pcfold.kernels import TSCManifoldKernel

        if isinstance(self.kernel, TSCManifoldKernel):
            raise NotImplementedError(
                "Kernels of type 'TSCManifoldKernel' are not " "supported yet"
            )

        check_scalar(
            self.n_svdtriplet, "n_svdtriplet", target_type=(int, np.integer), min_val=1
        )

        check_scalar(
            self.time_exponent,
            "time_exponent",
            target_type=(float, int, np.floating, np.integer),
        )

        if isinstance(self.landmarks, float):
            check_scalar(
                self.landmarks,
                "landmarks",
                target_type=float,
                min_val=0.0,
                max_val=1.0,
                include_boundaries="right",
            )
        elif isinstance(self.landmarks, int):
            check_scalar(
                self.landmarks,
                "landmarks",
                target_type=int,
                min_val=1,
                max_val=n_samples,
            )
        elif isinstance(self.landmarks, np.ndarray):
            pass  # will get checked later
        else:
            raise TypeError(
                f"Type of parameter landmarks (={type(self.landmarks)}) "
                "is not supported"
            )

    def _get_default_kernel(self):
        return GaussianKernel(epsilon=1.0)

    def _feature_names(self):
        if hasattr(self, "target_coords_"):
            feature_names = pd.Index(
                [f"rose{i}" for i in self.target_coords_],
                name=TSCDataFrame.tsc_feature_col_name,
            )

        else:
            feature_names = pd.Index(
                [f"rose{i}" for i in range(self.n_svdtriplet)],
                name=TSCDataFrame.tsc_feature_col_name,
            )

        return feature_names

    def _setup_default_dist_kwargs(self):
        from copy import deepcopy

        self.dist_kwargs_ = deepcopy(self.dist_kwargs) or {}
        self.dist_kwargs_.setdefault("cut_off", np.inf)

    def _subsample_landmarks(self, X: np.ndarray):
        """Subsamples landmarks from training data `X` when no `landmarks` are
        provided."""

        if isinstance(self.landmarks, float):
            n_landmarks = int(X.shape[0] * self.landmarks)
        else:  # isinstance(self.landmarks, int):
            n_landmarks = self.landmarks

        if n_landmarks <= 1:
            raise ValueError("The landmark set size must contain at least two samples.")

        if n_landmarks == X.shape[0]:
            landmarks = X
        else:
            landmarks, _ = random_subsample(
                X, n_landmarks, random_state=self.random_state
            )

        if isinstance(landmarks, pd.DataFrame):
            landmarks = landmarks.to_numpy()

        return landmarks

    def _compute_kernel_svd(
        self,
        kernel_matrix: KernelType,
        n_svdtriplet: int,
        normalize_diagonal: Optional[np.ndarray] = None,
        index_from: Optional[TSCDataFrame] = None,
    ) -> Tuple[np.ndarray, Union[np.ndarray, TSCDataFrame], np.ndarray]:

        svdvec_left, svdvals, svdvec_right = compute_kernel_svd(
            kernel_matrix=kernel_matrix, n_svdtriplet=n_svdtriplet
        )

        if normalize_diagonal is not None:
            # Change coordinates of the left singular vectors by using the diagonal matrix
            svdvec_left = diagmat_dot_mat(normalize_diagonal, svdvec_left)

        if index_from is not None:
            svdvec_left = TSCDataFrame.from_same_indices_as(
                index_from,
                svdvec_left,
                except_columns=[f"sv{i}" for i in range(n_svdtriplet)],
            )

        # transpose such that the right SVD vectors (typically denoted as V) are row-wise
        svdvec_right = svdvec_right.T
        return svdvec_left, svdvals, svdvec_right

    def _select_svdpairs_target_coords(self):
        """Returns either
        * all svd-triplets, or
        * the ones that were selected during set_target_coords

        It is assumed that the model is already fit.
        """

        if hasattr(self, "target_coords_"):
            svdvec_left = self.svdvec_left_[:, self.target_coords_]
            svdvec_right = self.svdvec_right_[:, self.target_coords_]
            svdvals = self.svdvalues_[self.target_coords_]
        else:
            svdvec_left, svdvals, svdvec_right = (
                self.svdvec_left_,
                self.svdvalues_,
                self.svdvec_right_,
            )
        return svdvec_left, svdvals, svdvec_right

    def _nystrom(self, kernel_cdist, svdvec_right, svdvals, normalize_diagonal):
        _kernel_cdist = kernel_cdist

        _magic_tol = 1e-14

        if (np.abs(svdvals) < _magic_tol).any():
            warnings.warn(
                "Roseland singular values are close to zero, which can cause "
                "numerical instabilities when applying the Nystroem extension."
            )

        svd_right_mapped = _kernel_cdist @ mat_dot_diagmat(
            svdvec_right, np.reciprocal(svdvals)
        )

        svd_right_mapped = diagmat_dot_mat(normalize_diagonal, svd_right_mapped)

        return svd_right_mapped

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
            # TODO: why is here an additional factor of 2? In Eq. 4 of the paper there is
            #  such factor
            svdvals_time = np.power(svdvalues, 2 * self.time_exponent)
            roseland_embedding = mat_dot_diagmat(svdvectors, svdvals_time)

        return roseland_embedding

    def set_target_coords(self, indices: Union[np.ndarray, List[int]]) -> "Roseland":
        """Set specific singular vector coordinates for a parsimonious mapping.

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

        indices = np.sort(np.asarray(indices))

        if indices.dtype != int:
            raise TypeError(f"The indices must be integers. Got type {indices.dtype}.")

        if indices[0] < 0 or indices[-1] >= self.n_svdtriplet:
            raise ValueError(
                f"Indices {indices} are out of bound. Only integer values "
                f"in [0, {self.n_svdtriplet}] are allowed."
            )

        self.target_coords_ = indices
        self.n_features_out_ = len(self.target_coords_)

        if hasattr(self, "feature_names_out_") and self.feature_names_out_ is not None:
            self.feature_names_out_ = self._feature_names()

        return self

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

        X = self._validate_datafold_data(
            X=X,
            array_kwargs=dict(ensure_min_samples=max(2, self.n_svdtriplet)),
            tsc_kwargs=dict(ensure_min_samples=max(2, self.n_svdtriplet)),
        )

        store_kernel_matrix = self._read_fit_params(
            attrs=[("store_kernel_matrix", False)],
            fit_params=fit_params,
        )

        self._validate_setting(n_samples=X.shape[0])
        self._setup_default_dist_kwargs()

        self._setup_feature_attrs_fit(X, features_out=self._feature_names())

        if isinstance(self.landmarks, (int, float)):
            self.landmarks_ = self._subsample_landmarks(X=X)
        else:
            self.landmarks_ = self.landmarks.view()

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
            kernel=RoselandKernel(internal_kernel=self.kernel, alpha=self.alpha),
            dist_kwargs=self.dist_kwargs_,
        )

        kernel_output = self.landmarks_.compute_kernel_matrix(X)

        (
            kernel_matrix,
            self._cdist_kwargs,
            ret_extra,
        ) = PCManifoldKernel.read_kernel_output(kernel_output=kernel_output)

        (
            self.svdvec_left_,
            self.svdvalues_,
            self.svdvec_right_,
        ) = self._compute_kernel_svd(
            kernel_matrix=kernel_matrix,
            n_svdtriplet=self.n_svdtriplet,
            normalize_diagonal=ret_extra["normalize_diagonal"],
            index_from=X if isinstance(X, TSCDataFrame) else None,
        )

        if store_kernel_matrix:
            self.kernel_matrix_ = kernel_matrix

        return self

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
        svdvec_left, svdvals, _ = self._select_svdpairs_target_coords()

        if isinstance(X, TSCDataFrame):
            svdvec_left = TSCDataFrame.from_same_indices_as(
                indices_from=X,
                values=svdvec_left,
                except_columns=self.feature_names_out_,
            )

        return svdvec_left

    def transform(self, X: TransformType) -> TransformType:
        r"""Embed out-of-sample points with the Nyström extension:

        .. math::

            K(Y, X) V \Lambda^{-1} = U_{new}

        where :math:`K(Y, X)` is a component-wise evaluation of the new kernel matrix,
        :math:`V` the right singular vectors associated with the fitted model,
        :math:`\Lambda` the singular values of the fitted model, and
        :math:`U_{new}` the approximated left singular vectors of the (normalized) new
        kernel matrix.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data points to be embedded.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            the new coordinates of the points in `X`
        """

        check_is_fitted(
            self, ("landmarks_", "svdvalues_", "svdvec_left_", "svdvec_right_")
        )

        X = self._validate_datafold_data(X)
        self._validate_feature_input(X, direction="transform")

        kernel_output = self.landmarks_.compute_kernel_matrix(X, **self._cdist_kwargs)
        kernel_matrix_cdist, _, ret_extra = PCManifoldKernel.read_kernel_output(
            kernel_output=kernel_output
        )

        _, svdvals, svdvec_right = self._select_svdpairs_target_coords()

        svdvec_right_embedded = self._nystrom(
            kernel_matrix_cdist,
            svdvec_right=svdvec_right,
            svdvals=svdvals,
            normalize_diagonal=ret_extra["normalize_diagonal"],
        )

        roseland_embedding = self._perform_roseland_embedding(
            svdvec_right_embedded, svdvals
        )

        if isinstance(X, TSCDataFrame) and kernel_matrix_cdist.shape[0] == X.shape[0]:
            roseland_embedding = TSCDataFrame.from_same_indices_as(
                indices_from=X,
                values=roseland_embedding,
                except_columns=self._feature_names(),
            )

        return roseland_embedding
