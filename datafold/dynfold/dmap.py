import sys
from typing import List, Tuple, Union

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_scalar

from datafold.dynfold.base import TransformType, TSCTransformerMixIn
from datafold.pcfold import DmapKernelFixed, PCManifold
from datafold.pcfold.eigsolver import NumericalMathError, compute_kernel_eigenpairs
from datafold.pcfold.kernels import DmapKernelVariable, GaussianKernel, PCManifoldKernel
from datafold.utils.general import (
    diagmat_dot_mat,
    if1dim_colvec,
    mat_dot_diagmat,
    random_subsample,
)


class _DmapKernelAlgorithms:
    """Collection of re-useable algorithms that appear in models that have a diffusion
    map kernel.

    See Also
    --------

    :class:`.DiffusionMaps`
    :class:`.DiffusionMapsVariable`
    :class:`.GeometricHarmonicsInterpolator`
    """

    @staticmethod
    def solve_eigenproblem(
        kernel_matrix, n_eigenpairs, is_symmetric, is_stochastic, basis_change_matrix
    ) -> Tuple[np.ndarray, np.ndarray]:

        try:
            eigvals, eigvect = compute_kernel_eigenpairs(
                kernel_matrix=kernel_matrix,
                n_eigenpairs=n_eigenpairs,
                is_symmetric=is_symmetric,
                is_stochastic=is_stochastic,
                # only normalize after potential basis change
                normalize_eigenvectors=False,
                backend="scipy",
            )

        except NumericalMathError:
            # re-raise with more details for DMAP kernel
            raise_numerical_error = True
        else:
            max_imag_eigvect = np.abs(np.imag(eigvect)).max()
            max_imag_eigval = np.abs(np.imag(eigvals)).max()

            if max(max_imag_eigval, max_imag_eigvect) > 1e2 * sys.float_info.epsilon:
                raise_numerical_error = True
            else:
                raise_numerical_error = False

        if raise_numerical_error:
            raise NumericalMathError(
                "Eigenpairs have non-negligible imaginary part (larger than "
                f"{1e2 * sys.float_info.epsilon}. First try to use "
                f"parameter 'symmetrize_kernel=True' (improves numerical stability) and "
                f"only if this is not working adjust kernel scale."
            )
        else:
            eigvals = np.real(eigvals)
            eigvect = np.real(eigvect)

        if basis_change_matrix is not None:
            eigvect = basis_change_matrix @ eigvect

        eigvect /= np.linalg.norm(eigvect, axis=0)[np.newaxis, :]

        return eigvals, eigvect

    @staticmethod
    def unsymmetric_kernel_matrix(
        kernel_matrix: Union[np.ndarray, scipy.sparse.csr_matrix], basis_change_matrix,
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Transform a kernel matrix obtained from a symmetric conjugate
        transformation to the diffusion kernel matrix.

        Parameters
        ----------
        kernel_matrix
            Symmetric conjugate kernel matrix of shape `(n_samples, n_samples)`

        basis_change_matrix
            diagonal elements of basis change matrix

        Returns
        -------
        numpy.ndarray
            Generally non-symmetric matrix of same shape and type as `kernel_matrix`.
        """
        inv_basis_change_matrix = scipy.sparse.diags(
            np.reciprocal(basis_change_matrix.data.ravel())
        )

        return inv_basis_change_matrix @ kernel_matrix @ inv_basis_change_matrix


class DiffusionMaps(BaseEstimator, TSCTransformerMixIn):
    """Define diffusion processes on point clouds to find meaningful geometric
    descriptions.

    The model can be used for

    * non-linear dimensionality reduction
    * approximating eigenfunctions of operators (see ``alpha`` parameter):

        - Laplace-Beltrami
        - Fokker-Plank
        - Graph Laplacian

    Parameters
    ----------
    kernel
        Internal kernel to describe proximity between points. The kernel is passed
        as an `internal_kernel` to :class:`.DmapKernelFixed`, which describes
        the diffusion process.

    n_eigenpairs
        Number of eigenpairs to compute from kernel matrix.

    time_exponent
        Time of diffusion process (exponent of eigenvalues in embedding).

    is_stochastic
        If True, the diffusion kernel matrix is normalized (row stochastic).

    alpha
        Re-normalization parameter between `(0,1)`. ``alpha=1`` corrects the sampling
        density in the data as an artifact of the collection process. Special values are

        * `alpha=0` graph Laplacian,
        * `alpha=0.5` Fokker-Plank operator,
        * `alpha=1` Laplace-Beltrami operator

        Note, that `is_stochastic=True` is required in all three cases.

    symmetrize_kernel
        If True, a conjugate transformation is performed if the settings
        lead to a non-symmetric kernel matrix. This improves numerical stability when
        solving the eigenvectors of the kernel matrix as it allows algorithms designed
        for (sparse) Hermitian matrices to be used. If the kernel matrix is symmetric
        already (`is_stochastic=False`), then the parameter has no effect.

    dist_kwargs
        Keyword arguments passed to the internal distance matrix computation. See
        :py:meth:`datafold.pcfold.compute_distance_matrix` for parameter arguments.

    Attributes
    ----------
    X_: PCManifold
        Training data during fit. The data is required for out-of-sample mappings.
        Equipped with kernel :py:class:`DmapKernelFixed`.

    eigenvalues_ : numpy.ndarray
        Eigenvalues of diffusion kernel matrix in decreasing order.

    eigenvectors_: TSCDataFrame, pandas.DataFrame, numpy.ndarray
        Eigenvectors of the kernel matrix to parametrize the data manifold.

    inv_coeff_matrix_: numpy.ndarray
        Coefficient matrix to map points from embedding space back to original space.\
        Computation is delayed until `inverse_transform` is called for the first time.

    kernel_matrix_ : Union[numpy.ndarray, scipy.sparse.csr_matrix]
        Computed kernel matrix, stored only if `store_kernel_matrix=True` is set during
        fit.

    References
    ----------
    :cite:`lafon_diffusion_2004`
    :cite:`coifman_diffusion_2006`
    """

    _cls_valid_operator_names = (
        "laplace_beltrami",
        "fokker_planck",
        "graph_laplacian",
        "rbf",
    )

    def __init__(
        self,
        kernel: PCManifoldKernel = GaussianKernel(epsilon=1.0),
        n_eigenpairs: int = 10,
        time_exponent: float = 0,
        is_stochastic: bool = True,
        alpha: float = 1,
        symmetrize_kernel: bool = True,
        dist_kwargs=None,
    ) -> None:

        self.kernel = kernel
        self.n_eigenpairs = n_eigenpairs
        self.time_exponent = time_exponent
        self.is_stochastic = is_stochastic
        self.alpha = alpha
        self.symmetrize_kernel = symmetrize_kernel
        self.dist_kwargs = dist_kwargs

    @classmethod
    def laplace_beltrami(
        cls, kernel=GaussianKernel(epsilon=1.0), n_eigenpairs=10, **kwargs,
    ) -> "DiffusionMaps":
        """Instantiate new model to approximate Laplace-Beltrami operator.

        Returns
        -------
        DiffusionMaps
            new instance
        """
        return cls(
            kernel=kernel,
            n_eigenpairs=n_eigenpairs,
            is_stochastic=True,
            alpha=1.0,
            **kwargs,
        )

    @classmethod
    def fokker_planck(
        cls, kernel=GaussianKernel(epsilon=1.0), n_eigenpairs=10, **kwargs,
    ) -> "DiffusionMaps":
        """Instantiate new model to approximate Fokker-Planck operator.

        Returns
        -------
        DiffusionMaps
            new instance
        """
        return cls(
            kernel=kernel,
            n_eigenpairs=n_eigenpairs,
            is_stochastic=True,
            alpha=0.5,
            **kwargs,
        )

    @classmethod
    def graph_laplacian(
        cls, kernel=GaussianKernel(epsilon=1.0), n_eigenpairs=10, **kwargs,
    ) -> "DiffusionMaps":
        """Instantiate new model to approximate graph Laplacian.

        Returns
        -------
        DiffusionMaps
            new instance
        """
        return cls(
            kernel=kernel,
            n_eigenpairs=n_eigenpairs,
            is_stochastic=True,
            alpha=0.0,
            **kwargs,
        )

    def _nystrom(self, kernel_cdist, eigvec, eigvals):
        return kernel_cdist @ mat_dot_diagmat(eigvec, np.reciprocal(eigvals))

    def _perform_dmap_embedding(self, eigenvectors: np.ndarray) -> np.ndarray:

        check_scalar(
            self.time_exponent,
            "time_exponent",
            target_type=(float, np.floating, int, np.integer),
            min_val=0,
            max_val=None,
        )

        if self.time_exponent == 0:
            dmap_embedding = eigenvectors
        else:
            eigvals_time = np.power(self.eigenvalues_, self.time_exponent)
            dmap_embedding = diagmat_dot_mat(eigvals_time, eigenvectors)

        return dmap_embedding

    def _setup_default_dist_kwargs(self):
        from copy import deepcopy

        self.dist_kwargs_ = deepcopy(self.dist_kwargs) or {}
        self.dist_kwargs_.setdefault("cut_off", np.inf)
        self.dist_kwargs_.setdefault("kmin", 0)
        self.dist_kwargs_.setdefault("backend", "guess_optimal")

    def set_coords(self, indices: Union[np.ndarray, List[int]]) -> "DiffusionMaps":
        """Set eigenvector coordinates for parsimonious mapping.

        Parameters
        ----------
        indices
            Index values of ``eigenvalues_`` and respective ``eigenvectors_`` to keep in
            the model.

        Returns
        -------
        DiffusionMaps
            self
        """

        check_is_fitted(self, attributes=["eigenvectors_", "eigenvalues_"])

        indices = np.asarray(indices)

        # type hints for mypy
        self.eigenvectors_: np.ndarray = if1dim_colvec(self.eigenvectors_[:, indices])
        self.eigenvalues_: np.ndarray = self.eigenvalues_[indices]

        return self

    def fit(
        self, X: TransformType, y=None, store_kernel_matrix=False
    ) -> "DiffusionMaps":
        """Compute diffusion kernel matrix and its' eigenpairs.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Training data of shape `(n_samples, n_features)`.
        
        y: None
            ignored

        store_kernel_matrix
            If True, store the kernel matrix in attribute ``kernel_matrix_``.

        Returns
        -------
        DiffusionMaps
            self
        """

        X = self._validate_data(X=X, validate_array_kwargs=dict(ensure_min_samples=2))

        self._setup_features_fit(
            X, features_out=[f"dmap{i}" for i in range(self.n_eigenpairs)]
        )

        self._setup_default_dist_kwargs()

        # Note the DmapKernel is a kernel that wraps another kernel to provides the
        # DMAP specific functionality.
        self._dmap_kernel = DmapKernelFixed(
            internal_kernel=self.kernel,
            is_stochastic=self.is_stochastic,
            alpha=self.alpha,
            symmetrize_kernel=self.symmetrize_kernel,
        )

        # Need to hold X in class to be able to compute cdist distance matrix during
        # out-of-sample transforms.
        self.X_ = PCManifold(
            X, kernel=self._dmap_kernel, dist_kwargs=self.dist_kwargs_,
        )

        kernel_output = self.X_.compute_kernel_matrix()
        (
            kernel_matrix_,
            self._cdist_kwargs,
            ret_extra,
        ) = PCManifoldKernel.read_kernel_output(kernel_output=kernel_output)

        # if key is not present, this is a bug. The value for the key can also be None.
        basis_change_matrix = ret_extra["basis_change_matrix"]

        (
            self.eigenvalues_,
            self.eigenvectors_,
        ) = _DmapKernelAlgorithms.solve_eigenproblem(
            kernel_matrix=kernel_matrix_,
            n_eigenpairs=self.n_eigenpairs,
            is_symmetric=self._dmap_kernel.is_symmetric,
            is_stochastic=self.is_stochastic,
            basis_change_matrix=basis_change_matrix,
        )

        self.eigenvectors_ = self._same_type_X(
            X, values=self.eigenvectors_, feature_names=self.features_out_[1]
        )

        if self._dmap_kernel.is_symmetric_transform(is_pdist=True):
            kernel_matrix_ = _DmapKernelAlgorithms.unsymmetric_kernel_matrix(
                kernel_matrix=kernel_matrix_, basis_change_matrix=basis_change_matrix,
            )

        if store_kernel_matrix:
            self.kernel_matrix_ = kernel_matrix_

        return self

    def transform(self, X: TransformType) -> TransformType:

        r"""Embed out-of-sample points with Nyström extension.

        From solving the eigenproblem of the diffusion kernel :math:`K`
        (:class:`.DmapKernelFixed`)

        .. math::
            K(X,X) \Psi = \Psi \Lambda

        follows the Nyström extension for out-of-sample mappings:

        .. math::
            K(X, Y) \Psi \Lambda^{-1} = \Psi

        where :math:`K(X, Y)` is a component-wise evaluation of the kernel.

        Note, that the Nyström mapping can be used for image mappings irrespective of
        whether the computed kernel matrix :math:`K(X,X)` is symmetric.
        For details on this see :cite:`fernandez_diffusion_2015` (especially equation 5).

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data points of shape `(n_samples, n_features)` to be embedded.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` of shape `(n_samples, n_coords)`
        """
        check_is_fitted(self, ("X_", "eigenvalues_", "eigenvectors_"))

        X = self._validate_data(X, validate_array_kwargs=dict(ensure_min_samples=1))
        self._validate_feature_input(X, direction="transform")

        kernel_output = self.X_.compute_kernel_matrix(X, **self._cdist_kwargs)
        kernel_matrix_cdist, _, _ = PCManifoldKernel.read_kernel_output(
            kernel_output=kernel_output
        )

        eigvec_nystroem = self._nystrom(
            kernel_matrix_cdist,
            eigvec=np.asarray(self.eigenvectors_),
            eigvals=self.eigenvalues_,
        )

        dmap_embedding = self._perform_dmap_embedding(eigvec_nystroem)

        return self._same_type_X(
            X, values=dmap_embedding, feature_names=self.features_out_[1]
        )

    def fit_transform(self, X: TransformType, y=None, **fit_params) -> TransformType:
        """Compute diffusion map from data and apply embedding on same data.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Training data of shape `(n_samples, n_features)`

        y: None
            ignored

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` of shape `(n_samples, n_eigenpairs)`
        """

        X = self._validate_data(X, validate_array_kwargs=dict(ensure_min_samples=2))
        self.fit(X=X, y=y)

        dmap_embedding = self._perform_dmap_embedding(self.eigenvectors_)
        return self._same_type_X(
            X, values=dmap_embedding, feature_names=self.features_out_[1]
        )

    def inverse_transform(self, X: TransformType) -> TransformType:
        """Pre-image from embedding space back to original (ambient) space.

        .. note::
            Currently, this is only a linear map in a least squares sense. Overwrite
            this function for more advanced pre-image mappings.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Out-of-sample data of shape `(n_samples, n_coords)` to
            map from embedding space to original space.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` of shape (`n_samples, n_features)`
        """

        check_is_fitted(self)
        X = self._validate_data(X)
        self._validate_feature_input(X, direction="inverse_transform")

        if not hasattr(self, "inv_coeff_matrix_"):
            self.inv_coeff_matrix_ = scipy.linalg.lstsq(
                np.asarray(self.eigenvectors_), self.X_, cond=None
            )[0]

        X_orig_space = np.asarray(X) @ self.inv_coeff_matrix_
        return self._same_type_X(
            X, values=X_orig_space, feature_names=self.features_in_[1]
        )


class DiffusionMapsVariable(TSCTransformerMixIn):
    """(experimental, not documented)
    .. warning::
        This class is not documented. Contributions are welcome
            * documentation
            * unit- or functional-testing

    References
    ----------
    :cite:`berry_nonparametric_2015`
    :cite:`berry_variable_2016`

    """

    def __init__(
        self,
        epsilon=1.0,
        n_eigenpairs=10,
        nn_bandwidth=10,
        expected_dim=2,
        beta=-0.5,
        symmetrize_kernel=False,
        dist_kwargs=None,
    ):
        self.epsilon = epsilon
        self.n_eigenpairs = n_eigenpairs
        self.expected_dim = expected_dim
        self.beta = beta
        self.nn_bandwidth = nn_bandwidth

        # TODO: To implement: cut_off: float = np.inf (allow also sparsity!)
        # TODO: generalize DmapKernelVariable also to arbitrary kernels

        self.dmap_kernel_ = DmapKernelVariable(
            epsilon=self.epsilon,
            k=nn_bandwidth,
            expected_dim=expected_dim,
            beta=beta,
            symmetrize_kernel=symmetrize_kernel,
        )
        self.alpha = self.dmap_kernel_.alpha  # is computed (depends on beta) in kernel
        self.dist_kwargs = dist_kwargs

    @property
    def peq_est_(self):
        """Estimation of the equilibrium density (p_eq)."""

        #  TODO: there are different suggestions,
        #    q_eps_s as noted pdfp. 5,  OR  eq. (2.3) pdfp 4 rho \approx peq^(-1/2)

        nr_samples = self.q_eps_s_.shape[0]
        return self.q_eps_s_ / (
            nr_samples * (4 * np.pi * self.epsilon) ** (self.expected_dim / 2)
        )

    def fit(self, X: TransformType, y=None, **fit_params):

        X = self._validate_data(X, validate_array_kwargs=dict(ensure_min_samples=2))

        self._setup_features_fit(
            X, features_out=[f"dmap{i}" for i in range(self.n_eigenpairs)]
        )

        self.dist_kwargs = self.dist_kwargs or {}
        self.dist_kwargs.setdefault("cut_off", np.inf)
        self.dist_kwargs.setdefault("kmin", self.nn_bandwidth)
        self.dist_kwargs.setdefault("backend", "guess_optimal")

        pcm = PCManifold(X, kernel=self.dmap_kernel_, dist_kwargs=self.dist_kwargs,)

        # basis_change_matrix is None if not required
        (
            self.operator_matrix_,
            _basis_change_matrix,
            self.rho0_,
            self.rho_,
            self.q0_,
            self.q_eps_s_,
        ) = pcm.compute_kernel_matrix()

        (
            self.eigenvalues_,
            self.eigenvectors_,
        ) = _DmapKernelAlgorithms.solve_eigenproblem(
            kernel_matrix=self.operator_matrix_,
            n_eigenpairs=self.n_eigenpairs,
            is_symmetric=self.dmap_kernel_.is_symmetric,
            is_stochastic=True,
            basis_change_matrix=_basis_change_matrix,
        )

        # TODO: note here the kernel is actually NOT the kernel but the operator matrix
        #  ("L") -> see "Variable bandwidth diffusion maps" by Berry et al.
        #  Maybe think about a way to transform this?

        if self.dmap_kernel_.is_symmetric_transform(is_pdist=True):
            self.operator_matrix_ = _DmapKernelAlgorithms.unsymmetric_kernel_matrix(
                kernel_matrix=self.operator_matrix_,
                basis_change_matrix=_basis_change_matrix,
            )

        # TODO: transformation to match eigenvalues with true kernel matrix.
        self.eigenvalues_ = np.power(
            self.eigenvalues_, 1 / self.epsilon, out=self.eigenvalues_
        )

        # assumes that the eigenvectors are already normed to 1
        #   (which is the default)
        self.eigenvectors_ = np.multiply(
            self.eigenvectors_, np.sqrt(X.shape[0]), out=self.eigenvectors_
        )

        self.eigenvectors_ = self._same_type_X(
            X, values=self.eigenvectors_, feature_names=self.features_out_[1]
        )

        return self

    def transform(self, X: TransformType):
        raise NotImplementedError(
            "A transform for out-of-sample points is currently "
            "not supported. This requires to implement the "
            "'cdist' case for the variable bandwidth diffusion "
            "maps kernel."
        )

    def fit_transform(self, X: TransformType, y=None, **fit_params):
        self.fit(X=X, y=y, **fit_params)
        return self._same_type_X(X, self.eigenvectors_, self.features_out_[1])


class LocalRegressionSelection(BaseEstimator, TSCTransformerMixIn):
    """Automatic selection of functional independent geometric harmonic vectors for
    parsimonious data manifold embedding.

    To measure the functional dependency a local regression regression is performed: The
    larger the residuals between eigenvetor sets the more information they add and are
    therefore more likely to be considered in an embedding.

    The kernel used for the local linear regression has a scale of

    .. code::

        scale = bandwidth_type(distances) / eps_med_scale

    In the referenced paper this is described on page 6, Eq. 11.

    ...

    Parameters
    ----------
    eps_med_scale
        Epsilon scale in kernel of the local linear regression.

    n_subsample
        Number of randomly uniform selected samples to reduce the computational cost of \
        the linear regressions. Lower numbers boost the performance of the selection at \
        the cost of accuracy. The minimum value is 100 samples.

    strategy
        * "dim" - set the expected dimension (fixed set of eigenvectors)
        * "threshold" - choose all eigenvectors that are above the threshold (variable \
        set of eigenpairs)

    intrinsic_dim
        Number of eigenvectors to select with largest residuals.
        
    regress_threshold
        Threshold for local residual to include eigenvectors that are above,
        if strategy="threshold".

    bandwidth_type
        "median" or "mean"

    Attributes
    ----------

    evec_indices_

    residuals_

    References
    ----------

    :cite:`dsilva_parsimonious_2018`

    """

    _cls_valid_strategy = ("dim", "threshold")
    _cls_valid_bandwidth = ("median", "mean")

    def __init__(
        self,
        eps_med_scale=3,
        n_subsample=np.inf,
        strategy="dim",
        intrinsic_dim=2,
        regress_threshold=0.9,
        bandwidth_type="median",
    ):

        self.strategy = strategy

        self.intrinsic_dim = intrinsic_dim
        self.regress_threshold = regress_threshold
        self.bandwidth_type = bandwidth_type

        self.eps_med_scale = eps_med_scale
        self.n_subsample = n_subsample

    def _validate_parameter(self, num_eigenvectors):

        check_scalar(
            self.eps_med_scale,
            name="eps_med_scale",
            target_type=(float, np.floating, int, np.integer),
            min_val=np.finfo(np.float64).eps,  # exclusive zero
            max_val=np.inf,
        )

        check_scalar(
            self.n_subsample,
            name="n_subsample",
            target_type=(int, np.integer),
            min_val=100,
            max_val=np.inf,
        )

        if self.strategy not in self._cls_valid_strategy:
            raise ValueError(f"strategy={self.strategy} is invalid.")

        if self.strategy == "dim":
            check_scalar(
                self.intrinsic_dim,
                name="intrinsic_dim",
                target_type=(int, np.integer),
                min_val=1,
                max_val=num_eigenvectors - 1,
            )

        if self.strategy == "threshold":
            check_scalar(
                self.regress_threshold,
                name="regress_threshold",
                target_type=(float, np.floating),
                min_val=np.finfo(np.float64).eps,
                max_val=1 - np.finfo(np.float64).eps,
            )

        if self.bandwidth_type not in self._cls_valid_bandwidth:
            raise ValueError(
                f"Valid options for 'locregress_bandwidth_type' are "
                f"{self._cls_valid_bandwidth}. Got: {self.bandwidth_type}"
            )

    def _single_residual_local_regression(
        self, domain_eigenvectors, target_eigenvector
    ):
        # Helper function for residuals_local_regression_eigenvectors

        nr_samples = domain_eigenvectors.shape[0]

        distance_matrix_eigvec = scipy.spatial.distance.pdist(
            domain_eigenvectors, "euclidean"
        )
        distance_matrix_eigvec = scipy.spatial.distance.squareform(
            distance_matrix_eigvec
        )

        # epsilon used for regression see referenced paper at page 6, equation 11
        # kernel scale (bandwidth) for the regression algorithm
        if self.bandwidth_type == "median":
            eps_regression = (
                np.median(np.square(distance_matrix_eigvec.flatten()))
                / self.eps_med_scale
            )
        else:  # self.bandwidth_type == "mean":
            eps_regression = (
                np.mean(np.square(distance_matrix_eigvec.flatten()))
                / self.eps_med_scale
            )

        if eps_regression == 0:
            eps_regression = np.finfo(np.float64).eps

        # equation 11 in referenced paper, corresponding to the weights
        kernel_eigvec = np.exp(-1 * np.square(distance_matrix_eigvec) / eps_regression)

        assert not np.isnan(eps_regression).any()

        collected_solution_coeffs = np.zeros((nr_samples, nr_samples))

        const_ones = np.ones((domain_eigenvectors.shape[0], 1))

        for i in range(nr_samples):  # TODO: this loop is extremely expensive...
            # This is a weighted least squares problem
            # see https://en.wikipedia.org/wiki/Weighted_least_squares
            # X.T W X = X.T W y
            #
            # notation used here
            #   data_matrix.T @ kernel_eigvec[i, :] data_matrix =
            #   data_matrix.T @ kernel_eig_vec[i, :] @ target_values

            # From original MATLAB code (Dsilva):
            # % W = diag(kernel_eigvec(i,:))  -- these are the weights
            # % Xx = like above  -- data matrix
            # % A  = solution matrix
            #
            # Wx = diag(kernel_eigvec(i,:))
            # A = (Xx'*Wx*Xx)\(Xx'*Wx)

            # const first column for the shift of the linear regression
            data_matrix = np.hstack(
                [const_ones, domain_eigenvectors - domain_eigenvectors[i, :]]
            )

            # data_matrix.T @ np.diag(kernel_eigvec[i, :])
            # --> partial evaluation of of X.T @ W (using the above notation) as it is
            #     used twice
            # --> use elementwise because W is a diagonal matrix
            weight_data_matrix = data_matrix.T * kernel_eigvec[i, :]

            # weight_data_matrix @ data_matrix --> X.T W X from the above notation,
            # the RHS is y=1
            current_solution_coeff, res_, rank_, singular_values_ = np.linalg.lstsq(
                (weight_data_matrix @ data_matrix), weight_data_matrix, rcond=1e-6
            )

            # TODO: when we are taking only the 0-th row, why are we computing the full
            #  matrix then?
            # TODO: why are we only taking the 0-th row?
            collected_solution_coeffs[i, :] = current_solution_coeff[0, :]

        estimated_target_values = collected_solution_coeffs @ target_eigenvector

        ######
        # START
        # The following code was provided. However, this part
        # diverges from the paper. I use the paper version as it also passes the tests.
        ###
        # target_stddev = np.std(target_eigenvector)
        #
        # # used to scale the (target_eigenvector - estimated_target_values)**2 values
        # error_scale = 1 - np.diag(collected_solution_coeffs)
        # if target_stddev == 0:
        #     raise NumericalMathError(
        #           "Standard deviation is zero. Cannot divide by zero.")
        # if (error_scale == 0).any():
        #     error_scale[error_scale == 0] = 1  # avoid dividing by zero
        #
        # # NOTE: differs from  eq. 12 in paper, was provided in code
        # # RMSE with error scale and dividing the residual by the target standard
        # #  deviation
        # # Not quite sure where this residual measure comes from, therefore the other
        # #   one is used...
        # residual = np.sqrt(np.mean(np.square(
        #     (target_eigenvector - estimated_target_values) / error_scale))) /
        #     target_stddev
        # END
        ######

        # residual according to paper, equation 12  (also used in PCM code)
        residual = np.sqrt(
            np.sum(np.square((target_eigenvector - estimated_target_values)))
            / np.sum(np.square(target_eigenvector))
        )

        return residual

    def _set_indices(self):

        # For the strategies numerical values are required. Therefore, set the first
        # residual (nan) here to the invalid value -1. This value makes sure, that
        # the first coordinate is always chosen.
        residuals = self.residuals_.copy()
        residuals[0] = -1

        # Strategy 1, according to user input:
        # User provides knowledge of the (expected) intrinsic dimension -- could also be
        # a result of estimation
        if self.strategy == "dim":
            if self.intrinsic_dim == 1:
                self.evec_indices_ = np.array([1])
            else:
                # NOTE: negative residuals to obtain the largest -- argpartition are the
                # respective indices
                self.evec_indices_ = np.sort(
                    np.argpartition(-residuals, self.intrinsic_dim)[
                        : self.intrinsic_dim
                    ]
                )

        # User provides a threshold for the residuals. All eigenfunctions above this value
        # are included to parametrize the manifold.
        else:  #  self.strategy == "threshold":
            self.evec_indices_ = np.sort(
                np.where(residuals > self.regress_threshold)[0]
            )

    def fit(self, X: TransformType, y=None, **fit_params) -> "LocalRegressionSelection":
        """Select indices according to strategy.

        Parameters
        ----------
        X
            Eigenvectors of shape `(n_samples, n_eigenvectors)` to make selection on.

        Returns
        -------
        LocalRegressionSelection
            self
        """

        # Code received from Yannis and adapted (author unknown). The code that was
        # received was translated from a MATLAB version from Dsilva.

        # TODO: performance issues: There are 2 loops, the inner loop goes over all
        #  samples
        #  1: use numba, numexpr or try to vectorize numpy code, last resort: cython
        #  2: parallelize code (outer loop)

        # Note: this saves self._transform_columns = X.columns
        # Later on not all of these columns are required because of the selection
        # performed.

        X = self._validate_data(X, validate_array_kwargs=dict(ensure_min_features=2))
        num_eigenvectors = X.shape[1]

        self._validate_parameter(num_eigenvectors)

        if self.n_subsample is not None:
            eigvec, _ = random_subsample(X, self.n_subsample)
        else:
            eigvec = np.asarray(X)

        self.residuals_ = np.zeros(num_eigenvectors)
        self.residuals_[0] = np.nan  # the const (trivial) eigenvector is ignored
        # the first eigenvector is always taken, therefore receives a 1
        self.residuals_[1] = 1.0

        for i in range(2, num_eigenvectors):
            self.residuals_[i] = self._single_residual_local_regression(
                domain_eigenvectors=eigvec[:, 1:i], target_eigenvector=eigvec[:, i]
            )

        self._set_indices()

        # Cannot use self._setup_features_fit here because the columns (if they exist)
        # are partially selected.
        if self._has_feature_names(X):
            self._setup_frame_input_fit(
                features_in=X.columns, features_out=X.columns[self.evec_indices_],
            )
        else:
            self._setup_array_input_fit(
                features_in=X.shape[1], features_out=len(self.evec_indices_)
            )

        return self

    def transform(self, X: TransformType) -> TransformType:
        """Select parsimonious representation of full set of eigenvectors.

        Parameters
        ----------

        X
            Eigenvectors of shape `(n_samples, n_eigenvectors)` to carry out selection.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` of shape `(n_samples, n_evec_indices)`
        """

        X = self._validate_data(X)
        self._validate_feature_input(X, direction="transform")

        # choose eigenvectors
        X_selected = self._same_type_X(
            X,
            np.asarray(X)[:, self.evec_indices_],
            feature_names=self.features_out_[1],
        )

        return X_selected

    def inverse_transform(self, X: TransformType):
        """
        .. warning::
            Not implemented.
        """

        # TODO: from the philosophy the inverse_transform should map
        #   \Psi_selected -> \Psi_full
        #  However this is usually not what we are interested in. Instead it is more
        #  likely that someone wants to do the inverse_transform like in DMAP:
        #   \Psi_selected -> X (original data).

        raise NotImplementedError(
            "The inverse_transform should be carried out with an DMAP, which has all "
            "eigenvectors (not selected)."
        )
