from typing import Union

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_scalar

from datafold.dynfold.base import DmapKernelMethod, TransformType, TSCTransformerMixIn
from datafold.pcfold import DmapKernelFixed, PCManifold
from datafold.pcfold.kernels import DmapKernelVariable
from datafold.utils.general import (
    diagmat_dot_mat,
    if1dim_colvec,
    is_float,
    is_integer,
    mat_dot_diagmat,
    random_subsample,
)


class DiffusionMaps(DmapKernelMethod, TSCTransformerMixIn):
    """Geometrical parametrization of data manifold based on diffusion processes with
    (fixed) bandwidth.

    The diffusion maps allows to perform non-linear dimension reduction
    (also known as unsupervervised learning or manifold learning) and also approximates
    operators

        - Laplace-Beltrami
        - Fokker-Plank
        - Graph Laplacian

    Parameters
    ----------
    epsilon
        Bandwidth/scale of diffusion map kernel (see :py:class:`DmapKernelFixed`).

    n_eigenpairs
        Number of eigenpairs to compute from computed diffusion kernel matrix.

    time_exponent
        Exponent of eigenvalues as the time progress of the diffusion process.

    cut_off
        Distance cut off, kernel values with a corresponding larger Euclidean distance
        are set to zero. Lower values increases the sparsity of kernel matrices and
        faster computation of eigenpairs at the cost of accuracy.

    is_stochastic
        If True the diffusion kernel matrix is normalized (stochastic rows).

    alpha
        Re-normalization parameter. Set to `alpha=0` for graph laplacian, `alpha=0.5`
        Fokker-Plank and `alpha=1` for Laplace-Beltrami (`is_stochastic=True` in all
        cases).

    symmetrize_kernel
        If True a conjugate transformation of non-symmetric kernel matrices is performed.
        This improves numerical stability and allows to use eigensolver algorithms
        designed for Hermitian matrices. If kernel is symmetric already (if
        `is_stochastic=False`, then the parameter has no effect).

    dist_backend
        Backend of distance matrix computation. Defaults to `guess_optimal`,
        which selects the backend based on the selectin of ``cut_off`` and the
        available algorithms. See also :class:`.DistanceAlgorithm`.

    dist_backend_kwargs,
        Keyword arguments handled to distance matrix backend.

    Attributes
    ----------
    X_: PCManifold
        Training data during fit, is required for out-of-sample mappings. Equipped with \
        kernel :py:class:`DmapKernelFixed`

    eigenvalues_ : numpy.ndarray
        Eigenvalues of diffusion kernel in decreasing magnitude order.

    eigenvectors_: TSCDataFrame, pandas.DataFrame, numpy.ndarray
        Eigenvectors of the kernel matrix to parametrizes the data manifold.

    inv_coeff_matrix_: numpy.ndarray
        Coeffficient matrix to map points from embedding space back to original space.\
        Computation is delayed until `inverse_transform` is called for the first time.

    kernel_matrix_ : numpy.ndarray
        Diffusion kernel matrix computed during fit.

        .. note::
            Currently, the kernel matrix is only used for testing. It may be removed.


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
        epsilon: float = 1.0,
        n_eigenpairs: int = 10,
        time_exponent: float = 0,
        cut_off: float = np.inf,
        is_stochastic: bool = True,
        alpha: float = 1,
        symmetrize_kernel: bool = True,
        dist_backend: str = "guess_optimal",
        dist_backend_kwargs=None,
    ) -> None:

        self.time_exponent = time_exponent

        super(DiffusionMaps, self).__init__(
            epsilon=epsilon,
            n_eigenpairs=n_eigenpairs,
            cut_off=cut_off,
            is_stochastic=is_stochastic,
            alpha=alpha,
            symmetrize_kernel=symmetrize_kernel,
            dist_backend=dist_backend,
            dist_backend_kwargs=dist_backend_kwargs,
        )

    @classmethod
    def from_operator_name(cls, name: str, **kwargs) -> "DiffusionMaps":
        """Instantiate new model to approximate operator (to be selected by name).

        Parameters
        ----------
        name
            "laplace_beltrami", "fokker_plank", "graph_laplacian" or "rbf".

        **kwargs
            All parameters in :py:class:`DiffusionMaps` but ``alpha``.

        Returns
        -------
        DiffusionMaps
            self
        """

        if name == "laplace_beltrami":
            eigfunc_interp = cls.laplace_beltrami(**kwargs)
        elif name == "fokker_planck":
            eigfunc_interp = cls.fokker_planck(**kwargs)
        elif name == "graph_laplacian":
            eigfunc_interp = cls.graph_laplacian(**kwargs)
        elif name == "rbf":
            eigfunc_interp = cls.rbf_kernel(**kwargs)
        else:
            raise ValueError(
                f"name='{name}' not known. Choose from {cls._cls_valid_operator_names}"
            )

        if name not in cls._cls_valid_operator_names:
            raise NotImplementedError(
                f"This is a bug. name={name} each name has to be "
                f"listed in VALID_OPERATOR_NAMES"
            )

        return eigfunc_interp

    @classmethod
    def laplace_beltrami(
        cls, epsilon=1.0, n_eigenpairs=10, **kwargs,
    ) -> "DiffusionMaps":
        """Instantiate new model to approximate Laplace-Beltrami operator.

        Returns
        -------
        DiffusionMaps
            new instance
        """
        return cls(
            epsilon=epsilon,
            n_eigenpairs=n_eigenpairs,
            is_stochastic=True,
            alpha=1.0,
            **kwargs,
        )

    @classmethod
    def fokker_planck(cls, epsilon=1.0, n_eigenpairs=10, **kwargs,) -> "DiffusionMaps":
        """Instantiate new model to approximate Fokker-Planck operator.

        Returns
        -------
        DiffusionMaps
            new instance
        """
        return cls(
            epsilon=epsilon,
            n_eigenpairs=n_eigenpairs,
            is_stochastic=True,
            alpha=0.5,
            **kwargs,
        )

    @classmethod
    def graph_laplacian(
        cls, epsilon=1.0, n_eigenpairs=10, **kwargs,
    ) -> "DiffusionMaps":
        """Instantiate new model to approximate graph Laplacian.

        Returns
        -------
        DiffusionMaps
            new instance
        """
        return cls(
            epsilon=epsilon,
            n_eigenpairs=n_eigenpairs,
            is_stochastic=True,
            alpha=0.0,
            **kwargs,
        )

    @classmethod
    def rbf_kernel(cls, epsilon=1.0, n_eigenpairs=10, **kwargs,) -> "DiffusionMaps":
        """Instantiate new model to approximate geometric harmonic functions of
        radial basis function kernel.

        Returns
        -------
        DiffusionMaps
            new instance
        """
        return cls(
            epsilon=epsilon, n_eigenpairs=n_eigenpairs, is_stochastic=False, **kwargs,
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

    def _setup_kernel(self):
        self._kernel = DmapKernelFixed(
            epsilon=self.epsilon,
            is_stochastic=self.is_stochastic,
            alpha=self.alpha,
            symmetrize_kernel=self.symmetrize_kernel,
        )

    def set_coords(self, indices: np.ndarray) -> "DiffusionMaps":
        """Set eigenvector coordinates for parsimonious mapping.

        Parameters
        ----------
        indices
            Index values (columns) of ``eigenvalues_`` and ``eigenvectors_`` to keep in
            the model.

        Returns
        -------
        DiffusionMaps
            self
        """

        check_is_fitted(self, attributes=["eigenvectors_", "eigenvalues_"])

        # type hints for mypy
        self.eigenvectors_: np.ndarray = if1dim_colvec(self.eigenvectors_[:, indices])
        self.eigenvalues_: np.ndarray = self.eigenvalues_[indices]

        return self

    def fit(self, X: TransformType, y=None, **fit_params) -> "DiffusionMaps":
        """Compute diffusion kernel matrix and its' eigenpairs for manifold
        parametrization.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Training data with shape `(n_samples, n_features)`.
        
        y: None
            ignored

        Returns
        -------
        DiffusionMaps
            self
        """

        X = self._validate_data(X=X, validate_array_kwargs=dict(ensure_min_samples=2))

        self._setup_features_fit(
            X, features_out=[f"dmap{i}" for i in range(self.n_eigenpairs)]
        )

        self._setup_kernel()

        # Need to hold X in class to be able to compute cdist distance matrix which is
        # required for out-of-sample transforms
        self.X_ = PCManifold(
            X,
            kernel=self.kernel_,
            cut_off=self.cut_off,
            dist_backend=self.dist_backend,
            **(self.dist_backend_kwargs or {}),
        )

        # basis_change_matrix is None if not required
        # save kernel_matrix for now to use it for testing, but it may not be necessary
        # for larger problems
        (
            self.kernel_matrix_,
            _basis_change_matrix,
            self._row_sums_alpha,
        ) = self.X_.compute_kernel_matrix()

        self.eigenvalues_, self.eigenvectors_ = self._solve_eigenproblem(
            self.kernel_matrix_, _basis_change_matrix
        )

        self.eigenvectors_ = self._same_type_X(
            X, values=self.eigenvectors_, set_columns=self.features_out_[1]
        )

        if self.kernel_.is_symmetric_transform(is_pdist=True):
            self.kernel_matrix_ = self._unsymmetric_kernel_matrix(
                kernel_matrix=self.kernel_matrix_,
                basis_change_matrix=_basis_change_matrix,
            )

        return self

    def transform(self, X: TransformType) -> TransformType:
        r"""Map out-of-sample points into embedding space with Nyström extension.

        From solving the eigenproblem of the kernel diffusion matrix :math:`K`

        .. math::
            K(X,X) \Psi = \Psi \Lambda

        follows the Nyström extension for out-of-sample mappings:

        .. math::
            K(X, Y) \Psi \Lambda^{-1} = \Psi

        where :math:`K(X, Y)` is a component-wise evaluation of the kernel.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Out-of-sample points with shape `(n_samples, n_features)` to map to embedding
            space.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` with shape `(n_samples, n_coords)`
        """

        check_is_fitted(self, ("X_", "eigenvalues_", "eigenvectors_", "kernel_"))

        X = self._validate_data(X, validate_array_kwargs=dict(ensure_min_samples=1))
        self._validate_feature_input(X, direction="transform")

        kernel_matrix_cdist, _, _ = self.X_.compute_kernel_matrix(
            X, row_sums_alpha_fit=self._row_sums_alpha
        )

        eigvec_nystroem = self._nystrom(
            kernel_matrix_cdist,
            eigvec=np.asarray(self.eigenvectors_),
            eigvals=self.eigenvalues_,
        )

        dmap_embedding = self._perform_dmap_embedding(eigvec_nystroem)

        return self._same_type_X(
            X, values=dmap_embedding, set_columns=self.features_out_[1]
        )

    def fit_transform(self, X: TransformType, y=None, **fit_params) -> TransformType:
        """Fit model and map data directly to embedding space.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Training data with shape `(n_samples, n_features)`

        y: None
            ignored

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` with shape `(n_samples, n_eigenpairs)`
        """

        X = self._validate_data(X, validate_array_kwargs=dict(ensure_min_samples=2))
        self.fit(X=X, y=y)

        dmap_embedding = self._perform_dmap_embedding(self.eigenvectors_)
        return self._same_type_X(
            X, values=dmap_embedding, set_columns=self.features_out_[1]
        )

    def inverse_transform(self, X: TransformType) -> TransformType:
        """Map points from embedding space back to original (ambient) space.

        .. note::
            Currently, this is only  a linear map in a least squares sense. Overwrite
            this function for more advanced inverse mappings.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Out-of-sample points with shape `(n_samples, n_coords)` to
            map from embedding space to original space.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` with shape (`n_samples, n_features)`
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
            X, values=X_orig_space, set_columns=self.features_in_[1]
        )


class DiffusionMapsVariable(DmapKernelMethod, TSCTransformerMixIn):
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
        dist_backend="brute",
        dist_backend_kwargs=None,
    ):

        self.expected_dim = expected_dim
        self.beta = beta
        self.nn_bandwidth = nn_bandwidth

        # TODO: To implement: cut_off: float = np.inf (allow also sparsity!)
        super(DiffusionMapsVariable, self).__init__(
            epsilon=epsilon,
            n_eigenpairs=n_eigenpairs,
            cut_off=None,
            is_stochastic=False,
            alpha=-1,
            symmetrize_kernel=symmetrize_kernel,
            dist_backend=dist_backend,
            dist_backend_kwargs=dist_backend_kwargs,
        )

        self._kernel = DmapKernelVariable(
            epsilon=self.epsilon,
            k=nn_bandwidth,
            expected_dim=expected_dim,
            beta=beta,
            symmetrize_kernel=symmetrize_kernel,
        )
        self.alpha = self.kernel_.alpha  # is computed (depends on beta) in kernel

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

        pcm = PCManifold(
            X,
            kernel=self.kernel_,
            cut_off=self.cut_off,
            dist_backend=self.dist_backend,
            **(self.dist_backend_kwargs or {}),
        )

        # basis_change_matrix is None if not required
        (
            self.operator_matrix_,
            _basis_change_matrix,
            self.rho0_,
            self.rho_,
            self.q0_,
            self.q_eps_s_,
        ) = pcm.compute_kernel_matrix()

        self.eigenvalues_, self.eigenvectors_ = self._solve_eigenproblem(
            self.operator_matrix_, _basis_change_matrix
        )

        # TODO: note here the kernel is actually NOT the kernel but the operator matrix
        #  ("L") -> see "Variable bandwidth diffusion maps" by Berry et al.
        #  Maybe think about a way to transform this?

        if self.kernel_.is_symmetric_transform(is_pdist=True):
            self.operator_matrix_ = self._unsymmetric_kernel_matrix(
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
            X, values=self.eigenvectors_, set_columns=self.features_out_[1]
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
    """Automatic selection of functional independent geometric harmonic vectors for or
    parsimonious representation.

    To measure the functional dependency a local regression regression is performed: The
    larger the residual between eigenvetor sets the larger the sets are.

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
        # The following code was provided by the received code. However, this part
        # diverges from the paper. Therefore, I use the paper as it also passes the tests.
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
            Eigenvectors with shape `(n_samples, n_eigenvectors)` to make selection on.

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
            Eigenvectors  with shape `(n_samples, n_eigenvectors)` to carry out selection.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` with shape `(n_samples, n_evec_indices)`
        """

        X = self._validate_data(X)
        self._validate_feature_input(X, direction="transform")

        # choose eigenvectors
        X_selected = self._same_type_X(
            X, np.asarray(X)[:, self.evec_indices_], set_columns=self.features_out_[1],
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
