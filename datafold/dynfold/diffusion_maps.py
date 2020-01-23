"""Diffusion maps module.

This module implements the diffusion maps method for dimensionality reduction,
as introduced in:

# TODO: include as cite in documention!
Coifman, R. R., & Lafon, S. (2006). Diffusion maps. Applied and Computational Harmonic
Analysis, 21(1), 5–30. DOI:10.1016/j.acha.2006.04.006
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
from sklearn.base import TransformerMixin

from datafold.dynfold.kernel import DmapKernelFixed, DmapKernelVariable, KernelMethod
from datafold.dynfold.utils import downsample
from datafold.pcfold.pointcloud import PCManifold
from datafold.pcfold.timeseries.base import (
    TSCTransformMixIn,
    TF_ALLOWED_TYPES,
)
from datafold.utils.datastructure import is_float, is_integer, if1dim_rowvec
from datafold.utils.maths import diagmat_dot_mat, mat_dot_diagmat


class DiffusionMaps(KernelMethod, TSCTransformMixIn):
    """Nonlinear dimension reduction by parametrizing a manifold with diffusion maps.
    Attributes:

    eigenvalues_ : np.ndarray
        Eigenvalues of kernel matrix.
    eigenvectors_ : np.ndarray
        Eigenvectors of the kernel matrix, which can be used to parametrize the
        manifold of X.
    """

    VALID_OPERATOR_NAMES = [
        "laplace_beltrami",
        "fokker_planck",
        "graph_laplacian",
        "rbf",
    ]

    def __init__(
        self,
        epsilon: float = 1.0,
        num_eigenpairs: int = 10,
        cut_off: float = np.inf,  # TODO: can provide to optimize the cut_off via PCM
        is_stochastic: bool = True,
        alpha: float = 1,
        # NOTE for docu: if is_stochastic=False, then this is not really required
        symmetrize_kernel=True,
        use_cuda=False,
        dist_backend="guess_optimal",
        dist_backend_kwargs=None,
    ) -> None:

        """Initialize base of diffusion maps object.

        This function computes the eigen-decomposition of the transition matrix
        associated to a random walk on the data using a (fixed) bandwidth equal to
        epsilon.

        NOTE for docu: if is_stochastic=False, it is actually not DMAP anymore, but a
          RadialBasisKernel, so this option is only for convenience.

        """

        super(DiffusionMaps, self).__init__(
            epsilon,
            num_eigenpairs,
            cut_off,
            is_stochastic,
            alpha,
            symmetrize_kernel,
            use_cuda,
            dist_backend,
            dist_backend_kwargs,
        )

        self._kernel = DmapKernelFixed(
            epsilon=epsilon,
            is_stochastic=is_stochastic,
            alpha=alpha,
            symmetrize_kernel=symmetrize_kernel,
        )

    @classmethod
    def from_operator_name(cls, name, **kwargs):

        if name == "laplace_beltrami":
            eigfunc_interp = cls.laplace_beltrami(**kwargs)
        elif name == "fokker_planck":
            eigfunc_interp = cls.fokker_planck(**kwargs)
        elif name == "graph_laplacian":
            eigfunc_interp = cls.graph_laplacian(**kwargs)
        elif name == "rbf":
            eigfunc_interp = cls.rbf(**kwargs)
        else:
            raise ValueError(
                f"name='{name}' not known. Choose from {cls.VALID_OPERATOR_NAMES}"
            )

        if name not in cls.VALID_OPERATOR_NAMES:
            raise NotImplementedError(
                f"This is a bug. name={name} each name has to be "
                f"listed in VALID_OPERATOR_NAMES"
            )

        return eigfunc_interp

    @classmethod
    def laplace_beltrami(
        cls, epsilon=1.0, num_eigenpairs=10, **kwargs,
    ):
        return cls(
            epsilon=epsilon,
            num_eigenpairs=num_eigenpairs,
            is_stochastic=True,
            alpha=1.0,
            **kwargs,
        )

    @classmethod
    def fokker_planck(
        cls, epsilon=1.0, num_eigenpairs=10, **kwargs,
    ):
        return cls(
            epsilon=epsilon,
            num_eigenpairs=num_eigenpairs,
            is_stochastic=True,
            alpha=0.5,
            **kwargs,
        )

    @classmethod
    def graph_laplacian(
        cls, epsilon=1.0, num_eigenpairs=10, **kwargs,
    ):
        return cls(
            epsilon=epsilon,
            num_eigenpairs=num_eigenpairs,
            is_stochastic=True,
            alpha=0.0,
            **kwargs,
        )

    @classmethod
    def rbf(
        cls, epsilon=1.0, num_eigenpairs=10, **kwargs,
    ):
        return cls(
            epsilon=epsilon,
            num_eigenpairs=num_eigenpairs,
            is_stochastic=False,
            **kwargs,
        )

    def _nystrom(self, kernel_cdist, eigvec, eigvals):
        """From eigenproblem:
            K(X,X) \Psi = \Psi \Lambda

        follows Nyström (out-of-sample):

            K(X, Y) \Psi \Lambda^-1 = \Psi
        """

        return kernel_cdist @ mat_dot_diagmat(eigvec, np.reciprocal(eigvals))

    def _select_eigpairs_indices(self, indices):
        if indices is not None:
            _selected_eigvect = self.eigenvectors_[:, indices]
            _selected_eigvals = self.eigenvalues_[indices]
        else:
            _selected_eigvect = self.eigenvectors_
            _selected_eigvals = self.eigenvalues_

        return if1dim_rowvec(_selected_eigvect), _selected_eigvals

    def _dmap_embedding(self, eigvect, eigvals, t):
        if t == 0:
            return eigvect
        else:
            eigvals_time = np.power(eigvals, t)
            return diagmat_dot_mat(eigvals_time, eigvect)

    def fit(self, X: TF_ALLOWED_TYPES, y=None, **fit_params) -> "DiffusionMaps":
        """

        Returns
        -------
        DiffusionMaps
            self
        """

        super(DiffusionMaps, self).fit(
            X, y, transform_columns=[f"dmap{i}" for i in range(self.num_eigenpairs)]
        )

        # Need to hold X in class to be able to compute cdist distance matrix which is
        # required for out-of-sample transforms
        self.X = PCManifold(
            X,
            kernel=self.kernel_,
            cut_off=self.cut_off,
            dist_backend=self.dist_backend,
            **self.dist_backend_kwargs,
        )

        # basis_change_matrix is None if not required
        # save kernel_matrix for now to use it for testing, but it may not be necessary
        # for larger problems
        (
            self.kernel_matrix_,
            _basis_change_matrix,
            self._row_sums_alpha,
        ) = self.X.compute_kernel_matrix()

        self.eigenvalues_, self.eigenvectors_ = self.solve_eigenproblem(
            self.kernel_matrix_, _basis_change_matrix, self.use_cuda
        )

        if self.kernel_.is_symmetric_transform(is_pdist=True):
            self.kernel_matrix_ = self._unsymmetric_kernel_matrix(
                kernel_matrix=self.kernel_matrix_,
                basis_change_matrix=_basis_change_matrix,
            )

        return self

    def transform(self, X, indices=None, t=0):

        """
        Uses Nyström for out-of-sample functionality.

        Parameters
        ----------
        X
        y
        indices

        Returns
        -------

        """

        if t < 0:
            raise ValueError("'t' must be greater than zero")

        super(DiffusionMaps, self).transform(X)

        kernel_matrix_cdist, _, _ = self.X.compute_kernel_matrix(
            X, row_sums_alpha_fit=self._row_sums_alpha
        )

        _selected_eigvect, _selected_eigvals = self._select_eigpairs_indices(indices)

        eigvec_embedding = self._nystrom(
            kernel_matrix_cdist, eigvec=_selected_eigvect, eigvals=_selected_eigvals,
        )

        dmap_embedding = self._dmap_embedding(
            eigvect=eigvec_embedding, eigvals=_selected_eigvals, t=t
        )
        return self._return_same_type_X(
            X, values=dmap_embedding, columns=self._transform_columns
        )

    def fit_transform(self, X, y=None, indices=None, t=0):

        if t < 0:
            raise ValueError("'t' must be greater than zero")

        self.fit(X, y)

        _selected_eigvect, _selected_eigvals = self._select_eigpairs_indices(indices)

        dmap_embedding = self._dmap_embedding(
            eigvect=_selected_eigvect, eigvals=_selected_eigvals, t=t
        )

        return self._return_same_type_X(
            X, values=dmap_embedding, columns=self._transform_columns
        )

    def inverse_transform(self, X: TF_ALLOWED_TYPES):
        super(DiffusionMaps, self).inverse_transform(X)

        import scipy.linalg

        coeff_matrix = scipy.linalg.lstsq(self.eigenvectors_.T, self.X, rcond=1e-14)

        X_orig_space = np.asarray(X) @ coeff_matrix
        return self._return_same_type_X(
            X, values=X_orig_space, columns=self._fit_columns
        )


class DiffusionMapsVariable(KernelMethod, TSCTransformMixIn):
    def __init__(
        self,
        epsilon=1.0,
        num_eigenpairs=10,
        nn_bandwidth=10,
        expected_dim=2,
        beta=-0.5,
        symmetrize_kernel=False,
        use_cuda=False,
        dist_backend="brute",
        dist_backend_kwargs=None,
    ):

        self.expected_dim = expected_dim
        self.beta = beta
        self.nn_bandwidth = nn_bandwidth

        # TODO: To implement: cut_off: float = np.inf (allow also sparsity!)
        super(DiffusionMapsVariable, self).__init__(
            epsilon,
            num_eigenpairs,
            None,
            False,
            -1,
            symmetrize_kernel,
            use_cuda,
            dist_backend,
            dist_backend_kwargs,
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

    def fit(self, X: TF_ALLOWED_TYPES, y=None, **fit_params):

        super(DiffusionMapsVariable, self).fit(
            X, y, transform_columns=[f"dmap{i}" for i in range(self.num_eigenpairs)]
        )

        pcm = PCManifold(
            X,
            kernel=self.kernel_,
            cut_off=self.cut_off,
            dist_backend=self.dist_backend,
            **self.dist_backend_kwargs,
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

        self.eigenvalues_, self.eigenvectors_ = self.solve_eigenproblem(
            self.operator_matrix_, _basis_change_matrix, self.use_cuda
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

        self.eigenvectors_ = (
            self.eigenvectors_
            / np.linalg.norm(self.eigenvectors_, axis=0)[np.newaxis, :]
            * np.sqrt(pcm.shape[0])
        )

        return self

    def transform(self, X: TF_ALLOWED_TYPES):
        raise NotImplementedError(
            "A transform for out-of-sample points is currently "
            "not supported. This requires to implement the "
            "'cdist' case for the variable bandwidth diffusion "
            "maps kernel."
        )

    def fit_transform(self, X: TF_ALLOWED_TYPES, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self._return_same_type_X(X, self.eigenvectors_, self._transform_columns)


class LocalRegressionSelection(TransformerMixin):

    VALID_STRATEGY = ("dim", "threshold")

    def __init__(
        self,
        eps_med_scale=3,
        n_subsample=np.inf,
        strategy="dim",
        intrinsic_dim=2,
        regress_threshold=0.9,
        bandwidth_type="median",
    ):
        """
        Select independent eigenfunctions via local linear regression.

        See:
        # TODO: use Sphinx for citation to the paper.
        Dsilva et al. 2015, https://arxiv.org/abs/1505.06118v1
        'Parsimonious Representation of Nonlinear Dynamical Systems Through Manifold
        Learning: A Chemotaxis Case Study'

        Parameters
        ----------

        eps_med_scale: float = 3
            Scale to use in the local linear regression kernel. The kernel is Gaussian
            with width median(distances)/eps_med_scale. A typical value is
            eps_med_scale=3. In the referenced paper this parameter is described on
            page 6 at equation 11.
        n_subsample: Optional[int] = None
            Computing the residuals is expensive. By setting the number of randomly
            chosen samples, the computation speed can speed up.
        intrinsic_dim: Optional[int] = None
            Select (expected) intrinsic dimensionality of the manifold.
        regress_threshold: Optional[float] = None
            Select threshold for local residual. All eigenvectors are included that
            have a larger residual.
        n_subsample: Optional[int] = np.inf
            See method `residuals_local_regression_eigenvectors`.
        """

        # TODO: all parameter checks

        if n_subsample < 100:
            raise ValueError(
                f"Parameter use_samples has to be larger than 100 samples. Got "
                f"n_subsample={n_subsample}."
            )

        if strategy not in self.VALID_STRATEGY:
            raise ValueError(f"strategy={strategy} is invaliv.")

        if strategy == "dim" and (not is_integer(intrinsic_dim) or intrinsic_dim < 0):
            # first condition: only raise error if variable in use
            raise ValueError("'intrinsic_dim' has to be non-negative integer.")

        if strategy == "threshold" and (
            not is_float(regress_threshold) or 0.0 > regress_threshold > 1.0
        ):
            raise ValueError(
                f"'regress_threshold' has to be non-negative float value between (0, 1). "
                f"Got {regress_threshold}"
            )

        valid_bandwidth_types = ["median", "mean"]
        if bandwidth_type not in valid_bandwidth_types:
            raise ValueError(
                f"Valid options for 'locregress_bandwidth_type' are "
                f"{valid_bandwidth_types}. Got: {bandwidth_type}"
            )

        self.strategy = strategy

        self.intrinsic_dim = intrinsic_dim
        self.regress_threshold = regress_threshold
        self.bandwidth_type = bandwidth_type

        self.eps_med_scale = eps_med_scale  # TODO: checks
        self.n_subsample = n_subsample  # TODO: checks

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
        elif self.bandwidth_type == "mean":
            eps_regression = (
                np.mean(np.square(distance_matrix_eigvec.flatten()))
                / self.eps_med_scale
            )
        else:
            # usually an error is raised in __init__, but user could set it afterwards
            # to an invalid value
            raise RuntimeError("Invalid bandwidth type.")

        if eps_regression == 0:
            eps_regression = np.finfo(float).eps

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

    def fit(self, X: np.ndarray, y=None):
        """

        Returns
        -------
        X
            Residuals of local linear regression for each eigenvector.
        """

        # Code received from Yannis and adapted (author unknown). The code that was
        # received was translated from a MATLAB version from Dsilva.

        # TODO: performance issues: There are 2 loops, the inner loop goes over all
        #  samples
        #  1: use numba, numexpr or try to vectorize numpy code, last resort: cython
        #  2: parallelize code (outer loop)

        num_eigenvectors = X.shape[1]

        if num_eigenvectors <= 1:
            raise ValueError(
                "There must be more than one eigenvector to compute the residual."
            )

        if self.strategy == "dim":
            if not (0 < self.intrinsic_dim < num_eigenvectors - 1):
                # num_eigenpairs-1 because the first eigenvector is trivial.
                raise ValueError(
                    f"intrinsic_dim has to be an integer larger than 1 and smaller than "
                    f"num_eigenpairs-1={num_eigenvectors}. Got intrinsic_dim"
                    f"={self.intrinsic_dim}"
                )
        elif self.strategy == "threshold":
            if not (0 < self.regress_threshold < 1):
                raise ValueError(
                    f"residual_threshold has to between [0, 1], exclusive. Got residual "
                    f"threshold={self.regress_threshold}"
                )
        else:
            raise ValueError(f"strategy={self.strategy} not known")

        if self.n_subsample is not None:
            eigvec = downsample(X, self.n_subsample)
        else:
            eigvec = X

        self.residuals_ = np.zeros(num_eigenvectors)
        self.residuals_[0] = np.nan  # the const (trivial) eigenvector is ignored
        # the first eigenvector is always taken, therefore receives a 1
        self.residuals_[1] = 1

        for i in range(2, num_eigenvectors):
            self.residuals_[i] = self._single_residual_local_regression(
                domain_eigenvectors=eigvec[:, 1:i], target_eigenvector=eigvec[:, i]
            )

        return self

    def transform(self, X, y=None):
        """Automatic parsimonious parametrization of the manifold by selecting appropriate
        residuals from local linear least squares fit.
        """

        # Strategy 1, according to user input:
        # User provides knowledge of the (expected) intrinsic dimension -- could also be
        # a result of estimation
        if self.strategy == "dim":
            if self.intrinsic_dim == 1:
                self.evec_indices_ = np.array([1])
            else:
                self.residuals_[
                    0
                ] = -1  # set the trivial cases invalid  # TODO: do this somewhere else?

                # NOTE: negative residuals to obtain the largest -- argpartition are the
                # respective indices
                self.evec_indices_ = np.argpartition(
                    -self.residuals_, self.intrinsic_dim
                )[: self.intrinsic_dim]

        # User provides a threshold for the residuals. All eigenfunctions above this value
        # are incldued to parametrize the manifold.
        elif self.strategy == "threshold":
            self.residuals_[0] = -1  # make trivial to invalid value
            self.evec_indices_ = np.where(self.residuals_ > self.regress_threshold)[0]
        else:
            raise ValueError(f"strategy={self.strategy} not known")

        return X[self.evec_indices_, :]  # transform the eigenvectors # TODO: #44
