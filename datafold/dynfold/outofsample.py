import enum
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_array, check_consistent_length, check_X_y
from sklearn.utils.validation import check_is_fitted, check_scalar

from datafold.decorators import warn_experimental_class, warn_known_bug
from datafold.dynfold.dmap import _DmapKernelAlgorithms
from datafold.pcfold import PCManifold
from datafold.pcfold.distance import compute_distance_matrix
from datafold.pcfold.kernels import DmapKernelFixed, GaussianKernel, PCManifoldKernel
from datafold.utils.general import mat_dot_diagmat


class GeometricHarmonicsInterpolator(BaseEstimator, RegressorMixin, MultiOutputMixin):
    """Interpolation of function values on high dimensional data with manifold assumption.

    Parameters
    ----------
    kernel
        Internal kernel to describe proximity between points. The kernel is passed
        as an `internal_kernel` to :class:`.DmapKernelFixed`, which describes
        the diffusion process.

    n_eigenpairs
        Number of eigenpairs to compute from kernel matrix.

    is_stochastic
        If True, the diffusion kernel matrix is normalized (row stochastic).

    alpha
        Re-normalization parameter. ``alpha=1`` corrects the sampling density in the
        data as an artifact of the collection process.

    symmetrize_kernel
        If True, a conjugate transformation is performed if the current settings would
        lead to a non-symmetric kernel matrix. This improves numerical stability when
        solving the eigenvectors of the kernel matrix as it allows algorithms designed
        for (sparse) Hermitian matrices to be used. If the kernel matrix is symmetric
        already (`is_stochastic=False`), then the parameter has no effect.

    dist_kwargs,
        Keyword arguments passed to the internal distance matrix computation. See
        :py:meth:`datafold.pcfold.compute_distance_matrix` for parameter arguments.

    Attributes
    ----------

    X_: PCManifold
        Training data during fit of shape `(n_samples, n_features)`. The data is required
        to be stored to perform out-of-sample interpolations. Equipped with kernel
        :py:class:`DmapKernelFixed`.

    y_: numpy.ndarray
        Target function values of shape `(n_samples, n_targets)`, can be multi-dimensional.

    eigenvalues_: numpy.ndarray
        Eigenvalues of diffusion kernel in decreasing order.

    eigenvectors_: numpy.ndarray
        Eigenvectors of the kernel matrix. Corresponds to the geometric harmonics.

    kernel_matrix_ : Union[numpy.ndarray, scipy.sparse.csr_matrix]
        Computed kernel matrix, stored if `store_kernel_matrix=True` during fit.

    References
    ----------

    :cite:`coifman_geometric_2006`

    See Also
    --------

    :class:`.LaplacianPyramidsInterpolator`
    """

    def __init__(
        self,
        kernel: PCManifoldKernel = GaussianKernel(epsilon=1.0),
        n_eigenpairs: int = 10,
        is_stochastic: bool = False,
        alpha: float = 1,
        symmetrize_kernel=True,
        dist_kwargs=None,
    ) -> None:

        self.kernel = kernel
        self.n_eigenpairs = n_eigenpairs
        self.is_stochastic = is_stochastic
        self.alpha = alpha
        self.symmetrize_kernel = symmetrize_kernel
        self.dist_kwargs = dist_kwargs

    def _precompute_aux(self) -> None:
        # TODO: [style, minor] "aux" should get a better name

        # Alternative/legacy way of computing self._aux
        # a little bit faster than  legacy "n^3"
        # self._aux = (ev.T * (1. / ew)) @ (ev @ self.values)
        # self._aux = ev.T @ np.diag(1. / ew) @ ev @ self.values # legacy "n^3"

        assert self.eigenvectors_ is not None and self.eigenvalues_ is not None

        # fast "n^2" complexity "AUX = EVEC @ 1/EVAL @ EVEC.T @ y"
        self._aux = mat_dot_diagmat(
            self.eigenvectors_, np.reciprocal(self.eigenvalues_)
        ) @ (self.eigenvectors_.T @ self.y_)

    def _validate(
        self, X: np.ndarray, y: np.ndarray = None, ensure_min_samples=1
    ) -> Union[np.ndarray, np.ndarray]:

        check_consistent_length(X, y)

        if isinstance(X, np.memmap):
            copy = True
        else:
            copy = False

        kwargs = {
            "accept_sparse": False,
            "copy": copy,
            "accept_large_sparse": False,
            "dtype": "numeric",
            "force_all_finite": True,
            "ensure_2d": True,
            "allow_nd": False,
            "ensure_min_samples": ensure_min_samples,
            "ensure_min_features": 1,
        }

        if y is None:
            X = check_array(X, **kwargs)
        else:
            if isinstance(y, np.ndarray) and y.ndim == 1:
                y = y[:, np.newaxis]

            kwargs["multi_output"] = True
            kwargs["y_numeric"] = True
            X, y = check_X_y(X, y, **kwargs)

        return X, y

    def _get_tags(self):
        # _tags = super(GeometricHarmonicsInterpolator, self)._get_tags()
        self._more_tags()["multioutput"] = True
        return super(GeometricHarmonicsInterpolator, self)._get_tags()

    def _setup_default_dist_kwargs(self):
        from copy import deepcopy

        self.dist_kwargs_ = deepcopy(self.dist_kwargs) or {}
        self.dist_kwargs_.setdefault("cut_off", np.inf)
        self.dist_kwargs_.setdefault("kmin", 0)
        self.dist_kwargs_.setdefault("backend", "guess_optimal")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Evaluate model for out-of-sample points.

        Parameters
        ----------
        X
            Points of shape `(n_samples, n_features)`.

        Returns
        -------
        numpy.ndarray
            The interpolated function values of shape `(n_samples, n_targets)`.
        """

        check_is_fitted(self)

        X, _ = self._validate(X, ensure_min_samples=1)

        kernel_output = self.X_.compute_kernel_matrix(Y=X, **self._cdist_kwargs)
        kernel_matrix_, _, _ = PCManifoldKernel.read_kernel_output(
            kernel_output=kernel_output
        )

        return np.squeeze(kernel_matrix_ @ self._aux)

    def fit(
        self, X: np.ndarray, y: np.ndarray, store_kernel_matrix: bool = False
    ) -> "GeometricHarmonicsInterpolator":
        """Fit model with training data.

        Parameters
        ----------
        X
            Training points of shape `(n_samples, n_features)`.

        y
            Target function values of shape `(n_samples, n_targets)`

        store_kernel_matrix
            If True, store the kernel matrix in attribute ``kernel_matrix_``.

        Returns
        -------
        GeometricHarmonicsInterpolator
            self
        """
        X, y = self._validate(X, y=y, ensure_min_samples=2)

        self._setup_default_dist_kwargs()

        self._dmap_kernel = DmapKernelFixed(
            internal_kernel=self.kernel,
            is_stochastic=self.is_stochastic,
            alpha=self.alpha,
            symmetrize_kernel=self.symmetrize_kernel,
        )

        self.X_ = PCManifold(
            X, kernel=self._dmap_kernel, dist_kwargs=self.dist_kwargs_,
        )
        self.y_ = y

        check_scalar(
            self.n_eigenpairs,
            "n_eigenpairs",
            target_type=(np.integer, int),
            min_val=1,
            max_val=self.X_.shape[0] - 1,
        )

        kernel_output = self.X_.compute_kernel_matrix()
        (
            kernel_matrix_,
            self._cdist_kwargs,
            ret_extra,
        ) = PCManifoldKernel.read_kernel_output(kernel_output=kernel_output)
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

        self._precompute_aux()

        if store_kernel_matrix:
            if self._dmap_kernel.is_symmetric_transform(is_pdist=True):
                self.kernel_matrix_ = _DmapKernelAlgorithms.unsymmetric_kernel_matrix(
                    kernel_matrix=kernel_matrix_,
                    basis_change_matrix=basis_change_matrix,
                )
            else:
                self.kernel_matrix_ = kernel_matrix_

        return self

    @warn_known_bug(gitlab_issue=16)
    def gradient(self, X: np.ndarray, vcol: Optional[int] = None) -> np.ndarray:
        """Evaluate gradient of interpolator at the given points.

        .. warning::
            This code is known to have a bug. (see gitlab issue #16).
            Contributions are welcome.

        Parameters
        ----------
        X
            Out-of-sample points to compute the gradient at.
            
        vcol
            Column index of the corresponding function value to compute the gradient of.
            Has to be given for multivariate interpolation.

        Returns
        -------
        np.ndarray
            Gradients (row-wise)
        """

        # TODO: generalize to all columns (if required...). Note that this will be a
        #  tensor then.

        X, _ = self._validate(X, ensure_min_samples=1)

        assert self.X_ is not None and self.y_ is not None  # prevents mypy warnings

        if vcol is None and self.y_.ndim > 1 and self.y_.shape[1] > 1:
            raise NotImplementedError(
                "Currently vcol has to be provided to indicate for which values to get "
                "the gradient. Jacobi matrix is currently not supported."
            )

        if vcol is not None and not (0 <= vcol <= self.y_.shape[1]):
            raise ValueError(
                f"vcol is not in the valid range between {0} and "
                f"{self.y_.shape[1]} (number of columns in values). Got vcol={vcol}"
            )

        if vcol is not None:
            values = self.y_[:, vcol]
        else:
            values = self.y_[:, 0]

        kernel_output = self.X_.compute_kernel_matrix(X)
        kernel_matrix, _, _sanity_check = PCManifoldKernel.read_kernel_output(
            kernel_output=kernel_output
        )

        assert _sanity_check == {}

        # TODO: see issue #54 the to_ndarray() kills memory, when many points
        #  (xi.shape[0]) are requested

        if isinstance(kernel_matrix, scipy.sparse.coo_matrix):
            kernel_matrix = np.squeeze(kernel_matrix.toarray())
        elif isinstance(kernel_matrix, scipy.sparse.csr_matrix):
            kernel_matrix = kernel_matrix.toarray()

        # Gradient computation
        ki_psis = kernel_matrix * values

        # NOTE: see also file misc/microbenchmark_gradient.py, using numexpr can squeeze
        # out some computation speed for large numbers of xi.shape[0]
        grad = np.zeros_like(X)
        v = np.empty_like(self.X_)
        for p in range(X.shape[0]):
            np.subtract(X[p, :], self.X_, out=v)
            np.matmul(v.T, ki_psis[p, :], out=grad[p, :])
        return grad

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        multioutput: str = "raw_values",
    ) -> float:
        """Score interpolation model with negative mean squared error metric.

        .. note::
            The mean squared error is negated to comply with "higher score is better"
            from scikit-learn.

        Parameters
        ----------
        X
            Data of shape `(n_samples, n_features)` to evaluate the model at.

        y
            True target values of shape `(n_samples, n_target_values)` to score
            predicted values against.

        sample_weight
            Sample weights of shape (n_samples,)

        multioutput
            String in `['raw_values', 'uniform_average']` to define aggregating of
            multiple output values, or ``numpy.ndarray`` of shape `(n_outputs,)`
            defines weights used to average errors.

            "raw_values":
                Returns a full set of errors in case of multioutput input.

            "uniform_average" :
                Errors of all outputs are averaged with uniform weight.

        Returns
        -------
        float
            score
        """
        X, y = self._validate(X, y=y, ensure_min_samples=1)
        y_pred = self.predict(X)

        score = mean_squared_error(
            y, y_pred, sample_weight=None, multioutput=multioutput
        )

        if multioutput == "raw_values" and len(score) == 1:
            # in case score is np.array([score])
            score = score[0]

        # root mean squared error (NOTE: if upgrading scikit learn > 0.22 , the
        # mean_squared_error supports another input to get the RMSE
        return -1 * np.sqrt(score)


@NotImplementedError
class MultiScaleGeometricHarmonicsInterpolator(GeometricHarmonicsInterpolator):
    """
    .. warning::
        This class is not documented and in experimental state. Contributions are welcome:
            * documentation
            * write unit tests
            * improve code
    """

    def __init__(
        self,
        initial_scale=1.0,
        n_eigenpairs: int = 11,
        condition=1.0,  # nu
        admissible_error=1.0,  # tau
        is_stochastic: bool = False,
        alpha: float = 1,
        symmetrize_kernel=False,
        dist_kwargs=None,
    ):
        """
        TODO: This is a work in progress algorithm.
         See: Chiavazzo et al. Reduced Models in Chemical Kinetics via Nonlinear
              Data-Mining
        """
        super(MultiScaleGeometricHarmonicsInterpolator, self).__init__(
            kernel=GaussianKernel(),
            n_eigenpairs=n_eigenpairs,
            is_stochastic=is_stochastic,
            alpha=alpha,
            symmetrize_kernel=symmetrize_kernel,
            dist_kwargs=dist_kwargs,
        )

        self.condition = condition
        self.admissible_error = admissible_error
        self.initial_scale = initial_scale

    def _multi_scale_optimize(self, X, y):

        scale = self.initial_scale
        error_not_in_range = True

        from datafold.pcfold import PCManifold
        from datafold.pcfold.kernels import GaussianKernel
        from datafold.utils.general import diagmat_dot_mat, sort_eigenpairs
        from scipy.sparse.linalg import eigsh

        mu_l_ = None
        phi_l_ = None

        while error_not_in_range:

            kernel = GaussianKernel(epsilon=scale)
            X = PCManifold(X, kernel)

            kernel_matrix = X.compute_kernel_matrix()

            mu, phi = eigsh(kernel_matrix, k=300, which="LM")
            mu, phi = np.real(mu), np.real(phi)
            mu, phi = sort_eigenpairs(mu, phi)

            # use orthogonality to solve phi c = y
            coeff_ = phi.T @ y

            ratio_eigenvalues = mu[0] / mu
            max_l = np.argmax(~(ratio_eigenvalues < self.condition))

            coeff_l_, coeff_l_err_ = (coeff_[:max_l], coeff_[max_l:])

            mu_l_ = mu[:max_l]
            phi_l_ = phi[:, :max_l]

            error = np.sqrt(np.sum(np.abs(coeff_l_err_) ** 2))

            print(f"scale={scale}  --  error={error} -- max_l={max_l}")

            if error <= self.admissible_error:
                error_not_in_range = False
            else:
                scale = scale / 2

        self.X = X
        self.y = y

        self._aux = phi_l_ @ diagmat_dot_mat(np.reciprocal(mu_l_), phi_l_.T) @ y

    def __call__(self, X):
        X, _ = self._validate(X=X, y=None, ensure_min_samples=1)

        kernel_matrix = self.X.compute_kernel_matrix(
            Y=X, row_sums_alpha_fit=self._row_sums_alpha
        )

        return np.squeeze(kernel_matrix @ self._aux)

    def fit(
        self, X: np.ndarray, y=None, store_kernel_matrix=False, **fit_params
    ) -> "MultiScaleGeometricHarmonicsInterpolator":
        X, y = self._validate(X, y=y, ensure_min_samples=2)

        self._multi_scale_optimize(X, y)
        # self._precompute_aux()

        return self

    def score(self, X, y, sample_weight=None, multioutput="uniform_average") -> float:

        if y is None:
            y = self.eigenvectors_  # target functions

        return super(MultiScaleGeometricHarmonicsInterpolator, self).score(
            X=X, y=y, sample_weight=sample_weight, multioutput=multioutput
        )


class LaplacianPyramidsInterpolator(BaseEstimator, RegressorMixin, MultiOutputMixin):
    """Laplacian pyramids interpolation of function values on data manifold using
    kernels with different scales.

    The model implementation is generalized to vector valued targets: the kernel scales
    are decreased (i.e. a new kernel with lower scale is computed) until for each target
    function the corresponding stopping criteria is reached (based on residual).

    Parameters
    ----------

    initial_epsilon
        Scale of kernel in first iteration.

    mu
        Factor by which epsilon is decreased in every iteration
        :code:`(new_epsilon = old_epsilon / mu)`. Must be strictly larger than 1.

    residual_tol
        Decreasing kernel scale terminates if interpolation residual gets
        smaller than tolerance. If ``auto_adaptive=False`` a parameter must be provided.

    auto_adaptive
        If True, decreasing the kernel scale terminates based on LOOCV (leave
        one out cross validation) estimation in each iteration.

    alpha
        Parameter handled to the diffusion maps kernel used (see
        :class:`DmapKernelFixed`).

    Attributes
    ----------

    X_: numpy.ndarray
        Point cloud during fit.

    level_: int
        The number of levels and kernels used in the model.

    n_targets_: int
        The number of target functions during fit. (Note: the target values are not hold
        in the model).

    References
    ----------

    :cite:`fernandez_auto-adaptative_2014`
    :cite:`rabin_heterogeneous_2012`

    """

    # Internal Enum class to indicate the state of a function in the loop during fit
    class _LoopCond(enum.Enum):
        NO_TERMINATION = 0
        BELOW_RES_TOL = 1
        LOOCV_INCREASES = 2
        TINY_RES = 3
        TERMINATED = 4

    def __init__(
        self,
        initial_epsilon: float = 10.0,
        mu: float = 2.0,
        residual_tol: Optional[float] = None,
        auto_adaptive: bool = False,
        alpha: float = 0,
    ):

        self.initial_epsilon = initial_epsilon
        self.mu = mu
        self.residual_tol = residual_tol
        self.auto_adaptive = auto_adaptive
        self.alpha = alpha

    @property
    def level_(self):
        return 0 if self._level_tracker == {} else max(self._level_tracker.keys())

    def _validate(self, X, y=None, ensure_min_samples=1):

        if self.residual_tol is None and not self.auto_adaptive:
            raise ValueError(
                "Need to specify a stopping criteria by either providing a "
                "residual tolerance or auto_adaptive=True"
            )

        if self.residual_tol is not None:
            check_scalar(
                self.residual_tol,
                name="residual_tol",
                target_type=(float, np.floating),
                min_val=0,
                max_val=np.inf,
            )

        check_scalar(
            self.mu,
            "mu",
            target_type=(int, np.integer, float, np.floating),
            min_val=1 + np.finfo(float).eps,
            max_val=np.inf,
        )

        if isinstance(X, np.memmap):
            copy = True
        else:
            copy = False

        kwargs = {
            "accept_sparse": False,
            "copy": copy,
            "force_all_finite": True,
            "accept_large_sparse": False,
            "dtype": "numeric",
            "ensure_2d": True,
            "allow_nd": False,
            "ensure_min_samples": ensure_min_samples,
            "ensure_min_features": 1,
        }

        if y is not None:
            kwargs["multi_output"] = True
            kwargs["y_numeric"] = True
            X, y = check_X_y(X, y, **kwargs)

            if y.ndim == 1:
                y = y[:, np.newaxis]

        else:
            X = check_array(X, **kwargs)

        return X, y

    def _setup(self):
        self._level_tracker = dict()

    def _get_tags(self):
        _tags = super(LaplacianPyramidsInterpolator, self)._get_tags()
        _tags["multioutput"] = True
        return _tags

    def _termination_rules_single(self, current_residual_norm, last_residual_norm):
        signal = self._LoopCond.NO_TERMINATION

        if self.auto_adaptive:
            if (
                not np.isnan(last_residual_norm)  # nan if self.level_ <= 1
                # stop if residual is  increasing
                and current_residual_norm > last_residual_norm
            ):
                signal = self._LoopCond.LOOCV_INCREASES

        if (
            signal == self._LoopCond.NO_TERMINATION
            and self.residual_tol is not None
            and current_residual_norm <= self.residual_tol
        ):
            signal = self._LoopCond.BELOW_RES_TOL

        MAGIC_TINY_RESIDUAL = 1e-15

        if (
            signal == self._LoopCond.NO_TERMINATION
            and current_residual_norm < MAGIC_TINY_RESIDUAL
        ):
            # Stop in any configuration, below this threshold
            signal = self._LoopCond.TINY_RES

        return signal

    def _terminate_condition_loop(self, current_residual_norm, last_residual_norm):

        assert current_residual_norm.ndim == 1
        assert current_residual_norm.shape == last_residual_norm.shape

        nr_tests = current_residual_norm.shape[0]
        signals = np.array([self._LoopCond.NO_TERMINATION] * nr_tests)

        for i in range(nr_tests):
            signals[i] = self._termination_rules_single(
                current_residual_norm[i], last_residual_norm[i]
            )

        return signals

    def _distance_matrix(self, X, Y=None):
        return compute_distance_matrix(
            X=X,
            Y=Y,
            metric="sqeuclidean",  # for now only Gaussian kernel
            backend="brute",  # for now no support for sparse distance matrix
        )

    def _get_next_level_(self):
        if self._level_tracker == {}:
            return 0
        else:
            return max(self._level_tracker.keys()) + 1

    def _track_new_level(
        self,
        kernel,
        target_values,
        residual_norm,
        active_func_indices,
        row_sums_fit_alpha,
    ):
        new_level = self._get_next_level_()

        assert new_level not in self._level_tracker.keys()
        assert active_func_indices.shape[0] == target_values.shape[1]

        self._level_tracker[new_level] = {
            "kernel": kernel,
            "target_values": target_values,
            "residual_norm": residual_norm,
            "active_indices": active_func_indices,
            "row_sums_fit_alpha": row_sums_fit_alpha,
        }

    def _remove_increase_loocv_indices(
        self,
        active_func_indices,
        loop_condition,
        target_values,
        current_residual,
        current_residual_norm,
    ):
        bool_idx = loop_condition != self._LoopCond.LOOCV_INCREASES
        return (
            active_func_indices[bool_idx],
            target_values[:, bool_idx],
            current_residual[:, bool_idx],
            current_residual_norm[bool_idx],
        )

    def _remove_residual_based_indices(
        self,
        active_func_indices,
        loop_condition,
        current_residual,
        current_residual_norm,
    ):

        # remove LOOCV-termination reason as it is handeled separately
        loop_condition = loop_condition[
            loop_condition != self._LoopCond.LOOCV_INCREASES
        ]

        bool_idx = np.logical_or(
            loop_condition == self._LoopCond.BELOW_RES_TOL,
            loop_condition == self._LoopCond.TINY_RES,
        )

        return (
            active_func_indices[~bool_idx],
            current_residual[:, ~bool_idx],
            current_residual_norm[~bool_idx],
        )

    def _prepare_kernel_and_matrix(self, distance_matrix, epsilon):
        dmap_kernel = DmapKernelFixed(
            internal_kernel=GaussianKernel(epsilon),
            is_stochastic=True,
            alpha=self.alpha,
            symmetrize_kernel=False,
        )

        kernel_output = dmap_kernel.eval(distance_matrix=distance_matrix, is_pdist=True)
        kernel_matrix, cdist_kwargs, _check = PCManifoldKernel.read_kernel_output(
            kernel_output=kernel_output
        )

        row_sums_alpha_fit = cdist_kwargs["row_sums_alpha_fit"]

        assert (
            _check["basis_change_matrix"] is None
        ), "no symmetrize of kernel supported"

        if self.auto_adaptive:
            # inplace: set diagonal to zero to obtain LOOCV estimation
            # TODO: this requires special handling for a sparse kernel matrix
            np.fill_diagonal(kernel_matrix, 0)

        return dmap_kernel, kernel_matrix, row_sums_alpha_fit

    def _compute_residual(
        self, y, func_approx, active_func_indices, current_residual_norm
    ):
        current_residual = np.subtract(
            y[:, active_func_indices], func_approx[:, active_func_indices]
        )

        last_residual_norm, current_residual_norm = (
            current_residual_norm,
            np.linalg.norm(current_residual, axis=0),
        )

        return current_residual, current_residual_norm, last_residual_norm

    def _laplacian_pyramid(self, X, y):

        func_approx = np.zeros_like(y, dtype=np.float)

        # compute once and only apply eval
        distance_matrix = self._distance_matrix(X)

        # set up variables for first iteration
        target_values = y
        epsilon = self.initial_epsilon
        current_residual_norm = np.array([np.nan] * self.n_targets_)
        active_func_indices = np.arange(self.n_targets_)

        # at least one iteration
        func_loop_conditions = np.array(
            [self._LoopCond.NO_TERMINATION] * self.n_targets_
        )

        while self._LoopCond.NO_TERMINATION in func_loop_conditions:

            (
                dmap_kernel,
                kernel_matrix,
                row_sums_fit_alpha,
            ) = self._prepare_kernel_and_matrix(distance_matrix, epsilon)

            # improve function approximation
            func_approx[:, active_func_indices] += kernel_matrix @ target_values

            (
                current_residual,
                current_residual_norm,
                last_residual_norm,
            ) = self._compute_residual(
                y=y,
                func_approx=func_approx,
                active_func_indices=active_func_indices,
                current_residual_norm=current_residual_norm,
            )

            curr_loop_conditions = self._terminate_condition_loop(
                current_residual_norm=current_residual_norm,
                last_residual_norm=last_residual_norm,
            )
            func_loop_conditions[active_func_indices] = curr_loop_conditions

            # remove function indices that terminate due to increasing LOOCV estimate
            # (auto-adaptive) first (i.e. before setting the new level). This is
            # because in the current iteration the error is already increasing,
            # so it shouldn't be used for the function evaluation.
            (
                active_func_indices,
                target_values,
                current_residual,
                current_residual_norm,
            ) = self._remove_increase_loocv_indices(
                active_func_indices,
                curr_loop_conditions,
                target_values,
                current_residual,
                current_residual_norm,
            )

            if len(active_func_indices) > 0:
                # insert all information about this run
                # (mainly required for function evaluation in __call__ !)
                self._track_new_level(
                    kernel=dmap_kernel,
                    target_values=target_values,
                    residual_norm=current_residual_norm,
                    active_func_indices=active_func_indices,
                    row_sums_fit_alpha=row_sums_fit_alpha,
                )

                # Remove all indices now that fulfill the residual criteria and do not
                # need another loop.
                (
                    active_func_indices,
                    current_residual,
                    current_residual_norm,
                ) = self._remove_residual_based_indices(
                    active_func_indices,
                    curr_loop_conditions,
                    current_residual,
                    current_residual_norm,
                )

            if self._LoopCond.NO_TERMINATION in curr_loop_conditions:
                # prepare for next loop iteration
                epsilon = epsilon / self.mu
                current_residual_norm = current_residual_norm
                target_values = current_residual

        return self

    def fit(
        self, X: np.ndarray, y: np.ndarray, **fit_params
    ) -> "LaplacianPyramidsInterpolator":
        """Train model by decreasing kernel scales until termination.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Training point cloud.
        y: numpy.ndarray of shape (n_samples, n_target_values)
            Target function values.

        Returns
        -------
        LaplacianPyramidsInterpolator
            self
        """
        self.X_, y = self._validate(X, y, ensure_min_samples=2)
        self._setup()

        self.n_targets_ = y.shape[1]
        self._laplacian_pyramid(self.X_, y)

        return self

    def predict(self, X: np.ndarray):
        """Out-of-sample point evaluation.

        Parameters
        ----------
        X: numpy.ndarray
            Out-of-sample data of shape `(n_samples, n_features)`.

        Returns
        -------
        numpy.ndarray
            Predicted function values of shape `(n_samples, n_targets_)`.
        """

        X, _ = self._validate(X)

        check_is_fitted(self)

        # allocate memory for return
        y_hat = np.zeros([X.shape[0], self.n_targets_])
        distance_matrix = self._distance_matrix(X=self.X_, Y=X)

        for level, level_content in self._level_tracker.items():

            kernel_matrix, _, _ = level_content["kernel"].eval(
                distance_matrix, row_sums_alpha_fit=level_content["row_sums_fit_alpha"]
            )

            active_indices = level_content["active_indices"]
            y_hat[:, active_indices] += kernel_matrix @ level_content["target_values"]

        if self.n_targets_ == 1:
            y_hat = y_hat.flatten()

        return y_hat

    def plot_eps_vs_residual(self) -> None:
        """Plot residuals versus kernel scales (epsilon) from model fit.
        """

        check_is_fitted(self)

        norm_residuals = np.zeros([self.level_ + 1, self.n_targets_]) * np.nan

        for i, info in enumerate(self._level_tracker.values()):
            residuals = np.linalg.norm(info["target_values"], axis=0)
            norm_residuals[i, info["active_indices"]] = residuals

        epsilons = [
            r["kernel"].internal_kernel.epsilon for r in self._level_tracker.values()
        ]

        plt.figure()
        plt.plot(epsilons, norm_residuals, "-+")
        plt.xscale("log")
        plt.title(f"levels: {self.level_}")
