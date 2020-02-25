"""Geometric harmonics module.

This module implements out-of-sample evaluation of functions using the Geometric Harmonics
method introduced in:

Coifman, R. R., & Lafon, S. (2006). Geometric harmonics: A novel tool for multiscale
out-of-sample extension of empirical functions. Applied and Computational Harmonic
Analysis, 21(1), 31–52. DOI:10.1016/j.acha.2005.07.005
"""

import enum
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from sklearn.base import MultiOutputMixin, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_array, check_consistent_length, check_X_y
from sklearn.utils.validation import check_is_fitted, check_scalar

import datafold.pcfold as pcfold
from datafold.dynfold.kernel import DmapKernelFixed, KernelMethod
from datafold.pcfold.distance import compute_distance_matrix
from datafold.utils.maths import mat_dot_diagmat
from sklearn.base import BaseEstimator


class GeometricHarmonicsInterpolator(KernelMethod, RegressorMixin, MultiOutputMixin):
    def __init__(
        self,
        epsilon: float = 1.0,
        n_eigenpairs: int = 10,
        cut_off: float = np.inf,
        is_stochastic: bool = False,
        alpha: float = 1,
        # NOTE for docu: if is_stochastic=False, then this is not really required
        symmetrize_kernel=True,
        use_cuda=False,
        dist_backend="guess_optimal",
        dist_backend_kwargs=None,
    ) -> None:

        """Geometric Harmonics Interpolator.

        """
        super(GeometricHarmonicsInterpolator, self).__init__(
            epsilon=epsilon,
            n_eigenpairs=n_eigenpairs,
            cut_off=cut_off,
            is_stochastic=is_stochastic,
            alpha=alpha,
            symmetrize_kernel=symmetrize_kernel,
            use_cuda=use_cuda,
            dist_backend=dist_backend,
            dist_backend_kwargs=dist_backend_kwargs,
        )

    def _setup_kernel(self):
        self._kernel = DmapKernelFixed(
            epsilon=self.epsilon,
            is_stochastic=self.is_stochastic,
            alpha=self.alpha,
            symmetrize_kernel=self.symmetrize_kernel,
        )

    def _precompute_aux(self) -> None:
        # TODO: [style] "aux" should get a better name

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
    ) -> np.ndarray:

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
        _tags = super(GeometricHarmonicsInterpolator, self)._get_tags()
        _tags["multioutput"] = True
        return _tags

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Evaluate interpolator at the given points.

        Parameters
        ----------
        X : np.ndarray
            Out-of-sample points to interpolate. The points are expected to lie on the
            same manifold as the data used in the fit function.
        """

        check_is_fitted(
            self,
            attributes=[
                "_kernel",
                "X_",
                "_aux",
                "eigenvalues_",
                "eigenvectors_",
                "y_",
                "_row_sums_alpha",
            ],
        )

        X, _ = self._validate(X, ensure_min_samples=1)
        (
            kernel_matrix,
            _sanity_check_basis,
            _sanity_check_rowsamples,
        ) = self.X_.compute_kernel_matrix(Y=X, row_sums_alpha_fit=self._row_sums_alpha)

        assert (
            _sanity_check_basis is None and _sanity_check_rowsamples is None
        ), "cdist case, the symmetrize_kernel only works for the pdist case"

        return np.squeeze(kernel_matrix @ self._aux)

    def fit(self, X, y):

        X, y = self._validate(X, y=y, ensure_min_samples=2)
        self._setup_kernel()

        self.X_ = pcfold.PCManifold(
            X,
            kernel=self.kernel_,
            cut_off=self.cut_off,
            dist_backend=self.dist_backend,
            **(self.dist_backend_kwargs or {}),
        )

        self.y_ = y

        check_scalar(
            self.n_eigenpairs,
            "n_eigenpairs",
            target_type=(np.integer, int),
            min_val=1,
            max_val=self.X_.shape[0] - 1,
        )

        (
            self.kernel_matrix_,
            _basis_change_matrix,
            self._row_sums_alpha,
        ) = self.X_.compute_kernel_matrix()

        self.eigenvalues_, self.eigenvectors_ = self._solve_eigenproblem(
            self.kernel_matrix_, _basis_change_matrix, self.use_cuda
        )

        if self.kernel_.is_symmetric_transform(is_pdist=True):
            self.kernel_matrix_ = self._unsymmetric_kernel_matrix(
                kernel_matrix=self.kernel_matrix_,
                basis_change_matrix=_basis_change_matrix,
            )

        self._precompute_aux()
        return self

    def predict(self, X):
        return self(X)

    def gradient(self, X: np.ndarray, vcol: Optional[int] = None) -> np.ndarray:
        """Evaluate gradient of interpolator at the given points.

        # TODO: explain or link to where the gradient is computed (literature links).
            The code is not self explanatory.

        Parameters
        ----------
        X : np.ndarray
            Out-of-sample points to compute the gradient for. The points are expected
            to lie on the original manifold.
        vcol : Optional[int]
            The index of the corresponding function values (i.e. column in parameter
            `values` given to GeometricHarmonicsInterpolator) to compute the gradient.
            Has to be given for multivariate interpolation.

        Returns
        -------
        np.ndarray
            Gradients for each point (row-wise) for the requested points `xi`.
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

        kernel_matrix, basis_change_matrix, _ = self.X_.compute_kernel_matrix(
            X
        )  # TODO: _
        assert basis_change_matrix is None  # TODO: catch this case before computing...

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

    def score(self, X, y, sample_weight=None, multioutput="raw_values") -> float:

        X, y = self._validate(X, y=y, ensure_min_samples=1)
        y_pred = self(X)

        score = mean_squared_error(
            y, y_pred, sample_weight=None, multioutput=multioutput
        )

        if multioutput == "raw_values" and len(score) == 1:
            # in case score is np.array([score])
            score = score[0]

        # root mean squared error (NOTE: if upgrading scikit learn > 0.22 , the
        # mean_squared_error supports another input to get the RMSE
        return np.sqrt(score)


class MultiScaleGeometricHarmonicsInterpolator(GeometricHarmonicsInterpolator):
    # TODO: use (*args, **kwargs) and simply note that it is the same as in
    #  GeometricHarmonicsInterpolator?

    def __init__(
        self,
        initial_scale=1.0,
        n_eigenpairs: int = 11,
        condition=1.0,  # nu
        admissible_error=1.0,  # tau
        cut_off: float = np.inf,
        is_stochastic: bool = False,
        alpha: float = 1,
        symmetrize_kernel=False,
        use_cuda=False,
        dist_backend="guess_optimal",
        dist_backend_kwargs=None,
    ):
        """
        TODO: This is a work in progress algorithm.
         See: Chiavazzo et al. Reduced Models in Chemical Kinetics via Nonlinear
              Data-Mining
        """
        super(MultiScaleGeometricHarmonicsInterpolator, self).__init__(
            epsilon=-1,
            n_eigenpairs=n_eigenpairs,
            cut_off=cut_off,
            is_stochastic=is_stochastic,
            alpha=alpha,
            symmetrize_kernel=symmetrize_kernel,
            use_cuda=use_cuda,
            dist_backend=dist_backend,
            dist_backend_kwargs=dist_backend_kwargs,
        )

        self.condition = condition
        self.admissible_error = admissible_error
        self.initial_scale = initial_scale

    def _multi_scale_optimize(self, X, y):

        scale = self.initial_scale
        error_not_in_range = True

        from datafold.pcfold import PCManifold
        from datafold.pcfold.kernels import RadialBasisKernel
        from datafold.utils.maths import diagmat_dot_mat
        from datafold.utils.maths import sort_eigenpairs
        from scipy.sparse.linalg import eigsh

        mu_l_ = None
        phi_l_ = None

        while error_not_in_range:

            kernel = RadialBasisKernel(epsilon=scale)
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
        self, X: np.ndarray, y=None, **fit_params
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


class LaplacianPyramidsInterpolator(BaseEstimator, RegressorMixin):
    # TODO: LIST OF THINGS TO IMPROVE; TRY OUT OR TO DO
    #   1. allow sparse matrix (integrate cut_off etc.) --> problem: For large scales,
    #       there is almost no memory saving
    #   2. performance: currently computing the exp from the distance matrix is most
    #       expensive -- for the pdist, actually only half (upper triangle and
    #       diagonal) has to be computed -- this is not supported yet by the kernels.
    #       But before refactor, test the performance impact!

    # Internal Enum class to indicate the state of a function in the loop during fit
    class LoopCond(enum.Enum):
        NO_TERMINATION = 0
        BELOW_RES_TOL = 1
        LOOCV_INCREASES = 2
        TINY_RES = 3
        TERMINATED = 4

    def __init__(
        self,
        initial_epsilon=10.0,
        mu=2,
        residual_tol=None,
        auto_adaptive=False,
        alpha=0,
    ):

        self.initial_epsilon = initial_epsilon
        self.mu = mu
        self.residual_tol = residual_tol
        self.auto_adaptive = auto_adaptive
        self.alpha = alpha

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
        signal = self.LoopCond.NO_TERMINATION

        if self.auto_adaptive:
            if (
                not np.isnan(last_residual_norm)  # nan if self.level_ <= 1
                # stop if residual is  increasing
                and current_residual_norm > last_residual_norm
            ):
                signal = self.LoopCond.LOOCV_INCREASES

        if (
            signal == self.LoopCond.NO_TERMINATION
            and self.residual_tol is not None
            and current_residual_norm <= self.residual_tol
        ):
            signal = self.LoopCond.BELOW_RES_TOL

        MAGIC_TINY_RESIDUAL = 1e-15

        if (
            signal == self.LoopCond.NO_TERMINATION
            and current_residual_norm < MAGIC_TINY_RESIDUAL
        ):
            # Stop in any configuration, below this threshold
            signal = self.LoopCond.TINY_RES

        return signal

    def _terminate_condition_loop(self, current_residual_norm, last_residual_norm):

        assert current_residual_norm.ndim == 1
        assert current_residual_norm.shape == last_residual_norm.shape

        nr_tests = current_residual_norm.shape[0]
        signals = np.array([self.LoopCond.NO_TERMINATION] * nr_tests)

        for i in range(nr_tests):
            signals[i] = self._termination_rules_single(
                current_residual_norm[i], last_residual_norm[i]
            )

        return signals

    def _distance_matrix(self, Y=None):
        return compute_distance_matrix(
            X=self.X_,
            Y=Y,
            metric="sqeuclidean",  # for now only Gaussian kernel
            backend="brute",  # for now no support for sparse distance matrix
        )

    @property
    def level_(self):
        if self._level_tracker == {}:
            return 0
        else:
            return max(self._level_tracker.keys())

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
        bool_idx = loop_condition != self.LoopCond.LOOCV_INCREASES
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
        loop_condition = loop_condition[loop_condition != self.LoopCond.LOOCV_INCREASES]

        bool_idx = np.logical_or(
            loop_condition == self.LoopCond.BELOW_RES_TOL,
            loop_condition == self.LoopCond.TINY_RES,
        )

        return (
            active_func_indices[~bool_idx],
            current_residual[:, ~bool_idx],
            current_residual_norm[~bool_idx],
        )

    def _prepare_kernel_and_matrix(self, distance_matrix, epsilon):
        dmap_kernel = DmapKernelFixed(
            epsilon=epsilon,
            is_stochastic=True,
            alpha=self.alpha,
            symmetrize_kernel=False,
        )

        kernel_matrix, basis_change_matrix, row_sums_fit_alpha = dmap_kernel.eval(
            distance_matrix=distance_matrix, is_pdist=True
        )

        assert basis_change_matrix is None, "no symmetrize of kernel supported"

        if self.auto_adaptive:
            # inplace: set diagonal to zero to obtain LOOCV estimation
            # TODO: this requires special handling for a sparse kernel matrix
            np.fill_diagonal(kernel_matrix, 0)

        return dmap_kernel, kernel_matrix, row_sums_fit_alpha

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
        distance_matrix = self._distance_matrix()

        # set up variables for first iteration
        target_values = y
        epsilon = self.initial_epsilon
        current_residual_norm = np.array([np.nan] * self.nr_targets_)
        active_func_indices = np.arange(self.nr_targets_)

        # at least one iteration
        func_loop_conditions = np.array(
            [self.LoopCond.NO_TERMINATION] * self.nr_targets_
        )

        while self.LoopCond.NO_TERMINATION in func_loop_conditions:

            (
                dmap_kernel,
                kernel_matrix,
                row_sums_fit_alpha,
            ) = self._prepare_kernel_and_matrix(distance_matrix, epsilon)

            ### improve function approximation
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
            # because the in the current iteration the error is already increasing,
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

            if self.LoopCond.NO_TERMINATION in curr_loop_conditions:
                # prepare for next loop iteration
                epsilon = epsilon / self.mu
                current_residual_norm = current_residual_norm
                target_values = current_residual

        return self

    def fit(self, X: np.ndarray, y, **fit_params) -> "LaplacianPyramidsInterpolator":

        self.X_, y = self._validate(X, y, ensure_min_samples=2)
        self._setup()

        self.nr_targets_ = y.shape[1]
        self._laplacian_pyramid(self.X_, y)

        return self

    def predict(self, X):

        X, _ = self._validate(X)

        check_is_fitted(self, attributes=["X_", "nr_targets_", "_level_tracker",])

        # allocate memory for return
        y_hat = np.zeros([X.shape[0], self.nr_targets_])
        distance_matrix = self._distance_matrix(Y=X)

        for level, level_content in self._level_tracker.items():

            kernel_matrix, _, _ = level_content["kernel"].eval(
                distance_matrix, row_sums_alpha_fit=level_content["row_sums_fit_alpha"]
            )

            active_indices = level_content["active_indices"]
            y_hat[:, active_indices] += kernel_matrix @ level_content["target_values"]

        if self.nr_targets_ == 1:
            y_hat = y_hat.flatten()

        return y_hat

    # def score(self, X, y, sample_weight=None, multioutput="uniform_average") -> float:
    #     return mean_squared_error(
    #         self.predict(X),
    #         y_pred=y,
    #         sample_weight=sample_weight,
    #         squared=True,
    #         multioutput=multioutput,
    #     )

    def plot_eps_vs_residual(self):

        norm_residuals = np.zeros([self.level_ + 1, self.nr_targets_]) * np.nan

        for i, info in enumerate(self._level_tracker.values()):
            residuals = np.linalg.norm(info["target_values"], axis=0)
            norm_residuals[i, info["active_indices"]] = residuals

        epsilons = [r["kernel"].epsilon for r in self._level_tracker.values()]

        plt.figure()
        plt.plot(epsilons, norm_residuals, "-+")
        plt.xscale("log")
        plt.title(f"levels: {self.level_}")


if __name__ == "__main__":
    from sklearn.datasets import make_swiss_roll
    from sklearn.model_selection import GridSearchCV, ParameterGrid

    data, _ = make_swiss_roll(4000, random_state=1)
    func_values = np.random.rand(4000)

    from sklearn.base import BaseEstimator

    # print(GeometricHarmonicsFunctionBasis())
    # print(GeometricHarmonicsFunctionBasis().get_params())
    # print(isinstance(GeometricHarmonicsFunctionBasis(), BaseEstimator))
    # print(issubclass(GeometricHarmonicsFunctionBasis, BaseEstimator))

    pg = ParameterGrid({"epsilon": np.linspace(0.5, 1.5, 5)})
    grid_search = GridSearchCV(
        estimator=GeometricHarmonicsInterpolator(), param_grid=pg.param_grid, cv=3
    )
    gcv = grid_search.fit(data, func_values)

    print(gcv.best_score_)
    print(gcv.best_estimator_)
    print(gcv.cv_results_)
