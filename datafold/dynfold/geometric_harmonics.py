"""Geometric harmonics module.

This module implements out-of-sample evaluation of functions using the Geometric Harmonics method introduced in:

Coifman, R. R., & Lafon, S. (2006). Geometric harmonics: A novel tool for multiscale out-of-sample extension of
empirical functions. Applied and Computational Harmonic Analysis, 21(1), 31–52. DOI:10.1016/j.acha.2005.07.005
"""

__all__ = [
    "GeometricHarmonicsInterpolator",
    "GeometricHarmonicsFunctionBasis",
    "estimate_regression_error",
]

from typing import Optional

import numpy as np
from sklearn.base import MultiOutputMixin, RegressorMixin
from sklearn.metrics.regression import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y, check_array, check_consistent_length

import datafold.pcfold as pcfold
from datafold.dynfold.kernel import DmapKernelFixed, KernelMethod
from datafold.dynfold.utils import to_ndarray


class GeometricHarmonicsInterpolator(KernelMethod, RegressorMixin, MultiOutputMixin):
    def __init__(
        self,
        epsilon: float = 1.0,
        num_eigenpairs: int = 10,
        cut_off: float = np.inf,
        is_stochastic: bool = False,
        alpha: float = 1,
        symmetrize_kernel=True,  # NOTE for docu: if is_stochastic=False, then this is not really required
        use_cuda=False,
        dist_backend="guess_optimal",
        dist_backend_kwargs=None,
    ) -> None:

        """Geometric Harmonics Interpolator.

        """
        super(GeometricHarmonicsInterpolator, self).__init__(
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
            epsilon=self.epsilon,
            is_stochastic=self.is_stochastic,
            alpha=self.alpha,
            symmetrize_kernel=self.symmetrize_kernel,
        )

    def fit(self, X, y):

        X, y = self._check_X_y(X, y)

        self.X = pcfold.PCManifold(
            X,
            kernel=self.kernel_,
            cut_off=self.cut_off,
            dist_backend=self.dist_backend,
            **self.dist_backend_kwargs,
        )
        self.y = y

        self._kernel_matrix, self._basis_change_matrix = self.X.compute_kernel_matrix()
        self.eigenvalues_, self.eigenvectors_ = self.solve_eigenproblem(
            self._kernel_matrix, self._basis_change_matrix, self.use_cuda
        )

        self._precompute_aux()
        return self

    def score(self, X, y, sample_weight=None) -> float:

        X, y = self._check_X_y(X, y)
        y_pred = self(X)

        score: float = mean_squared_error(
            y, y_pred, sample_weight=None, multioutput="uniform_average"
        )
        return score

    def predict(self, X):
        return self(X)

    def _precompute_aux(self) -> None:
        # TODO: aux should get a better name!

        # Alternative/legacy way of computing self._aux
        # self._aux = (ev.T * (1. / ew)) @ (ev @ self.values)  # a little bit faster than legacy "n^3"
        # self._aux = ev.T @ np.diag(1. / ew) @ ev @ self.values # legacy "n^3"

        assert self.eigenvectors_ is not None and self.eigenvalues_ is not None

        # fast "n^2" complexity
        self._aux = (self.eigenvectors_.T * np.reciprocal(self.eigenvalues_)) @ (
            self.eigenvectors_ @ self.y
        )

    def _check_X_y(self, X: np.ndarray, y: np.ndarray = None):

        # TODO: need to check if X and y have same shape[1] dimensions to fitted data!

        check_consistent_length(X, y)

        kwargs = {
            "accept_sparse": False,
            "accept_large_sparse": False,
            "dtype": "numeric",
            "ensure_2d": True,
            "allow_nd": False,
            "ensure_min_samples": 1,
            "ensure_min_features": 1,
        }

        if y is None:
            return check_array(X, **kwargs)
        else:
            if y.ndim == 1:
                y = y[:, np.newaxis]

            kwargs["multi_output"] = True
            kwargs["y_numeric"] = True
            return check_X_y(X, y, **kwargs)

    def __call__(self, X: np.ndarray):
        """Evaluate interpolator at the given points.

        Parameters
        ----------
        X : np.ndarray
            Out-of-sample points to interpolate. The points are expected to lie on a manifold.
        """

        X = self._check_X_y(X)

        kernel_matrix, basis_change_matrix = self.X.compute_kernel_matrix(Y=X)

        # TODO: catch this case before computation:
        assert (
            basis_change_matrix is None
        )  # this is a cdist case, the symmetrize_kernel only works for the pdist case

        return np.squeeze(kernel_matrix @ self._aux)

    def gradient(self, X: np.ndarray, vcol: Optional[int] = None) -> np.ndarray:
        """Evaluate gradient of interpolator at the given points.

        # TODO: explain or link to where the gradient is computed (literature links). The code is not self explanatory.

        Parameters
        ----------
        X : np.ndarray
            Out-of-sample points to compute the gradient for. The points are expected to lie on the original manifold.
        vcol : Optional[int]
            The index of the corresponding function values (i.e. column in parameter `values` given to
            GeometricHarmonicsInterpolator) to compute the gradient. Has to be given for multivariate interpolation.

        Returns
        -------
        np.ndarray
            Gradients for each point (row-wise) for the requested points `xi`.
        """

        # TODO: generalize to all columns (if required...). Note that this will be a tensor then.

        X = self._check_X_y(X)

        assert self.X is not None and self.y is not None  # prevents mypy warnings

        if vcol is None and self.y.ndim > 1 and self.y.shape[1] > 1:
            raise NotImplementedError(
                "Currently vcol has to be provided to indicate for which values to get the "
                "gradient. Jacobi matrix is currently not supported."
            )

        if vcol is not None and not (0 <= vcol <= self.y.shape[1]):
            raise ValueError(
                f"vcol is not in the valid range between {0} and {self.y.shape[1]} (number of "
                f"columns in values). Got vcol={vcol}"
            )

        if vcol is not None:
            values = self.y[:, vcol]
        else:
            values = self.y[:, 0]

        kernel_matrix, basis_change_matrix = self.X.compute_kernel_matrix(X)
        assert basis_change_matrix is None  # TODO: catch this case before computing...

        # TODO: see issue #54 the to_ndarray() kills memory, when many points (xi.shape[0]) are requested
        kernel_matrix = to_ndarray(kernel_matrix)

        # Gradient computation
        ki_psis = kernel_matrix * values

        # NOTE: see also file misc/microbenchmark_gradient.py, using numexpr can squeeze out some computation speed for
        # large numbers of xi.shape[0]
        grad = np.zeros_like(X)
        v = np.empty_like(self.X)
        for p in range(X.shape[0]):
            np.subtract(X[p, :], self.X, out=v)
            np.matmul(v.T, ki_psis[p, :], out=grad[p, :])
        return grad


class GeometricHarmonicsFunctionBasis(GeometricHarmonicsInterpolator):
    # TODO: use (*args, **kwargs) and simply note that it is the same as in GeometricHarmonicsInterpolator?
    def __init__(
        self,
        epsilon: float = 1.0,
        num_eigenpairs: int = 11,
        cut_off: float = np.inf,
        is_stochastic: bool = False,
        alpha: float = 1,
        symmetrize_kernel=False,
        use_cuda=False,
        dist_backend="guess_optimal",
        dist_backend_kwargs=None,
    ):
        """
        See GeometricHarmonicsInterpolator.
        """
        super(GeometricHarmonicsFunctionBasis, self).__init__(
            epsilon,
            num_eigenpairs,
            cut_off,
            is_stochastic,
            alpha,
            use_cuda,
            symmetrize_kernel,
            dist_backend,
            dist_backend_kwargs,
        )

    def fit(self, X, y=None) -> "GeometricHarmonicsFunctionBasis":
        # y is only there to provide the same function from the base, but y is never needed...
        assert y is None

        X = self._check_X_y(X)

        self.X = pcfold.PCManifold(
            X,
            kernel=self.kernel_,
            cut_off=self.cut_off,
            dist_backend=self.dist_backend,
            **self.dist_backend_kwargs,
        )

        self._kernel_matrix, self._basis_change_matrix = self.X.compute_kernel_matrix()

        self.eigenvalues_, self.eigenvectors_ = self.solve_eigenproblem(
            self._kernel_matrix, self._basis_change_matrix, self.use_cuda
        )

        assert self.eigenvectors_ is not None
        self.y = self.eigenvectors_.T  # TODO: see #44
        self._precompute_aux()
        return self

    def score(self, X, y=None, sample_weight=None) -> float:
        if y is None:
            y = self.eigenvectors_.T  # target functions

        return super(GeometricHarmonicsFunctionBasis, self).score(X, y)


@DeprecationWarning
def estimate_regression_error(
    points: np.ndarray, values: np.ndarray, train_size: float, **gh_options
) -> float:
    # TODO: see issue #62
    """Compute error for a random split of the data.

    Parameters
    ----------
    points: np.ndarray
        Points in the domain of the interpolator.
    values: np.ndarray
        Scalars in the range of the interpolator.
    train_size: float = 0.7
        Ratio of points in the training dataset

    Returns
    -------
    Union[float, np.ndarray]
        Mean squared error of the test set.
    """

    # avoid warning by setting train_size and test_size (even though it it complementary)
    points_train, points_test, values_train, values_test = train_test_split(
        points,
        np.expand_dims(values, 1),
        train_size=train_size,
        test_size=1 - train_size,
    )

    gh_interpolator = GeometricHarmonicsInterpolator(**gh_options).fit(points, values)

    mean_sq_error = gh_interpolator.score(points_test, values_test)
    return mean_sq_error


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
