import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
import pydmd
import scipy.linalg
import scipy.sparse
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, Ridge, ridge_regression
from sklearn.utils.validation import check_is_fitted

from datafold.decorators import warn_experimental_class
from datafold.dynfold.base import PRE_FIT_TYPES, PRE_IC_TYPES, TSCPredictMixIn
from datafold.dynfold.system_evolution import LinearDynamicalSystem
from datafold.pcfold.timeseries import TSCDataFrame
from datafold.utils.datastructure import if1dim_rowvec
from datafold.utils.maths import diagmat_dot_mat, mat_dot_diagmat, sort_eigenpairs

try:
    import pydmd
except ImportError:
    pydmd = None
    IS_IMPORTED_PYDMD = False
else:
    IS_IMPORTED_PYDMD = True


class DMDBase(BaseEstimator, TSCPredictMixIn):
    r"""Dynamic Mode Decomposition (DMD) approximates the Koopman operator with
    a matrix :math:`K`.

    The Koopman matrix :math:`K` defines a linear dynamical system of the form

    .. math::
        K x_k = x_{k+1} \\
        K^k x_0 = x_k

    where :math:`x_k` is the (column) state vector of the system at time :math:`k`. All
    subclasses should provide the right eigenvectors :math:`\Psi_r` and corresponding
    eigenvalues :math:`\omega` of the Koopman matrix to efficiently evolve the linear
    system TODO: link to method

    Evolving the linear system over many time steps is expensive due to the matrix power:

    .. math::
        K^k x_0 &= x_k

    Therefore, the (right) eigenpairs (:math:`\omega, \Psi_r`) of the Koopman matrix is
    computed and the initial condition is written in terms of the eigenvectors in a
    least-squares sense.

    .. math::
        K^k \Psi_r &= \Psi_r \Omega \\
        \Psi_r b &= x_0 \\
        K^k \Psi_r b &= x_k \\
        \Psi_r \Omega^k b &= x_k

    where the eigenproblem is stated in matrix form for all computed eigenpairs. Because
    :math:`\Omega` is a a diagonal matrix the power is very cheap compared to the
    generally full matrix :math:`K^k`.
    """

    def _evolve_dmd_system(
        self,
        X_ic: pd.DataFrame,
        time_values: np.ndarray,
        time_invariant=True,
        post_map: Optional[np.ndarray] = None,
        qoi_columns=None,
    ):
        """
        Evolve the linear system.

        Parameters
        ----------
        ic
            Initial condition in same space where EDMD was fit. The initial condition may
            be transformed internally.
        time_values
            Array of times where the dynamical system should be evaluated.
        dynmatrix
            If not provided, the dynmatrix corresponds to the eigenvectors of EDMD.
        time_series_ids
            Time series ids in same order to initial conditions.
        qoi_columns
            List of quantity of interest names that are set in the TSCDataFrame
            returned.

        Returns
        -------
        TSCDataFrame
            The resulting time series collection for each initial condition collected.

        """

        # type hints for mypy
        self.eigenvectors_left_: Optional[np.ndarray]
        self.eigenvectors_right_: np.ndarray
        self.eigenvalues_: np.ndarray

        if qoi_columns is None:
            qoi_columns = self.features_in_[1]

        # initial condition is numpy array only, from now on
        ic = X_ic.to_numpy().T
        time_series_ids = X_ic.index.get_level_values("ID").to_numpy()

        if len(np.unique(time_series_ids)) != len(time_series_ids):
            # check if duplicate ids are present
            raise ValueError("time series ids have to be unique")

        # Choose alternative of how to evolve the linear system:
        if hasattr(self, "eigenvectors_left_") and (
            self.eigenvectors_left_ is not None and self.eigenvectors_right_ is not None
        ):
            # uses both eigenvectors (left and right). Used if is_diagonalize=True in
            # DMDFull
            ic = self.eigenvectors_left_ @ ic
        elif (
            hasattr(self, "eigenvectors_right_")
            and self.eigenvectors_right_ is not None
        ):
            # represent the initial condition in terms of right eigenvectors (by solving a
            # least-squares problem) -- only the right eigenvectors are required
            ic = np.linalg.lstsq(self.eigenvectors_right_, ic, rcond=1e-15)[0]

        else:
            raise NotFittedError(
                "DMD is not properly fit. "
                "Missing attributes: eigenvectors_left_ / eigenvectors_right_ "
            )

        if time_invariant:
            shift = np.min(time_values)
        else:
            # If the dmd time is shifted during data (e.g. the minimum processed data
            # starts with time=5, some positive value) then normalize the time_samples
            # with this shift. The linear system handles the shifted time start as time
            # zero.
            shift = self.time_interval_[0]

        norm_time_samples = time_values - shift

        tsc_df = LinearDynamicalSystem(
            mode="continuous", time_invariant=True
        ).evolve_system_spectrum(
            eigenvectors=self.eigenvectors_right_,
            eigenvalues=self.eigenvalues_,
            dt=self.dt_,
            ic=ic,
            post_map=post_map,
            time_values=norm_time_samples,
            time_series_ids=time_series_ids,
            qoi_columns=qoi_columns,
        )

        # correct the time shift again to return the correct time according to the
        # training data (not necessarily "normed time steps" [0, 1, 2, ...]
        # One way is to shift the time again, i.e.
        #
        #    tsc_df.tsc.shift_time(shift_t=shift)
        #
        # However, this can sometimes introduce numerical noise (forward/backwards
        # shifting), therefore the user-requested `time_values` set directly into the
        # index. This way it matches for all time series.
        #
        # Because hard-setting the time indices can introduce problems, the following
        # assert makes sure that both ways match (up to numerical differences).
        assert (
            tsc_df.tsc.shift_time(shift_t=shift).time_values(unique_values=True)
            - time_values
            < 1e-15
        ).all()

        # Hard set of time_values
        tsc_df.index = tsc_df.index.set_levels(
            time_values, level=1
        ).remove_unused_levels()

        return tsc_df

    def _convert_array2frame(self, X):
        if isinstance(X, np.ndarray):
            assert X.ndim == 2
            nr_ic = X.shape[0]
            index = pd.Index(data=np.arange(nr_ic), name="ID")
            X = pd.DataFrame(X, index=index, columns=self.features_in_[1])

        return X

    def fit(self, X: PRE_FIT_TYPES, **fit_params):
        raise NotImplementedError("base class")

    def predict(self, X: PRE_IC_TYPES, time_values=None, **predict_params):
        check_is_fitted(self)

        X = self._convert_array2frame(X)
        self._validate_data(X)

        X, time_values = self._validate_features_and_time_values(
            X=X, time_values=time_values
        )

        # This is for compatibility with the koopman based surrogate model
        post_map = predict_params.pop("post_map", None)
        qoi_columns = predict_params.pop("qoi_columns", None)
        if len(predict_params.keys()) > 0:
            raise KeyError(f"predict_params are invalid: {predict_params.keys()}")

        return self._evolve_dmd_system(
            X_ic=X, time_values=time_values, post_map=post_map, qoi_columns=qoi_columns
        )

    def reconstruct(self, X: TSCDataFrame):
        check_is_fitted(self)

        self._validate_data(X)
        self._validate_feature_names(X)

        X_latent_ts_folds = []
        for X_latent_ic, time_values in X.tsc.initial_states_folds():
            current_ts = self.predict(X=X_latent_ic, time_values=time_values)
            X_latent_ts_folds.append(current_ts)

        X_est_ts = pd.concat(X_latent_ts_folds, axis=0)
        return X_est_ts

    def fit_reconstruct(self, X: TSCDataFrame, **fit_params):
        return self.fit(X, **fit_params).reconstruct(X)

    def score(self, X: TSCDataFrame, y=None, sample_weight=None):
        self._check_attributes_set_up(check_attributes=["score_eval"])
        assert y is None

        X_est_ts = self.reconstruct(X)

        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight

        return self.score_eval(X, X_est_ts, sample_weight)


class DMDFull(DMDBase):
    r"""Full (i.e. using entire data matrix) EDMD method.

    The Koopman matrix is approximated

    .. math::
        K X &= X' \\
        K &= X' X^{\dagger},

    where :math:`\dagger` defines the Moore–Penrose inverse.

    """

    def __init__(self, is_diagonalize: bool = False):
        self._setup_default_tsc_scorer_and_metric()
        self.is_diagonalize = is_diagonalize

    def _diagonalize_left_eigenvectors(self):
        """Compute right eigenvectors (not normed) such that
        Koopman matrix = right_eigenvectors @ diag(eigenvalues) @ left_eigenvectors .
        """

        # lhs_matrix = (np.diag(self.eigenvalues_) @ self.eigenvectors_right_)
        lhs_matrix = self.eigenvectors_right_ * self.eigenvalues_

        # NOTE: the left eigenvectors are not normed (i.e. ||ev|| != 1
        self.eigenvectors_left_ = np.linalg.solve(lhs_matrix, self.koopman_matrix_)

    def _compute_koopman_matrix(self, X: TSCDataFrame):

        # It is more suitable to get the shift_start and end in row orientation as this
        # is closer to the normal least squares parameter definition
        shift_start_transposed, shift_end_transposed = X.tsc.shift_matrices(
            snapshot_orientation="row"
        )

        # The easier to read version is:
        # koopman_matrix shift_start_transposed = shift_end_transposed
        # koopman_matrix.T = np.linalg.lstsq(shift_start_transposed,
        # shift_end_transposed, rcond=1E-14)[0]
        #
        # However, it is much more efficient to multiply shift_start from right
        # K^T (shift_start^T * shift_start) = (shift_end^T * shift_start)
        # K^T G = G'
        # This is because (shift_start^T * shift_start) is a smaller matrix and faster
        # to solve. For further info, see Williams et al. Extended DMD and DMD book,
        # Kutz et al. (book page 168).

        if shift_start_transposed.shape[1] > shift_start_transposed.shape[0]:

            warnings.warn(
                "There are more observables than snapshots. The current implementation "
                "favors more snapshots than obserables. This may result in a bad "
                "computational performance."
            )

        G = shift_start_transposed.T @ shift_start_transposed
        G_dash = shift_start_transposed.T @ shift_end_transposed

        # If a is square and of full rank, then x (but for round-off error) is the
        # “exact” solution of the equation.
        koopman_matrix, residual, rank, _ = np.linalg.lstsq(G, G_dash, rcond=1e-14)

        if rank != G.shape[1]:
            warnings.warn(
                f"Shift matrix (shape={G.shape}) has not full rank (={rank}), falling "
                f"back to least squares solution. The sum of residuals is: "
                f"{np.sum(residual)}"
            )

        # # TODO: Experimental (test other solvers, with more functionality)
        # #  ridge_regression, and sparisty promoting least squares solutions could be
        #    included here
        # # TODO: clarify if the ridge regression should be done better on lstsq with
        #     shift matrices (instead of the G, G_dash)

        # #  also possible to integrate "RidgeCV" which allows to select the best
        # #  alpha from a list
        # #  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV

        # TODO: fit_intercept option useful to integrate?
        # koopman_matrix = ridge_regression(
        #     G, G_dash, alpha=self.alpha, verbose=0, return_intercept=False
        # )
        # koopman_matrix = koopman_matrix.T
        # # TODO: END Experimental

        # koopman_matrix = (
        #     LinearRegression(fit_intercept=False, normalize=False).fit(G, G_dash).coef_
        # )

        # The reason why it is transposed:
        # K * G_k = G_{k+1}
        # (G_k)^T * K = G_{k+1}^T  (therefore the row snapshot orientation at the
        #                           beginning)

        koopman_matrix = koopman_matrix.T
        return koopman_matrix

    def fit(self, X: PRE_FIT_TYPES, y=None, **fit_params):

        self._validate_data(
            X=X,
            ensure_feature_name_type=True,
            validate_tsc_kwargs={"ensure_const_delta_time": True},
        )
        self._setup_features_and_time_fit(X=X)

        self.koopman_matrix_ = self._compute_koopman_matrix(X)
        self.eigenvalues_, self.eigenvectors_right_ = sort_eigenpairs(
            *np.linalg.eig(self.koopman_matrix_)
        )

        if self.is_diagonalize:
            self._diagonalize_left_eigenvectors()

        return self


class DMDEco(DMDBase):
    r"""Approximates the Koopman matrix economically (compared to EDMDFull). It computes
    the principal components of the input data and computes the Koopman matrix there.
    Finally, it reconstructs the eigenpairs of the original system.

    1. Compute the singular value decomposition of the data and use the leading `k`
    singular values and corresponding vectors in :math:`U` and :math:`V`.

    .. math::
        X \approx U \Sigma V^*

    2. Compute the Koopman matrix on the PCA coordinates:

    .. math::
        K = U^T X' V \Sigma^{-1}

    3. Compute the eigenpairs (stated in matrix form):

    .. math::
        K W_r = W_r \Omega

    4. Reconstruct the (exact) eigendecomposition of :math:`K`

    .. math::
        \Psi_r = X' V \Sigma^{-1} W

    .. note::
        The eigenvectors in step 4 can also be computed with :math:`\Psi_r = U W`, which
        is then referred to the projected reconstruction.

    .. todo::
        cite Tu, 2014
    """

    def __init__(self, svd_rank=10):
        self._setup_default_tsc_scorer_and_metric()
        self.k = svd_rank

    def _compute_internals(self, X: TSCDataFrame):
        # TODO: different orientations are good for different cases:
        #  1 more snapshots than quantities
        #  2 more quantities than snapshots
        #  Currently it is optimized for the case 2.

        shift_start, shift_end = X.tsc.shift_matrices(snapshot_orientation="column")
        U, S, Vh = np.linalg.svd(shift_start, full_matrices=False)  # (1.18)

        U = U[:, : self.k]
        S = S[: self.k]
        S_inverse = np.reciprocal(S, out=S)

        V = Vh.T
        V = V[:, : self.k]

        koopman_matrix_low_rank = (
            U.T @ shift_end @ mat_dot_diagmat(V, S_inverse)
        )  # (1.20)

        self.eigenvalues_, eigenvector = np.linalg.eig(
            koopman_matrix_low_rank
        )  # (1.22)

        # As noted in the resource, there is also an alternative way
        # self.eigenvectors = U @ W

        self.eigenvectors_right_ = (
            shift_end @ V @ diagmat_dot_mat(S_inverse, eigenvector)
        )  # (1.23)

        return koopman_matrix_low_rank

    def fit(self, X: PRE_FIT_TYPES, y=None, **fit_params):
        self._validate_data(
            X,
            ensure_feature_name_type=True,
            validate_tsc_kwargs={"ensure_const_delta_time": True},
        )
        self._setup_features_and_time_fit(X)
        self._compute_internals(X)
        return self


class PyDMDWrapper(DMDBase):
    def __init__(
        self, method: str, svd_rank, tlsq_rank, exact, opt, **init_params,
    ):

        if not IS_IMPORTED_PYDMD:
            raise ImportError(
                "Python package pydmd could not be imported. Check installation."
            )

        self._setup_default_tsc_scorer_and_metric()

        self.method_ = method.lower()

        standard_params = {
            "svd_rank": svd_rank,
            "tlsq_rank": tlsq_rank,
            "exact": exact,
            "opt": opt,
        }

        if method == "dmd":
            self.dmd_ = pydmd.DMD(**standard_params)
        elif method == "hodmd":
            standard_params["d"] = init_params.pop("d", 1)
            self.dmd_ = pydmd.HODMD(**standard_params)

        elif method == "fbdmd":
            self.dmd_ = pydmd.FbDMD(**standard_params)
        elif method == "mrdmd":
            standard_params["max_cycles"] = init_params.pop("max_cycles", 1)
            standard_params["max_level"] = init_params.pop("max_level", 6)
            self.dmd_ = pydmd.MrDMD(**standard_params)
        elif method == "cdmd":
            standard_params["compression_matrix"] = init_params.pop(
                "max_level", "uniform"
            )
            self.dmd_ = pydmd.CDMD(**standard_params)
        elif method == "optdmd":
            standard_params["factorization"] = init_params.pop("factorization", "evd")
            self.dmd_ = pydmd.OptDMD(**standard_params)
        elif method == "dmdc":
            # self.dmd_ = pydmd.DMDc(**init_params)
            raise NotImplementedError(
                "Currently not implemented as it requires " "additional input"
            )
        else:
            raise ValueError(f"method={method} not known")

    def fit(self, X: PRE_FIT_TYPES, y=None, **fit_params) -> "PyDMDWrapper":

        self._validate_data(
            X,
            ensure_feature_name_type=True,
            validate_tsc_kwargs={"ensure_const_delta_time": True},
        )
        self._setup_features_and_time_fit(X=X)

        if len(X.ids) > 1:
            raise NotImplementedError(
                "Provided DMD methods only allow single time series analysis."
            )

        # data is column major
        self.dmd_.fit(X=X.to_numpy().T)
        self.eigenvectors_right_ = self.dmd_.modes
        self.eigenvalues_ = self.dmd_.eigs

        return self
