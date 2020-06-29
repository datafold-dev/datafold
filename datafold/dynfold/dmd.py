import abc
import warnings
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, Ridge, ridge_regression
from sklearn.utils.validation import check_is_fitted

from datafold.decorators import warn_experimental_class
from datafold.dynfold.base import InitialConditionType, TimePredictType, TSCPredictMixIn
from datafold.pcfold import InitialCondition, TSCDataFrame, allocate_time_series_tensor
from datafold.utils.general import (
    diagmat_dot_mat,
    if1dim_colvec,
    mat_dot_diagmat,
    sort_eigenpairs,
)

try:
    import pydmd
except ImportError:
    pydmd = None
    IS_IMPORTED_PYDMD = False
else:
    IS_IMPORTED_PYDMD = True


class LinearDynamicalSystem(object):
    r"""Evolve a linear dynamical system forward in time.

    A mathematical description of a linear dynamical system is

    - continuous
        .. math::
            \frac{d}{dt} x(t) = \mathcal{A} \cdot x(t),
            \mathcal{A} \in \mathbb{R}^{[m \times m]}

    This continuous-system representation can also be written in terms of a discrete-time
    system

    - discrete
        .. math::
            x_{n+1} = A \cdot x_{n}

    and :math:`A = \exp(\mathcal{A} \Delta t)`, a constant matrix, which describes the
    linear evolution of the systems' states :math:`x` with state length :math:`m`.

    Parameters
    ----------

    mode
        Type of linear system:

        * "continuous"
        * "discrete" (restricts time values to integer values)

    time_invariant
        If True, the system internally always starts with `time=0`. \
        This is irrespective of the time given in the time values. If the initial
        time is larger than zero, the internal times are corrected to the requested time.

    References
    ----------

    :cite:`kutz_dynamic_2016` (pages 3 ff.)
    
    """

    _cls_valid_modes = ("continuous", "discrete")

    def __init__(self, mode: str = "continuous", time_invariant: bool = True):
        self.mode = mode
        self.time_invariant = time_invariant

    def _check_time_values(self, time_values):

        if (time_values < 0).any():
            raise ValueError("time samples contain negative values")

        if np.isnan(time_values).any() or np.isinf(time_values).any():
            raise ValueError("time samples contain invalid vales (nan/inf)")

        if self.mode == "discrete":

            if time_values.dtype == np.integer:
                pass  # restrict time_sample
            elif (
                time_values.dtype == np.floating and (np.mod(time_values, 1) == 0).all()
            ):
                time_values = time_values.astype(np.int)
            else:
                raise TypeError(
                    "For mode=discrete the time_samples have to be integers"
                )

        return time_values

    def _check_initial_condition(self, ic, state_length):

        if ic.ndim == 1:
            ic = if1dim_colvec(ic)

        if ic.ndim != 2:
            raise ValueError(  # in case ndim > 2
                f"Parameter 'ic' must have 2 dimensions. Got ic.ndim={ic.ndim}"
            )

        if ic.shape[0] != state_length:
            raise ValueError(
                f"Mismatch in ic.shape[0]={ic.shape[0]} is not "
                f"dynmatrix.shape[1]={state_length}."
            )

        return ic

    def evolve_system_spectrum(
        self,
        dynmatrix: np.ndarray,
        eigenvalues: np.ndarray,
        time_delta: float,
        initial_conditions: np.ndarray,
        time_values: np.ndarray,
        time_series_ids: Optional[Dict] = None,
        feature_columns: Optional[Union[pd.Index, list]] = None,
    ):
        r"""Evolve the dynamical system with spectral components of the dynamical
        matrix.

        Using the eigenvalues on the diagonal matrix :math:`\Lambda` and (right)
        eigenvectors :math:`\Psi_r` of the constant matrix :math:`A`

        .. math::
            A \Psi_r = \Psi_r \Lambda

        the linear system evolves

        - continuous :math:`\left(t \in \mathbb{R}^{+}\right)`
            .. math::
                \frac{d}{dt} x(t) &= \Psi \cdot \exp(\Omega \cdot t) \cdot b(0) \\
                \Omega &= \frac{\log(\Lambda)}{\Delta t}

        - discrete :math:`\left(n = \frac{t}{\Delta t} \in \mathbb{N}\right)`
            .. math::
                x_{n+1} = \Psi \cdot \Lambda^n \cdot b_{0}

        where :math:`b(0)` and :math:`b_{0}` are the initial
        conditions of the respective system.

        .. note::
            Initial condition states :math:`x_0` of the original system need to be
            aligned to the right eigenvectors beforehand:

            * By using the right eigenvectors and solving in a least square sense

                .. math::
                    \Psi_r b_0 = x_0

            * , or by using the left eigenvectors and computing the matrix-vector product

                .. math::
                    \Psi_l x_0 = b_0

        Parameters
        ----------
        dynmatrix
            Spectral linear time map of shape `(n_feature, n_feature_states)`, \
            where `n_feature_states` is the length of the initial condition.

            * right eigenvectors :math:`\Psi` of matrix :math:`A` (in this case \
              `n_feature=n_feature_states`), or
            * linear transformation of right eigenvectors :math:`D \cdot \Psi`. This \
              allows `n_feature` to be larger or smaller than `n_feature_states`. \
              The matrix :math:`D` maps the states directly to another \
              space, e.g., only a selection of states for reduce memory footprint.

        eigenvalues
            eigenvalues of matrix :math:`A`

        time_delta
            Time delta :math:`\Delta t` for a continuous system.

        initial_conditions
            Single initial condition of shape `(n_features,)` or multiple of shape \
            `(n_features, n_initial_conditions)`.

        time_values
           Time values to evaluate the linear system at

           * `mode="continuous"` - :math:`t \in \mathbb{R}^{+}`
           * `mode="discrete"` - :math:`n \in \mathbb{N}_0`

        time_series_ids
           Unique integer time series IDs of shape `(n_initial_conditions,)` for each \
           respective initial condition. Defaults to `(0, 1, 2, ...)`.

        feature_columns
            Unique feature columns names of shape `(n_feature,)`.
            Defaults to `(0, 1, 2, ...)`.

        Returns
        -------
        TSCDataFrame
            Collection with a time series for each initial condition with \
            shape `(n_time_values, n_features)`.
        """

        n_feature, state_length = dynmatrix.shape

        self._check_time_values(time_values)
        self._check_initial_condition(initial_conditions, state_length=state_length)

        if time_series_ids is None:
            time_series_ids = np.arange(initial_conditions.shape[1])

        if feature_columns is None:
            feature_columns = np.arange(state_length)

        if len(feature_columns) != n_feature:
            raise ValueError(
                f"len(feature_columns)={feature_columns} != state_length={state_length}"
            )

        time_series_tensor = allocate_time_series_tensor(
            n_time_series=initial_conditions.shape[1],
            n_timesteps=time_values.shape[0],
            n_feature=n_feature,
        )

        if self.mode == "continuous":
            # TODO: see gitlab #82
            omegas = np.log(eigenvalues.astype(np.complex)) / time_delta

            for idx, time in enumerate(time_values):
                time_series_tensor[:, idx, :] = np.real(
                    dynmatrix
                    @ diagmat_dot_mat(np.exp(omegas * time), initial_conditions)
                ).T
        else:  # self.mode == "discrete"
            for idx, time in enumerate(time_values):
                time_series_tensor[:, idx, :] = np.real(
                    dynmatrix
                    @ diagmat_dot_mat(np.power(eigenvalues, time), initial_conditions)
                ).T

        return TSCDataFrame.from_tensor(
            time_series_tensor,
            time_series_ids=time_series_ids,
            columns=feature_columns,
            time_values=time_values,
        )


class DMDBase(BaseEstimator, TSCPredictMixIn, metaclass=abc.ABCMeta):
    r"""Abstract base class for Dynamic Mode Decomposition (DMD) models.

    A DMD model decomposes time series data linearly into spatial-temporal components.
    Due to it's strong connection to non-linear dynamical systems with Koopman spectral
    theory, the DMD variants (subclasses) are framed in the context of this theory.

    A DMD model approximates the Koopman operator with a matrix :math:`K`,
    which defines a linear dynamical system

    .. math:: K^n x_0 &= x_n

    with :math:`x_n` being the (column) state vectors of the system at time :math:`n`.
    Note, that the state vectors :math:`x` are often not the true original observations
    of a system but states from a functional coordinate basis that seeks to linearize the
    dynamics (see reference for details).

    The spectrum of the Koopman matrix \
    (:math:`\Psi_r` right eigenvectors, and :math:`\Lambda` eigenvalues on diagonal)

    .. math:: K \Psi_r = \Psi_r \Lambda

    enables further analysis about the system and inexpensive evaluation of the
    Koopman system (matrix power of diagonal matrix :math:`\Lambda` instead of
    :math:`K`):

    .. math::
        x_n &= K^n x_0 \\
        &= K^n \Psi_r b  \\
        &= \Psi_r \Lambda^n b

    The vector :math:`b` contains the Koopman modes for a specified initial conditions
    :math:`x_0`. The modes remain constant for a prediction and are aligned to the
    linear system, described by the spectral decomposition of the Koopman matrix. The
    modes can be either computed

    1. in a least squares sense with the right eigenvectors of the Koopman matrix

       .. math::
           \Psi_r b = x_0

    2. , or using the left Koopman matrix eigenvectors (:math:`\Psi_l`, if available) and
       inexpensive matrix-vector product \

       .. math::
           \Psi_l x_0 = b

    All subclasses of ``DMDBase`` must provide the (right) eigenpairs
    :math:`\left(\Lambda, \Psi_r\right)`, in respective attributes
    :code:`eigenvalues_` and :code:`eigenvectors_right_`. If the left eigenvectors
    (attribute :code:`eigenvectors_left_`) are available the initial condition always
    solves with the second case for :math:`b`, because this is more efficient.

    References
    ----------
    :cite:`kutz_dynamic_2016`

    See Also
    --------

    ``LinearDynamicalSystem``
    """

    def _evolve_dmd_system(
        self,
        X_ic: pd.DataFrame,
        time_values: np.ndarray,
        time_invariant=True,
        post_map: Optional[np.ndarray] = None,
        feature_columns=None,
    ):

        check_is_fitted(self, attributes=["eigenvectors_right_"])

        # type hints for mypy
        self.eigenvectors_left_: Optional[np.ndarray]
        self.eigenvectors_right_: np.ndarray
        self.eigenvalues_: np.ndarray

        if feature_columns is None:
            feature_columns = self.features_in_[1]

        # initial condition is numpy-only, from now on
        initial_conditions = X_ic.to_numpy().T
        time_series_ids = X_ic.index.get_level_values(
            TSCDataFrame.tsc_id_idx_name
        ).to_numpy()

        if len(np.unique(time_series_ids)) != len(time_series_ids):
            # check if duplicate ids are present
            raise ValueError("time series ids have to be unique")

        # Choose alternative of how to evolve the linear system:
        if hasattr(self, "eigenvectors_left_") and (
            self.eigenvectors_left_ is not None and self.eigenvectors_right_ is not None
        ):
            # Uses both eigenvectors (left and right). Used if is_diagonalize=True in
            # DMDFull
            # The Koopman modes are b_0 in the documentation
            koopman_modes = self.eigenvectors_left_ @ initial_conditions
        elif (
            hasattr(self, "eigenvectors_right_")
            and self.eigenvectors_right_ is not None
        ):
            # represent the initial condition in terms of right eigenvectors (by solving a
            # least-squares problem) -- only the right eigenvectors are required
            koopman_modes = np.linalg.lstsq(
                self.eigenvectors_right_, initial_conditions, rcond=None
            )[0]
        else:
            raise NotImplementedError(
                "The DMD subclass does not provide the attribute 'eigenvectors_right_', "
                "which is required to evolve the dynamical system. Please check "
                "implementation or report bug. "
            )

        if post_map is not None:
            # transform eigenvectors with post_map
            try:
                post_map = np.asarray(post_map, dtype=np.float64)
            except Exception:
                raise TypeError("Cannot convert post_map to numpy array.")
            else:
                if post_map.ndim != 2:
                    raise TypeError("'post_map' must be two dimensional")

            dynmatrix = post_map @ self.eigenvectors_right_
        else:
            dynmatrix = self.eigenvectors_right_

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
            dynmatrix=dynmatrix,
            eigenvalues=self.eigenvalues_,
            time_delta=self.dt_,
            initial_conditions=koopman_modes,
            time_values=norm_time_samples,
            time_series_ids=time_series_ids,
            feature_columns=feature_columns,
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
            tsc_df.tsc.shift_time(shift_t=shift).time_values() - time_values < 1e-15
        ).all()

        # Hard set of time_values
        tsc_df.index = tsc_df.index.set_levels(
            time_values, level=1
        ).remove_unused_levels()

        return tsc_df

    @abc.abstractmethod
    def fit(self, X: TimePredictType, **fit_params) -> "DMDBase":
        """Abstract method to train DMD model.

        Parameters
        ----------
        X
            Training data
        """
        raise NotImplementedError("base class")

    def predict(
        self, X: InitialConditionType, time_values=None, **predict_params
    ) -> TSCDataFrame:
        """Predict time series data for each initial condition and time values.

        Parameters
        ----------
        X: pandas.DataFrame, numpy.ndarray
            Initial conditions of shape `(n_initial_condition, n_features)`.

        time_values
            Time values to evaluate the model at.

        Keyword Args
        ------------

        post_map: numpy.ndarray
            A matrix that is combined with the right eigenvectors. \
            :code:`post_map @ eigenvectors_right_`.

        feature_columns: pandas.Index
            If `post_map` is given with a changed state length, then new feature names
            must be provided.

        Returns
        -------
        TSCDataFrame
            time series predictions of shape `(n_time_values, n_features)` for each
            initial condition
        """

        check_is_fitted(self)

        if isinstance(X, np.ndarray):
            # work internally only with DataFrames
            X = InitialCondition.from_array(X, columns=self.features_in_[1])
        else:
            InitialCondition.validate(X)

        self._validate_data(X)

        X, time_values = self._validate_features_and_time_values(
            X=X, time_values=time_values
        )

        post_map = predict_params.pop("post_map", None)
        feature_columns = predict_params.pop("feature_columns", None)

        if len(predict_params.keys()) > 0:
            raise KeyError(f"predict_params keys are invalid: {predict_params.keys()}")

        return self._evolve_dmd_system(
            X_ic=X,
            time_values=time_values,
            post_map=post_map,
            feature_columns=feature_columns,
        )

    def reconstruct(self, X: TSCDataFrame):
        """Reconstruct time series collection.

        Extract the same initial states from the time series in the collection and
        predict the other states with the model at the same time values.

        Parameters
        ----------
        X: TSCDataFrame
            Time series to reconstruct.

        Returns
        -------
        TSCDataFrame
            same shape as input `X`
        """

        check_is_fitted(self)
        X = self._validate_data(
            X,
            ensure_feature_name_type=True,
            validate_tsc_kwargs={"ensure_const_delta_time": True},
        )
        self._validate_feature_names(X)

        X_reconstruct_ts = []

        for X_ic, time_values in InitialCondition.iter_reconstruct_ic(
            X, n_samples_ic=1
        ):
            X_ts = self.predict(X=X_ic, time_values=time_values)
            X_reconstruct_ts.append(X_ts)

        X_reconstruct_ts = pd.concat(X_reconstruct_ts, axis=0)
        return X_reconstruct_ts

    def fit_predict(self, X: TSCDataFrame, **fit_params):
        """Fit model and reconstruct the time series data.

        Parameters
        ----------
        X
            Training time series data.

        Returns
        -------
        TSCDataFrame
            same shape as input `X`
        """
        return self.fit(X, **fit_params).reconstruct(X)

    def score(self, X: TSCDataFrame, y=None, sample_weight=None) -> float:
        """Score model by reconstructing time series data.

        The default metric (see :class:`.TSCMetric` used is mode="feature", "metric=rmse"
        and "min-max" scaling.

        Parameters
        ----------
        X
            Time series data to reconstruct with `(n_samples, n_features)`.

        y: None
            ignored

        sample_weight
            passed to :py:meth:`TSCScoring.__call__`.

        Returns
        -------
        float
            score
        """
        self._check_attributes_set_up(check_attributes=["_score_eval"])

        # does checks:
        X_est_ts = self.reconstruct(X)

        return self._score_eval(X, X_est_ts, sample_weight)


class DMDFull(DMDBase):
    r"""Full Dynamic Mode Decomposition of time series data.

    The model approximates the Koopman matrix with

    .. math::
        K X &= X' \\
        K &= X' X^{\dagger},

    where :math:`\dagger` defines the Moore–Penrose inverse and column oriented
    snapshots in :math:`X`.
    
    ...

    Parameters
    ----------

    is_diagonalize
        If True, also the left eigenvectors are computed. This is more efficient to
        solve for initial conditions, because there is no least
        squares computation required for evaluating the linear dynamical
        system (see :class:`LinearDynamicalSystem`).

    rcond: Optional[float]
        Parameter handled to :class:`numpy.linalg.lstsq`.

    Attributes
    ----------

    eigenvalues_: numpy.ndarray
        Eigenvalues of Koopman matrix.

    eigenvectors_right_: numpy.ndarray
        All right eigenvectors of Koopman matrix.

    eigenvectors_left_: numpy.ndarray
        All left eigenvectors of Koopman matrix if ``is_diagonalize=True``.

    koopman_matrix_: numpy.ndarray
        Koopman matrix obtained from least squares. Only stored if
        `store_koopman_matrix=True` during fit.

    References
    ----------

    :cite:`schmid_dynamic_2010`
    :cite:`kutz_dynamic_2016`

    """

    def __init__(self, is_diagonalize: bool = False, rcond: Optional[float] = None):
        self._setup_default_tsc_metric_and_score()
        self.is_diagonalize = is_diagonalize
        self.rcond = rcond

    def _diagonalize_left_eigenvectors(self, koopman_matrix):
        """Compute right eigenvectors (not normed) such that
        Koopman matrix = right_eigenvectors @ diag(eigenvalues) @ left_eigenvectors .
        """

        # lhs_matrix = (np.diag(self.eigenvalues_) @ self.eigenvectors_right_)
        lhs_matrix = self.eigenvectors_right_ * self.eigenvalues_

        # NOTE: the left eigenvectors are not normed (i.e. ||ev|| != 1
        self.eigenvectors_left_ = np.linalg.solve(lhs_matrix, koopman_matrix)

    def _compute_koopman_matrix(self, X: TSCDataFrame):

        # It is more suitable to get the shift_start and end in row orientation as this
        # is closer to the normal least squares parameter definition
        shift_start_transposed, shift_end_transposed = X.tsc.compute_shift_matrices(
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
        koopman_matrix, residual, rank, _ = np.linalg.lstsq(G, G_dash, rcond=self.rcond)

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

    def fit(
        self, X: TimePredictType, y=None, store_koopman_matrix=False, **fit_params
    ) -> "DMDFull":
        """Build Koopman matrix and its spectral components from time series data.

        Parameters
        ----------
        X
            Training time series data.

        y: None
            ignored

        store_koopman_matrix
            If True, the model stores the Koopman matrix in attribute
            ``koopman_matrix_``, otherwise only the spectral components are stored.

        Returns
        -------
        DMDFull
            self
        """

        self._validate_data(
            X=X,
            ensure_feature_name_type=True,
            validate_tsc_kwargs={"ensure_const_delta_time": True},
        )
        self._setup_features_and_time_fit(X=X)

        koopman_matrix_ = self._compute_koopman_matrix(X)
        self.eigenvalues_, self.eigenvectors_right_ = sort_eigenpairs(
            *np.linalg.eig(koopman_matrix_)
        )

        if self.is_diagonalize:
            self._diagonalize_left_eigenvectors(koopman_matrix_)

        if store_koopman_matrix:
            self.koopman_matrix_ = koopman_matrix_

        return self


class DMDEco(DMDBase):
    r"""Dynamic Mode Decomposition of time series data with prior singular value
    decomposition (SVD).

    The singular value decomposition (SVD) reduces the data and the Koopman operator is
    computed in this reduced space. This DMD model is particularly interesting for high
    dimensional data (large number of features), for example, solutions of partial
    differential equations (PDE) with a fine grid.

    The procedure of ``DMDEco`` is as follows:

    1. Compute the singular value decomposition of the data and use the leading `k`
    singular values and corresponding vectors in :math:`U` and :math:`V`.

      .. math::
          X \approx U \Sigma V^*

    2. Compute the Koopman matrix in the SVD coordinates:

      .. math::
          K = U^T X' V \Sigma^{-1}

    3. Compute all eigenpairs of Koopman matrix:

      .. math::
          K W_r = W_r \Omega

    4. Reconstruct the (exact) eigendecomposition of :math:`K`

      .. math::
          \Psi_r = X' V \Sigma^{-1} W

      .. note::
          The eigenvectors in step 4 can also be computed with :math:`\Psi_r = U W`, which
          is then referred to the projected reconstruction.

    ...

    Parameters
    ----------
    svd_rank: int
        Number of eigenpairs (with largest eigenvalues, in magnitude) to keep.

    Attributes
    ----------

    eigenvalues_ : numpy.ndarray
        All eigenvalues of shape `(svd_rank,)` of the (reduced) Koopman matrix .

    eigenvectors_right_ : numpy.ndarray
        All right eigenvectors of shape `(svd_rank, svd_rank)` of the reduced Koopman
        matrix.

    References
    ----------

    :cite:`kutz_dynamic_2016`
    :cite:`tu_dynamic_2014`

    """

    def __init__(self, svd_rank=10):
        self._setup_default_tsc_metric_and_score()
        self.svd_rank = svd_rank

    def _compute_internals(self, X: TSCDataFrame):
        # TODO: different orientations are good for different cases:
        #  1 more snapshots than quantities
        #  2 more quantities than snapshots
        #  Currently it is optimized for the case 2.

        shift_start, shift_end = X.tsc.compute_shift_matrices(
            snapshot_orientation="col"
        )
        U, S, Vh = np.linalg.svd(shift_start, full_matrices=False)  # (1.18)

        U = U[:, : self.svd_rank]
        S = S[: self.svd_rank]
        S_inverse = np.reciprocal(S, out=S)

        V = Vh.T
        V = V[:, : self.svd_rank]

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

    def fit(self, X: TimePredictType, y=None, **fit_params):
        self._validate_data(
            X,
            ensure_feature_name_type=True,
            validate_tsc_kwargs={"ensure_const_delta_time": True},
        )
        self._setup_features_and_time_fit(X)
        self._compute_internals(X)
        return self


@warn_experimental_class
class PyDMDWrapper(DMDBase):
    """
    .. warning::
        This class is not documented and clsasified as experimental.
        Contributions are welcome:
            * documentation
            * write unit tests
            * improve code
    """

    def __init__(
        self, method: str, svd_rank, tlsq_rank, exact, opt, **init_params,
    ):

        if not IS_IMPORTED_PYDMD:
            raise ImportError(
                "Python package pydmd could not be imported. Check installation."
            )
        assert pydmd is not None

        self._setup_default_tsc_metric_and_score()
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
                "Currently not implemented because DMD with control requires "
                "additional input."
            )
        else:
            raise ValueError(f"method={method} not known")

    def fit(self, X: TimePredictType, y=None, **fit_params) -> "PyDMDWrapper":

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
