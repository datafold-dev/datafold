import abc
import copy
import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.linalg
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_scalar

from datafold.dynfold.base import InitialConditionType, TimePredictType, TSCPredictMixin
from datafold.dynfold.dynsystem import LinearDynamicalSystem
from datafold.pcfold import InitialCondition, TSCDataFrame
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


def compute_spectal_components(system_matrix, is_diagonalize):
    eigenvalues, eigenvectors_right = sort_eigenpairs(
        *np.linalg.eig(system_matrix)
    )
    eigenvectors_right /= np.linalg.norm(eigenvectors_right, axis=0)

    if is_diagonalize:
        # Compute left eigenvectors such that
        #     system_matrix = eigenvectors_right_ @ diag(eigenvalues) @ eigenvectors_left_
        #
        #  NOTE:
        #     The left eigenvectors are
        #          * not normed
        #          * row-wise in returned matrix
        eigenvectors_left_ = np.linalg.solve(mat_dot_diagmat(eigenvectors_right, eigenvalues), system_matrix)
    else:
        eigenvectors_left_ = None

    return eigenvectors_right, eigenvalues, eigenvectors_left_


class DMDBase(
    BaseEstimator, LinearDynamicalSystem, TSCPredictMixin, metaclass=abc.ABCMeta
):
    r"""Abstract base class for Dynamic Mode Decomposition (DMD) models.

    A DMD model decomposes time series data linearly into spatial-temporal components.
    The decomposition defines a linear dynamical system. Due to it's strong connection to
    non-linear dynamical systems with Koopman spectral theory
    (see e.g. introduction in :cite:t:`tu-2014`), the DMD variants (subclasses)
    are framed in the context of this theory.

    A DMD model approximates the Koopman operator with a matrix :math:`K`, which defines a
    linear dynamical system

    .. math:: K^n x_0 &= x_n

    with :math:`x_n` being the (column) state vectors of the system at timestep :math:`n`.
    Note, that the state vectors :math:`x`, when used in conjunction with the
    :py:meth:`EDMD` model are not the original observations of a system, but states from a
    functional coordinate basis that seeks to linearize the dynamics (see reference for
    details).

    A subclass can either provide :math:`K`, the spectrum of :math:`K` or the generator
    :math:`U` of :math:`K`

    .. math::
        U = \frac{K-I}{\Delta t}

    The spectrum of the Koopman matrix (or equivalently its generator) \
    (:math:`\Psi_r` right eigenvectors, and :math:`\Lambda` matrix with eigenvalues on
    diagonal)

    .. math:: K \Psi_r = \Psi_r \Lambda

    enables further analysis (e.g. stability) about the system and inexpensive
    evaluation of the Koopman system (matrix power of diagonal matrix :math:`\Lambda`
    instead of :math:`K`):

    .. math::
        x_n &= K^n x_0 \\
        &= K^n \Psi_r b_0  \\
        &= \Psi_r \Lambda^n b_0

    The vector :math:`b_0` contains the initial state (adapted from :math:`x_0` to the
    spectral system state). In the Koopman analysis this corresponds to the initial
    Koopman eigenfunctions, whereas in a 'pure' DMD setting this is often referred to the
    initial amplitudes.

    The DMD modes :math:`\Psi_r` remain constant.

    All subclasses of ``DMDBase`` are also subclasses of
    :py:class:`.LinearDynamicalSystem` and must therefore set up and specify the system
    (see :py:meth:`setup_sys_spectral` and :py:meth:`setup_sys_matrix`).

    References
    ----------
    :cite:t:`schmid-2010` - DMD method in the original sense
    :cite:t:`rowley-2009` - connects the DMD method to Koopman operator theory
    :cite:t:`tu-2014` - generalizes the DMD to temporal snapshot pairs
    :cite:t:`williams-2015` - generalizes the approximation to a lifted space
    :cite:t:`kutz-2016` - an introductory book for DMD and its connection to Koopman
    theory

    See Also
    --------

    :py:class:`.LinearDynamicalSystem`

    """

    @property
    def dmd_modes(self):
        if not self.is_spectral_mode:
            raise AttributeError(
                "The DMD modes are not available because the system is "
                "not set up in spectral mode."
            )
        if self.is_linear_system_setup(raise_error_if_not_setup=True):
            return self.eigenvectors_right_

        raise NotImplementedError("Please report bug.")  # should not get here

    def _read_predict_params(self, predict_params):

        # user defined post_map
        post_map = predict_params.pop("post_map", None)
        user_set_modes = predict_params.pop("modes", None)
        feature_columns = predict_params.pop("feature_columns", None)

        if len(predict_params.keys()) > 0:
            raise KeyError(f"predict_params keys are invalid: {predict_params.keys()}")

        if post_map is not None and user_set_modes is not None:
            raise ValueError("Can only provide 'post_map' or 'modes' in **kwargs")
        elif post_map is not None or user_set_modes is not None:
            if feature_columns is None:
                raise ValueError(
                    "If 'post_map' or 'modes' are provided it is necessary "
                    "to set 'feature_columns' in **kwargs"
                )

        if self.is_matrix_mode and (post_map is not None or user_set_modes is not None):
            raise ValueError("post_map can only be provided with 'sys_type=spectral'")

        return post_map, user_set_modes, feature_columns

    @abc.abstractmethod
    def fit(
        self, X: TimePredictType, *, U: Optional[TSCDataFrame], y=None, **fit_params
    ) -> "DMDBase":
        """Abstract method to train DMD model.

        Parameters
        ----------
        X
            Training data

        U
            Control data (set to None as a default if the subclass does not support control
            input)

        y
            ignored (Hint for future dev.: This parameter is reserved to specify an extra map,
            e.g. to specific features in `X` or a separate system feature)

        """
        raise NotImplementedError("base class")

    def _read_user_sys_matrix(self, post_map, user_set_modes):
        assert not (post_map is not None and user_set_modes is not None)

        if post_map is not None:
            post_map = post_map.astype(float)
            modes = post_map @ self.eigenvectors_right_
        elif user_set_modes is not None:
            modes = user_set_modes
        else:
            modes = None

        return modes

    def _evolve_dmd_system(
        self,
        X_ic: TSCDataFrame,
        overwrite_sys_matrix: Optional[np.ndarray],
        control_input: Optional[TSCDataFrame],
        time_values: np.ndarray,
        feature_columns=None,
    ):
        self.is_linear_system_setup(raise_error_if_not_setup=True)

        if feature_columns is None:
            feature_columns = self.feature_names_in_

        # initial condition is numpy-only, from now on, and column-oriented
        initial_states_origspace = X_ic.to_numpy().T

        if control_input is not None:
            # 3-dim tensor for the control input over the predicted time horizon
            control_input = control_input.to_numpy().reshape(
                len(control_input.ids), -1, len(control_input.columns)
            )

        time_series_ids = X_ic.index.get_level_values(
            TSCDataFrame.tsc_id_idx_name
        ).to_numpy()

        if len(np.unique(time_series_ids)) != len(time_series_ids):
            # check if duplicate ids are present
            raise ValueError("time series ids have to be unique")

        if self.is_matrix_mode:
            # no adaptation required
            initial_states_dmd = initial_states_origspace
        else:  # self.is_spectral_mode()
            initial_states_dmd = self.compute_spectral_system_states(
                states=initial_states_origspace
            )

        if self.time_invariant:
            shift = np.min(time_values)
        else:
            # If the dmd time is shifted during data (e.g. the minimum processed data
            # starts with time=5, some positive value) then normalize the time_samples
            # with this shift. The linear system handles the shifted time start as time
            # zero.
            # shift = self.time_interval_[0]
            raise NotImplementedError(
                "'time_invariant = False' is currently not supported"
            )

        norm_time_samples = time_values - shift

        tsc_df = self.evolve_system(
            initial_conditions=initial_states_dmd,
            time_values=norm_time_samples,
            control_input=control_input,
            overwrite_sys_matrix=overwrite_sys_matrix,
            time_delta=self.dt_,
            time_series_ids=time_series_ids,
            feature_names_out=feature_columns,
        )

        # correct the time shift again according to the training data
        # (not necessarily normed time steps [0, 1, 2, ...])
        # One way is to shift the time again, i.e.
        #
        #    tsc_df.tsc.shift_time(shift_t=shift)
        #
        # However, this can sometimes introduce numerical noise (forward/backwards
        # shifting). Therefore, the user-requested `time_values` are set directly into the
        # index. This way the time values are exactly the same accross for all time
        # series.
        #
        # Because hard-setting the time indices can be problematic, the following
        # assert makes sure that both ways match (up to numerical differences).

        if time_values.dtype == float:
            assert (
                tsc_df.tsc.shift_time(shift_t=shift).time_values() - time_values < 1e-14
            ).all()
        elif time_values.dtype == int:
            assert (
                tsc_df.tsc.shift_time(shift_t=shift).time_values() - time_values == 0
            ).all()

        # Set time_values from user input
        tsc_df.index = tsc_df.index.set_levels(
            time_values, level=1
        ).remove_unused_levels()

        return tsc_df

    def get_feature_names_out(self, input_features=None):
        if input_features is None and hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        else:
            return input_features

    def predict(
        self,
        X: InitialConditionType,
        *,
        U: Optional[TSCDataFrame] = None,
        time_values: Optional[np.ndarray] = None,
        **predict_params,
    ) -> TSCDataFrame:
        """Predict time series data for each initial condition and time values.

        Parameters
        ----------
        X: TSCDataFrame, numpy.ndarray
            Initial conditions of shape `(n_initial_condition, n_features)`.

        time_values
            Time values to evaluate the model at. If not provided, then the time at the
            initial condition plus ``dt_`` is set (i.e. predict a single step).

        Keyword Args
        ------------

        post_map: Union[numpy.ndarray, scipy.sparse.spmatrix]
            A matrix that is combined with the right eigenvectors. \
            :code:`post_map @ eigenvectors_right_`. If set, then also the input
            `feature_columns` is required. It cannot be set with 'modes' at the same
            time and requires "sys_type=spectral".

        modes: Union[numpy.ndarray]
            A matrix that sets the DMD modes directly. This must not be given at the
            same time with ``post_map``. If set, then also the input ``feature_columns``
            is required. It cannot be set with 'modes' at the same time and requires
            "sys_type=spectral".

        feature_columns: pandas.Index
            If ``post_map`` is given with a changed state length, then new feature names
            must be provided.

        Returns
        -------
        TSCDataFrame
            The computed time series predictions, where each time series has shape
            `(n_time_values, n_features)`.
        """

        check_is_fitted(self)

        time_values = self._set_and_validate_time_values_predict(
            time_values=time_values, X=X, U=U
        )

        if isinstance(X, np.ndarray):
            # work internally only with DataFrames
            X = InitialCondition.from_array(
                X,
                time_value=time_values[0],
                feature_names=self.feature_names_in_,
                ts_ids=U.ids if isinstance(U, TSCDataFrame) else None,
            )
        else:
            # for DMD the number of samples per initial condition is always 1
            InitialCondition.validate(X, n_samples_ic=1)

        self._validate_datafold_data(X)

        if U is not None:

            if X.n_timeseries > 1:
                raise NotImplementedError(
                    "If U is a numpy array, then only a prediction with "
                    "a single initial condition is allowed. "
                    f"Got {X.n_timeseries}"
                )

            if isinstance(U, np.ndarray):
                U = InitialCondition.from_array_control(
                    U,
                    control_names=self.control_names_in_,
                    dt=self.dt_,
                    time_values=time_values,
                    ts_id=int(X.ids[0]) if isinstance(X, TSCDataFrame) else None,
                )
                # TODO: include pd.DataFrame type (turn to TSCDataFrame)

            self._validate_datafold_data(
                U, ensure_tsc=True, tsc_kwargs=dict(ensure_same_time_values=True)
            )

            InitialCondition.validate_control(X_ic=X, U=U)

        X, U, time_values = self._validate_features_and_time_values(
            X=X, U=U, time_values=time_values
        )

        post_map, user_set_modes, feature_columns = self._read_predict_params(
            predict_params=predict_params
        )

        overwrite_sys_matrix = self._read_user_sys_matrix(
            post_map=post_map, user_set_modes=user_set_modes
        )

        return self._evolve_dmd_system(
            X_ic=X,
            overwrite_sys_matrix=overwrite_sys_matrix,
            control_input=U,
            time_values=time_values,
            feature_columns=feature_columns,
        )

    def reconstruct(
        self,
        X: TSCDataFrame,
        *,
        U: Optional[TSCDataFrame] = None,
        qois: Optional[Union[np.ndarray, pd.Index, List[str]]] = None,
    ) -> TSCDataFrame:
        """Reconstruct time series collection.

        Extract the same initial states from the time series in the collection and
        predict the other states with the model at the same time values.

        Parameters
        ----------
        X
            Time series to reconstruct.

        qois
            A list of feature names of interest to be include in the returned
            predictions. Passed to :py:meth:`.predict`.

        Returns
        -------
        TSCDataFrame
            same shape as input `X`
        """

        check_is_fitted(self)
        X = self._validate_datafold_data(
            X,
            ensure_tsc=True,
            tsc_kwargs=dict(ensure_const_delta_time=True),
        )
        self._validate_feature_names(X)

        # TODO: qois flag is currently not supported in DMD, bc. predict does not
        #  support it gitlab issue #125
        # self._validate_qois(qois=qois, valid_feature_names=self.feature_names_in_)

        X_reconstruct_ts = []

        for X_ic, time_values in InitialCondition.iter_reconstruct_ic(
            X, n_samples_ic=1
        ):
            if U is not None:
                U_ic = U.loc[pd.IndexSlice[X_ic.ids, :], :]
            else:
                U_ic = None
            X_ts = self.predict(X=X_ic, U=U_ic)
            X_reconstruct_ts.append(X_ts)

        return pd.concat(X_reconstruct_ts, axis=0)

    def fit_predict(
        self, X: TSCDataFrame, *, U: Optional[TSCDataFrame] = None, y=None, **fit_params
    ) -> TSCDataFrame:
        """Fit model and reconstruct the time series data.

        Parameters
        ----------
        X
            Training time series data.
        y
            ignored

        Returns
        -------
        TSCDataFrame
            same shape as input `X`
        """
        return self.fit(X, U=U, **fit_params).reconstruct(X, U=U)

    def score(
        self,
        X: TSCDataFrame,
        *,
        U: Optional[TSCDataFrame] = None,
        y=None,
        sample_weight=None,
    ) -> float:
        """Score model by reconstructing time series data.

        The default metric (see :class:`.TSCMetric` used is mode="feature", "metric=rmse"
        and "min-max" scaling.

        Parameters
        ----------
        X
            Time series data to reconstruct with `(n_samples, n_features)`.

        U
            Time series with control states (only necessary if the model was fit with control
            input).

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

        X_est_ts = self.reconstruct(X, U=U)  # does all the validation checks
        return self._score_eval(X, X_est_ts, sample_weight)


class DMDFull(DMDBase):
    r"""Full dynamic mode decomposition of time series collection data.

    The model computes a Koopman matrix :math:`K` with

    .. math::
        K X &= X^{+} \\
        K &= X^{+} X^{\dagger},

    where :math:`X` is the data with column oriented snapshots, :math:`\dagger`
    the Moore–Penrose inverse and :math:`+` the future time shifted data.

    The actual decomposition contains the spectral elements of the matrix :math:`K`.

    ...

    Parameters
    ----------

    sys_mode
       Select a mode to evolve the linear system with

       * "spectral" to use spectral components of the system matrix. The evaluation of
         the linear system is cheap and it provides valuable information about the
         underlying process. On the downside this mode has numerical issues if the
         system matrix is badly conditioned.
       * "matrix" to use system matrix directly. The evaluation of the system is more
         robust, but the system evaluation is computationally more expensive.

    is_diagonalize
        If True, also the left eigenvectors are computed to diagonalize the system matrix.
        This affects of how initial conditions are adapted for the spectral system
        representation (instead of a least squares :math:`\Psi_r^\dagger x_0` with right
        eigenvectors it performs :math:`\Psi_l x_0`). The parameter is ignored if
        ``sys_mode=matrix``.

    approx_generator
        If True, approximate the generator of the system

        * `mode=spectral` compute (complex) eigenvalues of the
          Koopman generator :math:`log(\lambda) / \Delta t`, with eigenvalues `\lambda`
          of the Koopman matrix. Note, that the left and right eigenvectors remain the
          same.
        * `mode=matrix` compute generator matrix with
          :math:`logm(K) / \Delta t`, where :math:`logm` is the matrix logarithm.

        .. warning::

            This operation can fail if the eigenvalues of the matrix :math:`K` are too
            close to zero or the matrix logarithm is ill-defined because of
            non-uniqueness. For details see :cite:t:`dietrich-2020` (Eq.
            3.2. and 3.3. and discussion). Currently, there are no counter measurements
            implemented to increase numerical robustness (work is needed). Consider
            also :py:class:`.gDMDFull`, which provides an alternative way to
            approximate the Koopman generator by using finite differences.

    rcond: Optional[float]
        Cut-off ratio for small singular values
        Passed to `rcond` of py:method:`numpy.linalg.lstsq`.

    Attributes
    ----------

    eigenvalues_ : numpy.ndarray
        Eigenvalues of Koopman matrix.

    eigenvectors_right_ : numpy.ndarray
        All right eigenvectors of Koopman matrix; ordered column-wise.

    eigenvectors_left_ : numpy.ndarray
        All left eigenvectors of Koopman matrix with ordered row-wise.
        Only accessible if ``is_diagonalize=True``.

    koopman_matrix_ : numpy.ndarray
        Koopman matrix obtained from least squares. Only available if
        ``store_system_matrix=True`` during fit.

    generator_matrix_ : numpy.ndarray
        Koopman generator matrix obtained from Koopman matrix via matrix-logarithm.
        Only available if ``store_system_matrix=True`` during fit.

    References
    ----------

    * :cite:t:`schmid-2010` - DMD method in the original sense
    * :cite:t:`rowley-2009` - connects the DMD method to Koopman operator theory
    * :cite:t:`tu-2014` - generalizes the DMD to temporal snapshot pairs
    * :cite:t:`williams-2015` - generalizes the approximation to a lifted space
    * :cite:t:`kutz-2016` - an introductory book for DMD and Koopman connection
    """

    def __init__(
        self,
        *,  # keyword-only
        sys_mode: str = "spectral",
        is_diagonalize: bool = False,
        approx_generator: bool = False,
        rcond: Optional[float] = None,
    ):
        self.is_diagonalize = is_diagonalize
        self.approx_generator = approx_generator
        self.rcond = rcond

        self._setup_default_tsc_metric_and_score()

        super(DMDFull, self).__init__(
            sys_type="differential" if self.approx_generator else "flowmap",
            sys_mode=sys_mode,
            time_invariant=True,
        )

    def _compute_koopman_matrix(self, X: TSCDataFrame):

        # It is more suitable to get the shift_start and shift_end in row orientation as
        # this is closer to the normal least squares parameter definition
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

        # see Eq. (13 a) and (13 b) in `williams_datadriven_2015`
        G = shift_start_transposed.T @ shift_start_transposed
        G = np.multiply(1 / X.shape[0], G, out=G)

        G_dash = shift_start_transposed.T @ shift_end_transposed
        G_dash = np.multiply(1 / X.shape[0], G_dash, out=G_dash)

        # If the matrix is square and of full rank, then 'koopman_matrix' is the exact
        # solution of the linear equation system.
        koopman_matrix, residual, rank, _ = np.linalg.lstsq(G, G_dash, rcond=self.rcond)
        if rank != G.shape[1]:
            warnings.warn(
                f"Shift matrix ({G.shape=}) has not full rank ({rank=}), falling "
                f"back to least squares solution. The sum of residuals is: "
                f"{np.sum(residual)=}"
            )

        # # TODO: START Experimental (test other solvers, with more functionality)
        # #  ridge_regression, and sparisty promoting least squares solutions could be
        #    included here
        # # TODO: clarify if the ridge regression should be done better on lstsq with
        #     shift matrices (instead of the G, G_dash)

        # TODO: fit_intercept option useful to integrate?
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.
        # html#sklearn.linear_model.RidgeCV
        # from sklearn.linear_model import LinearRegression, Ridge, ridge_regression
        # from sklearn.linear_model import RidgeCV
        #
        # ridge = RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.05, 1],
        # normalize=False, fit_intercept=False)
        # ridge.fit(X=shift_start_transposed, y=shift_end_transposed)
        # koopman_matrix = ridge.coef_.T
        #
        # print(f"best alpha value {ridge.alpha_}")

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
        # (G_k)^T * K^T = G_{k+1}^T  (therefore the row snapshot orientation at the
        #                             beginning)

        koopman_matrix = koopman_matrix.conj().T
        return koopman_matrix

    def fit(self, X: TimePredictType, *, U=None, y=None, **fit_params) -> "DMDFull":
        """Compute Koopman matrix and if applicable the spectral components.

        Parameters
        ----------
        X
            Training time series data.

        y: None
            ignored

        U: None
            ignored (the method does not support control input)

        **fit_params

         - store_system_matrix
            If True, the model stores the system matrix -- either Koopman
            matrix or Koopman generator matrix -- in attribute ``koopman_matrix_`` or
            ``generator_matrix_`` respectively. The parameter is ignored if
            ``sys_mode=="matrix"`` (the system matrix is then in attribute
            ``sys_matrix_``).

        Returns
        -------
        DMDFull
            self
        """

        self._validate_datafold_data(
            X=X,
            ensure_tsc=True,
            tsc_kwargs=dict(ensure_const_delta_time=True),
        )
        self._validate_and_setup_fit_attrs(X=X)

        store_system_matrix = self._read_fit_params(
            attrs=[("store_system_matrix", False)], fit_params=fit_params
        )

        koopman_matrix_ = self._compute_koopman_matrix(X)

        if self.is_spectral_mode:
            (
                eigenvectors_right_,
                eigenvalues_,
                eigenvectors_left_,
            ) = compute_spectal_components(koopman_matrix_, is_diagonalize=self.is_diagonalize)

            if self.approx_generator:
                # see e.g.https://arxiv.org/pdf/1907.10807.pdf pdfp. 10
                # Eq. 3.2 and 3.3.
                eigenvalues_ = np.log(eigenvalues_.astype(complex)) / self.dt_

            self.setup_spectral_system(
                eigenvectors_right=eigenvectors_right_,
                eigenvalues=eigenvalues_,
                eigenvectors_left=eigenvectors_left_,
            )

            if store_system_matrix:
                if self.approx_generator:
                    self.generator_matrix_ = (
                        scipy.linalg.logm(koopman_matrix_) / self.dt_
                    )
                else:
                    self.koopman_matrix_ = koopman_matrix_
        else:  # self.is_matrix_mode()
            if self.approx_generator:
                generator_matrix_ = scipy.linalg.logm(koopman_matrix_) / self.dt_
                self.setup_matrix_system(system_matrix=generator_matrix_)
            else:
                self.setup_matrix_system(system_matrix=koopman_matrix_)

        return self


class gDMDFull(DMDBase):
    r"""Full Dynamic Mode Decomposition of time series data to approximate the Koopman
    generator.

    The model computes the Koopman generator matrix :math:`L` with

    .. math::
        L X &= \dot{X} \\
        L &= \dot{X} X^{\dagger},

    where :math:`X` is the data with column oriented snapshots, :math:`\dagger`
    the Moore–Penrose inverse, and :math:`\dot{X}` contains the time derivative.

    .. warning::
        The time derivative is currently computed with finite differences (using the
        `findiff <https://github.com/maroba/findiff>`__ package). For some systems the
        time derivatives is also available in analytical form (or can be computed with
        automatic differentiation). These cases are currently not supported and require
        further implementation.

    ...

    Parameters
    ----------
    sys_mode
        Select a mode to evolve the linear system with

       * "spectral" to use spectral components of the system matrix. The
         evaluation of the linear system is cheap and it provides valuable information
         about the underlying process. On the downside this mode has numerical issues
         if the system matrix is badly conditioned.
       * "matrix" to use system matrix directly. The evaluation of the system is more
         robust. The evaluation of the system is computationally more expensive.

    is_diagonalize
        If True, also the left eigenvectors are computed. This is more efficient to
        solve for initial conditions, because there is no least
        squares computation required for evaluating the linear dynamical
        system (see :class:`LinearDynamicalSystem`).

    rcond
        Parameter passed to :class:`numpy.linalg.lstsq`.

    kwargs_fd
        Keyword arguments, divergent to the default settings, passed to
        :py:meth:`.TSCAccessor.time_derivative`. Note that ``diff_order`` must be 1 and
        should not be included in the kwargs.

    Attributes
    ----------

    eigenvalues_ : numpy.ndarray
        Eigenvalues of Koopman generator matrix.

    eigenvectors_right_ : numpy.ndarray
        All right eigenvectors of Koopman generator matrix; ordered column-wise.

    eigenvectors_left_ : numpy.ndarray
        All left eigenvectors of Koopman generator matrix with ordered row-wise.
        Only accessible if ``is_diagonalize=True``.

    generator_matrix_ : numpy.ndarray
        Koopman generator matrix obtained from least squares. Only available if
        `store_generator_matrix=True` during fit.

    References
    ----------

    :cite:`klus-2020`

    """

    def __init__(
        self,
        *,  # keyword-only
        sys_mode: str = "spectral",
        is_diagonalize: bool = False,
        rcond: Optional[float] = None,
        kwargs_fd: Optional[dict] = None,
    ):
        self._setup_default_tsc_metric_and_score()
        self.is_diagonalize = is_diagonalize
        self.rcond = rcond
        self.kwargs_fd = kwargs_fd

        super(gDMDFull, self).__init__(
            sys_type="differential", sys_mode=sys_mode, time_invariant=True
        )

    def _compute_koopman_generator(self, X: TSCDataFrame, X_grad: TSCDataFrame):
        # X and X_grad are both in row-wise orientation
        X_numpy = X.to_numpy()
        X_grad_numpy = X_grad.to_numpy()

        # the maths behind it:  (X -- row-wise)
        # L X^T = \dot{X}^T         -- rearrange to standard lstsq problem
        # X L^T = \dot{X}           -- normal equations
        # X^T X L^T = X^T \dot{X}   -- solve for L^T

        data_sq = X_numpy.T @ X_numpy
        data_deriv = X_numpy.T @ X_grad_numpy

        generator = np.linalg.lstsq(data_sq, data_deriv, rcond=self.rcond)[0]

        # transpose to get L (in standard lstsq problem setting we solve for L^T)
        return generator.conj().T

    def _compute_spectral_components(self, generator_matrix_):
        eigenvalues_, eigenvectors_right_ = sort_eigenpairs(
            *np.linalg.eig(generator_matrix_)
        )

        eigenvectors_left_ = None
        if self.is_diagonalize:
            eigenvectors_left_ = self._compute_left_eigenvectors(
                system_matrix=generator_matrix_,
                eigenvalues=eigenvalues_,
                eigenvectors_right=eigenvectors_right_,
            )
        return eigenvectors_right_, eigenvalues_, eigenvectors_left_

    def _generate_fd_kwargs(self):

        ret_kwargs = copy.deepcopy(self.kwargs_fd) or {}

        if "diff_order" in ret_kwargs.keys():
            if self.kwargs_fd["diff_order"] != 1:
                raise ValueError(
                    f"'diff_order' must be 1 in kwargs_fd. "
                    f"Got diff_order={self.kwargs_fd['diff_order']}"
                )

        ret_kwargs.setdefault("diff_order", 1)
        ret_kwargs.setdefault("accuracy", 2)
        ret_kwargs.setdefault("shift_index", False)

        return ret_kwargs

    def fit(self, X: TimePredictType, *, U=None, y=None, **fit_params) -> "gDMDFull":
        """Compute Koopman generator matrix and spectral components.

        Parameters
        ----------
        X
            Training time series data.

        U: None
            ignored (the method does not support control input)

        y: None
            ignored

        **fit_params

            - store_generator_matrix
                If provided and True, then store the generator matrix separately in
                attribute `generator_matrix_`. The parameter is ignored if system mode
                is `matrix` (in this case the system matrix is available in
                ``sys_matrix_``).
        Returns
        -------
        gDMDFull
            self
        """

        self._validate_datafold_data(
            X=X,
            ensure_tsc=True,
            tsc_kwargs=dict(ensure_const_delta_time=True),
        )
        self._validate_and_setup_fit_attrs(X=X)

        store_generator_matrix = self._read_fit_params(
            attrs=[("store_generator_matrix", False)], fit_params=fit_params
        )

        kwargs_fd = self._generate_fd_kwargs()

        X_grad = X.tsc.time_derivative(**kwargs_fd)
        X = X.loc[X_grad.index, :]

        generator_matrix_ = self._compute_koopman_generator(X, X_grad)

        if self.is_spectral_mode:
            (
                eigenvectors_right_,
                eigenvalues_,
                eigenvectors_left_,
            ) = self._compute_spectral_components(generator_matrix_=generator_matrix_)

            self.setup_spectral_system(
                eigenvectors_right=eigenvectors_right_,
                eigenvalues=eigenvalues_,
                eigenvectors_left=eigenvectors_left_,
            )

            if store_generator_matrix:
                # store separately -- only for information
                # i.e. it is not used to solve the linear dynamical system.
                self.generator_matrix_ = generator_matrix_

        else:  # self.is_matrix_mode()
            self.setup_matrix_system(system_matrix=generator_matrix_)

        return self


class DMDEco(DMDBase):
    r"""Dynamic Mode Decomposition of time series data with prior singular value
    decomposition.

    The singular value decomposition (SVD) reduces the data and the Koopman operator is
    computed in this reduced space. This DMD model is particularly interesting for high
    dimensional data (large number of features), for example, solutions of partial
    differential equations (PDEs) with a fine grid.

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

      Alternatively, the eigenvectors can also be reconstructed with

      .. math::
          \Psi_r = U W ,

      which refers to the 'projected' version (see parameter).

    ...

    Parameters
    ----------
    svd_rank : int
        Number of eigenpairs to keep (largest eigenvalues in magnitude).

    reconstruct_mode : str
        Either 'exact' (default) or 'projected'.

    Attributes
    ----------

    eigenvalues_ : numpy.ndarray
        All eigenvalues of shape `(svd_rank,)` of the (reduced) Koopman matrix .

    eigenvectors_right_ : numpy.ndarray
        All right eigenvectors of shape `(svd_rank, svd_rank)` of the reduced Koopman
        matrix.

    References
    ----------

    :cite:`kutz-2016,tu-2014`
    """

    def __init__(self, svd_rank=10, *, reconstruct_mode: str = "exact"):
        self._setup_default_tsc_metric_and_score()
        self.svd_rank = svd_rank

        if reconstruct_mode not in ["exact", "projected"]:
            raise ValueError(
                f"reconstruct_mode={reconstruct_mode} must be in {['exact', 'projected']}"
            )
        self.reconstruct_mode = reconstruct_mode

        super(DMDEco, self).__init__(
            sys_type="flowmap", sys_mode="spectral", time_invariant=True
        )

    def _compute_internals(self, X: TSCDataFrame):
        # TODO: different orientations are good for different cases:
        #  1 more snapshots than quantities
        #  2 more quantities than snapshots
        #  Currently it is optimized for the case 2.

        shift_start, shift_end = X.tsc.shift_matrices(snapshot_orientation="col")
        U, S, Vh = np.linalg.svd(shift_start, full_matrices=False)  # (1.18)

        U = U[:, : self.svd_rank]
        S = S[: self.svd_rank]
        S_inverse = np.reciprocal(S, out=S)

        V = Vh.conj().T
        V = V[:, : self.svd_rank]

        koopman_matrix_low_rank = (
            U.T @ shift_end @ mat_dot_diagmat(V, S_inverse)
        )  # (1.20)

        eigenvalues_, eigenvectors_low_rank = np.linalg.eig(
            koopman_matrix_low_rank
        )  # (1.22)

        # As noted in the resource, there is also an alternative way
        # self.eigenvectors = U @ W

        if self.reconstruct_mode == "exact":
            eigenvectors_right_ = (
                shift_end @ V @ diagmat_dot_mat(S_inverse, eigenvectors_low_rank)
            )  # (1.23)
        else:  # self.reconstruct_mode == "projected"
            eigenvectors_right_ = U @ eigenvectors_low_rank

        return eigenvectors_right_, eigenvalues_, koopman_matrix_low_rank

    def fit(self, X: TimePredictType, *, U=None, y=None, **fit_params) -> "DMDEco":
        """Compute spectral components of Koopman matrix in low dimensional singular
        value coordinates.

        Parameters
        ----------
        X
            Training time series data.

        U: None
            ignored (the method does not support control input)

        y
            ignored

        **fit_params: Dict[str, object]
            None

        Returns
        -------
        DMDEco
            self
        """
        self._validate_datafold_data(
            X,
            ensure_tsc=True,
            tsc_kwargs=dict(ensure_const_delta_time=True),
        )
        self._validate_and_setup_fit_attrs(X)
        self._read_fit_params(attrs=None, fit_params=fit_params)

        eigenvectors_right_, eigenvalues_, koopman_matrix = self._compute_internals(X)

        self.setup_spectral_system(
            eigenvectors_right=eigenvectors_right_, eigenvalues=eigenvalues_
        )

        return self


class DMDControl(DMDBase):
    r"""Dynamic Mode Decomposition of time series data with control input to
    approximate the Koopman operator.

    The model computes the system and control matrices :math:`A` and :math:`B` from data
    (corresponding to mode ``matrix``)

    .. math::
        \mathbf{x}_{k+1} &= A \mathbf{x}_{k} + B \mathbf{u}_k

    where :math:`\mathbf{x}` are the system states and :math:`\mathbf{u}` the control input.

    if the system matrix is further decomposed into spectral terms
    (:math:`A \Psi = \Psi \Lambda`), then the system is described with

    .. math::
        \mathbf{x}_{k+1} &= \Psi \Lambda \mathbf{z}_{k} + B \mathbf{u}_k

    where :math:`\mathbf{z}_k` are the spectrally aligned system states
    (see :py:meth:`datafold.dynfold.dynsystem.LinearDynamicalSystem.compute_spectral_system_states`).

    ...

    Parameters
    ----------

    sys_mode
       Select the mode of how to evolve the linear system:

       * "spectral" to decompose the system matrix (`A`) into spectral components. The
         evaluation of the linear system is cheap and it provides valuable information about
         the identified system. On the downside this mode has numerical issues if the
         system matrix is badly conditioned.
       * "matrix" to use system matrix (`A`) directly. The evaluation of the system is more
         robust, but the system evaluation is computationally more expensive.

    rcond
        Cut-off ratio for small singular values.
        Passed to `rcond` of py:method:`numpy.linalg.lstsq`.

    Attributes
    -------
    sys_matrix : np.ndarray
        Koopman approximation of the state matrix

    control_matrix : np.ndarray
        Koopman approximation of the control matrix

    References
    ----------

    :cite:`kutz-2016` (Chapter 6)
    :cite:`korda-2018`
    """ # noqa

    def __init__(
        self,
        *,  # keyword-only
        sys_mode: str = "spectral",
        is_diagonalize=False,
        rcond: Optional[float] = None,
        **kwargs,
    ):

        self.is_diagonalize = is_diagonalize
        self.rcond = rcond
        super().__init__(
            sys_type="flowmap",
            sys_mode=sys_mode,
            is_controlled=True,
            time_invariant=True,
        )

    def _compute_koopman_and_control_matrix(
        self,
        X: TSCDataFrame,
        U: TSCDataFrame,
    ):
        Xm, Xp = X.tsc.shift_matrices(snapshot_orientation="row")
        Um = U.to_numpy()  # there is no need to apply shift matrices!

        if Xm.shape[1] > Xm.shape[0]:
            warnings.warn(
                "There are more observables than snapshots. The current implementation "
                "favors more snapshots than observables. This may result in a bad "
                "computational performance."
            )

        XU = np.vstack([Xm.T, Um.T])

        # from :cite:`korda-2018` - Eq. 22
        G = XU @ XU.T
        np.multiply(1 / X.shape[0], G, out=G)  # improve condition?
        V = Xp.T @ XU.T
        np.multiply(1 / X.shape[0], V, out=V)  # improve condition?

        # V = Mu @ G => V.T = G.T @ Mu.T
        MuT, residual, rank, _ = np.linalg.lstsq(G.T, V.T, rcond=self.rcond)
        if rank != G.shape[1]:
            warnings.warn(
                f"Shift matrix (shape={G.shape}) has not full rank (={rank}), falling "
                f"back to least squares solution. The sum of residuals is: "
                f"{np.sum(residual)}"
            )
        state_cols = Xm.shape[1]
        sys_matrix = MuT.conj().T[:, :state_cols]

        control_matrix = MuT.conj().T[:, state_cols:]

        return sys_matrix, control_matrix

    def fit(self, X: TSCDataFrame, *, U: TSCDataFrame, y=None, **fit_params) -> "DMDControl":  # type: ignore[override] # noqa
        """Fit model to approximate Koopman and control matrix.

        Parameters
        ----------
        X : TSCDataFrame
            Input state data

        U: None
            ignored (the method does not support control input)

        y: None
            ignored

        Returns
        -------
        DMDControl
            fitted model
        """

        self._validate_datafold_data(
            X=X,
            ensure_tsc=True,
            tsc_kwargs={"ensure_const_delta_time": True},
        )

        self._validate_datafold_data(
            X=U,
            ensure_tsc=True,
        )

        self._validate_and_setup_fit_attrs(X=X, U=U)
        self._read_fit_params(None, fit_params=fit_params)

        sys_matrix, control_matrix = self._compute_koopman_and_control_matrix(X, U)

        if self.sys_mode == "spectral":
            eigenvectors_right_, eigenvalues_, eigenvectors_left_ = compute_spectal_components(sys_matrix, is_diagonalize=self.is_diagonalize)

            self.setup_spectral_system(
                eigenvectors_right=eigenvectors_right_,
                eigenvalues=eigenvalues_,
                eigenvectors_left=eigenvectors_left_,
                control_matrix=control_matrix
            )
        else:
            self.setup_matrix_system(sys_matrix, control_matrix=control_matrix)

        return self


class gDMDAffine(DMDBase):
    r"""Dynamic mode decomposition of time series data with control input to
    approximate the Koopman generator for an input affine system.

    The model computes the system matrix :math:`A` and control tensor :math:`B` with

    .. math::
        X &= [x^{(1)} \ldots x^{(n)}] \\
        B &= [B_{e_1} \ldots B_{e_q}] \\
        \Psi &= \begin{bmatrix}
                x^{(1)}                & \ldots & x^{(n)} \\
                u^{(1)} \otimes x^{(1)}& \ldots & u^{(n)} \otimes x^{(n)} \\
                \end{bmatrix} \\
        \Psi &= [\dot{x}^{(1)} \ldots \dot{x}^{(n)}] \\
        [A,B] &= \dot{\Psi} (\Psi)^{\dagger},

    where :math:`X` is the data with :math:`n` column oriented snapshots,
    :math:`\dagger` is the Moore–Penrose inverse and
    :math:`\otimes` is the Kronecker product.

    The derivative :math:`\dot{x}` is computed via a finite-difference scheme as determined by
    the model arguments and passed to
    :py:meth:`datafold.pcfold.timeseries.accessor.TSCAccessor.time_derivative`. Samples are
    dropped for which no derivative are available.

    ...

    Parameters
    ----------
    diff_scheme
        The finite difference scheme 'backward', 'center' or 'forward'.
        Default is center.

    diff_accuracy
        The accuracy (even positive integer) of the derivative scheme.
        Default is 2. Passed to `accuracy` of
        :py:meth:`datafold.pcfold.timeseries.accessor.TSCAccessor.time_derivative`

    rcond
        Cut-off ratio for small singular values
        Passed to `rcond` of py:method:`numpy.linalg.lstsq`.

    Attributes
    -------
    sys_matrix_ : np.ndarray
        Approximation of the Koopman operator as the state matrix.

    control_matrix_ : np.ndarray
        Computed control matrix (as a tensor)

    References
    ----------

    :cite:`peitz-2020`
    """

    def __init__(
        self,
        *,  # keyword-only
        diff_scheme: str = "center",
        diff_accuracy: int = 2,
        rcond: Optional[float] = None,
    ):
        self.diff_scheme = diff_scheme
        self.diff_accuracy = diff_accuracy
        self.rcond = rcond
        super(gDMDAffine, self).__init__(
            sys_type="differential",
            sys_mode="matrix",
            is_controlled=True,
            is_control_affine=True,
        )

    def _compute_koopman_and_control_matrix(
        self,
        state: TSCDataFrame,
        control_inp: TSCDataFrame,
    ):
        Xdot_tsc = state.tsc.time_derivative(
            scheme=self.diff_scheme, accuracy=self.diff_accuracy
        )
        # trim samples where derivative is unknown
        # TODO: improve this by using pandas' indexing state and control_inp
        X = state.select_time_values(Xdot_tsc.time_values()).to_numpy()
        U = control_inp.select_time_values(Xdot_tsc.time_values()).to_numpy()
        Xdot = Xdot_tsc.to_numpy()

        n_snapshots = X.shape[0]
        state_cols = X.shape[1]
        control_cols = U.shape[1]
        if state_cols > n_snapshots:
            warnings.warn(
                "There are more observables than snapshots. The current implementation "
                "favors more snapshots than obserables. This may result in a bad "
                "computational performance."
            )

        # column-wise kronecker product
        u_x_cwise_kron = np.einsum("ij,ik->ijk", U, X).reshape(
            n_snapshots, control_cols * state_cols
        )

        # match naming convention from cite:`peitz-2020`
        Psi_XU = np.vstack([X.T, u_x_cwise_kron.T])
        Psidot_XU = Xdot.T

        # Solve via normal equations
        G = Psi_XU @ Psi_XU.T
        np.multiply(1 / n_snapshots, G, out=G)  # improve condition?
        V = Psidot_XU @ Psi_XU.T
        np.multiply(1 / n_snapshots, V, out=V)  # improve condition?

        # V = Mu @ G => V.T = G.T @ Mu.T
        MuT, residual, rank, _ = np.linalg.lstsq(G.T, V.T, rcond=self.rcond)

        if rank != G.shape[1]:
            warnings.warn(
                f"Shift matrix ({G.shape=}) has not full rank (={rank=}), falling "
                f"back to least squares solution. The sum of residuals is: "
                f"{np.sum(residual)=}"
            )

        sys_matrix = MuT.conj().T[:, :state_cols]
        control_matrix = MuT.conj().T[:, state_cols:]

        # reshape the matrix to a tensor
        control_tensor = control_matrix.reshape(
            state_cols, control_cols, state_cols
        ).swapaxes(1, 2)

        return sys_matrix, control_tensor

    # re-use function from DMDControl (but there is no inheritance relation)
    fit = DMDControl.fit  # type: ignore


class PyDMDWrapper(DMDBase):
    """A wrapper for dynamic mode decompositions models of Python package *PyDMD*.

    For further details of the underlying models please go to
    `PyDMD documentation <https://mathlab.github.io/PyDMD/>`__

    .. warning::

        The models provided by *PyDMD* can only deal with single time series. See also
        `github issue #86 <https://github.com/mathLab/PyDMD/issues/86>`__. This means that the
        input `X` in `fit` can only consist of one time series.

    .. warning::

        A main purpose of this wrapper is to use it for cross testing. The wrapper itself is
        not properly tested.

    Parameters
    ----------

    method
        Choose a method by string.

        - "dmd" - standard DMD
        - "hodmd" - higher order DMD
        - "fbdmd" - forwards backwards DMD
        - "mrdmd" - multi resolution DMD
        - "cdmd" - compressed DMD
        - "dmdc" - DMD with control

    svd_rank
        The rank of the singular value decomposition.
            - If `-1`: no truncation is performed (NOTE: the SVD is still performed)
            - If `0`: compute optimal rank.
            - A positive integer defines the actual rank.
            - A float between 0 and 1 defines the 'energy' of biggest singular value.

    tlsq_rank
        The rank of the total least squares. If 0, then no total least squares is applied.

    exact
        If True, perform the 'exact DMD', else a 'projected DMD'.

    opt
        If True, compute optimal amplitudes.

    init_params
        All further keyword arguments will be passed to the underlying model.

    References
    ----------

    :cite:`demo-2018`

    """

    def __init__(
        self,
        method: str,
        *,
        svd_rank: Union[int, float] = 0,
        tlsq_rank=0,
        exact: bool = False,
        opt: bool = False,
        **init_params,
    ):

        if not IS_IMPORTED_PYDMD:
            raise ImportError(
                "The optional Python package 'pydmd' (https://github.com/mathLab/PyDMD) "
                "could not be imported. Please check your installation or install "
                "with 'python -m pip install pydmd'."
            )
        else:
            assert pydmd is not None  # mypy

        self._setup_default_tsc_metric_and_score()
        self.method = method
        self.svd_rank = svd_rank
        self.tlsq_rank = tlsq_rank
        self.exact = exact
        self.opt = opt
        self.init_params = init_params

        # TODO: pydmd also provides the Koopman operator --> sys_mode="matrix" is also
        #  possible but requires implementation.
        super().__init__(sys_type="flowmap", sys_mode="spectral", time_invariant=True)

    def _setup_pydmd_model(self):

        # TODO: support HankelDMD, SpDMD, ParametricDMD ?

        if self.method == "dmd":
            self.dmd_ = pydmd.DMD(
                svd_rank=self.svd_rank,
                tlsq_rank=self.tlsq_rank,
                exact=self.exact,
                opt=self.opt,
            )
        elif self.method == "hodmd":
            self.dmd_ = pydmd.HODMD(
                svd_rank=self.svd_rank,
                tlsq_rank=self.tlsq_rank,
                exact=self.exact,
                opt=self.opt,
                **self.init_params,
            )

        elif self.method == "fbdmd":
            self.dmd_ = pydmd.FbDMD(
                svd_rank=self.svd_rank,
                tlsq_rank=self.tlsq_rank,
                exact=self.exact,
                opt=self.opt,
            )
        elif self.method == "mrdmd":
            self.dmd_ = pydmd.MrDMD(**self.init_params)
        elif self.method == "cdmd":
            self.dmd_ = pydmd.CDMD(
                svd_rank=self.svd_rank,
                tlsq_rank=self.tlsq_rank,
                opt=self.opt,
                **self.init_params,
            )

        elif self.method == "dmdc":
            self.dmd_ = pydmd.DMDc(
                svd_rank=self.svd_rank,
                tlsq_rank=self.tlsq_rank,
                opt=self.opt,
                **self.init_params,
            )
        else:
            raise ValueError(f"method={self.method} not known")

    def fit(
        self,
        X: TimePredictType,
        *,
        U: Optional[TSCDataFrame] = None,
        y=None,
        **fit_params,
    ) -> "PyDMDWrapper":
        """Compute Dynamic Mode Decomposition from data.

        Parameters
        ----------
        X
            Training time series data.

        U
            Control input time series. Only available for models that support control.

        y: None
            ignored

        **fit_params: Dict[str, object]
            None

        Returns
        -------
        PyDMDWrapper
            self
        """

        self._validate_datafold_data(
            X,
            ensure_tsc=True,
            tsc_kwargs=dict(ensure_const_delta_time=True),
        )
        self._validate_and_setup_fit_attrs(X=X)
        self._read_fit_params(attrs=None, fit_params=fit_params)

        self._setup_pydmd_model()

        if len(X.ids) > 1:
            raise ValueError(
                "The PyDMD package only works for single coherent time series. See \n "
                "https://github.com/mathLab/PyDMD/issues/86"
            )

        # data is column major
        if self.method == "dmdc":
            assert isinstance(self.dmd_, pydmd.dmdc.DMDc)
            assert U is not None
            self.dmd_.fit(X=X.to_numpy().T, I=U.to_numpy()[1:, :].T)
            # Pydmd does not support a .predict() method for controlled systems

        else:
            self.dmd_.fit(X=X.to_numpy().T)
            self.setup_spectral_system(
                eigenvectors_right=self.dmd_.modes,
                eigenvalues=self.dmd_.eigs,
            )

        return self


class StreamingDMD(DMDBase):
    r"""Dynamic mode decomposition on streaming data.

    Parameters
    ----------
    max_rank
        The maximal rank of the system matrix. If set to `None`, then there is no limit.

    ngram
        Number of Gram-Schmidt iterations.

    incr_basis_tol
        Tolerance of when to expand the basis,
        :math:`\vert\vert e \vert\vert / \vert \vert x \vert \vert`, where :math:`e` is the
        error from the Gram-Schmidt iterations and :math:`x` the new sample for the update.

    Attributes
    ----------
    A
        system matrix

    References
    ----------
    :cite:`hemati-2014`

    This implementation is adapted and extended from
    `dmdtools <https://github.com/cwrowley/dmdtools>`__ (for the compatible license see the
    `LICENSE_bundeled <https://gitlab.com/datafold-dev/datafold/-/blob/master/LICENSES_bundled>`__
    in datafold).
    """  # noqa E501

    # TODO: there is also an adaptation for total least squares (only Matlab):
    #   https://github.com/cwrowley/dmdtools/blob/master/matlab/StreamingTDMD.m

    def __init__(self, max_rank=None, ngram=5, incr_basis_tol=1.0e-10):
        self.max_rank = max_rank
        self.ngram = ngram
        self.incr_basis_tol = incr_basis_tol
        super(StreamingDMD, self).__init__(sys_type="flowmap", sys_mode="spectral")

    def _gram_schmidt(self, xm, xp):
        # TODO: can the Gram-Schmidt be replaced? Usually GS has numerical troubles

        # classical Gram-Schmidt re-orthonormalization
        rx = self._Qx.shape[1]
        ry = self._Qy.shape[1]
        xtilde = np.zeros([rx])
        ytilde = np.zeros([ry])

        em = xm.copy()
        ep = xp.copy()

        for i in range(self.ngram):
            dx = self._Qx.T @ em
            dy = self._Qy.T @ ep

            xtilde += dx
            ytilde += dy
            em -= self._Qx @ dx
            ep -= self._Qy @ dy

        return em, ep

    def _validate_parameter(self):
        if self.max_rank is not None:
            check_scalar(self.max_rank, name="max_rank", target_type=int, min_val=1)

        check_scalar(self.ngram, name="ngram", target_type=int, min_val=1)

        check_scalar(
            self.incr_basis_tol, "incr_basis_tol", target_type=float, min_val=0
        )

    def _increase_basis(self, em, norm_m, ep, norm_p):

        # ---- Algorithm step 2 ----
        # check basis for x and expand, if necessary
        if np.linalg.norm(em) / norm_m > self.incr_basis_tol:
            rx = self._Qx.shape[1]
            # update basis for x
            self._Qx = np.column_stack([self._Qx, em / np.linalg.norm(em)])
            # increase size of Gx and A (by zero-padding)
            self._Gx = np.block(
                [[self._Gx, np.zeros([rx, 1])], [np.zeros([1, rx + 1])]]
            )
            self.A = np.block([self.A, np.zeros([self.A.shape[0], 1])])

        # check basis for y and expand if necessary
        if np.linalg.norm(ep) / norm_p > self.incr_basis_tol:
            ry = self._Qy.shape[1]
            # update basis for y
            self._Qy = np.column_stack([self._Qy, ep / np.linalg.norm(ep)])
            # increase size of Gy and A (by zero-padding)
            self._Gy = np.block(
                [[self._Gy, np.zeros([ry, 1])], [np.zeros([1, ry + 1])]]
            )
            self.A = np.block([[self.A], [np.zeros([1, self.A.shape[1]])]])

    def _pod_compression(self):
        n_qx = self._Qx.shape[1]
        n_qy = self._Qy.shape[1]

        # check if compression is needed
        if self.max_rank is not None:
            if n_qx > self.max_rank:
                evals, evecs = np.linalg.eig(self._Gx)
                idx = np.argsort(evals)

                # indices of largest eigenvalues
                idx = idx[: self.max_rank]

                self._Qx = self._Qx @ evecs[:, idx]
                self.A = self.A @ evecs[:, idx]
                self._Gx = np.diag(evals[idx])
            if n_qy > self.max_rank:
                evals, evecs = np.linalg.eig(self._Gy)
                idx = np.argsort(evals)

                # indices of largest eigenvalues
                idx = idx[: self.max_rank]

                self._Qy = self._Qy @ evecs[:, idx]
                self.A = evecs[:, idx].T @ self.A
                self._Gy = np.diag(evals[idx])

    def _update_sys_matrix(self, Xm, Xp):
        # ---- Algorithm step 4 ----
        xtilde = self._Qx.T @ Xm
        ytilde = self._Qy.T @ Xp

        # update A and Gx
        self.A += np.outer(ytilde, xtilde)
        self._Gx += np.outer(xtilde, xtilde)
        self._Gy += np.outer(ytilde, ytilde)

    def _update(self, Xm, Xp, norm_m, norm_p):
        em, ep = self._gram_schmidt(Xm, Xp)
        self._increase_basis(em, norm_m, ep, norm_p)
        self._pod_compression()
        self._update_sys_matrix(Xm, Xp)

    def fit(self, X: TimePredictType, y=None, **fit_params) -> "DMDBase":
        """Initial fit of the model (used within :py:meth`partial_fit`).

        .. note::
            This function is not intended to be used directly. Use only py:meth:`partial_fit`
            for the initial fit and model updates
        """

        self._validate_datafold_data(
            X, ensure_tsc=True, tsc_kwargs=dict(ensure_n_timeseries=1)
        )

        if X.n_timesteps != 2:
            raise ValueError(
                "Only a single time series with two samples is permitted for the initial fit."
            )

        self._validate_and_setup_fit_attrs(X)

        s1, s2 = X.iloc[0, :].to_numpy(), X.iloc[1, :].to_numpy()

        norm_s1 = np.linalg.norm(s1)
        norm_s2 = np.linalg.norm(s2)

        self._Qx = if1dim_colvec(s1 / norm_s1)
        self._Qy = if1dim_colvec(s2 / norm_s2)

        # copy operations included to allocate new memory
        self._Gx = np.atleast_2d(np.square(norm_s1)).copy()
        self._Gy = np.atleast_2d(np.square(norm_s2)).copy()
        self.A = np.atleast_2d(norm_s1 * norm_s2).copy()

        return self

    def _separate_init_pairs(self, X: TSCDataFrame):

        X_init_fit, X_other = X.iloc[0:2, :], X.iloc[2:, :]
        X_init_fit.tsc.check_tsc(ensure_n_timeseries=1)

        if X.has_degenerate():
            raise ValueError(
                "The time series must contain time series with at least two "
                "samples per time series. Note that the very first time series (to "
                "fit the model initially) should contain either exactly two or "
                "more than three time samples."
            )

        return X_init_fit, X_other

    def _compute_koopman_matrix(self):
        # original: self.Qx.T @ self.Qy @ self.A @ np.linalg.pinv(self.Gx)
        # here I write it specifically as a linear regression
        # return self.Qx.T @ self.Qy @ self.A @ np.linalg.pinv(self.Gx)
        return np.linalg.lstsq(self._Gx.T, (self._Qx.T @ self._Qy @ self.A).T, rcond=0)[
            0
        ].T

    def _compute_spectral_components(self):
        Ktilde = self._compute_koopman_matrix()

        evals, right_evec = np.linalg.eig(Ktilde)
        evals, right_evec = sort_eigenpairs(evals, right_eigenvectors=right_evec)

        right_evec = self._Qx @ right_evec
        return right_evec, evals

    def partial_fit(self, X: TSCDataFrame, y=None, **fit_params) -> "StreamingDMD":
        """Perform a single epoch of updates on the system matrices on a given time series
        collection.

        Parameters
        ----------
        X
            Initial or new time series collection data.

        y
            ignored

        **fit_params
            None

        Returns
        -------
        StreamingDMD
            updated model
        """
        self._validate_parameter()
        self._read_fit_params(attrs=None, fit_params=fit_params)

        if not hasattr(self, "dt_"):  # initial fit
            X_init_fit, X = self._separate_init_pairs(X)
            self.fit(X_init_fit)

            if X.empty:
                return self
        else:
            self._validate_feature_names(X)
            self._validate_delta_time(X.delta_time)

        self._validate_datafold_data(X, ensure_tsc=True, ensure_min_samples=2)

        (
            Xm,
            Xp,
        ) = X.tsc.shift_matrices(snapshot_orientation="row")

        norm_m = np.linalg.norm(Xm, axis=1)
        norm_p = np.linalg.norm(Xp, axis=1)

        for i in range(Xm.shape[0]):
            self._update(Xm[i, :], Xp[i, :], norm_m[i], norm_p[i])

        # required to perform predictions:
        (
            eigenvectors_right,
            eigenvalues,
        ) = self._compute_spectral_components()

        self.setup_spectral_system(
            eigenvectors_right=eigenvectors_right, eigenvalues=eigenvalues
        )

        return self

    def predict(
        self,
        X: TimePredictType,
        *,
        U=None,
        time_values=None,
        **predict_params,
    ):
        """Predict time series data for each initial condition and time values.

        Parameters
        ----------
        X: TSCDataFrame, numpy.ndarray
            Initial conditions of shape `(n_initial_condition, n_features)`.

        time_values
            Time values to evaluate the model at. If not provided, then predict a single step
            from initial condition plus time in ``dt_``.

        Returns
        -------
        TSCDataFrame
            The computed time series predictions, where each time series has shape
            `(n_time_values, n_features)`.
        """

        check_is_fitted(self)
        X, _, time_values = self._validate_features_and_time_values(
            X, U=None, time_values=time_values
        )

        post_map, user_set_modes, feature_columns = self._read_predict_params(
            predict_params=predict_params
        )

        user_sys_matrix = self._read_user_sys_matrix(
            post_map=post_map, user_set_modes=user_set_modes
        )

        if user_sys_matrix is None:
            system_matrix = self.eigenvectors_right_
            _feat_names = self.feature_names_in_
        else:
            system_matrix = user_sys_matrix
            _feat_names = feature_columns

        X_predict = self._evolve_dmd_system(
            X_ic=X,
            overwrite_sys_matrix=system_matrix,
            control_input=None,
            time_values=time_values,
            feature_columns=_feat_names,
        )

        return X_predict


class OnlineDMD(DMDBase):
    r"""Online dynamic mode decomposition on time-varying system data.

    The system attributes become only available after a warm up phase with two times the
    number (see also ``ready_`` attribute).

    Parameters
    ----------

    weighting
        Exponential weighing factor in (0, 1] for adaptive learning rates. Smaller values
        allow for more adpative learning but can also result in instabilities as the model
        relies only on limited recent snapshots). Defaults to 1.0.

    is_diagonalize
        If True, also the left eigenvectors are computed to diagonalize the system matrix.
        This afftects of how initial conditions are adapted for the spectral system
        representation (instead of a least squares :math:`\Psi_r^\dagger x_0` with right
        eigenvectors it performs :math:`\Psi_l x_0`).

    Attributes
    ----------

    timestep_: int
        Counts the number of samples that have been processed.

    eigenvalues_ : numpy.ndarray
        Most recent eigenvalues of system matrix.

    eigenvectors_right_ : numpy.ndarray
        Most recent right eigenvectors of system matrix; ordered column-wise.

    eigenvectors_left_ : numpy.ndarray
        All left eigenvectors of Koopman matrix; ordered row-wise.
        Only available if ``is_diagonalize=True``.

    A: numpy.ndarray
        Most recent system matrix.

    References
    ----------
    :cite:`zhang-2019`

    This implementation is adapted and extended from
    `odmd <https://github.com/haozhg/odmd>`__ (for the compatible license see the
    `LICENSE_bundeled <https://gitlab.com/datafold-dev/datafold/-/blob/master/LICENSES_bundled>`__
    in datafold).
    """  # noqa E501

    # TODO for an implementation with control see
    #        https://github.com/VArdulov/online_dmd/blob/master/online_dmd/control.py

    def __init__(self, weighting: float = 1.0, is_diagonalize: bool = False) -> None:
        self.weighting = weighting
        self.is_diagonalize = is_diagonalize
        super(OnlineDMD, self).__init__(sys_type="flowmap", sys_mode="spectral")

    def _validate_parameters(self):
        check_scalar(
            self.weighting,
            name="weighing",
            target_type=float,
            min_val=0,
            max_val=1,
            include_boundaries="right",
        )

    def _compute_spectral_components(self):
        """Compute spectral components based on the current system matrix."""
        if not self.ready_:
            raise ValueError(
                f"Model has not seen enough data. Requires at least {2 * self.n_features_in_} "
                f"samples (currently {self.timestep_})."
            )
        evals, right_evec = np.linalg.eig(self.A)

        if self.is_diagonalize:
            left_evec = self._compute_left_eigenvectors(
                system_matrix=self.A,
                eigenvalues=evals,
                eigenvectors_right=right_evec,
            )
            return sort_eigenpairs(
                evals, right_eigenvectors=right_evec, left_eigenvectors=left_evec
            )
        else:
            evals, right_evec = sort_eigenpairs(evals, right_evec)
            return evals, right_evec, None

    def _basic_initialize(self, X) -> None:
        """Initialize online DMD with epsilon small (1e-15) ghost snapshot pairs before t=0"""
        self._validate_parameters()
        self._validate_and_setup_fit_attrs(X)
        n_states = X.shape[1]

        epsilon = 1e-15
        alpha = 1.0 / epsilon

        # TODO: maybe provide random_state in __init__?
        self.A = np.random.default_rng(1).normal(size=(n_states, n_states))
        self._P = alpha * np.identity(n_states)
        self.timestep_ = 0

    @property
    def ready_(self) -> bool:
        """Indicates if enough samples have been processed to perform predictions and
        access the spectral system components."""
        return self.timestep_ >= 2 * self.n_features_in_

    def fit(self, X, y=None, **fit_params):
        """Initialize the model with the first time series data in a batch.

        .. note::

            This function is not intended to be used directly. Use py:meth:`partial_fit` from
            the start (also initial fit).
        """

        self._validate_and_setup_fit_attrs(X)
        self._validate_parameters()
        X = self._validate_datafold_data(X, ensure_tsc=True)

        Xm, Xp = X.tsc.shift_matrices(snapshot_orientation="row")

        # necessary condition for over-constrained initialization
        p = Xp.shape[0]

        if self.weighting < 1:
            weights = np.power(np.sqrt(self.weighting), np.arange(p)[::-1])
            weights = if1dim_colvec(weights)
            Xmhat, Xphat = weights * Xm, weights * Xp
        else:
            Xmhat, Xphat = Xm, Xp

        # TODO: cannot find a way to not use pinv (lstsq leads to more unstable solutions)
        self.A = (np.linalg.pinv(Xmhat) @ Xphat).T

        # original np.linalg.inv(Xmhat @ Xmhat.T) / self.weighting
        self._P = (
            np.linalg.lstsq(Xmhat.T @ Xmhat, np.identity(X.shape[1]), rcond=0)[0]
            / self.weighting
        )

        self.timestep_ = p

        if self.ready_:
            (
                self.eigenvalues_,
                self.eigenvectors_right_,
                self.eigenvectors_left_,
            ) = self._compute_spectral_components()

        return self

    def partial_fit(self, X, y=None, **fit_params):
        """Perform a single epoch of updates on data to update the the system matrix and its
        spectral components.

        Parameters
        ----------
        X
            Initial or new time series collection data.

        y
            ignored

        **fit_params
            batch_initialize: bool
                If True then the entire initial batch is used to initialize the system matrix.
                Parameter is ignored if the model has been fitted once already.

        Returns
        -------
        OnlineDMD
            updated model
        """

        # TODO: all the checks etc. cost quite a lot of computational resources, if iterating
        #  on partial_fit sample-by-sample. Maybe this can be relaxed (disable validation with
        #  parameter) or also allow processing NumPy data directly (the user then needs to
        #  tell in which format the snapshots are -- i) single time series or ii) snapshot
        #  pairs       --- this is improved when needed
        # TODO: Another performance issue is, that every partial_fit the spectral components
        #  are computed. This may not be necessary however (they could be re-computed lazily if
        #  predict is called or if the attributes (eigenvalues eigenvectors) are accessed).
        #  Again, this may be integrated if needed.

        batch_initialize = self._read_fit_params(
            attrs=[("batch_initialize", False)], fit_params=fit_params
        )
        is_fitted = hasattr(self, "A")

        if batch_initialize and not is_fitted:
            return self.fit(X)
        elif not is_fitted:
            self._basic_initialize(X)

        self._validate_datafold_data(
            X,
            ensure_tsc=True,
            ensure_min_samples=2,
            tsc_kwargs=dict(ensure_no_degenerate_ts=True),
        )
        self._validate_delta_time(X.delta_time)
        self._validate_feature_names(X)

        Xm, Xp = X.tsc.shift_matrices(validate=False)
        n_states, n_pairs = Xm.shape

        for i in range(n_pairs):
            xm, xp = Xm[:, i], Xp[:, i]

            # compute P*xm matrix vector product beforehand
            Pxm = self._P @ xm

            # compute gamma
            gamma = np.reciprocal(1 + xm.T @ Pxm)

            # update A
            self.A += np.outer(gamma * (xp - self.A @ xm), Pxm)

            # update P, group Pxm*Pxm' to ensure positive definite
            self._P -= gamma * np.outer(Pxm, Pxm)
            self._P /= self.weighting

            # ensure P is SPD by taking its symmetric part
            # TODO: there is a function in datafold that performs this -- can it be avoided
            #  or only performed after the loop?
            self._P += self._P.T
            self._P /= 2

            # time step + 1
            self.timestep_ += 1

        if self.ready_:
            (
                eigenvalues,
                eigenvectors_right,
                eigenvectors_left,
            ) = self._compute_spectral_components()

            self.setup_spectral_system(
                eigenvectors_right=eigenvectors_right,
                eigenvalues=eigenvalues,
                eigenvectors_left=eigenvectors_left,
            )

        return self
