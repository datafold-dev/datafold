import abc
import copy
import warnings
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import scipy.linalg
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_scalar

from datafold._decorators import warn_experimental_class, warn_experimental_function
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


def compute_spectral_components(system_matrix, is_diagonalize):
    eigenvalues, eigenvectors_right = sort_eigenpairs(*np.linalg.eig(system_matrix))
    eigenvectors_right /= np.linalg.norm(eigenvectors_right, axis=0)

    if is_diagonalize:
        # Compute left eigenvectors such that
        #     system_matrix = eigenvectors_right_ @ diag(eigenvalues) @ eigenvectors_left_
        #
        #  NOTE:
        #     The left eigenvectors are
        #          * not normed
        #          * row-wise in returned matrix
        eigenvectors_left = np.linalg.solve(
            mat_dot_diagmat(eigenvectors_right, eigenvalues), system_matrix
        )
    else:
        eigenvectors_left = None

    return eigenvectors_right, eigenvalues, eigenvectors_left


class DMDBase(
    BaseEstimator, LinearDynamicalSystem, TSCPredictMixin, metaclass=abc.ABCMeta
):
    r"""Abstract base class for variations of the Dynamic Mode Decomposition (DMD).

    A DMD model decomposes a linearly identified dynamical system matrix --
    obtained from example time series data -- into intrinsic spatial-temporal components.
    Specifically, this corresponds to an eigendecomposition of the system matrix.
    Due to its connection to non-linear dynamical systems with the Koopman operator
    (see e.g. introduction in :cite:t:`tu-2014`), the DMD variants (in the subclasses)
    use the terminology of this.

    A DMD model approximates the Koopman operator with a matrix :math:`K`, which defines a
    linear dynamical system

    .. math::

        K^n x_0 &= x_n

    with :math:`x_n` being the (column) state vectors of the system at timestep :math:`n`.
    Note, that the state vectors :math:`x`, when used in conjunction with the
    :py:meth:`EDMD` model are not the original observations of a system, but states from a
    functional coordinate basis that seeks to linearize the dynamics (see reference for
    details).

    A :code:`DMDBase` subclass can either provide the matrix :math:`K`, the generator
    :math:`U` of :math:`K`

    .. math::
        U = \frac{K-I}{\Delta t}

    or the spectrum of either :math:`K` or :math:`U`. The spectrum of the Koopman matrix
    (or similar its generator) is obtained by solving the eigenproblem of the matrix

    .. math:: K \Psi_r = \Psi_r \Lambda

    where :math:`\Psi_r` are the right eigenvectors and :math:`\Lambda` a matrix with
    the eigenvalues on the diagonal. The spectral components enable further system analysis
    (e.g. stability, mode analysis) and lead also to an inexpensive system representation
    with powers of a diagonal matrix instead of a full matrix:

    .. math::
        x_n &= K^n x_0 \\
        &= K^n \Psi_r b_0  \\
        &= \Psi_r \Lambda^n b_0

    The DMD modes :math:`\Psi_r` - or eigenvectors - remain constant and the vector
    :math:`b_0` can be understood as a "spectrally adapted initial state", which is obtained
    from :math:`x_0` and the eigenvectors.

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
        self,
        X: TimePredictType,
        *,
        U: Optional[TSCDataFrame],
        P=None,
        y=None,
        **fit_params,
    ) -> "DMDBase":
        """Abstract method to train DMD model.

        Parameters
        ----------
        X
            Training time series data.

        U
            Control input (set to ``None`` by default if the subclass does not support
            control input).

        P
            ignored
                reserved for parameter input

        y
            ignored
              TODO for future devevelopment: This parameter is reserved to specify an
               extra map. Currently, this is already implemented for EDMD.

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

        if self.is_time_invariant:
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

        # correct the time shift according to the training data
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
                tsc_df.tsc.shift_time_by_delta(shift_t=shift).time_values()
                - time_values
                < 1e-14
            ).all()
        elif time_values.dtype == int:
            assert (
                tsc_df.tsc.shift_time_by_delta(shift_t=shift).time_values()
                - time_values
                == 0
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

    def predict(  # type: ignore
        self,
        X: InitialConditionType,
        *,
        U: Optional[TSCDataFrame] = None,  # type: ignore
        time_values: Optional[np.ndarray] = None,
        **predict_params,
    ) -> TSCDataFrame:  # type: ignore
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

        time_values = self._validate_and_set_time_values_predict(
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
            if isinstance(U, np.ndarray):
                if X.n_timeseries > 1:
                    raise NotImplementedError(
                        "If U is a numpy array, then only a prediction with "
                        "a single initial condition is allowed. "
                        f"Got {X.n_timeseries}"
                    )

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
        qois: Optional[Union[np.ndarray, pd.Index, list[str]]] = None,
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

            # use time_values from U_ic if available, else set time_values
            X_ts = self.predict(
                X=X_ic, U=U_ic, time_values=time_values if U is None else None
            )
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


class PretrainedDMD(DMDBase):
    def __init__(
        self,
        sys_type: Literal["flowmap", "differential"],
        sys_mode: Literal["spectral", "matrix"],
    ):
        super().__init__(sys_type=sys_type, sys_mode=sys_mode)

    @classmethod
    def from_available_system_matrix(
        cls,
        sys_type: Literal["flowmap", "differential"],
        sys_mode: Literal["matrix", "spectral"],
        system_matrix: np.ndarray,
        is_diagonalize: bool,
    ):
        dmd: LinearDynamicalSystem = PretrainedDMD(sys_type=sys_type, sys_mode=sys_mode)
        if sys_mode == "spectral":
            (
                eigenvectors_right,
                eigenvalues,
                eigenvectors_left,
            ) = compute_spectral_components(
                system_matrix=system_matrix, is_diagonalize=is_diagonalize
            )

            dmd = dmd.setup_spectral_system(
                eigenvectors_right=eigenvectors_right,
                eigenvalues=eigenvalues,
                eigenvectors_left=eigenvectors_left,
            )
        else:
            dmd = dmd.setup_matrix_system(system_matrix=system_matrix)

        return dmd

    def fit(
        self, X: TSCDataFrame, *, U: Optional[TSCDataFrame] = None, y=None, **fit_params
    ) -> "PretrainedDMD":
        self._validate_and_setup_fit_attrs(X, U=None)
        self._read_fit_params(attrs=None, fit_params=fit_params)

        # TODO: perform validation if dimensions with setup system are correct...
        return self


class DMDStandard(DMDBase):
    r"""Standard dynamic mode decomposition.

    The standard DMD computes a system matrix :math:`K` (which can be interpreted as an
    approximation of a Koopman operator approximation in a matrix) with

    .. math::
        K X &= X^{+} \\
        K &= X^{+} X^{\dagger},

    where :math:`X` is the data with column-oriented snapshots, :math:`\dagger`
    the Moore–Penrose inverse and :math:`+` the future time shifted data.

    The actual decomposition contains the spectral elements of the matrix :math:`K`.

    If the parameter ``rank`` is set, then an economic DMD is computed. For this the data
    :math:`X` is first represented in a singular value decomposition

      .. math::
          X \approx U_k \Sigma_k V_k^*

    with singular values in the diagonal matrix :math:`\Sigma` and vectors in :math:`U` and
    :math:`V`. Instead of using all components, only the leading `k` (corresponding to
    ``rank``) are used. Using this representation the system matrix is then computed with the
    reduced SVD coordinates

      .. math::
          K = U^T X' V_k \Sigma_k^{-1}

    Again the eigenpairs of the system matrix are computed :math:`K W_k = W_k \Omega`. There
    are two ways to reconstruct the eigendecomposition of the full system matrix

    1. ``reconstruct_mode="exact"``
        .. math::
            \Psi_r = X' V \Sigma^{-1} W
    2. ``reconstruct_mode="projected"``
        .. math::
            \Psi_r = U W

    ...

    Parameters
    ----------
    sys_mode
       Select a mode to evolve the linear system with

       * "spectral" to compute spectral components from the system matrix. The evaluation of
         the linear system is cheap and it provides valuable information about the
         underlying process. Predictions may be numerically corrupted If the system matrix is
         badly conditioned.
       * "matrix" to use system matrix directly. The evaluation of the system is more robust,
         but computationally more expensive.

    rank
        If set, the economic DMD is performed. The rank should be less than the number of
        features in the data.

    reconstruct_mode
        How to reconstruct the eigenvectors of the reduced system matrix (valid values are
        ``exact`` and ``project``). The parameter is ignored if ``rank is None``.

    diagonalize
        If True, the right and left eigenvectors are computed to diagonalize the system matrix.
        This affects how initial conditions are adapted for the spectral system
        representation (instead of a least squares :math:`\Psi_r^\dagger x_0` with right
        eigenvectors it performs :math:`\Psi_l x_0`). The parameter is ignored if
        ``sys_mode=matrix``.

    approx_generator
        If True, approximate the generator of the system

        * ``mode=spectral`` compute (complex) eigenvalues of the
          generator matrix :math:`log(\lambda) / \Delta t`, with eigenvalues `\lambda`
          of the system matrix. The left and right eigenvectors remain the same.
        * ``mode=matrix`` compute generator matrix with
          :math:`logm(K) / \Delta t`, where :math:`logm` is the matrix logarithm.

        .. warning::

            This operation can fail if the eigenvalues of the matrix :math:`K` are too
            close to zero or the matrix logarithm is ill-defined because of
            non-uniqueness. For details see :cite:t:`dietrich-2020` (Eq.
            3.2. and 3.3. and discussion). Currently, there are no counter measurements
            implemented to increase numerical robustness (work is needed). Consider
            also :py:class:`.gDMDFull`, which provides an alternative way to
            approximate the Koopman generator by using finite differences.

    rcond
        Cut-off ratio for small singular values
        Passed to `rcond` of py:method:`numpy.linalg.lstsq`.

    res_threshold
        Residual threshold to filter spurious spectral components. This follows Algorithm 2
        in :cite:t:`colbrook-2021`. If set, this requires ``sys_mode="spectral``.

    compute_pseudospectrum
        Flag to indicate whether the method ``pesudospectrum`` is required. If True, then
        additional (internal) matrices are stored that are required for the computations.

    Attributes
    ----------
    eigenvalues_ : numpy.ndarray
        Eigenvalues of Koopman matrix.

    eigenvectors_right_ : numpy.ndarray
        All right eigenvectors of Koopman matrix; ordered column-wise.

    eigenvectors_left_ : numpy.ndarray
        All left eigenvectors of Koopman matrix with ordered row-wise.
        Only accessible if ``is_diagonalize=True``.

    system_matrix_ : numpy.ndarray
        System matrix describing the discrete flow map obtained from linear system
        identification. Only available if ``store_system_matrix=True`` is set during fit and
        ``approx_generator=False``.

    generator_matrix_ : numpy.ndarray
        The generator matrix describing a vector field for the continuous system obtained by
        taking the logarithm of the system matrix (or is eigenvalues). Only available if
        ``store_system_matrix=True`` during fit and ``approx_generator=True``.

    References
    ----------
    * :cite:t:`schmid-2010` - DMD method in the original sense
    * :cite:t:`rowley-2009` - connects the DMD method to Koopman operator theory
    * :cite:t:`tu-2014` - generalizes the DMD to temporal snapshot pairs
    * :cite:t:`williams-2015` - generalizes the approximation to a lifted space
    * :cite:t:`kutz-2016` - an introductory book for DMD and its connection to the
      Koopman operator
    * :cite:t:`colbrook-2021` - residual DMD (ResDMD) and spectral properties of the
      Koopman operator
    """

    _valid_reconstruct_modes = ["exact", "project"]

    def __init__(
        self,
        *,  # keyword-only
        sys_mode: Literal["spectral", "matrix"] = "spectral",
        rank: Optional[int] = None,
        reconstruct_mode: Literal["exact", "project"] = "exact",
        diagonalize: bool = False,
        approx_generator: bool = False,
        rcond: Optional[float] = None,
        residual_filter: Optional[float] = None,
        compute_pseudospectrum: bool = False,
    ):
        self.rank = rank
        self.reconstruct_mode = reconstruct_mode
        self.diagonalize = diagonalize
        self.approx_generator = approx_generator
        self.rcond = rcond
        self.residual_filter = residual_filter
        self.compute_pseudospectrum = compute_pseudospectrum

        if residual_filter is not None and sys_mode != "spectral":
            raise ValueError(
                f'Residual computation requires sys_mode="spectral". '
                f"Got {sys_mode=}."
            )

        self._setup_default_tsc_metric_and_score()

        super().__init__(
            sys_type="differential" if self.approx_generator else "flowmap",
            sys_mode=sys_mode,
            is_time_invariant=True,
        )

    def _validate_parameters(self):
        if self.rank is not None:
            check_scalar(
                self.rank,
                "rank",
                target_type=int,
                min_val=1,
                max_val=self.n_features_in_,
            )

            if self.diagonalize:
                raise NotImplementedError(
                    f"Currently diagonalization ({self.diagonalize=}) is not "
                    f"supported for reduced ranks ({self.rank=}."
                )

    def _compute_full_system_matrix(self, X: TSCDataFrame, sample_weights=None):
        # It is more suitable to get the shift_start and shift_end in row orientation as
        # this is closer to the common least squares A K = B
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
                "favors more snapshots than observables. This may result in a bad "
                "computational performance.",
                stacklevel=2,
            )

        # see Eq. (13 a) and (13 b) in `williams_datadriven_2015`
        if sample_weights is None:
            # assume uniform sample weights
            G = shift_start_transposed.T @ shift_start_transposed
            G = np.multiply(1 / X.shape[0], G, out=G)

            G_dash = shift_start_transposed.T @ shift_end_transposed
            G_dash = np.multiply(1 / X.shape[0], G_dash, out=G_dash)
        else:
            G = shift_start_transposed.T @ diagmat_dot_mat(
                sample_weights, shift_start_transposed
            )
            G_dash = shift_start_transposed.T @ diagmat_dot_mat(
                sample_weights, shift_end_transposed
            )

        if self.residual_filter is not None:
            if sample_weights is None:
                _shiftYTY = shift_end_transposed.T @ shift_end_transposed
                R = np.multiply(1 / X.shape[0], _shiftYTY, out=_shiftYTY)
            else:
                R = shift_end_transposed.T @ diagmat_dot_mat(
                    sample_weights, shift_end_transposed
                )
        else:
            R = None

        from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

        alpha = 0
        l1_ratio = 0
        # TODO: integrate later also CV of each version (depending if user requests it)
        # TODO: need to also adapt the text above (note that here no transpose is necessary)
        if alpha == 0 and l1_ratio == 0:
            # If the matrix is square and of full rank, then 'koopman_matrix' is the exact
            # (numerical) solution of the linear system of equations.
            linregress_model = LinearRegression(fit_intercept=False)
        elif alpha > 0 and l1_ratio == 0:
            linregress_model = Ridge(alpha=alpha, fit_intercept=False)
        elif alpha == 0 and l1_ratio > 0:
            linregress_model = Lasso(fit_intercept=False)
        else:  # alpha > 0 and l1_ratio > 0:
            linregress_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

        linregress_model = linregress_model.fit(G, G_dash)
        koopman_matrix = linregress_model.coef_

        if linregress_model.rank_ != G.shape[1]:
            warnings.warn(
                f"Shift matrix ({G.shape=}) has not full rank ({linregress_model.rank_=}), "
                f"falling back to least squares solution.",
                stacklevel=2,
            )

        if self.residual_filter is None:
            # free memory and to make sure that they are not used again
            G, G_dash, R = [None] * 3

        reconstruct = None  # not required for the case with rank = None

        # TODO: return NamedTuple
        return koopman_matrix, G, G_dash, R, reconstruct

    def _compute_reduced_system_matrix(self, X, sample_weights):
        if sample_weights is not None:
            raise NotImplementedError(
                "sample_weights are currently not implemented for the reduced DMD"
            )

        shift_start, shift_end = X.tsc.shift_matrices(snapshot_orientation="col")

        U, S, Vh = np.linalg.svd(shift_start, full_matrices=False)  # (1.18)

        U = U[:, : self.rank]
        S = S[: self.rank]
        S_inverse = np.reciprocal(S, out=S)

        V = Vh.conj().T
        V = V[:, : self.rank]

        koopman_matrix_low_rank = (
            U.T @ shift_end @ mat_dot_diagmat(V, S_inverse)
        )  # (1.20)

        G = shift_start @ shift_start.T
        G_dash = shift_end @ shift_start.T
        R = shift_end @ shift_end.T

        reconstruct = dict(V=V, S_inverse=S_inverse, U=U, shift_end=shift_end)

        return koopman_matrix_low_rank, G, G_dash, R, reconstruct

    def _remove_spectral_pollution(
        self, eigenvalues, eigenvectors_right, eigenvectors_left, G, G_dash, R
    ):
        res_squared = np.ones(len(eigenvalues)) * np.inf

        for i, eigval, eigvec in zip(
            *(range(len(eigenvalues)), eigenvalues, eigenvectors_right.T)
        ):
            # Eq. 4.6 (page 18) in https://arxiv.org/pdf/2111.14889.pdf
            eigvec = if1dim_colvec(eigvec)
            eigvec_conj = eigvec.conj().T

            num = (
                R
                - eigval * G_dash.T
                - np.conj(eigval) * G_dash
                + np.square(np.abs(eigval)) * G
            )
            denom = np.real(eigvec_conj @ G @ eigvec)

            if np.abs(denom) > 1e-15:
                res_squared[i] = (np.real(eigvec_conj @ num @ eigvec)) / denom[0]

        residuals = np.sqrt(res_squared, out=res_squared)

        if isinstance(self.residual_filter, (float, int)):
            self.residual_filter = dict(threshold=self.residual_filter)
        elif not isinstance(self.residual_filter, dict):
            raise TypeError("residual_filter must be a float, int or dict.")

        # TODO: there is currently no validation if the user sets a wrong value
        if self.residual_filter.get("threshold", False):
            mask_keep = residuals <= self.residual_filter["threshold"]
        elif self.residual_filter.get("quantile", False):
            mask_keep = residuals <= np.quantile(
                residuals, self.residual_filter["quantile"]
            )
        elif self.residual_filter.get("keep_n", False):
            value = self.residual_filter["keep_n"]
            res_n = np.partition(residuals, value - 1)[value - 1]
            mask_keep = residuals <= res_n
        else:
            raise ValueError(
                f"{self.residual_filter=} is set incorrectly. Key must be 'threshold', "
                "'quantile' or 'keep_n' must be available"
            )

        if self.residual_filter.get("store_residuals", False):
            self.residuals_ = residuals
            self.residuals_mask_keep_ = mask_keep

        res_finite = np.isfinite(residuals)
        min_res, max_res = np.min(residuals[res_finite]), np.max(residuals[res_finite])
        median_res = np.median(residuals[res_finite])

        if np.all(~mask_keep):
            raise ValueError(
                "All system components are removed as spectral pollution. "
                f"Consider setting a higher threshold (currently {self.residual_filter=}).\n"
                f"{min_res=:.3e}, {max_res=:.3e} and {median_res=:.3e}"
            )
        elif np.all(mask_keep):
            warnings.warn(
                "No system component was removed as spectral pollution.\n"
                f"{min_res=:.3e}, {max_res=:.3e} and {median_res=:.3e}",
                stacklevel=2,
            )

        eigenvalues = eigenvalues[mask_keep]
        eigenvectors_right = eigenvectors_right[:, mask_keep]

        if eigenvectors_left is not None:
            eigenvectors_left = eigenvectors_left[mask_keep, :]

        if self.residual_filter.get("sort", False):
            sortidx = np.argsort(residuals[mask_keep])  # from low to high
            eigenvalues = eigenvalues[sortidx]
            eigenvectors_right = eigenvectors_right[:, sortidx]

            if eigenvectors_left is not None:
                eigenvectors_left = eigenvectors_left[sortidx, :]

        return eigenvalues, eigenvectors_right, eigenvectors_left

        # TODO: remove
        # if np.sqrt(np.real(res_squared)) <= self.res_threshold:
        #     new_vals.append(eval)
        #     new_rvecs.append(if1dim_colvec(g))
        #     if not (eigenvectors_left is None):
        #         new_lvecs.append(if1dim_rowvec(lvec))
        #
        # return (
        #     np.array(new_vals),
        #     (np.hstack(new_rvecs) if len(new_rvecs) else np.empty((0, 0))),
        #     (None if eigenvectors_left is None else np.vstack(new_lvecs)),
        # )

    def _compute_left_eigenvectors(
        self, system_matrix, eigenvalues, eigenvectors_right
    ):
        """Compute left eigenvectors such that
        system_matrix = eigenvectors_right_ @ diag(eigenvalues) @ eigenvectors_left_.

        .. note::
             The eigenvectors are

             * not normed
             * row-wise in returned matrix

        """
        lhs_matrix = mat_dot_diagmat(eigenvectors_right, eigenvalues)
        return np.linalg.solve(lhs_matrix, system_matrix)

    def _compute_spectral_components(self, system_matrix, reconstruct):
        eigenvalues_, eigenvectors_right_ = sort_eigenpairs(
            *np.linalg.eig(system_matrix)
        )

        if self.rank is not None:
            shift_end = reconstruct["shift_end"]
            V = reconstruct["V"]
            U = reconstruct["U"]
            S_inverse = reconstruct["S_inverse"]

            if self.reconstruct_mode == "exact":
                eigenvectors_right_ = (
                    shift_end @ V @ diagmat_dot_mat(S_inverse, eigenvectors_right_)
                )  # (1.23)
            else:  # self.reconstruct_mode == "projected"
                eigenvectors_right_ = U @ eigenvectors_right_

        if self.diagonalize:
            eigenvectors_left_ = self._compute_left_eigenvectors(
                system_matrix=system_matrix,
                eigenvalues=eigenvalues_,
                eigenvectors_right=eigenvectors_right_,
            )
        else:
            eigenvectors_left_ = None

        return eigenvectors_right_, eigenvalues_, eigenvectors_left_

    def pseudospectrum(
        self, grid: np.ndarray, regparam=1e-14, return_eigfuncs=False
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Compute the pseudospectrum of the Koopman operator according to
        Algorithm 3 in :cite:t:`colbrook-2021`.

        Parameters
        ----------
        grid  # TODO: default to the eigenvalues of DMD?
            Sampling grid on which to evaluate the minimum of the residual
            of candidate eigenvalues

        regparam
            Regularization parameter for the Gram matrix.

        return_eigfuncs
            If True, returns both the residuals and corresponding
            pseudoeigenfunctions, by default False

        Returns
        -------
        np.ndarray, Optional[np.ndarray]
            Residual at the grid points and (if requested) the corresponding
            pseudo eigenfunctions
        """
        if not self.compute_pseudospectrum:
            raise ValueError(
                "Calculating the pseudospectrum requires fitting"
                f"self.compute_pesudospectrum=True (got {self.compute_pesudospectrum=}) and"
                f"res_threshold is not None (got {self.residual_filter=}."
            )

        import warnings

        warnings.warn(
            "This is an experimental function and is not thoroughly tested.",
            stacklevel=1,
        )

        zs = grid.ravel()
        n_samples = zs.shape[0]

        G_adapt = self._G.copy()
        G_adapt.ravel()[:: G_adapt.shape[1] + 1] += np.linalg.norm(self._G) * regparam

        # SQ is approximate inverse of G  # TODO check this
        DG, VG = np.linalg.eigh(G_adapt)
        DG[DG != 0.0] = np.sqrt(1.0 / np.abs(DG[DG != 0.0]))
        SQ = VG @ diagmat_dot_mat(
            DG, VG.T
        )  # TODO: this should be symmetric? Currently it is corrupted by numerical noise

        residuals = np.inf * np.ones(n_samples)

        if return_eigfuncs:
            eigfuncs = np.zeros((self._G.shape[0], n_samples), dtype=complex)
        else:
            eigfuncs = None

        for i in range(n_samples):
            # print(f"{i}/{n_samples}")

            z = zs[i]
            try:
                # num = (L - z * A' - conj(z) * A + (abs(z)^2)*G)
                # RES= sqrt(real(eigs( SQ*num*SQ, 1, 'smallestabs')))

                num = (
                    self._R
                    - z * self._G_dash.T
                    - np.conj(z) * self._G_dash
                    + np.square(np.abs(z)) * self._G
                )

                # TODO: the original code only computes the smallest eigenvalue
                # in magnitude. Check shift-invere mode as the smallest
                # eigenvalues are tricky to find numerically stable
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html#scipy.sparse.linalg.eigsh # noqa
                # TODO: could improve by finding good starting vector from
                # previous computations? // v0=eigvec if i > 1 else None)
                eigval, eigvec = scipy.sparse.linalg.eigs(
                    SQ @ num @ SQ, k=1, sigma=0, which="LM", return_eigenvectors=True
                )

                if return_eigfuncs:
                    eigfuncs[:, i] = eigvec.ravel()

                # eigval, eigvec = np.linalg.eig(SQ @ num @ SQ)
                # ix_min = np.argmin(np.abs(eigval))
                residuals[i] = np.sqrt(np.real(eigval))

            except np.linalg.LinAlgError:
                # if eig did not converge for the residual computation at a grid point
                residuals[i] = np.inf

        if return_eigfuncs:
            return residuals.reshape(grid.shape), eigfuncs
        else:
            return residuals.reshape(grid.shape)

    def _calc_z(self, m):
        return 1 + 1j * ((2 * np.arange(1, m + 1)) / (m + 1) - 1)

    def _calc_c(self, m, z, eps):
        m = len(z)
        sigma = (1.0 / (1 + eps * np.conjugate(z)) - 1.0) / eps
        c = np.linalg.solve(np.vstack([sigma**i for i in range(m)]), np.eye(m, 1))
        return c

    def _calc_d(self, m, z):
        d = np.linalg.solve(np.vstack([z**i for i in range(m)]), np.eye(m, 1))
        return d

    @warn_experimental_function
    def spectral_measure(
        self,
        observable: np.ndarray,
        kernel_order: int,
        smoothing: float,
        evaluation_points: np.ndarray,
    ) -> np.ndarray:
        """Compute the approximate spectral measure corresponding to
        the koopman operator of the fitted system according to
        Algorthm 4 in :cite:t:`colbrook-2021`
        Requires that attribute ``res_threshold`` is set.

        Parameters
        ----------
        observable : np.ndarray
            Observable with respect to which to compute the spectral measure
            Should be projected into the dictionary space
            (see eq. 5.13 in :cite:t:`colbrook-2021`)
        kernel_order : int
            Order for the filtering kernel
        smoothing : float
            Filter parameter of the filtering kernel
        evaluation_points : np.ndarray
            Points between -pi and pi on which to evaluate the spectral measure

        Returns
        -------
        np.ndarray
            Discrete value of the spectral measure at the evaluation points
        """
        if self.residual_filter is None:
            raise AttributeError(
                "Calculating the spectral_measure requires fitting"
                " the data with res_threshold != None"
            )

        observable = if1dim_colvec(observable)
        if observable.shape != (self._G.shape[0], 1):
            raise ValueError(
                f"The shape of the projected observable {observable.shape} "
                f"does not match the observables during fit {(self._G.shape[0], 1)}."
            )
        observable /= np.linalg.norm(observable, 2)

        z = self._calc_z(kernel_order)
        d = self._calc_d(kernel_order, z)
        c = self._calc_c(kernel_order, z, smoothing)

        S, T, Q, Zstar = scipy.linalg.qz(self._G_dash, self._G)
        v1 = T @ Zstar @ observable
        Qstarg = Q.conjugate().T @ observable
        v2 = T.conjugate().T @ Qstarg
        v3 = S.conjugate().T @ Qstarg

        nu = np.zeros_like(evaluation_points)

        for k, theta in enumerate(evaluation_points):
            for j in range(kernel_order):
                lam = np.exp(1j * theta) * (1 + smoothing * z[j])
                Ij = scipy.linalg.solve((S - lam * T), v1)
                nu[k] -= np.real(
                    c[j] * np.conj(lam) * (Ij.conjugate().T @ v2)
                    + d[j] * (v3.conjugate().T @ Ij)
                ) / (2 * np.pi)

        return nu

    def fit(self, X: TimePredictType, *, U=None, y=None, **fit_params) -> "DMDStandard":
        """Fit model.

        Parameters
        ----------
        X
            Training time series data.

        y: None
            ignored

        U: None
            ignored (the method does not support control input)

        **fit_params

         - store_system_matrix: bool
            If True, the model stores either the system or generator matrix in attribute
            ``system_matrix_`` or ``generator_matrix_`` respectively. The parameter is ignored
            if ``sys_mode=="matrix"`` (the system matrix is then in attribute ``sys_matrix_``).
         - sample_weights: np.ndarray
            Sample weights

        Returns
        -------
        DMDStandard
            self
        """
        # TODO: note if rank is set, then the system matrix is only reduced!
        # TODO: need to validate all parameters properly

        self._validate_datafold_data(
            X=X,
            ensure_tsc=True,
            tsc_kwargs=dict(ensure_const_delta_time=True),
        )
        self._validate_and_setup_fit_attrs(X=X)
        self._validate_parameters()

        store_system_matrix, sample_weights = self._read_fit_params(
            attrs=[
                ("store_system_matrix", False),
                ("sample_weights", None),
            ],
            fit_params=fit_params,
        )

        if self.rank is not None:
            check_scalar(
                self.rank, "rank", target_type=int, min_val=1, max_val=X.shape[1] - 1
            )

        if self.reconstruct_mode not in self._valid_reconstruct_modes:
            raise ValueError(
                f"Valid arguments for 'reconstruct_mode' are {self._valid_reconstruct_modes}."
                f" Got {self.reconstruct_mode=}."
            )

        if self.rank is None:
            (
                system_matrix_,
                G,
                G_dash,
                R,
                reconstruct,
            ) = self._compute_full_system_matrix(X, sample_weights=sample_weights)
        else:
            (
                system_matrix_,
                G,
                G_dash,
                R,
                reconstruct,
            ) = self._compute_reduced_system_matrix(X, sample_weights=sample_weights)

        if self.is_spectral_mode:
            (
                eigenvectors_right_,
                eigenvalues_,
                eigenvectors_left_,
            ) = self._compute_spectral_components(system_matrix_, reconstruct)

            if self.residual_filter is not None:
                # TODO: by removing spectral components the (reconstructed) system matrix also
                #  changes -- should this be considered?
                (
                    eigenvalues_,
                    eigenvectors_right_,
                    eigenvectors_left_,
                ) = self._remove_spectral_pollution(
                    eigenvalues_,
                    eigenvectors_right_,
                    eigenvectors_left_,
                    G,
                    G_dash,
                    R,
                )

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
                        scipy.linalg.logm(system_matrix_) / self.dt_
                    )
                else:
                    self.system_matrix_ = system_matrix_

            if self.compute_pseudospectrum:
                self._G = G
                self._G_dash = G_dash
                self._R = R

        else:  # self.is_matrix_mode()
            if self.approx_generator:
                generator_matrix_ = scipy.linalg.logm(system_matrix_) / self.dt_
                self.setup_matrix_system(system_matrix=generator_matrix_)
            else:
                self.setup_matrix_system(system_matrix=system_matrix_)

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
        sys_mode: Literal["matrix", "spectral"] = "spectral",
        is_diagonalize: bool = False,
        rcond: Optional[float] = None,
        kwargs_fd: Optional[dict] = None,
    ):
        self._setup_default_tsc_metric_and_score()
        self.is_diagonalize = is_diagonalize
        self.rcond = rcond
        self.kwargs_fd = kwargs_fd

        super().__init__(
            sys_type="differential", sys_mode=sys_mode, is_time_invariant=True
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
        # TODO instead of having gDMDFull() rewrite this function:
        #       * rename to gDMDStandard
        #       * include a rank=None parameter
        #       * if rank is not None, then compute low rank Koopman generator in SVD
        #         coordinates
        #       * provide a way to steer finite difference method (consider also the Python
        #         package "derivative", which is used by PySINDy)

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
            ) = compute_spectral_components(
                system_matrix=generator_matrix_, is_diagonalize=self.is_diagonalize
            )

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


class DMDControl(DMDBase):
    r"""Dynamic Mode Decomposition with control input.

    The model computes the system and control matrices :math:`K` and :math:`B`

    .. math::
        \mathbf{x}_{k+1} = K \mathbf{x}_{k} + B \mathbf{u}_k

    where :math:`\mathbf{x}` are the system states and :math:`\mathbf{u}` the control input.

    If the system matrix is further decomposed into spectral terms
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
         the identified system. If the system matrix is badly conditioned this can lead to
         numerical issues.
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
    :cite:`proctor-2016`
    """  # noqa

    _requires_last_control_state = False

    def __init__(
        self,
        *,  # keyword-only
        sys_mode: Literal["matrix", "spectral"] = "matrix",
        rcond: Optional[float] = None,
        **kwargs,
    ):
        self.rcond = rcond
        super().__init__(
            sys_type="flowmap",
            sys_mode=sys_mode,
            is_controlled=True,
            is_time_invariant=True,
        )

    def _compute_koopman_and_control_matrix(
        self,
        X: TSCDataFrame,
        U: TSCDataFrame,
    ):
        Xm, Xp = X.tsc.shift_matrices(snapshot_orientation="row")
        Um = U.to_numpy()  # there is no need to apply the shift matrix

        if Xm.shape[1] > Xm.shape[0]:
            warnings.warn(
                "There are more observables than snapshots. The current implementation "
                "favors more snapshots than observables. This may result in a bad "
                "computational performance.",
                stacklevel=2,
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
                f"back to least squares solution.",
                stacklevel=2,
            )
        state_cols = Xm.shape[1]
        sys_matrix = MuT.conj().T[:, :state_cols]

        control_matrix = MuT.conj().T[:, state_cols:]

        return sys_matrix, control_matrix

    def fit(self, X: TSCDataFrame, *, U: TSCDataFrame, y=None, **fit_params) -> "DMDControl":  # type: ignore[override] # noqa
        """Fit model to compute a system and control matrix.

        Parameters
        ----------
        X : TSCDataFrame
            System state time series

        U: TSCDataFrame
            Control input. Each control input must have a matching system state in ``X``
            (same ID and time value) for all but the last state (i.e. all time series have one
            timestep less than the time series in ``X``).

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
            # need same dtype in time axis in U as in X
            tsc_kwargs=dict(
                ensure_dtype_time=X.index.get_level_values(
                    TSCDataFrame.tsc_time_idx_name
                ).dtype
            ),
            ensure_tsc=True,
        )

        self._validate_and_setup_fit_attrs(X=X, U=U)
        self._read_fit_params(None, fit_params=fit_params)

        sys_matrix, control_matrix = self._compute_koopman_and_control_matrix(X, U)

        if self.sys_mode == "spectral":
            (
                eigenvectors_right_,
                eigenvalues_,
                eigenvectors_left_,
            ) = compute_spectral_components(sys_matrix, is_diagonalize=True)

            self.setup_spectral_system(
                eigenvectors_right=eigenvectors_right_,
                eigenvalues=eigenvalues_,
                eigenvectors_left=eigenvectors_left_,
                control_matrix=control_matrix,
            )
        else:
            self.setup_matrix_system(sys_matrix, control_matrix=control_matrix)

        return self


@warn_experimental_class
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
    ----------
    sys_matrix_ : np.ndarray
        Approximation of the Koopman operator as the state matrix.

    control_matrix_ : np.ndarray
        Computed control matrix (as a tensor)

    References
    ----------
    :cite:`peitz-2020`
    """

    _requires_last_control_state = True

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
        super().__init__(
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
                "is more efficient for the case of more snapshots than observables. "
                "Implementation effort is required if the performance is too bad.",
                stacklevel=2,
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
                f"{np.sum(residual)=}",
                stacklevel=2,
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
    `PyDMD documentation <https://github.com/PyDMD/PyDMD>`__

    .. warning::

        The models provided by *PyDMD* can only deal with single time series. See also
        `github issue #86 <https://github.com/PyDMD/PyDMD/issues/86>`__. This means that the
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
            - If `-1`: no truncation is performed (NOTE: the SVD is still performed, just no
              components are discarded).
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
                "The optional Python package 'pydmd' (https://github.com/PyDMD/PyDMD) "
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
        super().__init__(
            sys_type="flowmap", sys_mode="spectral", is_time_invariant=True
        )

    def _setup_pydmd_model(self):
        # TODO: support other DMD variants?

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
                "https://github.com/PyDMD/PyDMD/issues/86"
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
    r"""Dynamic mode decomposition for streaming data.

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
    `dmdtools <https://github.com/cwrowley/dmdtools>`__ (for a copyright notice with the
    compatible BSD 3-Clause license see the `LICENSE_bundeled
    <https://gitlab.com/datafold-dev/datafold/-/blob/master/LICENSES_bundled>`__ file).
    """

    # TODO: there is also an adaptation for total least squares (only Matlab):
    #   https://github.com/cwrowley/dmdtools/blob/master/matlab/StreamingTDMD.m

    # TODO: IncrementalPCA has the attribute self.n_samples_seen_ -- can use here too
    # TODO: IncrementalPCA has a parameter check_input=True in "predict_fit" -- this should
    #  also be used here
    def __init__(self, max_rank=None, ngram=5, incr_basis_tol=1.0e-10):
        self.max_rank = max_rank
        self.ngram = ngram
        self.incr_basis_tol = incr_basis_tol
        super().__init__(sys_type="flowmap", sys_mode="spectral")

    def _gram_schmidt(self, xm, xp):
        # TODO: can the Gram-Schmidt be replaced? Usually GS has numerical troubles

        # classical Gram-Schmidt re-orthonormalization
        rx = self._Qx.shape[1]
        ry = self._Qy.shape[1]
        xtilde = np.zeros([rx])
        ytilde = np.zeros([ry])

        em = xm.copy()
        ep = xp.copy()

        for _i in range(self.ngram):
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

        # required to perform predictions
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
    """

    # TODO for an implementation with control see
    #        https://github.com/VArdulov/online_dmd/blob/master/online_dmd/control.py

    # TODO: IncrementalPCA has the attribute self.n_samples_seen_ -- can use here too
    # TODO: IncrementalPCA has a parameter check_input=True in "predict_fit" -- this should
    #  also be used here
    def __init__(
        self, weighting: float = 1.0, is_diagonalize: bool = False, with_warm_up=True
    ) -> None:
        self.weighting = weighting
        self.is_diagonalize = is_diagonalize
        self.with_warm_up = with_warm_up
        super().__init__(sys_type="flowmap", sys_mode="spectral")

    def _validate_parameters(self):
        check_scalar(
            self.weighting,
            name="weighing",
            target_type=(float, int),
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
        """Initialize online DMD with epsilon small (1e-15) ghost snapshot pairs before t=0."""
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
        access the spectral system components. Returns always True if ``with_warm_up=False``.
        """
        if self.with_warm_up:
            return self.timestep_ >= 2 * self.n_features_in_
        else:
            return True

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
            batch_fit: bool
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

        batch_fit = self._read_fit_params(
            attrs=[("batch_fit", False)], fit_params=fit_params
        )
        is_fitted = hasattr(self, "A")

        if batch_fit and not is_fitted:
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
