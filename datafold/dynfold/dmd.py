import abc
import copy
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.linalg
from pandas.api.types import is_datetime64_dtype, is_timedelta64_dtype
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Ridge, ridge_regression
from sklearn.utils.validation import check_is_fitted

from datafold._decorators import warn_experimental_class
from datafold.dynfold.base import InitialConditionType, TimePredictType, TSCPredictMixin
from datafold.pcfold import InitialCondition, TSCDataFrame, allocate_time_series_tensor
from datafold.utils.general import (
    diagmat_dot_mat,
    if1dim_colvec,
    is_scalar,
    mat_dot_diagmat,
    projection_matrix_from_features,
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
    r"""Evolve linear dynamical system forward in time.

    A mathematical description of a linear dynamical system is

    - differential
        .. math::
            \frac{d}{dt} x(t) = \mathcal{A} \cdot x(t),
            \mathcal{A} \in \mathbb{R}^{[m \times m]}

    This continuous system representation can also be written in terms of a
    discrete-time system

    - flowmap
        .. math::
            x_{n+1} = A \cdot x_{n}

    and :math:`A = \exp(\mathcal{A} \Delta t)`, a constant matrix, which describes the
    linear evolution of the systems' states :math:`x` with state length :math:`m`.

    Parameters
    ----------

    sys_type
        Type of linear system:

        * "differential"
        * "flowmap"

    sys_mode
        Whether the system is evaluted with

        * "matrix" (i.e. :math:`A` or :math:`\mathcal{A}` are given)
        * "spectral" (i.e. eigenpairs of :math:`A` or :math:`\mathcal{A}` are given)

    time_invariant
        If True, the system internally always starts with `time=0`. \
        This is irrespective of the time given in the time values. If the initial
        time is larger than zero, the internal times are corrected to the requested time.

    References
    ----------

    :cite:`kutz_dynamic_2016` (pages 3 ff.)

    """

    _cls_valid_sys_type = ("differential", "flowmap")
    _cls_valid_sys_mode = ("matrix", "spectral")

    def __init__(self, sys_type: str, sys_mode: str, time_invariant: bool = True):
        self.sys_type = sys_type
        self.sys_mode = sys_mode
        self.time_invariant = time_invariant

        self._check_system_type()
        self._check_system_mode()

    def _check_system_type(self) -> None:
        if self.sys_type not in self._cls_valid_sys_type:
            raise ValueError(
                f"system_type={self.sys_type} is invalid. "
                f"Choose from {self._cls_valid_sys_type}"
            )

    def _check_system_mode(self) -> None:
        if self.sys_mode not in self._cls_valid_sys_mode:
            raise ValueError(
                f"'sys_mode={self.sys_mode}' is invalid."
                f"Choose from {self._cls_valid_sys_mode}"
            )

    def _check_and_set_system_params(
        self,
        sys_matrix: Optional[np.ndarray],
        initial_condition: np.ndarray,
        time_values: np.ndarray,
        time_delta: Optional[Union[float, int]],
        time_series_ids,
        feature_names_out,
    ):
        # SYSTEM MATRIX
        if sys_matrix is None:
            # The system matrix can be overwritten from outside
            # (if sys_matrix is not None).
            # This is particularily useful when A is the system matrix,
            # but there is a post-map, e.g. D @ A.
            if self.is_spectral_mode():
                sys_matrix = self.eigenvectors_right_
            else:  # self.is_matrix_mode()
                sys_matrix = self.sys_matrix_

        if not isinstance(sys_matrix, np.ndarray) or sys_matrix.ndim != 2:
            raise ValueError("'sys_matrix' must be 2-dim. and of type np.ndarray")

        n_features, state_length = sys_matrix.shape

        # INITIAL CONDITION
        try:
            if is_scalar(initial_condition):
                initial_condition = [initial_condition]
            initial_condition = np.asarray(initial_condition)
        except:
            raise TypeError(
                "Parameter ic must be be an array like object. "
                f"Got type(ic)={type(initial_condition)}"
            )

        if initial_condition.ndim == 1:
            initial_condition = if1dim_colvec(initial_condition)

        if initial_condition.ndim != 2:
            raise ValueError(  # in case ndim > 2
                f"Initial conditions 'ic' must have 2 dimensions. "
                f"Got ic.ndim={initial_condition.ndim}."
            )

        if initial_condition.shape[0] != state_length:
            raise ValueError(
                f"Mismatch in dimensions between initial condition and system matrix. "
                f"ic.shape[0]={initial_condition.shape[0]} is not dynmatrix.shape[1]={state_length}."
            )

        # TIME VALUES
        try:
            if is_scalar(time_values):
                time_values = [time_values]
            time_values = np.asarray(time_values)
        except:
            raise TypeError(
                "The parameter 'time_values' must be an array-like object. "
                f"Got type(time_values)={time_values}"
            )

        # see https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
        if time_values.ndim != 1 and time_values.dtype.kind in "buif":
            raise ValueError(
                "The array must be 1-dim. and only contain real-valued numeric data."
            )

        if is_timedelta64_dtype(time_values) or is_datetime64_dtype(time_values):
            time_values = time_values.astype(int)

        # TIME DELTA
        if self.is_differential_system():
            # for a differential system there is no time_delta -- all input is ignored
            time_delta = None
        elif time_delta is None or not is_scalar(time_delta):
            raise TypeError(
                "In a 'flowmap' system the parameter 'time_delta' must be provided "
                f"and a scalar value. Got type(time_delta)={type(time_delta)}"
            )
        else:
            assert time_delta is not None  # mypy
            # cast to built-in Python value
            if np.asarray(time_delta).dtype.kind in "mli":
                time_delta = int(time_delta)
            else:
                time_delta = float(time_delta)
            if time_delta <= 0:
                raise ValueError(f"time_delta={time_delta} must be positive.")

        # TIME SERIES IDS and FEATURE COLUMNS
        # Note: all the other checks are made during TSCDataFrame allocation.
        if time_series_ids is None:
            time_series_ids = np.arange(initial_condition.shape[1])

        if feature_names_out is None:
            feature_names_out = np.arange(state_length)

        if len(feature_names_out) != n_features:
            raise ValueError(
                f"len(feature_columns)={feature_names_out} != state_length={state_length}"
            )

        return (
            sys_matrix,
            initial_condition,
            time_values,
            time_delta,
            n_features,
            state_length,
            time_series_ids,
            feature_names_out,
        )

    def _evolve_system_states(
        self,
        time_series_tensor: np.ndarray,
        sys_matrix: np.ndarray,
        initial_conditions: np.ndarray,
        time_values: np.ndarray,
        time_delta: Optional[Union[float]],
    ) -> np.ndarray:

        if self.is_spectral_mode():
            # NOTE: The code can be optimized, but the current version is better readable
            # and so far no computational problems were encountered.
            if self.is_differential_system():
                for idx, time in enumerate(time_values):
                    time_series_tensor[:, idx, :] = np.real(
                        sys_matrix
                        @ diagmat_dot_mat(
                            np.exp(self.eigenvalues_ * time), initial_conditions
                        )
                    ).T

            elif self.is_flowmap_system():  # self.system_type == "flowmap":
                # Usually, for a differential system the eigenvalues are written as:
                # omegas = np.log(eigenvalues.astype(complex)) / time_delta
                # --> evolve system with
                #               exp(omegas * t)
                # because this matches the notation of the differential system.

                # An disadvantage is, that it requires the complex logarithm, which for
                # complex (eigen-)values can happen to be not well-defined.

                # A numerical more stable way is:
                # exp(omegas * t)
                # exp(log(ev) / time_delta * t)
                # exp(log(ev^(t/time_delta)))  -- logarithm rules
                # --> evolve system with, using `float_power`
                #               ev^(t / time_delta)

                _eigenvalues = self.eigenvalues_.astype(complex)

                for idx, time in enumerate(time_values):
                    time_series_tensor[:, idx, :] = np.real(
                        sys_matrix
                        @ diagmat_dot_mat(
                            np.float_power(_eigenvalues, time / time_delta),
                            initial_conditions,
                        )
                    ).T
            else:
                self._check_system_type()

        elif self.is_matrix_mode():
            # TODO: computational aspects:
            #  - treat equidistant sampling differently? Then the system can be
            #    iterated more efficiently, bc. scipy.linalg.expm(sys_matrix *
            #    time_delta) has to be only computed once, and can be iterated given
            #    the previous solution.
            #  - how is the fractional_matrix_power implemented? It computes internally
            #    singular values, so it'd be better to avoid calling it too often, if
            #    possible
            #    see: https://github.com/scipy/scipy/blob/c1372d8aa90a73d8a52f135529293ff4edb98fc8/scipy/linalg/_matfuncs_inv_ssq.py

            if self.is_differential_system():
                for idx, time in enumerate(time_values):
                    time_series_tensor[:, idx, :] = np.real(
                        scipy.linalg.expm(sys_matrix * time) @ initial_conditions
                    ).T

            elif self.is_flowmap_system():
                for idx, time in enumerate(time_values):
                    # TODO: this is really expensive -- can also store intermediate
                    #  results and only add the incremental fraction?
                    time_series_tensor[:, idx, :] = np.real(
                        scipy.linalg.fractional_matrix_power(
                            sys_matrix, time / time_delta
                        )
                        @ initial_conditions
                    ).T
            else:
                # Something is really wrong
                self._check_system_type()

        else:
            self._check_system_mode()

        return time_series_tensor

    def compute_spectral_system_states(self, states) -> np.ndarray:
        r"""Compute the spectral states of the system.

        If the linear system is written in its spectral form:

        .. math::
            \Psi_r \Lambda^n \Psi_l x_0 &= x_n \\
            \Psi_r \Lambda^n b_0 &= x_n \\

        then `b_0` is the spectral state, which is computed in this function. It does
        not necessarily need to be an initial state but instead can be arbitrary states.

        In the context of dynamic mode decomposition, the spectral state is also often
        referred to as "amplitudes". E.g., see :cite:`kutz_dynamic_2016`, page 8. In
        the context of `EDMD`, where the DMD model acts on a dictionary space, then the
        spectral states are the evaluation of the Koopman eigenfunctions. See e.g.,
        :cite:`williams_datadriven_2015` Eq. 3 or 6.

        There are two alternatives in how to compute the states.

        1. By using the right eigenvectors and solving in a least square sense

            .. math::
                \Psi_r b_0 = x_0

        2. , or by using the left eigenvectors and computing the matrix-vector
          product

            .. math::
                \Psi_l x_0 = b_0

        If the left eigenvectors where set during :py:meth:`.setup_sys_spectral`,
        then alternative 2 is used always and otherwise alternative 1.

        Parameters
        ----------
        states
            The states of original data space in column-orientation.

        Returns
        -------
        numpy.ndarray
            Transformed states.
        """

        if not self.is_spectral_mode():
            raise AttributeError(
                f"To compute the spectral system states sys_mode='spectral' is required. "
                f"Got self.sys_mode={self.sys_mode}"
            )

        # Choose between two alternatives:
        if hasattr(self, "eigenvectors_left_") and (
            self.eigenvectors_left_ is not None and self.eigenvectors_right_ is not None
        ):
            # uses both eigenvectors (left and right).
            # this is Eq. 18 in :cite:`williams_datadriven_2015` (note that in the
            # paper the Koopman matrix is transposed, therefore here left and right
            # eigenvectors are exchanged.
            states = self.eigenvectors_left_ @ states
        elif (
            hasattr(self, "eigenvectors_right_")
            and self.eigenvectors_right_ is not None
        ):
            # represent the initial condition in terms of right eigenvectors (by solving a
            # least-squares problem)
            # -- in this case only the right eigenvectors are required
            states = np.linalg.lstsq(self.eigenvectors_right_, states, rcond=None)[0]
        else:
            raise ValueError(
                f"Attribute 'eigenvectors_right_ is None'. Please report bug."
            )

        return states

    def is_matrix_mode(self) -> bool:
        r"""Whether the set up linear system is in "matrix" mode.

        The system uses either matrix :math:`A` for flowmap or :math:`\mathcal{A}`
        for differential.

        Returns
        -------

        """
        self._check_system_mode()
        return self.sys_mode == "matrix"

    def is_spectral_mode(self) -> bool:
        r"""Whether the set up linear system is in "spectral" mode.

        The system uses the spectral components of either matrix :math:`A` for flowmap or
        :math:`\mathcal{A}` for differential.

        Returns
        -------

        """
        self._check_system_mode()
        return self.sys_mode == "spectral"

    def is_differential_system(self) -> bool:
        r"""Whether the set up linear system is of "differential" type.

        The system uses either the matrix :math:`\mathcal{A}` directly or its spectral
        components to evolve the system.

        Returns
        -------

        """
        self._check_system_type()
        return self.sys_type == "differential"

    def is_flowmap_system(self) -> bool:
        r"""Whether the set up linear system is of "flowmap" type.

        The system uses either the matrix :math:`A` directly or its spectral
        components to evolve the system.

        Returns
        -------

        """
        self._check_system_type()
        return self.sys_type == "flowmap"

    def is_linear_system_setup(self, raise_error_if_not_setup: bool = False) -> bool:
        """Whether the set up linear system is set up.

        Returns
        -------

        """

        if self.is_matrix_mode():
            is_setup = hasattr(self, "sys_matrix_")
        else:  # self.is_spectrum_mode():
            is_setup = hasattr(self, "eigenvectors_right_") and hasattr(
                self, "eigenvalues_"
            )
        if not is_setup and raise_error_if_not_setup:
            raise RuntimeError("Linear system has not been setup.")
        else:
            return is_setup

    def setup_spectral_system(
        self, eigenvectors_right, eigenvalues, eigenvectors_left=None
    ) -> "LinearDynamicalSystem":
        r"""Set up linear system with spectral components of system matrix.

        If the left eigenvectors (attribute :code:`eigenvectors_left_`) are available the
        initial condition always solves with the second case for :math:`b_0` in note of
        :py:meth:`.evolve_linear_system` because this is more efficient.

        Parameters
        ----------
        eigenvectors_right
            The right eigenvectors :math:`\Psi_r` of system matrix.

        eigenvalues
            The eigenvalues :math:`\Lambda` of system matrix.

        eigenvectors_left
            The left eigenvectors :math:`\Psi_l` of system matrix.

        Returns
        -------
        LinearDynamicalSystem
            self
        """

        if not self.is_spectral_mode():
            raise RuntimeError(
                f"The 'sys_mode' was set to {self.sys_mode}. Cannot setup "
                f"system with spectral."
            )

        if self.is_linear_system_setup():
            raise RuntimeError("Linear system is already setup.")

        self.eigenvectors_right_ = eigenvectors_right
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_left_ = eigenvectors_left
        return self

    def setup_matrix_system(self, system_matrix):
        r"""Set up linear system with system matrix.

        Parameters
        ----------
        system_matrix
            The system matrix (either :math:`A` for flowmap or :math:`\mathcal{A}` for
            differential type).

        Returns
        -------
        LinearDynamicalSystem
            self
        """
        if self.is_linear_system_setup():
            raise RuntimeError("Linear system is already setup.")

        self.sys_matrix_ = system_matrix
        return self

    def evolve_system(
        self,
        initial_conditions: np.ndarray,
        time_values: np.ndarray,
        overwrite_sys_matrix: Optional[np.ndarray] = None,
        time_delta: Optional[float] = None,
        time_series_ids: Optional[np.ndarray] = None,
        feature_names_out: Optional[Union[pd.Index, list]] = None,
    ):
        r"""Evolve specified linear dynamical system.

        The system evolves depending on the system mode (matrix or spectral) and
        depending on the system type (differential or flowmap). In all cases the time
        values can be positive real values :math:`t \in \mathbb{R}^+`.

        * **matrix** -- Using the system matrix directly.

          - differential
            The system is evaluated with the analytical solution of a linear dynamical
            system by using the matrix exponential

            .. math::
                x(t) = \exp(\mathcal{A \cdot t}) x(0)

          - flowmap
            The system is evaluated with `matix_fractional_power`

            .. math::
                x(t) = A^{t / \delta t} \cdot x_0

        * **spectral** -- Using the eigenvalues in the diagonal matrix :math:`\Lambda`
          and (right) eigenvectors :math:`\Psi_r` of the constant matrix
          :math:`\mathcal{A}` in the differential case or :math:`A` in the flowmap case
          (see definitions in class description).

          - differential
                The system is evaluated with the analytical solution of a linear dynamical
                system by using the exponential of eigenvalues

                .. math::
                    x(t) = \Psi \cdot \exp(\Lambda \cdot t) \cdot b(0)

          - flowmap
                Non-integer values are interpolated with ``float_power``. For this case
                the parameter `time_delta` must be provided.

                .. math::
                    x(t) = \Psi \cdot \Lambda^{t / \delta t}) \cdot b_0

          where :math:`b(0)` and :math:`b_{0}` are the initial conditions of the
          respective system.

          .. note::
              Contrasting to the `matrix` case, the initial condition states
              :math:`x_0` of the original system need to be aligned to the right
              eigenvectors beforehand. See :py:meth:`.compute_spectral_system_states`

        Parameters
        ----------
        initial_conditions
            Single initial condition of shape `(n_features,)` or multiple initial
            conditions of shape `(n_features, n_initial_conditions)`.

        time_values
           Time values to evaluate the linear system at :math:`t \in \mathbb{R}^{+}`

        overwrite_sys_matrix
            Primarily for performance reasons the a system matrix :math:`A` can also be
            overwritten. An example is to include perform a projection matrix :math:`P`
            to only return some quantities of interest :math:`A^{*} = P \cdot A`

        time_delta
            Time delta :math:`\delta t` for reference. This is a required parameter in a
            "flowmap" system.

        time_series_ids
           Unique integer time series IDs of shape `(n_initial_conditions,)` for each \
           respective initial condition. Defaults to `(0, 1, 2, ...)`.

        feature_names_out
            Unique feature columns names of shape `(n_features,)`.
            Defaults to `(0, 1, 2, ...)`.

        Returns
        -------
        TSCDataFrame
            Collection with a time series for each initial condition with \
            shape `(n_time_values, n_features)`.
        """

        (
            sys_matrix,
            initial_conditions,
            time_values,
            time_delta,
            n_features,
            state_length,
            time_series_ids,
            feature_names_out,
        ) = self._check_and_set_system_params(
            sys_matrix=overwrite_sys_matrix,
            initial_condition=initial_conditions,
            time_values=time_values,
            time_delta=time_delta,
            time_series_ids=time_series_ids,
            feature_names_out=feature_names_out,
        )

        time_series_tensor = allocate_time_series_tensor(
            n_time_series=initial_conditions.shape[1],
            n_timesteps=time_values.shape[0],
            n_feature=n_features,
        )

        time_series_tensor = self._evolve_system_states(
            time_series_tensor=time_series_tensor,
            sys_matrix=sys_matrix,
            initial_conditions=initial_conditions,
            time_values=time_values,
            time_delta=time_delta,
        )

        return TSCDataFrame.from_tensor(
            time_series_tensor,
            time_series_ids=time_series_ids,
            columns=feature_names_out,
            time_values=time_values,
        )


class DMDBase(
    BaseEstimator, LinearDynamicalSystem, TSCPredictMixin, metaclass=abc.ABCMeta
):
    r"""Abstract base class for Dynamic Mode Decomposition (DMD) models.

    A DMD model decomposes time series data linearly into spatial-temporal components.
    The decomposition defines a linear dynamical system. Due to it's strong connection to
    non-linear dynamical systems with Koopman spectral theory
    (see e.g. introduction in :cite:`tu_dynamic_2014`), the DMD variants (subclasses)
    are framed in the context of this theory.

    A DMD model approximates the Koopman operator with a matrix :math:`K`,
    which defines a linear dynamical system

    .. math:: K^n x_0 &= x_n

    with :math:`x_n` being the (column) state vectors of the system at time :math:`n`.
    Note, that the state vectors :math:`x`, when used in conjunction with the
    :py:meth:`EDMD` model are not the original observations of a system, but states from a
    functional coordinate basis that seeks to linearize the dynamics (see reference for
    details).

    A subclass can either provide :math:`K`, the spectrum of :math:`K` or the generator
    :math:`U` of :math:`K`

    .. math::
        U = \frac{K-I}{\delta t}

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
    :cite:`schmid_dynamic_2010` - DMD method in the original sense
    :cite:`rowley_spectral_2009` - connects the DMD method to Koopman operator theory
    :cite:`tu_dynamic_2014` - generalizes the DMD to temporal snapshot pairs
    :cite:`williams_datadriven_2015` - generalizes the approximation to a lifted space
    :cite:`kutz_dynamic_2016` - an introductory book for DMD and its connection to Koopman
    theory

    See Also
    --------

    :py:class:`.LinearDynamicalSystem`

    """

    @property
    def dmd_modes(self):
        if not self.is_spectral_mode():
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

        if self.is_matrix_mode() and (
            post_map is not None or user_set_modes is not None
        ):
            raise ValueError(f"post_map can only be provided with 'sys_type=spectral'")

        return post_map, user_set_modes, feature_columns

    def _compute_left_eigenvectors(
        self, system_matrix, eigenvalues, eigenvectors_right
    ):
        """Compute left eigenvectors such that
        system_matrix = eigenvectors_right_ @ diag(eigenvalues) @ eigenvectors_left_

        .. note::
             The eigenvectors are

             * not normed
             * row-wise in returned matrix

        """
        lhs_matrix = mat_dot_diagmat(eigenvectors_right, eigenvalues)
        return np.linalg.solve(lhs_matrix, system_matrix)

    @abc.abstractmethod
    def fit(self, X: TimePredictType, **fit_params) -> "DMDBase":
        """Abstract method to train DMD model.

        Parameters
        ----------
        X
            Training data
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
        time_values: np.ndarray,
        feature_columns=None,
    ):
        self.is_linear_system_setup(raise_error_if_not_setup=True)

        if feature_columns is None:
            feature_columns = self.feature_names_in_

        # initial condition is numpy-only, from now on, and column-oriented
        initial_states_origspace = X_ic.to_numpy().T

        time_series_ids = X_ic.index.get_level_values(
            TSCDataFrame.tsc_id_idx_name
        ).to_numpy()

        if len(np.unique(time_series_ids)) != len(time_series_ids):
            # check if duplicate ids are present
            raise ValueError("time series ids have to be unique")

        if self.is_matrix_mode():
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
            shift = self.time_interval_[0]

        norm_time_samples = time_values - shift

        tsc_df = self.evolve_system(
            time_delta=self.dt_,
            initial_conditions=initial_states_dmd,
            overwrite_sys_matrix=overwrite_sys_matrix,
            time_values=norm_time_samples,
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

    def predict(
        self,
        X: InitialConditionType,
        time_values: Optional[np.ndarray] = None,
        **predict_params,
    ) -> TSCDataFrame:
        """Predict time series data for each initial condition and time values.

        Parameters
        ----------
        X: TSCDataFrame, numpy.ndarray
            Initial conditions of shape `(n_initial_condition, n_features)`.

        time_values
            Time values to evaluate the model at.

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

        if isinstance(X, np.ndarray):
            # work internally only with DataFrames
            X = InitialCondition.from_array(X, columns=self.feature_names_in_)
        else:
            # for DMD the number of samples per initial condition is always 1
            InitialCondition.validate(X, n_samples_ic=1)

        self._validate_datafold_data(X)

        X, time_values = self._validate_features_and_time_values(
            X=X, time_values=time_values
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
            time_values=time_values,
            feature_columns=feature_columns,
        )

    def reconstruct(
        self,
        X: TSCDataFrame,
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
            tsc_kwargs={"ensure_const_delta_time": True},
        )
        self._validate_feature_names(X)

        # TODO: qois flag is currently not supported in DMD, bc. predict does not
        #  support it # 125
        # self._validate_qois(qois=qois, valid_feature_names=self.feature_names_in_)

        X_reconstruct_ts = []

        for X_ic, time_values in InitialCondition.iter_reconstruct_ic(
            X, n_samples_ic=1
        ):
            X_ts = self.predict(X=X_ic, time_values=time_values)
            X_reconstruct_ts.append(X_ts)

        return pd.concat(X_reconstruct_ts, axis=0)

    def fit_predict(self, X: TSCDataFrame, y=None, **fit_params) -> TSCDataFrame:
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
    r"""Full Dynamic Mode Decomposition of time series data to approximate the Koopman
    operator.

    The model computes the Koopman matrix :math:`K` with

    .. math::
        K X &= X^{+} \\
        K &= X^{+} X^{\dagger},

    where :math:`X` is the data with column oriented snapshots, :math:`\dagger` the
    the Moore–Penrose inverse and :math:`+` the future time shifted data.

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
        If True, also the left eigenvectors are also computed if
        ``sys_mode='spectral'`` (the parameter is ignored for `matrix` mode). This is
        more efficient to solve for initial conditions, because there is no least
        squares computation required for evaluating the linear dynamical
        system.

    approx_generator
        If True, approximate the generator of the system

        * `mode=spectral` compute (complex) eigenvalues of the
          Koopman generator :math:`log(\lambda) / \delta t`, with eigenvalues `\lambda`
          of the Koopman matrix. Note, that the left and right eigenvectors remain the
          same.
        * `mode=matrix` compute generator matrix with
          :math:`logm(K) / \delta t` (where :math:`logm` is the matrix logarithm.

        .. warning::

            This operation can fail if the eigenvalues of the matrix :math:`K` are too
            close to zero or the matrix logarithm is not well-defined because because of
            non-uniqueness. For details see :cite:`dietrich_koopman_2019` (Eq.
            3.2. and 3.3. and discussion). Currently, there are no counter measurements
            implemented to increase numerical robustness (work is needed). Consider
            also :py:class:`.gDMDFull`, which provides an alternative way to
            approximate the Koopman generator by using finite differences.

    rcond: Optional[float]
        Parameter passed to :class:`numpy.linalg.lstsq`.

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

    :cite:`schmid_dynamic_2010` - DMD method in the original sense
    :cite:`rowley_spectral_2009` - connects the DMD method to Koopman operator theory
    :cite:`tu_dynamic_2014` - generalizes the DMD to temporal snapshot pairs
    :cite:`williams_datadriven_2015` - generalizes the approximation to a lifted space
    :cite:`kutz_dynamic_2016` - an introductory book for DMD and Koopman connection
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
                f"Shift matrix (shape={G.shape}) has not full rank (={rank}), falling "
                f"back to least squares solution. The sum of residuals is: "
                f"{np.sum(residual)}"
            )

        # # TODO: START Experimental (test other solvers, with more functionality)
        # #  ridge_regression, and sparisty promoting least squares solutions could be
        #    included here
        # # TODO: clarify if the ridge regression should be done better on lstsq with
        #     shift matrices (instead of the G, G_dash)

        # TODO: fit_intercept option useful to integrate?
        # #  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV
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

    def _compute_spectal_components(self, system_matrix):
        eigenvalues_, eigenvectors_right_ = sort_eigenpairs(
            *np.linalg.eig(system_matrix)
        )
        eigenvectors_right_ /= np.linalg.norm(eigenvectors_right_, axis=0)

        if self.is_diagonalize:
            # must be computed with the Koopman eigenvalues
            # (NOT the generator eigenvalues)
            eigenvectors_left_ = self._compute_left_eigenvectors(
                system_matrix=system_matrix,
                eigenvalues=eigenvalues_,
                eigenvectors_right=eigenvectors_right_,
            )

        else:
            eigenvectors_left_ = None

        if self.approx_generator:
            # see e.g.https://arxiv.org/pdf/1907.10807.pdf pdfp. 10
            # Eq. 3.2 and 3.3.
            eigenvalues_ = np.log(eigenvalues_.astype(complex)) / self.dt_

        return eigenvectors_right_, eigenvalues_, eigenvectors_left_

    def fit(self, X: TimePredictType, y=None, **fit_params) -> "DMDFull":
        """Compute Koopman matrix and if applicable the spectral components.

        Parameters
        ----------
        X
            Training time series data.

        y: None
            ignored

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
            tsc_kwargs={"ensure_const_delta_time": True},
        )
        self._setup_features_and_time_attrs_fit(X=X)

        store_system_matrix = self._read_fit_params(
            attrs=[("store_system_matrix", False)], fit_params=fit_params
        )

        koopman_matrix_ = self._compute_koopman_matrix(X)

        if self.is_spectral_mode():
            (
                eigenvectors_right_,
                eigenvalues_,
                eigenvectors_left_,
            ) = self._compute_spectal_components(koopman_matrix_)
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

    :cite:`klus_data-driven_2020`

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

    def fit(self, X: TimePredictType, y=None, **fit_params) -> "gDMDFull":
        """Compute Koopman generator matrix and spectral components.

        Parameters
        ----------
        X
            Training time series data.

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
            tsc_kwargs={"ensure_const_delta_time": True},
        )
        self._setup_features_and_time_attrs_fit(X=X)

        store_generator_matrix = self._read_fit_params(
            attrs=[("store_generator_matrix", False)], fit_params=fit_params
        )

        kwargs_fd = self._generate_fd_kwargs()

        X_grad = X.tsc.time_derivative(**kwargs_fd)
        X = X.loc[X_grad.index, :]

        generator_matrix_ = self._compute_koopman_generator(X, X_grad)

        if self.is_spectral_mode():
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

    :cite:`kutz_dynamic_2016`
    :cite:`tu_dynamic_2014`

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

    def fit(self, X: TimePredictType, y=None, **fit_params) -> "DMDEco":
        """Compute spectral components of Koopman matrix in low dimensional singular
        value coordinates.

        Parameters
        ----------
        X
            Training time series data.
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
            tsc_kwargs={"ensure_const_delta_time": True},
        )
        self._setup_features_and_time_attrs_fit(X)
        self._read_fit_params(attrs=None, fit_params=fit_params)

        eigenvectors_right_, eigenvalues_, koopman_matrix = self._compute_internals(X)

        self.setup_spectral_system(
            eigenvectors_right=eigenvectors_right_, eigenvalues=eigenvalues_
        )

        return self


class PyDMDWrapper(DMDBase):
    """A wrapper for dynamic mode decompositions models of Python package *PyDMD*.

    For further details of the underlying models please go to
    `PyDMD documentation <https://mathlab.github.io/PyDMD/>`__

    .. warning::

        The models provided by *PyDMD* can only deal with single time series. See also
        `github issue #86 <https://github.com/mathLab/PyDMD/issues/86>`__.

    Parameters
    ----------

    method
        Choose a method by string.

        - "dmd" - standard DMD
        - "hodmd" - higher order DMD
        - "fbdmd" - forwards backwards DMD
        - "mrdmd" - multi resolution DMD
        - "cdmd" - compressed DMD

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

    :cite:`demo_pydmd_2018`

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
            self.dmd_ = pydmd.MrDMD(
                svd_rank=self.svd_rank,
                tlsq_rank=self.tlsq_rank,
                exact=self.exact,
                opt=self.opt,
                **self.init_params,
            )
        elif self.method == "cdmd":
            self.dmd_ = pydmd.CDMD(
                svd_rank=self.svd_rank,
                tlsq_rank=self.tlsq_rank,
                opt=self.opt,
                **self.init_params,
            )

        elif self.method == "dmdc":
            raise NotImplementedError(
                "Currently not implemented because DMD with control requires "
                "additional input."
            )
        else:
            raise ValueError(f"method={self.method} not known")

    def fit(self, X: TimePredictType, y=None, **fit_params) -> "PyDMDWrapper":
        """Compute Dynamic Mode Decomposition from data.

        Parameters
        ----------
        X
            Training time series data.

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
            tsc_kwargs={"ensure_const_delta_time": True},
        )
        self._setup_features_and_time_attrs_fit(X=X)
        self._read_fit_params(attrs=None, fit_params=fit_params)

        self._setup_pydmd_model()

        if len(X.ids) > 1:
            raise ValueError(
                "The PyDMD package only works for single coherent time series. See \n "
                "https://github.com/mathLab/PyDMD/issues/86"
            )

        # data is column major
        self.dmd_.fit(X=X.to_numpy().T)

        self.setup_spectral_system(
            eigenvectors_right=self.dmd_.modes, eigenvalues=self.dmd_.eigs
        )

        return self
