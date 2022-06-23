from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.linalg
from pandas.api.types import is_datetime64_dtype, is_timedelta64_dtype
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from datafold.pcfold import TSCDataFrame, allocate_time_series_tensor
from datafold.utils.general import diagmat_dot_mat, if1dim_colvec, is_matrix, is_scalar


class SystemSolveStrategy:
    _cls_valid_sys_type = ("differential", "flowmap")
    _cls_valid_sys_mode = ("matrix", "spectral")
    r"""A mathematical description of a standard linear dynamical system is

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

    """

    @staticmethod
    def differential_spectral(
        time_series_tensor,
        initial_conditions,
        sys_matrix,
        eigenvalues,
        time_values,
        **ignored,
    ):
        for idx, time in enumerate(time_values):
            time_series_tensor[:, idx, :] = np.real(
                sys_matrix
                @ diagmat_dot_mat(np.exp(eigenvalues * time), initial_conditions)
            ).T
        return time_series_tensor

    @staticmethod
    def differential_matrix(
        time_series_tensor, initial_conditions, sys_matrix, time_values, **ignored
    ):
        for idx, time in enumerate(time_values):
            time_series_tensor[:, idx, :] = np.real(
                scipy.linalg.expm(sys_matrix * time) @ initial_conditions
            ).T
        return time_series_tensor

    @staticmethod
    def flowmap_spectral(
        time_series_tensor,
        initial_conditions,
        sys_matrix,
        eigenvalues,
        time_values,
        time_delta,
        **ignored,
    ):
        # Usually, for a differential system the eigenvalues are written as:
        # omegas = np.log(eigenvalues.astype(complex)) / time_delta
        # --> evolve system with
        #               exp(omegas * t)
        # because this matches the notation of the differential system.

        # A disadvantage is that it requires the complex logarithm, which for
        # complex (eigen-)values can happen to be not well-defined.

        # A numerical more stable way is:
        # exp(omegas * t)
        # exp(log(ev) / time_delta * t)
        # exp(log(ev^(t/time_delta)))  -- logarithm rules
        # --> evolve system with, using `float_power`
        #               ev^(t / time_delta)
        _eigenvalues = eigenvalues.astype(complex)

        for idx, time in enumerate(time_values):
            time_series_tensor[:, idx, :] = np.real(
                sys_matrix
                @ diagmat_dot_mat(
                    np.float_power(_eigenvalues, time / time_delta),
                    initial_conditions,
                )
            ).T

        return time_series_tensor

    @staticmethod
    def flowmap_matrix(
        time_series_tensor,
        initial_conditions,
        sys_matrix,
        time_values,
        time_delta,
        **ignored,
    ):
        for idx, time in enumerate(time_values):
            # TODO: this is really expensive -- can also store intermediate
            #  results and only add the incremental fraction?
            time_series_tensor[:, idx, :] = np.real(
                scipy.linalg.fractional_matrix_power(sys_matrix, time / time_delta)
                @ initial_conditions
            ).T

        return time_series_tensor

    @staticmethod
    def flowmap_matrix_controlled(
        time_series_tensor,
        initial_conditions,
        sys_matrix,
        control_matrix,
        control_input,
        time_values,
        **ignored,
    ):

        time_series_tensor[:, 0, :] = initial_conditions.T

        for idx in range(len(time_values) - 1):
            # x_n+1 = A*x_n + B*u_n
            next_state = (
                sys_matrix @ time_series_tensor[:, idx, :].T
                + control_matrix @ control_input[:, idx, :].T
            )
            time_series_tensor[:, idx + 1, :] = next_state.T

        return time_series_tensor

    @staticmethod
    def flowmap_matrix_controlled_affine(
        time_series_tensor,
        initial_conditions,
        sys_matrix,
        control_matrix,
        control_input,
        time_values,
        **ignored,
    ):
        n_ic = initial_conditions.shape[1]
        if n_ic != control_input.shape[0]:
            raise ValueError("Control inputs and initial conditions don't match!")

        for i in range(n_ic):
            interp_control = interp1d(
                time_values, control_input[i], axis=0, kind="cubic"
            )

            affine_system_func = lambda t, state: (
                sys_matrix @ state + control_matrix @ interp_control(t) @ state
            )

            ivp_solution = solve_ivp(
                affine_system_func,
                t_span=(time_values[0], time_values[-1]),
                y0=initial_conditions[:, i],
                t_eval=time_values,
                method="RK23",
            )

            if not ivp_solution.success:
                raise RuntimeError(
                    f"The system could not be envolved for the requested "
                    f"timespan for initial condition {initial_conditions[i]}."
                )

            time_series_tensor[i] = ivp_solution.y.T

        return time_series_tensor

    @staticmethod
    def select_strategy(
        system_type, system_mode, is_time_invariant, is_controlled, is_affine
    ):
        # TODO: include interpolate flag?

        if not is_time_invariant:
            raise NotImplementedError(
                "Currently there are only time invariant system solvers implemented"
            )

        if (
            system_type == "differential"
            and system_mode == "spectral"
            and not is_controlled
        ):
            return SystemSolveStrategy.differential_spectral
        elif (
            system_type == "differential"
            and system_mode == "matrix"
            and not is_controlled
        ):
            return SystemSolveStrategy.differential_matrix
        elif (
            system_type == "flowmap" and system_mode == "spectral" and not is_controlled
        ):
            return SystemSolveStrategy.flowmap_spectral
        elif system_type == "flowmap" and system_mode == "matrix" and not is_controlled:
            return SystemSolveStrategy.flowmap_matrix
        elif (
            system_type == "flowmap"
            and system_mode == "matrix"
            and is_controlled
            and not is_affine
        ):
            return SystemSolveStrategy.flowmap_matrix_controlled
        elif (
            system_type == "differential"
            and system_mode == "matrix"
            and is_controlled
            and is_affine
        ):
            return SystemSolveStrategy.flowmap_matrix_controlled_affine
        else:
            raise ValueError(
                "no strategy found to solve specified linear dynamical system"
            )


class LinearDynamicalSystem(object):
    r"""Evolve linear dynamical system forward in time.

        There are many definitions for a linear dynamical system. See the class
        :py:class:`??` to specify the form

        Parameters
        ----------

        sys_type
            Type of linear system:

            * "differential"
            * "flowmap"
            * "controlled"

        sys_mode
            Whether the system is evaluated with

            * "matrix" (i.e. :math:`A` or :math:`\mathcal{A}` are given)
            * "spectral" (i.e. eigenpairs of :math:`A` or :math:`\mathcal{A}` are given)

        time_invariant
            If True, the system internally always starts with `time=0`. \
            This is irrespective of the time given in the time values. If the initial
            time is larger than zero, the internal times are corrected to the requested time.

        References
        ----------
        :cite:`kutz-2016` (pages 3 ff.)
        """

    _cls_valid_sys_type = ("differential", "flowmap")
    _cls_valid_sys_mode = ("matrix", "spectral")

    def __init__(
        self,
        sys_type: str,
        sys_mode: str,
        is_controlled=False,
        is_affine_control=False,
        time_invariant: bool = True,
    ):
        self.sys_type = sys_type
        self.sys_mode = sys_mode
        self.is_controlled = is_controlled
        self.is_affine_control = is_affine_control
        self.time_invariant = time_invariant

        self._check_system_type()
        self._check_system_mode()

        self._evolve_system_states = SystemSolveStrategy.select_strategy(
            system_type=self.sys_type,
            system_mode=self.sys_mode,
            is_time_invariant=time_invariant,
            is_controlled=is_controlled,
            is_affine=is_affine_control,
        )

    def _validate_matrix(self, matrix, name):
        is_matrix(matrix, name, square=False, allow_sparse=False, handle="raise")

    def _check_system_type(self) -> None:
        if self.sys_type not in self._cls_valid_sys_type:
            raise ValueError(
                f"'{self.sys_type=}' is invalid. "
                f"Choose from {self._cls_valid_sys_type}"
            )

    def _check_system_mode(self) -> None:
        if self.sys_mode not in self._cls_valid_sys_mode:
            raise ValueError(
                f"'{self.sys_mode=}' is invalid."
                f"Choose from {self._cls_valid_sys_mode}"
            )

    def _check_sys_matrix(self, sys_matrix: Optional[np.ndarray]):
        if sys_matrix is None:
            # The system matrix can be overwritten from outside
            # (if sys_matrix is not None).
            # This is particularily useful when A is the system matrix,
            # but there is a post-map, e.g. D @ A.
            if self.is_spectral_mode():
                sys_matrix = self.eigenvectors_right_
            else:  # self.is_matrix_mode()
                sys_matrix = self.sys_matrix_

        return sys_matrix

    def _check_control_input(
        self,
        control_input: np.ndarray,
        n_time_values: int,
        n_initial_condition: int = 1,
    ):
        control_input = if1dim_colvec(control_input)
        control_input = (
            control_input[np.newaxis] if control_input.ndim == 2 else control_input
        )
        if control_input.shape[0] != n_initial_condition:
            raise ValueError(
                "control_input does not match the number of time series in the ic"
            )
        if control_input.shape[1] != n_time_values:
            raise ValueError("control_input should have the same length as time_values")
        if control_input.shape[2] != self.control_matrix_.shape[-1]:
            raise ValueError(
                "control_input columns should match the last dimension of the control_matrix"
            )
        return control_input

    def _check_initial_condition(
        self, initial_condition: np.ndarray, state_length: int
    ) -> np.ndarray:
        try:
            if is_scalar(initial_condition):
                initial_condition = [initial_condition]
            initial_condition = np.asarray(initial_condition)
        except:
            raise TypeError(
                "Parameter 'ic' must be be an array-like object. "
                f"Got {type(initial_condition)=}"
            )

        if initial_condition.ndim == 1:
            initial_condition = if1dim_colvec(initial_condition)

        if initial_condition.ndim != 2:
            raise ValueError(  # in case ndim > 2
                f"Initial conditions 'ic' must have 2 dimensions. "
                f"Got {initial_condition.ndim=}."
            )

        if initial_condition.shape[0] != state_length:
            raise ValueError(
                f"Mismatch in dimensions between initial condition and system matrix. "
                f"{initial_condition.shape[0]=} is not "
                f"{state_length=}."
            )
        return initial_condition

    def _check_time_values(self, time_values: np.ndarray):
        try:
            if is_scalar(time_values):
                time_values = np.array([time_values])
            time_values = np.asarray(time_values)
        except:
            raise TypeError(
                "The parameter 'time_values' must be an array-like object. "
                f"Got {type(time_values)=}"
            )

        # see https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
        if time_values.ndim != 1 and time_values.dtype.kind in "buif":
            raise ValueError(
                f"The array 'time_values' must be 1-dim. (got {time_values.ndim=}) and only "
                f"contain real-valued numeric data (got {time_values.dtype.kind=})."
            )

        if is_timedelta64_dtype(time_values) or is_datetime64_dtype(time_values):
            time_values = time_values.astype(np.int64)

        return time_values

    def _check_time_delta(self, time_delta: Optional[Union[float, int]]):
        if self.is_differential_system():
            # for a differential system there is no time_delta -- all input is ignored
            time_delta = None
        elif time_delta is None or not is_scalar(time_delta):
            raise TypeError(
                "In a 'flowmap' system the parameter 'time_delta' must be provided as a "
                f"scalar value and a scalar value. Got {type(time_delta)=}"
            )
        else:
            assert time_delta is not None  # mypy
            # cast to built-in Python value
            if np.asarray(time_delta).dtype.kind in "mli":
                time_delta = int(time_delta)
            else:
                time_delta = float(time_delta)
            if time_delta <= 0:
                raise ValueError(f"{time_delta=} must be a positive number.")
        return time_delta

    def _check_and_set_system_params(
        self,
        initial_conditions: np.ndarray,
        sys_matrix: Optional[np.ndarray],
        control_input: Optional[np.ndarray],
        time_values: np.ndarray,
        time_delta: Optional[Union[float, int]],
        time_series_ids,
        feature_names_out,
    ):
        # SYSTEM MATRIX
        sys_matrix = self._check_sys_matrix(sys_matrix)
        n_features, state_length = sys_matrix.shape

        # INITIAL CONDITION
        initial_conditions = self._check_initial_condition(
            initial_conditions, state_length
        )

        # TIME VALUES
        time_values = self._check_time_values(time_values)

        # CONTROL_INPUT
        if self.is_controlled:
            control_input = self._check_control_input(
                control_input=control_input,
                n_time_values=len(time_values),
                n_initial_condition=initial_conditions.shape[1],
            )

        # TIME DELTA
        time_delta = self._check_time_delta(time_delta)

        # TIME SERIES IDS and FEATURE COLUMNS
        # Note: all the other checks are made during TSCDataFrame allocation.
        if time_series_ids is None:
            time_series_ids = np.arange(initial_conditions.shape[1])

        if feature_names_out is None:
            feature_names_out = np.arange(state_length)

        if len(feature_names_out) != n_features:
            raise ValueError(f"{len(feature_names_out)=} != {state_length=}")

        return (
            initial_conditions,
            sys_matrix,
            control_input,
            time_values,
            time_delta,
            n_features,
            state_length,
            time_series_ids,
            feature_names_out,
        )

    def is_matrix_mode(self) -> bool:
        r"""Indicate whether the linear system is in "matrix" mode.

        The system uses either matrix :math:`A` for flowmap or :math:`\mathcal{A}`
        for a differential system.
        """
        self._check_system_mode()
        return self.sys_mode == "matrix"

    def is_spectral_mode(self) -> bool:
        r"""Indicate whether the linear system is in "spectral" mode.

        The system uses the spectral components of either matrix :math:`A` for flowmap or
        :math:`\mathcal{A}` for differential.
        """
        self._check_system_mode()
        return self.sys_mode == "spectral"

    def is_differential_system(self) -> bool:
        r"""Indicate whether the linear system is of "differential" type.

        The system uses either the matrix :math:`\mathcal{A}` or the spectral components to
        evolve the system.
        """
        self._check_system_type()
        return self.sys_type == "differential"

    def is_flowmap_system(self) -> bool:
        r"""Indicate whether the linear system is a "flowmap" system.

        The system uses either the matrix :math:`A` or its spectral
        components to evolve the system.
        """
        self._check_system_type()
        return self.sys_type == "flowmap"

    def is_linear_system_setup(self, raise_error_if_not_setup: bool = False) -> bool:
        """Indicate whether the linear system is set up."""

        if self.is_matrix_mode():
            is_setup = hasattr(self, "sys_matrix_")
        else:  # self.is_differential_system():
            is_setup = hasattr(self, "eigenvectors_right_") and hasattr(
                self, "eigenvalues_"
            )

        if not is_setup and raise_error_if_not_setup:
            raise RuntimeError("Linear system has not been set up.")
        else:
            return is_setup

    def compute_spectral_system_states(self, states) -> np.ndarray:
        r"""Compute the spectral states of the system.

        If the linear system is written in its spectral form:

        .. math::
            \Psi_r \Lambda^n \Psi_l x_0 &= x_n \\
            \Psi_r \Lambda^n b_0 &= x_n \\

        then `b_0` is the spectral state, which is computed in this function. It does
        not necessarily need to be an initial state but instead can be arbitrary states.

        In the context of dynamic mode decomposition, the spectral state is also often
        referred to as "amplitudes". E.g., see :cite:t:`kutz-2016`, page 8. In
        the context of `EDMD`, where the DMD model acts on a dictionary space, then the
        spectral states are the evaluation of the Koopman eigenfunctions. See e.g.,
        :cite:t:`williams-2015` Eq. 3 or 6.

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
                f"To compute the spectral system states self.sys_mode='spectral' is required. "
                f"Got {self.sys_mode=}."
            )

        # Choose between two alternatives:
        if hasattr(self, "eigenvectors_left_") and (
            self.eigenvectors_left_ is not None and self.eigenvectors_right_ is not None
        ):
            # uses both eigenvectors (left and right).
            # this is Eq. 18 in :cite:`williams-2015` (note that in the
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
                "Attribute 'eigenvectors_right_ is None'. Please report bug."
            )

        return states

    def setup_spectral_system(  # TODO: is there a better way to accomplish this?
        self,
        eigenvectors_right: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvectors_left: Optional[np.ndarray] = None,
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

        # TODO: provide an update mode for the streaming case!

        if not self.is_spectral_mode():
            raise RuntimeError(
                f"The 'sys_mode' was set to {self.sys_mode}. Cannot setup "
                f"system with spectral."
            )

        if self.is_linear_system_setup():
            raise RuntimeError("Linear system is already setup.")

        self._validate_matrix(eigenvectors_right, "eigenvectors_right_")

        self.eigenvectors_right_ = eigenvectors_right
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_left_ = eigenvectors_left

        return self

    def setup_matrix_system(self, system_matrix, *, control_matrix=None):
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

        self._validate_matrix(system_matrix, "system_matrix")
        self.sys_matrix_ = system_matrix

        if self.is_controlled:
            self.control_matrix_ = control_matrix

            if not self.is_affine_control:
                self._validate_matrix(control_matrix, "control_matrix")
                if self.control_matrix_.shape[0] != self.sys_matrix_.shape[0]:
                    raise ValueError(
                        "control_matrix and system_matrix must have the same number of rows."
                        f"Got {self.control_matrix_.shape[0]=} and "
                        f"{self.sys_matrix_.shape[0]=}."
                    )
            else:
                if (
                    not isinstance(self.control_matrix_, np.ndarray)
                    or self.control_matrix_.ndim != 3
                ):
                    raise ValueError(
                        "The control matrix tensor must be 3-dim. and of type np.ndarray"
                    )

                if self.control_matrix_.shape[:2] != self.sys_matrix_.shape:
                    raise ValueError(
                        "control_matrix and system_matrix must have "
                        "the same number of rows and columns"
                    )

        else:
            if control_matrix is not None:
                raise ValueError(
                    f"If {self.is_controlled} is set, a control matrix must be provided!"
                )
        return self

    def evolve_system(  # TODO: better name?
        self,
        initial_conditions: np.ndarray,
        *,
        time_values: Union[np.ndarray, float, int, list],
        control_input: Optional[np.ndarray] = None,
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
            The system is evaluated with `matrix_fractional_power`

            .. math::
                x(t) = A^{t / \Delta t} \cdot x_0

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
                    x(t) = \Psi \cdot \Lambda^{t / \Delta t}) \cdot b_0

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
            Time delta :math:`\Delta t` for reference. This is a required parameter in a
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
            initial_conditions,
            sys_matrix,
            control_input,
            time_values,
            time_delta,
            n_features,
            state_length,
            time_series_ids,
            feature_names_out,
        ) = self._check_and_set_system_params(
            initial_conditions=initial_conditions,
            sys_matrix=overwrite_sys_matrix,
            control_input=control_input,
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
            initial_conditions=initial_conditions,
            sys_matrix=sys_matrix,
            control_matrix=self.control_matrix_ if self.is_controlled else None,
            control_input=control_input,
            time_values=time_values,
            eigenvalues=self.eigenvalues_ if self.is_spectral_mode() else None,
            time_delta=time_delta,
        )

        return TSCDataFrame.from_tensor(
            time_series_tensor,
            time_series_ids=time_series_ids,
            columns=feature_names_out,
            time_values=time_values,
        )
