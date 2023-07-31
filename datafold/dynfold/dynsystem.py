from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import scipy.linalg
from pandas.api.types import is_datetime64_dtype, is_timedelta64_dtype
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from datafold.pcfold import TSCDataFrame, allocate_time_series_tensor
from datafold.utils.general import (
    diagmat_dot_mat,
    if1dim_colvec,
    is_matrix,
    is_scalar,
    is_vector,
)


class SystemSolveStrategy:
    """Collection of linear dynamical system solvers."""

    @staticmethod
    def differential_matrix(
        time_series_tensor, initial_conditions, sys_matrix, time_values, **ignored
    ):
        r"""Continuous linear dynamical system.

        The system is evaluated with the analytical solution of a linear dynamical system by
        using the matrix exponential

        .. math::
            x(t) = \exp(\mathcal{A \cdot t}) x(0)

        with

        * :math:`\mathcal{A}`, a constant matrix to describe the vector field of the systems'
          states
        """
        for idx, time in enumerate(time_values):
            time_series_tensor[:, idx, :] = np.real(
                scipy.linalg.expm(sys_matrix * time) @ initial_conditions
            ).T
        return time_series_tensor

    @staticmethod
    def differential_matrix_controlled(
        time_series_tensor,
        initial_conditions,
        sys_matrix,
        control_matrix,
        control_input,
        time_values,
        **ignored,
    ):
        r"""Solves system:
        \dot{x} = A x + B u.
        """
        time_series_tensor[:, 0, :] = initial_conditions
        time_diff = np.diff(time_values)

        n_states = sys_matrix.shape[0]
        n_inputs = control_matrix.shape[1]

        for idx, tdiff in enumerate(time_diff):
            M = np.vstack(
                [np.hstack([sys_matrix * tdiff]), control_matrix * tdiff],
                np.zeros([n_inputs, n_states + n_inputs]),
            )
            expM = np.linalg.expm(M)
            Ad = expM[:n_states, :n_states]
            Bd = expM[n_states:, :n_states]

            time_series_tensor[:, idx + 1, :] = (
                Ad @ time_series_tensor[:, idx, :] + Bd @ control_input[:, idx, :]
            )

        return time_series_tensor

    @staticmethod
    def differential_matrix_controlled_affine(
        time_series_tensor,
        initial_conditions,
        sys_matrix,
        control_matrix,
        control_input,
        time_values,
        **ignored,
    ):
        """Documentation and review welcome.
        This is based on the implementation of :cite:`peitz-2020`.
        """
        # solve for each initial condition separately
        for i in range(initial_conditions.shape[1]):
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
                    f"The system could not be simulated for the requested "
                    f"timespan and initial condition {i=} with values "
                    f"{initial_conditions[i]=}."
                )

            time_series_tensor[i] = ivp_solution.y.T

        return time_series_tensor

    @staticmethod
    def differential_spectral(
        time_series_tensor,
        initial_conditions,
        sys_matrix,
        eigenvalues,
        time_values,
        **ignored,
    ):
        r"""Continuous linear dynamical system in spectral components.

        The base system is the same as with ``differential_matrix``, where the system matrix
        is now in spectral components. The system is evaluated with the analytical
        solution of a linear dynamical system using the exponential of the eigenvalues

        .. math::
            x(t) = \Psi \cdot \exp(\Lambda \cdot t) \cdot b(0)

        with the following system components:

        * :math:`b(0)` are the spectrally-aligned initial condition (see
          :py:meth:`.compute_spectral_system_states`)
        * :math:`\Psi` the right eigenvectors
        * :math:`\Lambda` the eigenvalues
        """
        for idx, time in enumerate(time_values):
            time_series_tensor[:, idx, :] = np.real(
                sys_matrix
                @ diagmat_dot_mat(np.exp(eigenvalues * time), initial_conditions)
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
        r"""Discrete flowmap of a linear dynamical system.

        The standard form is

        .. math::
             x_{n+1} = A \cdot x_{n}

        where :math:`n = \frac{t}{\Delta t}`. Note that if `n` is not an integer this
        essentially interpolates system states. For this reason the function uses
        `matrix_fractional_power`, which is computationally demanding.

        Note for future optimization (if required):
            * If :code:`time_values / time_delta` are all (near) integers this function could
              also use np.power()
            * If :code:`time_values / time_delta` are `[0, 1, 2, ...]`, within the loop the
              system matrix can be iteratively updated
        """
        for idx, time in enumerate(time_values):
            # TODO: this is really expensive -- can also store intermediate
            #  results and only add the incremental fraction?
            time_series_tensor[:, idx, :] = np.real(
                scipy.linalg.fractional_matrix_power(sys_matrix, time / time_delta)
                @ initial_conditions
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
        r"""Discrete flowmap of a linear dynamical system in spectral components.

        The base system is the same as with ``flowmap_matrix`` where the system matrix
        is now in spectral components. The system is evaluated with the analytical
        solution of a linear dynamical system using the exponential of the eigenvalues

        .. math::
            x(n+1) = \Psi \cdot \Lambda^{n}) \cdot b_0

        where :math:`n = \frac{t}{\Delta t}`. Note that if `n` is not an integer this
        essentially interpolates the system states. For this reason the function uses
        `float_power`. In particular if the eigenvectors are complex-valued this can result in
        ill-defined mapping
        (see https://en.wikipedia.org/wiki/Exponentiation#Complex_exponentiation). The
        `float_power` always uses the principal value.

        System components:

        * :math:`b_0` are the spectrally-aligned initial conditions (see
          :py:meth:`.compute_spectral_system_states`)
        * :math:`\Psi` the right eigenvectors
        * :math:`\Lambda` the eigenvalues

        """
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
    def flowmap_matrix_controlled(
        time_series_tensor,
        initial_conditions,
        sys_matrix,
        control_matrix,
        control_input,
        time_values,
        time_delta,
        **ignored,
    ):
        r"""A linear dynamical system as a discrete flowmap with control input.

        .. math::
            x_{n+1} = A \cdot x_{n} + B \cdot u_{n}

        with
        * :math:`A \in \mathbb{R}^{[m \times m]}`, a constant matrix, which describes
          the linear evolution of the systems' states
        * :math:`x` is the state with length :math:`m`,
        * :math:`B \in \mathbb{R}^{[m \times q]}`, another constant matrix which describes
          the effect of the control input
          :math:`u` is the control input with length :math:`q`.

        Note that the control input has to be available for the entire time horizon of the
        prediction.
        """
        time_series_tensor[:, 0, :] = initial_conditions.T

        for idx, time_diff in enumerate(np.diff(time_values)):
            # x_n+1 = A*x_n + B*u_n

            time_fraction = time_diff / time_delta

            if False and np.abs(time_fraction - 1.0) > 1e-15:
                # TODO: how to "reduce" the control_matrix? It is generally rectangular!
                # TODO: what if the system_matrix includes a post map, then this has to be
                #  performed after each step (it cannot be combined with the eigenvectors!)
                import warnings

                warnings.warn(
                    "Solving the control system with a different time sampling to "
                    "the original sampling rate is potentially wrong.",
                    stacklevel=1,
                )

                _f_sys_matrix = scipy.linalg.fractional_matrix_power(
                    sys_matrix, time_fraction
                )
            else:
                _f_sys_matrix = sys_matrix

            next_state = (
                _f_sys_matrix @ time_series_tensor[:, idx, :].T
                + control_matrix @ control_input[:, idx, :].T
            )
            time_series_tensor[:, idx + 1, :] = next_state.real.T

        return time_series_tensor

    @staticmethod
    def flowmap_spectral_controlled(
        time_series_tensor,
        initial_conditions,
        sys_matrix,
        eigenvectors_left,
        eigenvalues,
        control_matrix,
        control_input,
        time_values,
        **ignored,
    ):
        time_series_tensor[:, 0, :] = np.real(sys_matrix @ initial_conditions).T

        # because of the structure it is not possible to directly make use of the spectral
        # representation. It is therefore more efficient to reconstruct the matrix form. While
        # it seems unnecessary to compute the spectral components in the first place, the
        # components can still be used for analysis and the reconstruction
        reconstruct_system_matrix = sys_matrix @ diagmat_dot_mat(
            eigenvalues, eigenvectors_left
        )

        for idx, _time in enumerate(time_values[1:]):
            next_state = (
                reconstruct_system_matrix @ time_series_tensor[:, idx, :].T
                + control_matrix @ control_input[:, idx, :].T
            )
            time_series_tensor[:, idx + 1, :] = np.real(next_state).T

        return time_series_tensor

    @staticmethod
    def select_strategy(
        *, system_type, system_mode, is_time_invariant, is_controlled, is_control_affine
    ):
        """Selects the solver for the linear dynamical system based on the given system
        parameters.
        """
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
            system_type == "differential" and system_mode == "matrix" and is_controlled
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
            and not is_control_affine
        ):
            return SystemSolveStrategy.flowmap_matrix_controlled
        elif (
            system_type == "flowmap"
            and system_mode == "spectral"
            and is_controlled
            and not is_control_affine
        ):
            return SystemSolveStrategy.flowmap_spectral_controlled
        elif (
            system_type == "differential"
            and system_mode == "matrix"
            and is_controlled
            and is_control_affine
        ):
            return SystemSolveStrategy.differential_matrix_controlled_affine
        else:
            raise ValueError(
                "No strategy found to solve the specified linear dynamical system\n"
                f"{system_type=}, {system_mode=}, {is_time_invariant=}, {is_controlled=}, "
                f"{is_control_affine=}"
            )


class LinearDynamicalSystem:
    r"""Evolve linear dynamical system forward in time.

    There are various definitions of a linear dynamical system, the specific form is
    selected from

    Parameters
    ----------
    sys_type
        Type of linear system:

        * "differential"
        * "flowmap"

    sys_mode
        Whether the system is evaluated with

        * "matrix" (i.e. :math:`A` or :math:`\mathcal{A}` are given)
        * "spectral" (i.e. eigenpairs of :math:`A` or :math:`\mathcal{A}` are given)

    is_controlled:
        Whether the system is controlled. If set to True a control matrix must be passed to
        setup_matrix_system (currently there is no implementation for spectral systems)

    is_control_affine
        Whether the system is a control affine. The control matrix must be a 3-dim. array
        (tensor) and the implementation is based on :cite:t:`peitz-2020`.

    is_time_invariant
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
        sys_type: Literal["differential", "flowmap"],
        sys_mode: Literal["matrix", "spectral"],
        is_controlled: bool = False,
        is_control_affine: bool = False,
        is_time_invariant: bool = True,
    ):
        self.sys_type = sys_type
        self.sys_mode = sys_mode
        self.is_controlled = is_controlled
        self.is_control_affine = is_control_affine
        self.is_time_invariant = is_time_invariant

        self._check_system_type()
        self._check_system_mode()

        self._evolve_system_states = SystemSolveStrategy.select_strategy(
            system_type=self.sys_type,
            system_mode=self.sys_mode,
            is_controlled=self.is_controlled,
            is_control_affine=self.is_control_affine,
            is_time_invariant=self.is_time_invariant,
        )

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

    def _set_sys_matrix(self, sys_matrix: Optional[np.ndarray]):
        if sys_matrix is None:
            # The system matrix can be overwritten from outside
            # (if sys_matrix is not None).
            # This is particularly useful when A is the system matrix,
            # but there is a post-map, e.g. D @ A.
            if self.is_spectral_mode:
                sys_matrix = self.eigenvectors_right_
            else:  # self.is_matrix_mode()
                sys_matrix = self.sys_matrix_
        else:
            is_matrix(sys_matrix, "overwrite_sys_matrix")

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
                f"{control_input.shape[0]=} does not match the number of initial "
                f"conditions (={n_initial_condition})"
            )

        if self.control_matrix_ is None:
            raise ValueError(
                "If control input is provided, then a control_matrix needs to be set up!"
            )

        req_last_control_state = getattr(self, "_requires_last_control_state", False)
        req_control_input = control_input.shape[1] + int(not req_last_control_state)

        if req_control_input != n_time_values:
            raise ValueError(
                f"{req_control_input=} does not match number of time values {n_time_values=}"
            )

        if control_input.shape[2] != self.control_matrix_.shape[-1]:
            raise ValueError(
                f"{control_input.shape[2]=} should match the last dimension of the control "
                f"matrix {self.control_matrix_.shape[-1]=}"
            )
        return control_input

    def _check_initial_condition(
        self, initial_condition: np.ndarray, state_length: int
    ) -> np.ndarray:
        try:
            if is_scalar(initial_condition):
                initial_condition = [initial_condition]
            initial_condition = np.asarray(initial_condition)
        except Exception:
            raise TypeError(
                "Parameter 'ic' must be be an array-like object. "
                f"Got {type(initial_condition)=}"
            )

        if initial_condition.ndim == 1:
            initial_condition = if1dim_colvec(initial_condition)

        is_matrix(initial_condition, "initial_condition")

        if initial_condition.shape[0] != state_length:
            raise ValueError(
                f"Mismatch in dimensions between initial condition and system matrix. "
                f"{initial_condition.shape[0]=} is not {state_length=}."
            )
        return initial_condition

    def _check_time_values(self, time_values: np.ndarray):
        try:
            if is_scalar(time_values):
                time_values = np.array([time_values])
            time_values = np.asarray(time_values)
        except Exception:
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
        if self.is_differential_system:
            # for a differential system there is no time_delta -- all input is ignored
            time_delta = None
        elif time_delta is None or not is_scalar(time_delta):
            raise TypeError(
                "In a 'flowmap' system the parameter 'time_delta' must be provided as a "
                f"scalar value. Got {type(time_delta)=}"
            )
        else:
            assert time_delta is not None  # mypy
            # cast to built-in Python value
            np_time_delta = np.asarray(time_delta)

            if np_time_delta.dtype.kind in "mi":
                # Note: it looks over-complicated but this is works for both timedelta and
                #       signed int
                time_delta = int(np_time_delta.astype(int))
            else:
                time_delta = float(time_delta)
            if time_delta <= 0:
                raise ValueError(f"{time_delta=} must be a positive number.")
        return time_delta

    def _set_and_check_system_params(
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
        sys_matrix = self._set_sys_matrix(sys_matrix)
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

    @property
    def is_matrix_mode(self) -> bool:
        r"""Indicate whether the linear system is in "matrix" mode.

        The system uses either matrix :math:`A` for flowmap or :math:`\mathcal{A}`
        for a differential system.
        """
        self._check_system_mode()
        return self.sys_mode == "matrix"

    @property
    def is_spectral_mode(self) -> bool:
        r"""Indicate whether the linear system is in "spectral" mode.

        The system uses the spectral components of either matrix :math:`A` for flowmap or
        :math:`\mathcal{A}` for differential.
        """
        self._check_system_mode()
        return self.sys_mode == "spectral"

    @property
    def is_differential_system(self) -> bool:
        r"""Indicate whether the linear system is of "differential" type.

        The system uses either the matrix :math:`\mathcal{A}` or the spectral components to
        evolve the system.
        """
        self._check_system_type()
        return self.sys_type == "differential"

    @property
    def is_flowmap_system(self) -> bool:
        r"""Indicate whether the linear system is a "flowmap" system.

        The system uses either the matrix :math:`A` or its spectral
        components to evolve the system.
        """
        self._check_system_type()
        return self.sys_type == "flowmap"

    def is_linear_system_setup(self, raise_error_if_not_setup: bool = False) -> bool:
        """Indicate whether the linear system is set up."""
        if self.is_matrix_mode:
            is_setup = hasattr(self, "sys_matrix_")
        else:  # self.is_differential_system():
            is_setup = hasattr(self, "eigenvectors_right_") and hasattr(
                self, "eigenvalues_"
            )

        if not is_setup and raise_error_if_not_setup:
            raise RuntimeError("Linear system has not been set up.")
        else:
            return is_setup

    def compute_spectral_system_states(self, states: np.ndarray) -> np.ndarray:
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

        2. or by using the left eigenvectors and computing the matrix-vector \
           product

            .. math::
                \Psi_l x_0 = b_0

        If the left eigenvectors where set during :py:meth:`.setup_sys_spectral`,
        then alternative 2 is used always and otherwise alternative 1.

        Parameters
        ----------
        states
            The states of original data space in column-orientation `(state_length, n_states)`.

        Returns
        -------
        numpy.ndarray
            spectrally aligned states
        """
        self.is_linear_system_setup(raise_error_if_not_setup=True)

        if not self.is_spectral_mode:
            raise AttributeError(
                f"To compute the spectral system states sys_mode='spectral' is required. "
                f"Got {self.sys_mode=}."
            )

        # Choose between two alternatives:
        if hasattr(self, "eigenvectors_left_") and (
            self.eigenvectors_left_ is not None and self.eigenvectors_right_ is not None
        ):
            # uses both eigenvectors (left and right).
            # this is Eq. 18 in :cite:`williams-2015` (note that in the
            # paper the Koopman matrix is transposed, therefore here left and right
            # eigenvectors are exchanged).
            states = self.eigenvectors_left_ @ states
        elif (
            hasattr(self, "eigenvectors_right_")
            and self.eigenvectors_right_ is not None
        ):
            # represent the initial condition in terms of right eigenvectors (by solving a
            # least-squares problem) -- in this case only the right eigenvectors are required
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
        control_matrix: Optional[np.ndarray] = None,
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

        control_matrix
            An additional control matrix (note that currently the control matrix is not
            described in spectral components.

        Returns
        -------
        LinearDynamicalSystem
            self
        """
        if not self.is_spectral_mode:
            raise RuntimeError(
                f"With '{self.sys_mode=}' this function is not supported."
            )

        is_matrix(eigenvectors_right, name="eigenvectors_right")

        if eigenvectors_left is not None:
            is_matrix(eigenvectors_right, name="eigenvectors_left")

        is_vector(eigenvalues)

        if eigenvalues.shape[0] != eigenvectors_right.shape[1]:
            raise ValueError("")

        self.eigenvectors_right_ = eigenvectors_right
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_left_ = eigenvectors_left

        self.control_matrix_ = control_matrix

        return self

    def setup_matrix_system(
        self, system_matrix, *, control_matrix=None
    ) -> "LinearDynamicalSystem":
        r"""Set up linear system with system matrix.

        Parameters
        ----------
        system_matrix
            The system matrix (either :math:`A` for flowmap or :math:`\mathcal{A}` for
            differential type).

        control_matrix
            The control matrix. Required if the linear system is controlled.

        Returns
        -------
        LinearDynamicalSystem
            self
        """
        if self.is_linear_system_setup():
            raise RuntimeError("Linear system is already setup.")

        is_matrix(system_matrix, "system_matrix")
        self.sys_matrix_ = system_matrix

        if self.is_controlled:
            self.control_matrix_ = control_matrix

            if not self.is_control_affine:
                is_matrix(control_matrix, "control_matrix")
                if self.control_matrix_.shape[0] != self.sys_matrix_.shape[0]:
                    raise ValueError(
                        "control_matrix and system_matrix must have the same number of rows."
                        f"Got {self.control_matrix_.shape[0]=} and "
                        f"{self.sys_matrix_.shape[0]=}."
                    )
            else:  # if is_affine_control, then provide a 3-dim tensor
                if (
                    not isinstance(self.control_matrix_, np.ndarray)
                    or self.control_matrix_.ndim != 3
                ):
                    raise ValueError(
                        f"If the system is set to affine control, then the control matrix "
                        f"must be a 3-dim. array (got {control_matrix.ndim=}) and of type "
                        f"np.ndarray (got {type(control_matrix)})."
                    )

                if self.control_matrix_.shape[:2] != self.sys_matrix_.shape:
                    raise ValueError(
                        "control_matrix and system_matrix must have "
                        "the same number of rows and columns "
                    )

        else:
            if control_matrix is not None:
                raise ValueError(
                    f"If {self.is_controlled=}, no control matrix should be provided!"
                )
        return self

    def evolve_system(
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

        The system evolves depending on the specified system solver (set during
        initialization).

        Parameters
        ----------
        initial_conditions
            Single initial condition of shape `(n_features,)` or multiple initial
            conditions of shape `(n_features, n_initial_conditions)`.

        time_values
            Time values to evaluate the linear system at :math:`t \in \mathbb{R}^{+}`

        control_input
            Control states over the time horizon acting to the system dynamics. The array has
            to have a shape of `(n_timesteps, n_control_features)` for a single initial
            condition and be a tensor with
            `(n_initial_condition, n_timesteps, n_control_features)` for multiple initial
            conditions.

        overwrite_sys_matrix
            Primarily for performance reasons the system matrix :math:`A` can also be
            overwritten. An example is to include linear post-mappings of the system (e.g.
            a projection matrix :math:`P`, resulting in only returning some quantities of
            interest; :math:`A^{*} = P \cdot A`).

        time_delta
            Time delta :math:`\Delta t`. This is a required parameter in a "flowmap" system.

        time_series_ids
           Unique integer time series IDs of shape `(n_initial_conditions,)` for each
           respective initial condition. Defaults to `(0, 1, 2, ...)`.

        feature_names_out
            Unique feature columns names of shape `(n_features,)`.
            Defaults to `(0, 1, 2, ...)`.

        Returns
        -------
        TSCDataFrame
            Time series for each initial condition, each time series has \
            shape `(n_time_values, n_features)`
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
        ) = self._set_and_check_system_params(
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

        # write the predicted states in time_series_tensor
        time_series_tensor = self._evolve_system_states(
            time_series_tensor=time_series_tensor,
            initial_conditions=initial_conditions,
            sys_matrix=sys_matrix,
            control_matrix=self.control_matrix_ if self.is_controlled else None,
            control_input=control_input,
            time_values=time_values,
            eigenvalues=self.eigenvalues_ if self.is_spectral_mode else None,
            eigenvectors_left=self.eigenvectors_left_
            if hasattr(self, "eigenvectors_left_")
            else None,
            time_delta=time_delta,
        )

        return TSCDataFrame.from_tensor(
            time_series_tensor,
            time_series_ids=time_series_ids,
            feature_names=feature_names_out,
            time_values=time_values,
        )
