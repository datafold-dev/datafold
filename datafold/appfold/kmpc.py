import warnings
from typing import Any, List, Optional, Union

import numpy as np
from qpsolvers import solve_qp

from datafold.appfold import EDMDControl
from datafold.dynfold.base import InitialConditionType, TransformType
from datafold.utils.general import if1dim_colvec


class LinearKMPC:
    r"""
    Class to implement Lifting based Model Predictive Control

    Given a linear controlled model evolving at timestep :math:`k`

    .. math::
        x_{k+1}=Ax_k + Bu_k
        x \in \mathbb{R}^{N}; A \in \mathbb{R}^{[N \times N]};
        u \in \mathbb{R}^{m}; B \in \mathbb{R}^{[N \times m]};

    the class can construct control input :math:`u` to track a given reference
    :math:`y_r` (see :py:func:`.generate_control_signal`).

    The resulting control input is chosen such as to minimize

    .. math::
        J = z_{N_p}^TQ_{N_p}z_{N_p} + \sum_{k=0}^{N_p-1} z_k^T Q z_k + u_k^T R u_k

    and requiring :math:`z` to satisfy the state bounds and :math:`u` the input bounds

    Parameters
    ----------
    predictor : EDMDControl
        Prediction model to use, must be already fitted.
        See also: :py:class:`.EDMDControl`
    horizon : int
        prediction horizon, number of timesteps to predict, :math:`N_p`
    state_bounds : np.ndarray(shape=(n,2))
        [[x1max, x1min],
         ...
         [xnmax, xnmin]]
    input_bounds : np.ndarray(shape=(m,2))
        [[u1max, u1min],
         ...
         [ummax, ummin]]
    qois : List[str] | List[int], optional
        Quantities of interest - the state to be controlled via the reference.
        It is used to form matrix :math:`C \in \mathbb{R}^{[n \times N]}`,
        where :math:`n` is the number of quantities of interest and  :math:`N`
        is the lifted state size such that :math:`z=Cx`

        Example:
        ["z1",...,"zn"] or [index("z1"), ..., index("zn")]
    cost_running : float | np.ndarray(shape=(1,n)), optional, by default 0.1
        Quadratic cost of the state for internal time steps :math:`Q`.
        If float, the same cost will be applied to all state dimensions.
    cost_terminal : float | np.ndarray(shape=(1,n)), optional, by default 100
        Quadratic cost of the state at the end of the preidction :math:`Q_{N_p}`.
        If float, the same cost will be applied to all state dimensions.
    cost_input : float | np.ndarray(shape=(1,n)), optional, by default 0.01
        Quadratic cost of the input :math:`R`.
        If float, the same cost will be applied to all input dimensions.

    Attributes
    ----------

    H : np.ndarray
        Quadratic cost/weight term for the input.
    h : np.ndarray
        Linear cost/weight term for the input.
    G : np.ndarray
        Linear cost/weight term for the initial state.
    Y : np.ndarray
        Linear cost/weight term for the reference state.
    M : np.ndarray
        Linear constraint weight for the initial state.
    c : np.ndarray
        Linear constraint.

    """

    def __init__(
        self,
        predictor: EDMDControl,
        horizon: int,
        state_bounds: np.ndarray,
        input_bounds: np.ndarray,
        qois: Optional[Union[List[str], List[int]]] = None,
        cost_running: Optional[Union[float, np.ndarray]] = 0.1,
        cost_terminal: Optional[Union[float, np.ndarray]] = 100,
        cost_input: Optional[Union[float, np.ndarray]] = 0.01,
    ) -> None:
        # utilize the lifting functions from EDMD
        self.lifting_function = predictor.transform

        # define default values of properties
        self.horizon = horizon

        self.cost_running = cost_running
        self.cost_terminal = cost_terminal
        self.cost_input = cost_input

        self.A = predictor.sys_matrix
        self.B = predictor.control_matrix
        self.lifted_state_size, self.input_size = self.B.shape
        self.state_size = len(predictor.state_columns)

        # setup conversion from lifted state to output quantities of interest
        self.Cb, self.output_size = self._setup_qois(qois, predictor)

        try:
            self.input_bounds = input_bounds.reshape(self.input_size, 2)
            self.state_bounds = state_bounds.reshape(self.output_size, 2)
        except ValueError as e:
            raise ValueError("the bounds should be ") from e

        self.H, self.h, self.G, self.Y, self.L, self.M, self.c = self._setup_optimizer()

        # check for positive-definiteness, as the optimizer requires it.
        try:
            np.linalg.cholesky(self.H)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Cost matrix H is not positive-definite, using H^T@H instead."
            )
            self.H = np.dot(self.H.T, self.H)

    def _setup_qois(self, qois, predictor):

        # handle default case
        if qois is None:
            output_size = self.state_size
            C = np.zeros((output_size, self.lifted_state_size))
            C[:output_size, :output_size] = np.eye(output_size)
            Cb = np.kron(np.eye(self.horizon + 1), C)
            return Cb, output_size

        if not isinstance(qois, list):
            raise ValueError("qois should be a list of strings or ints")

        output_size = len(qois)
        if output_size <= 0 or output_size > self.state_size:
            raise ValueError("qois should satisfy 0 < len(qois) <= state_size.")
        qtype = str if isinstance(qois[0], str) else int

        C = np.zeros((output_size, self.lifted_state_size))
        ixs = []
        for qoi in qois:
            try:
                qoi = qtype(qoi)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Entries in qoi should be only strings or ints (is {qoi})."
                ) from e
            try:
                if qtype is str:
                    ixs.append(predictor.state_columns.index(qoi))
                else:
                    assert qoi < self.state_size
                    ixs.append(qoi)
            except (ValueError, AssertionError):
                raise ValueError(
                    f"Entries in qois should be contained in the predictor state (is {qoi})."
                )

        for i, ix in enumerate(ixs):
            C[i, ix] = 1.0

        Cb = np.kron(np.eye(self.horizon + 1), C)

        return Cb, output_size

    def _setup_optimizer(self):
        # implements relevant parts of korda-mezic-2018 for setting up the optimization problem

        Ab, Bb = self._create_evolution_matrices()
        Q, q, R, r = self._create_cost_matrices()
        F, E, c = self._create_constraint_matrices()

        H = R + Bb.T @ Q @ Bb
        h = r + Bb.T @ q
        G = 2 * Ab.T @ Q @ Bb
        Y = 2 * Q @ Bb

        L = F + E @ Bb
        M = E @ Ab

        return H, h, G, Y, L, M, c

    def _create_evolution_matrices(self):
        # implemenets appendix from korda-mezic-2018
        # same as Sabin 2.44
        Np = self.horizon
        N = self.lifted_state_size
        m = self.input_size
        A = self.A
        B = self.B
        Ab = np.eye((Np + 1) * N, N)
        Bb = np.zeros(((Np + 1) * N, Np * m))

        for i in range(Np):
            Ab[(i + 1) * N : (i + 2) * N, :] = A @ Ab[i * N : (i + 1) * N, :]
            Bb[(i + 1) * N : (i + 2) * N, i * m : (i + 1) * m] = B
            if i > 0:
                Bb[(i + 1) * N : (i + 2) * N, : i * m] = (
                    A @ Bb[i * N : (i + 1) * N, : i * m]
                )

        # transform the evolution matrices from the lifted to the referenced state
        return self.Cb @ Ab, self.Cb @ Bb

    def _create_constraint_matrices(self):
        # implemenets appendix from korda-mezic-2018
        # same as Sabin 2.44, assuming
        # bounds vector is ordered [zmax; -zmin; umax; -umin]
        Np = self.horizon
        N = self.output_size
        m = self.input_size

        # constraint equations
        E = np.vstack([np.eye(N), -np.eye(N), np.zeros((2 * m, N))])
        Eb = np.kron(np.eye(Np + 1), E)

        F = np.vstack([np.zeros((2 * N, m)), np.eye(m), -np.eye(m)])
        Fb = np.vstack([np.kron(np.eye(Np), F), np.zeros((2 * (N + m), m * Np))])

        # constraint
        c = np.zeros((1, 2 * (N + m)))
        c[:, :N] = self.state_bounds[:, 0]
        c[:, N : 2 * N] = -self.state_bounds[:, 1]
        c[:, 2 * N : 2 * N + m] = self.input_bounds[:, 0]
        c[:, 2 * N + m :] = -self.input_bounds[:, 1]
        c = c.T
        cb = np.tile(c, (Np + 1, 1))

        return Fb, Eb, cb

    def _cost_to_array(self, cost: Union[float, np.ndarray], N: int):
        if isinstance(cost, np.ndarray):
            try:
                cost = cost.flatten()
                assert len(cost) == N
            except AssertionError:
                raise ValueError(f"Cost should have length {N}, received {len(cost)}.")
            return cost
        else:
            try:
                cost = float(cost)
            except:
                raise ValueError(
                    f"Cost must be numeric value or array, received {cost}."
                )
            return np.ones(N) * cost

    def _create_cost_matrices(self):
        # implemenets appendix from korda-mezic-2018
        # same as Sabin 2.44
        Np = self.horizon
        N = self.output_size
        m = self.input_size

        # optimization - linear
        # assume linear optimization term is 0
        # TODO: improve or set h = 0
        q = np.zeros((N * (Np + 1), 1))
        r = np.zeros((m * Np, 1))

        # optimization - quadratic
        # assuming only autocorrelation
        vec_running = self._cost_to_array(self.cost_running, N)
        vec_terminal = self._cost_to_array(self.cost_terminal, N)
        vec_input = self._cost_to_array(self.cost_input, m)

        Qb = np.diag(np.hstack([np.tile(vec_running, Np), vec_terminal]))

        Rb = np.diag(np.tile(vec_input, Np))

        return Qb, q, Rb, r

    def generate_control_signal(
        self, initial_conditions: InitialConditionType, reference: TransformType
    ) -> np.ndarray:
        """
        Method to generate a control sequence, given some initial conditions and a reference trajectory,
        as in Korda-Mezic, Algorithm 1. This method solves thefollowing optimization problem (Korda-Mezic, Equation 24).

        .. math::
            \text{minimize : } U^{T} H U^{T} + h^{T} U + z_0^{T} GU - y_{r}^{T} U
            \text{subject to : } LU + Mz_{0} \leq c
            \text{parameter : } z_{0} = \Psi(x_{k})

        Here, :math:`U` is the optimal control sequence to be estimated.

        Parameters
        ----------
        initial_conditions : TSCDataFrame or np.ndarray
            Initial conditions for the model
        reference : np.ndarray
            Reference trajectory. Required to optimize the control sequence

        Returns
        -------
        U : np.ndarray(shape=(horizon,1))
            Sequence of control inputs.

        Raises
        ------
        ValueError
            In case of mis-shaped input
        """
        z0 = self.lifting_function(initial_conditions)
        z0 = if1dim_colvec(z0)
        z0 = if1dim_colvec(np.array(z0))

        try:
            z0 = z0.reshape(self.lifted_state_size, 1)
        except ValueError as e:
            raise ValueError(
                "The initial state should match the shape of the system state before the lifting."
            ) from e

        try:
            yr = np.array(reference)
            assert yr.shape[1] == self.output_size
            yr = yr.reshape(((self.horizon + 1) * self.output_size, 1))
        except:
            raise ValueError(
                "The reference signal should be a frame or array with n (output_size) columns and  Np (prediction horizon) rows."
            )

        U = solve_qp(
            P=2 * self.H,
            q=(self.h.T + z0.T @ self.G - yr.T @ self.Y).flatten(),
            G=self.L,
            h=(self.c - self.M @ z0).flatten(),
            A=None,
            b=None,
            solver="quadprog",
        )

        if U is None:
            raise ValueError("The solver did not converge.")

        return U

    def compute_cost(self, U, reference, initial_conditions):
        z0 = self.lifting_function(initial_conditions)
        z0 = if1dim_colvec(z0)

        try:
            z0 = z0.to_numpy().reshape(self.lifted_state_size, 1)
        except ValueError as e:
            raise ValueError(
                "The initial state should match the shape of the system state before the lifting."
            ) from e

        try:
            yr = np.array(reference)
            assert yr.shape[1] == self.output_size
            yr = yr.reshape(((self.horizon + 1) * self.output_size, 1))
        except:
            raise ValueError(
                "The reference signal should be a frame or array with n (output_size) columns and  Np (prediction horizon) rows."
            )

        U = U.reshape(-1, 1)
        e1 = U.T @ self.H @ U
        e2 = self.h.T @ U
        e3 = z0.T @ self.G @ U
        e4 = -yr.T @ self.Y @ U

        return (e1 + e2 + e3 + e4)[0, 0]
