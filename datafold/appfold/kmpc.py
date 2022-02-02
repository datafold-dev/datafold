from multiprocessing.sharedctypes import Value
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from qpsolvers import solve_qp

from datafold.appfold import EDMDControl


class KoopmanMPC:
    """
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
    input_bounds : np.ndarray(shape=(n,2))
        [[u1max, u1min],
         ...
         [unmax, unmin]]
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
        # take some properties/methods from the other class
        self.lifting_function = predictor.transform

        # define default values of properties
        self.horizon = horizon

        self.cost_running = cost_running
        self.cost_terminal = cost_terminal
        self.cost_input = cost_input

        self.A = predictor.sys_matrix
        self.B = predictor.control_matrix
        self.lifted_state_size, self.input_size = self.B.shape

        # setup conversion from lifted state to output quantities of interest
        self.Cb, self.output_size = self._setup_qois(predictor, qois)

        try:
            self.input_bounds = input_bounds.reshape(self.input_size, 2)
            self.state_bounds = state_bounds.reshape(self.output_size, 2)
        except ValueError as e:
            raise ValueError("the bounds should ") from e

        self.H, self.h, self.G, self.L, self.M, self.c = self._setup_optimizer()

    def _setup_qois(self, predictor, qois):

        # handle default case
        if qois is None:
            output_size = len(predictor.state_columns)
            C = np.zeros(output_size, self.lifted_state_size)
            C[:output_size, :output_size] = np.eye(output_size)
            Cb = np.kron(np.eye(self.horizon + 1), C)
            return Cb, output_size

        if not isinstance(qois, list):
            raise ValueError("qois should be a list of strings or ints")

        output_size = len(qois)
        if output_size <= 0 or output_size > len(predictor.state_columns):
            raise ValueError(
                "qois should satisfy 0 < len(qois) <= len(predictor.state_columns)."
            )
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
                    assert qoi < len(predictor.state_columns)
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
        L = F + E @ Bb
        M = E @ Ab

        return H, h, G, L, M, c

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

    def generate_control_signal(self, trajectory, reference):
        # implement the generation of control signal as in
        Np = self.horizon
        N = self.lifted_state_size
        m = self.input_size

        H, h, G, L, M, c = self._setup_optimizer()
        z0 = self.lifting_function(trajectory).T
        y = np.reshape(reference.T, ((Np + 1) * reference.shape[1], 1))

        U = solve_qp(
            P=H,
            q=(h.T + z0 @ G),
            G=L,
            h=(c - M @ z0),
            A=None,
            b=None,
            solver="quadprog",
        )

        return U
