from typing import Any, Optional, Union

import numpy as np
from qpsolvers import solve_qp

from datafold.appfold import EDMDControl


class KoopmanMPC:
    """
    Class to implement Lifting based Model Predictive Control

    [summary]

    Parameters
    ----------
    predictor : EDMDControl
        [description]
    horizon : float
        [description]
    state_bounds : np.ndarray
        [description]
    input_bounds : np.ndarray
        [description]
    cost_running : Optional[Union[float,np.ndarray]], optional
        [description], by default 0.1
    cost_terminal : Optional[Union[float,np.ndarray]], optional
        [description], by default 100
    cost_input : Optional[Union[float,np.ndarray]], optional
        [description], by default 0.01
    """

    def __init__(
        self,
        predictor: EDMDControl,
        horizon: float,
        state_bounds: np.ndarray,
        input_bounds: np.ndarray,
        cost_running: Optional[Union[float, np.ndarray]] = 0.1,
        cost_terminal: Optional[Union[float, np.ndarray]] = 100,
        cost_input: Optional[Union[float, np.ndarray]] = 0.01,
    ) -> None:
        # take some properties/methods from the other class
        self.lifting_function = predictor.transform

        # define default values of properties
        self.horizon = horizon

        self.input_bounds = input_bounds  # [min,max] can be 1x2 or mx2
        self.state_bounds = state_bounds  # [min,max] can be 1x2 or nx2

        self.cost_running = cost_running
        self.cost_terminal = cost_terminal
        self.cost_input = cost_input

        self.A = predictor.sys_matrix
        self.B = predictor.control_matrix

        self.lifted_state_size = self.B.shape[0]
        self.input_size = self.B.shape[1]

        self.H, self.h, self.G, self.L, self.M, self.c = self._setup_optimizer()

    def _setup_optimizer(self):
        # implements relevant parts of korda-mezic-2018 for setting up the optimization problem
        Np = self.horizon
        m = self.input_size

        Ab, Bb = self._create_bold_AB()
        F, Q, q, R, r, E = self.create_bold_FQRE()

        H = R + Bb.T @ Q @ Bb
        h = r + Bb.T @ q
        G = 2 * Ab.T @ Q @ Bb
        L = F + E @ Bb
        M = E @ Ab
        c = np.zeros((m * Np, 1))

        return H, h, G, L, M, c

    def _create_bold_AB(self):
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

        return Ab, Bb

    def _create_bold_FQRE(self):
        # implemenets appendix from korda-mezic-2018
        # same as Sabin 2.44, assuming
        # bounds vector is ordered [zmax, -zmin, umax, -umin]
        Np = self.horizon
        N = self.lifted_state_size
        m = self.input_size

        # constraints
        E = np.vstack([np.eye(N), -np.eye(N), np.zeros((2 * m, N))])
        Eb = np.kron(np.eye(Np + 1), E)

        F = np.vstack([np.zeros((2 * N, m)), np.eye(m), -np.eye(m)])
        Fb = np.vstack([np.kron(np.eye(Np), F), np.zeros((2 * (N + m), m * Np))])

        # optimization - linear
        # assume linear optimization term is 0
        # TODO: improve or set h = 0
        q = np.zeros((N * (Np + 1), 1))
        r = np.zeros((m * Np, 1))

        # optimization - quadratic
        # assuming only autocorrelation
        Qb = np.diag(
            np.hstack(
                [np.ones(Np * N) * self.cost_running, np.ones(N) * self.cost_terminal]
            )
        )

        Rb = np.diag(np.ones(Np * m) * self.cost_input)

        return Fb, Qb, q, Rb, r, Eb

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
