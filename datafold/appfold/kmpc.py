import math
import sys
from typing import Any, Optional, Union

import numpy as np
import quadprog
import scipy as sp
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
        Ab, Bb = self._create_bold_AB()
        F, Q, q, R, r, E = self.create_bold_FQRE()

        # TODO:
        pass

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

    # TODO: remove
    def _create_cost_matrix(self):
        """
        Generate matrices Q,R
        """
        self.C = np.zeros((self.projmtx.shape[0], self.model.Nlift))
        self.C = np.kron(np.eye(self.horizon + 1), self.C)
        num_projections = np.size(self.C, 0)

        Q = np.kron(
            np.eye(self.horizon + 1), np.eye(num_projections) * self.cost_running
        )
        endr, endc = Q.shape
        Q[endr - num_projections :, endc - num_projections :] = (
            np.eye(num_projections) * self.cost_terminal
        )
        R = np.kron(np.eye(self.horizon), np.eye(self.model.m) * self.cost_input)

        H = self.B.T @ self.C.T @ Q @ self.C @ self.B + R
        G = 2 * self.A.T @ self.C.T @ Q @ self.C @ self.B
        D = -2 * Q @ self.C @ self.B

        self.Q = Q
        self.R = R
        self.H = H
        self.G = G
        self.D = D

    # TODO: remove
    def _optimize_cost(self, P, Q, G, H, A, B):
        """
        Optimize matrices Q,R
        """

        G_optimized = 0.5 * (P + P.T)
        A_optimized = -Q
        C_optimized = -np.vstack([A, G]).T
        B_optimized = -np.vstack([B, H]).T
        m_eq = A.shape[0]

        return quadprog.solve_qp(
            G_optimized, A_optimized, C_optimized, B_optimized, m_eq
        )[0]

    def generate_control_signal(self, trajectory, reference):
        # TODO: reimplement
        Np = self.horizon

        if reference.shape[1] > Np + 1:
            reference = reference[: Np + 1, :]
        elif reference.shape[1] < Np + 1:
            temp = np.kron(np.ones((Np + 1, 1)), reference[-1, :])
            temp[: reference.shape[0], :] = reference
            reference = temp

        z0 = self.lifting_function(trajectory).T

        y = np.reshape(reference.T, ((Np + 1) * reference.shape[1], 1))

        P = 2 * self.H
        q = (z0.T @ self.G + y.T @ self.D).T
        G = self.G
        h = -self.M @ z0 + self.C
        A = self.A
        b = self.B
        Q = self.Q

        U = solve_qp(P, q, G, h, A, b, solver="quadprog")

        return U
