import math
import sys

import numpy as np
import quadprog
from appfold import EDMDControl
from qpsolvers import solve_qp


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
    cost_running : float, optional
        [description], by default 0.1
    cost_terminal : float, optional
        [description], by default 100
    cost_input : float, optional
        [description], by default 0.01
    """

    def __init__(
        self,
        predictor: EDMDControl,
        horizon: float,
        state_bounds: np.ndarray,
        input_bounds: np.ndarray,
        cost_running: float = 0.1,
        cost_terminal: float = 100,
        cost_input: float = 0.01,
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
        # TODO:
        pass

    def _create_bold_FQRE(self):
        # TODO:
        pass

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
