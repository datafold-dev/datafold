import math
import numpy as np
from qpsolvers import solve_qp
import quadprog
import sys

class KoopmanMPC:
    """
    Class to implement Lifting based Model Predictive Control
    """

    def __init__(self, model, system_matrix, control_matrix, edmd_basis) -> None:
        # take some properties/methods from the other class
        self.model = model
        self.lifting_function = edmd_basis

        # define default values of properties
        self.horizon = math.floor(1 / self.model.dt)

        self.input_bounds = np.array([-12, 12])  # [min,max] can be 1x2 or mx2
        self.state_bounds = np.array([-10, 10])  # [min,max] can be 1x2 or nx2

        self.cost_running = 0.1
        self.cost_terminal = 100
        self.cost_input = 0.01
        self.projmtx = self.model.C_select
        self.A = system_matrix
        self.B = control_matrix
        self.C = []
        self.Q, self.R, self.H, self.G, self.D = []
        self.constraints = []

    def _create_predictor_matrix(self):
        """
        Generate matrices A,B
        """
        return self.A, self.B

    def _create_cost_matrix(self):
        """
        Generate matrices Q,R
        """
        self.C = np.zeros((self.projmtx.shape[0], self.model.Nlift))
        self.C = np.kron(np.eye(self.horizon + 1), self.C)
        num_projections = np.size(self.C , 0)

        Q = np.kron(np.eye(self.horizon + 1) , np.eye(num_projections) * self.cost_running)
        endr, endc = Q.shape
        Q[endr-num_projections: , endc-num_projections:] = np.eye(num_projections) * self.cost_terminal
        R = np.kron(np.eye(self.horizon) , np.eye(self.model.m) * self.cost_input)

        H = self.B.T @ self.C.T @ Q @ self.C @ self.B + R
        G = 2 * self.A.T @ self.C.T @ Q @ self.C @ self.B
        D = -2 * Q @ self.C @ self.B

        self.Q = Q
        self.R = R
        self.H = H
        self.G = G
        self.D = D

    def _create_constraint_matrix(self):
        """
        Generate matrices E, F, constraints
        """



    def _optimize_cost(self, P, Q, G, H, A, B):
        """
        Optimize matrices Q,R
        """

        G_optimized = 0.5 * (P + P.T)
        A_optimized = -Q
        C_optimized = -np.vstack([A, G]).T
        B_optimized = -np.vstack([B, H]).T
        m_eq = A.shape[0]

        return quadprog.solve_qp(G_optimized, A_optimized, C_optimized, B_optimized, m_eq)[0]

    def generate_control_signal(self, trajectory, reference):
        Np = self.horizon

        if reference.shape[1] > Np + 1:
            reference = reference[:Np+1, :]
        elif reference.shape[1] < Np + 1:
            temp = np.kron(np.ones((Np+1, 1)), reference[-1,:])
            temp[:reference.shape[0], :] = reference
            reference = temp

        z0 = self.lifting_function(trajectory).T

        y = np.reshape(reference.T, ((Np+1) * reference.shape[1], 1))

        P = 2 * self.H
        q = (z0.T @ self.G + y.T @ self.D).T
        G = self.G
        h = -self.M @ z0 + self.C
        A = self.A
        b = self.B
        Q = self.Q

        U = solve_qp(P, q, G, h, A, b, solver='quadprog')

        return U

