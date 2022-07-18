import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from datafold.appfold import EDMD
from datafold.dynfold.base import InitialConditionType, TransformType
from datafold.utils.general import if1dim_colvec

try:
    import quadprog  # noqa: F401
    from qpsolvers import solve_qp
except ImportError:
    solve_qp = None


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
    predictor : EDMD(dmd_model=DMDControl())
        Prediction model to use, must be already fitted.
        The underlying DMDModel should be based on a LinearControlledDynamicalSystem
        See also: :py:class:`.EDMD`, :py:class:`.DMDControl`
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

    References
    ----------

    :cite:`korda-2018`
    """

    def __init__(
        self,
        predictor: EDMD,
        horizon: int,
        state_bounds: np.ndarray,
        input_bounds: np.ndarray,
        qois: Optional[Union[List[str], List[int]]] = None,
        cost_running: Optional[Union[float, np.ndarray]] = 0.1,
        cost_terminal: Optional[Union[float, np.ndarray]] = 100,
        cost_input: Optional[Union[float, np.ndarray]] = 0.01,
    ) -> None:
        if solve_qp is None:
            raise ImportError(
                "The optional dependencies qpsolvers and quadprog are required "
                "for LinearKMPC. They can be installed using option mpc. "
                "E.g. `pip install datafold[mpc]`"
            )

        # utilize the lifting functions from EDMD
        self.lifting_function = predictor.transform

        # define default values of properties
        self.horizon = horizon

        self.cost_running = cost_running
        self.cost_terminal = cost_terminal
        self.cost_input = cost_input

        self.A = predictor.dmd_model.sys_matrix_
        self.B = predictor.dmd_model.control_matrix_
        try:
            self.lifted_state_size, self.input_size = self.B.shape
        except ValueError:
            raise TypeError(
                "The shape of the control matrix is not as expected. "
                "This is likely due to an incompatible dmd_model type of the predictor. "
                "The predictor.dmd_model should support linear controlled systems "
                "(e.g. DMDControl)."
            )
        self.state_size = len(predictor.feature_names_in_)

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
                    ixs.append(list(predictor.feature_names_in_).index(qoi))
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
        # implements relevant part of :cite:`korda-2018` to set up the optimization problem

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
        # implemenets appendix from :cite:`korda-2018`
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
        # implemenets appendix from :cite:`korda-2018`
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
        # implemenets appendix from :cite:`korda-2018`
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
        r"""
        Method to generate a control sequence, given some initial conditions and
        a reference trajectory, as in :cite:`korda-2018` , Algorithm 1. This
        method solves the following optimization problem (:cite:`korda-2018` ,
        Equation 24).

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
        U : np.ndarray(shape=(horizon,m))
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
                "The initial state should match the shape of the system state "
                "before the lifting."
            ) from e

        try:
            yr = np.array(reference)
            assert yr.shape[1] == self.output_size
            yr = yr.reshape(((self.horizon + 1) * self.output_size, 1))
        except:
            raise ValueError(
                "The reference signal should be a frame or array with n (output_size) "
                "columns and  Np (prediction horizon) rows."
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

        return U.reshape((-1, self.input_size))

    def compute_cost(self, U, reference, initial_conditions):
        z0 = self.lifting_function(initial_conditions)
        z0 = if1dim_colvec(z0)

        try:
            z0 = z0.to_numpy().reshape(self.lifted_state_size, 1)
        except ValueError as e:
            raise ValueError(
                "The initial state should match the shape of "
                "the system state before the lifting."
            ) from e

        try:
            yr = np.array(reference)
            assert yr.shape[1] == self.output_size
            yr = yr.reshape(((self.horizon + 1) * self.output_size, 1))
        except:
            raise ValueError(
                "The reference signal should be a frame or array with n (output_size) "
                "columns and  Np (prediction horizon) rows."
            )

        U = U.reshape(-1, 1)
        e1 = U.T @ self.H @ U
        e2 = self.h.T @ U
        e3 = z0.T @ self.G @ U
        e4 = -yr.T @ self.Y @ U

        return (e1 + e2 + e3 + e4)[0, 0]


class AffineKgMPC(object):
    def __init__(
        self,
        predictor: EDMD,
        horizon: int,
        input_bounds: np.ndarray,
        cost_state: Optional[Union[float, np.ndarray]] = 1,
        cost_input: Optional[Union[float, np.ndarray]] = 1,
        interpolation="cubic",
        ivp_method="RK23",
    ):
        r"""
        Class to implement Lifting based (Koopman generator) Model Predictive Control,
        given a controll-affine model in differential form

        .. math::
            \dot{x}= Ax + \sum_{i=1}^m B_i u_i x
            x \in \mathbb{R}^{N}; A \in \mathbb{R}^{[N \times N]};
            u \in \mathbb{R}^{m}; B_i \in \mathbb{R}^{[N \times N]} for i=[1,...,m];

        the class can construct control input :math:`u` to track a given reference
        :math:`x_r` (see :py:func:`.generate_control_signal`).
        :math:`u` and :math:`x_r` are discretized with timestep :math:`k`, however internally
        the differential equations use an adaptive time-step scheme based on interpolation.

        The resulting control input is chosen such as to minimize

        .. math::
            J = \sum_{k=0}^{N_p} || Q * (x^{(k)}-x_r^{(k)})||^2 + ||R u_k||^2

        and requiring :math:`u` the input bounds

        Parameters
        ----------
        predictor : EDMD(dmd_model=gDMDAffine)
            Prediction model to use, must be already fitted.
            The underlying DMDModel should be based on a linear system with affine control
            See also: :py:class:`.EDMD`, :py:class:`.gDMDAffine`
        horizon : int
            prediction horizon, number of timesteps to predict, :math:`N_p`
        input_bounds : np.ndarray(shape=(m,2))
            [[u1max, u1min],
            ...
            [ummax, ummin]]
        cost_state : float | np.ndarray(shape=(n,1)), optional, by default 1
            Linear cost of the state  :math:`Q`.
            If float, the same cost will be applied to all state dimensions.
        cost_input : float | np.ndarray(shape=(m,1)), optional, by default 1
            Linear cost of the input :math:`R`.
            If float, the same cost will be applied to all input dimensions.
        interpolation: string (default 'cubic')
            Interpolation type passed to `scipy.interpolate.interp1d`
        ivp_method: string (default 'RK23')
            Initial value problem solution scheme passed to `scipy.integrate.solve_ivp`

        References
        ----------

        :cite:`peitz-2020`
        """
        self.horizon = horizon

        self.predictor = predictor
        self.L = predictor.dmd_model.sys_matrix_
        self.B = predictor.dmd_model.control_matrix_
        try:
            self.lifted_state_size, _, self.input_size = self.B.shape
        except ValueError:
            raise TypeError(
                "The shape of the control tensor is not as expected "
                "(n_state,n_state,n_control). This is likely due to an incompatible "
                "dmd_model type of the predictor. The predictor.dmd_model should "
                "support affine controlled systems (e.g. gDMDAffine)."
            )

        if not self.predictor._dmd_model.is_differential_system:
            raise TypeError(
                "The predictor.dmd_model should be a differential "
                "affine controlled system (e.g. gDMDAffine)."
            )

        self.state_size = len(predictor.feature_names_in_)

        if input_bounds.shape != (self.input_size, 2):
            raise ValueError(
                f"input_bounds is of shape {input_bounds.shape}, "
                f"should be ({self.input_size},2)"
            )

        if isinstance(cost_input, np.ndarray):
            try:
                self.cost_input = cost_input.reshape((self.input_size, 1))
            except ValueError:
                raise ValueError(
                    f"cost_input is of shape {cost_input.shape}, "
                    f"should be ({self.input_size},1)"
                )
        else:
            self.cost_input = cost_input * np.ones((self.input_size, 1))

        if isinstance(cost_state, np.ndarray):
            try:
                self.cost_state = cost_state.reshape((self.state_size, 1))
            except ValueError:
                raise ValueError(
                    f"cost_state is of shape {cost_state.shape}, "
                    f"should be ({self.state_size},1)"
                )
        else:
            self.cost_state = cost_state * np.ones((self.state_size, 1))

        self.input_bounds = np.repeat(
            np.fliplr(input_bounds).T, self.horizon + 1, axis=1
        ).T

        self._cached_init_state = None
        self._cached_lifted_state = None

        self.interpolation = interpolation
        self.ivp_method = ivp_method

    def _predict(self, x0, u, t):
        # -> shape (self.lifted_state_size, self.horizon+1)
        # if (self._cached_input != u).any() or (self._cached_state != x0).any:
        if (self._cached_init_state != x0).any():
            lifted_x0 = self.predictor.transform(
                pd.DataFrame(x0.T, columns=self.predictor.feature_names_in_)
                # InitialCondition.from_array(x0.T, self.predictor.feature_names_in_)
            ).to_numpy()
            self._cached_init_state = x0
            self._cached_lifted_state = lifted_x0
        else:
            lifted_x0 = self._cached_lifted_state

        interp_u = interp1d(t, u, axis=1, kind=self.interpolation)

        affine_system_func = lambda t, state: (
            self.L @ state + self.B @ interp_u(t) @ state
        )

        ivp_solution = solve_ivp(
            affine_system_func,
            t_span=(t[0], t[-1]),
            y0=lifted_x0.ravel(),
            t_eval=t,
            method=self.ivp_method,
            max_step=np.min(np.diff(t)),
        )

        if not ivp_solution.success:
            raise RuntimeError(
                f"The system could not be envolved for the "
                f"requested timespan for initial condition {x0}."
            )

        return ivp_solution.y

    def cost_and_jacobian(
        self,
        u: np.ndarray,
        x0: np.ndarray,
        xref: np.ndarray,
        t: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Calculate the cost and its Jacobian

        Parameters
        ----------
        u : np.array
            Control input
            shape = (n*m,)
            [u1(t0) u1(t1) ... u1(tn) u2(t1) ... um(tn)]
            with `n = self.horizon+1`; `m = self.input_size`
        x0 : np.array
            Initial conditions
            shape = `(self.state_size, 1)`
        xref : np.array
            Reference state
            shape = `(self.state_size, 1)`
        t : np.array
            Time values for the evaluation

        Returns
        -------
        (float, np.array)
            cost, jacobian
        """
        u = u.reshape(self.input_size, self.horizon + 1)
        x = self._predict(x0, u, t)
        cost = self.cost(u, x0, xref, t, x[: self.state_size, :])
        jacobian = self.jacobian(u, x0, xref, t, x)
        return (cost, jacobian)

    def cost(
        self,
        u: np.ndarray,
        x0: np.ndarray,
        xref: np.ndarray,
        t: np.ndarray,
        x: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the cost for the given reference

        Parameters
        ----------
        u : np.array
            Control input
            shape = (n*m,)
            [u1(t0) u1(t1) ... u1(tn) u2(t1) ... um(tn)]
            with `n = self.horizon+1`; `m = self.input_size`
        x0 : np.array
            Initial conditions
            shape = `(self.state_size, 1)`
        xref : np.array
            Reference state
            shape = `(self.state_size, 1)`
        t : np.array
            Time values for the evaluation
        x : Optional[np.array]
            State to use. If not provided is calculated by self.predictor
        Returns
        ---------
        float
        """
        u = u.reshape(self.input_size, self.horizon + 1)
        if x is None:
            x = self._predict(x0, u, t)[: self.state_size, :]
        Lhat = np.linalg.norm(self.cost_state * (x - xref), axis=0) ** 2
        Lhat += np.linalg.norm(self.cost_input * u, axis=0) ** 2
        J = np.sum(Lhat)
        return J

    def jacobian(
        self,
        u: np.ndarray,
        x0: np.ndarray,
        xref: np.ndarray,
        t: np.ndarray,
        x: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the Jacobian of the cost function

        Parameters
        ----------
        u : np.array
            shape = (n*m,)
            [u1(t0) u1(t1) ... u1(tn) u2(t1) ... um(tn)]
            with `n = self.horizon+1`; `m = self.input_size`
        x0 : np.array
            shape = `(self.state_size, 1)`
        xref : np.array
            shape = `(self.state_size, 1)`
        t : np.array
            Time values for the evaluation
        x : Optional[np.array]
            State to use. If not provided is calculated by self.predictor
        Returns
        ---------
        array_like, shape (n*m,)
        [dJ/du1(t0) dJ/du1(t1) ... dJ/du1(tn) dJ/du2(t1) ... dJ/dum(tn)]
        """
        u = u.reshape(self.input_size, self.horizon + 1)
        if x is None:
            x = self._predict(x0, u, t)  # shape = (lifted_state_size, horizon+1)
        interp_gamma = interp1d(t, self._dcost_dx(x, xref), axis=1, kind="previous")
        interp_u = interp1d(t, u, axis=1, kind="previous")

        lambda_adjoint = self._compute_lambda_adjoint(
            interp_u, interp_gamma, t
        )  # shape = (lifted_state_size,horizon+1)
        rho = self._dcost_du(u)  # shape = (input_size, horizon+1)

        # self.B.shape = (lifted_state_size, lifted_state_size, input_size)
        # (x.T @ self.B.T).shape = (input_size, horizon+1, lifted_state_size)
        # einsum(...).shape = rho.shape = (input_size, horizon+1)
        jac = np.einsum("ijk,kj->ij", x.T @ self.B.T, lambda_adjoint) + rho

        return jac.ravel()

    def _lambda_ajoint_ODE(self, t, y, interp_u, interp_gamma):
        return (
            -interp_gamma(t)
            - self.L.T @ y
            - np.einsum("ijk,i->jk", self.B.T, interp_u(t)) @ y
        )

    def _compute_lambda_adjoint(self, interp_u, interp_gamma, t):
        sol = solve_ivp(
            self._lambda_ajoint_ODE,
            y0=np.zeros(self.lifted_state_size),
            t_span=[t[-1], t[0]],
            t_eval=t[::-1],
            args=(interp_u, interp_gamma),
            method=self.ivp_method,
            max_step=np.min(np.diff(t)),
        )

        if not sol.success:
            raise RuntimeError(
                "Could not integrate the adjoint dynamics. Solver says '{sol.message}'"
            )
        return np.fliplr(sol.y)

    def _dcost_dx(self, x, xref):
        # gamma(t0:te)
        gamma = np.zeros((self.lifted_state_size, self.horizon + 1))
        gamma[: self.state_size, :] = (
            2 * self.cost_state * (x[: self.state_size] - xref)
        )
        return gamma

    def _dcost_du(self, u):
        # rho(t0:te)
        return 2 * self.cost_input * u

    def generate_control_signal(
        self,
        initial_conditions: InitialConditionType,
        reference: TransformType,
        time_values: Optional[np.ndarray] = None,
        **minimization_kwargs,
    ) -> np.ndarray:
        r"""
        Method to generate a control sequence, given some initial conditions and a reference
        trajectory, as in :cite:`peitz-2020` , Section 4.1. This method solves the following
        optimization problem (:cite:`peitz-2020` , Equation K-MPC).

        .. math::
            \text{given : } x_{0}, x_r
            \text{find :} u
            \text{minimizing : } J(x_0,x_r,u)
            \text{subject to : } \dot{x}= Ax + \sum_{i=1}^m B_i u_i x


        Here, :math:`u` is the optimal control sequence to be estimated.

        The optimization is done using scipy.optimize.minimize. Suitable methods use
        a calculated jacobian (but not hessian) and support bounds on the variable.

        Parameters
        ----------
        initial_conditions : TSCDataFrame or np.ndarray
            Initial conditions for the model
        reference : TSCDataFrame or np.ndarray
            Reference trajectory. Required to optimize the control sequence.
            If TSCDataFrame and time_values is not provided, the time index is used.
        time_vlues : Optional[np.ndarray]
            Time values of the the reference trajectory at which control inputs will
            be generated. If not provided is tried to be inferred from the reference.
        **minimization_kwargs:
            Passed to scipy.optimize.minimize. If method is not provided, 'L-BFGS-B' is used.
        Returns
        -------
        U : np.ndarray(shape=(horizon,m))
            Sequence of control inputs.

        Raises
        ------
        ValueError
            In case of mis-shaped input
        """
        # x0.shape = (nc,1)
        # xref.shape = (nc,n)
        # t.shape = (n,)
        if time_values is None:
            try:
                time_values = reference.time_values()
            except AttributeError:
                raise TypeError(
                    "If time_values is not provided, "
                    "the reference needs to be of type TSCDataFrame"
                )

        if time_values.shape != (self.horizon + 1,):
            raise ValueError(
                f"time_values is of shape {time_values.shape}, should be ({self.horizon+1},)"
            )

        xref = (
            reference if isinstance(reference, np.ndarray) else reference.to_numpy().T
        )
        xref = if1dim_colvec(xref)
        if xref.shape != (self.state_size, self.horizon + 1):
            raise ValueError(
                f"reference is of shape {reference.shape}, "
                f"should be ({self.state_size},{self.horizon+1})"
            )

        x0 = (
            initial_conditions
            if isinstance(initial_conditions, np.ndarray)
            else initial_conditions.to_numpy().T
        )
        x0 = if1dim_colvec(x0)
        if x0.shape != (self.state_size, 1):
            raise ValueError(
                f"initial_conditions is of shape {initial_conditions.shape}, "
                f"should be ({self.state_size},1)"
            )

        try:
            method = minimization_kwargs.pop("method")
        except KeyError:
            method = "L-BFGS-B"

        u_init = np.zeros(
            (self.input_size, self.horizon + 1)
        ).ravel()  # [u1(t0) u1(t1) ... u1(tn-1) u2(t0) ... um(t1)]
        xref = np.copy(xref)
        xref[
            :, 0
        ] = x0.ravel()  # we can't change the initial state no matter how much we try
        res = minimize(
            fun=self.cost_and_jacobian,
            x0=u_init,
            args=(x0, xref, time_values),
            method=method,
            jac=True,
            bounds=self.input_bounds,
            **minimization_kwargs,
        )

        if not res.success:
            warnings.warn(
                f"Could not find a minimum solution. Solver says '{res.message}'. "
                "Using closest solution."
            )

        return res.x.reshape(self.input_size, self.horizon + 1).T[:-1, :]
