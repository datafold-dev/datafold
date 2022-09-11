import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datafold.appfold import EDMD
from datafold.dynfold.base import InitialConditionType, TransformType
from datafold.utils.general import if1dim_colvec, if1dim_rowvec

try:
    import quadprog  # noqa: F401
    from qpsolvers import solve_qp
except ImportError:
    solve_qp = None


class LinearKMPC:
    r"""Class to implement Lifting based Model Predictive Control.

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

    and requiring :math:`z` to satisfy the state bounds and :math:`u` the input bounds.

    Parameters
    ----------
    predictor : EDMD
        Prediction model to use, must be already fitted.
        The underlying DMD model in :py:class:`EDMD` must support control, such as
        :py:class:`.DMDControl`.

    horizon : int
        prediction horizon, number of time steps to predict, :math:`N_p`.

    state_bounds : np.ndarray(shape=(n,2))
        The min/max bounds of the system states:

        .. code::

            [[x1max, x1min],
             ...
             [xnmax, xnmin]]

    input_bounds : np.ndarray(shape=(m,2))
        The min/max bounds of the control states:

        .. code::

            [[u1max, u1min],
             ...
             [ummax, ummin]]

    qois : List[str] | List[int], optional
        Quantities of interest - the state to be controlled via the reference. It is used to
        form matrix :math:`C \in \mathbb{R}^{[n \times N]}`, where :math:`n` is the number of
        quantities of interest and :math:`N` is the lifted state size such that :math:`z=Cx`.

        Example:
        ["z1",...,"zn"] or [index("z1"), ..., index("zn")]

    cost_running : float | np.ndarray(shape=(1,n)), optional, by default 0.1
        Quadratic cost of the state for internal time steps :math:`Q`. If of type float, the
        same cost will be applied to all state dimensions.

    cost_terminal : float | np.ndarray(shape=(1,n)), optional, by default 100
        Quadratic cost of the state at the end of the prediction :math:`Q_{N_p}`. If of type
        float, the same cost will be applied to all state dimensions.

    cost_input : float | np.ndarray(shape=(1,n)), optional, by default 0.01
        Quadratic cost of the input :math:`R`. If of type float, the same cost will be applied
        to all input dimensions.

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
                "The optional dependencies `qpsolvers` and `quadprog` are required "
                "for `LinearKMPC`. These need to be installed separately with \n"
                "`pip install qpsolvers quadprog`"
            )

        self.predictor = predictor

        # define default values of properties
        # TODO: horizon could / should? be replaced with time_values
        self.horizon = horizon

        self.cost_running = cost_running
        self.cost_terminal = cost_terminal
        self.cost_input = cost_input

        self.account_initial = False

        try:
            # Note that these are defined on the lifted space:
            self.A = predictor.dmd_model.sys_matrix_
            self.B = predictor.dmd_model.control_matrix_
            self.lifted_state_size, self.input_size = self.B.shape  # type: ignore
        except ValueError:
            raise TypeError(
                "The shape of the control matrix is not as expected. "
                "This is likely due to an incompatible dmd_model type of the predictor. "
                "The predictor.dmd_model should support control (e.g. DMDControl)."
            )
        self.state_size = len(predictor.feature_names_in_)

        # setup conversion from lifted state to output quantities of interest

        from datafold.utils.general import projection_matrix_from_feature_names

        self.Cb = projection_matrix_from_feature_names(features_all=predictor.feature_names_out_, features_select=qois)
        self.Cb = self.Cb.T.toarray()
        self.output_size = len(qois)

        # if input_bounds.shape != (self.input_size, 2):
        #     raise ValueError("")
        #
        # if state_bounds.shape != (self.output_size, 2):
        #     raise ValueError("")

        self.input_bounds = input_bounds
        self.state_bounds = state_bounds

        self.H, self.h, self.G, self.Y, self.L, self.M, self.c = self._setup_optimizer()


    def _setup_optimizer(self):
        # implements relevant part of :cite:`korda-2018` to set up the optimization problem

        self.Ab, self.Bb = self._create_evolution_matrices()
        Q, q, R, r = self._create_cost_matrices()
        F, E, c = self._create_constraint_matrices()

        H = 2 * (self.Bb.T @ Q @ self.Bb + R)
        h = r + self.Bb.T @ q  # TODO: currently h is always zero
        G = 2 * self.Bb.T @ Q @ self.Ab
        Y = (-2 * Q @ self.Bb).T

        L = None
        M = None

        return H, h, G, Y, L, M, c

    def _create_evolution_matrices(self):
        # appendix from :cite:`korda-2018`
        # same as Sabin 2.44
        Np = self.horizon
        N = self.lifted_state_size
        m = self.input_size

        A = self.A
        B = self.B
        # TODO: C is here a projection matrix, and applied after Ab and Bb are set up.
        #  -- as of my understanding it is unnecessary to store the full Ab, we can apply the
        #  projection already (we only need to store the last "full A") -- similarily Bb
        Ab = np.eye((Np + 1) * self.output_size, N)
        Bb = np.zeros(((Np+1) * self.output_size, Np * m))

        A_last = A.copy()

        # set up A
        for i in range(1, Np + 1):
            s = i * self.output_size
            e = (i + 1) * self.output_size
            Ab[s:e, :] = self.Cb @ A_last
            A_last = A @ A_last

        B_current = B

        # set up B
        for i in range(1, Np+1):
            for k, j in enumerate(range(i, Np+1)):
                sr = j * self.output_size
                er = (j + 1) * self.output_size
                sc = k * m
                ec = (k+1) * m

                Bb[sr:er, sc:ec] = self.Cb @ B_current

            B_current = A @ B_current  # TODO: the last can be saved!

        if not self.account_initial:
            Ab = Ab[self.output_size:]
            Bb = Bb[self.output_size:]

        return Ab, Bb

    def _create_constraint_matrices(self):
        # implements appendix from :cite:`korda-2018`
        # same as Sabin 2.44, assuming
        # bounds vector is ordered [zmax; -zmin; umax; -umin]
        Np = self.horizon
        N = self.output_size
        m = self.input_size

        # TODO: Eb and Fb are very sparse matrices (~ 99 %)
        # constraint equations
        # E = np.vstack([np.eye(N), -np.eye(N), np.zeros((2 * m, N))])
        # Eb = np.kron(np.eye(Np+1), E) # TODO Np+1 if not ignoring initial

        rc = (Np + 1) * self.lifted_state_size
        Eb = np.zeros((rc, rc))

        # from scipy.sparse import block_diag
        # block_diag([np.eye(Np+1) for _ in range(self.lifted_state_size)]).to_array()


        # F = np.vstack([np.zeros((2 * N, m)), np.eye(m), -np.eye(m)])
        # Fb = np.vstack([np.kron(np.eye(Np), F), np.zeros((2 * (N + m), m * Np))])

        Fb = np.zeros(((Np+1)*self.lifted_state_size, Np*m))

        # constraint
        cb = np.zeros(((Np+1)*self.lifted_state_size, 1))


        # c[:, :N] = self.state_bounds[:, 0]
        # c[:, N : 2 * N] = -self.state_bounds[:, 1]  # TODO: does it make any sense to negate here??
        # c[:, 2 * N : 2 * N + m] = self.input_bounds[:, 0]
        # c[:, 2 * N + m :] = -self.input_bounds[:, 1]
        # c = c.T
        # cb = np.tile(c, (Np, 1))  # Np+1 if initial state is accounted

        return Eb, Fb, cb

    def _cost_to_array(self, cost: Union[float, np.ndarray], N: int):
        if isinstance(cost, np.ndarray):
            try:
                cost = cost.flatten()
                assert len(cost) == N
            except AssertionError:
                raise ValueError(
                    f"Cost should have length {N=}, received {len(cost)=}."
                )
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
        # implements appendix from :cite:`korda-2018`
        # same as Sabin 2.44
        Np = self.horizon
        N = self.output_size
        m = self.input_size

        # optimization - linear
        # assume linear optimization term is 0
        # TODO: improve or set h = 0
        q = np.zeros((N * (Np+int(self.account_initial)), 1))  # TODO: Np-1 to not account for initial state -- q is always zero!
        r = np.zeros((m * Np, 1))  # TODO: r is always zero currently...

        # optimization - quadratic
        # assuming only autocorrelation
        vec_running = self._cost_to_array(self.cost_running, N)

        if self.cost_terminal == 0 or self.cost_terminal is None:
            # add the running cost for the last iteration...
            vec_terminal = self._cost_to_array(self.cost_running, N)
        else:
            vec_terminal = self._cost_to_array(self.cost_terminal, N)

        vec_input = self._cost_to_array(self.cost_input, m)

        # TODO: do not store the diagonal matrices explicitly (or use sparse matrices).
        # quadratic matrix for state cost
        # TODO: -1 because initial state is not accounted!
        # TODO: include vec_terminal again
        Qb = np.diag(np.hstack([np.tile(vec_running, Np-1 + int(self.account_initial)), vec_terminal]))

        # quadratic matrix for input cost
        Rb = np.diag(np.tile(vec_input, Np))

        return Qb, q, Rb, r

    def generate_control_signal(
        self, X: InitialConditionType, reference: TransformType, initvals=None
    ) -> np.ndarray:
        r"""Method to generate a control sequence, given some initial conditions and
        a reference trajectory, as in :cite:`korda-2018` Algorithm 1. This method solves the
        following optimization problem (:cite:`korda-2018`, Equation 24).

        .. math::
            \text{minimize : } U^{T} H U^{T} + h^{T} U + z_0^{T} GU - y_{r}^{T} U
            \text{subject to : } LU + Mz_{0} \leq c
            \text{parameter : } z_{0} = \Psi(x_{k})

        Here, :math:`U` is the optimal control sequence to be estimated.

        Parameters
        ----------
        X : TSCDataFrame or np.ndarray
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

        # TODO: need validation here
        # TODO: work more with native TSCDataFrame here
        X_dict = self.predictor.transform(X).to_numpy().T

        try:
            yr = np.asarray(reference)
            assert yr.shape[1] == self.output_size  # TODO: make error and validation
            # TODO reference signal should contain horizon (and not horizon+1, as this is confusing!)
            yr = yr.reshape(((self.horizon + int(self.account_initial)) * self.output_size, 1))
        except:
            raise ValueError(
                "The reference signal should be a frame or array with n (output_size) "
                "columns and Np (prediction horizon) rows."
            )

        U = solve_qp(
            P=self.H,
            q=(self.G @ X_dict + self.Y @ yr).flatten(),
            G=None, # , self.L
            h=None, # (self.c - self.M @ X_dict).flatten(),  # (,
            A=None,
            b=None,
            lb=-0.1 * np.ones(38), #,
            ub=0.1* np.ones(38),
            solver="quadprog",
            verbose=False,
            initvals=initvals,
        )

        if U is None:
            raise ValueError("the solver did not converge")

        # y = self.Cb @ X_dict # TODO: comment from source % Should be y - yr, but yr adds just a constant term


        # TODO: U should be a TSCDataFrame -- use the columns from edmd
        # TODO: There should be a parameter time_values to set the time values in U
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

        # TODO: Need to return a TSCDataFrame
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
        given a control-affine model in differential form

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
            self.lifted_state_size, _, self.input_size = self.B.shape  # type: ignore
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
            raise ValueError(f"{input_bounds.shape=}, should be ({self.input_size=},2)")

        if isinstance(cost_input, np.ndarray):
            try:
                self.cost_input = cost_input.reshape((self.input_size, 1))
            except ValueError:
                raise ValueError(
                    f"{cost_input.shape=}, should be ({self.input_size},1)"
                )
        else:
            self.cost_input = cost_input * np.ones((self.input_size, 1))

        if isinstance(cost_state, np.ndarray):
            try:
                self.cost_state = cost_state.reshape((self.state_size, 1))
            except ValueError:
                raise ValueError(
                    f"{cost_state.shape=}, should be ({self.state_size},1)"
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
                "The system could not be envolved for the "
                "requested time span for initial condition."
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
        u : np.ndarray
            Control input
            shape = (n*m,)
            [u1(t0) u1(t1) ... u1(tn) u2(t1) ... um(tn)]
            with `n = self.horizon+1`; `m = self.input_size`
        x0 : np.ndarray
            Initial conditions
            shape = `(self.state_size, 1)`
        xref : np.ndarray
            Reference state
            shape = `(self.state_size, 1)`
        t : np.ndarray
            Time values for the evaluation

        Returns
        -------
        (float, np.ndarray)
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
        u
            Control input of shape `(n*m,)`
            [u1(t0) u1(t1) ... u1(tn) u2(t1) ... um(tn)]
            with `n = self.horizon+1`; `m = self.input_size`
        x0
            Initial conditions
            shape = `(self.state_size, 1)`
        xref
            Reference state
            shape = `(self.state_size, 1)`
        t
            Time values for the evaluation
        x
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
        u
            shape = (n*m,)
            [u1(t0) u1(t1) ... u1(tn) u2(t1) ... um(tn)]
            with `n = self.horizon+1`; `m = self.input_size`
        x0
            shape = `(self.state_size, 1)`
        xref
            shape = `(self.state_size, 1)`
        t
            Time values for the evaluation
        x
            State to use. If not provided is calculated by self.predictor
        Returns
        ---------
        np.ndarray, shape (n*m,)
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
        jac = np.einsum("ijk,kj->ij", x.T @ self.B.T, lambda_adjoint) + rho  # type: ignore

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
                f"Could not integrate the adjoint dynamics. Solver says '{sol.message}'"
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
        r"""Method to generate a control sequence, given some initial conditions and a
        reference trajectory. This method solves the following
        optimization problem (:cite:t:`peitz-2020` , Equation K-MPC).

        .. math::
            \text{given: } x_{0}, x_r
            \text{find: } u
            \text{minimizing: } J(x_0,x_r,u)
            \text{subject to: } \dot{x}= Ax + \sum_{i=1}^m B_i u_i x

        Here, :math:`u` is the optimal control sequence to be estimated.

        The optimization is done using scipy.optimize.minimize. Suitable methods use
        a calculated Jacobian (but not Hessian) and support bounds on the variable.

        Parameters
        ----------
        initial_conditions
            Initial conditions for the model

        reference
            The reference trajectory, which is required to optimize the control sequence.
            If ``TSCDataFrame`` and ``time_values`` is not provided, the time index of the
            reference is used.

        time_values
            Time values of the reference trajectory at which control inputs will
            be generated. If not provided tje  inferred from the reference.

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

        References
        ----------
            :cite:t:`peitz-2020`, Section 4.1


        """

        if time_values is None:
            try:
                time_values = reference.time_values()
            except AttributeError:
                raise TypeError(
                    "If time_values is not provided, the reference needs to be of type "
                    f"TSCDataFrame. Got {type(reference)=}"
                )

        if time_values.shape != (self.horizon + 1,):
            raise ValueError(
                f"time_values is of shape {time_values.shape} but should be "
                f"({self.horizon+1},)."
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
