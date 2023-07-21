import inspect
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.utils.validation import check_is_fitted, check_scalar

from datafold import EDMD, TSCDataFrame
from datafold._decorators import warn_experimental_class
from datafold.dynfold.base import InitialConditionType, TransformType
from datafold.utils.general import if1dim_colvec, projection_matrix_from_feature_names

try:
    import qpsolvers

    IMPORTED_QPSOLVERS = True
except ImportError:
    IMPORTED_QPSOLVERS = False


def _cost_to_array(cost: Union[float, np.ndarray], n_elements):
    if isinstance(cost, np.ndarray):
        if cost.ndim != 1 or cost.shape[0] != n_elements:
            raise ValueError(
                f"The cost vector must be 1-dim. with {n_elements} elements. "
                f"Got {cost.ndim=} and {cost.shape=}"
            )
        if (cost < 0).any():
            raise ValueError(
                "All cost values must be non-negative. "
                f"Found {(cost < 0).sum()} negative values."
            )

        return cost
    elif isinstance(cost, (float, int)):
        cost = float(cost)

        if cost < 0:
            raise ValueError(f"Cost must be a non-negative numeric value. Got {cost=}")
        return np.ones(n_elements) * cost
    else:
        raise TypeError(
            f"{type(cost)=} not understood, use numeric value (float/int) or 1-dim. array "
            f"with {n_elements} elements."
        )


class KMPC:
    r"""Class to implement Koopman operator-based model predictive control.

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
    edmd: EDMD
        Model to use, must be already fitted. The underlying DMD model in :py:class:`EDMD`
        must support control, such as :py:class:`.DMDControl`.

    horizon: int
        Prediction horizon in number of time steps to predict, :math:`N_p`.

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
        Quadratic cost of the state for internal time steps :math:`Q`. If the argument is of
        type float, then the same cost will be applied to all state dimensions.

    cost_terminal : float | np.ndarray(shape=(1,n)), optional, by default 100
        Quadratic cost of the state at the end of the prediction :math:`Q_{N_p}`. If the
        argument is of type float, then the same cost will be applied to all state dimensions.

    cost_input : float | np.ndarray(shape=(1,n)), optional, by default 0.01
        Quadratic cost of the input :math:`R`. If the argument is of type float, then the same
        cost will be applied to all state dimensions.

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
        edmd: EDMD,
        horizon: int,
        state_bounds: Optional[np.ndarray],
        input_bounds: Optional[np.ndarray],
        qois: Optional[list[str]] = None,
        cost_running: Union[float, np.ndarray] = 1,
        cost_terminal: Optional[Union[float, np.ndarray]] = 1,
        cost_input: Optional[Union[float, np.ndarray]] = 1,
        solver: str = "quadprog",
    ) -> None:
        # TODO: option to set multiple horizons? E.g. setting horizon = np.arange(5)

        if not IMPORTED_QPSOLVERS:
            raise ImportError(
                "The optional dependency `qpsolvers` is required by this class. "
                " The package can be installed with `pip install qpsolvers`."
            )

        self.edmd = edmd
        check_is_fitted(self.edmd)

        # define default values of properties
        self.horizon = horizon

        check_scalar(self.horizon, "horizon", target_type=int, min_val=2)

        self.cost_running = cost_running
        self.cost_terminal = cost_terminal
        self.cost_input = cost_input

        if qois is None:
            self.qois = self.edmd.dmd_model.feature_names_in_
            self.is_all_features = True
        else:
            # TODO: currently no validation
            self.qois = qois
            self.is_all_features = False

        self.n_qois = len(self.qois)

        self.n_features = self.edmd.dmd_model.n_features_in_
        self.n_control_input = self.edmd.dmd_model.n_control_in_

        self.input_bounds = input_bounds
        self.state_bounds = state_bounds

        self.solver = solver

        if self.solver not in qpsolvers.available_solvers:
            raise ValueError(
                f"{solver=} is not in the available solvers of qpsolvers\n "
                f"{qpsolvers.available_solvers=}"
            )

        if isinstance(self.input_bounds, np.ndarray) and self.input_bounds.shape != (
            self.n_control_input,
            2,
        ):
            raise ValueError(
                f"input_bounds must be of shape ({self.n_control_input=}, 2). "
                f"Got {self.input_bounds.shape=}."
            )

        if isinstance(self.state_bounds, np.ndarray) and self.state_bounds.shape != (
            self.n_qois,
            2,
        ):
            raise ValueError(
                f"state_bounds must be of shape ({self.n_qois=}, 2). "
                f"Got {self.state_bounds.shape=}\nNote that it is possible to also impose "
                "bounds on the other (not QoI) state dimensions, but this requires further "
                "implementation."
            )

        (
            self.H,
            self.l,
            self.M,
            self.Y,
            self.G,
            self.h,
            self.Ab,
            self.lb,
            self.ub,
        ) = self._setup_optimizer()

    def _setup_optimizer(self):
        # implements relevant part of :cite:`korda-2018` to set up the optimization problem
        Ab, Bb = self._create_evolution_matrices()
        Q, q, R, r = self._create_cost_matrices()
        G, h, lb, ub = self._create_constraint_matrices(Bb)

        # tmp variable to save computations
        BbTQ = 2 * Bb.T @ Q

        H = BbTQ @ Bb + 2 * R
        a = (
            r + Bb.T @ q
        )  # TODO: currently a is always zero (maybe better name for `a`?)
        M = BbTQ @ Ab
        Y = BbTQ

        if G is None:
            # Ab is not required to store if there are no bounds on state
            Ab = None

        return H, a, M, Y, G, h, Ab, lb, ub

    def _create_evolution_matrices(self):
        # appendix from :cite:`korda-2018`
        # same as Sabin 2.44

        A = self.edmd.dmd_model.sys_matrix_
        B = self.edmd.dmd_model.control_matrix_

        Ab = np.zeros((self.horizon * self.n_qois, self.n_features))
        Bb = np.zeros((self.horizon * self.n_qois, self.horizon * self.n_control_input))

        is_project_coordinates = not self.is_all_features

        if is_project_coordinates:
            Cb = projection_matrix_from_feature_names(
                features_all=self.edmd.feature_names_out_, features_select=self.qois
            ).T
        else:
            Cb = None

        A_last = A.copy()

        # set up Ab
        for i in range(self.horizon):
            s = i * self.n_qois  # start index
            e = (i + 1) * self.n_qois  # end index

            if is_project_coordinates:
                Ab[s:e, :] = Cb @ A_last
            else:
                Ab[s:e, :] = A_last
            A_last = A.dot(A_last, out=A_last)

        B_current = B.copy()

        # set up Bb
        for i in range(self.horizon):
            if is_project_coordinates:
                _B_tmp = Cb @ B_current
            else:
                _B_tmp = B_current.view()

            # Copy along diagonal blocks of matrix
            for k, j in enumerate(range(i, self.horizon)):
                sr = j * self.n_qois  # start row
                er = (j + 1) * self.n_qois  # end row
                sc = k * self.n_control_input  # start columns
                ec = (k + 1) * self.n_control_input  # end column
                Bb[sr:er, sc:ec] = _B_tmp

            if i != self.horizon:
                # avoid unnecessary matrix-matrix multiplication in last iteration
                B_current = A.dot(B_current, out=B_current)

        return Ab, Bb

    def _create_cost_matrices(self):
        # implements appendix from :cite:`korda-2018`
        # same as Sabin 2.44

        # optimization - linear
        # TODO: q and r are always zero -- either remove completely or need a user parameter
        q = np.zeros((self.n_qois * self.horizon, 1))

        # optimization - quadratic diagonal matrix
        # quadratic matrix for state cost and terminal cost
        vec_running = _cost_to_array(self.cost_running, self.n_qois)

        if (np.asarray(self.cost_terminal) == 0).all() or self.cost_terminal is None:
            # add the running cost for the last iteration...
            vec_terminal = _cost_to_array(self.cost_running, self.n_qois)
        else:
            vec_terminal = _cost_to_array(self.cost_terminal, self.n_qois)

        diag = np.hstack([np.tile(vec_running, self.horizon - 1), vec_terminal])
        Qb = scipy.sparse.spdiags(diag, 0, diag.size, diag.size)

        # linear part input cost  # TODO: currently always zero
        r = np.zeros((self.n_control_input * self.horizon, 1))

        # quadratic matrix for input cost
        vec_input = _cost_to_array(self.cost_input, self.n_control_input)

        diag = np.tile(vec_input, self.horizon)
        Rb = scipy.sparse.spdiags(diag, 0, diag.size, diag.size)

        return Qb, q, Rb, r

    def _create_constraint_matrices(self, Bb):
        # implements appendix from :cite:`korda-2018`
        # same as Sabin 2.44, assuming
        # bounds vector is ordered [zmax; -zmin; umax; -umin]

        if self.state_bounds is not None:
            # inequality matrix
            G = np.vstack([Bb, -Bb])

            # right hand side of inequality. NOTE: at this point 'h' is incomplete and
            # must be adapted with the initial condition of the system (which is unknown here)
            Xlb = np.tile(self.state_bounds[:, 0], self.horizon)
            Xub = np.tile(self.state_bounds[:, 1], self.horizon)
            h = np.hstack([Xub, -Xlb])
        else:
            G, h = [None, None]

        if self.input_bounds is not None:
            lb = np.tile(self.input_bounds[:, 0], self.horizon)
            ub = np.tile(self.input_bounds[:, 1], self.horizon)
        else:
            lb, ub = [None, None]

        return G, h, lb, ub

    def control_sequence_horizon(
        self, X: InitialConditionType, reference: TransformType
    ) -> TSCDataFrame:
        r"""Generate a control sequence, given some initial condition and
        a reference time series (target trajectory).

        This method solves the following optimization problem (from :cite:t:`korda-2018`,
        Eq. 24):

        .. math::
            \text{minimize : } U^{T} H U^{T} + (G \mathbf{z}_0 - Y^{T}) U \\
            \text{subject to : } U_{lb} <= U_i <= U_{ub} \\
            \text{parameter: } z_{0} = \Psi(x_{k})


        where :math:`U` is the optimal control sequence to be estimated (represented as a time
        series of shape `(horizon, n_control_input)`), `Y` the reference time series,
        :math:`\mathbf{z}_0` the initial condition in EDMD dictionary coordinates,
        :math:`U_{lb}` and :math:`U_{ub}` the control bounds.

        If successful, performing a prediction with the internal model leads to a state
        evolution that steers towards the reference.
        I.e. ``edmd.predict(X, U)``, with ``U`` being the control sequence.

        Parameters
        ----------
        X: TSCDataFrame
            Initial conditions for the model. Passed to :code:`edmd.transform(X)`. The result
            of this transformation must be a single state. This means `X` must have
            `edmd.n_samples_ic_`.

        reference: TSCDataFrame
            Target time series over the prediction horizon with shape `(horizon, n_qois)`.

        Returns
        -------
        U : TSCDataFrame
            Sequence of control inputs.

        Raises
        ------
        ValueError
            In case of mis-shaped input
        """
        # TODO: linear cost is currently not supported

        # Currently, only the control of a single time series is supported
        # X.tsc.check_required_n_timeseries(1)
        # X.tsc.check_required_n_timesteps(self.edmd.n_samples_ic_)

        X_dict = self.edmd.transform(X)

        if X_dict.shape[0] != 1:
            raise ValueError(
                f"The transformed dictionary state must consist of a single sample only "
                f"(i.e. X.shape[0] == 0). Got {X.shape[0]=}."
            )

        t_ic = X_dict.time_values()[0]
        t_ref = reference.time_values()[0]

        if not np.allclose(t_ic + self.edmd.dt_, t_ref, rtol=1e-11, atol=1e-15):
            raise ValueError(
                f"The reference time value of the initial state is at time {t_ic=}. "
                f"The reference time series must start one time step in the future at "
                f"t_ref={t_ic+self.edmd.dt_} (diff={t_ic+self.edmd.dt_ - t_ref})."
            )

        if reference.shape != (self.horizon, self.n_qois):
            raise ValueError(
                f"Reference time series must have shape ({self.horizon=}, {self.n_qois=}). "
                f"Got {reference.shape=} instead."
            )

        np_reference = reference.to_numpy()
        np_reference = np_reference.reshape((self.horizon * self.n_qois, 1))

        if self.h is not None:
            # This is required to fulfill the state bounds
            # if Ab is None than there is a bug, so there is no check
            h_ic_adapt = self.Ab @ X_dict.to_numpy().ravel()
            h = self.h + np.hstack([-h_ic_adapt, h_ic_adapt])
        else:
            h = None

        U = qpsolvers.solve_qp(
            P=self.H,
            q=(self.M @ X_dict.to_numpy().T - self.Y @ np_reference).flatten(),
            G=self.G,
            h=h,
            A=None,
            b=None,
            lb=self.lb,
            ub=self.ub,
            solver=self.solver,
            verbose=False,
            initvals=None,  # TODO: make use of initvals for iteration?
        )

        if U is None:
            raise ValueError("The quadratic solver did not converge.")

        # TODO: comment from source % Should be y - yr, but yr adds just a constant term
        # y = self.Cb @ X_dict

        # the actual control input is obtained from the previous state the reference time
        # series, therefore the actual control sequence starts at the initial state (not the
        # first time value in the reference time series

        # it is safer to use the time values from the initial condition and reference
        # this does not introduce potential numerical noise or changes the type
        control_time_values = np.append(t_ic, reference.time_values()[:-1])

        U = TSCDataFrame.from_array(
            U.reshape((self.horizon, self.n_control_input)),
            time_values=control_time_values,
            feature_names=self.edmd.control_names_in_,
        )
        return U

    def control_system_reference(
        self,
        sys,
        sys_ic: TSCDataFrame,
        X_ref: TSCDataFrame,
        X_ic: TSCDataFrame,
        augment_control: bool = False,
    ):
        """
        TODO.

        Parameters
        ----------
        sys
        X_ref
        X_ic
        U_ic
        augment_control

        Returns
        -------

        """
        # TODO: validation  # TODO: fill_horizon_with_last_state option
        # TODO: it should also be possible that the system terminates the simulation

        if inspect.ismethod(sys):
            eval_sys = sys
        else:
            eval_sys = sys.predict  # set to function

        sys_seq = sys_ic.copy()
        X_seq = X_ic.copy()
        U_seq = None
        n_ic = self.edmd.n_samples_ic_

        for i in range(X_ref.shape[0] - 1):
            # obtain the reference time series over the time horizon that we want to optimize
            _ref = X_ref.iloc[i : i + self.horizon, :]

            # fill up with last state to obtain the required time series length, if necessary
            if _ref.shape[0] != self.horizon:
                _ref = _ref.tsc.fill_timeseries_with_last_state(
                    n_timesteps=self.horizon
                )

            # obtain optimal control sequence from KMPC
            ukmpc = self.control_sequence_horizon(
                X=X_seq.iloc[-n_ic:, :], reference=_ref
            )

            # TODO: currently we take only the next input. This could be generalized to take
            #  more control input
            U_next = ukmpc.iloc[[0], :]

            time_values = [U_next.time_values()[0], _ref.time_values()[0]]

            # forward the true model with the next optimal control
            try:
                state, _ = eval_sys(
                    sys_seq.iloc[[-1], :],
                    U=U_next,
                    time_values=time_values,
                )
            except RuntimeError:
                print("Simulation has terminated")
                # TODO: maybe own error to indicate termination?
                break

            if augment_control:
                # augment the state for the next prediction
                edmd_state = state.tsc.augment_control_input(U_next).iloc[[-1], :]
                edmd_state = edmd_state.loc[:, self.edmd.feature_names_in_]
            else:
                edmd_state = state.loc[:, self.edmd.feature_names_in_]

            # TODO: preallocating memory and writing in is more efficient than concat
            # store control input and state the sequence, which is used later for plotting
            if sys_seq is None:
                sys_seq = state.copy()
            else:
                sys_seq = pd.concat([sys_seq, state.iloc[[-1], :]], axis=0)

            U_seq = pd.concat([U_seq, U_next], axis=0)
            X_seq = pd.concat([X_seq, edmd_state], axis=0)

        return sys_seq, U_seq

    # def compute_cost(self, U, reference, initial_conditions):
    #     z0 = self.lifting_function(initial_conditions)
    #     z0 = if1dim_colvec(z0)
    #
    #     try:
    #         z0 = z0.to_numpy().reshape(self.lifted_state_size, 1)
    #     except ValueError as e:
    #         raise ValueError(
    #             "The initial state should match the shape of "
    #             "the system state before the lifting."
    #         ) from e
    #
    #     try:
    #         yr = np.array(reference)
    #         assert yr.shape[1] == self.output_size
    #         yr = yr.reshape(((self.horizon + 1) * self.output_size, 1))
    #     except:
    #         raise ValueError(
    #             "The reference signal should be a frame or array with n (output_size) "
    #             "columns and  Np (prediction horizon) rows."
    #         )
    #
    #     U = U.reshape(-1, 1)
    #     e1 = U.T @ self.H @ U
    #     e2 = self.h.T @ U
    #     e3 = z0.T @ self.G @ U
    #     e4 = -yr.T @ self.Y @ U
    #
    #     # TODO: Need to return a TSCDataFrame
    #     return (e1 + e2 + e3 + e4)[0, 0]


class LQR:
    def __init__(self, edmd, cost_running, cost_input):
        self.edmd = edmd

        self.cost_running = cost_running
        self.cost_input = cost_input

        self.Flqr = self._setup_optimizer()

    def preset_target_state(self, target_state: TSCDataFrame):
        self._preset_target_state = self.edmd.transform(target_state)
        return self

    def _setup_optimizer(self):
        Q, R = self._create_cost_matrices()

        Ad = self.edmd.dmd_model.sys_matrix_
        Bd = self.edmd.dmd_model.control_matrix_
        Pd = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
        Flqr = np.linalg.inv(R + Bd.T @ Pd @ Bd) @ Bd.T @ Pd @ Ad

        return Flqr

    def _create_cost_matrices(self):
        cost_diagonal = _cost_to_array(
            self.cost_running, n_elements=self.edmd.n_features_out_
        )
        Q = np.diag(cost_diagonal)  # sparse matrices don't work in solve_discrete_are

        cost_input = _cost_to_array(self.cost_input, n_elements=self.edmd.n_control_in_)
        R = np.diag(cost_input)

        return Q, R

    def control_sequence(
        self, X: TSCDataFrame, target_state: Optional[TSCDataFrame] = None
    ):
        X.tsc.check_required_n_timeseries(1)

        if target_state is not None:
            target_state_dict = self.edmd.transform(target_state).to_numpy()
        elif hasattr(self, "_preset_target_state"):
            target_state_dict = self._preset_target_state
        else:
            raise ValueError(
                "Target state must be provided either as a parameter or set with "
                "'preset_target_state'"
            )

        u = -self.Flqr @ (self.edmd.transform(X).to_numpy() - target_state_dict).T
        u = TSCDataFrame.from_array(
            u,
            time_values=X.time_values()[-1],
            feature_names=self.edmd.control_names_in_,
            ts_id=X.ids[0],
        )

        return u


@warn_experimental_class
class AffineKgMPC:  # prama: no cover
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
        given a control-affine model in differential form.

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

        if not self.predictor.dmd_model.is_differential_system:
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
    ) -> tuple[float, np.ndarray]:
        """Calculate the cost and its Jacobian.

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
        """Compute the cost for the given reference.

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
        -------
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
        """Compute the Jacobian of the cost function.

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
        -------
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
                "Using closest solution.",
                stacklevel=2,
            )

        return res.x.reshape(self.input_size, self.horizon + 1).T[:-1, :]
