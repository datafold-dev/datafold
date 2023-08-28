#!/usr/bin/env python3
import abc
from typing import Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d

from datafold import InitialCondition
from datafold.dynfold.base import TSCPredictMixin
from datafold.pcfold import TSCDataFrame


class DynamicalSystem(TSCPredictMixin, metaclass=abc.ABCMeta):
    def __init__(self, feature_names_in) -> None:
        self.feature_names_in_ = feature_names_in
        self.n_features_in_ = len(self.feature_names_in_)

    @abc.abstractmethod
    def predict(
        self,
        X,
        *,
        time_values: Optional[np.ndarray] = None,
        **kwargs,
    ):
        raise NotImplementedError("base class")


class ODE(DynamicalSystem, metaclass=abc.ABCMeta):
    def __init__(self, feature_names_in, **ivp_kwargs) -> None:
        super().__init__(feature_names_in=feature_names_in)
        self._default_step_size = 0.01

        self.ivp_kwargs = ivp_kwargs
        self.ivp_kwargs.setdefault("method", "RK45")
        self.ivp_kwargs.setdefault("vectorized", True)

    def _prepare_ic_and_time_values(self, X, time_values, feature_names_in):
        if isinstance(time_values, (float, int)):
            time_values = np.array([0, time_values])
            dt = time_values[1]
        if time_values is None:
            time_values = np.array([0.0, self._default_step_size])
        else:
            time_values = np.asarray(time_values)
            if len(time_values) == 1:
                dt = time_values[0]
                time_values = np.append(0, time_values)
            else:
                dt = time_values[1] - time_values[0]

        if isinstance(X, (np.ndarray, list)):
            X = InitialCondition.from_array(
                X=np.asarray(X),
                time_value=time_values[0],
                feature_names=feature_names_in,
            )

        return X, time_values, dt

    def predict(
        self, X: TSCDataFrame, *, time_values: Optional[np.ndarray] = None, **kwargs
    ):
        X, time_values, dt = self._prepare_ic_and_time_values(
            X, time_values, self.feature_names_in_
        )
        time_values = self._validate_time_values_format(time_values=time_values)

        self._validate_datafold_data(X, tsc_kwargs=dict(ensure_n_timesteps=1))

        feature_names_out = self._read_fit_params(
            attrs=[("feature_names_out", self.get_feature_names_out())],
            fit_params=kwargs,
        )

        t_span = np.array([time_values[0], time_values[-1]])

        X_ret = []
        for i, ic in X.itertimeseries():
            ic = ic.to_numpy().ravel()
            X_sol = solve_ivp(
                fun=self._f, t_span=t_span, y0=ic, t_eval=time_values, **self.ivp_kwargs
            )
            if X_sol.success:
                X_sol = TSCDataFrame.from_array(
                    X_sol.y.T,
                    time_values=time_values,
                    feature_names=feature_names_out,
                    ts_id=i,
                )
                X_ret.append(X_sol)
            else:
                raise ValueError(f"Initial condition {i} failed.")

        X_ret = TSCDataFrame.from_frame_list(X_ret)
        return X_ret

    @abc.abstractmethod
    def _f(self, t, Y, U):
        """Right-hand side of the ODE.

        The return of the function must match the 'fun' parameter in Scipy's 'solve_ivp'
        function.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

        The U parameter is an extra parameter for the control input.
        """
        raise NotImplementedError("base class")


class ControllableODE(ODE, metaclass=abc.ABCMeta):
    def __init__(
        self,
        feature_names_in,
        control_names_in,
        **ivp_kwargs,
    ):
        super().__init__(feature_names_in=feature_names_in, **ivp_kwargs)
        self.control_names_in_ = control_names_in
        self.n_control_in_ = len(self.control_names_in_)

    def predict(  # type: ignore
        self,
        X,
        *,
        U: Optional[Union[np.ndarray, TSCDataFrame, Callable]] = None,
        time_values: Optional[np.ndarray] = None,
        require_last_control_state=False,
    ):
        # TODO: make U really optional (do not apply any control if U is None) --
        #  this needs to be addressed in _f, where U is ignored (or set to zero)

        # validation
        if isinstance(U, np.ndarray) and U.ndim == 1:
            U = U[:, np.newaxis]

        if (
            isinstance(U, (TSCDataFrame, np.ndarray))
            and U.shape[1] != self.n_control_in_
        ):
            raise ValueError(f"{U.shape[1]=} must match {self.n_control_in_=}")

        if isinstance(X, np.ndarray) and X.ndim == 1:
            X = X[np.newaxis, :]

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"{X.shape[1]=} must match {self.n_features_in_=}")

        # TODO: cast time_values only containing a float/int to np.array([value])
        #   then check for one_step_sim

        if isinstance(U, TSCDataFrame) and U.shape[0] > 1:
            U.tsc.check_const_time_delta()
            dt = U.delta_time
        elif time_values is not None:
            time_values = np.asarray(time_values)
            if len(time_values) < 2:
                raise ValueError(
                    "Parameter time_values must include at least two elements. "
                    f"Got {len(time_values)=}"
                )

            dt = time_values[1] - time_values[0]
        else:
            dt = self._default_step_size

        if not callable(U):
            if U is None:
                # uncontrolled system evaluation
                if time_values is None:
                    raise ValueError(
                        "If U is not provided (uncontrolled system) then time_values "
                        "cannot be None. "
                    )
                U = np.zeros([X.shape[0], len(time_values) - 1, self.n_control_in_])
                U = TSCDataFrame.from_tensor(U, time_values=time_values[:-1])
            elif isinstance(U, np.ndarray) and X.shape[0] == U.shape[0]:
                if time_values is None:
                    raise ValueError(
                        "For multiple one-step predictions, the parameter "
                        "'time_values' cannot be None."
                    )

                # Interpret as one-step prediction of multiple initial conditions
                idx = pd.MultiIndex.from_arrays(
                    [np.arange(U.shape[0]), np.ones(U.shape[0]) * time_values[0]]
                )
                U = TSCDataFrame(U, index=idx, columns=self.control_names_in_)

            elif X.shape[0] > 1 and not isinstance(U, TSCDataFrame):
                raise ValueError(
                    "To solve for multiple initial conditions `U` must be of type "
                    f"TSCDataFrame. Got {type(U)}"
                )

        self._requires_last_control_state = False
        time_values = self._validate_and_set_time_values_predict(
            time_values=time_values, X=X, U=U, dt=dt
        )

        if isinstance(U, TSCDataFrame):
            U.tsc.check_equal_timevalues()
            U.tsc.check_required_n_timesteps(len(time_values) - 1)
        elif isinstance(U, np.ndarray):
            U = TSCDataFrame.from_array(
                U, time_values=time_values[:-1], feature_names=self.control_names_in_
            )

        if isinstance(X, pd.DataFrame):
            # TODO: work with TSCDataFrame instead and cast to it if X is np.ndarray
            X = X.to_numpy()

        X_sol = list()

        for i in range(X.shape[0]):
            ic = X[i]

            if callable(U):
                # user specified input
                Ufunc = U
            elif len(time_values) == 2:
                Ufunc = lambda t, x: U.iloc[[i], :].to_numpy()
            else:
                # interpolates control input from data
                U_interp = U.loc[[U.ids[i]], :].to_numpy()

                interp_control = []

                for j in range(self.n_control_in_):
                    func = lambda t, x: interp1d(
                        time_values[:-1],
                        U_interp[:, j],
                        kind="previous",
                        fill_value="extrapolate",
                    )(t)
                    interp_control.append(func)

                Ufunc = lambda t, x: np.array([[u(t, x) for u in interp_control]])

            sol = solve_ivp(
                # U should be a row-major mapping in datafold
                # to align with Scipy's ODE solver (column-major), the control mapping is
                # transposed
                fun=lambda t, x: self._f(t, x, Ufunc(t, x).T),
                t_span=(time_values[0], time_values[-1]),
                y0=ic,
                t_eval=time_values,
                **self.ivp_kwargs,
            )

            if not sol.success:
                raise RuntimeError(
                    f"The prediction was not successful \n Reason: \n"
                    f" {sol.message=}"
                )

            X_sol.append(
                TSCDataFrame.from_array(
                    sol.y.T,
                    time_values=time_values,
                    feature_names=self.feature_names_in_,
                )
            )

        X_sol = TSCDataFrame.from_frame_list(X_sol)

        if callable(U):
            X_sol_but_last = X_sol.tsc.drop_last_n_samples(1)

            tv = X_sol_but_last.index.get_level_values(
                TSCDataFrame.tsc_time_idx_name
            ).to_numpy()

            # turn callable into actual data -- needs to be re-computed as I do not see a way
            # to access this from the scipy ODE solver

            # TODO: this only works if U is vectorized, maybe need an element-by-element
            #  way too...
            U = U(tv, X_sol_but_last.to_numpy())
            U = TSCDataFrame.from_same_indices_as(
                X_sol_but_last, values=U, except_columns=self.control_names_in_
            )

        return X_sol, U


class LimitCycle(DynamicalSystem):
    def __init__(self, eps=1, analytical=True):
        self.obs = None
        self.eps = eps
        self.analytical = analytical

    def _compute_angle(self, x1, x2):
        e_vec = np.array([1, 0])
        vec = np.array([x1, x2])

        norm_vec = np.linalg.norm(vec)

        if norm_vec < 1e-15:
            return 0

        if x2 < 0:
            angle = 2 * np.pi - np.arccos(np.dot(e_vec, vec) / norm_vec)
        else:
            angle = np.arccos(np.dot(e_vec, vec) / norm_vec)

        return angle

    def _as_cartesian_coordinates(self, ang, rad):
        vals = rad * np.exp(0 + 1j * ang)
        return np.real(vals), np.imag(vals)

    def predict(self, **kwargs):
        # TODO: adapt to the predict in metaclass and do not put all in **kwargs
        if "t_eval" in kwargs:
            kwargs["nr_steps"] = len(kwargs["t_eval"])
            t_diff = np.diff(kwargs["t_eval"])
            assert (
                len(np.unique(np.round(t_diff, decimals=10))) == 1
            )  # TODO only equidistant sampling is supported at the moment!
            kwargs["dt"] = t_diff[0]
            del kwargs["t_eval"]

        if self.analytical:
            return self.eval_analytical(**kwargs)
        else:
            return self.eval_finite_differences(**kwargs)

    def eval_finite_differences(self, x1, x2, dt, nr_steps):
        # TODO: make private function!
        # use Euler, could also be solved analytically
        # diss, p. 52 t_end=10^-3 and nr_steps=10, eps=1E-2
        t = np.linspace(0, dt * (nr_steps - 1), nr_steps)
        r0 = np.linalg.norm(np.array([x1, x2]))
        a0 = self._compute_angle(x1=x1, x2=x2)

        a_vals = np.zeros(nr_steps)
        a_vals[0] = a0

        r_vals = np.zeros(nr_steps)
        r_vals[0] = r0

        for i in range(1, nr_steps):
            a_vals[i] = a_vals[i - 1] + dt * 1 / self.eps
            r_vals[i] = r_vals[i - 1] + dt * ((-r_vals[i - 1] ** 2 + 1) * r_vals[i - 1])

        # x, y = self._as_cartesian_coordinates(rad=r_vals, ang=a_vals)
        self.obs = pd.DataFrame(
            np.vstack([a_vals, r_vals]).T, index=t, columns=["alpha", "radius"]
        )
        return self.obs

    def eval_analytical(self, x1, x2, dt, nr_steps):
        # TODO: make private function!
        t = np.linspace(0, dt * (nr_steps - 1), nr_steps)

        a0 = self._compute_angle(x1=x1, x2=x2)
        r0 = (
            np.linalg.norm(np.array([x1, x2])) + 1e-10
        )  # add a small number to avoid division by zero

        a_vals = 1 / self.eps * t + a0

        r_vals = np.exp(t) / np.sqrt(-1 + np.exp(2 * t) + 1 / r0**2)

        # x, y = self._as_cartesian_coordinates(rad=r_vals, ang=a_vals)
        self.obs = pd.DataFrame(
            np.vstack([a_vals, r_vals]).T, index=t, columns=["alpha", "radius"]
        )
        return self.obs


class ThreeStablePoints(ODE):
    def __init__(self):
        super().__init__(feature_names_in=["x0", "x1"])

    def get_feature_names_out(self, input_features=None):
        return ["x0", "x1"]

    def _f(self, t, x):
        dx0 = x[0] - x[0] * x[1]
        dx1 = x[0] ** 2 - 2 * x[1]
        return np.array([dx0, dx1])


class Hopf(ODE):
    """From.

    Lawrence Perko. Differential equations and dynamical systems, volume 7. Springer
    Science & Business Media, 2013. page 350

    https://link.springer.com/book/10.1007/978-1-4613-0003-8
    """

    def __init__(
        self,
        mu: float = 1,
        return_cart: bool = True,
        return_polar: bool = False,
        **ivp_kwargs,
    ):
        self.mu = mu
        self.return_cart = return_cart
        self.return_angular = return_polar

        if not self.return_cart and not return_polar:
            raise ValueError(f"canot have both {return_cart=} and {return_polar=}")

        super().__init__(feature_names_in=["x1", "x2"], **ivp_kwargs)

    def get_feature_names_out(self, input_features=None):
        cols = []
        if self.return_cart:
            cols += ["x1", "x2"]  # Cartesian coordinates
        if self.return_angular:
            cols += ["r", "angle"]  # radius and angle
        return cols

    def _f(self, t, y):
        """Hopf system as planar ODE system."""
        y_dot = np.zeros(2)
        factor = self.mu - y[0] ** 2 - y[1] ** 2

        y_dot[0] = -y[1] + y[0] * factor
        y_dot[1] = y[0] + y[1] * factor
        return y_dot

    def predict(  # type: ignore
        self,
        X,
        *,
        time_values: Optional[np.ndarray] = None,
        ic_type: Literal["cart", "polar"] = "cart",
    ) -> TSCDataFrame:
        if ic_type not in ["cart", "polar"]:
            raise ValueError("")

        if ic_type == "polar":
            if isinstance(X, pd.DataFrame):
                rt_ic = X.to_numpy()
            else:
                rt_ic = X
            adapt_ic = np.copy(rt_ic)
            adapt_ic[:, 0] = rt_ic[:, 0] * np.cos(rt_ic[:, 1])
            adapt_ic[:, 1] = rt_ic[:, 0] * np.sin(rt_ic[:, 1])
            if isinstance(X, pd.DataFrame):
                X[:] = adapt_ic
                X.columns = pd.Index(["x1", "x2"])
            else:
                X, time_values, _ = self._prepare_ic_and_time_values(
                    X=X, time_values=time_values, feature_names_in=["x1", "x2"]
                )

        X_cart = super().predict(
            X=X, time_values=time_values, feature_names_out=["x1", "x2"]
        )

        if self.return_angular:
            # compute angular solution

            X_cart_np = X_cart.to_numpy()

            theta = np.arctan2(X_cart_np[:, 1], X_cart_np[:, 0])
            radius = X_cart_np[:, 0] / np.cos(theta)
            X_ang_np = np.column_stack([radius, theta])

            X_ang = TSCDataFrame.from_same_indices_as(
                indices_from=X_cart,
                values=X_ang_np,
                except_columns=["r", "angle"],
            )

        if self.return_cart and self.return_angular:
            return pd.concat([X_cart, X_ang], axis=1)
        elif self.return_cart:
            return X_cart
        else:  # self.return_angular:
            return X_ang


class ClosedPeriodicalCurve(DynamicalSystem):
    def __init__(self, consts=(3, 1, 1, 5, 2), noise_std=0):
        assert len(consts) == 5
        self.consts = consts
        self.noise = noise_std

    def _closed_system(self, t_eval):
        if self.noise > 0:
            noise_x = np.random.default_rng(1).normal(0, self.noise, size=len(t_eval))
            noise_y = np.random.default_rng(2).normal(0, self.noise, size=len(t_eval))
            noise_z = np.random.default_rng(3).normal(0, self.noise, size=len(t_eval))
        else:
            noise_x, noise_y, noise_z = [0, 0, 0]

        x = noise_x + np.sin(self.consts[0] * t_eval) * np.cos(self.consts[1] * t_eval)
        y = noise_y + np.sin(self.consts[2] * t_eval) * np.sin(self.consts[3] * t_eval)
        z = noise_z + np.sin(self.consts[4] * t_eval)

        return pd.DataFrame(
            np.column_stack([x, y, z]), index=t_eval, columns=["x", "y", "z"]
        )

    def predict(self, X, time_values, **kwargs):
        assert X is None
        return self._closed_system(time_values)


class Pendulum(DynamicalSystem):
    """System explained:
    https://towardsdatascience.com/a-beginners-guide-to-simulating-dynamical-systems-with-python-a29bc27ad9b1.
    """

    def __init__(self, mass_kg=1, length_rod_m=1, friction=0, gravity=9.81):
        self.mass_kg = mass_kg
        self.rod_length_m = length_rod_m
        self.friction = friction
        self.gravity = gravity

    def _integrate_pendulum_sim(self, theta_init, t):
        theta_dot_1 = theta_init[1]
        theta_dot_2 = -self.friction / self.mass_kg * theta_init[
            1
        ] - self.gravity / self.rod_length_m * np.sin(theta_init[0])
        return theta_dot_1, theta_dot_2

    def _compute_cart_parameters(self):
        # 10 * circle_area = mass -- the 10 is artificial
        self.radius_mass_ = np.sqrt(self.mass_kg / np.pi) / 10

        self.fixation_point_ = np.array([0, self.rod_length_m], dtype=float)
        self.equilibrium_point_ = np.array([0, 0])

    def _convert_cartesian(self, theta_position):
        x = self.rod_length_m * np.cos(theta_position - np.pi / 2)
        y = self.rod_length_m * np.sin(theta_position - np.pi / 2)

        return self.fixation_point_ + np.column_stack([x, y])

    def predict(self, X, time_values, **kwargs):
        # initial_conditions = theta_0 -- theta_1
        self._compute_cart_parameters()

        X = np.asarray(X)

        # TODO: use np.atleast2d
        if X.ndim == 1:
            X = X[np.newaxis, :]

        solution_frames = []

        for ic_idx in range(X.shape[0]):
            theta_coord = odeint(
                self._integrate_pendulum_sim, X[ic_idx, :], time_values
            )

            cartesian_coord = self._convert_cartesian(theta_coord[:, 0].copy())

            theta_coord = pd.DataFrame(
                theta_coord, index=time_values, columns=["theta", "dot_theta"]
            )
            cartesian_coord = pd.DataFrame(
                cartesian_coord, index=time_values, columns=["x", "y"]
            )

            solution_frames.append(pd.concat([theta_coord, cartesian_coord], axis=1))

        tsc_df = TSCDataFrame.from_frame_list(solution_frames)

        return tsc_df


class InvertedPendulum(ControllableODE):
    """An inverted pendulum on a cart controlled by an electric motor.

    The system is parametrized with the voltage of the electric motor. The states include
    four observations: 1) position, 2) velocity, 3) angle from horizon and 4) angular velocity.

    Parameters
    ----------
    pendulum_mass: float
        Mass of pendulum, defaults to 0.0905 kg

    cart_mass: float
        Mass of the cart, defaults to 1.12 kg

    g: float
        Gravitational acceleration, defaults to 9.81 m/s^2

    tension_force_gain: float
        Conversion between electric motor input voltage in V and tension
        force in N, defaults to 7.5 N/V

    pendulum_length: float
        Length of the pendulum, defaults to 0.365 m

    cart_friction: float
        Dynamic damping coefficient on the cart, defaults to 6.65 kg/s

    """

    def __init__(
        self,
        pendulum_mass=0.0905,  # kg
        cart_mass=1.12,  # kg
        g=9.81,  # m/s^2
        tension_force_gain=7.5,  # N/V
        pendulum_length=0.365,  # m
        cart_friction=6.65,  # kg/s
    ):
        self.pendulum_mass = pendulum_mass
        self.cart_mass = cart_mass
        self.g = g
        self.tension_force_gain = tension_force_gain
        self.pendulum_length = pendulum_length
        self.cart_friction = cart_friction

        super().__init__(
            feature_names_in=["x", "xdot", "theta", "thetadot"],
            control_names_in=["u"],
        )

    def _f(self, t, y, u):
        _, xdot, theta, thetadot = y

        m = self.pendulum_mass
        M = self.cart_mass

        sin_th = np.sin(theta)
        cos_th = np.cos(theta)

        alpha = M + m * np.square(sin_th)

        f1 = xdot

        f2 = (
            self.tension_force_gain
            + m * self.g * sin_th * cos_th
            - m * self.pendulum_length * thetadot**2 * sin_th
            - 2 * self.cart_friction * xdot
            + u
        ) / alpha

        f3 = thetadot

        f4 = (
            self.tension_force_gain * cos_th
            - m * self.pendulum_length * thetadot**2 * sin_th * cos_th
            + (M + m) * self.g * sin_th
            - 2 * self.cart_friction * xdot * cos_th
        ) / (self.pendulum_length * alpha)

        return np.row_stack([f1, f2, f3, f4])

    def theta_to_trigonometric(self, X):
        theta = X["theta"].to_numpy()
        trig_values = np.column_stack([np.cos(theta), np.sin(theta)])
        X = X.drop("theta", axis=1)
        X[["theta_cos", "theta_sin"]] = trig_values

        thetadot = X["thetadot"].to_numpy()
        trig_values = np.column_stack([np.cos(thetadot), np.sin(thetadot)])
        X = X.drop("thetadot", axis=1)
        X[["thetadot_cos", "thetadot_sin"]] = trig_values

        return X

    def trigonometric_to_theta(self, X):
        trig_values = X[["theta_sin", "theta_cos"]].to_numpy()
        theta = np.arctan2(trig_values[:, 0], trig_values[:, 1])
        X = X.drop(["theta_sin", "theta_cos"], axis=1)
        X["theta"] = theta

        trig_values = X[["thetadot_sin", "thetadot_cos"]].to_numpy()
        thetadot = np.arctan2(trig_values[:, 0], trig_values[:, 1])
        X = X.drop(["thetadot_sin", "thetadot_cos"], axis=1)
        X["thetadot"] = thetadot

        return X.loc[:, self.feature_names_in_]

    def animate(self, X: TSCDataFrame, U: Optional[TSCDataFrame] = None):
        assert X.n_timeseries == 1

        import matplotlib.pyplot as plt
        from matplotlib import animation

        fig = plt.figure()

        min_x, max_x = X.iloc[:, 0].min(), X.iloc[:, 0].max()

        ax = plt.axes(
            xlim=(min_x - self.pendulum_length, max_x + self.pendulum_length),
            ylim=(-self.pendulum_length * 1.05, self.pendulum_length * 1.05),
        )

        def pendulum_pos(x_pos, theta):
            _cos, _sin = np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2)
            pos = np.array([_cos, _sin]) * self.pendulum_length
            pos[0] = pos[0] + x_pos
            return pos

        ax.set_aspect("equal")
        ax.plot()
        (line,) = ax.plot([], [], lw=2, label="pendulum")
        (point,) = ax.plot([], [], marker="o", lw=2, label="mounting")
        (pendulum_unstable,) = ax.plot(
            [], [], lw=2, linestyle="--", color="red", label="unstable location"
        )
        control_arrow = ax.arrow([], [], [], [], label="force")
        plt.legend()

        def init():
            mounting = np.array([float(X.iloc[0, 0]), 0])
            pendulum = pendulum_pos(float(X.iloc[0, 0]), float(X.iloc[0].loc["theta"]))
            unstable = pendulum_pos(mounting[0], 0)

            line.set_data(*np.row_stack([mounting, pendulum]).T)
            point.set_data(*mounting.T)
            pendulum_unstable.set_data(*np.row_stack([mounting, unstable]).T)

            if U is not None:
                control_arrow.set_data(
                    x=mounting[0], y=mounting[1], dx=float(U.iloc[0, 0]), dy=0
                )

            return (
                line,
                point,
                pendulum_unstable,
                control_arrow,
            )

        def _animate(i):
            mounting = np.array([float(X.iloc[i, 0]), 0])
            pendulum = pendulum_pos(float(X.iloc[i, 0]), float(X.iloc[i].loc["theta"]))
            unstable = pendulum_pos(mounting[0], 0)

            line.set_data(*np.row_stack([mounting, pendulum]).T)
            point.set_data(*mounting.T)
            pendulum_unstable.set_data(*np.row_stack([mounting, unstable]).T)

            if U is not None:
                control_arrow.set_data(
                    x=mounting[0], y=mounting[1], dx=float(U.iloc[i, 0]), dy=0
                )

            return (
                line,
                point,
                pendulum_unstable,
                control_arrow,
            )

        anim = animation.FuncAnimation(
            fig, _animate, init_func=init, frames=X.shape[0], interval=20, blit=True
        )
        return anim


class InvertedPendulum2(ControllableODE):
    """An inverted pendulum on a cart controlled by an electric motor.

    The system is parametrized with the voltage of the electric motor. The states include
    four observations: 1) position, 2) velocity, 3) angle from horizon and 4) angular velocity.

    Parameters
    ----------
    pendulum_mass: float
        Mass of pendulum, defaults to 0.0905 kg

    cart_mass: float
        Mass of the cart, defaults to 1.12 kg

    g: float
        Gravitational acceleration, defaults to 9.81 m/s^2

    tension_force_gain: float
        Conversion between electric motor input voltage in V and tension
        force in N, defaults to 7.5 N/V

    pendulum_length: float
        Length of the pendulum, defaults to 0.365 m

    cart_friction: float
        Dynamic damping coefficient on the cart, defaults to 6.65 kg/s

    """

    def __init__(
        self,
        pendulum_mass=0.0905,  # kg
        cart_mass=1.12,  # kg
        tension_force_gain=7.5,  # N/V
        pendulum_length=0.365,  # m
        cart_friction=6.65,  # kg/s
    ):
        self.pendulum_mass = pendulum_mass
        self.cart_mass = cart_mass
        self.g = 9.81  # m/s^2
        self.b = 0.1
        self.tension_force_gain = tension_force_gain
        self.pendulum_length = pendulum_length
        self.cart_friction = cart_friction
        self.l = 0.3

        super().__init__(
            feature_names_in=["x", "xdot", "theta", "thetadot"],
            control_names_in=["u"],
        )

    def _f(self, t, y, u):
        _, xdot, theta, thetadot = y

        M = 0.5
        m = 0.2
        b = 0.1
        ftheta = 0.1
        l = 0.3  # noqa: E741
        g = 9.81

        F = u
        v = y[1]
        theta = y[2]
        omega = y[3]
        der = np.zeros(4)
        der[0] = v
        der[1] = (
            m * l * np.sin(theta) * omega**2
            - m * g * np.sin(theta) * np.cos(theta)
            + m * ftheta * np.cos(theta) * omega
            + F
            - b * v
        ) / (M + m * (1 - np.cos(theta) ** 2))
        der[2] = omega
        der[3] = (
            (M + m) * (g * np.sin(theta) - ftheta * omega)
            - m * l * omega**2 * np.sin(theta) * np.cos(theta)
            - (F - b * v) * np.cos(theta)
        ) / (l * (M + m * (1 - np.cos(theta) ** 2)))
        return der

    def theta_to_trigonometric(self, X):
        theta = X["theta"].to_numpy()
        trig_values = np.column_stack([np.cos(theta), np.sin(theta)])
        X = X.drop("theta", axis=1)
        X[["theta_cos", "theta_sin"]] = trig_values

        thetadot = X["thetadot"].to_numpy()
        trig_values = np.column_stack([np.cos(thetadot), np.sin(thetadot)])
        X = X.drop("thetadot", axis=1)
        X[["thetadot_cos", "thetadot_sin"]] = trig_values

        return X

    def trigonometric_to_theta(self, X):
        trig_values = X[["theta_sin", "theta_cos"]].to_numpy()
        theta = np.arctan2(trig_values[:, 0], trig_values[:, 1])
        X = X.drop(["theta_sin", "theta_cos"], axis=1)
        X["theta"] = theta

        trig_values = X[["thetadot_sin", "thetadot_cos"]].to_numpy()
        thetadot = np.arctan2(trig_values[:, 0], trig_values[:, 1])
        X = X.drop(["thetadot_sin", "thetadot_cos"], axis=1)
        X["thetadot"] = thetadot

        return X.loc[:, self.feature_names_in_]

    def animate(self, X: TSCDataFrame, *, U: Optional[TSCDataFrame] = None):
        assert X.n_timeseries == 1

        import matplotlib.pyplot as plt
        from matplotlib import animation

        fig = plt.figure()

        min_x, max_x = X.iloc[:, 0].min(), X.iloc[:, 0].max()

        ax = plt.axes(
            xlim=(min_x - self.pendulum_length, max_x + self.pendulum_length),
            ylim=(-self.pendulum_length * 1.05, self.pendulum_length * 1.05),
        )

        def pendulum_pos(x_pos, theta):
            _cos, _sin = np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2)
            pos = np.array([_cos, _sin]) * self.pendulum_length
            pos[0] = pos[0] + x_pos
            return pos

        ax.set_aspect("equal")
        ax.plot()
        (line,) = ax.plot([], [], lw=2, label="pendulum")
        (point,) = ax.plot([], [], marker="o", lw=2, label="mounting")
        (pendulum_unstable,) = ax.plot(
            [], [], lw=2, linestyle="--", color="red", label="unstable location"
        )
        control_arrow = ax.arrow([], [], [], [], label="force")
        plt.legend()

        def init():
            mounting = np.array([float(X.iloc[0, 0]), 0])
            pendulum = pendulum_pos(float(X.iloc[0, 0]), float(X.iloc[0].loc["theta"]))
            unstable = pendulum_pos(mounting[0], 0)

            line.set_data(*np.row_stack([mounting, pendulum]).T)
            point.set_data(*mounting.T)
            pendulum_unstable.set_data(*np.row_stack([mounting, unstable]).T)

            if U is not None:
                control_arrow.set_data(
                    x=mounting[0], y=mounting[1], dx=float(U.iloc[0, 0]), dy=0
                )

            return (
                line,
                point,
                pendulum_unstable,
                control_arrow,
            )

        def _animate(i):
            mounting = np.array([float(X.iloc[i, 0]), 0])
            pendulum = pendulum_pos(float(X.iloc[i, 0]), float(X.iloc[i].loc["theta"]))
            unstable = pendulum_pos(mounting[0], 0)

            line.set_data(*np.row_stack([mounting, pendulum]).T)
            point.set_data(*mounting.T)
            pendulum_unstable.set_data(*np.row_stack([mounting, unstable]).T)

            if U is not None:
                control_arrow.set_data(
                    x=mounting[0], y=mounting[1], dx=float(U.iloc[i, 0]), dy=0
                )

            return (
                line,
                point,
                pendulum_unstable,
                control_arrow,
            )

        anim = animation.FuncAnimation(
            fig, _animate, init_func=init, frames=X.shape[0], interval=20, blit=True
        )
        return anim


class Burger1DPeriodicBoundary(ControllableODE):
    def __init__(self, n_spatial_points: int = 100, nu=0.01):
        self.nu = nu
        self.x_nodes = np.linspace(0, 2 * np.pi, n_spatial_points)
        self.dx = self.x_nodes[1] - self.x_nodes[0]

        super().__init__(
            feature_names_in=[f"x{i}" for i in range(n_spatial_points)],
            control_names_in=[f"u{i}" for i in range(n_spatial_points)],
            method="RK23",
            vectorized=True,
        )

    def _f(self, t, y, u):
        # TODO: should use an upwind scheme? But waves where more "averaged" on the top.
        # y_pad = np.concatenate([x[-3:-1], x])
        # advection = (-1.5 * y_pad[:-2] + 2 * y_pad[1:-1] - 0.5 * y_pad[2:]) / (2 * self.dx)

        # use central scheme as the advection seems
        y_pad = np.concatenate([y[[-1]], y, y[[0]]])
        advection = (-y_pad[:-2] + y_pad[2:]) / (2 * self.dx)

        # central finite difference scheme
        y_pad = np.concatenate([y[[-1]], y, y[[0]]])
        convection = (y_pad[:-2] - 2 * y_pad[1:-1] + y_pad[2:]) / (self.dx**2)

        x_dot = -np.multiply(advection, y, out=advection)
        x_dot += np.multiply(self.nu, convection, out=convection)
        x_dot += u

        return x_dot

    def _jac(self, t, x):
        n_nodes = len(self.x_nodes)

        if x.ndim == 1:
            x = x[:, np.newaxis]

        # x_pad = np.concatenate([x[[-1]], x, x[[0]]])
        x_pad = np.concatenate([x[-3:-1], x])
        xdiag = np.diag(x_pad.flatten())

        c = self.d_dx["coefficients"]

        first = np.zeros((n_nodes, n_nodes))

        for i in range(xdiag.shape[1] - 2):
            first[:, i] = (
                c[0] * xdiag[:-2, i + 1]
                + c[1] * xdiag[1:-1, i + 1]
                + c[2] * xdiag[2:, i + 1]
            ) / (2 * self.dx)

        first *= 2

        main_diagonal = np.diag(np.ones(n_nodes) * (-2))
        off_diagonal = np.diag(np.ones(n_nodes - 1), 1)

        second = main_diagonal + off_diagonal + off_diagonal.T
        second /= self.dx**2

        return -first - (np.eye(n_nodes) - self.nu * second)

    def _step(self, x, dt, u):
        # TODO: remove when finished
        if x.ndim == 1:
            x = x[:, np.newaxis]

        if x.ndim == 1:
            x = x[:, np.newaxis]

        # compute residual
        # R = SimPar.dt*(D*v).*v + (eye(N,N) - SimPar.dt*SimPar.nu*D2)*v - X(:,t-1)
        # - SimPar.dt*u;

        n_nodes = len(self.x_nodes)

        for _j in range(1, 20):
            x_pad = np.concatenate([x[[-1]], x, x[[0]]])
            advection_central = (-x_pad[:-2] + x_pad[2:]) / (2 * self.dx)
            advection_central *= dt
            advection_central = advection_central * x

            dxsq = self.dx**2
            factor = -(dt * self.nu) / dxsq

            convection = (
                factor * x_pad[:-2]
                + (1 + (2 * dt * self.nu) / dxsq) * x_pad[1:-1]
                + factor * x_pad[2:]
            )

            R = advection_central + convection - x - dt * u.T

            x_pad = np.concatenate([x[[-1]], x, x[[0]]])
            # x_pad = np.concatenate([x[-3:-1], x])
            xdiag = np.diag(x_pad.flatten())

            first = np.zeros((n_nodes, n_nodes))

            for i in range(xdiag.shape[1] - 2):
                first[:, i] = (-xdiag[:-2, i + 1] + xdiag[2:, i + 1]) / (2 * self.dx)

            first *= dt * 2

            main_diagonal = np.diag(np.ones(n_nodes) * (-2))
            off_diagonal = np.diag(np.ones(n_nodes - 1), 1)
            off_diagonal[0, -1] = 1

            second = main_diagonal + off_diagonal + off_diagonal.T
            second /= dxsq

            J = first + (np.eye(n_nodes) - dt * self.nu * second)
            x = x - np.linalg.solve(J, R)
        return x

    def solve_ivp(self, X_ic, U, time_values):
        # TODO: remove when finished
        x = X_ic

        ts = np.zeros((len(time_values), len(x)))
        ts[0, :] = X_ic.flatten()

        dts = np.append([0], np.diff(time_values))

        for i in range(1, len(time_values)):
            if isinstance(U, Callable):
                Ut = U(time_values[i], x)
            elif isinstance(U, pd.DataFrame):
                Ut = U.iloc[[i - 1]].to_numpy()
            else:
                Ut = U[[i - 1]]

            ts[i, :] = self._step(ts[i - 1], dts[i], Ut).flatten()

        return ts


class DCMotor(ControllableODE):
    r"""Model from https://arxiv.org/pdf/1611.03537.pdf
    Section 8.2 (Feedback control of a bilinear motor).

    Original source for detailed description (see Eq. 9):
     > S. Daniel-Berhe and H. Unbehauen. Experimental physical parameter
     > estimation of a thyristor driven DC-motor using the HMF-method.
     > Control Engineering Practice, 6:615?626, 1998.

    The model

    .. math::
       \dot{x}_1 = -(R_a/L_a)*x_1 - (k_m/L_a)*x_2 * u + u_a/L_a
       \dot{x}_2   = -(B/J)*x_2 + (k_m/J)*x_1 * u - \tau_l/J
       y = x_1

    Parameters L_a = 0.314, R_a = 12.345; k_m = 0.253, J = 0.00441, B = 0.00732;
       taul = 1.47, u_a = 60

    Constraints (scaled to [-1,1] in the final model)
    x_1: min = -10; max = 10
    x_2: min = -100; max = 100
    u: min = -4; max = 4
    """

    def __init__(self, ivp_kwargs=None):
        ivp_kwargs = ivp_kwargs or {}
        ivp_kwargs.setdefault("method", "RK23")
        super().__init__(
            feature_names_in=["x1", "x2"], control_names_in=["u"], **ivp_kwargs
        )

    def _runge_kutta(self, X, U, dt):
        U = U.flatten()

        k1 = self._f(None, X.T, U)
        k2 = self._f(None, X.T + k1 * dt / 2, U)
        k3 = self._f(None, X.T + k2 * dt / 2, U)
        k4 = self._f(None, X.T + k1 * dt, U)
        X_next = X.T + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return X_next.T

    def predict_vectorize(self, X_ic: TSCDataFrame, U: TSCDataFrame, time_values=None):
        """This is an adapted code to speed up the computation with a standard Runge Kutta 4
        scheme. To create a large dataset, the advanced integrators (RK23 or RK45 from scipy)
        turned out to be too costly.

        Parameters
        ----------
        X_ic
            initial conditions
        U
            Control input for each time series. The time delta and prediction horizon are
            extracted from here.

        time_values
            This parameter is required if U contains a single row. Otherwise the time sampling
            is inferred from U and the parameter is ignored.

        Returns
        -------
        TSCDataFrame
        """
        if X_ic.n_timeseries != U.n_timeseries:
            raise ValueError(
                f"The number of initial conditions ({X_ic.n_timeseries=}) must match the "
                f"number of control time series ({U.n_timeseries=})"
            )

        n_timesteps = U.tsc.check_const_timesteps()

        if n_timesteps > 1:
            is_single_step = False
            dt = U.tsc.check_const_time_delta()
        else:  # n_timesteps == 1:
            is_single_step = True
            if time_values is None or len(time_values) != 2:
                raise ValueError(
                    "If U contains only a single control input, then the parameter "
                    "time_values must be provided and contain the start and end "
                    "time."
                )
            dt = time_values[1] - time_values[0]

        # we always perform one more prediction than the number of control inputs
        n_timesteps += 1

        # allocate memory for prediction an set initial condition
        X_all = np.zeros([X_ic.shape[0], n_timesteps, X_ic.shape[1]])
        X_all[:, 0, :] = X_ic

        for i in range(n_timesteps - 1):
            # take every i-th element from the time series
            current_control_action = U.iloc[i :: n_timesteps - 1].to_numpy()

            # perform the next step according to the model
            X_all[:, i + 1, :] = self._runge_kutta(
                X_all[:, i, :], current_control_action, dt
            )

        if not is_single_step:
            time_values = np.append(U.time_values(), U.time_values()[-1] + dt)

        X_all = TSCDataFrame.from_tensor(
            tensor=X_all,
            feature_names=self.feature_names_in_,
            time_values=time_values,
        )
        return X_all, U

    def _f(self, t, y, u):
        dy = np.zeros_like(y)
        dy[0] = 19.10828025 - 39.3153 * y[0, :] - 32.2293 * y[1, :] * u
        dy[1] = -3.333333333 - 1.6599 * y[1, :] + 22.9478 * y[0, :] * u
        return dy


class VanDerPol(ControllableODE):
    _allowed_control_input = ["x", "y", "both"]

    def __init__(self, eps=1.0, control_coord="both", **solver_kwargs):
        solver_kwargs.setdefault("method", "RK45")
        solver_kwargs.setdefault("vectorized", True)

        self.eps = eps

        if control_coord in ["x", "y"]:
            control_names_in = [f"u{control_coord}"]
        elif control_coord == "both":
            control_names_in = ["ux", "uy"]
        else:
            raise ValueError(
                f"{control_coord=} invalid. Choose from {self._allowed_control_input}"
            )

        super().__init__(
            feature_names_in=["x1", "x2"],
            control_names_in=control_names_in,
            **solver_kwargs,
        )

    def _f(self, t, y, u):
        y1, y2 = y

        partial = len(self.control_names_in_) == 1
        if partial and "ux" in self.control_names_in_:
            u1, u2 = u, 0
        elif partial and "uy" in self.control_names_in_:
            u1, u2 = 0, u
        else:
            u1, u2 = u

        xdot = np.row_stack([y2 + u1, -y1 + self.eps * (1 - y1**2) * y2 + u2])
        return xdot


class Duffing(ControllableODE):
    def __init__(self, alpha=-1, beta=1, delta=0.6):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

        super().__init__(
            feature_names_in=["x1", "x2"],
            control_names_in=["u"],
        )

    def _f(self, t, Y, U):
        x1, x2 = Y.ravel()

        dx1 = x2
        dx2 = -self.delta * x2 - x1 * (self.beta + self.alpha * np.square(x1)) + U
        return np.array([dx1, float(dx2)])


class Lorenz(ODE):
    def __init__(self, sigma=10, rho=28, beta=8 / 3, **ivp_kwargs) -> None:
        """Lorenz system as a common system to analze for chaotic behavior.

        The standard parameter settings are taken from
        `<Wikipedia https://en.wikipedia.org/wiki/Lorenz_system>`__:

        "The system exhibits chaotic behavior for these (and nearby) values. If rho<1 then
        there is only one equilibrium point, which is at the origin. This point corresponds
        to no convection. All orbits converge to the origin, which is a global attractor."
        """
        super().__init__(feature_names_in=self.get_feature_names_out(), **ivp_kwargs)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def get_feature_names_out(self, input_features=None):
        return ["x1", "x2", "x3"]

    def _f(self, t, y):
        dy = np.zeros([3])
        dy[0] = self.sigma * (y[1] - y[0])
        dy[1] = y[0] * (self.rho - y[2]) - y[1]
        dy[2] = y[0] * y[1] - self.beta * y[2]
        return dy


# TODO:
#  include benchmark systems from: https://arxiv.org/pdf/2008.12874.pdf

if __name__ == "__main__":
    pass
    # import matplotlib.pyplot as plt
    #
    # X = np.array(100)
    #
    # Burger1DPeriodicBoundary().predict()
    #
    # exit()
    # if False:
    #     n_ic = 500
    #     state = np.random.uniform(-3.0, 3.0, size=(n_ic, 2))
    #
    #     x = np.linspace(-3, 3, 50)
    #     y = np.linspace(-3, 3, 50)
    #     xv, yv = np.meshgrid(x, y)
    #
    #     sys = VanDerPol()
    #
    #     state = np.column_stack([xv.flatten(), yv.flatten()])
    #     control = np.random.uniform(-3.0, 3.0, size=(state.shape[0], 1))
    #     control = np.zeros((state.shape[0], 1))
    #     trajectory, U = sys.predict(X=state, U=control, time_values=np.array([0.03]))
    #
    #     group = trajectory.groupby("ID")
    #     start, end = group.head(1).to_numpy(), group.tail(1).to_numpy()
    #
    #     for i in range(start.shape[0]):
    #         plt.plot(
    #             np.array([start[i, 0], end[i, 0]]),
    #             np.array([start[i, 1], end[i, 1]]),
    #             "black",
    #         )
    #
    #     n_timesteps = 500
    #     state = np.random.uniform(-3.0, 3.0, size=(1, 2))
    #     control = np.zeros((n_timesteps, 1))
    #     timevals = np.linspace(0, 10, n_timesteps)
    #     trajectory, _ = sys.predict(X=state, U=control, time_values=timevals)
    #
    #     plt.plot(trajectory["x1"].to_numpy(), trajectory["x2"].to_numpy())
    #
    # else:
    #     n_timesteps = 500
    #     state = np.random.uniform(-3.0, 3.0, size=(1, 2))
    #     control = np.random.uniform(-3.0, 3.0, size=(n_timesteps, 1))
    #     control = np.zeros((n_timesteps, 1))
    #     timevals = np.linspace(0, 10, n_timesteps)
    #     trajectory, U = VanDerPol(eps=1).predict(
    #         X=state, U=control, time_values=timevals
    #     )
    #
    #     plt.plot(trajectory["x1"].to_numpy(), trajectory["x2"].to_numpy())
    #
    # plt.show()


# if __name__ == "__main__":
#
#     from datafold.dynfold import DiffusionMaps
#     from datafold.pcfold import GaussianKernel
#     from datafold.utils.plot import plot_pairwise_eigenvector
#
#     import matplotlib.pyplot as plt
#     from datafold.pcfold import TSCDataFrame
#     from mpl_toolkits.mplot3d import Axes3D
#
#     t_eval = np.sort(np.random.default_rng(1).uniform(0, np.pi, size=(1000,)))
#     t_eval_oos = np.linspace(0, np.pi, 5000)
#     X = ClosedPeriodicalCurve((3, 1, 1, 5, 2), noise_std=0.05).eval(None, t_eval=t_eval)
#     X_oos = ClosedPeriodicalCurve((3, 1, 1, 5, 2)).eval(None, t_eval=t_eval_oos)
#
#     X = TSCDataFrame.from_single_timeseries(X)
#     X_oos = TSCDataFrame.from_single_timeseries(X_oos)
#
#     fig = plt.figure()
#     ax = fig.gca(projection="3d")
#     ax.scatter(
#         X["x"].to_numpy().ravel(),
#         X["y"].to_numpy().ravel(),
#         X["z"].to_numpy().ravel(),
#         label="parametric curve",
#         c=t_eval,
#         cmap=plt.cm.Spectral,
#     )
#     ax.legend()
#
#     dmap = DiffusionMaps(
#         kernel=GaussianKernel(epsilon=0.01),
#         n_eigenpairs=5,
#         # dist_kwargs=dict(cut_off=0.2),
#     ).fit(X)
#
#     Y = dmap.transform(X)
#     Y_oos = dmap.transform(X_oos)
#
#     f, ax = plt.subplots()
#
#     ax.scatter(
#         Y["dmap1"].to_numpy(), Y["dmap2"].to_numpy(), c=t_eval, cmap=plt.cm.Spectral,
#     )
#
#     ax.axis("equal")
#
#     f, ax = plt.subplots()
#
#     ax.scatter(
#         Y_oos["dmap1"].to_numpy(),
#         Y_oos["dmap2"].to_numpy(),
#         c=t_eval_oos,
#         cmap=plt.cm.Spectral,
#     )
#     ax.axis("equal")
#
#     Y_oos.loc[:, ("dmap1", "dmap2")].plot()
#
#     # plot_pairwise_eigenvector(
#     #     dmap.eigenvectors_, n=1, scatter_params=dict(c=t_eval, cmap=plt.cm.Spectral)
#     # )
#
#     plt.show()
