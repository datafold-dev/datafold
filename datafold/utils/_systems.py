#!/usr/bin/env python3
from scipy.interpolate import interp1d
import abc
from typing import Callable, Optional, Union

import findiff as fd
import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp

from datafold.dynfold.base import TSCPredictMixin
from datafold.pcfold import TSCDataFrame


class DynamicalSystem(TSCPredictMixin, metaclass=abc.ABCMeta):

    # TODO: initial_conditions should be "X" to align with the Predict models
    @abc.abstractmethod
    def predict(self, X, *, U=None, time_values=None, **kwargs):
        raise NotImplementedError("base class")


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


class HopfSystem(DynamicalSystem):
    """From

    Lawrence Perko. Differential equations and dynamical systems, volume 7. Springer
    Science & Business Media, 2013. page 350

    https://link.springer.com/book/10.1007/978-1-4613-0003-8
    """

    def __init__(self, mu: float = 1, return_xx: bool = True, return_rt: bool = True):
        # TODO: rename "return_xx" and "return_rt"
        self.mu = mu
        self.return_xx = return_xx
        self.return_rt = return_rt

        if not self.return_xx and not return_rt:
            raise ValueError(f"cannot have both {return_xx=} and {return_rt=}")

    def hopf_system(self, t, y):
        """Autonomous, planar ODE System"""
        # TODO make private

        y_dot = np.zeros(2)
        factor = self.mu - y[0] ** 2 - y[1] ** 2

        y_dot[0] = -y[1] + y[0] * factor
        y_dot[1] = y[0] + y[1] * factor
        return y_dot

    def predict(self, X, time_values, ic_type="xx"):
        assert ic_type in ["xx", "rt"]
        assert X.ndim == 2
        assert X.shape[1] == 2

        if ic_type == "rt":
            new_ic = np.copy(X)
            new_ic[:, 0] = X[:, 0] * np.cos(X[:, 1])
            new_ic[:, 1] = X[:, 0] * np.sin(X[:, 1])
            X = new_ic

        tsc_dfs = []

        for _id, ic in enumerate(X):
            solution = solve_ivp(
                self.hopf_system,
                t_span=(time_values[0], time_values[-1]),
                y0=ic,
                t_eval=time_values,
            )
            current_solution = solution["y"].T
            theta = np.arctan2(current_solution[:, 1], current_solution[:, 0])
            radius = current_solution[:, 0] / np.cos(theta)

            current_solution = np.column_stack([current_solution, radius, theta])

            solution = pd.DataFrame(
                data=current_solution,
                index=pd.MultiIndex.from_arrays(
                    [np.ones(len(solution["t"])) * _id, solution["t"]]
                ),
                columns=["x1", "x2", "r", "theta"],
            )

            tsc_dfs.append(solution)

        result = pd.concat(tsc_dfs, axis=0)

        if not self.return_xx:
            result = result.drop(["x1", "x2"], axis=1)
        elif not self.return_rt:
            result = result.drop(["r", "theta"], axis=1)

        # TODO: return as TSCDataFrame
        return result


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
    https://towardsdatascience.com/a-beginners-guide-to-simulating-dynamical-systems-with-python-a29bc27ad9b1
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


class ControllableODE(DynamicalSystem, metaclass=abc.ABCMeta):
    def __init__(
        self,
        feature_names_in,
        control_names_in,
        **ivp_kwargs,
    ):
        # TODO: possibly move one up to DynamicalSystem

        self.feature_names_in_ = feature_names_in
        self.n_features_in_ = len(self.feature_names_in_)
        self.control_names_in_ = control_names_in
        self.n_control_in_ = len(self.control_names_in_)
        self._default_step_size = 0.01

        self.ivp_kwargs = ivp_kwargs
        self.ivp_kwargs.setdefault("method", "RK45")
        self.ivp_kwargs.setdefault("vectorized", True)

    @abc.abstractmethod
    def _f(self, t, X, U):
        """Right-hand side of the ODE.

        The return of the function must match the 'fun' parameter in Scipy's 'solve_ivp'
        function.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

        The U parameter is an extra parameter for the control input.
        """
        raise NotImplementedError("base class")

    def predict(
        self,
        X,
        *,
        U: Optional[Union[np.ndarray, TSCDataFrame, Callable]] = None,
        time_values: Optional[np.ndarray] = None,
        require_last_control_state=False,
    ):
        # TODO: need to support if X is TSCDataFrame!

        # TODO: make U really optional (do not apply any control if U is None) --
        #  this needs to be addressed in _f, where U is ignored (or set to zero, but this
        #  needs computations.

        # some validation
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

        if time_values is None:
            self.dt_ = self._default_step_size
        else:
            time_values = np.asarray(time_values)
            if len(time_values) < 2:
                raise ValueError(f"Parameter time_values must include at least two elements. Got {len(time_values)=}")

            self.dt_ = time_values[1] - time_values[0]

        if not isinstance(U, Callable):

            if isinstance(U, np.ndarray) and X.shape[0] == U.shape[0]:
                if time_values is None:
                    raise ValueError("For multiple one-step predictions, the parameter "
                                     "'time_values' cannot be None")

                # Interpret as one-step prediction of multiple initial conditions
                idx = pd.MultiIndex.from_arrays(
                    [np.arange(U.shape[0]), np.ones(U.shape[0]) * time_values[0]])
                U = TSCDataFrame(U, index=idx, columns=self.control_names_in_)

            elif X.shape[0] > 1 and not isinstance(U, TSCDataFrame):
                raise ValueError(
                    "To solve for multiple initial conditions `U` must be of type "
                    f"TSCDataFrame. Got {type(U)}"
                )

        self._requires_last_control_state = False
        time_values = self._validate_and_set_time_values_predict(
            time_values=time_values, X=X, U=U
        )

        if isinstance(U, TSCDataFrame):
            U.tsc.check_equal_timevalues()
            U.tsc.check_required_n_timesteps(len(time_values) - 1)
        elif isinstance(U, np.ndarray):
            U = TSCDataFrame.from_array(
                U, time_values=time_values[:-1], feature_names=self.control_names_in_
            )

        if isinstance(X, pd.DataFrame): # TODO: work with TSCDataFrame instead and cast to it if X is np.ndarray
            X = X.to_numpy()

        X_sol = list()

        for i in range(X.shape[0]):
            ic = X[i]

            if isinstance(U, Callable):
                # user specified input
                Ufunc = U
            elif len(time_values) == 2:
                Ufunc = lambda t, x: U.iloc[[i], :].to_numpy()
            else:
                # interpolates control input from data
                U_interp = U.loc[[U.ids[i]], :].to_numpy()

                interp_control = []

                for i in range(self.n_control_in_):
                    func = lambda t, x: interp1d(time_values[:-1], U_interp[:, i], kind='previous', fill_value="extrapolate")(t)
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

        if isinstance(U, Callable):
            X_sol_but_last = X_sol.tsc.drop_last_n_samples(1)

            tv = X_sol_but_last.index.get_level_values(TSCDataFrame.tsc_time_idx_name).to_numpy()

            # turn callable into actual data -- needs to be re-computed as I do not see a way
            # to access this from the scipy ODE solver

            # TODO: this only works if U is vectorized, maybe need an element-by-element way too...
            U = U(tv, X_sol_but_last.to_numpy())
            U = TSCDataFrame.from_same_indices_as(
                X_sol_but_last, values=U, except_columns=self.control_names_in_
            )

        return X_sol, U


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

        super(InvertedPendulum, self).__init__(
            n_features_in=4,
            feature_names_in=["x", "xdot", "theta", "thetadot"],
            n_control_in=1,
            control_names_in=["u"],
        )

    def _f(self, t, x, u):
        _, xdot, theta, thetadot = x

        m = self.pendulum_mass
        M = self.cart_mass

        sin_th = np.sin(theta)
        cos_th = np.cos(theta)

        alpha = M + m * np.square(sin_th)

        f1 = xdot

        f2 = (
            self.tension_force_gain * u
            + m * self.g * sin_th * cos_th
            - m * self.pendulum_length * thetadot**2 * sin_th
            - 2 * self.cart_friction * xdot
        ) / alpha

        f3 = thetadot

        f4 = (
            self.tension_force_gain * u * cos_th
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

    def animate(self, X: TSCDataFrame, U: TSCDataFrame):
        assert X.n_timeseries == 1

        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

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
        plt.legend()

        def init():
            mounting = np.array([float(X.iloc[0, 0]), 0])
            pendulum = pendulum_pos(float(X.iloc[0, 0]), float(X.iloc[0].loc["theta"]))
            unstable = pendulum_pos(mounting[0], 0)

            line.set_data(*np.row_stack([mounting, pendulum]).T)
            point.set_data(*mounting.T)
            pendulum_unstable.set_data(*np.row_stack([mounting, unstable]).T)

            return (
                line,
                point,
                pendulum_unstable,
            )

        def _animate(i):
            mounting = np.array([float(X.iloc[i, 0]), 0])
            pendulum = pendulum_pos(float(X.iloc[i, 0]), float(X.iloc[i].loc["theta"]))
            unstable = pendulum_pos(mounting[0], 0)

            line.set_data(*np.row_stack([mounting, pendulum]).T)
            point.set_data(*mounting.T)
            pendulum_unstable.set_data(*np.row_stack([mounting, unstable]).T)
            return (
                line,
                point,
                pendulum_unstable,
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

        self.d_dx = fd.coefficients(deriv=1, acc=2)["forward"]
        self.d2_dx2 = fd.coefficients(deriv=2, acc=2)["center"]

        super(Burger1DPeriodicBoundary, self).__init__(
            feature_names_in=[f"x{i}" for i in range(n_spatial_points)],
            control_names_in=[f"u{i}" for i in range(n_spatial_points)],
            **{"method": "RK23", "vectorized": True},
        )

    def _f(self, t, x, u):
        # TODO: should use an upwind scheme? But waves where more "averaged" on the top.
        # x_pad = np.concatenate([x[-3:-1], x])
        # advection = (-1.5 * x_pad[:-2] + 2 * x_pad[1:-1] - 0.5 * x_pad[2:]) / (2 * self.dx)

        # use central scheme as the advection seems
        x_pad = np.concatenate([x[[-1]], x, x[[0]]])
        advection = (-x_pad[:-2] + x_pad[2:]) / (2 * self.dx)

        # central finite difference scheme
        x_pad = np.concatenate([x[[-1]], x, x[[0]]])
        convection = (x_pad[:-2] - 2 * x_pad[1:-1] + x_pad[2:]) / (self.dx ** 2)

        x_dot = -np.multiply(advection, x, out=advection)
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

        for i in range(xdiag.shape[1]-2):
            first[:, i] = (c[0] * xdiag[:-2, i+1] + c[1] * xdiag[1:-1, i+1] + c[2] * xdiag[2:, i+1]) / (2 * self.dx)

        first *= 2

        main_diagonal = np.diag(np.ones(n_nodes)*(-2))
        off_diagonal = np.diag(np.ones(n_nodes-1), 1)

        second = main_diagonal + off_diagonal + off_diagonal.T
        second /= (self.dx ** 2)

        return -first - (np.eye(n_nodes) - self.nu * second)


    def _step(self, x, dt, u):
        # TODO: remove when finished
        if x.ndim == 1:
            x = x[:, np.newaxis]

        if x.ndim == 1:
            x = x[:, np.newaxis]

        # compute residual
        # R = SimPar.dt*(D*v).*v + (eye(N,N) - SimPar.dt*SimPar.nu*D2)*v - X(:,t-1) - SimPar.dt*u;

        n_nodes = len(self.x_nodes)

        for j in range(1, 20):

            x_pad = np.concatenate([x[[-1]], x, x[[0]]])
            advection_central = (-x_pad[:-2] + x_pad[2:]) / (2 * self.dx)
            advection_central *= dt
            advection_central = advection_central*x


            dxsq = (self.dx ** 2)
            factor = - (dt * self.nu) / dxsq

            convection = factor * x_pad[:-2] + (1 + (2 * dt * self.nu)/dxsq) * x_pad[1:-1] + factor * x_pad[2:]

            R = advection_central + convection - x - dt*u.T

            x_pad = np.concatenate([x[[-1]], x, x[[0]]])
            # x_pad = np.concatenate([x[-3:-1], x])
            xdiag = np.diag(x_pad.flatten())



            first = np.zeros((n_nodes, n_nodes))

            for i in range(xdiag.shape[1]-2):
                first[:, i] = (-xdiag[:-2, i+1] + xdiag[2:, i+1]) / (2 * self.dx)

            first *= dt*2

            main_diagonal = np.diag(np.ones(n_nodes)*(-2))
            off_diagonal = np.diag(np.ones(n_nodes-1), 1)
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
                Ut =  U(time_values[i], x)
            elif isinstance(U, pd.DataFrame):
                Ut = U.iloc[[i-1]].to_numpy()
            else:
                Ut = U[[i-1]]

            ts[i, :] = self._step(ts[i-1], dts[i], Ut).flatten()

        return ts




class VanDerPol(ControllableODE):
    def __init__(self, eps=1.0):
        super(VanDerPol, self).__init__(
            n_features_in=2,
            feature_names_in=["x1", "x2"],
            n_control_in=2,
            control_names_in=["u1", "u2"],
        )
        self.eps = eps

    def _f(self, t, x, u):

        x1, x2 = x
        u1, u2 = u

        xdot = np.row_stack([x2 + u1, -x1 + self.eps * (1 - x1**2) * x2 + u2])
        return xdot


class Duffing1D(ControllableODE):
    def __init__(self, alpha=-1, beta=1, delta=0.6):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

        super(Duffing1D, self).__init__(
            n_features_in=2,
            feature_names_in=["x1", "x2"],
            n_control_in=1,
            control_names_in=["u"],
        )

    def _f(self, t, X, U):

        x1, x2 = X.ravel()

        f1 = x2
        f2 = -self.delta * x2 - self.alpha * x1 - self.beta * x1**3 + U
        return np.array([f1, f2])


# TODO:
#  include benchmark systems from: https://arxiv.org/pdf/2008.12874.pdf

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X = np.array(100)

    Burger1DPeriodicBoundary().predict()

    exit()

    if False:
        n_ic = 500
        state = np.random.uniform(-3.0, 3.0, size=(n_ic, 2))

        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        xv, yv = np.meshgrid(x, y)

        sys = VanDerPol()

        state = np.column_stack([xv.flatten(), yv.flatten()])
        control = np.random.uniform(-3.0, 3.0, size=(state.shape[0], 1))
        control = np.zeros((state.shape[0], 1))
        trajectory, U = sys.predict(X=state, U=control, time_values=np.array([0.03]))

        group = trajectory.groupby("ID")
        start, end = group.head(1).to_numpy(), group.tail(1).to_numpy()

        for i in range(start.shape[0]):
            plt.plot(
                np.array([start[i, 0], end[i, 0]]),
                np.array([start[i, 1], end[i, 1]]),
                "black",
            )

        n_timesteps = 500
        state = np.random.uniform(-3.0, 3.0, size=(1, 2))
        control = np.zeros((n_timesteps, 1))
        timevals = np.linspace(0, 10, n_timesteps)
        trajectory, _ = sys.predict(X=state, U=control, time_values=timevals)

        plt.plot(trajectory["x1"].to_numpy(), trajectory["x2"].to_numpy())

    else:
        n_timesteps = 500
        state = np.random.uniform(-3.0, 3.0, size=(1, 2))
        control = np.random.uniform(-3.0, 3.0, size=(n_timesteps, 1))
        control = np.zeros((n_timesteps, 1))
        timevals = np.linspace(0, 10, n_timesteps)
        trajectory, U = VanDerPol(eps=1).predict(
            X=state, U=control, time_values=timevals
        )

        plt.plot(trajectory["x1"].to_numpy(), trajectory["x2"].to_numpy())

    plt.show()


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
