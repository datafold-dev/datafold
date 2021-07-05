#!/usr/bin/env python3

import abc

import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp

from datafold.pcfold import TSCDataFrame


# TODO: could be a TSCPredictMixin
class DynamicalSystem(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def predict(self, initial_conditions, time_values, **kwargs):
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

        if "t_eval" in kwargs:
            kwargs["nr_steps"] = len(kwargs["t_eval"])
            t_diff = np.diff(kwargs["t_eval"])
            assert (
                len(np.unique(np.round(t_diff, decimals=10))) == 1
            )  # TODO only equidistant is supported at the moment!
            kwargs["dt"] = t_diff[0]
            del kwargs["t_eval"]

        if self.analytical:
            return self.eval_analytical(**kwargs)
        else:
            return self.eval_finite_differences(**kwargs)

    def eval_finite_differences(self, x1, x2, dt, nr_steps):
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
        t = np.linspace(0, dt * (nr_steps - 1), nr_steps)

        a0 = self._compute_angle(x1=x1, x2=x2)
        r0 = (
            np.linalg.norm(np.array([x1, x2])) + 1e-10
        )  # add a small number to avoid division by zero

        a_vals = 1 / self.eps * t + a0

        r_vals = np.exp(t) / np.sqrt(-1 + np.exp(2 * t) + 1 / r0 ** 2)

        # x, y = self._as_cartesian_coordinates(rad=r_vals, ang=a_vals)
        self.obs = pd.DataFrame(
            np.vstack([a_vals, r_vals]).T, index=t, columns=["alpha", "radius"]
        )
        return self.obs


class HopfSystem(DynamicalSystem):
    """
    Lawrence Perko. Differential equations and dynamical systems, volume 7. Springer Science & Business
    Media, 2013. page 350

    https://link.springer.com/book/10.1007/978-1-4613-0003-8
    """

    def __init__(self, mu: float = 1, return_xx: bool = True, return_rt: bool = True):

        self.mu = mu
        self.return_xx = return_xx
        self.return_rt = return_rt

        if not self.return_xx and not return_rt:
            raise ValueError("Cannot have both return_xx=False and return_rt=False")

    def hopf_system(self, t, y):
        """Autonmous, planar ODE System"""

        y_dot = np.zeros(2)
        factor = self.mu - y[0] ** 2 - y[1] ** 2

        y_dot[0] = -y[1] + y[0] * factor
        y_dot[1] = y[0] + y[1] * factor
        return y_dot

    def predict(self, initial_conditions, time_values, ic_type="xx"):

        assert ic_type in ["xx", "rt"]
        assert initial_conditions.ndim == 2
        assert initial_conditions.shape[1] == 2

        if ic_type == "rt":
            new_ic = np.copy(initial_conditions)
            new_ic[:, 0] = initial_conditions[:, 0] * np.cos(initial_conditions[:, 1])
            new_ic[:, 1] = initial_conditions[:, 0] * np.sin(initial_conditions[:, 1])
            initial_conditions = new_ic

        tsc_dfs = []

        for _id, ic in enumerate(initial_conditions):
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

    def predict(self, initial_conditions, time_values, **kwargs):
        assert initial_conditions is None
        return self._closed_system(time_values)


class Pendulum(DynamicalSystem):
    """
    System explained:
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

    def predict(self, initial_conditions, time_values, **kwargs):
        # initial_conditions = theta_0 -- theta_1

        self._compute_cart_parameters()

        initial_conditions = np.asarray(initial_conditions)

        if initial_conditions.ndim == 1:
            initial_conditions = initial_conditions[np.newaxis, :]

        solution_frames = []

        for ic_idx in range(initial_conditions.shape[0]):
            theta_coord = odeint(
                self._integrate_pendulum_sim, initial_conditions[ic_idx, :], time_values
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


# TODO:
#  include benchmark systems from: https://arxiv.org/pdf/2008.12874.pdf


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
