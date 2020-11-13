import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


class LimitCycle(object):
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

    def run(self, **kwargs):

        if "t_eval" in kwargs:
            kwargs["nr_steps"] = len(kwargs["t_eval"])
            t_diff = np.diff(kwargs["t_eval"])
            assert (
                len(np.unique(np.round(t_diff, decimals=10))) == 1
            )  # TODO only equidistant is supported at the moment!
            kwargs["dt"] = t_diff[0]
            del kwargs["t_eval"]

        if self.analytical:
            return self.run_analytical(**kwargs)
        else:
            return self.run_differences(**kwargs)

    def run_differences(self, x1, x2, dt, nr_steps):
        # use Euler, could also be solved analytically easily...
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

    def run_analytical(self, x1, x2, dt, nr_steps):
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


def solve_hopf_system(initial_conditions, t_eval, ic_type="xx"):
    def hopf_system(t, y):
        mu = 1
        y_dot = np.zeros(2)

        factor = mu - y[0] ** 2 - y[1] ** 2

        y_dot[0] = -y[1] + y[0] * factor
        y_dot[1] = y[0] + y[1] * factor
        return y_dot

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
            hopf_system,
            t_span=(t_eval[0], t_eval[-1]),
            y0=ic,
            t_eval=t_eval,
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

    return pd.concat(tsc_dfs, axis=0)


if __name__ == "__main__":
    df = LimitCycle(analytical=True).run(x1=5, x2=3, dt=0.01, nr_steps=5000)
    plt.polar(df["alpha"], df["radius"])
    plt.show()
