import unittest

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest
import pytest
from matplotlib.animation import FuncAnimation

from datafold import TSCDataFrame
from datafold.utils._systems import (
    Burger1DPeriodicBoundary,
    Duffing,
    Hopf,
    InvertedPendulum,
    Lorenz,
    VanDerPol,
)


class TestSystems(unittest.TestCase):
    def test_vanderpol01(self, plot=False):
        x = np.linspace(-3, 3, 10)
        y = np.linspace(-3, 3, 10)
        xv, yv = np.meshgrid(x, y)

        sys = VanDerPol()
        state = np.column_stack([xv.flatten(), yv.flatten()])

        control = np.zeros((state.shape[0], 2))
        trajectory, U = sys.predict(X=state, U=control, time_values=np.array([0, 0.03]))

        group = trajectory.groupby("ID")
        start, end = group.head(1).to_numpy(), group.tail(1).to_numpy()

        if plot:
            f, ax = plt.subplots()

            for i in range(start.shape[0]):
                ax.plot(
                    np.array([start[i, 0], end[i, 0]]),
                    np.array([start[i, 1], end[i, 1]]),
                    c="black",
                )

            # also include a longer example trajectory
            n_timesteps = 500
            rng = np.random.default_rng(1)
            state = rng.uniform(-3.0, 3.0, size=(1, 2))
            control = np.zeros((n_timesteps - 1, 2))
            timevals = np.linspace(0, 10, n_timesteps)
            trajectory, _ = sys.predict(X=state, U=control, time_values=timevals)
            ax.plot(trajectory["x1"].to_numpy(), trajectory["x2"].to_numpy())
            plt.show()

    def test_vanderpol02(self, plot=False):
        # a single longer time series
        n_timesteps = 500

        rng = np.random.default_rng()
        state = rng.uniform(-3.0, 3.0, size=(1, 2))
        control = np.zeros((n_timesteps - 1, 2))

        timevals = np.linspace(0, 10, n_timesteps)

        vdp = VanDerPol()
        trajectory, U = vdp.predict(X=state, U=control, time_values=timevals)

        if plot:
            plt.plot(trajectory["x1"].to_numpy(), trajectory["x2"].to_numpy())
            plt.show()

    def test_vanderpol03(self, plot=False):
        # multiple time series
        n_timesteps = 500

        # simulate 3 timeseries
        rng = np.random.default_rng(1)
        state = rng.uniform(-3.0, 3.0, size=(3, 2))

        timevals = np.linspace(0, 10, n_timesteps)

        df = pd.DataFrame(
            np.zeros((n_timesteps - 1, 2)), index=timevals[:-1], columns=["u1", "u2"]
        )

        control = TSCDataFrame.from_frame_list([df, df, df])

        sys = VanDerPol(eps=1)
        X_predict, U = sys.predict(X=state, U=control)

        self.assertIsInstance(X_predict, TSCDataFrame)
        self.assertIsInstance(U, TSCDataFrame)
        self.assertEqual(X_predict.n_timeseries, 3)
        self.assertEqual(U.n_timeseries, 3)

        if plot:
            f, ax = plt.subplots()
            for i in X_predict.ids:
                trajectory = X_predict.loc[[i], :]
                ax.plot(trajectory["x1"].to_numpy(), trajectory["x2"].to_numpy())
            plt.show()

    def test_burger01(self, plot=False):
        # simulates the setting from https://arxiv.org/pdf/1804.05291.pdf

        rng = np.random.default_rng(1)

        time_values = np.linspace(0, 3, 300)

        umin, umax = (-0.1, 0.1)

        f1 = lambda x: np.atleast_2d(np.exp(-((15 * (x - 0.25)) ** 2)))
        f2 = lambda x: np.atleast_2d(np.exp(-((15 * (x - 0.75)) ** 2)))

        rand_vals = rng.uniform(umin, umax, size=(len(time_values), 2))
        U1rand = lambda t: np.atleast_2d(np.interp(t, time_values, rand_vals[:, 0])).T
        U2rand = lambda t: np.atleast_2d(np.interp(t, time_values, rand_vals[:, 1])).T

        def U(t, x):
            if x.shape[1] == 1:
                x = x.T
            return U1rand(t) * f1(x) + U2rand(t) * f2(x)

        sys = Burger1DPeriodicBoundary()

        a = rng.uniform(0, 1)
        ic1 = np.exp(-((((sys.x_nodes) - 0.5) * 5) ** 2))
        ic2 = np.square(np.sin(4 * np.pi * sys.x_nodes))
        ic = a * ic1 + (1 - a) * ic2

        values, _ = sys.predict(ic, U=U, time_values=time_values)

        if plot:
            values = values.to_numpy()

            f = plt.figure()
            (line,) = plt.plot(sys.x_nodes, ic)

            def func(i):
                line.set_data(sys.x_nodes, values[i, :])
                return (line,)

            anim = FuncAnimation(f, func=func, frames=values.shape[0])  # noqa: F841
            plt.show()

    @pytest.mark.skip(reason="This test needs an update")
    def test_inverted_pendulum01(self):
        # unstable equilibrium
        X_stable = np.array([0, 0, 0, 0])

        # stable equilibrium
        X_unstable = np.array([0, 0, np.pi, 0])

        U = np.zeros((20, 1))

        sys = InvertedPendulum()
        actual_unstable, _ = sys.predict(X_stable, U=U)
        actual_stable, _ = sys.predict(X_unstable, U=U)

        expected_unstable = np.tile(X_stable, (U.shape[0] + 1, 1))
        expected_stable = np.tile(X_unstable, (U.shape[0] + 1, 1))

        nptest.assert_array_equal(expected_unstable, actual_unstable.to_numpy())
        nptest.assert_allclose(
            expected_stable, actual_stable.to_numpy(), rtol=0, atol=1e-15
        )

        self.assertIsInstance(actual_unstable, TSCDataFrame)
        self.assertIsInstance(actual_stable, TSCDataFrame)
        self.assertTrue((actual_unstable.columns == sys.feature_names_in_).all())
        self.assertTrue((actual_stable.columns == sys.feature_names_in_).all())

    def test_inverted_pendulum02(self):
        # unstable equilibrium
        X = np.array([0, 0, 0.1, 0])
        U = np.zeros((20, 1))

        sys = InvertedPendulum()
        X_predict, _ = sys.predict(X, U=U)

        actual_trigon = sys.theta_to_trigonometric(X_predict)

        new_cols = ["theta_sin", "theta_cos", "thetadot_sin", "thetadot_cos"]
        dropped_cols = ["theta", "thetadot"]
        [self.assertTrue(c in actual_trigon) for c in new_cols]
        [self.assertFalse(c in actual_trigon) for c in dropped_cols]

        actual_theta = sys.trigonometric_to_theta(actual_trigon)
        self.assertTrue((actual_theta.columns == sys.feature_names_in_).all())

        pdtest.assert_frame_equal(actual_theta, X_predict)

    def test_duffing01(self, plot=False):
        # setting parameters, because default leads to error (requires smaller time step size)

        X1, _ = Duffing(alpha=1, beta=-1).predict(
            np.array([1, 2]), U=np.zeros((999, 1))
        )
        X2, _ = Duffing(alpha=1, beta=-1).predict(
            np.array([1, 2]), U=np.zeros((999, 1)), time_values=np.arange(0, 10, 0.01)
        )

        U = TSCDataFrame.from_array(
            np.zeros((999, 1)), time_values=np.arange(0, 10 - 0.01, 0.01)
        )
        X3, _ = Duffing(alpha=1, beta=-1).predict(
            np.array([1, 2]), U=U, time_values=np.arange(0, 10, 0.01)
        )
        X4, _ = Duffing(alpha=1, beta=-1).predict(np.array([1, 2]), U=U)

        pdtest.assert_frame_equal(X1, X2)
        pdtest.assert_frame_equal(X1, X3)
        pdtest.assert_frame_equal(X1, X4)

        if plot:
            _, ax = plt.subplots()
            ax.plot(X1["x1"].to_numpy(), X2["x2"].to_numpy())
            plt.show()

    def test_hopf01(self, plot=False):
        # 1) mesh in Cartesian return only cartesian

        from datafold.utils.general import generate_2d_regular_mesh

        X = generate_2d_regular_mesh((-2, -2), (2, 2), 10, 10, ["x1", "x2"])

        system = Hopf()
        X = system.predict(X=X, time_values=np.linspace(0, 5, 200))

        if plot:
            plt.figure()
            for _, df in X.itertimeseries():
                Xnp = df.to_numpy()
                plt.plot(Xnp[:, 0], Xnp[:, 1], c="black", linewidth=1)
                plt.plot(Xnp[0, 0], Xnp[0, 1], marker=".", c="blue")
            plt.show()

    def test_hopf02(self, plot=False):
        # 2) mesh in Angular return Cartesian

        from datafold.utils.general import generate_2d_regular_mesh

        X = generate_2d_regular_mesh((0.01, 0), (3, 2 * np.pi), 10, 10, ["r", "angle"])

        system = Hopf()
        X = system.predict(X=X, time_values=np.linspace(0, 5, 200), ic_type="polar")

        if plot:
            plt.figure()
            for _, df in X.itertimeseries():
                Xnp = df.to_numpy()
                plt.plot(Xnp[:, 0], Xnp[:, 1], c="black", linewidth=1)
                plt.plot(Xnp[0, 0], Xnp[0, 1], marker=".", c="blue")
            plt.show()

    def test_hopf03(self, plot=False):
        # 3) mesh in Cartesian return Angular
        from datafold.utils.general import generate_2d_regular_mesh

        X = generate_2d_regular_mesh((-2, -2), (2, 2), 10, 10, ["x1", "x2"])

        system = Hopf(return_cart=False, return_polar=True)
        X = system.predict(X, time_values=np.linspace(0, 5, 200))

        if plot:
            f, ax = plt.subplots(subplot_kw={"projection": "polar"})

            for _, df in X.itertimeseries():
                ax.plot(
                    df.loc[:, "angle"].to_numpy(),
                    df.loc[:, "r"].to_numpy(),
                    color="black",
                )
                ax.grid(True)
            plt.show()

    def test_hopf04(self):
        from datafold.utils.general import generate_2d_regular_mesh

        X = generate_2d_regular_mesh((-2, -2), (2, 2), 10, 10, ["x1", "x2"])

        system = Hopf(return_polar=True)
        X = system.predict(X, time_values=np.linspace(0, 1, 20))

        self.assertEqual(X.shape[1], 4)
        self.assertEqual(["x1", "x2", "r", "angle"], system.get_feature_names_out())

    def test_hopf05(self):
        from datafold.utils.general import generate_2d_regular_mesh

        X_ic = generate_2d_regular_mesh((-2, -2), (2, 2), 10, 10, ["x1", "x2"])
        X_ic_polar = generate_2d_regular_mesh(
            (0.01, 0), (3, 2 * np.pi), 10, 10, ["r", "angle"]
        )

        system = Hopf()
        X = system.predict(X=X_ic, time_values=0.1)
        X_polar = system.predict(X=X_ic_polar, time_values=0.1, ic_type="polar")
        self.assertEqual(X.n_timesteps, 2)
        self.assertEqual(X.n_timeseries, 100)
        pdtest.assert_index_equal(pd.Index(["x1", "x2"]), X.columns, check_names=False)
        self.assertEqual(X_polar.n_timesteps, 2)
        self.assertEqual(X_polar.n_timeseries, 100)
        pdtest.assert_index_equal(
            pd.Index(["x1", "x2"]), X_polar.columns, check_names=False
        )

        system = Hopf(return_cart=False, return_polar=True)
        X = system.predict(X=X_ic, time_values=0.1)
        X_polar = system.predict(X=X_ic_polar, time_values=0.1, ic_type="polar")
        self.assertEqual(X.n_timesteps, 2)
        self.assertEqual(X.n_timeseries, 100)
        pdtest.assert_index_equal(
            pd.Index(["r", "angle"]), X.columns, check_names=False
        )
        self.assertEqual(X_polar.n_timesteps, 2)
        self.assertEqual(X_polar.n_timeseries, 100)
        pdtest.assert_index_equal(
            pd.Index(["r", "angle"]), X_polar.columns, check_names=False
        )

    def test_hopf06(self):
        from datafold.utils.general import generate_2d_regular_mesh

        X_ic = generate_2d_regular_mesh((-2, -2), (2, 2), 10, 10)
        X_ic_polar = generate_2d_regular_mesh((0.01, 0), (3, 2 * np.pi), 10, 10)

        system = Hopf()
        X = system.predict(X=X_ic, time_values=0.1)
        X_polar = system.predict(X=X_ic_polar, time_values=0.1, ic_type="polar")
        self.assertEqual(X.n_timesteps, 2)
        self.assertEqual(X.n_timeseries, 100)
        pdtest.assert_index_equal(pd.Index(["x1", "x2"]), X.columns, check_names=False)
        self.assertEqual(X_polar.n_timesteps, 2)
        self.assertEqual(X_polar.n_timeseries, 100)
        pdtest.assert_index_equal(
            pd.Index(["x1", "x2"]), X_polar.columns, check_names=False
        )

        system = Hopf(return_cart=False, return_polar=True)
        X = system.predict(X=X_ic, time_values=0.1)
        X_polar = system.predict(X=X_ic_polar, time_values=0.1, ic_type="polar")
        self.assertEqual(X.n_timesteps, 2)
        self.assertEqual(X.n_timeseries, 100)
        pdtest.assert_index_equal(
            pd.Index(["r", "angle"]), X.columns, check_names=False
        )
        self.assertEqual(X_polar.n_timesteps, 2)
        self.assertEqual(X_polar.n_timeseries, 100)
        pdtest.assert_index_equal(
            pd.Index(["r", "angle"]), X_polar.columns, check_names=False
        )

    def test_lorenz(self, plot=False):
        time_values = np.arange(0, 200, 0.001)
        lorenz = Lorenz()
        X = lorenz.predict(X=[-5, 9, 30], time_values=time_values)

        self.assertIsInstance(X, TSCDataFrame)
        self.assertEqual(X.n_timeseries, 1)
        self.assertEqual(X.n_timesteps, len(time_values))
        self.assertEqual(X.columns.to_list(), lorenz.feature_names_in_)

        if plot:
            X = X.to_numpy()

            f = plt.figure()
            ax = f.add_subplot(projection="3d")

            ax.plot3D(X[:, 0], X[:, 1], X[:, 2], "-k")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

            f, ax = plt.subplots(nrows=3)

            ax[0].plot(time_values, X[:, 0], label="x")
            ax[0].set_ylabel("x")
            ax[0].set_xlim([-1, 100])

            ax[1].plot(time_values, X[:, 1], label="y")
            ax[1].set_ylabel("y")
            ax[1].set_xlim([-1, 100])

            ax[2].plot(time_values, X[:, 2], label="z")
            ax[2].set_ylabel("z")
            ax[2].set_xlabel("time")
            ax[2].set_xlim([-1, 100])
            plt.show()


if __name__ == "__main__":
    TestSystems().test_hopf02()
