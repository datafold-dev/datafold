import unittest
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import pandas.testing as pdtest
import numpy.testing as nptest

from datafold.utils._systems import VanDerPol, Duffing1D, InvertedPendulum, Burger1DPeriodicBoundary
from datafold import TSCDataFrame
import matplotlib.pyplot as plt


class TestSystems(unittest.TestCase):

    def test_vanderpol01(self, plot=True):

        x = np.linspace(-3, 3, 20)
        y = np.linspace(-3, 3, 20)
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
                ax.plot(np.array([start[i, 0], end[i, 0]]), np.array([start[i, 1], end[i, 1]]), c="black")

            # also include a longer example trajectory
            n_timesteps = 500
            state = np.random.uniform(-3., 3., size=(1, 2))
            control = np.zeros((n_timesteps-1, 2))
            timevals = np.linspace(0, 10, n_timesteps)
            trajectory, _ = sys.predict(X=state, U=control, time_values=timevals)
            ax.plot(trajectory["x1"].to_numpy(), trajectory["x2"].to_numpy())
            plt.show()

    def test_vanderpol02(self, plot=True):
        # a single longer time series
        n_timesteps = 500
        state = np.random.uniform(-3., 3., size=(1, 2))
        control = np.zeros((n_timesteps-1, 2))

        timevals = np.linspace(0, 10, n_timesteps)

        vdp = VanDerPol()
        trajectory, U = vdp.predict(X=state, U=control, time_values=timevals)

        if plot:
            plt.plot(trajectory["x1"].to_numpy(), trajectory["x2"].to_numpy())
            plt.show()

    def test_vanderpol03(self, plot=True):
        # multiple time series
        n_timesteps = 500

        # simulate 3 timeseries
        state = np.random.uniform(-3., 3., size=(3, 2))

        timevals = np.linspace(0, 10, n_timesteps)

        df = pd.DataFrame(np.zeros((n_timesteps-1, 2)), index=timevals[:-1], columns=["u1", "u2"])

        control = TSCDataFrame.from_frame_list([df, df, df])
        X_predict, U = VanDerPol(eps=1).predict(X=state, U=control, time_values=timevals)

        if plot:
            f, ax = plt.subplots()
            for i in X_predict.ids:
                trajectory = X_predict.loc[[i], :]
                ax.plot(trajectory["x1"].to_numpy(), trajectory["x2"].to_numpy())
            plt.show()

    def test_burger01(self, plot=True):
        # simulates the setting from https://arxiv.org/pdf/1804.05291.pdf

        rng = np.random.default_rng(1)

        time_values = np.linspace(0, 3, 300)

        umin, umax = (-0.1, 0.1)

        f1 = lambda x: np.atleast_2d(np.exp(-(15 * (x - 0.25)) ** 2))
        f2 = lambda x: np.atleast_2d(np.exp(-(15 * (x - 0.75)) ** 2))

        rand_vals = rng.uniform(umin, umax, size=(len(time_values), 2))
        U1rand = lambda t: np.atleast_2d(np.interp(t, time_values, rand_vals[:, 0])).T
        U2rand = lambda t: np.atleast_2d(np.interp(t, time_values, rand_vals[:, 1])).T

        def U(t, x):
            if x.shape[1] == 1:
                x = x.T
            return U1rand(t) * f1(x) + U2rand(t) * f2(x)

        sys = Burger1DPeriodicBoundary()

        a = rng.uniform(0, 1)
        ic1 = np.exp(-(((sys.x_nodes) - .5) * 5) ** 2)
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

            anim = FuncAnimation(f, func=func, frames=values.shape[0])
            plt.show()

    def test_inverted_pendulum01(self):

        # unstable equilibrium
        X_stable = np.array([0, 0, 0, 0])

        # stable equilibrium
        X_unstable = np.array([0, 0, np.pi, 0])

        U = np.zeros((20,1))

        sys = InvertedPendulum()
        actual_unstable, _ = sys.predict(X_stable, U=U)
        actual_stable, _ = sys.predict(X_unstable, U=U)

        expected_unstable = np.tile(X_stable, (U.shape[0]+1, 1))
        expected_stable = np.tile(X_unstable, (U.shape[0]+1, 1))

        nptest.assert_array_equal(expected_unstable, actual_unstable.to_numpy())
        nptest.assert_allclose(expected_stable, actual_stable.to_numpy(), rtol=0, atol=1E-15)

        self.assertIsInstance(actual_unstable, TSCDataFrame)
        self.assertIsInstance(actual_stable, TSCDataFrame)
        self.assertTrue((actual_unstable.columns == sys.feature_names_in_).all())
        self.assertTrue((actual_stable.columns == sys.feature_names_in_).all())

    def test_inverted_pendulum02(self):
        # unstable equilibrium
        X = np.array([0,0,0.1,0])
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


    def test_duffing01(self, plot=True):

        X1, U1 = Duffing1D().predict(np.array([1, 2]), U=np.zeros((999, 1)))
        X2, U2 = Duffing1D().predict(np.array([1, 2]), U=np.zeros((999, 1)), time_values=np.arange(0, 10, 0.01))

        U = TSCDataFrame.from_array(np.zeros((999, 1)), time_values=np.arange(0, 10-0.01, 0.01))
        X3, U3 = Duffing1D().predict(np.array([1, 2]), U=U, time_values=np.arange(0, 10, 0.01))
        X4, U4 = Duffing1D().predict(np.array([1, 2]), U=U)

        pdtest.assert_frame_equal(X1, X2)
        pdtest.assert_frame_equal(X1, X3)
        pdtest.assert_frame_equal(X1, X4)

        if plot:
            f, ax = plt.subplots()
            ax.plot(X1["x1"].to_numpy(), X2["x2"].to_numpy())
            plt.show()
