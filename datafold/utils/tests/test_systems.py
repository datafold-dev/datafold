import unittest

import numpy as np
import pandas as pd
import pandas.testing as pdtest

from datafold.utils._systems import VanDerPol, Duffing1D
from datafold import TSCDataFrame
import matplotlib.pyplot as plt


class TestSystems(unittest.TestCase):

    def test_vanderpol01(self, plot=True):

        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        xv, yv = np.meshgrid(x, y)

        sys = VanDerPol()
        state = np.column_stack([xv.flatten(), yv.flatten()])

        control = np.zeros((state.shape[0], 2))
        trajectory, U = sys.predict(X=state, U=control, time_values=np.array([0.03]))

        group = trajectory.groupby("ID")
        start, end = group.head(1).to_numpy(), group.tail(1).to_numpy()

        if plot:
            f, ax = plt.subplots()

            for i in range(start.shape[0]):
                ax.plot(np.array([start[i ,0], end[i, 0]]), np.array([start[i, 1], end[i, 1]]), c="black")

            # also include a longer example trajectory
            n_timesteps = 500
            state = np.random.uniform(-3., 3., size=(1, 2))
            control = np.zeros((n_timesteps, 2))
            timevals = np.linspace(0, 10, n_timesteps)
            trajectory, _ = sys.predict(X=state, U=control, time_values=timevals)
            ax.plot(trajectory["x1"].to_numpy(), trajectory["x2"].to_numpy())
            plt.show()

    def test_vanderpol02(self, plot=True):
        # a single longer time series
        n_timesteps = 500
        state = np.random.uniform(-3., 3., size=(1, 2))
        control = np.zeros((n_timesteps, 2))

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

        df = pd.DataFrame(np.zeros((n_timesteps, 2)), index=timevals, columns=["u1", "u2"])

        control = TSCDataFrame.from_frame_list([df, df, df])
        X_predict, U = VanDerPol(eps=1).predict(X=state, U=control, time_values=timevals)

        if plot:
            f, ax = plt.subplots()
            for i in X_predict.ids:
                trajectory = X_predict.loc[[i], :]
                ax.plot(trajectory["x1"].to_numpy(), trajectory["x2"].to_numpy())
            plt.show()


    def test_duffing01(self, plot=False):

        X1, U1 = Duffing1D().predict(np.array([1, 2]), U=np.zeros((1000, 1)))
        X2, U2 = Duffing1D().predict(np.array([1, 2]), U=np.zeros((1000, 1)), time_values=np.arange(0, 10, 0.01))

        U = TSCDataFrame.from_array(np.zeros((1000, 1)), time_values=np.arange(0, 10, 0.01))
        X3, U3 = Duffing1D().predict(np.array([1, 2]), U=U, time_values=np.arange(0, 10, 0.01))
        X4, U4 = Duffing1D().predict(np.array([1, 2]), U=U)


        pdtest.assert_frame_equal(X1, X2)
        pdtest.assert_frame_equal(X1, X3)
        pdtest.assert_frame_equal(X1, X4)

        f, ax = plt.subplots()
        ax.plot(X1["x1"].to_numpy(), X2["x2"].to_numpy())
        plt.show()


