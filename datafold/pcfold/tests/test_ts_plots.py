
import unittest

import numpy as np
import pandas as pd

from trajds import TSCDataFrame


class TestTimeSeriesCollectionPlots(unittest.TestCase):

    """NOTE: plotting tests only check if the methods run without error. A proper unit-testing is too difficult.
    To look at a plot for an example, simply run the specific test method and call plt.show() after the
    respective test. """

    def test_plot_density2d(self):
        number_ts = 100

        ids = np.array(np.arange(number_ts)).repeat(number_ts, axis=0)
        time = np.tile(np.arange(number_ts), number_ts)

        idx = pd.MultiIndex.from_arrays([ids, time])
        ts = TSCDataFrame(np.random.rand(ids.shape[0], 2), index=idx, columns=["x", "y"])

        ts.tsc.plot_density2d(0, 100, 100, np.eye(2) * 0.005)

        # Check for error
        ts.loc[:, "third_column"] = 0  # density plot only works for 2D data

        with self.assertRaises(ValueError):
            ts.tsc.plot_density2d(0, 100, 100, np.eye(2) * 0.005)

        # plt.show()


if __name__ == "__main__":
    TestTimeSeriesCollectionPlots().test_plot_density2d()
