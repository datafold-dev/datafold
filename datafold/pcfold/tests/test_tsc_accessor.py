import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error

from datafold.pcfold.timeseries import TSCDataFrame
from datafold.pcfold.timeseries.accessor import (
    TakensEmbedding,
    TimeSeriesError,
    NormalizeQoi,
)


class TestTscAccessor(unittest.TestCase):
    def setUp(self) -> None:
        # The last two elements are used
        idx = pd.MultiIndex.from_arrays(
            [[0, 0, 1, 1, 15, 15, 45, 45, 45], [0, 1, 0, 1, 0, 1, 17, 18, 19]]
        )
        col = ["A", "B"]
        self.simple_df = pd.DataFrame(np.random.rand(9, 2), index=idx, columns=col)

        # Requires non-random values
        self.takens_df = pd.DataFrame(
            np.arange(18).reshape([9, 2]), index=idx, columns=col
        )

    def test_normalize_time1(self):
        # NOTE: more tests are included in test_tsc_data_structre/test_is_normalize_time()
        to_convert = TSCDataFrame(self.simple_df)

        actual = to_convert.tsc.normalize_time()
        expected = to_convert

        pdtest.assert_frame_equal(actual, expected)

    def test_normalize_time2(self):
        simple_df = self.simple_df.copy()
        simple_df.index = pd.MultiIndex.from_arrays(
            [
                simple_df.index.get_level_values(0),
                simple_df.index.get_level_values(1) + 1,
            ]
        )

        to_convert = TSCDataFrame(simple_df)

        actual = to_convert.tsc.normalize_time()
        expected = TSCDataFrame(self.simple_df)  # undo the +1

        pdtest.assert_frame_equal(actual, expected)

    def test_normalize_time3(self):
        simple_df = self.simple_df.copy()
        simple_df.index = pd.MultiIndex.from_arrays(
            [
                simple_df.index.get_level_values(0),
                simple_df.index.get_level_values(1) + 0.5,
            ]
        )  # shift by float

        to_convert = TSCDataFrame(simple_df)

        actual = to_convert.tsc.normalize_time()
        expected = TSCDataFrame(self.simple_df)  # undo the +1

        pdtest.assert_frame_equal(actual, expected)

    def test_shift_matrices(self):
        # TODO: potentially do more tests (esp. with uneven number of trajectories,
        #  this  is a quite important functionality!)

        tc = TSCDataFrame(self.simple_df)
        actual_left, actual_right = tc.tsc.shift_matrices()

        original_values = self.simple_df.to_numpy()

        expected_left = np.zeros([2, 5])
        expected_left[:, 0] = original_values[0, :]
        expected_left[:, 1] = original_values[2, :]
        expected_left[:, 2] = original_values[4, :]
        expected_left[:, 3] = original_values[6, :]
        expected_left[:, 4] = original_values[7, :]

        expected_right = np.zeros_like(expected_left)
        expected_right[:, 0] = original_values[1, :]
        expected_right[:, 1] = original_values[3, :]
        expected_right[:, 2] = original_values[5, :]
        expected_right[:, 3] = original_values[7, :]
        expected_right[:, 4] = original_values[8, :]

        nptest.assert_equal(actual_left, expected_left)
        nptest.assert_equal(actual_right, expected_right)

        actual_left, actual_right = tc.tsc.shift_matrices(snapshot_orientation="row")

        nptest.assert_equal(actual_left, expected_left.T)
        nptest.assert_equal(actual_right, expected_right.T)

    def test_shift_matrices2(self):
        simple_df = self.simple_df.copy()
        simple_df = simple_df.drop(labels=[45])

        tc = TSCDataFrame(simple_df)

        actual_left, actual_right = tc.tsc.shift_matrices()

        original_values = simple_df.to_numpy()

        expected_left = np.zeros([2, 3])
        expected_left[:, 0] = original_values[0, :]
        expected_left[:, 1] = original_values[2, :]
        expected_left[:, 2] = original_values[4, :]

        expected_right = np.zeros_like(expected_left)
        expected_right[:, 0] = original_values[1, :]
        expected_right[:, 1] = original_values[3, :]
        expected_right[:, 2] = original_values[5, :]

        nptest.assert_equal(actual_left, expected_left)
        nptest.assert_equal(actual_right, expected_right)

    def test_takens_embedding(self):
        simple_df = self.takens_df.drop("B", axis=1)

        tc = TSCDataFrame(simple_df)

        # using class
        actual1 = TakensEmbedding(
            lag=0, delays=1, frequency=1, time_direction="backward"
        ).apply(tc)
        self.assertTrue(isinstance(actual1, TSCDataFrame))

        actual1 = actual1.values  # only compare the numeric values now

        # using function wrapper
        actual2 = tc.tsc.takens_embedding(
            lag=0, delays=1, frequency=1, time_direction="backward"
        ).values

        expected = np.array(
            [
                [0.0, np.nan],
                [2.0, 0.0],
                [4.0, np.nan],
                [6.0, 4.0],
                [8.0, np.nan],
                [10.0, 8.0],
                [12.0, np.nan],
                [14.0, 12.0],
                [16.0, 14.0],
            ]
        )

        nptest.assert_equal(actual1, expected)
        nptest.assert_equal(actual2, expected)

    def test_takens_delay_indices(self):
        nptest.assert_array_equal(
            TakensEmbedding(lag=0, delays=1, frequency=1).delay_indices, np.array([1])
        )

        nptest.assert_array_equal(
            TakensEmbedding(lag=0, delays=2, frequency=1).delay_indices,
            np.array([1, 2]),
        )

        nptest.assert_array_equal(
            TakensEmbedding(lag=0, delays=5, frequency=1).delay_indices,
            np.array([1, 2, 3, 4, 5]),
        )

        nptest.assert_array_equal(
            TakensEmbedding(lag=1, delays=1, frequency=1).delay_indices, np.array([2]),
        )

        nptest.assert_array_equal(
            TakensEmbedding(lag=1, delays=5, frequency=1).delay_indices,
            np.array([2, 3, 4, 5, 6]),
        )

        nptest.assert_array_equal(
            TakensEmbedding(lag=2, delays=2, frequency=2).delay_indices,
            np.array([3, 5]),
        )

        nptest.assert_array_equal(
            TakensEmbedding(lag=2, delays=4, frequency=2).delay_indices,
            np.array([3, 5, 7, 9]),
        )

        with self.assertRaises(ValueError):
            TakensEmbedding(lag=0, delays=1, frequency=2)

    def test_shift_time1(self):
        tsc_df = TSCDataFrame(self.simple_df)

        tsc_df.tsc.shift_time(5)
        nptest.assert_array_equal(
            tsc_df.index.get_level_values(1),
            self.simple_df.index.get_level_values(1) + 5,
        )

    def test_shift_time2(self):
        tsc_df = TSCDataFrame(self.simple_df)

        with self.assertRaises(AttributeError):
            # time is not allowed to be negative
            tsc_df.tsc.shift_time(-5)

    def test_normalize_id(self):
        tsc_df = TSCDataFrame(self.simple_df)

        norm_tsc, norm_info = tsc_df.tsc.normalize_qoi(normalize_strategy="id")
        pdtest.assert_frame_equal(tsc_df, norm_tsc)

    def test_normalize_min_max(self):
        tsc_df = TSCDataFrame(self.simple_df)

        norm_tsc, norm_info = tsc_df.tsc.normalize_qoi(normalize_strategy="min-max")

        # sanity check:
        nptest.assert_array_equal(norm_tsc.min().to_numpy(), np.zeros(2))
        nptest.assert_array_equal(norm_tsc.max().to_numpy(), np.ones(2))

        # Undoing normalization must give original TSCDataFrame back
        pdtest.assert_frame_equal(tsc_df, norm_tsc.tsc.undo_normalize_qoi(norm_info)[0])

    def test_normalize_mean(self):
        tsc_df = TSCDataFrame(self.simple_df)

        norm_tsc, norm_info = tsc_df.tsc.normalize_qoi(normalize_strategy="mean")

        # sanity check:
        nptest.assert_array_equal(tsc_df.mean(), norm_info["mean"])
        nptest.assert_array_equal(tsc_df.min(), norm_info["min"])
        nptest.assert_array_equal(tsc_df.max(), norm_info["max"])

        # Undoing normalization must give original TSCDataFrame back
        pdtest.assert_frame_equal(tsc_df, norm_tsc.tsc.undo_normalize_qoi(norm_info)[0])

    def test_normalize_standard(self):
        tsc_df = TSCDataFrame(self.simple_df)

        norm_tsc, norm_info = tsc_df.tsc.normalize_qoi(normalize_strategy="standard")

        # sanity check:
        nptest.assert_array_equal(tsc_df.mean(), norm_info["mean"])
        nptest.assert_array_equal(tsc_df.std(), norm_info["std"])

        # Undoing normalization must give original TSCDataFrame back
        pdtest.assert_frame_equal(tsc_df, norm_tsc.tsc.undo_normalize_qoi(norm_info)[0])

    def test_normalize_standard_on_second_tsc(self):
        tsc_df1 = TSCDataFrame(self.simple_df)

        tsc_df2 = TSCDataFrame(
            np.random.rand(*tsc_df1.to_numpy().shape),
            index=tsc_df1.index,
            columns=tsc_df1.columns,
        )

        norm_tsc1, norm_info1 = tsc_df1.tsc.normalize_qoi(normalize_strategy="standard")
        norm_tsc2, norm_info2 = tsc_df2.tsc.normalize_qoi(normalize_strategy=norm_info1)

        # sanity check:
        pdtest.assert_frame_equal(
            norm_tsc2, (tsc_df2 - norm_info1["mean"]) / norm_info1["std"]
        )

        # Undoing normalization must give original TSCDataFrame back
        pdtest.assert_frame_equal(
            tsc_df2, norm_tsc2.tsc.undo_normalize_qoi(norm_info2)[0]
        )


class TestErrorTimeSeries(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self._create_tsc_one()
        self._create_tsc_two()

    def _create_multi_qoi_ts(self, nr_timesteps=10):
        time_index = np.arange(nr_timesteps)
        columns = np.array(["qoi_A", "qoi_B", "qoi_C"])
        data = np.random.rand(time_index.shape[0], columns.shape[0])

        return pd.DataFrame(data, time_index, columns)

    def _create_tsc_one(self):
        self.tsc_one_left = TSCDataFrame.from_single_timeseries(
            df=self._create_multi_qoi_ts(10)
        )
        self.tsc_one_left = self.tsc_one_left.insert_ts(self._create_multi_qoi_ts(10))
        self.tsc_one_left = self.tsc_one_left.insert_ts(self._create_multi_qoi_ts(10))

        self.tsc_one_right = TSCDataFrame.from_same_indices_as(
            self.tsc_one_left, values=np.random.rand(*self.tsc_one_left.shape)
        )

    def _create_tsc_two(self):
        self.tsc_two_left = TSCDataFrame.from_single_timeseries(
            df=self._create_multi_qoi_ts(10)
        )

        # NOTE: here they have different length!
        self.tsc_two_left = self.tsc_two_left.insert_ts(self._create_multi_qoi_ts(5))
        self.tsc_two_left = self.tsc_two_left.insert_ts(self._create_multi_qoi_ts(20))

        self.tsc_two_right = TSCDataFrame.from_same_indices_as(
            self.tsc_two_left, values=np.random.rand(*self.tsc_two_left.shape)
        )

    def test_metrics_without_error(self):
        for metric in TimeSeriesError.VALID_METRIC:
            for mode in TimeSeriesError.VALID_MODES:
                for normalize_strategy in NormalizeQoi.VALID_STRATEGIES:
                    for multi_qoi in ["uniform_average", "raw_values"]:

                        err_obj = TimeSeriesError(
                            metric=metric,
                            mode=mode,
                            normalize_strategy=normalize_strategy,
                        )

                        if metric != "max":  # max does not support multi-output
                            err_obj.score(
                                self.tsc_one_left,
                                self.tsc_one_right,
                                multi_qoi=multi_qoi,
                            )
                            err_obj.score(
                                self.tsc_two_left,
                                self.tsc_two_right,
                                multi_qoi=multi_qoi,
                            )
                        else:
                            with self.assertRaises(ValueError):
                                err_obj.score(
                                    self.tsc_one_left,
                                    self.tsc_one_right,
                                    multi_qoi=multi_qoi,
                                )
                                err_obj.score(
                                    self.tsc_two_left,
                                    self.tsc_two_right,
                                    multi_qoi=multi_qoi,
                                )

        # Test to not fail

    def test_error_per_timeseries1(self):
        multi_output = "uniform_average"
        actual = TimeSeriesError(metric="mse", mode="timeseries").score(
            self.tsc_one_left, self.tsc_one_right, multi_qoi=multi_output
        )

        self.assertIsInstance(actual, pd.Series)

        idx = pd.IndexSlice

        for id_ in self.tsc_one_left.ids:
            expected_val = mean_squared_error(
                self.tsc_one_left.loc[idx[id_, :], :],
                self.tsc_one_right.loc[idx[id_, :], :],
                sample_weight=None,
                multioutput="uniform_average",
            )

            self.assertEqual(expected_val, actual.loc[id_])

    def test_error_per_timeseries2(self):
        multi_output = "raw_values"

        actual = TimeSeriesError(metric="mse", mode="timeseries").score(
            self.tsc_one_left, self.tsc_one_right, multi_qoi=multi_output
        )

        self.assertIsInstance(actual, pd.DataFrame)

        idx = pd.IndexSlice

        for id_ in self.tsc_one_left.ids:
            expected_val = mean_squared_error(
                self.tsc_one_left.loc[idx[id_, :], :],
                self.tsc_one_right.loc[idx[id_, :], :],
                sample_weight=None,
                multioutput=multi_output,
            )

            nptest.assert_array_equal(expected_val, actual.loc[id_].to_numpy())

    def test_error_per_timeseries3(self):
        # With different length TSC

        multi_output = "raw_values"

        actual = TimeSeriesError(metric="mse", mode="timeseries").score(
            self.tsc_two_left, self.tsc_two_right, multi_qoi=multi_output
        )

        self.assertIsInstance(actual, pd.DataFrame)

        idx = pd.IndexSlice

        for id_ in self.tsc_two_left.ids:
            expected_val = mean_squared_error(
                self.tsc_two_left.loc[idx[id_, :], :],
                self.tsc_two_right.loc[idx[id_, :], :],
                sample_weight=None,
                multioutput=multi_output,
            )

            nptest.assert_array_equal(expected_val, actual.loc[id_].to_numpy())

    def test_error_per_qoi1(self):
        sample_weight = np.ones(self.tsc_one_left.shape[0])
        actual = TimeSeriesError(metric="mse", mode="qoi").score(
            self.tsc_one_left, self.tsc_one_right, sample_weight=sample_weight
        )

        self.assertIsInstance(actual, pd.Series)

        nptest.assert_array_equal(
            mean_squared_error(
                self.tsc_one_left, self.tsc_one_right, multioutput="raw_values"
            ),
            actual.to_numpy(),
        )

    def test_error_per_qoi2(self):
        sample_weight = np.zeros(self.tsc_one_left.shape[0])
        sample_weight[0] = 1  # put whole weight on a single sample

        actual = TimeSeriesError(metric="mse", mode="qoi").score(
            self.tsc_one_left, self.tsc_one_right, sample_weight=sample_weight
        )

        self.assertIsInstance(actual, pd.Series)

        nptest.assert_array_equal(
            mean_squared_error(
                self.tsc_one_left,
                self.tsc_one_right,
                sample_weight=sample_weight,
                multioutput="raw_values",
            ),
            actual.to_numpy(),
        )

    def test_error_per_qoi3(self):
        sample_weight = np.ones(self.tsc_two_left.shape[0])
        actual = TimeSeriesError(metric="mse", mode="qoi").score(
            self.tsc_two_left, self.tsc_two_right, sample_weight=sample_weight
        )

        self.assertIsInstance(actual, pd.Series)

        nptest.assert_array_equal(
            mean_squared_error(
                self.tsc_two_left, self.tsc_two_right, multioutput="raw_values"
            ),
            actual.to_numpy(),
        )

    def test_error_per_timestep1(self):
        multi_qoi = "uniform_average"
        actual = TimeSeriesError(metric="mse", mode="timestep").score(
            self.tsc_one_left, self.tsc_one_right, multi_qoi=multi_qoi
        )

        self.assertIsInstance(actual, pd.Series)

        idx_slice = pd.IndexSlice
        for t in self.tsc_one_left.time_indices(unique_values=True):

            nptest.assert_array_equal(
                mean_squared_error(
                    self.tsc_one_left.loc[idx_slice[:, t], :],
                    self.tsc_one_right.loc[idx_slice[:, t], :],
                    sample_weight=None,
                    multioutput=multi_qoi,
                ),
                actual.loc[t],
            )

    def test_error_per_timestep2(self):
        multi_qoi = np.array([0.5, 0.5, 1])  # user defined weighing
        sample_weight = np.arange(
            len(self.tsc_one_left.ids)
        )  # increasing weight for each time series (three)

        actual = TimeSeriesError(metric="mse", mode="timestep").score(
            self.tsc_one_left,
            self.tsc_one_right,
            sample_weight=sample_weight,
            multi_qoi=multi_qoi,
        )

        self.assertIsInstance(actual, pd.Series)

        idx_slice = pd.IndexSlice
        for t in self.tsc_one_left.time_indices(unique_values=True):

            nptest.assert_array_equal(
                mean_squared_error(
                    self.tsc_one_left.loc[idx_slice[:, t], :],
                    self.tsc_one_right.loc[idx_slice[:, t], :],
                    sample_weight=sample_weight,
                    multioutput=multi_qoi,
                ),
                actual.loc[t],
            )

    def test_error_per_timestep3(self):
        # For tsc_two

        multi_qoi = "uniform_average"
        actual = TimeSeriesError(metric="mse", mode="timestep").score(
            self.tsc_two_left, self.tsc_two_right, multi_qoi=multi_qoi
        )

        self.assertIsInstance(actual, pd.Series)

        idx_slice = pd.IndexSlice
        for t in self.tsc_two_left.time_indices(unique_values=True):

            nptest.assert_array_equal(
                mean_squared_error(
                    self.tsc_two_left.loc[idx_slice[:, t], :],
                    self.tsc_two_right.loc[idx_slice[:, t], :],
                    sample_weight=None,
                    multioutput=multi_qoi,
                ),
                actual.loc[t],
            )


if __name__ == "__main__":
    # test = TestErrorTimeSeries()
    # test.setUp()
    # test.test_error_per_timestep3()
    # exit()

    unittest.main()
