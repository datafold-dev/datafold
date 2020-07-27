import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest

from datafold.pcfold import TSCDataFrame


class TestTscAccessor(unittest.TestCase):
    def setUp(self) -> None:
        # The last two elements are used
        idx = pd.MultiIndex.from_arrays(
            [[0, 0, 1, 1, 15, 15, 45, 45, 45], [0, 1, 0, 1, 0, 1, 17, 18, 19]]
        )
        col = ["A", "B"]
        self.simple_df = pd.DataFrame(np.random.rand(9, 2), index=idx, columns=col)

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

    def test_iter_timevalue_window(self):
        tsc_df = TSCDataFrame.from_single_timeseries(
            pd.DataFrame(np.random.rand(10, 2), columns=["A", "B"])
        )

        tsc_df2 = tsc_df.insert_ts(
            pd.DataFrame(np.random.rand(10, 2), columns=["A", "B"])
        )

        # tests for one time series
        self.assertEqual(
            len(list(tsc_df.tsc.iter_timevalue_window(blocksize=2, offset=2))), 5
        )
        self.assertEqual(
            len(list(tsc_df.tsc.iter_timevalue_window(blocksize=5, offset=5))), 2
        )
        self.assertEqual(
            len(list(tsc_df.tsc.iter_timevalue_window(blocksize=5, offset=1))), 6
        )

        # the same must be true if there are two time series present
        self.assertEqual(
            len(list(tsc_df2.tsc.iter_timevalue_window(blocksize=2, offset=2))), 5
        )
        self.assertEqual(
            len(list(tsc_df2.tsc.iter_timevalue_window(blocksize=5, offset=5))), 2
        )
        self.assertEqual(
            len(list(tsc_df2.tsc.iter_timevalue_window(blocksize=5, offset=1))), 6
        )

    def test_assign_ids_sequential(self):
        tsc_df = TSCDataFrame(self.simple_df)

        tsc_df.tsc.assign_ids_sequential()
        nptest.assert_array_equal(tsc_df.ids, np.arange(4))

        # makes sure that the correct type is returned
        tsc_df = tsc_df.tsc.assign_ids_sequential()
        self.assertIsInstance(tsc_df, TSCDataFrame)
        nptest.assert_array_equal(tsc_df.ids, np.arange(4))

    def test_assign_ids_train_test1(self):
        df = pd.DataFrame(np.arange(10).reshape(5, 2), columns=("A", "B"))

        train_indices = np.array([0, 1, 2])
        test_indices = np.array([3, 4])

        X = TSCDataFrame.from_single_timeseries(df)

        actual_train, actual_test = X.tsc.assign_ids_train_test(
            train_indices, test_indices
        )

        self.assertEqual(actual_train.n_timeseries, 1)
        self.assertEqual(actual_test.n_timeseries, 1)

        expected_train_timevalues = X.iloc[:3, :].time_values()
        expected_test_timevalues = X.iloc[3:, :].time_values()

        nptest.assert_array_equal(actual_train.time_values(), expected_train_timevalues)
        nptest.assert_array_equal(actual_test.time_values(), expected_test_timevalues)

        #### new case: the index 0 is not included in train or test
        train_indices = np.array([1, 2])
        test_indices = np.array([3, 4])

        actual_train, actual_test, actual_dropped = X.tsc.assign_ids_train_test(
            train_indices, test_indices, return_dropped=True
        )

        expected_train_timevalues = X.iloc[1:3, :].time_values()
        expected_test_timevalues = X.iloc[3:, :].time_values()
        expected_dropped_timevalues = np.array([0])

        nptest.assert_array_equal(actual_train.time_values(), expected_train_timevalues)
        nptest.assert_array_equal(actual_test.time_values(), expected_test_timevalues)
        nptest.assert_array_equal(
            actual_dropped.index.get_level_values(TSCDataFrame.tsc_time_idx_name),
            expected_dropped_timevalues,
        )

    def test_assign_ids_train_test2(self):
        df = pd.DataFrame(np.arange(20).reshape(10, 2), columns=("A", "B"))
        X = TSCDataFrame.from_single_timeseries(df)

        # the indices 2 and 7 are missing -> this must start a new time series ID
        train_indices = np.array([0, 1, 3, 4])
        test_indices = np.array([5, 6, 8, 9])

        actual_train, actual_test = X.tsc.assign_ids_train_test(
            train_indices=train_indices, test_indices=test_indices
        )

        # because of the dropped index "2" there must be two time series
        self.assertEqual(actual_train.n_timeseries, 2)
        self.assertEqual(actual_test.n_timeseries, 2)

        # the dropped indices are not part of the either of the time_values()
        self.assertFalse(2 in actual_train.time_values())
        self.assertFalse(7 in actual_test.time_values())

    def test_assign_ids_const_delta1(self):

        original_idx = pd.MultiIndex.from_arrays(
            [np.ones(8), np.hstack([np.arange(4), np.arange(10, 14)])]
        )
        data = np.arange(16).reshape((8, 2))

        base_tsc_df = TSCDataFrame(data, index=original_idx, columns=["A", "B"])

        expect_index = pd.MultiIndex.from_arrays(
            [
                np.hstack([np.zeros(4), np.ones(4)]),
                np.hstack([np.arange(4), np.arange(10, 14)]),
            ]
        )

        expect = base_tsc_df.copy().set_index(expect_index)
        actual = base_tsc_df.tsc.assign_ids_const_delta()

        # before: single time series with irregular time sampling
        # after: two time series with both the same uniform time sampling
        self.assertTrue(np.isnan(base_tsc_df.delta_time))
        self.assertEqual(actual.delta_time, 1)

        self.assertEqual(base_tsc_df.n_timeseries, 1)
        self.assertEqual(actual.n_timeseries, 2)

        pdtest.assert_frame_equal(expect, actual)

    def test_assign_ids_const_delta2(self):

        original_idx = pd.MultiIndex.from_arrays(
            [np.ones(6), np.hstack([np.arange(4), np.arange(10, 14, 2)])]
        )
        data = np.arange(12).reshape((6, 2))

        base_tsc_df = TSCDataFrame(data, index=original_idx, columns=["A", "B"])

        expect_index = pd.MultiIndex.from_arrays(
            [
                np.hstack([np.zeros(4), np.ones(2)]),
                np.hstack([np.arange(4), np.arange(10, 14, 2)]),
            ]
        )

        expect = base_tsc_df.copy().set_index(expect_index)
        actual = base_tsc_df.tsc.assign_ids_const_delta()

        pdtest.assert_frame_equal(expect, actual)
        nptest.assert_array_equal(actual.delta_time.to_numpy(), np.array([1, 2]))

    def test_assign_ids_const_delta3(self):

        original_idx = pd.MultiIndex.from_arrays(
            [np.ones(5), np.array([1, 7, 8, 9, 10])]
        )
        data = np.arange(10).reshape((5, 2))
        base_tsc_df = TSCDataFrame(data, index=original_idx, columns=["A", "B"])

        with self.assertRaises(ValueError):
            base_tsc_df.tsc.assign_ids_const_delta(drop_samples=False)

        actual = base_tsc_df.tsc.assign_ids_const_delta(drop_samples=True)

        expected_idx = pd.MultiIndex.from_arrays([np.zeros(4), np.array([7, 8, 9, 10])])
        expected = base_tsc_df.iloc[1:, :].set_index(expected_idx)

        self.assertEqual(actual.n_timeseries, 1)
        pdtest.assert_frame_equal(actual, expected)

    def test_assign_ids_const_delta4(self):

        # there is no time series possible with constant time sampling
        original_idx1 = pd.MultiIndex.from_arrays(
            [np.zeros(5), np.array([1, 5, 7, 14, 19])]
        )
        data = np.arange(10).reshape((5, 2))
        base_tsc_df = TSCDataFrame(data, index=original_idx1, columns=["A", "B"])

        with self.assertRaises(ValueError):
            # cannot assign any new ids in a completely irregular time series
            base_tsc_df.tsc.assign_ids_const_delta(drop_samples=False)

        actual = base_tsc_df.tsc.assign_ids_const_delta(drop_samples=True)

        # if time series are completely irregular and allowed to be dropped, then
        # in this case None is returned
        self.assertEqual(actual, None)

    def test_assign_ids_const_delta5(self):
        test_tsc = TSCDataFrame(self.simple_df)

        actual = test_tsc.copy().tsc.assign_ids_const_delta(drop_samples=False)
        expected = test_tsc.copy().tsc.assign_ids_sequential()

        pdtest.assert_frame_equal(actual, expected)

    def test_assign_ids_const_delta6(self):

        df = pd.DataFrame(
            np.arange(18).reshape(9, 2),
            index=[1, 2, 3, 5, 7, 9, 10, 11, 12],
            columns=["A", "B"],
        )

        tsc_df = TSCDataFrame.from_single_timeseries(df)
        actual = tsc_df.copy().tsc.assign_ids_const_delta(drop_samples=False)

        self.assertEqual(actual.n_timeseries, 3)
        # new time frequencies
        nptest.assert_array_equal(actual.delta_time.to_numpy(), np.array([1, 2, 1]))
        # no dropping of samples
        nptest.assert_array_equal(df.to_numpy(), actual.to_numpy())

    def test_assign_ids_const_delta7(self):

        tsc_df = TSCDataFrame(
            np.arange(6).reshape(3, 2),
            index=pd.MultiIndex.from_arrays([[0, 0, 1], [1, 2, 1]]),
            columns=["A", "B"],
        )

        actual = tsc_df.copy().tsc.assign_ids_const_delta(drop_samples=True)

        pdtest.assert_frame_equal(actual, tsc_df.drop(1, level=0))

        with self.assertRaises(ValueError):
            tsc_df.copy().tsc.assign_ids_const_delta(drop_samples=False)

    def test_shift_matrices(self):
        # TODO: potentially do more tests (esp. with uneven number of time series,
        #  this is a quite important functionality!)

        tc = TSCDataFrame(self.simple_df)
        actual_left, actual_right = tc.tsc.compute_shift_matrices()

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

        actual_left, actual_right = tc.tsc.compute_shift_matrices(
            snapshot_orientation="row"
        )

        nptest.assert_equal(actual_left, expected_left.T)
        nptest.assert_equal(actual_right, expected_right.T)

    def test_shift_matrices2(self):
        simple_df = self.simple_df.copy()
        simple_df = simple_df.drop(labels=[45])

        tc = TSCDataFrame(simple_df)

        actual_left, actual_right = tc.tsc.compute_shift_matrices()

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


if __name__ == "__main__":
    # test = TestErrorTimeSeries()
    # test.setUp()
    # test.test_error_per_timestep3()
    # exit()

    unittest.main()
