import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest

from datafold.pcfold import TSCDataFrame
from datafold.pcfold.timeseries.accessor import TSCAccessor
from datafold.pcfold.timeseries.collection import TSCException


class TestTSCAccessor(unittest.TestCase):
    rng = np.random.default_rng(5)

    def setUp(self) -> None:
        # The last two elements are used
        idx = pd.MultiIndex.from_arrays(
            [[0, 0, 1, 1, 15, 15, 45, 45, 45], [0, 1, 0, 1, 0, 1, 17, 18, 19]]
        )
        col = ["A", "B"]
        self.simple_df = pd.DataFrame(self.rng.random((9, 2)), index=idx, columns=col)

    def test_normalize_time1(self):
        # NOTE: more tests are included in test_tsc_data_structure/test_is_normalize_time()
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

    def test_shift_time_per_time_series01(self):
        simple_df = TSCDataFrame(self.simple_df)

        actual, shift_values = simple_df.tsc.shift_time_per_time_series(
            return_shift_values=True
        )

        self.assertIsInstance(actual, TSCDataFrame)
        self.assertIsInstance(shift_values, pd.Series)

        self.assertTrue(
            np.all(actual.initial_states().index.get_level_values("time") == 0)
        )

        actual2 = actual.tsc.shift_time_per_time_series(shift_values=-1 * shift_values)

        pdtest.assert_frame_equal(simple_df, actual2)

    def test_shift_time_per_time_series02(self):
        simple_df = TSCDataFrame(self.simple_df)

        actual, shift_values = simple_df.tsc.shift_time_per_time_series(
            return_shift_values=True
        )

        with self.assertRaises(ValueError):
            actual.tsc.shift_time_per_time_series(shift_values.iloc[:2])

        with self.assertRaises(TypeError):
            actual.tsc.shift_time_per_time_series(shift_values.to_numpy())

    def test_iter_timevalue_window(self):
        tsc_df = TSCDataFrame.from_single_timeseries(
            pd.DataFrame(self.rng.random((10, 2)), columns=["A", "B"])
        )

        tsc_df2 = tsc_df.insert_ts(
            pd.DataFrame(self.rng.random((10, 2)), columns=["A", "B"])
        )

        # tests for one time series
        self.assertEqual(
            len(list(tsc_df.tsc.iter_timevalue_window(window_size=2, offset=2))), 5
        )
        self.assertEqual(
            len(list(tsc_df.tsc.iter_timevalue_window(window_size=5, offset=5))), 2
        )
        self.assertEqual(
            len(list(tsc_df.tsc.iter_timevalue_window(window_size=5, offset=1))), 6
        )

        # the same must be true if there are two time series present
        self.assertEqual(
            len(list(tsc_df2.tsc.iter_timevalue_window(window_size=2, offset=2))), 5
        )
        self.assertEqual(
            len(list(tsc_df2.tsc.iter_timevalue_window(window_size=5, offset=5))), 2
        )
        self.assertEqual(
            len(list(tsc_df2.tsc.iter_timevalue_window(window_size=5, offset=1))), 6
        )

        # if per_time_series, the list becomes double as long
        self.assertEqual(
            len(
                list(
                    tsc_df2.tsc.iter_timevalue_window(
                        window_size=2, offset=2, per_time_series=True
                    )
                )
            ),
            10,
        )
        self.assertEqual(
            len(
                list(
                    tsc_df2.tsc.iter_timevalue_window(
                        window_size=5, offset=5, per_time_series=True
                    )
                )
            ),
            4,
        )
        self.assertEqual(
            len(
                list(
                    tsc_df2.tsc.iter_timevalue_window(
                        window_size=5, offset=1, per_time_series=True
                    )
                )
            ),
            12,
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

        # new case: the index 0 is not included in train or test
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

    def test_assign_ids_train_test3(self):
        df = pd.DataFrame(np.arange(20).reshape(10, 2), columns=("A", "B"))
        X = TSCDataFrame.from_single_timeseries(df)

        train_indices = np.arange(5)
        test_indices = np.arange(5, 10)

        # success
        X.tsc.assign_ids_train_test(train_indices, test_indices)

        with self.assertRaises(ValueError):
            train_indices_invalid = np.arange(5).reshape(2)
            X.tsc.assign_ids_train_test(train_indices_invalid, test_indices)

        with self.assertRaises(ValueError):
            train_indices_invalid = train_indices.copy()
            train_indices_invalid[0] = -1
            X.tsc.assign_ids_train_test(train_indices_invalid, test_indices)

        with self.assertRaises(ValueError):
            train_indices_invalid = train_indices.astype(float).copy()
            X.tsc.assign_ids_train_test(train_indices_invalid, test_indices)

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

    def test_check_equal_delta_time(self):
        tsc = TSCDataFrame(self.simple_df)
        dt1, dt2 = TSCAccessor.check_equal_delta_time(tsc, tsc, require_const=True)

        self.assertEqual(dt1, dt2)

        # insert new time series that now has a different time sampling
        tsc = tsc.insert_ts(
            pd.DataFrame(1, index=np.linspace(0, 1, 5), columns=tsc.columns)
        )

        with self.assertRaises(TSCException):
            TSCAccessor.check_equal_delta_time(tsc, tsc, require_const=True)

        dt1, dt2 = TSCAccessor.check_equal_delta_time(tsc, tsc, require_const=False)
        self.assertIsInstance(dt1, pd.Series)
        self.assertIsInstance(dt2, pd.Series)
        self.assertTrue(0.25 in dt1.to_numpy() and 0.25 in dt2.to_numpy())
        pdtest.assert_series_equal(dt1, dt2)

    def test_shift_matrices01(self):
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

    def test_shift_matrices02(self):
        # all time series are snapshot pairs (length 2)

        simple_df = self.simple_df.copy()
        simple_df = simple_df.drop(labels=[45])

        tsc_df = TSCDataFrame(simple_df)

        actual_left, actual_right = tsc_df.tsc.shift_matrices()

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

    def test_shift_matrices03(self):
        # single time series

        data = np.arange(8).reshape([4, 2])
        tsc_df = TSCDataFrame.from_array(data)

        actual_left, actual_right = tsc_df.tsc.shift_matrices(
            snapshot_orientation="row"
        )
        actual_left_T, actual_right_T = tsc_df.tsc.shift_matrices(
            snapshot_orientation="col"
        )

        expected_left = data[:-1, :]
        expected_left_T = data[:-1, :].T
        expected_right = data[1:, :]
        expected_right_T = data[1:, :].T

        nptest.assert_equal(actual_left, expected_left)
        nptest.assert_equal(actual_left_T, expected_left_T)

        nptest.assert_equal(actual_right, expected_right)
        nptest.assert_equal(actual_right_T, expected_right_T)

    def test_shift_matrices04(self):
        # test validation
        data = np.arange(8).reshape([4, 2])
        # uneven time delta
        tsc_df_okay = TSCDataFrame.from_array(data)
        tsc_df_uneven = TSCDataFrame.from_array(data, time_values=[0, 0.1, 0.3, 0.5])
        tsc_df_single = TSCDataFrame.from_array(data[0, :])

        with self.assertRaises(ValueError):
            tsc_df_okay.tsc.shift_matrices(snapshot_orientation="invalid")

        with self.assertRaises(TSCException):
            tsc_df_uneven.tsc.shift_matrices()

        with self.assertRaises(TSCException):
            tsc_df_single.tsc.shift_matrices()

    def test_shift_matrices05(self):
        # test validation
        data = pd.DataFrame(np.arange(6).reshape([3, 2]))
        tsc_df = TSCDataFrame.from_frame_list([data, data])

        actual_left, actual_right = tsc_df.tsc.shift_matrices(
            snapshot_orientation="row"
        )
        actual_left_T, actual_right_T = tsc_df.tsc.shift_matrices(
            snapshot_orientation="col"
        )

        expected_left = tsc_df.iloc[[0, 1, 3, 4], :].to_numpy()
        expected_right = tsc_df.iloc[[1, 2, 4, 5], :].to_numpy()

        nptest.assert_equal(actual_left, expected_left)
        nptest.assert_equal(actual_right, expected_right)
        nptest.assert_equal(actual_left_T, expected_left.T)
        nptest.assert_equal(actual_right_T, expected_right.T)

    def test_shift_time1(self):
        tsc_df = TSCDataFrame(self.simple_df).copy()

        tsc_df.tsc.shift_time_by_delta(5)
        nptest.assert_array_equal(
            tsc_df.index.get_level_values(1),
            self.simple_df.index.get_level_values(1) + 5,
        )

    def test_shift_time2(self):
        tsc_df = TSCDataFrame(self.simple_df)

        with self.assertRaises(AttributeError):
            # time is not allowed to be negative
            tsc_df.tsc.shift_time_by_delta(-5)

    def test_drop_last_n_samples(self):
        tsc_df = TSCDataFrame(self.simple_df)

        actual = tsc_df.tsc.drop_last_n_samples(1)
        expected = tsc_df.iloc[[0, 2, 4, 6, 7], :]

        pdtest.assert_frame_equal(actual, expected)

        with self.assertRaises(TSCException):
            tsc_df.tsc.drop_last_n_samples(999)

    def test_timederivative(self):
        data = pd.DataFrame(np.arange(20).reshape(10, 2), columns=["A", "B"])
        tsc_df = TSCDataFrame.from_single_timeseries(data)

        expected = np.ones((8, 2)) * 2
        actual = tsc_df.tsc.time_derivative(scheme="center", diff_order=1, accuracy=2)
        nptest.assert_equal(actual, expected)

        expected = np.zeros((8, 2))
        actual = tsc_df.tsc.time_derivative(scheme="center", diff_order=2, accuracy=2)
        nptest.assert_equal(actual, expected)

        expected = np.ones((8, 2)) * 2
        actual = tsc_df.tsc.time_derivative(scheme="backward", diff_order=1, accuracy=2)
        nptest.assert_equal(actual, expected)

        expected = np.zeros((7, 2))
        actual = tsc_df.tsc.time_derivative(scheme="backward", diff_order=2, accuracy=2)
        nptest.assert_allclose(actual, expected, rtol=1e-15, atol=1e-14)

        expected = np.ones((8, 2)) * 2
        actual = tsc_df.tsc.time_derivative(scheme="forward", diff_order=1, accuracy=2)
        nptest.assert_equal(actual, expected)

        expected = np.zeros((7, 2))
        actual = tsc_df.tsc.time_derivative(scheme="forward", diff_order=2, accuracy=2)
        nptest.assert_allclose(actual, expected, rtol=1e-15, atol=1e-14)

    def test_timederivative_index_center(self):
        data = pd.DataFrame(np.arange(20).reshape(10, 2), columns=["A", "B"])
        tsc_df = TSCDataFrame.from_single_timeseries(data)

        expected = np.arange(2, 10)
        actual = tsc_df.tsc.time_derivative(
            scheme="center", diff_order=1, accuracy=2, shift_index=True
        ).time_values()

        nptest.assert_equal(expected, actual)

        expected = np.arange(1, 9)
        actual = tsc_df.tsc.time_derivative(
            scheme="center", diff_order=1, accuracy=2, shift_index=False
        ).time_values()

        nptest.assert_equal(expected, actual)

    def test_timederivative_index_backward(self):
        data = pd.DataFrame(np.arange(20).reshape(10, 2), columns=["A", "B"])
        tsc_df = TSCDataFrame.from_single_timeseries(data)

        expected = np.arange(2, 10)
        actual = tsc_df.tsc.time_derivative(
            scheme="backward", diff_order=1, accuracy=2, shift_index=True
        ).time_values()

        nptest.assert_equal(expected, actual)

        expected = np.arange(2, 10)  # NOTE: is the same than with shift = True
        actual = tsc_df.tsc.time_derivative(
            scheme="backward", diff_order=1, accuracy=2, shift_index=False
        ).time_values()

        nptest.assert_equal(expected, actual)

    def test_timederivative_index_forward(self):
        data = pd.DataFrame(np.arange(20).reshape(10, 2), columns=["A", "B"])
        tsc_df = TSCDataFrame.from_single_timeseries(data)

        # shifted, even though the first derivative is  available at time 0
        expected = np.arange(2, 10)
        actual = tsc_df.tsc.time_derivative(
            scheme="forward", diff_order=1, accuracy=2, shift_index=True
        ).time_values()

        nptest.assert_equal(expected, actual)

        expected = np.arange(0, 8)
        actual = tsc_df.tsc.time_derivative(
            scheme="forward", diff_order=1, accuracy=2, shift_index=False
        ).time_values()

        nptest.assert_equal(expected, actual)

    def test_fill_timeseries_with_last_state01(self):
        tscdf = TSCDataFrame(self.simple_df).loc[[0], :]

        n_timesteps = 5
        actual = tscdf.tsc.fill_timeseries_with_last_state(n_timesteps=n_timesteps)

        self.assertEqual(actual.delta_time, tscdf.delta_time)
        self.assertEqual(actual.n_timesteps, n_timesteps)
        self.assertTrue(
            np.all(actual.iloc[1:, :].to_numpy() == tscdf.iloc[[-1], :].to_numpy())
        )

    def test_fill_timeseries_with_last_state02(self):
        tscdf = TSCDataFrame(self.simple_df).loc[[0], :]

        with self.assertRaises(TSCException):
            tscdf.tsc.fill_timeseries_with_last_state(n_timesteps=1)

        with self.assertRaises(TSCException):
            tscdf.tsc.fill_timeseries_with_last_state(n_timesteps=2)

    def test_augment_control_input(self):
        rng = np.random.default_rng(1)

        tscdf = TSCDataFrame.from_array(
            rng.uniform(size=(3, 2)), feature_names=["x1", "x2"]
        )
        control = TSCDataFrame.from_array(
            rng.uniform(size=(2, 2)), feature_names=["u1", "u2"]
        )

        expect = np.ones([3, 4]) * np.nan
        expect[:, :2] = tscdf.to_numpy()
        expect[1:, 2:] = control.to_numpy()
        expect = TSCDataFrame.from_array(expect, feature_names=["x1", "x2", "u1", "u2"])

        actual = tscdf.tsc.augment_control_input(U=control)

        pdtest.assert_frame_equal(expect, actual)
