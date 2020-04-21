import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest

from datafold.pcfold.timeseries.collection import (
    InitialCondition,
    TSCDataFrame,
    TSCException,
)


class TestTSCDataFrame(unittest.TestCase):
    def setUp(self) -> None:
        # The last two elements are used
        idx = pd.MultiIndex.from_arrays(
            [[0, 0, 1, 1, 15, 15, 45, 45, 45], [0, 1, 0, 1, 0, 1, 17, 18, 19]]
        )
        col = ["A", "B"]
        self.simple_df = pd.DataFrame(np.random.rand(9, 2), index=idx, columns=col)

    def test_simple1(self):
        tc = TSCDataFrame(self.simple_df)
        pdtest.assert_frame_equal(tc.loc[0, :], self.simple_df.loc[0, :])
        pdtest.assert_frame_equal(tc.loc[1, :], self.simple_df.loc[1, :])

    def test_nr_timeseries(self):
        tc = TSCDataFrame(self.simple_df)
        self.assertEqual(tc.n_timeseries, 4)

    def test_n_feature(self):
        tc = TSCDataFrame(self.simple_df)
        self.assertEqual(tc.n_features, 2)

    def test_shape(self):
        tc = TSCDataFrame(self.simple_df)
        self.assertEqual(tc.shape, (9, 2))

    def test_nelements_timeseries(self):
        tc = TSCDataFrame(self.simple_df)
        pdtest.assert_series_equal(
            tc.n_timesteps,
            pd.Series(
                [2, 2, 2, 3],
                index=pd.Index([0, 1, 15, 45], name=TSCDataFrame.tsc_id_idx_name),
                name="counts",
            ),
        )

        simple_df = self.simple_df.copy()
        simple_df = simple_df.drop(
            labels=[45]
        )  # is the only one which has time series length 4
        actual = TSCDataFrame(simple_df).n_timesteps
        expected = 2

        self.assertEqual(actual, expected)

    def test_from_same_indices_as01(self):

        tc = TSCDataFrame(self.simple_df)
        matrix = self.simple_df.to_numpy()

        actual = TSCDataFrame.from_same_indices_as(tc, matrix)
        pdtest.assert_frame_equal(actual, tc)

    def test_from_same_indices_as02(self):
        tc = TSCDataFrame(self.simple_df)
        matrix = self.simple_df.to_numpy()

        actual = TSCDataFrame.from_same_indices_as(
            tc, matrix, except_columns=["qoi0", "qoi1"]
        )

        expected = self.simple_df.copy()
        expected.columns = ["qoi0", "qoi1"]
        expected = TSCDataFrame(expected)

        pdtest.assert_frame_equal(actual, expected)

    def test_from_same_indices_as03(self):
        tc = TSCDataFrame(self.simple_df)
        matrix = self.simple_df.to_numpy()

        new_index = pd.MultiIndex.from_arrays(
            [tc.index.get_level_values(0), tc.index.get_level_values(1) + 100]
        )  # simply add to time +100

        actual = TSCDataFrame.from_same_indices_as(tc, matrix, except_index=new_index)

        expected = self.simple_df.copy()
        expected.index = new_index
        expected = TSCDataFrame(expected)

        pdtest.assert_frame_equal(actual, expected)

    def test_from_same_indices_as04(self):
        tc = TSCDataFrame(self.simple_df)
        matrix = self.simple_df.to_numpy()

        new_index = tc.index

        # check for error
        with self.assertRaises(ValueError):
            TSCDataFrame.from_same_indices_as(
                tc, matrix, except_columns=pd.Index(["A", "B"]), except_index=new_index
            )

    def test_from_frame_list(self):

        frame_list = [self.simple_df.loc[i, :] for i in self.simple_df.index.levels[0]]

        actual = TSCDataFrame.from_frame_list(frame_list)
        expected = TSCDataFrame(self.simple_df)
        expected.index = pd.MultiIndex.from_arrays(
            [
                [0, 0, 1, 1, 2, 2, 3, 3, 3],
                expected.index.get_level_values(TSCDataFrame.tsc_time_idx_name),
            ],
            names=[TSCDataFrame.tsc_id_idx_name, TSCDataFrame.tsc_time_idx_name],
        )
        pdtest.assert_frame_equal(actual, expected)

    def test_feature_to_datamatrix(self):

        with self.assertRaises(TSCException):
            TSCDataFrame(self.simple_df).feature_to_array(feature="A")

        simple_df = self.simple_df.copy()
        simple_df = simple_df.drop(labels=[45])

        expected_shape = (3, 2)  # 3 time series a 2 time steps

        for i in range(simple_df.shape[1]):
            feature = simple_df.columns[i]

            actual = TSCDataFrame(simple_df).feature_to_array(feature=feature)
            expected = np.reshape(simple_df.loc[:, feature].to_numpy(), expected_shape)

            nptest.assert_equal(actual, expected)

    def test_from_timeseries_tensor(self):
        matrix = np.zeros([3, 2, 2])  # 1st: time series ID, 2nd: time, 3rd: feature
        matrix[0, :, :] = 1
        matrix[1, :, :] = 2
        matrix[2, :, :] = 3

        feature_cols = pd.Index(["A", "B"])

        actual = TSCDataFrame.from_tensor(matrix, columns=feature_cols)

        time_index_expected = pd.MultiIndex.from_arrays(
            [[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
            names=[actual.tsc_id_idx_name, TSCDataFrame.tsc_time_idx_name],
        )

        feature_col_expected = pd.Index(
            ["A", "B"], name=TSCDataFrame.tsc_feature_col_name
        )
        data_expected = np.array(
            [[1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]], dtype=np.float
        )
        expected = pd.DataFrame(
            data=data_expected, index=time_index_expected, columns=feature_col_expected
        )
        pdtest.assert_frame_equal(actual, expected)

    def test_from_timeseries_tensor_time_index(self):
        matrix = np.zeros([3, 2, 2])  # 1st: time series ID, 2nd: time, 3rd: feature
        matrix[0, :, :] = 1
        matrix[1, :, :] = 2
        matrix[2, :, :] = 3

        feature_column = pd.Index(["A", "B"])

        actual = TSCDataFrame.from_tensor(
            matrix, columns=feature_column, time_values=np.array([100, 200])
        )

        time_index_expected = pd.MultiIndex.from_arrays(
            [[0, 0, 1, 1, 2, 2], [100, 200, 100, 200, 100, 200]],
            names=[actual.tsc_id_idx_name, TSCDataFrame.tsc_time_idx_name],
        )

        feature_col_expected = pd.Index(
            ["A", "B"], name=TSCDataFrame.tsc_feature_col_name
        )
        data_expected = np.array(
            [[1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]], dtype=np.float
        )
        expected = pd.DataFrame(
            data=data_expected, index=time_index_expected, columns=feature_col_expected
        )
        pdtest.assert_frame_equal(actual, expected)

    def test_from_shift_matrix_row(self):

        left_matrix = np.array([[1, 3, 5], [7, 9, 11]])
        right_matrix = np.array([[2, 4, 6], [8, 10, 12]])

        actual = TSCDataFrame.from_shift_matrices(
            left_matrix, right_matrix, snapshot_orientation="row"
        )

        # build expected
        values = np.array([[1, 3, 5], [2, 4, 6], [7, 9, 11], [8, 10, 12]])
        index = pd.MultiIndex.from_arrays(
            [np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])]
        )
        columns = [0, 1, 2]
        expected = TSCDataFrame(values.astype(np.float64), index=index, columns=columns)

        pdtest.assert_frame_equal(actual, expected)

    def test_from_shift_matrix_col(self):

        left_matrix = np.array([[1, 3, 5], [7, 9, 11]])
        right_matrix = np.array([[2, 4, 6], [8, 10, 12]])

        actual = TSCDataFrame.from_shift_matrices(
            left_matrix, right_matrix, snapshot_orientation="col"
        )

        # build expected
        values = np.array([[1, 7], [2, 8], [3, 9], [4, 10], [5, 11], [6, 12]])
        index = pd.MultiIndex.from_arrays(
            [np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 0, 1, 0, 1])]
        )
        columns = [0, 1]
        expected = TSCDataFrame(values.astype(np.float64), index=index, columns=columns)

        pdtest.assert_frame_equal(actual, expected)

    def test_time_interval(self):
        actual = TSCDataFrame(self.simple_df).time_interval()
        expected = (0, 19)

        self.assertEqual(actual, expected)

        actual = TSCDataFrame(self.simple_df).time_interval(0)
        expected = (0, 1)

        self.assertEqual(actual, expected)

        actual = TSCDataFrame(self.simple_df).time_interval(45)
        expected = (17, 19)

        self.assertEqual(actual, expected)

    def test_is_equal_length(self):
        actual = TSCDataFrame(self.simple_df).is_equal_length()
        expected = False
        self.assertEqual(actual, expected)

        simple_df = self.simple_df.copy(deep=True)
        simple_df = simple_df.drop(labels=45, axis=0)
        actual = TSCDataFrame(simple_df).is_equal_length()
        expected = True
        self.assertEqual(actual, expected)

    def test_is_const_deltatime(self):
        actual = TSCDataFrame(self.simple_df).is_const_delta_time()
        expected = True

        self.assertEqual(actual, expected)
        simple_df = self.simple_df.copy(deep=True)
        simple_df.loc[pd.IndexSlice[99, 1], :] = [1, 2]
        simple_df.loc[pd.IndexSlice[99, 5], :] = [1, 2]  # not equal time difference

        actual = TSCDataFrame(simple_df).is_equal_length()
        expected = False
        self.assertEqual(actual, expected)

    @unittest.skip(reason="see gitlab issue #85")
    def test_delta_time(self):
        # TODO: if adressing this issue, test for multiple n_values

        n_values = 100  # 100 -> delta_time=1.0, 20 delta_time=nan

        df1 = pd.DataFrame(
            np.arange(n_values), index=np.linspace(1, 100, n_values), columns=["A"]
        )
        df2 = pd.DataFrame(
            np.arange(n_values), index=np.linspace(101, 200, n_values), columns=["A"],
        )

        tsc = TSCDataFrame.from_frame_list([df1, df2])
        print(tsc.delta_time)

        exit()

        tsc = TSCDataFrame.from_single_timeseries(df1)

        raise NotImplementedError(
            "Finish implementation. Requires a "
            "'round_time_values' -- very small differences break "
            "the delta_time. At the same time a function to get the differences (with "
            "highest difference would also be nice."
        )

    def test_delta_time02(self):
        n_values = 100  # 100 -> delta_time=1.0, 20 delta_time=nan

        df1 = pd.DataFrame(
            np.arange(n_values),
            index=np.arange(n_values).astype(np.int64),
            columns=["A"],
        )
        df2 = pd.DataFrame(
            np.arange(n_values),
            index=np.arange(n_values).astype(np.int64),
            columns=["A"],
        )

        tsc = TSCDataFrame.from_frame_list([df1, df2])

        actual = tsc.delta_time
        expected = 1

        self.assertEqual(actual, expected)
        self.assertIsInstance(actual, np.int64)

    def test_is_normalized_time1(self):
        actual = TSCDataFrame(self.simple_df).is_normalized_time()
        self.assertTrue(actual)

        actual = TSCDataFrame(self.simple_df).tsc.normalize_time().is_normalized_time()
        self.assertTrue(actual)

    def test_is_normalized_time2(self):
        # not const time time delta
        simple_df = self.simple_df.copy(deep=True)
        simple_df.loc[pd.IndexSlice[99, 1], :] = [1, 2]
        simple_df.loc[pd.IndexSlice[99, 5], :] = [1, 2]

        actual = TSCDataFrame(simple_df).is_normalized_time()
        self.assertFalse(actual)

        with self.assertRaises(TSCException):
            # to normalize time we need a const time delta
            TSCDataFrame(simple_df).tsc.normalize_time()

    def test_is_normalized_time3(self):
        # not start zero
        simple_df = self.simple_df.copy(deep=True)
        simple_df.index = pd.MultiIndex.from_arrays(
            [
                simple_df.index.get_level_values(0),
                simple_df.index.get_level_values(1) + 1,
            ]
        )

        actual = TSCDataFrame(simple_df).is_normalized_time()
        self.assertFalse(actual)

        actual = TSCDataFrame(simple_df).tsc.normalize_time().is_normalized_time()
        self.assertTrue(actual)

    def test_is_normalized_time4(self):
        # time delta is not 1
        simple_df = self.simple_df.copy(deep=True)
        simple_df.index = pd.MultiIndex.from_arrays(
            [
                simple_df.index.get_level_values(0),
                simple_df.index.get_level_values(1) * 3,
            ]
        )

        actual = TSCDataFrame(simple_df).is_normalized_time()
        self.assertFalse(actual)

        actual = TSCDataFrame(simple_df).tsc.normalize_time().is_normalized_time()
        self.assertTrue(actual)

    def test_is_equal_time_values(self):
        actual = TSCDataFrame(self.simple_df).is_same_time_values()
        expected = False
        self.assertEqual(actual, expected)

        simple_df = self.simple_df.copy(deep=True)
        simple_df = simple_df.drop(labels=45, axis=0)

        actual = TSCDataFrame(simple_df).is_same_time_values()
        expected = True
        self.assertEqual(actual, expected)

    def test_isfinite1(self):
        actual = TSCDataFrame(self.simple_df).is_finite()
        self.assertTrue(actual)

    def test_isfinite2(self):
        tsc = TSCDataFrame(self.simple_df.copy())
        tsc.iloc[0, 0] = np.inf

        actual = TSCDataFrame(tsc).is_finite()
        self.assertFalse(actual)

    def test_isfinite3(self):
        tsc = TSCDataFrame(self.simple_df.copy())
        tsc.iloc[0, 0] = np.nan

        actual = TSCDataFrame(tsc).is_finite()
        self.assertFalse(actual)

    def test_iterator(self):
        tc = TSCDataFrame(self.simple_df)
        counter = 0

        for i, ts in tc.itertimeseries():
            # Test 1 - frame has to be equal to original DF
            pdtest.assert_frame_equal(ts, self.simple_df.loc[i, :])

            # Test 2 - id has to be in the id index level
            self.assertTrue(i in self.simple_df.index.levels[0])
            counter += 1

        # Test 3 - the number of iterations has to match
        self.assertEqual(counter, len(self.simple_df.index.levels[0]))

    def test_timeseries_initial_states(self):
        actual = TSCDataFrame(self.simple_df).initial_states()

        self.assertIsInstance(actual, pd.DataFrame)

        idx = pd.MultiIndex.from_arrays(
            [[0, 1, 15, 45], [0, 0, 0, 17]],
            names=[TSCDataFrame.tsc_id_idx_name, TSCDataFrame.tsc_time_idx_name],
        )
        col = pd.Index(["A", "B"], name=TSCDataFrame.tsc_feature_col_name)

        values = self.simple_df.to_numpy()[[0, 2, 4, 6], :]
        expected = pd.DataFrame(values, index=idx, columns=col)

        pdtest.assert_frame_equal(actual, expected)

    def test_timeseries_initial_states_n_samples(self):
        actual = TSCDataFrame(self.simple_df).initial_states(n_samples=2)

        self.assertIsInstance(actual, TSCDataFrame)

        idx = pd.MultiIndex.from_arrays(
            [[0, 0, 1, 1, 15, 15, 45, 45], [0, 1, 0, 1, 0, 1, 17, 18]],
            names=[TSCDataFrame.tsc_id_idx_name, TSCDataFrame.tsc_time_idx_name],
        )
        col = pd.Index(["A", "B"], name=TSCDataFrame.tsc_feature_col_name)

        values = self.simple_df.to_numpy()[[0, 1, 2, 3, 4, 5, 6, 7], :]
        expected = pd.DataFrame(values, index=idx, columns=col)

        pdtest.assert_frame_equal(actual, expected)

        with self.assertRaises(TSCException):
            # some time series have only length 2
            TSCDataFrame(self.simple_df).initial_states(n_samples=3)

    def test_timeseries_final_state(self):
        expected = TSCDataFrame(self.simple_df).final_states()

        idx = pd.MultiIndex.from_arrays(
            [[0, 1, 15, 45], [1, 1, 1, 19]],
            names=[TSCDataFrame.tsc_id_idx_name, TSCDataFrame.tsc_time_idx_name],
        )
        col = pd.Index(["A", "B"], name=TSCDataFrame.tsc_feature_col_name)

        values = self.simple_df.to_numpy()[[1, 3, 5, 8], :]
        actual = pd.DataFrame(values, index=idx, columns=col)

        pdtest.assert_frame_equal(expected, actual)

    def test_time_delta01(self):
        actual = TSCDataFrame(self.simple_df).delta_time
        expected = 1.0
        self.assertEqual(actual, expected)

        simple_df = self.simple_df.copy()
        simple_df.loc[pd.IndexSlice[45, 100], :] = [
            5,
            4,
        ]  # "destroy" existing time delta of id 45

        actual = TSCDataFrame(simple_df).delta_time
        expected = pd.Series(
            data=[1, 1, 1, np.nan],
            index=pd.Index([0, 1, 15, 45], name=TSCDataFrame.tsc_id_idx_name),
            name="delta_time",
        )

        pdtest.assert_series_equal(actual, expected)

        simple_df = simple_df.drop(labels=45)

        # id 45 now has only 1 time point (time delta cannot be computed)
        simple_df.loc[pd.IndexSlice[45, 1], :] = [1, 2]

    def test_time_delta02(self):

        # all time series have irregular time value frequency
        test_df = TSCDataFrame(
            np.random.rand(6, 2),
            index=pd.MultiIndex.from_product([[0, 1], [1, 3, 100]]),
            columns=["A", "B"],
        )

        actual = test_df.delta_time

        # expected to return single nan
        self.assertTrue(np.isnan(actual))

    def test_time_array(self):
        actual = TSCDataFrame(self.simple_df).time_values()
        expected = self.simple_df.index.levels[1].to_numpy()
        nptest.assert_equal(actual, expected)

        simple_df = self.simple_df.copy()

        # include non-const time delta
        simple_df.loc[pd.IndexSlice[45, 100], :] = (
            5,
            4,
        )

        actual = TSCDataFrame(self.simple_df).time_values()
        expected = np.unique(self.simple_df.index.levels[1].to_numpy())
        nptest.assert_equal(actual, expected)

    def test_time_array_fill(self):
        actual = TSCDataFrame(self.simple_df).time_values_delta_time()
        expected = np.arange(0, 19 + 1, 1)
        nptest.assert_equal(actual, expected)

        simple_df = self.simple_df.copy()
        simple_df.loc[pd.IndexSlice[45, 100], :] = [
            5,
            4,
        ]  # include non-const time delta
        # ... should raise error
        with self.assertRaises(TSCException):
            TSCDataFrame(simple_df).time_values_delta_time()

    def test_single_time_df(self):
        tsc = TSCDataFrame(self.simple_df)
        actual = tsc.select_time_values(time_values=0)

        # cannot be a TSC anymore, because only one time point does not make a time series
        self.assertIsInstance(actual, pd.DataFrame)

        expected = self.simple_df.iloc[[0, 2, 4], :]
        pdtest.assert_frame_equal(actual, expected)

        with self.assertRaises(KeyError):
            tsc.select_time_values(time_values=100)  # no time series with this time

        actual = tsc.select_time_values(time_values=17)

        # in 'key' use a list to enforce a data frame, not a series
        expected = self.simple_df.iloc[[6], :]
        pdtest.assert_frame_equal(actual, expected)

    def test_multi_time_tsc(self):
        ts = TSCDataFrame(self.simple_df)

        new_ts = ts.select_time_values(time_values=np.array([0, 1, 17, 18]))

        self.assertIsInstance(new_ts, TSCDataFrame)
        self.assertNotIn(
            19, new_ts.index.get_level_values(TSCDataFrame.tsc_time_idx_name)
        )

        self.assertTrue(np.in1d(ts.ids, new_ts.ids).all())

        self.assertTrue(np.in1d(new_ts.time_values(), (0, 1, 17, 18)).all())

    def test_multi_time_tsc2(self):
        ts = TSCDataFrame(self.simple_df)

        new_ts = ts.select_time_values(time_values=np.array([0, 1]))
        self.assertIsInstance(new_ts, TSCDataFrame)

        self.assertTrue(np.in1d(new_ts.time_values(), (0, 1)).all())
        self.assertTrue(np.in1d(new_ts.ids, (0, 1, 15)).all())

    def test_multi_time_tsc3(self):
        ts = TSCDataFrame(self.simple_df)
        actual = ts.select_time_values(time_values=np.array([0]))
        expected = self.simple_df.loc[pd.IndexSlice[:, 0], :]

        pdtest.assert_frame_equal(actual, expected)

    def test_multi_time_tsc4(self):
        # behaviour if not all time points are present
        ts = TSCDataFrame(self.simple_df)

        # -1 is even an illegal
        new_ts = ts.select_time_values(time_values=np.array([0, 1, -1]))

        self.assertTrue(np.in1d(new_ts.time_values(), (0, 1)).all())
        self.assertTrue(np.in1d(new_ts.ids, (0, 1, 15)).all())

    def test_loc_slice01(self):
        # get time series with ID = 0
        ts = TSCDataFrame(self.simple_df).loc[0, :]  # does not fail

        self.assertFalse(isinstance(ts, TSCDataFrame))  # is not a TSCDataFrame because
        self.assertTrue(isinstance(ts, pd.DataFrame))

    def test_loc_slice02(self):
        tscdf = TSCDataFrame(self.simple_df)
        idx = pd.IndexSlice
        tscdf_sliced = tscdf.loc[idx[:, 0], :]

        # after slicing for a single time, it is not a valid TSCDataFrame anymore, therefore fall back to pd.DataFrame
        self.assertFalse(isinstance(tscdf_sliced, TSCDataFrame))
        self.assertTrue(isinstance(tscdf_sliced, pd.DataFrame))

    def test_loc_slice03(self):
        tscdf = TSCDataFrame(self.simple_df)
        idx = pd.IndexSlice
        tscdf_sliced = tscdf.loc[idx[:, 17], :]

        # after slicing for a single time, it is not a valid TSCDataFrame anymore, therefore fall back to pd.DataFrame
        self.assertFalse(isinstance(tscdf_sliced, TSCDataFrame))
        self.assertTrue(isinstance(tscdf_sliced, pd.DataFrame))

    def test_loc_slice04(self):
        tscdf = TSCDataFrame(self.simple_df)
        sliced_df = tscdf.loc[0, :]  # only one time series -> fall back to pd.DataFrame

        self.assertFalse(isinstance(sliced_df, TSCDataFrame))
        self.assertTrue(isinstance(sliced_df, pd.DataFrame))

    def test_loc_slice05(self):
        tc = TSCDataFrame(self.simple_df)

        # Here we expect to obtain a pd.Series, it is not a valid TSCDataFrame anymore
        pdtest.assert_series_equal(tc.loc[0, "A"], self.simple_df.loc[0, "A"])
        pdtest.assert_series_equal(tc.loc[0, "B"], self.simple_df.loc[0, "B"])
        pdtest.assert_series_equal(tc.loc[1, "A"], self.simple_df.loc[1, "A"])
        pdtest.assert_series_equal(tc.loc[1, "B"], self.simple_df.loc[1, "B"])

    def test_loc_slice06(self):
        tc = TSCDataFrame(self.simple_df)

        actual_a = tc.loc[:, "A"]
        actual_b = tc.loc[:, "B"]

        # TODO: note the cast to pd.DataFrame -- currently there is no TSCSeries,
        #  i.e. also a single feature column is a DataFrame. This cold be changed in
        #  future to be closer to the pandas data structures...
        expected_a = pd.DataFrame(self.simple_df.loc[:, "A"])
        expected_a.columns.name = TSCDataFrame.tsc_feature_col_name

        expected_b = pd.DataFrame(self.simple_df.loc[:, "B"])
        expected_b.columns.name = TSCDataFrame.tsc_feature_col_name

        pdtest.assert_frame_equal(actual_a, expected_a)
        pdtest.assert_frame_equal(actual_b, expected_b)

    def test_loc_slice07(self):
        df = self.simple_df.copy()

        # index is float64 but can be converted to int64 without loss
        df.index.set_levels(
            df.index.levels[0].astype(np.float64), level=0, inplace=True
        )

        # index must be the same
        pdtest.assert_frame_equal(TSCDataFrame(df), TSCDataFrame(self.simple_df))

    def test_loc_slice08(self):
        df = self.simple_df.copy()

        # index is float64 and cannot be converted to int64 without loss
        df.index.set_levels(
            df.index.levels[0].astype(np.float64) + 0.01, level=0, inplace=True
        )

        with self.assertRaises(AttributeError):
            TSCDataFrame(df)

    def test_slice01(self):
        tsc = TSCDataFrame(self.simple_df)
        actual = tsc["A"]
        self.assertIsInstance(actual, TSCDataFrame)

    def test_slice02(self):
        tsc = TSCDataFrame(self.simple_df)
        bool_idx = np.ones(tsc.shape[0], dtype=np.bool)
        bool_idx[-3:] = False

        actual = tsc[bool_idx]

        self.assertIsInstance(actual, TSCDataFrame)

    def test_slice04(self):
        tsc = TSCDataFrame(self.simple_df)
        actual = tsc[tsc < 0.5]

        self.assertIsInstance(actual, TSCDataFrame)

    def test_at_index(self):

        # tests the .at index with two examples

        tsc = TSCDataFrame(self.simple_df)
        actual1 = tsc.at[(0, 1), "A"]
        actual2 = tsc.at[(45, 17), "B"]

        expected1 = self.simple_df.at[(0, 1), "A"]
        expected2 = self.simple_df.at[(45, 17), "B"]

        self.assertEqual(actual1, expected1)
        self.assertEqual(actual2, expected2)

    def test_xs_index01(self):
        tsc = TSCDataFrame(self.simple_df)

        actual = tsc.xs(0)

        self.assertIsInstance(actual, pd.DataFrame)
        pdtest.assert_frame_equal(actual, self.simple_df.xs(0))

    def test_xs_index02(self):
        tsc = TSCDataFrame(self.simple_df)

        actual = tsc.xs("A", axis=1)

        # still a valid TScDataFrame
        self.assertIsInstance(actual, TSCDataFrame)

        # xs returns a Series
        expected = pd.DataFrame(self.simple_df.xs("A", axis=1))
        pdtest.assert_frame_equal(actual, expected, check_names=False)

    def test_set_index(self):
        tsc = TSCDataFrame(self.simple_df)

        # It is generally difficult to set a time series with using .loc, it is much easier to add a new time series
        #  with `insert_new_time_series`

        with self.assertRaises(AttributeError):
            tsc.loc[(100, 0), :] = 1  # a time series has to more than 2

        with self.assertRaises(AttributeError):
            # for the second level (after 100) is now a whitespace '' because the time is not specified
            # this leads the tsc.index.dtype go to object (due to a string) and is therefore not numeric anymore
            tsc.loc[100, :] = 1

    def test_concat_new_timeseries(self):
        tsc = TSCDataFrame(self.simple_df)
        new_tsc = pd.DataFrame(
            np.random.rand(2, 2),
            index=pd.MultiIndex.from_tuples([(100, 0), (100, 1)]),
            columns=["A", "B"],
        )

        full_tsc = pd.concat([tsc, new_tsc], axis=0)

        self.assertTrue(isinstance(full_tsc, TSCDataFrame))

        # The order defines the type. This is correct, but risky, it is therefore
        # better to use the method `insert_new_timeseries`
        full_df = pd.concat([new_tsc, tsc], axis=0)
        self.assertFalse(isinstance(full_df, TSCDataFrame))
        self.assertTrue(isinstance(full_df, pd.DataFrame))

    def test_insert_timeseries01(self):
        tsc = TSCDataFrame(self.simple_df)
        new_ts = pd.DataFrame(np.random.rand(2, 2), index=[0, 1], columns=["A", "B"])
        tsc = tsc.insert_ts(df=new_ts)
        self.assertTrue(isinstance(tsc, TSCDataFrame))

    def test_insert_timeseries02(self):
        tsc = TSCDataFrame(self.simple_df)

        # Column is not present
        new_ts = pd.DataFrame(np.random.rand(2, 2), index=[0, 1], columns=["A", "NA"])

        with self.assertRaises(ValueError):
            tsc.insert_ts(new_ts)

    def test_insert_timeseries03(self):
        tsc = TSCDataFrame(self.simple_df)
        new_ts = pd.DataFrame(np.random.rand(2, 2), index=[0, 1], columns=["A", "B"])

        with self.assertRaises(ValueError):
            tsc.insert_ts(new_ts, 1.5)  # id has to be int

        with self.assertRaises(ValueError):
            tsc.insert_ts(new_ts, 1)  # id=1 already present

    def test_insert_timeseries04(self):
        tsc = TSCDataFrame(self.simple_df)

        # Not unique time points -> invalid
        new_ts = pd.DataFrame(np.random.rand(2, 2), index=[0, 0], columns=["A", "B"])

        with self.assertRaises(AttributeError):
            tsc.insert_ts(new_ts, None)

    def test_build_from_single_timeseries(self):
        df = pd.DataFrame(np.random.rand(5), index=np.arange(5, 0, -1), columns=["A"])
        tsc = TSCDataFrame.from_single_timeseries(df)

        self.assertIsInstance(tsc, TSCDataFrame)

    def test_time_not_disappear_initial_state(self):
        """One observation was that a feature-column named 'time' disappears because the
        index is set to a regular column. This is tested here, such a 'time'
        feature-column does not disappear. """

        tsc = TSCDataFrame(self.simple_df)
        tsc[TSCDataFrame.tsc_time_idx_name] = 1

        initial_states = tsc.initial_states()
        self.assertTrue(TSCDataFrame.tsc_time_idx_name in initial_states.columns)

    def test_str_time_indices(self):
        simple_df = self.simple_df.copy(deep=True)

        simple_df.index = simple_df.index.set_levels(
            self.simple_df.index.levels[1].astype(str), level=1
        )

        with self.assertRaises(AttributeError):
            TSCDataFrame(simple_df)

    def test_float_time_indices(self):
        simple_df = self.simple_df.copy(deep=True)

        simple_df.index = simple_df.index.set_levels(
            self.simple_df.index.levels[1].astype(np.float), level=1
        )

        self.assertIsInstance(TSCDataFrame(simple_df), TSCDataFrame)

    def test_datetime_time_indices(self):
        simple_df = self.simple_df.copy(deep=True)

        dates = pd.to_datetime(
            "2019-11-" + (self.simple_df.index.levels[1] + 1).astype(str).str.zfill(2),
            format="%Y-%m-%d",
        )

        simple_df.index = simple_df.index.set_levels(dates, level=1)

        self.assertIsInstance(TSCDataFrame(simple_df), TSCDataFrame)


class TestInitialCondition(unittest.TestCase):
    def _tsc01(self):
        time = np.linspace(1, 10, 10)
        values = np.column_stack([np.sin(time), np.sin(time + np.pi / 2)])

        return TSCDataFrame.from_single_timeseries(pd.DataFrame(values, index=time))

    def _tsc02(self):
        time1 = np.linspace(1, 10, 10)
        time2 = np.linspace(10, 20, 10)

        values01 = np.column_stack([np.sin(time1), np.sin(time1 + np.pi / 2)])
        values02 = np.column_stack([np.sin(time2), np.sin(time2 + np.pi / 2)])

        _tsc = TSCDataFrame.from_single_timeseries(pd.DataFrame(values01, index=time1))
        return _tsc.insert_ts(pd.DataFrame(values02, index=time2))

    def setUp(self) -> None:
        self.test_tsc01 = self._tsc01()
        self.test_tsc02 = self._tsc02()

    def test_from_array01(self):
        # single_sample (1D)
        actual = InitialCondition.from_array(np.array([1, 2, 3]), ["A", "B", "C"])

        expected = pd.DataFrame(
            np.array([[1, 2, 3]]),  # note it is 2D
            index=pd.Index([0], name=TSCDataFrame.tsc_id_idx_name),
            columns=pd.Index(["A", "B", "C"], name=TSCDataFrame.tsc_feature_col_name),
        )

        self.assertTrue(InitialCondition.validate(actual))
        pdtest.assert_frame_equal(actual, expected)

    def test_from_array02(self):
        actual = InitialCondition.from_array(
            np.array([[1, 2, 3], [4, 5, 6]]), ["A", "B", "C"]
        )

        expected = pd.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6]]),
            index=pd.Index([0, 1], name=TSCDataFrame.tsc_id_idx_name),
            columns=pd.Index(["A", "B", "C"], name=TSCDataFrame.tsc_feature_col_name),
        )

        self.assertTrue(InitialCondition.validate(actual))
        pdtest.assert_frame_equal(actual, expected)

    def test_from_tsc01(self):
        actual = InitialCondition.from_tsc(self.test_tsc01, n_samples_ic=1)

        expected = pd.DataFrame(self.test_tsc01).head(1)
        expected.index = expected.index.get_level_values(TSCDataFrame.tsc_id_idx_name)

        self.assertTrue(InitialCondition.validate(actual))
        pdtest.assert_frame_equal(actual, expected)

    def test_from_tsc02(self):
        actual = InitialCondition.from_tsc(self.test_tsc01, n_samples_ic=3)

        expected = pd.DataFrame(self.test_tsc01).head(3)

        self.assertTrue(InitialCondition.validate(actual))
        pdtest.assert_frame_equal(actual, expected)

    def test_iter_reconstruct_ic01(self):
        # test if it can handle a single time series

        n_samples_ic = 1

        for i, (actual_ic, actual_time_values) in enumerate(
            InitialCondition.iter_reconstruct_ic(self.test_tsc01, n_samples_ic=1)
        ):

            select_ts = pd.DataFrame(self.test_tsc01).loc[[i, None], :]
            expected_ic = select_ts.head(n_samples_ic)
            expected_ic.index = expected_ic.index.droplevel(
                TSCDataFrame.tsc_time_idx_name
            )

            expected_time_values = select_ts.index.get_level_values(
                TSCDataFrame.tsc_time_idx_name
            )

            self.assertTrue(InitialCondition.validate(actual_ic))
            pdtest.assert_frame_equal(actual_ic, expected_ic)
            nptest.assert_array_equal(actual_time_values, expected_time_values)

    def test_iter_reconstruct_ic02(self):
        # test if it can handle a multiple time series

        n_samples_ic = 1

        for i, (actual_ic, actual_time_values) in enumerate(
            InitialCondition.iter_reconstruct_ic(self.test_tsc02, n_samples_ic=1)
        ):

            select_ts = pd.DataFrame(self.test_tsc02).loc[[i, None], :]
            expected_ic = select_ts.head(n_samples_ic)
            expected_ic.index = expected_ic.index.droplevel(
                TSCDataFrame.tsc_time_idx_name
            )

            expected_time_values = select_ts.index.get_level_values(
                TSCDataFrame.tsc_time_idx_name
            )

            self.assertTrue(InitialCondition.validate(actual_ic))
            pdtest.assert_frame_equal(actual_ic, expected_ic)
            nptest.assert_array_equal(actual_time_values, expected_time_values)

    def test_iter_reconstruct_ic03(self):
        # test

        n_sample_ic = 3

        for i, (actual_ic, actual_time_values) in enumerate(
            InitialCondition.iter_reconstruct_ic(self.test_tsc02, n_samples_ic=3)
        ):

            select_ts = self.test_tsc02.loc[[i, None], :]
            expected_ic = select_ts.head(n_sample_ic)

            expected_time_values = select_ts.index.get_level_values(
                TSCDataFrame.tsc_time_idx_name
            )[n_sample_ic - 1 :]

            self.assertTrue(InitialCondition.validate(actual_ic))
            pdtest.assert_frame_equal(actual_ic, expected_ic)
            nptest.assert_array_equal(actual_time_values, expected_time_values)


if __name__ == "__main__":
    # test = TestTSCDataFrame()
    # test.setUp()
    # test.test_build_from_single_timeseries()
    #
    # exit()
    unittest.main()
