import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest

from datafold.pcfold.timeseries.collection import (
    TSCDataFrame,
    TimeSeriesCollectionError,
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
        self.assertEqual(tc.nr_timeseries, 4)

    def test_nr_qoi(self):
        tc = TSCDataFrame(self.simple_df)
        self.assertEqual(tc.nr_qoi, 2)

    def test_shape(self):
        tc = TSCDataFrame(self.simple_df)
        self.assertEqual(tc.shape, (9, 2))

    def test_nelements_timeseries(self):
        tc = TSCDataFrame(self.simple_df)
        pdtest.assert_series_equal(
            tc.lengths_time_series,
            pd.Series(
                [2, 2, 2, 3], index=pd.Index([0, 1, 15, 45], name="ID"), name="counts"
            ),
        )

        simple_df = self.simple_df.copy()
        simple_df = simple_df.drop(
            labels=[45]
        )  # is the only one which has time series length 4
        actual = TSCDataFrame(simple_df).lengths_time_series
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

    def test_qoi_to_datamatrix(self):

        with self.assertRaises(TimeSeriesCollectionError):
            TSCDataFrame(self.simple_df).qoi_to_ndarray(qoi="A")

        simple_df = self.simple_df.copy()
        simple_df = simple_df.drop(labels=[45])

        expected_shape = (3, 2)  # 3 time series a 2 time steps

        for i in range(simple_df.shape[1]):
            qoi = simple_df.columns[i]

            actual = TSCDataFrame(simple_df).qoi_to_ndarray(qoi=qoi)
            expected = np.reshape(simple_df.loc[:, qoi].to_numpy(), expected_shape)

            nptest.assert_equal(actual, expected)

    def test_from_timeseries_tensor(self):
        matrix = np.zeros([3, 2, 2])  # 1st: time series ID, 2nd: time, 3rd: qoi
        matrix[0, :, :] = 1
        matrix[1, :, :] = 2
        matrix[2, :, :] = 3

        qoi_column = pd.Index(["A", "B"])

        actual = TSCDataFrame.from_tensor(matrix, columns=qoi_column)

        time_index_expected = pd.MultiIndex.from_arrays(
            [[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]], names=[actual.IDX_ID_NAME, "time"]
        )

        qoi_column_expected = pd.Index(["A", "B"], name="qoi")
        data_expected = np.array(
            [[1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]], dtype=np.float
        )
        expected = pd.DataFrame(
            data=data_expected, index=time_index_expected, columns=qoi_column_expected
        )
        pdtest.assert_frame_equal(actual, expected)

    def test_from_timeseries_tensor_time_index(self):
        matrix = np.zeros([3, 2, 2])  # 1st: time series ID, 2nd: time, 3rd: qoi
        matrix[0, :, :] = 1
        matrix[1, :, :] = 2
        matrix[2, :, :] = 3

        qoi_column = pd.Index(["A", "B"])

        actual = TSCDataFrame.from_tensor(
            matrix, columns=qoi_column, time_index=np.array([100, 200])
        )

        time_index_expected = pd.MultiIndex.from_arrays(
            [[0, 0, 1, 1, 2, 2], [100, 200, 100, 200, 100, 200]],
            names=[actual.IDX_ID_NAME, "time"],
        )

        qoi_column_expected = pd.Index(["A", "B"], name="qoi")
        data_expected = np.array(
            [[1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]], dtype=np.float
        )
        expected = pd.DataFrame(
            data=data_expected, index=time_index_expected, columns=qoi_column_expected
        )
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

    def test_is_const_dt(self):
        actual = TSCDataFrame(self.simple_df).is_const_dt()
        expected = True

        self.assertEqual(actual, expected)
        simple_df = self.simple_df.copy(deep=True)
        simple_df.loc[pd.IndexSlice[99, 1], :] = [1, 2]
        simple_df.loc[pd.IndexSlice[99, 5], :] = [1, 2]  # not equal time difference

        actual = TSCDataFrame(simple_df).is_equal_length()
        expected = False
        self.assertEqual(actual, expected)

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

        with self.assertRaises(TimeSeriesCollectionError):
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

    def test_is_equal_time_points(self):
        actual = TSCDataFrame(self.simple_df).is_equal_time_index()
        expected = False
        self.assertEqual(actual, expected)

        simple_df = self.simple_df.copy(deep=True)
        simple_df = simple_df.drop(labels=45, axis=0)

        actual = TSCDataFrame(simple_df).is_equal_time_index()
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

    def test_timeseries_starts(self):
        expected = TSCDataFrame(self.simple_df).initial_states_df()

        idx = pd.MultiIndex.from_arrays(
            [[0, 1, 15, 45], [0, 0, 0, 17]], names=["ID", "initial_time"]
        )
        col = pd.Index(["A", "B"], name="qoi")

        values = self.simple_df.to_numpy()[[0, 2, 4, 6], :]
        actual = pd.DataFrame(values, index=idx, columns=col)

        pdtest.assert_frame_equal(expected, actual)

    def test_timeseries_ends(self):
        expected = TSCDataFrame(self.simple_df).final_states_df()

        idx = pd.MultiIndex.from_arrays(
            [[0, 1, 15, 45], [1, 1, 1, 19]], names=["ID", "final_time"]
        )
        col = pd.Index(["A", "B"], name="qoi")

        values = self.simple_df.to_numpy()[[1, 3, 5, 8], :]
        actual = pd.DataFrame(values, index=idx, columns=col)

        pdtest.assert_frame_equal(expected, actual)

    def test_time_delta(self):
        actual = TSCDataFrame(self.simple_df).dt
        expected = 1.0
        self.assertEqual(actual, expected)

        simple_df = self.simple_df.copy()
        simple_df.loc[pd.IndexSlice[45, 100], :] = [
            5,
            4,
        ]  # "destroy" existing time delta of id 45

        actual = TSCDataFrame(simple_df).dt
        expected = pd.Series(
            data=[1, 1, 1, np.nan],
            index=pd.Index([0, 1, 15, 45], name=TSCDataFrame.IDX_ID_NAME),
            name="dt",
        )

        pdtest.assert_series_equal(actual, expected)

        simple_df = simple_df.drop(labels=45)

        # id 45 now has only 1 time point (time delta cannot be computed)
        simple_df.loc[pd.IndexSlice[45, 1], :] = [1, 2]

    def test_time_array(self):
        actual = TSCDataFrame(self.simple_df).time_indices(unique_values=True)
        expected = self.simple_df.index.levels[1].to_numpy()
        nptest.assert_equal(actual, expected)

        simple_df = self.simple_df.copy()

        # include non-const time delta
        simple_df.loc[pd.IndexSlice[45, 100], :] = (
            5,
            4,
        )

        actual = TSCDataFrame(self.simple_df).time_indices(unique_values=True)
        expected = np.unique(self.simple_df.index.levels[1].to_numpy())
        nptest.assert_equal(actual, expected)

    def test_time_array_fill(self):
        actual = TSCDataFrame(self.simple_df).time_index_fill()
        expected = np.arange(0, 19 + 1, 1)
        nptest.assert_equal(actual, expected)

        simple_df = self.simple_df.copy()
        simple_df.loc[pd.IndexSlice[45, 100], :] = [
            5,
            4,
        ]  # include non-const time delta
        # ... should raise error
        with self.assertRaises(TimeSeriesCollectionError):
            TSCDataFrame(simple_df).time_index_fill()

    def test_single_time_df(self):
        tsc = TSCDataFrame(self.simple_df)
        actual = tsc.single_time_df(time=0)
        expected = self.simple_df.iloc[[0, 2, 4], :]
        pdtest.assert_frame_equal(actual, expected)

        with self.assertRaises(KeyError):
            tsc.single_time_df(time=100)  # no time series with this time

        actual = tsc.single_time_df(time=17)
        expected = self.simple_df.iloc[
            [6], :
        ]  # in 'key' use a list to enforce a data frame, not a series
        pdtest.assert_frame_equal(actual, expected)

    def test_index01(self):
        # get time series with ID = 0
        ts = TSCDataFrame(self.simple_df).loc[0, :]  # does not fail

        self.assertFalse(isinstance(ts, TSCDataFrame))  # is not a TSCDataFrame because
        self.assertTrue(isinstance(ts, pd.DataFrame))

    def test_index02(self):
        tscdf = TSCDataFrame(self.simple_df)
        idx = pd.IndexSlice
        tscdf_sliced = tscdf.loc[idx[:, 0], :]

        # after slicing for a single time, it is not a valid TSCDataFrame anymore, therefore fall back to pd.DataFrame
        self.assertFalse(isinstance(tscdf_sliced, TSCDataFrame))
        self.assertTrue(isinstance(tscdf_sliced, pd.DataFrame))

    def test_index03(self):
        tscdf = TSCDataFrame(self.simple_df)
        idx = pd.IndexSlice
        tscdf_sliced = tscdf.loc[idx[:, 17], :]

        # after slicing for a single time, it is not a valid TSCDataFrame anymore, therefore fall back to pd.DataFrame
        self.assertFalse(isinstance(tscdf_sliced, TSCDataFrame))
        self.assertTrue(isinstance(tscdf_sliced, pd.DataFrame))

    def test_index04(self):
        tscdf = TSCDataFrame(self.simple_df)
        sliced_df = tscdf.loc[0, :]  # only one time series -> fall back to pd.DataFrame

        self.assertFalse(isinstance(sliced_df, TSCDataFrame))
        self.assertTrue(isinstance(sliced_df, pd.DataFrame))

    def test_index_05(self):
        tc = TSCDataFrame(self.simple_df)

        # Here we expect to obtain a pd.Series, it is not a valid TSCDataFrame anymore
        pdtest.assert_series_equal(tc.loc[0, "A"], self.simple_df.loc[0, "A"])
        pdtest.assert_series_equal(tc.loc[0, "B"], self.simple_df.loc[0, "B"])
        pdtest.assert_series_equal(tc.loc[1, "A"], self.simple_df.loc[1, "A"])
        pdtest.assert_series_equal(tc.loc[1, "B"], self.simple_df.loc[1, "B"])

    def test_index_06(self):
        tc = TSCDataFrame(self.simple_df)

        actual_a = tc.loc[:, "A"]
        actual_b = tc.loc[:, "B"]

        # TODO: note the cast to pd.DataFrame -- currently there is no TSCSeries, i.e. also a single qoi column
        #  is a DataFrame. This cold be changed in future to be closer to the pandas data structures...
        expected_a = pd.DataFrame(self.simple_df.loc[:, "A"])
        expected_a.columns.name = "qoi"

        expected_b = pd.DataFrame(self.simple_df.loc[:, "B"])
        expected_b.columns.name = "qoi"

        pdtest.assert_frame_equal(actual_a, expected_a)
        pdtest.assert_frame_equal(actual_b, expected_b)

    def test_slice01(self):
        tsc = TSCDataFrame(self.simple_df)

        # TODO: this is tsc["A"] is actually still a valid TSCDataFrame
        self.assertTrue(isinstance(tsc["A"], pd.Series))

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

        # The order defines the type. This is correct, but risky, it is therefore better to use the method
        #  `insert_new_timeseries`
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
        """One observation was that a qoi-column named 'time' disappears because the index is set to a regular
        column. This is tested here, that the 'time' column not disappears. """

        tsc = TSCDataFrame(self.simple_df)
        tsc["time"] = 1

        initial_states = tsc.initial_states_df()
        self.assertTrue("time" in initial_states.columns)

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


if __name__ == "__main__":
    # test = TestTSCDataFrame()
    # test.setUp()
    # test.test_build_from_single_timeseries()
    #
    # exit()
    unittest.main()
