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
    rng = np.random.default_rng(2)

    def setUp(self) -> None:
        # The last two elements are used
        idx = pd.MultiIndex.from_arrays(
            [[0, 0, 1, 1, 15, 15, 45, 45, 45], [0, 1, 0, 1, 0, 1, 17, 18, 19]]
        )
        col = ["A", "B"]

        # TODO: make this a TSCDataFrame here already and save all the TSCDataFrame(...) wraps
        # TODO: in the long run use pytest fixtures
        self.simple_df = pd.DataFrame(
            self.rng.uniform(size=(9, 2)), index=idx, columns=col
        )

    def test_simple1(self):
        tc = TSCDataFrame(self.simple_df)
        pdtest.assert_frame_equal(tc.loc[0, :], self.simple_df.loc[0, :])
        pdtest.assert_frame_equal(tc.loc[1, :], self.simple_df.loc[1, :])

    def test_invalid_tsc(self):
        tc = TSCDataFrame(self.simple_df)

        # ID 45 appears at the start and end
        tsc_append = TSCDataFrame(
            np.arange(4).reshape(2, 2),
            index=pd.MultiIndex.from_arrays([[45, 45], [20, 21]]),
            columns=self.simple_df.columns,
        )

        with self.assertRaises(AttributeError):
            pd.concat([tsc_append, tc], axis=0)

        # time 1 for ID 0 is duplicated
        tsc_append = TSCDataFrame(
            np.arange(4).reshape(2, 2),
            index=pd.MultiIndex.from_arrays([[0, 0], [1, 2]]),
            columns=self.simple_df.columns,
        )

        with self.assertRaises(pd.errors.DuplicateLabelError):
            pd.concat([tsc_append, tc], axis=0)

        # time values of ID 0 are not sorted
        tsc_append = TSCDataFrame(
            np.arange(4).reshape(2, 2),
            index=pd.MultiIndex.from_arrays([[0, 0], [50, 51]]),
            columns=self.simple_df.columns,
        )

        with self.assertRaises(AttributeError):
            pd.concat([tsc_append, tc], axis=0)

    def test_n_timeseries(self):
        tc = TSCDataFrame(self.simple_df)
        self.assertEqual(tc.n_timeseries, 4)

    def test_n_feature(self):
        tc = TSCDataFrame(self.simple_df)
        self.assertEqual(tc.n_features, 2)

    def test_empty_tsc(self):
        df = pd.DataFrame(
            index=pd.MultiIndex.from_arrays(
                [[1], [1]],  # set 1's to set the types of the index
                names=[TSCDataFrame.tsc_id_idx_name, TSCDataFrame.tsc_time_idx_name],
            )
        )

        self.assertTrue(df.empty)
        self.assertTrue(TSCDataFrame(df).empty)

    def test_dropna(self):
        tc = TSCDataFrame(self.simple_df)
        tc_nan = tc.copy(deep=True)

        tc_nan.iloc[0, :] = np.nan
        tc_nan.iloc[-1, :] = np.nan
        actual = tc_nan.dropna()

        expect = tc.iloc[1:-1, :]

        pdtest.assert_frame_equal(actual, expect)

    def test_transpose(self):
        tc = TSCDataFrame(self.simple_df)

        actual = tc.transpose(copy=False)
        self.assertIsInstance(actual, pd.DataFrame)
        pdtest.assert_index_equal(actual.index, tc.columns)
        pdtest.assert_index_equal(actual.columns, tc.index)
        self.assertTrue(np.shares_memory(tc.to_numpy(), actual.to_numpy()))

        actual = tc.transpose(copy=True)
        self.assertFalse(np.shares_memory(tc.to_numpy(), actual.to_numpy()))

    def test_concat(self):
        tc1 = TSCDataFrame(self.simple_df)
        tc2 = TSCDataFrame(self.simple_df.copy())
        tc2.columns = pd.Index(["C", "D"])

        actual = pd.concat([tc1, tc2], axis=1)
        pdtest.assert_index_equal(
            actual.columns, pd.Index(["A", "B", "C", "D"], name="feature")
        )
        pdtest.assert_index_equal(actual.index, tc1.index)

        # concat with non-identical index
        with self.assertRaises(AttributeError):
            # raises error because the concat result not sorted and the time series broken in
            # two parts
            pd.concat([tc1.iloc[1:, :], tc2], axis=1)

        actual = pd.concat([tc1.iloc[1:, :], tc2], axis=1, sort=True)
        pdtest.assert_index_equal(
            actual.columns, pd.Index(["A", "B", "C", "D"], name="feature")
        )
        pdtest.assert_index_equal(actual.index, tc1.index)

    def test_shape(self):
        tc = TSCDataFrame(self.simple_df)
        self.assertEqual(tc.shape, (9, 2))

    def test_assign_column(self):
        tscdf = TSCDataFrame(self.simple_df)
        tscdf["A"] = 1
        tscdf["B"] = 2

        actualA = tscdf.to_numpy()[:, 0]
        expectA = np.ones(tscdf.shape[0])
        actualB = tscdf.to_numpy()[:, 1]
        expectB = np.ones(tscdf.shape[0]) * 2

        nptest.assert_equal(actualA, expectA)
        nptest.assert_equal(actualB, expectB)

    def test_set_index1(self):
        tsc_df = TSCDataFrame(self.simple_df)

        new_idx = pd.MultiIndex.from_arrays([np.zeros(9, dtype=float), np.arange(9)])

        actual = tsc_df.set_index(new_idx)

        # no change in the basis TSCDataFrame
        self.assertEqual(tsc_df.n_timeseries, 4)

        self.assertEqual(actual.n_timeseries, 1)
        nptest.assert_array_equal(actual.time_values(), np.arange(9))
        self.assertTrue(
            np.issubdtype(actual.index.get_level_values(0).dtype, np.integer)
        )

        # test inplace
        with self.assertRaises(ValueError):
            # Cannot specify 'inplace=True' when
            # 'flags.allows_duplicate_labels' is False.
            tsc_df.set_index(new_idx, inplace=True)

    def test_set_index2(self):
        tsc_df = TSCDataFrame(self.simple_df.copy())

        new_idx_float_id = pd.MultiIndex.from_arrays(
            [np.ones(9, dtype=float) * 0.5, np.arange(9)]
        )

        # test new_idx_float_id that
        with self.assertRaises(AttributeError):
            tsc_df.set_index(new_idx_float_id)

        with self.assertRaises(AttributeError):
            tsc_df.set_index(new_idx_float_id)

        # reset tsc_df
        tsc_df = TSCDataFrame(self.simple_df)

        # test new_idx_degenerated_ts
        # has only one time sample for ID 0
        new_idx_degenerated_ts = pd.MultiIndex.from_arrays(
            [np.hstack([0, np.ones(8)]), np.arange(9)]
        )

        self.assertTrue(tsc_df.set_index(new_idx_degenerated_ts).has_degenerate())
        self.assertFalse(tsc_df.has_degenerate())

    def test_set_index3(self):
        tsc_df = TSCDataFrame(self.simple_df.copy())

        with self.assertRaises(AttributeError):
            # only pd.Index is allowed (this is a restriction)
            tsc_df.index = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        with self.assertRaises(AttributeError):
            # go in index checks
            tsc_df.index = pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_set_datetime_index(self):
        tsc_df = TSCDataFrame(self.simple_df.copy())

        _ids = tsc_df.index.get_level_values(TSCDataFrame.tsc_id_idx_name)
        new_idx = np.arange(np.datetime64("2021-01-01"), np.datetime64("2021-01-10"))
        tsc_df.index = pd.MultiIndex.from_arrays([_ids, new_idx])

        self.assertTrue(tsc_df.is_datetime_index())
        self.assertIsInstance(tsc_df.delta_time, np.timedelta64)

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

    def test_from_array1(self):
        data = np.random.default_rng(None).uniform(size=(10, 2))

        actual = TSCDataFrame.from_array(data)

        self.assertIsInstance(actual, TSCDataFrame)

        nptest.assert_array_equal(
            actual.index.get_level_values(TSCDataFrame.tsc_time_idx_name), np.arange(10)
        )
        self.assertEqual(len(actual.ids), 1)
        self.assertEqual(actual.ids[0], 0)

    def test_from_array2(self):
        data = np.random.default_rng(None).uniform(size=(10, 2))

        expected_time_values = np.arange(10, 20)
        expected_id = 2
        actual = TSCDataFrame.from_array(
            data, time_values=expected_time_values, ts_id=2
        )

        self.assertIsInstance(actual, TSCDataFrame)

        nptest.assert_array_equal(
            actual.index.get_level_values(TSCDataFrame.tsc_time_idx_name),
            expected_time_values,
        )
        self.assertEqual(len(actual.ids), 1)
        self.assertEqual(actual.ids[0], expected_id)

    def test_from_frame_list0(self):
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

    def test_from_frame_list01(self):
        # include ts_ids

        frame_list = [self.simple_df.loc[i, :] for i in self.simple_df.index.levels[0]]

        actual = TSCDataFrame.from_frame_list(frame_list, ts_ids=[1, 3, 99, 101])
        expected = TSCDataFrame(self.simple_df)
        expected.index = pd.MultiIndex.from_arrays(
            [
                [1, 1, 3, 3, 99, 99, 101, 101, 101],
                expected.index.get_level_values(TSCDataFrame.tsc_time_idx_name),
            ],
            names=[TSCDataFrame.tsc_id_idx_name, TSCDataFrame.tsc_time_idx_name],
        )
        pdtest.assert_frame_equal(actual, expected)

    def test_from_frame_list02(self):
        df1 = self.simple_df.copy().loc[[0], :]  # pd.DataFrame but possible to transfer
        df2 = TSCDataFrame(self.simple_df.copy().loc[[1], :])

        # 1
        actual = TSCDataFrame.from_frame_list([self.simple_df])
        expected = TSCDataFrame.from_frame_list(
            [e[1] for e in list(TSCDataFrame(self.simple_df).itertimeseries())]
        )
        pdtest.assert_frame_equal(expected, actual)

        # 2
        actual = TSCDataFrame.from_frame_list(
            [self.simple_df], ts_ids=TSCDataFrame(self.simple_df).ids
        )
        pdtest.assert_frame_equal(TSCDataFrame(self.simple_df), actual)

        # 3
        actual = TSCDataFrame.from_frame_list([df1, df2], ts_ids=[99, 100])
        expected_values = np.row_stack([df1.to_numpy(), df2.to_numpy()])

        nptest.assert_equal(actual.ids.values, np.array([99, 100]))
        nptest.assert_equal(actual.to_numpy(), expected_values)

        self.assertIsInstance(actual, TSCDataFrame)

    def test_from_frame_list03(self):
        df = pd.DataFrame(np.arange(6).reshape([3, 2]))

        # use identical frame:
        actual = TSCDataFrame.from_frame_list([df, df])

        expected = pd.concat([df, df])
        expected.index = pd.MultiIndex.from_arrays(
            [np.array([0, 0, 0, 1, 1, 1]), expected.index]
        )

        pdtest.assert_frame_equal(
            actual, expected, check_names=False, check_flags=False
        )

    def test_from_frame_list04(self):
        df = pd.DataFrame(np.arange(6).reshape([3, 2]))
        tsc = TSCDataFrame.from_single_timeseries(df.copy())

        # use identical TSCDataFrane:
        actual = TSCDataFrame.from_frame_list([tsc, tsc])

        expected = pd.concat(
            [
                TSCDataFrame.from_single_timeseries(df.copy(), ts_id=0),
                TSCDataFrame.from_single_timeseries(df.copy(), ts_id=1),
            ],
            axis=0,
        )
        pdtest.assert_frame_equal(actual, expected, check_names=True, check_flags=True)

    def test_feature_to_array1(self):
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

    def test_feature_to_array2(self):
        simple_df = TSCDataFrame(self.simple_df.copy())
        simple_df = simple_df.drop(labels=[45])

        with self.assertRaises(ValueError):
            # feature must be given if multiple features are present
            simple_df.feature_to_array(feature=None)

        # numpy array
        simple_df = simple_df.drop(labels=["A"], axis=1)
        actual = simple_df.feature_to_array(feature=None)

        expected = simple_df.to_numpy().reshape(3, 2)
        nptest.assert_equal(actual, expected)

        # pandas frame
        actual = simple_df.feature_to_array(feature=None, as_frame=True)

        expected = pd.DataFrame(
            simple_df.to_numpy().reshape(3, 2), index=simple_df.ids, columns=[0, 1]
        )
        pdtest.assert_frame_equal(actual, expected)

    def test_from_timeseries_tensor(self):
        matrix = np.zeros([3, 2, 2])  # 1st: time series ID, 2nd: time, 3rd: feature
        matrix[0, :, :] = 1
        matrix[1, :, :] = 2
        matrix[2, :, :] = 3

        feature_cols = pd.Index(["A", "B"])

        actual = TSCDataFrame.from_tensor(matrix, feature_names=feature_cols)

        time_index_expected = pd.MultiIndex.from_arrays(
            [[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
            names=[actual.tsc_id_idx_name, TSCDataFrame.tsc_time_idx_name],
        )

        feature_col_expected = pd.Index(
            ["A", "B"], name=TSCDataFrame.tsc_feature_col_name
        )
        data_expected = np.array(
            [[1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]], dtype=float
        )
        expected = pd.DataFrame(
            data=data_expected, index=time_index_expected, columns=feature_col_expected
        )
        pdtest.assert_frame_equal(actual, expected, check_flags=False)

    def test_from_timeseries_tensor_time_index(self):
        matrix = np.zeros([3, 2, 2])  # 1st: time series ID, 2nd: time, 3rd: feature
        matrix[0, :, :] = 1
        matrix[1, :, :] = 2
        matrix[2, :, :] = 3

        feature_column = pd.Index(["A", "B"])

        actual = TSCDataFrame.from_tensor(
            matrix, feature_names=feature_column, time_values=np.array([100, 200])
        )

        time_index_expected = pd.MultiIndex.from_arrays(
            [[0, 0, 1, 1, 2, 2], [100, 200, 100, 200, 100, 200]],
            names=[actual.tsc_id_idx_name, TSCDataFrame.tsc_time_idx_name],
        )

        feature_col_expected = pd.Index(
            ["A", "B"], name=TSCDataFrame.tsc_feature_col_name
        )
        data_expected = np.array(
            [[1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]], dtype=float
        )
        expected = pd.DataFrame(
            data=data_expected, index=time_index_expected, columns=feature_col_expected
        )
        pdtest.assert_frame_equal(actual, expected, check_flags=False)

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
        expected = TSCDataFrame(values.astype(float), index=index, columns=columns)

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
        expected = TSCDataFrame(values.astype(float), index=index, columns=columns)

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

    def test_nonequal_delta_time(self):
        df_int = pd.DataFrame(
            np.arange(12).reshape(6, 2), index=[0, 1, 2, 5, 6, 7], columns=["A", "B"]
        )

        index = np.append(
            np.arange(
                np.datetime64("2022-12-13"), np.datetime64("2022-12-16")
            ),  # three days
            np.arange(np.datetime64("2023-01-01"), np.datetime64("2023-01-04")),
        )  # after gap another three days

        df_datetime = pd.DataFrame(
            np.arange(12).reshape(6, 2), index=index, columns=["A", "B"]
        )
        tscdf_int = TSCDataFrame.from_frame_list([df_int])
        tscdf_datetime = TSCDataFrame.from_frame_list([df_datetime])

        self.assertTrue(np.isnan(tscdf_int.delta_time))
        self.assertTrue(np.isnat(tscdf_datetime.delta_time))

    def test_delta_time01(self):
        n_values = 100

        df1 = pd.DataFrame(
            np.arange(n_values), index=np.linspace(1, 100, n_values), columns=["A"]
        )
        df2 = pd.DataFrame(
            np.arange(n_values),
            index=np.linspace(101, 200, n_values),
            columns=["A"],
        )

        tsc = TSCDataFrame.from_frame_list([df1, df2])
        expected = 1.0
        self.assertEqual(tsc.delta_time, expected)

    def test_delta_time02(self):
        n_values = 100  # 100 -> delta_time=1.0, 20 delta_time=nan

        df1 = pd.DataFrame(
            np.arange(n_values),
            index=np.arange(n_values).astype(int),
            columns=["A"],
        )
        df2 = pd.DataFrame(
            np.arange(n_values),
            index=np.arange(n_values).astype(int),
            columns=["A"],
        )

        tsc = TSCDataFrame.from_frame_list([df1, df2])

        actual = tsc.delta_time
        expected = 1

        self.assertEqual(actual, expected)
        self.assertIsInstance(actual, np.integer)

    def test_delta_time_03(self):
        tscdf = TSCDataFrame(self.simple_df)
        self.assertEqual(tscdf.delta_time, 1)

    def test_delta_time_04(self):
        tscdf = TSCDataFrame(self.simple_df)
        tscdf = tscdf.groupby("ID").head(1)  # all degenerate
        self.assertTrue(np.isnan(tscdf.delta_time))

    def test_delta_time_05(self):
        tscdf = TSCDataFrame(self.simple_df)
        tscdf = tscdf.iloc[1:, :]  # different n_timesteps
        expected = pd.Series(
            data=[np.nan, 1.0, 1.0, 1.0], index=tscdf.ids, name="delta_time"
        )
        actual = tscdf.delta_time
        pdtest.assert_series_equal(actual, expected)

    def test_delta_time_06(self):
        tscdf = TSCDataFrame(self.simple_df)
        tscdf = tscdf.iloc[1:-1, :]  # all have the same n_timesteps
        expected = pd.Series(
            data=[np.nan, 1.0, 1.0, 1.0], index=tscdf.ids, name="delta_time"
        )
        actual = tscdf.delta_time
        pdtest.assert_series_equal(actual, expected)

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

    def test_degenerate_timeseries0(self):
        tsc_df = TSCDataFrame(self.simple_df)
        tsc_df = tsc_df.drop(0, level=1)

        nptest.assert_equal(tsc_df.degenerate_ids(), np.array([0, 1, 15], dtype=int))
        self.assertTrue(tsc_df.has_degenerate())

    def test_degenerate_timeseries1(self):
        # test behavior with delta time
        tsc_df = TSCDataFrame(self.simple_df)
        self.assertEqual(tsc_df.delta_time, 1)

        tsc_df = tsc_df.drop(0, level=1)
        actual = tsc_df.delta_time

        self.assertIsInstance(actual, pd.Series)
        pdtest.assert_series_equal(
            pd.Series([np.nan, np.nan, np.nan, 1], index=tsc_df.ids),
            actual,
            check_names=False,
        )

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

        pdtest.assert_frame_equal(actual, expected, check_flags=False)

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

        pdtest.assert_frame_equal(actual, expected, check_flags=False)

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

        pdtest.assert_frame_equal(expected, actual, check_flags=False)

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
            self.rng.uniform(size=(6, 2)),
            index=pd.MultiIndex.from_product([[0, 1], [1, 3, 100]]),
            columns=["A", "B"],
        )

        actual = test_df.delta_time

        # expected to return single nan
        self.assertTrue(np.isnan(actual))

    def test_time_delta03(self):
        # there is a special (faster) routine to take the delta_time for n_timesteps==2
        X_left = np.reshape(np.arange(100), (10, 10))
        X_right = np.reshape(np.arange(100, 200), (10, 10))

        X_tsc = TSCDataFrame.from_shift_matrices(
            X_left, X_right, snapshot_orientation="col"
        )

        self.assertEqual(X_tsc.delta_time, 1)

    def test_time_delta04(self):
        # there is a special (faster) routine to take the delta_time for
        # isinstance(n_timesteps, int)

        idx = pd.MultiIndex.from_product([np.arange(10), np.arange(10)])
        X_tsc = TSCDataFrame(np.reshape(np.arange(1000), (100, 10)), index=idx)

        self.assertEqual(X_tsc.delta_time, 1)

    def test_time_delta05(self):
        # there is a special (faster) routine to take the delta_time for
        # isinstance(n_timesteps, int)

        # uneven sampling in the second time series should lead to nan
        idx = pd.MultiIndex.from_arrays(
            [np.array([0, 0, 0, 1, 1, 1]), np.array([1, 2, 3, 1, 5, 99])]
        )
        X_tsc = TSCDataFrame(np.reshape(np.arange(12), (6, 2)), index=idx)

        actual = X_tsc.delta_time
        self.assertIsInstance(actual, pd.Series)
        self.assertEqual(actual.iloc[0], 1)
        self.assertTrue(np.isnan(actual.iloc[1]))

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
        actual = TSCDataFrame(self.simple_df).const_sampled_time_values()
        expected = np.arange(0, 19 + 1, 1)
        nptest.assert_equal(actual, expected)

        simple_df = self.simple_df.copy()
        simple_df.loc[pd.IndexSlice[45, 100], :] = [
            5,
            4,
        ]  # include non-const time delta
        # ... should raise error
        with self.assertRaises(TSCException):
            TSCDataFrame(simple_df).const_sampled_time_values()

    def test_single_time_df(self):
        tsc = TSCDataFrame(self.simple_df)
        actual = tsc.select_time_values(time_values=0)

        # cannot be a TSC anymore, because only one time point does not make a time series
        self.assertIsInstance(actual, pd.DataFrame)

        expected = self.simple_df.iloc[[0, 2, 4], :]
        pdtest.assert_frame_equal(actual, expected, check_flags=False)

        with self.assertRaises(KeyError):
            tsc.select_time_values(time_values=100)  # no time series with this time

        actual = tsc.select_time_values(time_values=17)

        # in 'key' use a list to enforce a data frame, not a series
        expected = self.simple_df.iloc[[6], :]
        pdtest.assert_frame_equal(actual, expected, check_flags=False)

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

        pdtest.assert_frame_equal(actual, expected, check_flags=False)

    def test_multi_time_tsc4(self):
        # behaviour if not all time points are present
        ts = TSCDataFrame(self.simple_df)
        actual = ts.select_time_values(time_values=np.array([0, 1]))

        expected_time = (0, 1)
        expected_ids = (0, 1, 15)

        self.assertTrue(np.in1d(actual.time_values(), expected_time).all())
        self.assertTrue(np.in1d(actual.ids, expected_ids).all())

        with self.assertRaises(KeyError):
            # from pandas>=2.0 an illegal key (here 500) raises a KeyError
            ts.select_time_values(time_values=np.array([0, 1, 500]))

    def test_loc_slice01(self):
        # get time series with ID = 0 --> is not a TSCDataFrame anymore, because the ID
        # is missing.
        ts = TSCDataFrame(self.simple_df).loc[0, :]  # does not fail

        self.assertFalse(isinstance(ts, TSCDataFrame))
        self.assertIsInstance(ts, pd.DataFrame)

    def test_loc_slice02(self):
        tscdf = TSCDataFrame(self.simple_df)
        idx = pd.IndexSlice
        tscdf_sliced = tscdf.loc[idx[:, 0], :]

        self.assertIsInstance(tscdf_sliced, TSCDataFrame)

    def test_loc_slice03(self):
        tscdf = TSCDataFrame(self.simple_df)
        idx = pd.IndexSlice

        actual = tscdf.loc[idx[:, 17], :]

        # after slicing for a single time, it is not a valid TSCDataFrame anymore,
        # therefore fall back to pd.DataFrame
        self.assertIsInstance(actual, TSCDataFrame)

        self.assertTrue(actual.has_degenerate())
        nptest.assert_array_equal(actual.degenerate_ids(), np.array([45]))

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
        #  See also gitlab issue #61
        #  i.e. also a single feature column is a DataFrame. This cold be changed in
        #  future to be closer to the pandas data structures...
        expected_a = pd.DataFrame(self.simple_df.loc[:, "A"])
        expected_a.columns.name = TSCDataFrame.tsc_feature_col_name

        expected_b = pd.DataFrame(self.simple_df.loc[:, "B"])
        expected_b.columns.name = TSCDataFrame.tsc_feature_col_name

        pdtest.assert_frame_equal(actual_a, expected_a, check_flags=False)
        pdtest.assert_frame_equal(actual_b, expected_b, check_flags=False)

    def test_loc_slice07(self):
        df = self.simple_df.copy()

        # index is float64 but can be converted to int64 without loss
        df.index = df.index.set_levels(df.index.levels[0].astype(float), level=0)

        # index must be the same
        pdtest.assert_frame_equal(TSCDataFrame(df), TSCDataFrame(self.simple_df))

    def test_loc_slice08(self):
        df = self.simple_df.copy()

        # index is float64 and cannot be converted to int64 without loss
        df.index = df.index.set_levels(df.index.levels[0].astype(float) + 0.01, level=0)

        with self.assertRaises(AttributeError):
            TSCDataFrame(df)

    def test_loc_slice09(self):
        tsc = TSCDataFrame(self.simple_df)
        actual = tsc.loc[(0, 0), "A"]

        self.assertIsInstance(actual, pd.Series)

    def test_iloc_slice0(self):
        tsc = TSCDataFrame(self.simple_df)

        actual = tsc.iloc[:, 0]
        self.assertIsInstance(actual, pd.Series)
        self.assertIsInstance(TSCDataFrame(actual), TSCDataFrame)

    def test_iloc_slice1(self):
        tsc = TSCDataFrame(self.simple_df)

        actual = tsc.iloc[0, :]
        self.assertIsInstance(actual, pd.Series)

        with self.assertRaises(AttributeError):
            # a single row slice cannot be transformed to TSCDataFrame, because the
            # row-indices are not kept in the Series
            TSCDataFrame(actual)

    def test_iloc_slice2(self):
        tsc = TSCDataFrame(self.simple_df)
        actual = tsc.iloc[0, 0]
        self.assertIsInstance(actual, float)

    def test_iloc_sclice3(self):
        # test for bug reported in gitlab issue
        # https://gitlab.com/datafold-dev/datafold/-/issues/148

        df_list = [pd.DataFrame(data=[0], index=[0])]
        df_list += [pd.DataFrame(data=range(10), index=range(10)) for i in range(2)]
        df_list.append(pd.DataFrame(data=[0], index=[0]))

        tsc_full = TSCDataFrame.from_frame_list(df_list)

        test_one = tsc_full.iloc[[2, 4, 6, 13, 15, 18]]
        self.assertIsInstance(test_one, TSCDataFrame)

        test_two = tsc_full.iloc[[2, 4, 6, 13, 14, 18]]
        self.assertIsInstance(test_two, TSCDataFrame)

        with self.assertRaises(pd.errors.DuplicateLabelError):
            tsc_full.iloc[[1, 1]]

    def test_slice01(self):
        tsc = TSCDataFrame(self.simple_df)
        actual = tsc["A"]
        self.assertIsInstance(actual, TSCDataFrame)

    def test_slice02(self):
        tsc = TSCDataFrame(self.simple_df)
        bool_idx = np.ones(tsc.shape[0], dtype=bool)
        bool_idx[-3:] = False

        actual = tsc[bool_idx]

        self.assertIsInstance(actual, TSCDataFrame)

    def test_slice03(self):
        tsc = TSCDataFrame(self.simple_df)
        actual = tsc[tsc < 0.5]

        self.assertIsInstance(actual, TSCDataFrame)

    def test_set_invalid_index(self):
        tsc = TSCDataFrame(self.simple_df)

        with self.assertRaises(AttributeError):
            tsc.index = pd.Index(np.arange(tsc.shape[0]))

        with self.assertRaises(AttributeError):
            tsc.set_index(np.arange(tsc.shape[0]))

        with self.assertRaises(AttributeError):
            tsc.index = pd.Index(np.zeros(tsc.shape[0]))

        with self.assertRaises(AttributeError):
            tsc.set_index(np.zeros(tsc.shape[0]))

    def test_set_invalid_columns(self):
        tsc = TSCDataFrame(self.simple_df)
        with self.assertRaises(AttributeError):
            tsc.columns = pd.MultiIndex.from_product([[0], [0, 1]])

        with self.assertRaises(AttributeError):
            tsc.columns = pd.Index(np.zeros(tsc.shape[1]))

    def test_insert_invalid_dtype_row(self):
        tsc = TSCDataFrame(self.simple_df)

        insert = pd.Series(["a", "b"], index=tsc.columns, name=(999, 0))

        with self.assertRaises(AttributeError):
            tsc.concat(insert, axis=1)

        with self.assertRaises(AttributeError):
            pd.concat([tsc, insert], axis=0)

    def test_insert_invalid_dtype_data(self):
        with self.assertRaises(AttributeError):
            invalid_tsc = self.simple_df.copy()
            invalid_tsc.iloc[0, 0] = "a"
            TSCDataFrame(invalid_tsc)

        tsc = TSCDataFrame(self.simple_df)

        with self.assertRaises(AttributeError):
            tsc.iloc[(0, 0), 0] = "a"

        with self.assertRaises(AttributeError):
            tsc.iloc[0, :] = np.array([0, "not_numeric"])

        with self.assertRaises(AttributeError):
            tsc.loc[pd.IndexSlice[0, :], :] = pd.Series([0, "not_numeric"])

        with self.assertRaises(AttributeError):
            tsc.loc[(0, 0), "A"] = "a"

        with self.assertRaises(AttributeError):
            tsc.loc[pd.IndexSlice[0, :], :] = np.array([0, "not_numeric"])

        with self.assertRaises(AttributeError):
            tsc.loc[pd.IndexSlice[0, :], :] = pd.Series([0, "not_numeric"])

        # check that no attempt to change the data was sucessful (raise AttributeError
        # before changing the data)
        nptest.assert_equal(tsc.to_numpy(), self.simple_df.to_numpy())

    def test_insert_naninf_data(self):
        tsc = TSCDataFrame(self.simple_df)

        # allow nan and inf in tsc
        tsc.iloc[0, 0] = np.nan
        tsc.iloc[0, 1] = np.inf
        nptest.assert_equal(tsc.iloc[0, :].to_numpy(), np.array([np.nan, np.inf]))
        self.assertFalse(tsc.is_finite())

        tsc.loc[(0, 1), "A"] = np.nan
        tsc.loc[(0, 1), "B"] = np.inf
        nptest.assert_equal(tsc.loc[(0, 1), :].to_numpy(), np.array([np.nan, np.inf]))
        self.assertFalse(tsc.is_finite())

        tsc.loc[pd.IndexSlice[0, 0], :] = np.array([0, 1])
        nptest.assert_equal(tsc.loc[pd.IndexSlice[0, 0], :], np.array([0, 1]))

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
        pdtest.assert_frame_equal(
            actual, expected, check_names=False, check_flags=False
        )

    def test_getitem(self):
        tsc = TSCDataFrame(self.simple_df)

        actual = tsc["A"]
        self.assertIsInstance(actual, TSCDataFrame)
        pdtest.assert_frame_equal(actual, TSCDataFrame(self.simple_df["A"]))

    def test_set_false_row(self):
        tsc = TSCDataFrame(self.simple_df)

        with self.assertRaises(AttributeError):
            # for the second level (after 100) is now a whitespace '' because the time is
            # not specified this leads the tsc.index.dtype go to object (due to a
            # string) and is therefore not numeric anymore
            tsc.loc[100, :] = 1

    def test_set_new_timeseries(self):
        tsc = TSCDataFrame(self.simple_df)

        tsc.loc[(100, 0), :] = 1

        self.assertTrue(100 in tsc.ids)
        nptest.assert_array_equal(tsc.degenerate_ids(), np.array([100]))

        tsc.loc[(100, 1), :] = 2

        self.assertEqual(tsc.n_timesteps.loc[100], 2)
        self.assertEqual(tsc.degenerate_ids(), None)

    def test_concat_new_timeseries(self):
        tsc = TSCDataFrame(self.simple_df)
        new_tsc = pd.DataFrame(
            self.rng.uniform(size=(2, 2)),
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
        new_ts = pd.DataFrame(
            self.rng.uniform(size=(2, 2)), index=[0, 1], columns=["A", "B"]
        )
        tsc = tsc.insert_ts(df=new_ts)
        self.assertTrue(isinstance(tsc, TSCDataFrame))

    def test_insert_timeseries02(self):
        tsc = TSCDataFrame(self.simple_df)

        # Column is not present
        new_ts = pd.DataFrame(
            self.rng.uniform(size=(2, 2)), index=[0, 1], columns=["A", "NA"]
        )

        with self.assertRaises(ValueError):
            tsc.insert_ts(new_ts)

    def test_insert_timeseries03(self):
        tsc = TSCDataFrame(self.simple_df)
        new_ts = pd.DataFrame(
            self.rng.uniform(size=(2, 2)), index=[0, 1], columns=["A", "B"]
        )

        with self.assertRaises(ValueError):
            tsc.insert_ts(new_ts, 1.5)  # id has to be int

        with self.assertRaises(ValueError):
            tsc.insert_ts(new_ts, 1)  # id=1 already present

    def test_insert_timeseries04(self):
        tsc = TSCDataFrame(self.simple_df)

        # Not unique time points -> invalid
        new_ts = pd.DataFrame(
            self.rng.uniform(size=(2, 2)), index=[0, 0], columns=["A", "B"]
        )

        with self.assertRaises(pd.errors.DuplicateLabelError):
            tsc.insert_ts(new_ts, None)

    def test_delta_time_degenerate_timeseries(self):
        tsc = TSCDataFrame(self.simple_df.iloc[1:, :])
        actual = tsc.delta_time
        expected = pd.Series(
            [np.nan, 1, 1, 1.0], index=tsc.ids, name="delta_time", dtype=float
        )
        pdtest.assert_series_equal(actual, expected)

    def test_linspace_unique_delta_times(self):
        # The problem is that np.linspace(...) is often not equally spaced numerically
        # this function tests the tolerances set in the delta_time attribute.

        for n_samples in [10, 1000, 2000, 3000, 3300, 3400, 3500, 10000]:
            for stop in [0.01, 0.1, 10, 40, 50, 100, 1000, 10000, 100000000000]:
                # putting higher order of n_samples (next 100000) fails the pipeline -- for
                # these cases ajdusted tolerances would be necessary

                # useful to set a specific setting for debugging:
                # n_samples = 100000
                # stop = 10

                time_values, delta = np.linspace(0, stop, n_samples, retstep=True)

                tsc_single = TSCDataFrame(
                    self.rng.uniform(size=(time_values.shape[0], 2)),
                    index=pd.MultiIndex.from_product([[0], time_values]),
                )

                idx_two = pd.MultiIndex.from_arrays(
                    [np.append(np.zeros(5), np.ones(n_samples - 5)), time_values]
                )
                tsc_two = TSCDataFrame(
                    self.rng.uniform(size=(time_values.shape[0], 2)),
                    index=idx_two,
                )

                actual_delta_single = tsc_single.delta_time
                actual_delta_two = tsc_two.delta_time

                isinstance(actual_delta_single, float)
                isinstance(actual_delta_two, float)

                # print(f"{n_samples=} {stop=}")
                self.assertFalse(np.isnan(actual_delta_single))
                self.assertFalse(np.isnan(actual_delta_two))

                # Check that when increasing the distance in one sample minimally,
                # then the delta_time is not constant anymore
                time_values[-1] = time_values[-1] + delta * 1e-9
                tsc_single = TSCDataFrame(
                    self.rng.uniform(size=(time_values.shape[0], 2)),
                    index=pd.MultiIndex.from_product([[0], time_values]),
                )

                idx_two = pd.MultiIndex.from_arrays(
                    [np.append(np.zeros(5), np.ones(n_samples - 5)), time_values]
                )
                tsc_two = TSCDataFrame(
                    self.rng.uniform(size=(time_values.shape[0], 2)),
                    index=idx_two,
                )

                actual_delta_single = tsc_single.delta_time
                self.assertTrue(np.isnan(actual_delta_single))

                actual_delta_two = tsc_two.delta_time
                self.assertIsInstance(actual_delta_two, pd.Series)
                self.assertTrue(tsc_two.delta_time.iloc[0] - delta < 1e-15)
                self.assertTrue(np.isnan(tsc_two.delta_time.iloc[1]))

    def test_build_from_single_timeseries(self):
        df = pd.DataFrame(self.rng.random(5), index=np.arange(5, 0, -1), columns=["A"])
        tsc = TSCDataFrame.from_single_timeseries(df)

        self.assertIsInstance(tsc, TSCDataFrame)

    def test_time_not_disappear_initial_state(self):
        """One observation was that a feature-column named 'time' disappears because the
        index is set to a regular column. This is tested here, such a 'time'
        feature-column does not disappear.
        """
        tsc = TSCDataFrame(self.simple_df)
        tsc[TSCDataFrame.tsc_time_idx_name] = 1

        initial_states = tsc.initial_states()
        self.assertTrue(TSCDataFrame.tsc_time_idx_name in initial_states.columns)

    def test_column_dtypes(self):
        d = np.array([[1, 2], [3, 4]])

        tsc_int = TSCDataFrame.from_single_timeseries(pd.DataFrame(d, columns=[1, 2]))
        tsc_str = TSCDataFrame.from_single_timeseries(
            pd.DataFrame(d, columns=["1", "2"])
        )

        def _get_col_types(cols):
            types = list({type(v) for v in cols})
            return types[0] if len(types) == 1 else types

        self.assertEqual(_get_col_types(tsc_int.columns), int)
        self.assertEqual(_get_col_types(tsc_str.columns), str)

        with self.assertRaises(AttributeError):
            # mixed dtypes are not supported
            TSCDataFrame.from_single_timeseries(pd.DataFrame(d, columns=[1, "2"]))

        with self.assertRaises(AttributeError):
            # explicitly setting a mixed index also raises error
            tsc_int.columns = pd.Index([1, "2"])

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
            self.simple_df.index.levels[1].astype(float), level=1
        )

        self.assertIsInstance(TSCDataFrame(simple_df), TSCDataFrame)

    def test_datetime_time_indices(self):
        simple_df = self.simple_df.copy(deep=True)

        # set datetime index
        dates = pd.to_datetime(
            "2019-11-" + (self.simple_df.index.levels[1] + 1).astype(str).str.zfill(2),
            format="%Y-%m-%d",
        )
        simple_df.index = simple_df.index.set_levels(dates, level=1)

        actual = TSCDataFrame(simple_df)

        self.assertIsInstance(actual.delta_time, np.timedelta64)
        self.assertEqual(actual.delta_time, np.timedelta64(1, "D"))
        self.assertTrue(actual.is_datetime_index())

        new_ts_wo_datetime_index = pd.DataFrame(
            self.rng.random((2, 2)), columns=simple_df.columns
        )
        with self.assertRaises(ValueError):
            actual.insert_ts(new_ts_wo_datetime_index)

    def test_fixed_delta01(self):
        tscdf = TSCDataFrame(
            np.zeros((3, 2)),
            index=pd.MultiIndex.from_product([[0], [1, 2, 3]]),
            columns=["a", "b"],
            fixed_delta=0.1,
        )

        actual = tscdf.__repr__(with_fixed_delta=False)
        for expected_num_str in ["0.1", "0.2", "0.3"]:
            self.assertIn(expected_num_str, actual)

        self.assertEqual(tscdf.index.get_level_values("time").dtype, int)

        actual = tscdf.__repr__(with_fixed_delta=False)
        for expected_num_str in ["1", "2", "3"]:
            self.assertIn(expected_num_str, actual)

    def test_fixed_delta02(self):
        tscdf = TSCDataFrame(
            np.arange(6).reshape(3, 2),
            index=pd.MultiIndex.from_product([[0], [1, 2, 3]]),
            fixed_delta=0.1,
        )

        expected_dt = np.array([0.1, 0.2, 0.3])
        expected_int = np.array([1, 2, 3])

        actual_time_values_dt = tscdf.time_values(with_fixed_delta=True)
        actual_time_values_int = tscdf.time_values(with_fixed_delta=False)

        nptest.assert_allclose(
            actual_time_values_dt, expected_dt, rtol=1e-16, atol=1e-16
        )
        nptest.assert_equal(actual_time_values_int, expected_int)

    def test_fixed_delta03(self):
        tscdf = TSCDataFrame(self.simple_df, fixed_delta=1.0)
        self.assertEqual(tscdf.delta_time, 1.0)
        self.assertEqual(tscdf.fixed_delta, 1.0)
        self.assertTrue(tscdf.time_values().dtype == float)

    def test_fixed_delta04(self):
        tscdf = TSCDataFrame(self.simple_df, fixed_delta=1.0)

        tscdf_attach = TSCDataFrame.from_array(
            np.zeros((3, 2)),
            time_values=[99, 100, 101],
            feature_names=tscdf.columns,
            ts_id=101,
        )

        actual = pd.concat([tscdf, tscdf_attach], axis=0)

        self.assertIn(101, actual.ids)
        # should not propagate for newly constructed TSCDataFrame
        self.assertIsNone(actual.fixed_delta)


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
        actual = InitialCondition.from_array(
            np.array([1, 2, 3]), time_value=0, feature_names=["A", "B", "C"]
        )

        expected = TSCDataFrame(
            np.array([[1, 2, 3]]),  # note it is 2D
            index=pd.MultiIndex.from_arrays(
                [[0], [0]],
                names=[TSCDataFrame.tsc_id_idx_name, TSCDataFrame.tsc_time_idx_name],
            ),
            columns=pd.Index(["A", "B", "C"], name=TSCDataFrame.tsc_feature_col_name),
        )

        self.assertTrue(InitialCondition.validate(actual))
        pdtest.assert_frame_equal(actual, expected)

    def test_from_array02(self):
        actual = InitialCondition.from_array(
            np.array([[1, 2, 3], [4, 5, 6]]),
            time_value=0.0,
            feature_names=["A", "B", "C"],
        )

        expected = TSCDataFrame(
            np.array([[1, 2, 3], [4, 5, 6]]),
            index=pd.MultiIndex.from_arrays(
                [[0, 1], [0.0, 0.0]],
                names=[TSCDataFrame.tsc_id_idx_name, TSCDataFrame.tsc_time_idx_name],
            ),
            columns=pd.Index(["A", "B", "C"], name=TSCDataFrame.tsc_feature_col_name),
        )

        self.assertTrue(InitialCondition.validate(actual))
        pdtest.assert_frame_equal(actual, expected)

    def test_from_array03(self):
        actual = InitialCondition.from_array(
            np.array([[1, 2, 3], [4, 5, 6]]),
            time_value=0.0,
            feature_names=["A", "B", "C"],
            ts_ids=np.array([55, 99]),  # insert ts_ids
        )

        expected = TSCDataFrame(
            np.array([[1, 2, 3], [4, 5, 6]]),
            index=pd.MultiIndex.from_arrays(
                [[55, 99], [0.0, 0.0]],
                names=[TSCDataFrame.tsc_id_idx_name, TSCDataFrame.tsc_time_idx_name],
            ),
            columns=pd.Index(["A", "B", "C"], name=TSCDataFrame.tsc_feature_col_name),
        )

        self.assertTrue(InitialCondition.validate(actual))
        pdtest.assert_frame_equal(actual, expected)

    def test_from_tsc01(self):
        actual = InitialCondition.from_tsc(self.test_tsc01, n_samples_ic=1)
        expected = TSCDataFrame(self.test_tsc01).head(1)

        self.assertTrue(InitialCondition.validate(actual, n_samples_ic=1))
        pdtest.assert_frame_equal(actual, expected)

    def test_from_tsc02(self):
        actual = InitialCondition.from_tsc(self.test_tsc01, n_samples_ic=3)
        expected = pd.DataFrame(self.test_tsc01).head(3)

        self.assertTrue(InitialCondition.validate(actual))
        pdtest.assert_frame_equal(actual, expected, check_flags=False)

    def test_iter_reconstruct_ic01(self):
        # test if it can handle a single time series

        n_samples_ic = 1

        for i, (actual_ic, actual_time_values) in enumerate(
            InitialCondition.iter_reconstruct_ic(self.test_tsc01, n_samples_ic=1)
        ):
            select_ts = pd.DataFrame(self.test_tsc01).loc[[i], :]
            expected_ic = select_ts.head(n_samples_ic)
            expected_time_values = select_ts.index.get_level_values(
                TSCDataFrame.tsc_time_idx_name
            )

            self.assertTrue(InitialCondition.validate(actual_ic))
            pdtest.assert_frame_equal(actual_ic, expected_ic, check_flags=False)
            nptest.assert_array_equal(actual_time_values, expected_time_values)

    def test_iter_reconstruct_ic02(self):
        # test if it can handle a multiple time series

        n_samples_ic = 1

        for i, (actual_ic, actual_time_values) in enumerate(
            InitialCondition.iter_reconstruct_ic(self.test_tsc02, n_samples_ic=1)
        ):
            select_ts = pd.DataFrame(self.test_tsc02).loc[[i], :]
            expected_ic = select_ts.head(n_samples_ic)

            expected_time_values = select_ts.index.get_level_values(
                TSCDataFrame.tsc_time_idx_name
            )

            self.assertTrue(InitialCondition.validate(actual_ic))
            pdtest.assert_frame_equal(actual_ic, expected_ic, check_flags=False)
            nptest.assert_array_equal(actual_time_values, expected_time_values)

    def test_iter_reconstruct_ic03(self):
        # test

        n_sample_ic = 3

        for i, (actual_ic, actual_time_values) in enumerate(
            InitialCondition.iter_reconstruct_ic(self.test_tsc02, n_samples_ic=3)
        ):
            select_ts = self.test_tsc02.loc[[i], :]
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
