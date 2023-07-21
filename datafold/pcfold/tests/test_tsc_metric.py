#!/usr/bin/env python3
import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest
from sklearn.metrics import mean_squared_error

from datafold.pcfold import (
    TSCDataFrame,
    TSCKfoldSeries,
    TSCKFoldTime,
    TSCMetric,
    TSCScoring,
    TSCWindowFoldTime,
)
from datafold.pcfold.timeseries.collection import TSCException


class TestTSCMetric(unittest.TestCase):
    rng = np.random.default_rng(5)

    def setUp(self):
        self._create_tsc_one()
        self._create_tsc_two()

    def _create_multi_feature_timeseries(self, nr_timesteps=10):
        time_index = np.arange(nr_timesteps)
        columns = np.array(["qoi_A", "qoi_B", "qoi_C"])
        data = self.rng.uniform(size=(time_index.shape[0], columns.shape[0]))

        return pd.DataFrame(data, time_index, columns)

    def _create_tsc_one(self):
        self.tsc_ytrue = TSCDataFrame.from_single_timeseries(
            df=self._create_multi_feature_timeseries(10)
        )

        self.tsc_ytrue = self.tsc_ytrue.insert_ts(
            self._create_multi_feature_timeseries(10)
        )
        self.tsc_ytrue = self.tsc_ytrue.insert_ts(
            self._create_multi_feature_timeseries(10)
        )

        self.tsc_ypred = TSCDataFrame.from_same_indices_as(
            self.tsc_ytrue, values=self.rng.uniform(size=self.tsc_ytrue.shape)
        )

    def _create_tsc_two(self):
        self.tsc_ytrue2 = TSCDataFrame.from_single_timeseries(
            df=self._create_multi_feature_timeseries(10)
        )

        # NOTE: here they have different length!
        # "left" and "right" refer to "y_true" and "y_pred" to measure the metric
        self.tsc_ytrue2 = self.tsc_ytrue2.insert_ts(
            self._create_multi_feature_timeseries(5)
        )
        self.tsc_ytrue2 = self.tsc_ytrue2.insert_ts(
            self._create_multi_feature_timeseries(20)
        )

        self.tsc_ypred2 = TSCDataFrame.from_same_indices_as(
            self.tsc_ytrue2, values=self.rng.uniform(size=self.tsc_ytrue2.shape)
        )

    def test_metrics_without_error(self):
        # simply test if any of the possible configurations fails

        for metric in TSCMetric._cls_valid_metrics:
            for mode in TSCMetric._cls_valid_modes:
                for scale in TSCMetric._cls_valid_scaling:
                    for multioutput in ["uniform_average", "raw_values"]:
                        tsc_metric = TSCMetric(
                            metric=metric,
                            mode=mode,
                            scaling=scale,
                        )

                        try:
                            if metric != "max":  # max does not support multi-output
                                tsc_metric(
                                    self.tsc_ytrue,
                                    self.tsc_ypred,
                                    multioutput=multioutput,
                                )
                                if not (
                                    metric == "rrmse"
                                    and mode == "timestep"
                                    and scale == "min-max"
                                    and multioutput == "raw_values"
                                ):
                                    # rrmse fails if min-max scaling leads to a zero in a
                                    # sample for which there is only a single sample in the
                                    # time series collection
                                    tsc_metric(
                                        self.tsc_ytrue2,
                                        self.tsc_ypred2,
                                        multioutput=multioutput,
                                    )
                            else:
                                with self.assertRaises(ValueError):
                                    tsc_metric(
                                        self.tsc_ytrue,
                                        self.tsc_ypred,
                                        multioutput=multioutput,
                                    )
                                    tsc_metric(
                                        self.tsc_ytrue2,
                                        self.tsc_ypred2,
                                        multioutput=multioutput,
                                    )
                        except Exception as e:
                            print(
                                f"metric={metric}, mode={mode}, scale={scale}, "
                                f"multioutput={multioutput} failed"
                            )
                            raise e

        # Test to not fail

    def test_error_per_timeseries1(self):
        multioutput = "uniform_average"
        actual = TSCMetric(metric="mse", mode="timeseries")(
            self.tsc_ytrue, self.tsc_ypred, multioutput=multioutput
        )

        self.assertIsInstance(actual, pd.Series)

        idx = pd.IndexSlice

        for id_ in self.tsc_ytrue.ids:
            expected_val = mean_squared_error(
                self.tsc_ytrue.loc[idx[id_, :], :],
                self.tsc_ypred.loc[idx[id_, :], :],
                sample_weight=None,
                multioutput="uniform_average",
            )

            self.assertEqual(expected_val, actual.loc[id_])

    def test_error_per_timeseries2(self):
        multi_output = "raw_values"

        actual = TSCMetric(metric="mse", mode="timeseries")(
            self.tsc_ytrue, self.tsc_ypred, multioutput=multi_output
        )

        self.assertIsInstance(actual, pd.DataFrame)

        idx = pd.IndexSlice

        for id_ in self.tsc_ytrue.ids:
            expected_val = mean_squared_error(
                self.tsc_ytrue.loc[idx[id_, :], :],
                self.tsc_ypred.loc[idx[id_, :], :],
                sample_weight=None,
                multioutput=multi_output,
            )

            nptest.assert_array_equal(expected_val, actual.loc[id_].to_numpy())

    def test_error_per_timeseries3(self):
        # With different length TSC

        multi_output = "raw_values"

        actual = TSCMetric(metric="mse", mode="timeseries")(
            self.tsc_ytrue2, self.tsc_ypred2, multioutput=multi_output
        )

        self.assertIsInstance(actual, pd.DataFrame)

        idx = pd.IndexSlice

        for id_ in self.tsc_ytrue2.ids:
            expected_val = mean_squared_error(
                self.tsc_ytrue2.loc[idx[id_, :], :],
                self.tsc_ypred2.loc[idx[id_, :], :],
                sample_weight=None,
                multioutput=multi_output,
            )

            nptest.assert_array_equal(expected_val, actual.loc[id_].to_numpy())

    def test_error_per_feature1(self):
        sample_weight = np.ones(self.tsc_ytrue.shape[0])
        actual = TSCMetric(metric="mse", mode="feature")(
            self.tsc_ytrue, self.tsc_ypred, sample_weight=sample_weight
        )

        self.assertIsInstance(actual, pd.Series)

        nptest.assert_array_equal(
            mean_squared_error(
                self.tsc_ytrue, self.tsc_ypred, multioutput="raw_values"
            ),
            actual.to_numpy(),
        )

    def test_error_per_feature2(self):
        sample_weight = np.zeros(self.tsc_ytrue.shape[0])
        sample_weight[0] = 1  # put whole weight on a single sample

        actual = TSCMetric(metric="mse", mode="feature")(
            self.tsc_ytrue, self.tsc_ypred, sample_weight=sample_weight
        )

        self.assertIsInstance(actual, pd.Series)

        nptest.assert_array_equal(
            mean_squared_error(
                self.tsc_ytrue,
                self.tsc_ypred,
                sample_weight=sample_weight,
                multioutput="raw_values",
            ),
            actual.to_numpy(),
        )

    def test_error_per_feature3(self):
        sample_weight = np.ones(self.tsc_ytrue2.shape[0])
        actual = TSCMetric(metric="mse", mode="feature")(
            self.tsc_ytrue2, self.tsc_ypred2, sample_weight=sample_weight
        )

        self.assertIsInstance(actual, pd.Series)

        nptest.assert_array_equal(
            mean_squared_error(
                self.tsc_ytrue2, self.tsc_ypred2, multioutput="raw_values"
            ),
            actual.to_numpy(),
        )

    def test_error_per_timestep1(self):
        multioutput = "uniform_average"
        actual = TSCMetric(metric="mse", mode="timestep")(
            self.tsc_ytrue, self.tsc_ypred, multioutput=multioutput
        )

        self.assertIsInstance(actual, pd.Series)

        idx_slice = pd.IndexSlice
        for t in self.tsc_ytrue.time_values():
            nptest.assert_array_equal(
                mean_squared_error(
                    self.tsc_ytrue.loc[idx_slice[:, t], :],
                    self.tsc_ypred.loc[idx_slice[:, t], :],
                    sample_weight=None,
                    multioutput=multioutput,
                ),
                actual.loc[t],
            )

    def test_error_per_timestep2(self):
        multioutput = np.array([0.5, 0.5, 1])  # user defined weighing
        sample_weight = np.arange(
            len(self.tsc_ytrue.ids)
        )  # increasing weight for each time step (three)

        actual = TSCMetric(metric="mse", mode="timestep")(
            self.tsc_ytrue,
            self.tsc_ypred,
            sample_weight=sample_weight,
            multioutput=multioutput,
        )

        self.assertIsInstance(actual, pd.Series)

        idx_slice = pd.IndexSlice
        for t in self.tsc_ytrue.time_values():
            nptest.assert_array_equal(
                mean_squared_error(
                    self.tsc_ytrue.loc[idx_slice[:, t], :],
                    self.tsc_ypred.loc[idx_slice[:, t], :],
                    sample_weight=sample_weight,
                    multioutput=multioutput,
                ),
                actual.loc[t],
            )

    def test_error_per_timestep3(self):
        # For tsc_two

        multioutput = "uniform_average"
        actual = TSCMetric(metric="mse", mode="timestep")(
            self.tsc_ytrue2, self.tsc_ypred2, multioutput=multioutput
        )

        self.assertIsInstance(actual, pd.Series)

        idx_slice = pd.IndexSlice
        for t in self.tsc_ytrue2.time_values():
            nptest.assert_array_equal(
                mean_squared_error(
                    self.tsc_ytrue2.loc[idx_slice[:, t], :],
                    self.tsc_ypred2.loc[idx_slice[:, t], :],
                    sample_weight=None,
                    multioutput=multioutput,
                ),
                actual.loc[t],
            )

    def test_tsc_scorer(self):
        _metric_callable_actual = TSCMetric(metric="rmse", mode="feature", scaling="id")
        _metric_callable_expected = TSCMetric(
            metric="rmse", mode="feature", scaling="id"
        )

        pdtest.assert_series_equal(
            _metric_callable_expected(self.tsc_ytrue, self.tsc_ytrue),
            _metric_callable_actual(self.tsc_ytrue, self.tsc_ytrue),
        )

    def test_feature_uniform_avrg_score(self):
        _metric = TSCMetric(metric="rmse", mode="feature", scaling="id")
        _score = TSCScoring(_metric)
        _score_actual = _score(self.tsc_ytrue, self.tsc_ypred)

        _score_expected = _metric(self.tsc_ytrue, self.tsc_ypred)
        _score_expected = float(_score_expected.mean())

        self.assertEqual(-1 * _score_expected, _score_actual)

    def test_feature_weighted_avrg_score(self):
        sample_weight = np.array([1, 2, 3])
        _metric = TSCMetric(metric="rmse", mode="feature", scaling="id")
        _score = TSCScoring(_metric)
        _score_actual = _score(
            self.tsc_ytrue, self.tsc_ypred, sample_weight=sample_weight
        )

        _score_expected = _metric(self.tsc_ytrue, self.tsc_ypred)
        _score_expected = float(np.average(_score_expected, weights=sample_weight))

        self.assertEqual(-1 * _score_expected, _score_actual)


class TestTSCCV(unittest.TestCase):
    def _simple_tsc(self):
        idx = pd.MultiIndex.from_arrays(
            [
                [0, 0, 0, 0, 1, 1, 1, 1, 15, 15, 15, 15, 45, 45, 45, 45],
                [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            ]
        )
        col = ["A", "B"]
        data = np.arange(len(idx) * 2).reshape([len(idx), 2])
        self.simple_tsc = TSCDataFrame(data, index=idx, columns=col)

    def _single_id_tsc(self):
        idx = pd.MultiIndex.from_arrays(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 2, 3, 4, 5, 6, 7],
            ]
        )
        col = ["A", "B"]
        data = np.arange(len(idx) * 2).reshape([len(idx), 2])
        single_ts = TSCDataFrame(data, index=idx, columns=col)

        self.single_id_tsc = TSCDataFrame(single_ts)

    def _two_id_tsc(self):
        idx = pd.MultiIndex.from_arrays(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7],
            ]
        )
        col = ["A", "B"]
        data = np.arange(len(idx) * 2).reshape([len(idx), 2])
        single_ts = TSCDataFrame(data, index=idx, columns=col)

        self.two_id_tsc = TSCDataFrame(single_ts)

    def setUp(self) -> None:
        self._simple_tsc()
        self._single_id_tsc()
        self._two_id_tsc()

    def test_sklearn_check_cv(self):
        # NOTE: this is an import from internal module _split
        #  -- there is no guarantee for backwards compatibility and there is no
        #  deprecation cycle
        from sklearn.model_selection import check_cv
        from sklearn.model_selection._validation import is_classifier

        from datafold.appfold.edmd import EDMD

        check_cv(TSCKFoldTime, y=self.single_id_tsc, classifier=is_classifier(EDMD))
        check_cv(TSCKfoldSeries, y=self.single_id_tsc, classifier=is_classifier(EDMD))

    def test_kfold_series_simple_tsc(self):
        # there are 4 time series, so a 2-split should always contain 2 time series
        n_splits = 2

        for train, test in TSCKfoldSeries(n_splits).split(self.simple_tsc):
            # print(f"train {train} {self.simple_tsc.iloc[train, :]}")
            # print(f"test {test} {self.simple_tsc.iloc[test, :]}")

            train_part = self.simple_tsc.iloc[train, :]
            test_part = self.simple_tsc.iloc[test, :]

            self.assertIsInstance(train_part, TSCDataFrame)
            self.assertIsInstance(test_part, TSCDataFrame)

            self.assertEqual(train_part.n_timeseries, 2)
            self.assertEqual(test_part.n_timeseries, 2)

            # should keep original length:
            self.assertEqual(train_part.n_timesteps, 4)
            self.assertEqual(test_part.n_timesteps, 4)

            # checks that no time series id is in train and also test
            self.assertFalse(np.in1d(train_part.ids, test_part.ids).any())

    def test_kfold_series_single_id_tsc(self):
        with self.assertRaises(ValueError):
            # error is raised when trying to iterate it (also in sklearn)
            for _, _ in TSCKfoldSeries(2).split(self.single_id_tsc):
                pass

    def test_kfold_time_simple_tsc(self):
        # time series have all a length of 4, a n_splits should result into
        n_splits = 2

        for train, test in TSCKFoldTime(n_splits).split(self.simple_tsc):
            train_part: TSCDataFrame = self.simple_tsc.iloc[train, :]
            test_part: TSCDataFrame = self.simple_tsc.iloc[test, :]

            self.assertIsInstance(train_part, TSCDataFrame)
            self.assertIsInstance(test_part, TSCDataFrame)

            # all time series are still present
            self.assertEqual(train_part.n_timeseries, 4)
            self.assertEqual(test_part.n_timeseries, 4)

            # originally all time series are of length 4, now they should be 2
            self.assertEqual(train_part.n_timesteps, 2)
            self.assertEqual(test_part.n_timesteps, 2)

            # this tests that there was no shuffle, all time series should still be
            # connected
            self.assertTrue(train_part.is_const_delta_time())
            self.assertTrue(test_part.is_const_delta_time())

            nptest.assert_array_equal(train_part.ids, test_part.ids)

    def test_kfold_time_single_id_tsc(self):
        # time series is 8 long, so there should be always 4 samples in each split
        n_splits = 2

        for train, test in TSCKFoldTime(n_splits).split(self.single_id_tsc):
            train_part: TSCDataFrame = self.single_id_tsc.iloc[train, :]
            test_part: TSCDataFrame = self.single_id_tsc.iloc[test, :]

            self.assertIsInstance(train_part, TSCDataFrame)
            self.assertIsInstance(test_part, TSCDataFrame)

            self.assertEqual(train_part.n_timeseries, 1)
            self.assertEqual(test_part.n_timeseries, 1)

            # this tests that there was no shuffle, all time series should still be
            # connected
            self.assertTrue(train_part.is_const_delta_time())
            self.assertTrue(test_part.is_const_delta_time())

            nptest.assert_array_equal(train_part.ids, test_part.ids)

    def test_kfold_time_single_id_tsc2(self):
        n_splits = 4

        for train_idx, test_idx in TSCKFoldTime(n_splits=n_splits).split(
            X=self.single_id_tsc
        ):
            actual_train, actual_test = self.single_id_tsc.tsc.assign_ids_train_test(
                train_idx, test_idx
            )

            self.assertIsInstance(actual_train, TSCDataFrame)
            self.assertIsInstance(actual_test, TSCDataFrame)

            self.assertTrue(actual_train.is_const_delta_time())
            self.assertTrue(actual_test.is_const_delta_time())

            self.assertIn(len(actual_train.ids), (1, 2))
            self.assertIn(len(actual_test.ids), (1, 2))

            self.assertFalse(np.in1d(actual_train.ids, actual_test.ids).any())

    def test_kfold_time_two_id_tsc(self):
        n_splits = 4

        for train_idx, test_idx in TSCKFoldTime(n_splits=n_splits).split(
            X=self.two_id_tsc
        ):
            actual_train, actual_test = self.two_id_tsc.tsc.assign_ids_train_test(
                train_idx, test_idx
            )

            self.assertIsInstance(actual_train, TSCDataFrame)
            self.assertIsInstance(actual_test, TSCDataFrame)

            self.assertTrue(actual_train.is_const_delta_time())
            self.assertTrue(actual_test.is_const_delta_time())

            self.assertIn(len(actual_train.ids), (2, 3, 4))
            self.assertIn(len(actual_test.ids), (2, 3, 4))

            self.assertFalse(np.in1d(actual_train.ids, actual_test.ids).any())

    def test_window_split1(self):
        df1 = pd.DataFrame(np.arange(10).reshape(5, 2), index=np.arange(5))
        df2 = pd.DataFrame(np.arange(10).reshape(5, 2), index=np.arange(10, 15))

        X = TSCDataFrame.from_frame_list([df1, df2])

        for i, (train, test) in enumerate(
            TSCWindowFoldTime(test_window_length=3, train_min_timesteps=3).split(X)
        ):
            if i == 0:
                nptest.assert_array_equal(test, np.array([7, 8, 9]))
                nptest.assert_array_equal(train, np.array([0, 1, 2, 3, 4]))
            elif i == 1:
                nptest.assert_array_equal(test, np.array([2, 3, 4]))
                nptest.assert_array_equal(train, np.array([5, 6, 7, 8, 9]))
            else:
                raise AssertionError()

            self.assertTrue((~np.isin(train, test)).all())

        self.assertEqual(TSCWindowFoldTime(test_window_length=3).get_n_splits(X), 2)

    def test_window_split2(self):
        df1 = pd.DataFrame(np.arange(10).reshape(5, 2), index=np.arange(5))
        df2 = pd.DataFrame(np.arange(4).reshape(2, 2), index=np.arange(10, 12))

        X = TSCDataFrame.from_frame_list([df1, df2])

        for i, (train, test) in enumerate(
            TSCWindowFoldTime(test_window_length=2, train_min_timesteps=None).split(X)
        ):
            if i == 0:
                nptest.assert_array_equal(test, np.array([5, 6]))
                nptest.assert_array_equal(train, np.array([0, 1, 2, 3, 4]))
            elif i == 1:
                nptest.assert_array_equal(test, np.array([3, 4]))
                nptest.assert_array_equal(train, np.array([0, 1, 2, 5, 6]))
            elif i == 2:
                nptest.assert_array_equal(test, np.array([1, 2]))
                nptest.assert_array_equal(train, np.array([0, 3, 4, 5, 6]))
            else:
                raise AssertionError()

            self.assertTrue((~np.isin(train, test)).all())
        self.assertEqual(TSCWindowFoldTime(test_window_length=2).get_n_splits(X), 3)

    def test_window_split3(self):
        # test that empty training samples are skipped (if the samples [2,3] are in
        # test, then the train_min_timesteps is not fulfilled).

        df = pd.DataFrame(np.arange(12).reshape(6, 2), index=np.arange(6))
        X = TSCDataFrame.from_single_timeseries(df)

        for i, (train, test) in enumerate(
            TSCWindowFoldTime(test_window_length=2, train_min_timesteps=3).split(X)
        ):
            if i == 0:
                nptest.assert_array_equal(test, np.array([4, 5]))
                nptest.assert_array_equal(train, np.array([0, 1, 2, 3]))
            elif i == 1:
                nptest.assert_array_equal(test, np.array([0, 1]))
                nptest.assert_array_equal(train, np.array([2, 3, 4, 5]))
            else:
                raise AssertionError()

            self.assertTrue((~np.isin(train, test)).all())
        self.assertEqual(TSCWindowFoldTime(test_window_length=2).get_n_splits(X), 3)

    def test_window_split4(self):
        df1 = pd.DataFrame(np.arange(10).reshape(5, 2), index=np.arange(5))
        df2 = pd.DataFrame(np.arange(10).reshape(5, 2), index=np.arange(10, 15))
        X = TSCDataFrame.from_frame_list([df1, df2])

        with self.assertRaises(ValueError):
            list(TSCWindowFoldTime(test_window_length=6).split(X))

        df1 = pd.DataFrame(np.arange(10).reshape(5, 2), index=np.arange(5))
        df2 = df1.copy()
        X = TSCDataFrame.from_frame_list([df1, df2])

        with self.assertRaises(TSCException):
            list(TSCWindowFoldTime(test_window_length=2).split(X))

    def test_window_split5(self, plot=False):
        # Simply check if an error is raised for plotting

        df1 = pd.DataFrame(np.arange(10).reshape(5, 2), index=np.arange(5))
        df2 = pd.DataFrame(np.arange(10).reshape(5, 2), index=np.arange(10, 15))
        X = TSCDataFrame.from_frame_list([df1, df2])

        df_test = pd.DataFrame(np.arange(10).reshape(5, 2), index=np.arange(15, 20))
        X_test = TSCDataFrame.from_single_timeseries(df_test)

        # test if it errors
        TSCWindowFoldTime(test_window_length=2).plot_splits(X)
        TSCWindowFoldTime(test_window_length=2).plot_splits(X, X_test)
        self.assertTrue(True)

        df_test_invalid = pd.DataFrame(
            np.arange(10).reshape(5, 2), index=np.arange(14, 19)
        )
        X_test_invalid = TSCDataFrame.from_single_timeseries(df_test_invalid)

        with self.assertRaises(ValueError):
            # test_set is not completely separated from the data
            TSCWindowFoldTime(test_window_length=2).plot_splits(X, X_test_invalid)

        if plot:
            import matplotlib.pyplot as plt

            plt.show()
