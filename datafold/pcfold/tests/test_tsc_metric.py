#!/usr/bin/env python3
import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest
from sklearn.metrics import mean_squared_error

from datafold.pcfold.timeseries import TSCDataFrame
from datafold.pcfold.timeseries.metric import (
    TSCKfoldSeries,
    TSCKFoldTime,
    TSCMetric,
    make_tsc_scorer,
)


class TestTSCMetric(unittest.TestCase):
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

        # "left" and "right" refer to "y_true" and "y_pred" to measure the metric
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
        # "left" and "right" refer to "y_true" and "y_pred" to measure the metric
        self.tsc_two_left = self.tsc_two_left.insert_ts(self._create_multi_qoi_ts(5))
        self.tsc_two_left = self.tsc_two_left.insert_ts(self._create_multi_qoi_ts(20))

        self.tsc_two_right = TSCDataFrame.from_same_indices_as(
            self.tsc_two_left, values=np.random.rand(*self.tsc_two_left.shape)
        )

    def test_metrics_without_error(self):

        # simply test of any of the configuration fails

        for metric in TSCMetric.VALID_METRIC:
            for mode in TSCMetric.VALID_MODE:
                for scale in TSCMetric.VALID_SCALING:
                    for multi_qoi in ["uniform_average", "raw_values"]:

                        tsc_metric = TSCMetric(metric=metric, mode=mode, scaling=scale,)

                        try:
                            if metric != "max":  # max does not support multi-output
                                tsc_metric.eval_metric(
                                    self.tsc_one_left,
                                    self.tsc_one_right,
                                    multi_qoi=multi_qoi,
                                )
                                tsc_metric.eval_metric(
                                    self.tsc_two_left,
                                    self.tsc_two_right,
                                    multi_qoi=multi_qoi,
                                )
                            else:
                                with self.assertRaises(ValueError):
                                    tsc_metric.eval_metric(
                                        self.tsc_one_left,
                                        self.tsc_one_right,
                                        multi_qoi=multi_qoi,
                                    )
                                    tsc_metric.eval_metric(
                                        self.tsc_two_left,
                                        self.tsc_two_right,
                                        multi_qoi=multi_qoi,
                                    )
                        except Exception as e:
                            print(
                                f"metric={metric}, mode={mode}, scale={scale}, "
                                f"mulit_qoi={multi_qoi} failed"
                            )
                            raise e

        # Test to not fail

    def test_error_per_timeseries1(self):
        multi_output = "uniform_average"
        actual = TSCMetric(metric="mse", mode="timeseries").eval_metric(
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

        actual = TSCMetric(metric="mse", mode="timeseries").eval_metric(
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

        actual = TSCMetric(metric="mse", mode="timeseries").eval_metric(
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
        actual = TSCMetric(metric="mse", mode="qoi").eval_metric(
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

        actual = TSCMetric(metric="mse", mode="qoi").eval_metric(
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
        actual = TSCMetric(metric="mse", mode="qoi").eval_metric(
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
        actual = TSCMetric(metric="mse", mode="timestep").eval_metric(
            self.tsc_one_left, self.tsc_one_right, multi_qoi=multi_qoi
        )

        self.assertIsInstance(actual, pd.Series)

        idx_slice = pd.IndexSlice
        for t in self.tsc_one_left.time_values(unique_values=True):

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

        actual = TSCMetric(metric="mse", mode="timestep").eval_metric(
            self.tsc_one_left,
            self.tsc_one_right,
            sample_weight=sample_weight,
            multi_qoi=multi_qoi,
        )

        self.assertIsInstance(actual, pd.Series)

        idx_slice = pd.IndexSlice
        for t in self.tsc_one_left.time_values(unique_values=True):

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
        actual = TSCMetric(metric="mse", mode="timestep").eval_metric(
            self.tsc_two_left, self.tsc_two_right, multi_qoi=multi_qoi
        )

        self.assertIsInstance(actual, pd.Series)

        idx_slice = pd.IndexSlice
        for t in self.tsc_two_left.time_values(unique_values=True):

            nptest.assert_array_equal(
                mean_squared_error(
                    self.tsc_two_left.loc[idx_slice[:, t], :],
                    self.tsc_two_right.loc[idx_slice[:, t], :],
                    sample_weight=None,
                    multioutput=multi_qoi,
                ),
                actual.loc[t],
            )

    def test_tsc_scorer(self):
        _metric_callable_actual = TSCMetric.make_tsc_metric(
            metric="rmse", mode="qoi", scaling="id"
        )
        _metric_callable_expected = TSCMetric(
            metric="rmse", mode="qoi", scaling="id"
        ).eval_metric

        pdtest.assert_series_equal(
            _metric_callable_expected(self.tsc_one_left, self.tsc_one_left),
            _metric_callable_actual(self.tsc_one_left, self.tsc_one_left),
        )

    def test_qoi_uniform_avrg_score(self):

        _metric = TSCMetric.make_tsc_metric(metric="rmse", mode="qoi", scaling="id")
        _score = make_tsc_scorer(_metric)
        _score_actual = _score(self.tsc_one_left, self.tsc_one_right)

        _score_expected = _metric(self.tsc_one_left, self.tsc_one_right)
        _score_expected = float(_score_expected.mean())

        self.assertEqual(-1 * _score_expected, _score_actual)

    def test_qoi_weighted_avrg_score(self):

        sample_weight = np.array([1, 2, 3])
        _metric = TSCMetric.make_tsc_metric(metric="rmse", mode="qoi", scaling="id")
        _score = make_tsc_scorer(_metric)
        _score_actual = _score(
            self.tsc_one_left, self.tsc_one_right, sample_weight=sample_weight
        )

        _score_expected = _metric(self.tsc_one_left, self.tsc_one_right)
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
            [[0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6, 7],]
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

        from sklearn.model_selection import check_cv
        from datafold.appfold.edmd import EDMD

        # NOTE: this is an import from internal module _split
        #  -- there is no guarantee for backwards compatibility and there is no
        #  deprecation cycle
        from sklearn.model_selection._validation import is_classifier

        self.assertFalse(is_classifier(EDMD))

        check_cv(TSCKFoldTime, y=None, classifier=is_classifier(EDMD))
        check_cv(TSCKfoldSeries, y=None, classifier=is_classifier(EDMD))

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
            self.assertEquals(train_part.lengths_time_series, 4)
            self.assertEquals(test_part.lengths_time_series, 4)

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
            self.assertEqual(train_part.lengths_time_series, 2)
            self.assertEqual(test_part.lengths_time_series, 2)

            # this tests that there was no shuffle, all time series should still be
            # connected
            self.assertTrue(train_part.is_const_dt())
            self.assertTrue(test_part.is_const_dt())

            nptest.assert_array_equal(train_part.ids, test_part.ids)

    def test_kfold_time_single_id_tsc(self):

        # time series is 8 long, so there should be always 4 samples in each split
        n_splits = 2

        for train, test in TSCKFoldTime(n_splits).split(self.single_id_tsc):
            # print(f"train{train} {self.single_id_tsc.iloc[train, :]}")
            # print(f"test{train} {self.single_id_tsc.iloc[test, :]}")

            train_part: TSCDataFrame = self.single_id_tsc.iloc[train, :]
            test_part: TSCDataFrame = self.single_id_tsc.iloc[test, :]

            self.assertIsInstance(train_part, TSCDataFrame)
            self.assertIsInstance(test_part, TSCDataFrame)

            self.assertEqual(train_part.n_timeseries, 1)
            self.assertEqual(test_part.n_timeseries, 1)

            # this tests that there was no shuffle, all time series should still be
            # connected
            self.assertTrue(train_part.is_const_dt())
            self.assertTrue(test_part.is_const_dt())

            nptest.assert_array_equal(train_part.ids, test_part.ids)

    def test_kfold_time_single_id_tsc2(self):

        n_splits = 4

        for train, test in TSCKFoldTime(n_splits=n_splits).split(X=self.single_id_tsc):

            train, test = self.single_id_tsc.tsc.kfold_cv_reassign_ids(train, test)

            # print(f"train {train}")
            # print(f"test {test}")

            self.assertIsInstance(train, TSCDataFrame)
            self.assertIsInstance(test, TSCDataFrame)

            self.assertTrue(train.is_const_dt())
            self.assertTrue(test.is_const_dt())

            self.assertIn(len(train.ids), (1, 2))
            self.assertIn(len(test.ids), (1, 2))

            self.assertFalse(np.in1d(train.ids, test.ids).any())

    def test_kfold_time_two_id_tsc(self):
        n_splits = 4

        for train, test in TSCKFoldTime(n_splits=n_splits).split(X=self.two_id_tsc):
            train, test = self.two_id_tsc.tsc.kfold_cv_reassign_ids(train, test)

            print(f"train {train}")
            print(f"test {test}")

            self.assertIsInstance(train, TSCDataFrame)
            self.assertIsInstance(test, TSCDataFrame)

            self.assertTrue(train.is_const_dt())
            self.assertTrue(test.is_const_dt())

            self.assertIn(len(train.ids), (2, 3, 4))
            self.assertIn(len(test.ids), (2, 3, 4))

            self.assertFalse(np.in1d(train.ids, test.ids).any())
