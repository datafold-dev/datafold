#!/usr/bin/env python3
import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
from sklearn.metrics import mean_squared_error

from datafold.pcfold.timeseries import TSCDataFrame
from datafold.pcfold.timeseries.metric import TSCMetric


class TestTscMetric(unittest.TestCase):
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

        # simply test of any of the configuration fails

        for metric in TSCMetric.VALID_METRIC:
            for mode in TSCMetric.VALID_MODE:
                for scale in TSCMetric.VALID_SCALING:
                    for multi_qoi in ["uniform_average", "raw_values"]:

                        tsc_metric = TSCMetric(metric=metric, mode=mode, scaling=scale,)

                        try:
                            if metric != "max":  # max does not support multi-output
                                tsc_metric.score(
                                    self.tsc_one_left,
                                    self.tsc_one_right,
                                    multi_qoi=multi_qoi,
                                )
                                tsc_metric.score(
                                    self.tsc_two_left,
                                    self.tsc_two_right,
                                    multi_qoi=multi_qoi,
                                )
                            else:
                                with self.assertRaises(ValueError):
                                    tsc_metric.score(
                                        self.tsc_one_left,
                                        self.tsc_one_right,
                                        multi_qoi=multi_qoi,
                                    )
                                    tsc_metric.score(
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
        actual = TSCMetric(metric="mse", mode="timeseries").score(
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

        actual = TSCMetric(metric="mse", mode="timeseries").score(
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

        actual = TSCMetric(metric="mse", mode="timeseries").score(
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
        actual = TSCMetric(metric="mse", mode="qoi").score(
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

        actual = TSCMetric(metric="mse", mode="qoi").score(
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
        actual = TSCMetric(metric="mse", mode="qoi").score(
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
        actual = TSCMetric(metric="mse", mode="timestep").score(
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

        actual = TSCMetric(metric="mse", mode="timestep").score(
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
        actual = TSCMetric(metric="mse", mode="timestep").score(
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
