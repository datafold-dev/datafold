#!/usr/bin/env python3

import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest

from datafold.pcfold.timeseries import TSCDataFrame
from datafold.pcfold.timeseries.transform import TSCQoiTransform, TSCTakensEmbedding


class TestTSCQoiTransform(unittest.TestCase):
    def _setUp_simple_df(self):
        idx = pd.MultiIndex.from_arrays(
            [[0, 0, 1, 1, 15, 15, 45, 45, 45], [0, 1, 0, 1, 0, 1, 17, 18, 19]]
        )
        col = ["A", "B"]

        self.simple_df = pd.DataFrame(np.random.rand(9, 2), index=idx, columns=col)

    def _setUp_takens_df(self):

        idx = pd.MultiIndex.from_arrays(
            [[0, 0, 1, 1, 15, 15, 45, 45, 45], [0, 1, 0, 1, 0, 1, 17, 18, 19]]
        )
        col = ["A", "B"]

        # Requires non-random values
        self.takens_df = pd.DataFrame(
            np.arange(18).reshape([9, 2]), index=idx, columns=col
        )

    def setUp(self) -> None:
        self._setUp_simple_df()
        self._setUp_takens_df()

    def test_scale_min_max(self):
        tsc_df = TSCDataFrame(self.simple_df)

        scale = TSCQoiTransform.from_name("min-max")
        scaled_tsc = scale.fit_transform(tsc_df)

        # sanity check:
        nptest.assert_allclose(scaled_tsc.min().to_numpy(), np.zeros(2), atol=1e-16)
        nptest.assert_allclose(scaled_tsc.max().to_numpy(), np.ones(2), atol=1e-16)

        # Undoing normalization must give original TSCDataFrame back
        pdtest.assert_frame_equal(tsc_df, scale.inverse_transform(scaled_tsc))

    def test_scale_standard(self):

        tsc_df = TSCDataFrame(self.simple_df)

        scale = TSCQoiTransform.from_name("standard")
        scaled_tsc = scale.fit_transform(tsc_df)

        from sklearn.preprocessing import StandardScaler

        nptest.assert_array_equal(
            scaled_tsc.to_numpy(),
            StandardScaler(with_mean=True, with_std=True).fit_transform(
                tsc_df.to_numpy()
            ),
        )

        # Undoing normalization must give original TSCDataFrame back
        pdtest.assert_frame_equal(tsc_df, scale.inverse_transform(scaled_tsc))

    def test_sklearn_scaler(self):
        tsc_df = TSCDataFrame(self.simple_df)

        from sklearn.preprocessing import (
            MaxAbsScaler,
            PowerTransformer,
            QuantileTransformer,
            RobustScaler,
        )

        # each tuple has the class and a dictionary with the init-options
        scaler = [
            (MaxAbsScaler, dict()),
            (PowerTransformer, dict(method="yeo-johnson")),
            (PowerTransformer, dict(method="box-cox")),
            (
                QuantileTransformer,
                dict(n_quantiles=tsc_df.shape[0], output_distribution="uniform"),
            ),
            (
                QuantileTransformer,
                dict(n_quantiles=tsc_df.shape[0], output_distribution="normal"),
            ),
            (RobustScaler, dict()),
        ]

        for cls, kwargs in scaler:
            scale = TSCQoiTransform(cls=cls, **kwargs)
            tsc_transformed = scale.fit_transform(tsc_df)

            # Check the underlying array equals:
            nptest.assert_array_equal(
                cls(**kwargs).fit_transform(tsc_df.to_numpy()),
                tsc_transformed.to_numpy(),
            )

            # check inverse transform is equal the original TSCDataFrame:
            pdtest.assert_frame_equal(tsc_df, scale.inverse_transform(tsc_transformed))

    def test_error_wo_inverse(self):
        from sklearn.preprocessing import Normalizer

        with self.assertRaises(AttributeError):
            # Normalizer has no inverse_transform
            TSCQoiTransform(cls=Normalizer)

    def test_takens_embedding(self):
        simple_df = self.takens_df.drop("B", axis=1)

        tc = TSCDataFrame(simple_df)

        # using class
        actual = TSCTakensEmbedding(
            lag=0, delays=1, frequency=1, time_direction="backward"
        ).apply(tc)
        self.assertTrue(isinstance(actual, TSCDataFrame))

        actual = actual.values  # only compare the numeric values now

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

        nptest.assert_equal(actual, expected)

    def test_takens_delay_indices(self):
        nptest.assert_array_equal(
            TSCTakensEmbedding(lag=0, delays=1, frequency=1).delay_indices_,
            np.array([1]),
        )

        nptest.assert_array_equal(
            TSCTakensEmbedding(lag=0, delays=2, frequency=1).delay_indices_,
            np.array([1, 2]),
        )

        nptest.assert_array_equal(
            TSCTakensEmbedding(lag=0, delays=5, frequency=1).delay_indices_,
            np.array([1, 2, 3, 4, 5]),
        )

        nptest.assert_array_equal(
            TSCTakensEmbedding(lag=1, delays=1, frequency=1).delay_indices_,
            np.array([2]),
        )

        nptest.assert_array_equal(
            TSCTakensEmbedding(lag=1, delays=5, frequency=1).delay_indices_,
            np.array([2, 3, 4, 5, 6]),
        )

        nptest.assert_array_equal(
            TSCTakensEmbedding(lag=2, delays=2, frequency=2).delay_indices_,
            np.array([3, 5]),
        )

        nptest.assert_array_equal(
            TSCTakensEmbedding(lag=2, delays=4, frequency=2).delay_indices_,
            np.array([3, 5, 7, 9]),
        )

        with self.assertRaises(ValueError):
            TSCTakensEmbedding(lag=0, delays=1, frequency=2)
