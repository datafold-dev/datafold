#!/usr/bin/env python3

import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest

from datafold.pcfold.timeseries import TSCDataFrame
from datafold.pcfold.timeseries.transform import TSCQoiTransform


class TestTSCQoiTransform(unittest.TestCase):
    def setUp(self) -> None:
        # The last two elements are used
        idx = pd.MultiIndex.from_arrays(
            [[0, 0, 1, 1, 15, 15, 45, 45, 45], [0, 1, 0, 1, 0, 1, 17, 18, 19]]
        )
        col = ["A", "B"]
        self.simple_df = pd.DataFrame(np.random.rand(9, 2), index=idx, columns=col)

    def test_normalize_min_max(self):
        tsc_df = TSCDataFrame(self.simple_df)

        scale = TSCQoiTransform.from_name("min-max")
        scaled_tsc = scale.fit_transform(tsc_df)

        # sanity check:
        nptest.assert_allclose(scaled_tsc.min().to_numpy(), np.zeros(2), atol=1e-16)
        nptest.assert_allclose(scaled_tsc.max().to_numpy(), np.ones(2), atol=1e-16)

        # Undoing normalization must give original TSCDataFrame back
        pdtest.assert_frame_equal(tsc_df, scale.inverse_transform(scaled_tsc))

    def test_normalize_standard(self):

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

            # check the underlying array equals:
            nptest.assert_array_equal(
                cls(**kwargs).fit_transform(tsc_df.to_numpy()),
                tsc_transformed.to_numpy(),
            )

            # check inverse transform works:
            pdtest.assert_frame_equal(tsc_df, scale.inverse_transform(tsc_transformed))

    def test_error_wo_inverse(self):
        from sklearn.preprocessing import Normalizer

        with self.assertRaises(AttributeError):
            # Normalizer has no inverse_transform
            TSCQoiTransform(cls=Normalizer)
