#!/usr/bin/env python3

import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from datafold.pcfold.timeseries import TSCDataFrame
from datafold.pcfold.timeseries.transform import (
    TSCIdentity,
    TSCPrincipalComponent,
    TSCQoiPreprocess,
    TSCTakensEmbedding,
    TSCTransformerMixIn,
)


# TODO: write a check_tsc_transform() ?


def _all_tsc_transformers():
    # only finds the ones that are importated (DMAP e.g. is not here)
    print(TSCTransformerMixIn.__subclasses__())


class TestTSCTransform(unittest.TestCase):
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

    def test_is_valid_sklearn_estimator(self):
        from sklearn.utils.estimator_checks import check_estimator
        from sklearn.preprocessing import MinMaxScaler

        TEST_ESTIMATORS = (
            TSCIdentity,
            TSCPrincipalComponent,
            TSCQoiPreprocess(MinMaxScaler()),
            TSCQoiPreprocess(StandardScaler()),
        )

        for test_estimator in TEST_ESTIMATORS:
            for estimator, check in check_estimator(test_estimator, generate_only=True):
                try:
                    check(estimator)
                except Exception as e:
                    print(estimator)
                    print(check)
                    raise e

    def test_identity(self):
        tsc = TSCDataFrame(self.simple_df)

        _id = TSCIdentity()
        pdtest.assert_frame_equal(_id.fit_transform(tsc), tsc)
        pdtest.assert_frame_equal(_id.inverse_transform(tsc), tsc)

    def test_scale_min_max(self):
        tsc_df = TSCDataFrame(self.simple_df)

        scale = TSCQoiPreprocess.scale("min-max")
        scaled_tsc = scale.fit_transform(tsc_df)

        # sanity check:
        nptest.assert_allclose(scaled_tsc.min().to_numpy(), np.zeros(2), atol=1e-16)
        nptest.assert_allclose(scaled_tsc.max().to_numpy(), np.ones(2), atol=1e-16)

        # Undoing normalization must give original TSCDataFrame back
        pdtest.assert_frame_equal(tsc_df, scale.inverse_transform(scaled_tsc))

    def test_scale_standard(self):

        tsc_df = TSCDataFrame(self.simple_df)

        scale = TSCQoiPreprocess.scale("standard")
        scaled_tsc = scale.fit_transform(tsc_df)

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
            scale = TSCQoiPreprocess(sklearn_transformer=cls(**kwargs))
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
            TSCQoiPreprocess(sklearn_transformer=Normalizer)

    def test_pca_transform(self):

        tsc = TSCDataFrame(self.simple_df)
        pca = TSCPrincipalComponent(n_components=1).fit(tsc)

        data = pca.transform(tsc)
        self.assertIsInstance(data, TSCDataFrame)

        pca_sklearn = PCA(n_components=1).fit(tsc.to_numpy())
        data_sklearn = pca_sklearn.transform(tsc)

        nptest.assert_allclose(data, data_sklearn, atol=1e-15)

        nptest.assert_array_equal(
            pca.inverse_transform(data).to_numpy(),
            pca_sklearn.inverse_transform(data_sklearn),
        )

    def test_takens_embedding_fillnan(self):
        simple_df = self.takens_df.drop("B", axis=1)
        tsc_df = TSCDataFrame(simple_df)

        # using class
        actual = TSCTakensEmbedding(
            lag=0,
            delays=1,
            frequency=1,
            time_direction="backward",
            fillin_handle=np.nan,
        ).fit_transform(tsc_df)
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

    def test_takens_embedding_fillremove(self):
        simple_df = self.takens_df.drop("B", axis=1)
        tsc_df = TSCDataFrame(simple_df)

        # using class
        actual = TSCTakensEmbedding(
            lag=0,
            delays=1,
            frequency=1,
            time_direction="backward",
            fillin_handle="remove",
        ).fit_transform(tsc_df)

        self.assertIsInstance(actual, pd.DataFrame)

        actual = actual.values  # only compare the numeric values now

        expected = np.array(
            [[2.0, 0.0], [6.0, 4.0], [10.0, 8.0], [14.0, 12.0], [16.0, 14.0],]
        )

        nptest.assert_equal(actual, expected)

    def test_takens_delay_indices(self):
        tsc = TSCDataFrame(self.takens_df)

        nptest.assert_array_equal(
            TSCTakensEmbedding(lag=0, delays=1, frequency=1).fit(tsc).delay_indices_,
            np.array([1]),
        )

        nptest.assert_array_equal(
            TSCTakensEmbedding(lag=0, delays=2, frequency=1).fit(tsc).delay_indices_,
            np.array([1, 2]),
        )

        nptest.assert_array_equal(
            TSCTakensEmbedding(lag=0, delays=5, frequency=1).fit(tsc).delay_indices_,
            np.array([1, 2, 3, 4, 5]),
        )

        nptest.assert_array_equal(
            TSCTakensEmbedding(lag=1, delays=1, frequency=1).fit(tsc).delay_indices_,
            np.array([2]),
        )

        nptest.assert_array_equal(
            TSCTakensEmbedding(lag=1, delays=5, frequency=1).fit(tsc).delay_indices_,
            np.array([2, 3, 4, 5, 6]),
        )

        nptest.assert_array_equal(
            TSCTakensEmbedding(lag=2, delays=2, frequency=2).fit(tsc).delay_indices_,
            np.array([3, 5]),
        )

        nptest.assert_array_equal(
            TSCTakensEmbedding(lag=2, delays=4, frequency=2).fit(tsc).delay_indices_,
            np.array([3, 5, 7, 9]),
        )

        with self.assertRaises(ValueError):
            TSCTakensEmbedding(lag=0, delays=1, frequency=2).fit(tsc)
