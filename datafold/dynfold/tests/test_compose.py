#!/usr/bin/env python3

import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest
from sklearn.pipeline import Pipeline

from datafold.dynfold import (
    TSCPolynomialFeatures,
    TSCPrincipalComponent,
    TSCTakensEmbedding,
)
from datafold.dynfold.compose import TSCColumnTransformer, TSCFeatureUnion
from datafold.pcfold import TSCDataFrame


class TestTSCTransform(unittest.TestCase):
    @unittest.skip("Currently TSCFeatureUnion is not implemented")
    def test_tsc_union(self):

        X = TSCDataFrame.from_single_timeseries(
            pd.DataFrame(np.random.default_rng(1).uniform(low=0, high=1, size=(100, 2)))
        )

        transform1 = TSCPrincipalComponent(n_components=2)
        transform2 = TSCPolynomialFeatures(degree=2)

        union = TSCFeatureUnion(
            [("pca", transform1), ("poly", transform2)], n_jobs=1,
        ).fit(X)

        actual = union.transform(X)

        print(actual)

    def test_pipeline(self):
        X = TSCDataFrame.from_single_timeseries(
            pd.DataFrame(
                np.random.default_rng(1).uniform(low=0, high=1, size=(100, 2)),
                columns=["a", "b"],
            )
        )

        # NOTE: there is no final estimator set
        actual_transform = Pipeline(
            steps=[
                ("pca", TSCPrincipalComponent(n_components=2)),
                ("poly", TSCPolynomialFeatures(degree=2)),
            ]
        )
        actual_result = actual_transform.fit_transform(X)

        expected_index = pd.Index(["pca0^2", "pca0 pca1", "pca1^2"], name="features")
        nptest.assert_array_equal(expected_index, actual_result.columns)

    def test_pipeline_incl_timedrop(self):
        X = TSCDataFrame.from_single_timeseries(
            pd.DataFrame(
                np.random.default_rng(1).uniform(low=0, high=1, size=(100, 2)),
                columns=["a", "b"],
            )
        )

        # NOTE: there is no final estimator set
        actual_transform = Pipeline(
            steps=[
                ("pca", TSCPrincipalComponent(n_components=1)),
                ("poly", TSCTakensEmbedding(delays=2)),
            ]
        )
        actual_result = actual_transform.fit_transform(X)

        expected_index = pd.Index(["pca0", "pca0:d1", "pca0:d2"], name="features")
        nptest.assert_array_equal(expected_index, actual_result.columns)

        self.assertEquals(actual_result.n_timesteps, X.n_timesteps - 2)

    def test_column_transform(self):

        X = TSCDataFrame.from_single_timeseries(
            pd.DataFrame(
                np.random.default_rng(1).uniform(low=0, high=1, size=(100, 2)),
                columns=["a", "b"],
            )
        )

        transform1 = TSCPrincipalComponent(n_components=2)
        transform2 = TSCPolynomialFeatures(degree=2)

        actual_transform = TSCColumnTransformer(
            [("pca", transform1, X.columns), ("poly", transform2, X.columns)]
        )
        actual_result = actual_transform.fit_transform(X)

        expected_col_index = pd.Index(
            ["pca0", "pca1", "a^2", "a b", "b^2"], name="features"
        )

        self.assertEqual(actual_transform.n_features_out_, 2 + 3)
        nptest.assert_array_equal(
            expected_col_index, actual_transform.feature_names_out_
        )

        nptest.assert_array_equal(expected_col_index, actual_result.columns)

        self.assertEqual(actual_transform.n_features_in_, X.n_features)
        nptest.assert_array_equal(actual_transform.feature_names_in_, X.columns)

    def test_column_transform_incl_timedrop(self):
        # tests of TSCColumnTransformer can handle dictionary elements that drop samples

        X = TSCDataFrame.from_single_timeseries(
            pd.DataFrame(
                np.random.default_rng(1).uniform(low=0, high=1, size=(100, 2)),
                columns=["a", "b"],
            )
        )

        transform1 = TSCTakensEmbedding(delays=2)
        transform2 = TSCPolynomialFeatures(degree=2)

        actual_transform = TSCColumnTransformer(
            [("pca", transform1, X.columns), ("poly", transform2, X.columns)]
        )
        actual_result = actual_transform.fit_transform(X)

        expected_col_index = pd.Index(
            ["a", "b", "a:d1", "b:d1", "a:d2", "b:d2", "a^2", "a b", "b^2"],
            name="features",
        )

        self.assertEqual(actual_transform.n_features_out_, 9)
        nptest.assert_array_equal(
            expected_col_index, actual_transform.feature_names_out_
        )

        nptest.assert_array_equal(expected_col_index, actual_result.columns)
        self.assertFalse(np.isnan(actual_result.to_numpy()).any())

        # -2 for the dropped samples from the takens embedding
        self.assertTrue(actual_result.n_timesteps, X.n_timesteps - 2)

    def test_pipeline_with_column_transform(self, display_html=False):
        from sklearn.utils import estimator_html_repr

        X = TSCDataFrame.from_single_timeseries(
            pd.DataFrame(
                np.random.default_rng(1).uniform(low=0, high=1, size=(100, 2)),
                columns=["a", "b"],
            )
        )

        column_transform = TSCColumnTransformer(
            [
                ("pca", TSCTakensEmbedding(delays=2), X.columns),
                ("poly", TSCPolynomialFeatures(degree=2), X.columns),
            ]
        )

        actual_transform = Pipeline(
            steps=[
                ("col", column_transform),
                ("pca", TSCPrincipalComponent(n_components=2)),
            ]
        )

        actual_result = actual_transform.fit_transform(X)

        if display_html:
            import webbrowser
            import tempfile

            with tempfile.NamedTemporaryFile("w", suffix=".html") as fp:
                fp.write(estimator_html_repr(actual_transform))
                fp.flush()
                webbrowser.open_new_tab(fp.name)
                input("Press Enter to continue...")

        expected_col_index = pd.Index(["pca0", "pca1"], name="features")

        nptest.assert_array_equal(expected_col_index, actual_result.columns)
        self.assertFalse(np.isnan(actual_result.to_numpy()).any())

        # -2 for the dropped samples from the takens embedding
        self.assertTrue(actual_result.n_timesteps, X.n_timesteps - 2)

    def test_complicated_pipeline_with_pipeline_transform(self):
        # TODO.
        #  Start with FeatureScaling
        #     1.way Poly  2.way PCA
        #     1.way PCA   2.way Poly
        #     combine the two ways and perform embedding
        self.assertFalse(True)
