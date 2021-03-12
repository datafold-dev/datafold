#!/usr/bin/env python3

import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from datafold.dynfold.transform import (
    TSCApplyLambdas,
    TSCFeaturePreprocess,
    TSCFiniteDifference,
    TSCIdentity,
    TSCPolynomialFeatures,
    TSCPrincipalComponent,
    TSCRadialBasis,
    TSCTakensEmbedding,
    TSCTransformerMixin,
)
from datafold.pcfold.kernels import *
from datafold.pcfold.timeseries.collection import TSCDataFrame, TSCException


def _all_tsc_transformers():
    # only finds the ones that are importated (DMAP e.g. is not here)
    print(TSCTransformerMixin.__subclasses__())


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
        self.takens_df_short = pd.DataFrame(
            np.arange(18).reshape([9, 2]), index=idx, columns=col
        )

        n_samples_timeseries = 100
        idx = pd.MultiIndex.from_product(
            [np.array([0, 1]), np.arange(n_samples_timeseries)]
        )

        self.takens_df_long = pd.DataFrame(
            np.random.rand(n_samples_timeseries * 2, 2), index=idx, columns=col
        )

    def setUp(self) -> None:
        self._setUp_simple_df()
        self._setUp_takens_df()

    def test_is_valid_sklearn_estimator(self):
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.utils.estimator_checks import check_estimator

        TEST_ESTIMATORS = (
            TSCIdentity(),
            TSCPrincipalComponent(),
            TSCFeaturePreprocess(sklearn_transformer=MinMaxScaler()),
            TSCFeaturePreprocess(sklearn_transformer=StandardScaler()),
            TSCPolynomialFeatures(),
        )

        for test_estimator in TEST_ESTIMATORS:
            for estimator, check in check_estimator(test_estimator, generate_only=True):
                try:
                    check(estimator)
                except Exception as e:
                    print(estimator)
                    print(check)
                    raise e

    def test_identity0(self):
        tsc = TSCDataFrame(self.simple_df)

        _id = TSCIdentity()
        pdtest.assert_frame_equal(_id.fit_transform(tsc), tsc)
        pdtest.assert_frame_equal(_id.inverse_transform(tsc), tsc)

    def test_identity1(self):
        tsc = TSCDataFrame(self.simple_df)

        _id = TSCIdentity(include_const=True)

        tsc_plus_const = tsc.copy(deep=True)
        tsc_plus_const["const"] = 1

        pdtest.assert_frame_equal(_id.fit_transform(tsc.copy()), tsc_plus_const)
        pdtest.assert_frame_equal(_id.inverse_transform(tsc_plus_const), tsc)

    def test_identity2(self):
        data = np.random.rand(5, 5)

        data_wo_const = TSCIdentity(include_const=False).fit_transform(data)
        data_plus_const = TSCIdentity(include_const=True).fit_transform(data)

        nptest.assert_equal(data, data_wo_const)
        nptest.assert_equal(data_plus_const, np.column_stack([data, np.ones(5)]))

    def test_identity3(self):
        data = TSCDataFrame(self.simple_df)

        data_wo_const = TSCIdentity(
            include_const=False, rename_features=True
        ).fit_transform(data)

        data_with_const = TSCIdentity(
            include_const=True, rename_features=True
        ).fit_transform(data)

        data = data.add_suffix("_id")
        pdtest.assert_index_equal(data.columns, data_wo_const.columns)

        data["const"] = 1
        pdtest.assert_index_equal(data.columns, data_with_const.columns)

    def test_scale_min_max(self):
        tsc_df = TSCDataFrame(self.simple_df)

        scale = TSCFeaturePreprocess.from_name("min-max")
        scaled_tsc = scale.fit_transform(tsc_df)

        # sanity check:
        nptest.assert_allclose(scaled_tsc.min().to_numpy(), np.zeros(2), atol=1e-16)
        nptest.assert_allclose(scaled_tsc.max().to_numpy(), np.ones(2), atol=1e-16)

        # Undoing normalization must give original TSCDataFrame back
        pdtest.assert_frame_equal(tsc_df, scale.inverse_transform(scaled_tsc))

    def test_scale_standard(self):

        tsc_df = TSCDataFrame(self.simple_df)

        scale = TSCFeaturePreprocess.from_name("standard")
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
            scale = TSCFeaturePreprocess(sklearn_transformer=cls(**kwargs))
            tsc_transformed = scale.fit_transform(tsc_df)

            # Check the underlying array equals:
            nptest.assert_array_equal(
                cls(**kwargs).fit_transform(tsc_df.to_numpy()),
                tsc_transformed.to_numpy(),
            )

            # check inverse transform is equal the original TSCDataFrame:
            pdtest.assert_frame_equal(tsc_df, scale.inverse_transform(tsc_transformed))

    def test_polynomial_feature_transform01(self):
        from sklearn.preprocessing import PolynomialFeatures

        tsc = TSCDataFrame(self.simple_df)

        for degree in [2, 3, 4]:
            for include_bias in [True, False]:
                actual = TSCPolynomialFeatures(
                    degree=degree, include_bias=include_bias, include_first_order=True
                ).fit_transform(tsc)

                expected = PolynomialFeatures(
                    degree=degree, include_bias=include_bias
                ).fit_transform(tsc.to_numpy())
                nptest.assert_array_equal(actual.to_numpy(), expected)

    def test_polynomial_feature_transform02(self):
        tsc = TSCDataFrame(self.simple_df)

        for include_first_order in [True, False]:
            poly = TSCPolynomialFeatures(
                degree=2, include_bias=True, include_first_order=include_first_order
            ).fit(tsc)
            actual = poly.transform(tsc)

            expected = TSCPolynomialFeatures(
                degree=2, include_bias=True, include_first_order=False
            ).fit_transform(tsc)

        pdtest.assert_frame_equal(actual, expected)

    def test_polynomial_feature_transform03(self):
        tsc = TSCDataFrame(self.simple_df)

        actual = TSCPolynomialFeatures(
            degree=2, include_bias=True, include_first_order=False
        ).fit_transform(tsc)

        pdtest.assert_index_equal(
            actual.columns,
            pd.Index(["1", "A^2", "A B", "B^2"], name="feature"),
        )

        actual = TSCPolynomialFeatures(
            degree=2, include_bias=False, include_first_order=False
        ).fit_transform(tsc)

        pdtest.assert_index_equal(
            actual.columns,
            pd.Index(["A^2", "A B", "B^2"], name="feature"),
        )

    def test_apply_lambda_transform01(self):
        # use lambda identity function

        tsc = TSCDataFrame(self.simple_df)

        lambda_transform = TSCApplyLambdas(lambdas=[lambda x: x]).fit(tsc)

        actual = lambda_transform.transform(tsc)
        expected = tsc
        expected.columns = pd.Index(
            ["A_lambda0", "B_lambda0"], name=TSCDataFrame.tsc_feature_col_name
        )

        pdtest.assert_frame_equal(actual, expected)

    def test_apply_lambda_transform02(self):
        # use numpy function

        tsc = TSCDataFrame(self.simple_df)

        lambda_transform = TSCApplyLambdas(lambdas=[np.square]).fit(tsc)

        actual = lambda_transform.transform(tsc)
        expected = tsc.apply(np.square, axis=0, raw=True)
        expected.columns = pd.Index(
            ["A_lambda0", "B_lambda0"], name=TSCDataFrame.tsc_feature_col_name
        )

        pdtest.assert_frame_equal(actual, expected)

    def test_apply_lambda_transform03(self):
        # use numpy function

        tsc = TSCDataFrame(self.simple_df)

        lambda_transform = TSCApplyLambdas(lambdas=[lambda x: x, np.square]).fit(tsc)

        actual = lambda_transform.transform(tsc)

        identity = tsc
        identity.columns = pd.Index(
            ["A_lambda0", "B_lambda0"], name=TSCDataFrame.tsc_feature_col_name
        )

        squared = tsc.apply(np.square, axis=0, raw=True)
        squared.columns = pd.Index(
            ["A_lambda1", "B_lambda1"], name=TSCDataFrame.tsc_feature_col_name
        )

        expected = pd.concat([identity, squared], axis=1)

        pdtest.assert_frame_equal(actual, expected)

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

    def test_takens_embedding0(self):
        simple_df = self.takens_df_short.drop("B", axis=1)
        tsc_df = TSCDataFrame(simple_df)

        takens = TSCTakensEmbedding(
            delays=1,
            lag=0,
            frequency=1,
        )
        actual = takens.fit_transform(tsc_df)

        self.assertIsInstance(actual, TSCDataFrame)

        # First test
        actual_numerics = actual.to_numpy()  # only compare the numeric values
        expected = np.array(
            [
                [2.0, 0.0],
                [6.0, 4.0],
                [10.0, 8.0],
                [14.0, 12.0],
                [16.0, 14.0],
            ]
        )

        nptest.assert_equal(actual_numerics, expected)

        # Second test
        actual_inverse = takens.inverse_transform(actual)
        pdtest.assert_frame_equal(tsc_df.drop([0, 17], level=1), actual_inverse)

    def test_takens_embedding1(self):
        # test kappa = 1

        tsc_df = TSCDataFrame.from_single_timeseries(
            pd.DataFrame(
                np.column_stack([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]),
                columns=["A", "B"],
                dtype=float,
            )
        )

        takens = TSCTakensEmbedding(lag=0, delays=5, frequency=1, kappa=1)

        # embedd to a single instance
        actual = takens.fit_transform(tsc_df)

        self.assertIsInstance(actual, TSCDataFrame)
        self.assertTrue(actual.has_degenerate())
        self.assertEqual(actual.n_timeseries, 1)

        # First test
        actual_numerics = actual.to_numpy()  # only compare the numeric values

        expected = np.array([[5, 4, 3, 2, 1, 0]], dtype=float) * np.exp(
            -1.0 * np.array([0, 1, 2, 3, 4, 5])
        )
        expected = np.repeat(expected, 2, axis=1)

        nptest.assert_equal(actual_numerics, expected)

        # Second test
        actual_inverse = takens.inverse_transform(actual)
        expected = tsc_df.final_states(1)
        pdtest.assert_frame_equal(actual_inverse, expected)

    def test_takens_delay_indices(self):
        tsc_short = TSCDataFrame(self.takens_df_short)  # better check for errors
        tsc_long = TSCDataFrame(self.takens_df_long)

        nptest.assert_array_equal(
            TSCTakensEmbedding(delays=1, lag=0, frequency=1)
            .fit(tsc_short)
            .delay_indices_,
            np.array([1]),
        )

        nptest.assert_array_equal(
            TSCTakensEmbedding(delays=2, lag=0, frequency=1)
            .fit(tsc_long)
            .delay_indices_,
            np.array([1, 2]),
        )

        with self.assertRaises(TSCException):
            # Data too short
            TSCTakensEmbedding(delays=5, lag=0, frequency=1).fit(tsc_short)

        nptest.assert_array_equal(
            TSCTakensEmbedding(delays=1, lag=1, frequency=1)
            .fit(tsc_long)
            .delay_indices_,
            np.array([2]),
        )

        nptest.assert_array_equal(
            TSCTakensEmbedding(delays=5, lag=1, frequency=1)
            .fit(tsc_long)
            .delay_indices_,
            np.array([2, 3, 4, 5, 6]),
        )

        nptest.assert_array_equal(
            TSCTakensEmbedding(delays=2, lag=2, frequency=2)
            .fit(tsc_long)
            .delay_indices_,
            np.array([3, 5]),
        )

        nptest.assert_array_equal(
            TSCTakensEmbedding(delays=4, lag=2, frequency=2)
            .fit(tsc_long)
            .delay_indices_,
            np.array([3, 5, 7, 9]),
        )

        with self.assertRaises(ValueError):
            TSCTakensEmbedding(delays=1, lag=0, frequency=2).fit(tsc_short)

    def test_rbf_1d(self):
        func = lambda x: np.exp(x * np.cos(3 * np.pi * x)) - 1
        x_vals = np.linspace(0, 1, 100)
        y_vals = func(x_vals)

        df = pd.DataFrame(y_vals, index=x_vals, columns=["qoi"])
        tsc = TSCDataFrame.from_single_timeseries(df)

        rbf = TSCRadialBasis(kernel=MultiquadricKernel())
        rbf_coeff = rbf.fit_transform(tsc)

        rbf_coeff_inverse = rbf.inverse_transform(rbf_coeff)
        pdtest.assert_frame_equal(tsc, rbf_coeff_inverse, check_exact=False)

    def test_rbf_2d(self):
        func = lambda x: np.exp(x * np.cos(3 * np.pi * x)) - 1
        x_vals = np.linspace(0, 1, 15)
        y_vals = func(x_vals)

        df = pd.DataFrame(np.column_stack([x_vals, y_vals]), columns=["qoi1", "qoi2"])

        tsc = TSCDataFrame.from_single_timeseries(df)

        rbf = TSCRadialBasis(kernel=MultiquadricKernel(epsilon=1.0))
        rbf_coeff = rbf.fit_transform(tsc)

        rbf_coeff_inverse = rbf.inverse_transform(rbf_coeff)
        pdtest.assert_frame_equal(tsc, rbf_coeff_inverse, check_exact=False)

    def test_rbf_centers(self):
        func = lambda x: np.exp(x * np.cos(3 * np.pi * x)) - 1
        x_vals = np.linspace(0, 1, 15)
        y_vals = func(x_vals)

        df = pd.DataFrame(np.column_stack([x_vals, y_vals]), columns=["qoi1", "qoi2"])

        tsc = TSCDataFrame.from_single_timeseries(df)

        # Use centers at another location than the data. These can be selected in a
        # optimization routine (such as kmean), or randomly put into the phase space.
        x_vals_centers = np.linspace(0, 1, 10)
        y_vals_centers = func(x_vals_centers)

        centers = np.column_stack([x_vals_centers, y_vals_centers])
        centers = pd.DataFrame(centers, columns=tsc.columns)

        rbf = TSCRadialBasis(kernel=MultiquadricKernel(epsilon=1.0))
        rbf = rbf.fit(centers)

        rbf_coeff = rbf.transform(tsc)

        rbf_coeff_inverse = rbf.inverse_transform(rbf_coeff)

        pdtest.assert_index_equal(tsc.index, rbf_coeff_inverse.index)
        pdtest.assert_index_equal(tsc.columns, rbf_coeff_inverse.columns)
        # can only check against a reference solution:
        nptest.assert_allclose(tsc.to_numpy(), rbf_coeff_inverse, atol=1e-1, rtol=0)

    def test_time_difference01(self):

        from findiff import FinDiff

        # from example https://maroba.github.io/findiff-docs/source/examples-basic.html

        time_values = np.linspace(0, 10, 100)
        dt = time_values[1] - time_values[0]
        f = np.sin(time_values)
        g = np.cos(time_values)

        df = pd.DataFrame(
            data=np.column_stack([f, g]), index=time_values, columns=["sin", "cos"]
        )

        X = TSCDataFrame.from_single_timeseries(df)

        actual = TSCFiniteDifference(spacing="dt", diff_order=2).fit_transform(X)

        d2_dx2 = FinDiff(0, dt, 2)
        expected = np.column_stack([d2_dx2(f), d2_dx2(g)])

        # move the time information to the time value that lies furthest in the future
        # e.g. the centered finite difference scheme
        #  weights [-0.5, 0, 0,5]
        #  time    [-1, 0, 1]
        # is computed and set for time offset 1.
        # This is because we cannot use future samples

        expected = TSCDataFrame.from_single_timeseries(
            pd.DataFrame(
                data=expected[1:-1, :],  # takes the center
                index=time_values[2:],
                columns=["sin_dot2", "cos_dot2"],
            )
        )

        pdtest.assert_frame_equal(actual, expected)
