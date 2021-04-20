#!/usr/bin/env python3

import tempfile
import unittest
import warnings
import webbrowser
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest
from sklearn.model_selection import GridSearchCV
from sklearn.utils import estimator_html_repr

from datafold.appfold.edmd import EDMD, EDMDCV, EDMDWindowPrediction
from datafold.dynfold import DMDFull, gDMDFull
from datafold.dynfold.transform import (
    TSCFeaturePreprocess,
    TSCIdentity,
    TSCPrincipalComponent,
    TSCTakensEmbedding,
)
from datafold.pcfold import TSCDataFrame, TSCKfoldSeries, TSCKFoldTime
from datafold.pcfold.timeseries.collection import TSCException
from datafold.utils.general import is_df_same_index
from datafold.utils.plot import plot_eigenvalues


class EDMDTest(unittest.TestCase):
    @staticmethod
    def _setup_sine_wave_data() -> TSCDataFrame:
        time = np.linspace(0, 2 * np.pi, 100)
        df = pd.DataFrame(np.sin(time) + 10, index=time, columns=["sin"])
        return TSCDataFrame.from_single_timeseries(df)

    def _setup_multi_sine_wave_data(self) -> TSCDataFrame:
        time = np.linspace(0, 4 * np.pi, 100)

        omega = 1.5

        for i in range(1, 11):
            data = np.sin(i * omega * time)
            df = pd.DataFrame(data=data, index=time, columns=["sin"])
            if i == 1:
                tsc = TSCDataFrame.from_single_timeseries(df)
            else:
                tsc = tsc.insert_ts(df)

        self.assertTrue(tsc.is_same_time_values())

        return tsc

    def _setup_multi_sine_wave_data2(self) -> TSCDataFrame:
        time = np.linspace(0, 2 * np.pi, 100)

        omega = 1.5

        for i in range(1, 11):
            data = np.column_stack([np.sin(i * omega * time), np.cos(i * omega * time)])
            df = pd.DataFrame(data=data, index=time, columns=["sin", "cos"])
            if i == 1:
                tsc = TSCDataFrame.from_single_timeseries(df)
            else:
                tsc = tsc.insert_ts(df)

        self.assertTrue(tsc.is_same_time_values())

        return tsc

    def setUp(self) -> None:
        self.sine_wave_tsc = self._setup_sine_wave_data()
        self.multi_sine_wave_tsc = self._setup_multi_sine_wave_data()
        self.multi_waves = self._setup_multi_sine_wave_data2()

    def test_id_dict1(self):
        _edmd = EDMD(
            dict_steps=[("id", TSCIdentity())],
            include_id_state=False,
            use_transform_inverse=False,
            verbose=2,
        ).fit(self.sine_wave_tsc)

        pdtest.assert_frame_equal(
            _edmd.transform(self.sine_wave_tsc), self.sine_wave_tsc
        )

        actual = _edmd.inverse_transform(_edmd.transform(self.sine_wave_tsc))
        expected = self.sine_wave_tsc
        pdtest.assert_frame_equal(actual, expected)

        expected = _edmd.reconstruct(self.sine_wave_tsc)
        is_df_same_index(expected, self.sine_wave_tsc)

        self.assertEqual(_edmd.feature_names_in_, self.sine_wave_tsc.columns)
        self.assertEqual(_edmd.feature_names_out_, self.sine_wave_tsc.columns)
        self.assertEqual(_edmd.n_features_in_, self.sine_wave_tsc.shape[1])
        self.assertEqual(_edmd.n_features_out_, self.sine_wave_tsc.shape[1])

    def test_id_dict2(self):
        _edmd = EDMD(
            dict_steps=[("id", TSCIdentity())],
            include_id_state=False,
            use_transform_inverse=True,  # different to test_id_dict1
            verbose=2,
        ).fit(self.sine_wave_tsc)

        pdtest.assert_frame_equal(
            _edmd.transform(self.sine_wave_tsc), self.sine_wave_tsc
        )

        pdtest.assert_frame_equal(
            _edmd.inverse_transform(self.sine_wave_tsc), self.sine_wave_tsc
        )

        expected = _edmd.reconstruct(self.sine_wave_tsc)
        is_df_same_index(expected, self.sine_wave_tsc)

        self.assertEqual(_edmd.feature_names_in_, self.sine_wave_tsc.columns)
        self.assertEqual(_edmd.feature_names_out_, self.sine_wave_tsc.columns)
        self.assertEqual(_edmd.n_features_in_, self.sine_wave_tsc.shape[1])
        self.assertEqual(_edmd.n_features_out_, self.sine_wave_tsc.shape[1])

    def test_id_dict3(self):
        _edmd = EDMD(
            dict_steps=[("id", TSCIdentity(include_const=True))],
            include_id_state=False,
            use_transform_inverse=False,
        ).fit(self.sine_wave_tsc)

        actual = _edmd.inverse_transform(_edmd.transform(self.sine_wave_tsc))
        expected = self.sine_wave_tsc

        pdtest.assert_frame_equal(actual, expected)

        expected = _edmd.reconstruct(self.sine_wave_tsc)
        is_df_same_index(expected, self.sine_wave_tsc)

        self.assertEqual(_edmd.feature_names_in_, self.sine_wave_tsc.columns)
        self.assertEqual(_edmd.n_features_in_, self.sine_wave_tsc.shape[1])
        self.assertEqual(_edmd.n_features_out_, self.sine_wave_tsc.shape[1] + 1)

    def test_qoi_selection1(self):
        X = self.multi_waves

        # pre-selection
        edmd = EDMD(dict_steps=[("id", TSCIdentity())], include_id_state=False).fit(X)

        cos_values = edmd.predict(X.initial_states(), qois=["cos"])
        sin_values = edmd.predict(X.initial_states(), qois=["sin"])

        pdtest.assert_index_equal(X.loc[:, "cos"].columns, cos_values.columns)
        pdtest.assert_index_equal(X.loc[:, "sin"].columns, sin_values.columns)

        cos_values_reconstruct = edmd.reconstruct(X, qois=["cos"])
        sin_values_reconstruct = edmd.reconstruct(X, qois=["sin"])

        pdtest.assert_index_equal(
            X.loc[:, "cos"].columns, cos_values_reconstruct.columns
        )
        pdtest.assert_index_equal(
            X.loc[:, "sin"].columns, sin_values_reconstruct.columns
        )

    def test_qoi_selection2(self):
        tsc = self.multi_waves

        # pre-selection
        edmd = EDMD(
            dict_steps=[("id", TSCIdentity(include_const=False, rename_features=True))],
            include_id_state=True,
        ).fit(tsc)

        cos_values_predict = edmd.predict(tsc.initial_states(), qois=["cos"])
        sin_values_predict = edmd.predict(tsc.initial_states(), qois=["sin"])

        pdtest.assert_index_equal(tsc.loc[:, "cos"].columns, cos_values_predict.columns)
        pdtest.assert_index_equal(tsc.loc[:, "sin"].columns, sin_values_predict.columns)

        cos_values_reconstruct = edmd.reconstruct(tsc, qois=["cos"])
        sin_values_reconstruct = edmd.reconstruct(tsc, qois=["sin"])

        pdtest.assert_index_equal(
            tsc.loc[:, "cos"].columns, cos_values_reconstruct.columns
        )
        pdtest.assert_index_equal(
            tsc.loc[:, "sin"].columns, sin_values_reconstruct.columns
        )

    def test_qoi_selection3(self):
        tsc = self.multi_waves

        # pre-selection
        edmd = EDMD(
            dict_steps=[("id", TSCIdentity(include_const=False, rename_features=True))],
            include_id_state=True,
        ).fit(tsc)

        with self.assertRaises(ValueError):
            edmd.predict(tsc.initial_states(), qois=["INVALID"])

    def test_edmd_no_classifier(self):
        # import from internal module -- subject to change without warning!
        from sklearn.model_selection._validation import is_classifier

        self.assertFalse(is_classifier(EDMD))
        self.assertFalse(is_classifier(EDMDCV))

    def test_n_samples_ic(self):
        _edmd = EDMD(
            dict_steps=[
                ("scale", TSCFeaturePreprocess.from_name(name="min-max")),
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=2)),
            ],
            include_id_state=True,
        ).fit(X=self.multi_waves)

        actual = _edmd.transform(self.multi_waves.initial_states(_edmd.n_samples_ic_))

        # each initial-condition time series must result into a single state in
        # dictionary space
        self.assertIsInstance(actual, pd.DataFrame)

        # 2 ID states + 2 PCA components
        self.assertEqual(actual.shape, (self.multi_waves.n_timeseries, 2 + 2))

        # Take one sample more and transform the states
        actual = _edmd.transform(
            self.multi_waves.initial_states(_edmd.n_samples_ic_ + 1)
        )
        self.assertIsInstance(actual, TSCDataFrame)

        # Having not enough samples must result into error
        with self.assertRaises(TSCException):
            _edmd.transform(self.multi_waves.initial_states(_edmd.n_samples_ic_ - 1))

    def test_error_nonmatch_time_sample(self):
        _edmd = EDMD(
            dict_steps=[
                ("scale", TSCFeaturePreprocess.from_name(name="min-max")),
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=2)),
            ],
            include_id_state=True,
        ).fit(X=self.multi_waves)

        initial_condition = self.multi_waves.initial_states(_edmd.n_samples_ic_)
        # change time values to a different sampling interval
        initial_condition.index = pd.MultiIndex.from_arrays(
            [
                initial_condition.index.get_level_values(TSCDataFrame.tsc_id_idx_name),
                # change sample rate:
                initial_condition.index.get_level_values(TSCDataFrame.tsc_time_idx_name)
                * 2,
            ]
        )

        with self.assertRaises(TSCException):
            _edmd.predict(initial_condition)

    def test_access_koopman_triplet(self):
        # triplet = eigenvalues, Koopman modes and eigenfunctions

        _edmd = EDMD(
            dict_steps=[
                ("scale", TSCFeaturePreprocess.from_name(name="min-max")),
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=2)),
            ],
            include_id_state=True,
        ).fit(X=self.multi_waves)

        eval_waves = self.multi_waves.loc[pd.IndexSlice[0:1], :]

        actual_modes = _edmd.koopman_modes
        actual_eigvals = _edmd.koopman_eigenvalues
        actual_eigfunc = _edmd.koopman_eigenfunction(X=eval_waves)

        # 2 original states
        # 4 eigenvectors in dictionary space (2 ID states + 2 PCA states)
        expected = (2, 4)
        self.assertEqual(actual_modes.shape, expected)
        self.assertEqual(actual_eigvals.shape[0], expected[1])
        self.assertEqual(
            actual_eigfunc.shape,
            (
                eval_waves.shape[0]
                # correct the output samples by number of samples required for
                # initial condition
                - eval_waves.n_timeseries * (_edmd.n_samples_ic_ - 1),
                expected[1],
            ),
        )

        self.assertIsInstance(actual_modes, pd.DataFrame)
        self.assertIsInstance(actual_eigvals, pd.Series)
        self.assertIsInstance(actual_eigfunc, TSCDataFrame)

    def test_sort_koopman_triplets(self):
        _edmd_wo_sort = EDMD(
            dict_steps=[
                ("scale", TSCFeaturePreprocess.from_name(name="min-max")),
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=6)),
            ],
            dmd_model=DMDFull(is_diagonalize=True),
            include_id_state=True,
        ).fit(X=self.multi_waves)

        _edmd_w_sort = EDMD(
            dict_steps=[
                ("scale", TSCFeaturePreprocess.from_name(name="min-max")),
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=6)),
            ],
            dmd_model=DMDFull(is_diagonalize=True),
            sort_koopman_triplets=True,
            include_id_state=True,
        ).fit(X=self.multi_waves)

        expected = _edmd_wo_sort.reconstruct(self.multi_waves)
        actual = _edmd_w_sort.reconstruct(self.multi_waves)

        pdtest.assert_frame_equal(expected, actual)

    def test_koopman_eigenfunction_eval(self):
        _edmd = EDMD(
            dict_steps=[
                ("scale", TSCFeaturePreprocess.from_name(name="min-max")),
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=2)),
            ],
            include_id_state=True,
        ).fit(X=self.multi_waves)

        actual = _edmd.koopman_eigenfunction(
            self.multi_waves.initial_states(_edmd.n_samples_ic_ + 1)
        )

        self.assertIsInstance(actual, TSCDataFrame)

        actual = _edmd.koopman_eigenfunction(
            self.multi_waves.initial_states(_edmd.n_samples_ic_)
        )

        self.assertIsInstance(actual, pd.DataFrame)

        with self.assertRaises(TSCException):
            _edmd.koopman_eigenfunction(
                self.multi_waves.initial_states(_edmd.n_samples_ic_ - 1)
            )

    def test_dmap_kernels(self, plot=False):
        X = self._setup_multi_sine_wave_data2()

        from datafold.dynfold import DiffusionMaps
        from datafold.pcfold import (
            ConeKernel,
            ContinuousNNKernel,
            GaussianKernel,
            InverseMultiquadricKernel,
            MultiquadricKernel,
        )

        kernels = [
            ConeKernel(zeta=0.0, epsilon=0.5),
            ConeKernel(zeta=0.9, epsilon=0.1),
            GaussianKernel(epsilon=0.1),
            ContinuousNNKernel(k_neighbor=20, delta=0.9),
            # MultiquadricKernel(epsilon=1),
            InverseMultiquadricKernel(epsilon=0.5),
        ]

        f, ax = plt.subplots(nrows=len(kernels) + 1, ncols=1, sharex=True)
        X.plot(ax=ax[0])
        ax[0].set_title("original data")

        for i, kernel in enumerate(kernels):
            try:
                X_predict = EDMD(
                    [
                        ("takens", TSCTakensEmbedding(delays=20)),
                        ("dmap", DiffusionMaps(kernel, n_eigenpairs=220)),
                    ]
                ).fit_predict(X)
                X_predict.plot(ax=ax[i + 1])
                ax[i + 1].set_title(f"kernel={kernel}")

            except Exception as e:
                print(f"kernel={kernel} failed")
                raise e

        if plot:
            plt.show()

    def test_attach_illegal_id(self):
        _edmd = EDMD(
            dict_steps=[
                ("id", TSCIdentity()),
            ],
            include_id_state=True,  # cannot be attached to identity (same feature names)
        )

        with self.assertRaises(ValueError):
            _edmd.fit(X=self.sine_wave_tsc)

    def test_edmd_dict_sine_wave(self, plot=False):
        _edmd = EDMD(
            dict_steps=[
                ("scale", TSCFeaturePreprocess.from_name(name="min-max")),
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=2)),
            ]
        )

        forward_dict = _edmd.fit_transform(X=self.sine_wave_tsc)
        self.assertIsInstance(forward_dict, TSCDataFrame)

        inverse_dict = _edmd.inverse_transform(X=forward_dict)
        self.assertIsInstance(inverse_dict, TSCDataFrame)

        # index not the same because of Takens, so only check column
        pdtest.assert_index_equal(
            self.sine_wave_tsc.columns,
            inverse_dict.columns,
        )

        diff = inverse_dict - self.sine_wave_tsc
        # sort out the removed rows from Takens (NaN values)
        self.assertTrue((diff.dropna() < 1e-14).all().all())

        if plot:
            ax = self.sine_wave_tsc.plot()
            inverse_dict.plot(ax=ax)

            from datafold.utils.plot import plot_eigenvalues

            f, ax = plt.subplots()
            plot_eigenvalues(eigenvalues=_edmd.dmd_model.eigenvalues_, ax=ax)

            plt.show()

    def test_edmd_dict_sine_wave_generator(self, plot=False):
        # Use a DMD model that approximates the generator matrix and not the Koopman
        # operator
        _edmd = EDMD(
            dict_steps=[
                ("scale", TSCFeaturePreprocess.from_name(name="min-max")),
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=2)),
            ],
            dmd_model=gDMDFull(),
        )

        forward_dict = _edmd.fit_transform(
            X=self.sine_wave_tsc, **dict(dmd__store_generator_matrix=True)
        )
        self.assertIsInstance(forward_dict, TSCDataFrame)

        inverse_dict = _edmd.inverse_transform(X=forward_dict)
        self.assertIsInstance(inverse_dict, TSCDataFrame)

        # index not the same because of Takens, so only check column
        pdtest.assert_index_equal(
            self.sine_wave_tsc.columns,
            inverse_dict.columns,
        )

        diff = inverse_dict - self.sine_wave_tsc
        # sort out the removed rows from Takens (NaN values)
        self.assertTrue((diff.dropna() < 1e-14).all().all())

        # test that the fit_param dmd__store_generator_matrix was really passed to the
        # DMD model.
        self.assertTrue(hasattr(_edmd.dmd_model, "generator_matrix_"))

        if plot:
            ax = self.sine_wave_tsc.plot()
            inverse_dict.plot(ax=ax)
            f, ax = plt.subplots()
            plot_eigenvalues(eigenvalues=_edmd.dmd_model.eigenvalues_, ax=ax)
            plt.show()

    def test_spectral_and_matrix_mode(self):
        _edmd_spectral = EDMD(
            dict_steps=[
                ("scale", TSCFeaturePreprocess.from_name(name="min-max")),
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=2)),
            ],
            dmd_model=DMDFull(sys_mode="spectral"),
        )

        _edmd_matrix = EDMD(
            dict_steps=[
                ("scale", TSCFeaturePreprocess.from_name(name="min-max")),
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=2)),
            ],
            dmd_model=DMDFull(sys_mode="matrix"),
        )

        actual_spectral = _edmd_spectral.fit_predict(X=self.sine_wave_tsc)
        actual_matrix = _edmd_matrix.fit_predict(X=self.sine_wave_tsc)

        pdtest.assert_frame_equal(actual_spectral, actual_matrix)

        self.assertTrue(_edmd_spectral.koopman_modes is not None)
        self.assertTrue(_edmd_spectral.dmd_model.is_spectral_mode())

        self.assertTrue(_edmd_matrix.koopman_modes is None)
        self.assertTrue(_edmd_matrix.dmd_model.is_matrix_mode())

        # use qois argument
        actual_spectral = _edmd_spectral.reconstruct(X=self.sine_wave_tsc, qois=["sin"])
        actual_matrix = _edmd_matrix.reconstruct(X=self.sine_wave_tsc, qois=["sin"])
        pdtest.assert_frame_equal(actual_spectral, actual_matrix)

    def test_edmd_with_composed_dict(self, display_html=False, plot=False):

        from sklearn.compose import make_column_selector

        from datafold.dynfold import TSCColumnTransformer

        selector_sin = make_column_selector(pattern="sin")
        selector_cos = make_column_selector(pattern="cos")

        separate_transform = TSCColumnTransformer(
            transformers=[
                ("sin_path", TSCPrincipalComponent(n_components=10), selector_sin),
                ("cos_path", TSCPrincipalComponent(n_components=10), selector_cos),
            ]
        )

        edmd = EDMD(
            dict_steps=[
                ("delays", TSCTakensEmbedding(delays=15)),
                ("pca", separate_transform),
            ],
            include_id_state=True,
        ).fit(self.multi_waves)

        if display_html:
            with tempfile.NamedTemporaryFile("w", suffix=".html") as fp:
                fp.write(estimator_html_repr(edmd))
                fp.flush()
                webbrowser.open_new_tab(fp.name)
                input("Press Enter to continue...")

        f, ax = plt.subplots(nrows=2, sharex=True)
        self.multi_waves.plot(ax=ax[0])
        edmd.reconstruct(self.multi_waves).plot(ax=ax[1])

        if plot:
            plt.show()

    def test_edmd_sine_wave(self):

        edmd = EDMD(
            dict_steps=[
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=5)),
            ],
            include_id_state=True,
        )
        case_one_edmd = deepcopy(edmd)
        case_two_edmd = deepcopy(edmd)

        case_one = case_one_edmd.fit(self.multi_sine_wave_tsc).reconstruct(
            self.multi_sine_wave_tsc
        )
        case_two = case_two_edmd.fit_predict(self.multi_sine_wave_tsc)

        pdtest.assert_frame_equal(case_one, case_two)

    def test_edmd_cv_sine_wave(self):
        # Tests a specific setting of EDMDCV compared to sklearn.GridSearchCV,
        # where the results are expected to be the same. EDMDCV generalizes aspects
        # that fail for GridSearchCV

        edmd = EDMD(
            dict_steps=[
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=5)),
            ],
            include_id_state=False,
        )

        # NOTE: cv only TSCKfoldSeries can be compared and is equal to sklearn. not
        #  E.g. TSCKfoldTime requires to adapt the internal data (setting the time
        #  series correctly for the DMD model
        sklearn_cv = GridSearchCV(
            estimator=edmd,
            param_grid={"pca__n_components": [5, 7]},
            cv=TSCKfoldSeries(2),
            verbose=False,
            return_train_score=True,
            n_jobs=None,
        )

        edmdcv = EDMDCV(
            estimator=edmd,
            param_grid={"pca__n_components": [5, 7]},
            cv=TSCKfoldSeries(2),
            verbose=False,
            return_train_score=True,
            n_jobs=None,
        )

        sklearn_cv.fit(self.multi_sine_wave_tsc)
        edmdcv.fit(self.multi_sine_wave_tsc)

        # timings are very unlikely to be the same, so drop them for the comparison:
        drop_rows = {
            "mean_fit_time",
            "std_fit_time",
            "mean_score_time",
            "std_score_time",
        }

        expected_results = pd.DataFrame(sklearn_cv.cv_results_).T.drop(
            labels=drop_rows, axis=0
        )
        actual_results = pd.DataFrame(edmdcv.cv_results_).T.drop(
            labels=drop_rows, axis=0
        )

        pdtest.assert_frame_equal(expected_results, actual_results)

    def test_edmdcv_fail_all(self):
        edmd = EDMD(
            dict_steps=[
                ("timedelay", TSCTakensEmbedding(delays=1)),
            ]
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")  # suppress error messages
            edmdcv = EDMDCV(
                estimator=edmd,
                param_grid={"timedelay__delays": [9999999]},  # raises error
                cv=TSCKfoldSeries(2),
                verbose=2,
                refit=False,
                return_train_score=True,
                error_score=np.nan,
                n_jobs=1,
            ).fit(self.multi_sine_wave_tsc)

        df = pd.DataFrame(edmdcv.cv_results_).T

        self.assertTrue(np.isnan(df.loc["split1_test_score", :][0]))
        self.assertTrue(np.isnan(df.loc["split1_test_score", :][0]))
        self.assertTrue(np.isnan(df.loc["mean_test_score", :][0]))
        self.assertTrue(np.isnan(df.loc["std_test_score", :][0]))

        self.assertTrue(np.isnan(df.loc["split0_train_score", :][0]))
        self.assertTrue(np.isnan(df.loc["split1_train_score", :][0]))
        self.assertTrue(np.isnan(df.loc["mean_train_score", :][0]))
        self.assertTrue(np.isnan(df.loc["std_train_score", :][0]))

        with self.assertRaises(TSCException):
            EDMDCV(
                estimator=edmd,
                param_grid={"timedelay__delays": [9999999]},
                cv=TSCKfoldSeries(2),
                verbose=False,
                refit=False,
                return_train_score=True,
                error_score="raise",
                n_jobs=1,
            ).fit(self.multi_sine_wave_tsc)

    def test_edmdcv_seriescv_no_error(self):
        edmd = EDMD(
            dict_steps=[
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=5)),
            ]
        )

        edmdcv = EDMDCV(
            estimator=edmd,
            param_grid={"pca__n_components": [5, 7]},
            cv=TSCKfoldSeries(4),
            verbose=2,
            return_train_score=True,
            n_jobs=1,
        ).fit(self.multi_sine_wave_tsc)

        # passes reconstruct to best_estimator_ (EDMD)
        edmdcv.best_estimator_.reconstruct(self.multi_sine_wave_tsc)

    def test_edmdcv_parallel_no_error(self):
        edmd = EDMD(
            dict_steps=[
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=5)),
            ]
        )

        edmdcv = EDMDCV(
            estimator=edmd,
            param_grid={"pca__n_components": [2, 4, 6, 8]},
            cv=TSCKfoldSeries(4),
            verbose=False,
            error_score="raise",
            return_train_score=True,
            n_jobs=-1,
        ).fit(self.multi_sine_wave_tsc)

        self.assertIsInstance(edmdcv.cv_results_, dict)

    def test_edmdcv_timecv_no_error(self):
        edmd = EDMD(
            dict_steps=[
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=5)),
            ]
        )

        edmdcv = EDMDCV(
            estimator=edmd,
            param_grid={"pca__n_components": [2, 4]},
            cv=TSCKFoldTime(4),
            verbose=False,
            error_score="raise",
            return_train_score=True,
            n_jobs=1,
        ).fit(self.multi_sine_wave_tsc)

        self.assertIsInstance(edmdcv.cv_results_, dict)


class EDMDPredictionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sine_data = EDMDTest._setup_sine_wave_data()

    def test_sine_data(self):
        edmd = EDMD(dict_steps=[("id", TSCIdentity())], include_id_state=False)
        edmd.fit(self.sine_data)
        edmd_new = EDMDWindowPrediction(window_size=2, offset=2).adapt_model(edmd)

        actual = edmd_new.reconstruct(X=self.sine_data)

        self.assertTrue(actual.n_timeseries, 50)
        nptest.assert_equal(actual.time_values(), self.sine_data.time_values())

        actual_score = edmd_new.score(self.sine_data)
        self.assertIsInstance(actual_score, float)

    def test_sine_data2(self):
        edmd = EDMD(
            dict_steps=[("id", TSCTakensEmbedding(delays=2))], include_id_state=False
        )
        edmd.fit(self.sine_data)
        edmd_new = EDMDWindowPrediction(
            window_size=edmd.n_samples_ic_ + 1, offset=2
        ).adapt_model(edmd)
        actual = edmd_new.reconstruct(X=self.sine_data)

        self.assertTrue(actual.n_timeseries, 50)
        self.assertTrue(actual.n_timesteps, 2)

        self.assertIsInstance(edmd_new.score(self.sine_data), float)

        edmd_new.blocksize = 3  # equals number to obtain initial condition
        self.assertTrue(actual.n_timeseries, 50)
        self.assertTrue(actual.n_timesteps, 1)

        self.assertIsInstance(edmd_new.score(self.sine_data), float)


if __name__ == "__main__":
    test = EDMDTest()
    test.setUp()

    test.test_edmd_cv_sine_wave()
