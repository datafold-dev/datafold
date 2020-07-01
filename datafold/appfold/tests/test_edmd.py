#!/usr/bin/env python3

import unittest
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.testing as pdtest
from sklearn.model_selection import GridSearchCV

from datafold.appfold.edmd import EDMD, EDMDCV
from datafold.dynfold.transform import (
    TSCFeaturePreprocess,
    TSCIdentity,
    TSCPrincipalComponent,
    TSCTakensEmbedding,
)
from datafold.pcfold import TSCDataFrame, TSCKfoldSeries, TSCKFoldTime
from datafold.pcfold.timeseries.collection import TSCException
from datafold.utils.general import is_df_same_index


class EDMDTest(unittest.TestCase):
    def _setup_sine_wave_data(self) -> TSCDataFrame:
        time = np.linspace(0, 2 * np.pi, 100)
        df = pd.DataFrame(np.sin(time) + 10, index=time, columns=["sin"])
        return TSCDataFrame.from_single_timeseries(df)

    def _setup_multi_sine_wave_data(self) -> TSCDataFrame:
        time = np.linspace(0, 2 * np.pi, 100)

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
        _edmd_dict = EDMD(
            dict_steps=[("id", TSCIdentity())],
            include_id_state=False,
            compute_koopman_modes=True,
        ).fit(self.sine_wave_tsc)

        pdtest.assert_frame_equal(
            _edmd_dict.transform(self.sine_wave_tsc), self.sine_wave_tsc
        )

        actual = _edmd_dict.inverse_transform(_edmd_dict.transform(self.sine_wave_tsc))
        expected = self.sine_wave_tsc
        pdtest.assert_frame_equal(actual, expected)

        expected = _edmd_dict.reconstruct(self.sine_wave_tsc)
        is_df_same_index(expected, self.sine_wave_tsc)

    def test_id_dict2(self):
        _edmd_dict = EDMD(
            dict_steps=[("id", TSCIdentity())],
            include_id_state=False,
            compute_koopman_modes=False,  # different to test_id_dict1
        ).fit(self.sine_wave_tsc)

        pdtest.assert_frame_equal(
            _edmd_dict.transform(self.sine_wave_tsc), self.sine_wave_tsc
        )

        pdtest.assert_frame_equal(
            _edmd_dict.inverse_transform(self.sine_wave_tsc), self.sine_wave_tsc
        )

        expected = _edmd_dict.reconstruct(self.sine_wave_tsc)
        is_df_same_index(expected, self.sine_wave_tsc)

    def test_id_dict3(self):
        _edmd_dict = EDMD(
            dict_steps=[("id", TSCIdentity(include_const=True))],
            include_id_state=False,
            compute_koopman_modes=True,
        ).fit(self.sine_wave_tsc)

        actual = _edmd_dict.inverse_transform(_edmd_dict.transform(self.sine_wave_tsc))
        expected = self.sine_wave_tsc

        pdtest.assert_frame_equal(actual, expected)

        expected = _edmd_dict.reconstruct(self.sine_wave_tsc)
        is_df_same_index(expected, self.sine_wave_tsc)

    def test_qoi_selection1(self):
        tsc = self.multi_waves

        # pre-selection
        edmd = EDMD(dict_steps=[("id", TSCIdentity())], include_id_state=False).fit(tsc)

        cos_values = edmd.predict(tsc.initial_states(), qois=["cos"])
        sin_values = edmd.predict(tsc.initial_states(), qois=["sin"])

        pdtest.assert_index_equal(tsc.loc[:, "cos"].columns, cos_values.columns)
        pdtest.assert_index_equal(tsc.loc[:, "sin"].columns, sin_values.columns)

        cos_values_reconstruct = edmd.reconstruct(tsc, qois=["cos"])
        sin_values_reconstruct = edmd.reconstruct(tsc, qois=["sin"])

        pdtest.assert_index_equal(
            tsc.loc[:, "cos"].columns, cos_values_reconstruct.columns
        )
        pdtest.assert_index_equal(
            tsc.loc[:, "sin"].columns, sin_values_reconstruct.columns
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

    def test_access_koopman_system_triplet(self):
        # triplet = eigenvalues, Koopman modes and eigenfunctions

        _edmd = EDMD(
            dict_steps=[
                ("scale", TSCFeaturePreprocess.from_name(name="min-max")),
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=2)),
            ],
            include_id_state=True,
        ).fit(X=self.multi_waves)

        actual_modes = _edmd.koopman_modes
        actual_eigvals = _edmd.koopman_eigenvalues
        actual_eigfunc = _edmd.koopman_eigenfunction(X=self.multi_waves)

        # 2 original states
        # 4 eigenvectors in dictionary space (2 ID states + 2 PCA states)
        expected = (2, 4)
        self.assertTrue(actual_modes.shape, expected)
        self.assertTrue(actual_eigvals.shape, expected[1])
        self.assertTrue(actual_eigfunc.shape, (self.multi_waves.shape[0], expected[1]))

        self.assertIsInstance(actual_modes, pd.DataFrame)
        self.assertIsInstance(actual_eigvals, pd.Series)
        self.assertIsInstance(actual_eigfunc, TSCDataFrame)

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

    def test_edmd_dict_sine_wave(self, plot=False):
        _edmd_dict = EDMD(
            dict_steps=[
                ("scale", TSCFeaturePreprocess.from_name(name="min-max")),
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=2)),
            ]
        )

        forward_dict = _edmd_dict.fit_transform(X=self.sine_wave_tsc)
        self.assertIsInstance(forward_dict, TSCDataFrame)

        inverse_dict = _edmd_dict.inverse_transform(X=forward_dict)
        self.assertIsInstance(inverse_dict, TSCDataFrame)

        # index not the same because of Takens, so only check column
        pdtest.assert_index_equal(
            self.sine_wave_tsc.columns, inverse_dict.columns,
        )

        diff = inverse_dict - self.sine_wave_tsc
        # sort out the removed rows from Takens (NaN values)
        self.assertTrue((diff.dropna() < 1e-14).all().all())

        if plot:
            ax = self.sine_wave_tsc.plot()
            inverse_dict.plot(ax=ax)
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
            verbose=False,
            return_train_score=True,
            n_jobs=1,
        ).fit(self.multi_sine_wave_tsc)

        # passes reconstruct to best_estimator_ (EDMD)
        edmdcv.reconstruct(self.multi_sine_wave_tsc)

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


if __name__ == "__main__":
    test = EDMDTest()
    test.setUp()

    test.test_edmd_cv_sine_wave()
