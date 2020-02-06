#!/usr/bin/env python3

import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.testing as pdtest
from sklearn.model_selection import GridSearchCV

from datafold.appfold.edmd import EDMD, EDMDCV, EDMDDict
from datafold.pcfold.timeseries import TSCDataFrame
from datafold.pcfold.timeseries.metric import TSCKfoldSeries, TSCKFoldTime
from datafold.pcfold.timeseries.transform import (
    TSCIdentity,
    TSCPrincipalComponent,
    TSCQoiPreprocess,
    TSCTakensEmbedding,
)


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

        self.assertTrue(tsc.is_equal_time_values())

        return tsc

    def setUp(self) -> None:
        self.sine_wave_tsc = self._setup_sine_wave_data()
        self.multi_sine_wave_tsc = self._setup_multi_sine_wave_data()

    def test_id_dict(self):
        _edmd_dict = EDMDDict(steps=[("id", TSCIdentity())]).fit(self.sine_wave_tsc)

        pdtest.assert_frame_equal(
            _edmd_dict.transform(self.sine_wave_tsc), self.sine_wave_tsc
        )

        pdtest.assert_frame_equal(
            _edmd_dict.inverse_transform(self.sine_wave_tsc), self.sine_wave_tsc
        )

    def test_edmd_dict_sine_wave(self, plot=False):
        _edmd_dict = EDMDDict(
            steps=[
                ("scale", TSCQoiPreprocess.scale(name="min-max")),
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

    def test_edmd_cv_sine_wave(self):
        # Tests a specific setting of EDMDCV compared to sklearn.GridSearchCV,
        # where the results are expected to be the same. EDMDCV generalizes aspects
        # that fail for GridSearchCV

        edmd = EDMD(
            dict_steps=[
                # NOTE: in Takens fill-in handle *cannot* be "remove", because sklearn cv
                #  will because the number of samples changes during transformation (
                #  EDMDCV could handle this)
                ("delays", TSCTakensEmbedding(delays=10, fillin_handle=1)),
                ("pca", TSCPrincipalComponent(n_components=5)),
            ]
        )

        # NOTE: cv only TSCKfoldSeries can be compared and is equal to sklearn. not
        #  E.g. TSCKfoldTime requires to adapt the internal data (setting the time
        #  series correctly for the DMD model
        sklearn_cv = GridSearchCV(
            estimator=edmd,
            param_grid={"pca__n_components": [5, 10]},
            cv=TSCKfoldSeries(2),
            verbose=False,
            return_train_score=True,
            n_jobs=None,
        )

        edmdcv = EDMDCV(
            estimator=edmd,
            param_grid={"pca__n_components": [5, 10]},
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

    def test_edmdcv_parallel_no_error(self):
        edmd = EDMD(
            dict_steps=[
                ("delays", TSCTakensEmbedding(delays=10, fillin_handle=1)),
                ("pca", TSCPrincipalComponent(n_components=5)),
            ]
        )

        edmdcv = EDMDCV(
            estimator=edmd,
            param_grid={"pca__n_components": [2, 4, 6, 8]},
            cv=TSCKfoldSeries(4),
            verbose=False,
            return_train_score=True,
            n_jobs=-1,
        )

        edmdcv.fit(self._setup_multi_sine_wave_data())
        self.assertIsInstance(edmdcv.cv_results_, dict)

    def test_edmdcv_timecv_no_error(self):
        edmd = EDMD(
            dict_steps=[
                ("delays", TSCTakensEmbedding(delays=10, fillin_handle=1)),
                ("pca", TSCPrincipalComponent(n_components=5)),
            ]
        )

        edmdcv = EDMDCV(
            estimator=edmd,
            param_grid={"pca__n_components": [2, 4]},
            cv=TSCKFoldTime(4),
            verbose=False,
            return_train_score=True,
            n_jobs=1,
        )

        edmdcv.fit(self._setup_multi_sine_wave_data())
        self.assertIsInstance(edmdcv.cv_results_, dict)
