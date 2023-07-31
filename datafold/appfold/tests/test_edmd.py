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
import pytest
import scipy.sparse
from sklearn.compose import make_column_selector
from sklearn.model_selection import GridSearchCV
from sklearn.utils import estimator_html_repr

from datafold import (
    EDMD,
    EDMDCV,
    DMDControl,
    DMDStandard,
    EDMDWindowPrediction,
    OnlineDMD,
    StreamingDMD,
    TSCColumnTransformer,
    TSCDataFrame,
    TSCFeaturePreprocess,
    TSCIdentity,
    TSCKfoldSeries,
    TSCKFoldTime,
    TSCMetric,
    TSCPolynomialFeatures,
    TSCPrincipalComponent,
    TSCTakensEmbedding,
    gDMDFull,
)
from datafold.pcfold.timeseries.collection import TSCException
from datafold.utils._systems import InvertedPendulum
from datafold.utils.general import is_df_same_index
from datafold.utils.plot import plot_eigenvalues


class EDMDTest(unittest.TestCase):
    @staticmethod
    def _setup_sine_wave_data(end=2 * np.pi) -> TSCDataFrame:
        time = np.linspace(0, end, 100)
        df = pd.DataFrame(np.sin(time) + 10, index=time, columns=["sin"])
        return TSCDataFrame.from_single_timeseries(df)

    def _setup_multi_sine_wave_data(self, n_samples=100) -> TSCDataFrame:
        time = np.linspace(0, 4 * np.pi, n_samples)

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

    @staticmethod
    def setup_inverted_pendulum(
        sim_time_step=0.1,
        sim_num_steps=10,
        training_size=5,
        require_last_control_state=False,
        seed=42,
    ):
        gen = np.random.default_rng(seed)

        X_tsc, U_tsc = [], []

        for _ in range(training_size):
            control_amplitude = 0.1 + 0.9 * gen.random()
            control_frequency = np.pi + 2 * np.pi * gen.random()
            control_phase = 2 * np.pi * gen.random()
            Ufunc = lambda t, y: control_amplitude * np.sin(
                control_frequency * t + control_phase
            )

            time_values = np.arange(0, sim_time_step * sim_num_steps, sim_time_step)
            X_ic = np.array([[0, 0, np.pi, 0]])

            X, U = InvertedPendulum().predict(
                X=X_ic,
                U=Ufunc,
                time_values=time_values,
                require_last_control_state=require_last_control_state,
            )

            X_tsc.append(X)
            U_tsc.append(U)

        X_tsc = TSCDataFrame.from_frame_list(X_tsc)
        U_tsc = TSCDataFrame.from_frame_list(U_tsc)
        return X_tsc, U_tsc

    def setUp(self) -> None:
        self.sine_wave_tsc = self._setup_sine_wave_data()
        self.multi_sine_wave_tsc = self._setup_multi_sine_wave_data()
        self.multi_waves = self._setup_multi_sine_wave_data2()

    def test_slicing(self):
        _edmd = EDMD(
            dict_steps=[
                ("id1", TSCIdentity()),
                ("id2", TSCIdentity()),
                ("id3", TSCIdentity()),
            ],
            include_id_state=False,
            use_transform_inverse=False,
        ).fit(self.sine_wave_tsc)

        from sklearn.pipeline import Pipeline

        self.assertIsInstance(_edmd[0], TSCIdentity)
        self.assertIsInstance(_edmd[:1], Pipeline)
        self.assertIsInstance(_edmd[:2], Pipeline)
        self.assertIsInstance(_edmd[1:3], Pipeline)

        # the slices over transform are Pipelines, but no EDMD anymore
        self.assertFalse(isinstance(_edmd[:1], EDMD))
        self.assertFalse(isinstance(_edmd[:2], EDMD))
        self.assertFalse(isinstance(_edmd[1:3], EDMD))

        # If everything is sliced however, they are still EDMD models
        self.assertIsInstance(_edmd[:], EDMD)
        self.assertIsInstance(_edmd[:4], EDMD)

        self.assertEqual(list(_edmd[:2].named_steps.keys()), ["id1", "id2"])

        with self.assertRaises(ValueError):
            _edmd[2:]

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
        self.assertEqual(_edmd.get_feature_names_out(), self.sine_wave_tsc.columns)
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
        self.assertEqual(_edmd.get_feature_names_out(), self.sine_wave_tsc.columns)
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
            include_id_state=False,
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
        ).fit(X=self.multi_waves, **dict(dmd__store_system_matrix=True))

        eval_waves = self.multi_waves.loc[pd.IndexSlice[0:1], :]

        actual_matrix = _edmd.dmd_model.system_matrix_
        actual_modes = _edmd.koopman_modes
        actual_eigvals = _edmd.koopman_eigenvalues
        actual_eigfunc = _edmd.koopman_eigenfunction(X=eval_waves)

        self.assertIsInstance(actual_matrix, np.ndarray)
        self.assertTrue(actual_matrix.shape, (4, 4))

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
            dmd_model=DMDStandard(diagonalize=True),
            include_id_state=True,
        ).fit(X=self.multi_waves)

        _edmd_w_sort = EDMD(
            dict_steps=[
                ("scale", TSCFeaturePreprocess.from_name(name="min-max")),
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=6)),
            ],
            dmd_model=DMDStandard(diagonalize=True),
            sort_koopman_triplets=True,
            include_id_state=True,
        ).fit(X=self.multi_waves)

        expected = _edmd_wo_sort.reconstruct(self.multi_waves)
        actual = _edmd_w_sort.reconstruct(self.multi_waves)

        pdtest.assert_frame_equal(expected, actual)

    def test_is_time_invariant_system(self):
        # the system corresponds to the Pendulum example

        from datafold.utils._systems import Pendulum

        theta_init = np.array([[np.pi / 3, -4], [-3 * np.pi / 4, 2]])
        theta_init_oos = np.array([np.pi / 2, -2])

        t_eval = np.linspace(0, 8 * np.pi, 500)

        system = Pendulum(friction=0.45)
        cart_df = system.predict(theta_init, time_values=t_eval).loc[:, ("x", "y")]
        cart_df_oos = system.predict(theta_init_oos, time_values=t_eval).loc[
            :, ("x", "y")
        ]

        edmd = EDMD(
            dict_steps=[
                ("delay", TSCTakensEmbedding(delays=3, lag=0, kappa=0)),
            ],
            dmd_model=DMDStandard(approx_generator=False),
            include_id_state=False,
        )

        edmd_floatindex = deepcopy(edmd)
        edmd_floatindex.fit(cart_df)

        # modes @ eigenfunctions(X_ic) is the reconstructed initial state
        X_ic = cart_df_oos.initial_states(edmd_floatindex.n_samples_ic_)
        expected = (
            edmd_floatindex.koopman_modes.to_numpy()
            @ edmd_floatindex.koopman_eigenfunction(X_ic).to_numpy().T
        )
        expected = np.real(expected)
        actual = edmd_floatindex.reconstruct(cart_df_oos)

        nptest.assert_equal(expected.ravel(), actual.iloc[0, :].ravel())
        self.assertEqual(actual.time_values()[0], X_ic.time_values()[-1])

        # -------------------------------------------------
        # test that if also works with datetime index

        datetime_index = np.arange(
            np.datetime64("2021-02-15"),
            np.datetime64("2021-02-15")
            + np.timedelta64(1, "s") * (cart_df.n_timesteps),
            np.timedelta64(1, "s"),
        )
        np.hstack([datetime_index, datetime_index])
        cart_df.index = pd.MultiIndex.from_arrays(
            [
                cart_df.index.get_level_values(0),
                np.hstack([datetime_index, datetime_index]),
            ]
        )
        cart_df_oos.index = pd.MultiIndex.from_arrays(
            [cart_df_oos.index.get_level_values(0), datetime_index]
        )

        edmd_dateindex = deepcopy(edmd)
        edmd_dateindex.fit(cart_df)

        # modes @ eigenfunctions(X_ic) is the reconstructed initial state
        X_ic = cart_df_oos.initial_states(edmd_floatindex.n_samples_ic_)
        expected = (
            edmd_dateindex.koopman_modes.to_numpy()
            @ edmd_dateindex.koopman_eigenfunction(X_ic).to_numpy().T
        )
        expected = np.real(expected)
        actual = edmd_dateindex.reconstruct(cart_df_oos)

        self.assertTrue(actual.index.get_level_values("time").dtype.kind == "M")
        nptest.assert_equal(expected.ravel(), actual.iloc[0, :].ravel())
        self.assertEqual(actual.time_values()[0], X_ic.time_values()[-1])

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

        # TODO: make regression test (make sure that the result is replicated)

        from datafold.dynfold import DiffusionMaps
        from datafold.pcfold import ConeKernel  # MultiquadricKernel
        from datafold.pcfold import (
            ContinuousNNKernel,
            GaussianKernel,
            InverseMultiquadricKernel,
        )

        kernels = [
            ConeKernel(zeta=0.0, epsilon=0.1),
            ConeKernel(zeta=0.9, epsilon=0.1),
            GaussianKernel(epsilon=20),
            ContinuousNNKernel(k_neighbor=20, delta=0.9),
            # MultiquadricKernel(epsilon=0.05),
            InverseMultiquadricKernel(epsilon=0.5),
        ]

        if plot:
            f, ax = plt.subplots(nrows=len(kernels) + 1, ncols=1, sharex=True)
            X.plot(ax=ax[0])
            ax[0].set_title("original data")

        for i, kernel in enumerate(kernels):
            try:
                X_predict = EDMD(
                    [
                        ("takens", TSCTakensEmbedding(delays=20)),
                        ("dmap", DiffusionMaps(kernel, n_eigenpairs=100)),
                    ]
                ).fit_predict(X)

                if plot:
                    X_predict.plot(ax=ax[i + 1])
                    ax[i + 1].set_title(f"kernel={kernel}")

            except Exception as e:
                print(f"kernel={kernel} failed")
                raise e

        plt.show()

    def test_include_id_states(self):
        edmd1 = EDMD(
            dict_steps=[
                ("id", TSCIdentity(rename_features=True)),
            ],
            include_id_state=False,
        )

        edmd2 = EDMD(
            dict_steps=[
                ("id", TSCIdentity(rename_features=False)),
            ],
            include_id_state=True,
        )

        edmd3 = EDMD(
            dict_steps=[
                ("id", TSCIdentity(rename_features=False)),
            ],
            include_id_state=False,
        )

        edmd4 = EDMD(
            dict_steps=[
                ("id", TSCIdentity(rename_features=True)),
            ],
            include_id_state=True,
        )

        data = self.sine_wave_tsc

        actual1 = edmd1.fit_transform(data)
        actual2 = edmd2.fit_transform(data)
        actual3 = edmd3.fit_transform(data)
        actual4 = edmd4.fit_transform(data)

        # for the first a linear regression is solved, for the others a projection onto the ID
        # states is set up
        self.assertIsInstance(edmd1._inverse_map, np.ndarray)
        self.assertIsInstance(edmd2._inverse_map, scipy.sparse.csr_matrix)
        self.assertIsInstance(edmd3._inverse_map, scipy.sparse.csr_matrix)
        self.assertIsInstance(edmd4._inverse_map, scipy.sparse.csr_matrix)

        def _sin_column_is_in(sol):
            return np.any(np.isin("sin", sol.columns))

        self.assertFalse(_sin_column_is_in(actual1))
        self.assertTrue(_sin_column_is_in(actual2))
        self.assertTrue(_sin_column_is_in(actual3))
        self.assertTrue(_sin_column_is_in(actual4))

    def test_edmd_transform01(self):
        edmd = EDMD(
            dict_steps=[
                ("id", TSCIdentity(rename_features=False)),
            ],
            include_id_state=False,
        )
        actual = edmd.fit_transform(self.sine_wave_tsc)
        expected = self.sine_wave_tsc

        pdtest.assert_frame_equal(actual, expected)

    def test_edmd_transform02(self):
        edmd = EDMD(
            dict_steps=[
                ("id", TSCIdentity(rename_features=False)),
            ],
            include_id_state=False,
        )
        edmd = edmd.fit(self.sine_wave_tsc)

        # test if transform is inferred right
        X_tsc = self.sine_wave_tsc.to_numpy()
        actual = edmd.transform(X_tsc)

        self.assertIsInstance(actual, TSCDataFrame)
        self.assertEqual(actual.delta_time, edmd.dt_)

        expected = self.sine_wave_tsc
        pdtest.assert_frame_equal(actual, expected)

    def test_separate_target_data01(self):
        edmd1 = EDMD(
            dict_steps=[
                ("id", TSCIdentity(rename_features=True)),
            ],
            include_id_state=False,
        )

        edmd2 = deepcopy(edmd1)

        expected = edmd2.fit_transform(self.sine_wave_tsc)
        actual = edmd1.fit_transform(self.sine_wave_tsc, y=self.sine_wave_tsc)
        pdtest.assert_frame_equal(expected, actual)

    def test_separate_target_data02(self, plot=False):
        edmd1 = EDMD(
            dict_steps=[
                ("delay", TSCTakensEmbedding(delays=2)),
            ],
            include_id_state=False,
        )

        target = TSCDataFrame.from_same_indices_as(
            self.sine_wave_tsc,
            values=np.cos(self.sine_wave_tsc.time_values()),
            except_columns=["cos"],
        )

        edmd1.fit(self.sine_wave_tsc, y=target)
        actual_predict = edmd1.predict(
            self.sine_wave_tsc.head(3), time_values=self.sine_wave_tsc.time_values()[2:]
        )

        expected_error = TSCMetric(metric="l2", mode="feature")(
            actual_predict, target.iloc[2:, :]
        )[0]

        # detect changes in numerics, update if necessary
        self.assertLessEqual(expected_error, 7.36e-06)
        self.assertTrue(actual_predict.columns[0] == "cos")

        if plot:
            plt.plot(
                actual_predict.time_values(),
                actual_predict.iloc[:, 0],
                label="predict (cos)",
            )
            plt.plot(target.time_values(), target.iloc[:, 0], label="target (cos)")
            plt.plot(
                self.sine_wave_tsc.time_values(),
                self.sine_wave_tsc.iloc[:, 0],
                "input (sin)",
            )
            plt.show()

    def test_dict_preserved_id_state(self):
        edmd1 = EDMD(
            dict_steps=[
                ("id", TSCIdentity(rename_features=True)),
            ],
            include_id_state=False,
            dict_preserves_id_state="infer",
        )

        edmd2 = EDMD(
            dict_steps=[
                ("id", TSCIdentity(rename_features=False)),
            ],
            include_id_state=True,  # has no effect
            dict_preserves_id_state=True,
        )

        edmd3 = EDMD(
            dict_steps=[
                ("id", TSCIdentity(rename_features=False)),
            ],
            include_id_state=False,
            dict_preserves_id_state=True,
        )

        edmd4 = EDMD(
            dict_steps=[
                ("id", TSCIdentity(rename_features=True)),
            ],
            include_id_state=True,
            dict_preserves_id_state=True,
        )

        data = self.sine_wave_tsc

        actual1 = edmd1.fit_transform(data)
        actual2 = edmd2.fit_transform(data)
        actual3 = edmd3.fit_transform(data)

        with self.assertRaises(ValueError):
            edmd4.fit_transform(data)

        # for the first a linear regression is solved, for the others a projection onto the ID
        # states is set up
        self.assertIsInstance(edmd1._inverse_map, np.ndarray)
        self.assertIsInstance(edmd2._inverse_map, scipy.sparse.csr_matrix)
        self.assertIsInstance(edmd3._inverse_map, scipy.sparse.csr_matrix)

        def _sin_column_is_in(sol):
            return np.any(np.isin("sin", sol.columns))

        self.assertFalse(_sin_column_is_in(actual1))
        self.assertTrue(_sin_column_is_in(actual2))
        self.assertTrue(_sin_column_is_in(actual3))

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
            dmd_model=DMDStandard(sys_mode="spectral"),
        )

        _edmd_matrix = EDMD(
            dict_steps=[
                ("scale", TSCFeaturePreprocess.from_name(name="min-max")),
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=2)),
            ],
            dmd_model=DMDStandard(sys_mode="matrix"),
        )

        actual_spectral = _edmd_spectral.fit_predict(X=self.sine_wave_tsc)
        actual_matrix = _edmd_matrix.fit_predict(X=self.sine_wave_tsc)

        pdtest.assert_frame_equal(actual_spectral, actual_matrix)

        self.assertTrue(_edmd_spectral.koopman_modes is not None)
        self.assertTrue(_edmd_spectral.dmd_model.is_spectral_mode)

        self.assertTrue(_edmd_matrix.koopman_modes is None)
        self.assertTrue(_edmd_matrix.dmd_model.is_matrix_mode)

        # use qois argument
        actual_spectral = _edmd_spectral.reconstruct(X=self.sine_wave_tsc, qois=["sin"])
        actual_matrix = _edmd_matrix.reconstruct(X=self.sine_wave_tsc, qois=["sin"])
        pdtest.assert_frame_equal(actual_spectral, actual_matrix)

    @pytest.mark.filterwarnings("ignore:Shift matrix")
    def test_edmd_with_composed_dict(self, display_html=False, plot=False):
        # Ignore warning, because none of the other configurations results in a
        # satisfying reconstruction result.

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

        if plot:
            f, ax = plt.subplots(nrows=2, sharex=True)
            self.multi_waves.plot(ax=ax[0])
            edmd.reconstruct(self.multi_waves).plot(ax=ax[1])

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

    def test_streaming_dmd_multi_sine(self, plot=False):
        data = self._setup_multi_sine_wave_data(3000)

        _tse = TSCTakensEmbedding(delays=150)

        edmd = EDMD(
            dict_steps=[("delay", _tse)],
            dmd_model=StreamingDMD(),
        )

        test_batches = []
        test_batches_dmd = []

        tv_batches = np.array_split(data.time_values(), 10)

        apply_dmd = True
        for i in range(len(tv_batches) - 1):
            current_time_values = tv_batches[i]
            next_time_values = tv_batches[i + 1]
            X_train = data.loc[pd.IndexSlice[:, current_time_values], :]
            X_test = data.loc[pd.IndexSlice[:, next_time_values], :]

            if apply_dmd:
                dmd = DMDStandard().fit(_tse.fit_transform(X_train))
                test_batches_dmd.append(
                    dmd.reconstruct(_tse.fit_transform(X_test)).loc[:, "sin"]
                )

            edmd = edmd.partial_fit(X_train)
            test_batches.append(edmd.reconstruct(X_test))

        # test that eigenvalues are either close to zero or one
        # (values can be adapted if only slightly adaptations are necessary)
        idx_close_one = np.abs(np.abs(edmd.koopman_eigenvalues) - 1) < 3.5e-4
        idx_close_zero = np.abs(edmd.koopman_eigenvalues) < 1e-15

        self.assertTrue(np.all(np.logical_or(idx_close_one, idx_close_zero)))

        if plot:
            plot_eigenvalues(edmd.dmd_model.eigenvalues_, plot_unit_circle=True)

            ax = data.plot()
            for b in test_batches:
                b.plot(ax=ax, c="red")

            if apply_dmd:
                ax = data.plot()
                for b in test_batches_dmd:
                    b.plot(ax=ax, c="green")

            plt.show()

    def test_streaming_dmd(self, plot=False):
        data = self._setup_sine_wave_data(end=4 * np.pi)

        for dmd_model in [StreamingDMD(), OnlineDMD()]:
            edmd = EDMD(
                dict_steps=[("delay", TSCTakensEmbedding(delays=2))],
                dmd_model=dmd_model,
            )

            batches = np.array_split(data, 2)

            predicts_train = []
            predicts_test = []

            for i in range(len(batches) - 1):
                fit_batch = batches[i]
                reconstruct_batch = batches[i + 1]

                edmd = edmd.partial_fit(fit_batch)

                predicts_train.append(edmd.reconstruct(fit_batch))
                predicts_test.append(edmd.reconstruct(reconstruct_batch))

            if plot:
                ax = data.plot()
                ax.set_title(f"{dmd_model=}")
                for b in predicts_train:
                    b.plot(ax=ax, c="green")

                for b in predicts_test:
                    b.plot(ax=ax, c="red")

        plt.show()

    def test_edmdcontrol_pipe(self):
        from scipy.special import comb

        n_delays = 2
        n_degrees = 2
        sim_num_steps = 10
        lag = 5
        X_tsc, U_tsc = EDMDTest.setup_inverted_pendulum(sim_num_steps=sim_num_steps)

        dict_steps = [
            ("takens", TSCTakensEmbedding(delays=n_delays, lag=lag)),
            ("poly", TSCPolynomialFeatures(degree=n_degrees, include_first_order=True)),
        ]

        edmd = EDMD(
            dict_steps=dict_steps, include_id_state=False, dmd_model=DMDControl()
        )

        n_latent_states = X_tsc.shape[1] * (n_delays + 1)

        edmd.fit(X=X_tsc, U=U_tsc)
        actual_transform = edmd.transform(X_tsc)

        n_final = comb(n_latent_states, n_degrees) + 2 * n_latent_states

        self.assertEqual(len(actual_transform.columns), n_final)
        self.assertEqual(
            len(actual_transform.time_values()), sim_num_steps - lag - n_delays
        )

        X_ic = X_tsc.initial_states(edmd.n_samples_ic_)
        U = U_tsc.final_states(2)  # align with initial condition

        actual_predict = edmd.predict(X_ic, U=U)
        actual_predict2 = edmd.predict(
            X_ic, U=U, time_values=actual_transform.time_values()
        )

        nptest.assert_allclose(
            actual_predict.time_values(),
            actual_transform.time_values(),
            atol=1e-15,
            rtol=0,
        )
        pdtest.assert_frame_equal(actual_predict, actual_predict2)

    def test_edmdcontrol_id(self):
        ic = np.array([0, 0, np.pi, 0])
        X_tsc, U_tsc = EDMDTest.setup_inverted_pendulum(training_size=1)

        dmdc = DMDControl()
        dmdc.fit(X_tsc, U=U_tsc)

        edmdid = EDMD(
            dict_steps=[
                ("id", TSCIdentity()),
            ],
            include_id_state=False,
            dmd_model=DMDControl(),
        )
        edmdid.fit(X_tsc, U=U_tsc)

        self.assertIsInstance(ic, np.ndarray)

        expected = dmdc.predict(X=ic, U=U_tsc)
        actual = edmdid.predict(X=ic, U=U_tsc)

        pdtest.assert_frame_equal(expected, actual)

    def test_edmd_timedelay_stepwise_transform(self):
        expected_edmd = EDMD(
            dict_steps=[("delay", TSCTakensEmbedding(delays=2))],
            stepwise_transform=False,
        )
        expected_edmd = expected_edmd.fit(self.multi_sine_wave_tsc)

        actual_edmd = EDMD(
            dict_steps=[("delay", TSCTakensEmbedding(delays=2))],
            stepwise_transform=True,
        )
        actual_edmd = actual_edmd.fit(self.multi_sine_wave_tsc)

        expected = expected_edmd.reconstruct(self.multi_sine_wave_tsc)
        actual = actual_edmd.reconstruct(self.multi_sine_wave_tsc)

        pdtest.assert_index_equal(expected.index, actual.index)
        pdtest.assert_index_equal(expected.columns, actual.columns)

        win_size = expected_edmd.n_samples_ic_ + 1
        X_windows = TSCDataFrame.from_frame_list(
            list(
                self.multi_sine_wave_tsc.tsc.iter_timevalue_window(
                    window_size=win_size, offset=win_size
                )
            )
        )

        self.assertEqual(X_windows.n_timesteps, win_size)

        expected = expected_edmd.reconstruct(X_windows)
        actual = actual_edmd.reconstruct(X_windows)

        print(expected_edmd.score(self.multi_sine_wave_tsc))
        print(actual_edmd.score(self.multi_sine_wave_tsc))

        pdtest.assert_frame_equal(expected, actual)

    def test_edmd_control_stepwise_transform(self):
        X_tsc, U_tsc = EDMDTest.setup_inverted_pendulum()

        # make the data only snapshots of 2 steps, so that the code with
        # stepwise transform=True gives the same result than with
        # stepwise_transform=False
        X_tsc = X_tsc.drop(
            X_tsc.groupby(TSCDataFrame.tsc_id_idx_name).tail(1).index, axis=0
        )
        X_now, X_next = X_tsc.tsc.shift_matrices(snapshot_orientation="row")
        U_now, _ = U_tsc.tsc.shift_matrices(snapshot_orientation="row")

        X_tsc = TSCDataFrame.from_shift_matrices(
            left_matrix=X_now,
            right_matrix=X_next,
            snapshot_orientation="row",
            columns=X_tsc.columns,
        )
        U_index = X_tsc.groupby("ID").head(1).index
        U_tsc = TSCDataFrame(U_now, index=U_index, columns=U_tsc.columns)

        # two different procedures to get the same result
        stepwise_transform_expext = False
        stepwise_transform_actual = True

        X_exp = EDMD(
            dict_steps=[("id", TSCIdentity())],
            include_id_state=False,
            stepwise_transform=stepwise_transform_expext,
            dmd_model=DMDControl(),
        ).fit_predict(X=X_tsc, U=U_tsc)

        X_act = EDMD(
            dict_steps=[("id", TSCIdentity())],
            include_id_state=False,
            stepwise_transform=stepwise_transform_actual,
            dmd_model=DMDControl(),
        ).fit_predict(X=X_tsc, U=U_tsc)

        pdtest.assert_frame_equal(X_exp, X_act)

    def test_edmdcontrol_reconstruct(self):
        X_tsc, U_tsc = EDMDTest.setup_inverted_pendulum()

        dmdc = DMDControl()
        edmdid = EDMD(
            dict_steps=[
                ("id", TSCIdentity()),
            ],
            include_id_state=False,
            dmd_model=DMDControl(),
        )

        expected = dmdc.fit_predict(X_tsc, U=U_tsc)
        actual = edmdid.fit_predict(X_tsc, U=U_tsc)
        pdtest.assert_frame_equal(expected, actual)


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


class HAVOKTest(unittest.TestCase):
    def test_lorenz(self, plot=False):
        # TODO: try using TSCDataframe with fixed delta time and see how it works...

        from datafold.appfold.edmd import HAVOK
        from datafold.utils._systems import Lorenz

        dt = 0.001
        time_values = np.arange(0, 200 + 1e-15, dt)

        X = Lorenz(atol=1e-12, rtol=1e-12).predict([-8, 8, 27], time_values=time_values)

        n_samples = X.shape[0]

        # TODO: for now I work with integer time values as the other is not considered as
        #  equally spaced (numerical noise)
        # TODO: in future try to either fix this (see test_linspace_unique_delta_times) or
        #  check whether the fixed_delta_time within TSCDataFrame works in EDMD properly
        #  (requires a separate test)
        X.index = pd.MultiIndex.from_arrays([np.zeros(n_samples), np.arange(n_samples)])
        X = X.iloc[:, [0]]  # only on the first component

        havok = HAVOK(rank=15, delays=99).fit(X=X)

        if plot:
            f, ax = plt.subplots(ncols=2)

            ax[0].imshow(havok.state_matrix, aspect="equal", cmap=plt.get_cmap("RdBu"))
            ax[1].imshow(
                havok.control_matrix, aspect="equal", cmap=plt.get_cmap("RdBu")
            )

            plt.figure()
            plt.plot(
                havok.forcing_signal.time_values(),
                havok.forcing_signal.to_numpy().ravel(),
            )
            plt.xlim(0, 25000)
            plt.show()


if __name__ == "__main__":
    test = EDMDTest()
    test.setUp()

    test.test_edmd_cv_sine_wave()
