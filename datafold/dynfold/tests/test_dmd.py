#!/usr/bin/env python3

import unittest
import unittest.mock as mock

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest
import scipy.linalg

from datafold.dynfold import TSCTakensEmbedding
from datafold.dynfold.dmd import (
    DMDEco,
    DMDFull,
    LinearDynamicalSystem,
    OnlineDMD,
    PyDMDWrapper,
    StreamingDMD,
    gDMDFull,
)
from datafold.pcfold import TSCDataFrame
from datafold.utils.general import (
    assert_equal_eigenvectors,
    is_df_same_index,
    sort_eigenpairs,
)
from datafold.utils.plot import plot_eigenvalues


class LinearDynamicalSystemTest(unittest.TestCase):
    def _set_attrs_flowmap_system(self):

        # A too small time_delta increases error (its only a first order finite diff
        # scheme). Smallest time_delta by testing 1e-8
        self.time_delta_approx = 1e-8

        # Is \dot{x} = A x (i.e., A = generator_matrix)
        self.generator_matrix = np.array([[1, -3], [-2, 1]])

        # This is basically, that the Koopman matrix is
        # A = (K - I) / time_delta, now we can simply use a small time_delta, because
        # we have the true generator matrix A given
        self.dyn_matrix_flowmap = (
            np.eye(self.generator_matrix.shape[0])
            + self.time_delta_approx * self.generator_matrix
        )

        self.eigvals_flowmap, self.eigvec_right_flowmap = np.linalg.eig(
            self.dyn_matrix_flowmap
        )
        self.eigvec_left = np.linalg.solve(
            self.eigvec_right_flowmap * self.eigvals_flowmap, self.dyn_matrix_flowmap
        )

        # Check that diagonalization of the flowmap matrix is correct
        nptest.assert_allclose(
            self.eigvec_right_flowmap
            @ np.diag(self.eigvals_flowmap)
            @ self.eigvec_left,
            self.dyn_matrix_flowmap,
            atol=1e-15,
            rtol=0,
        )

    def setUp(self) -> None:
        self._set_attrs_flowmap_system()

    def test_approx_continuous_linear_system(self, plot=False):

        n_timesteps = 200
        time_values = np.linspace(0, 1, n_timesteps)

        ic = np.array([[3], [2]])

        expected = np.zeros([n_timesteps, self.generator_matrix.shape[0]])
        for i, t in enumerate(time_values):
            expected[i, :] = np.real(
                (scipy.linalg.expm(t * self.generator_matrix) @ ic).ravel()
            )

        actual = (
            LinearDynamicalSystem(
                sys_type="flowmap", sys_mode="spectral", time_invariant=True
            )
            .setup_spectral_system(
                eigenvectors_right=self.eigvec_right_flowmap,
                eigenvalues=self.eigvals_flowmap,
            )
            .evolve_system(
                initial_conditions=self.eigvec_left @ ic,
                time_values=time_values,
                time_delta=self.time_delta_approx,
            )
        )

        nptest.assert_allclose(expected, actual.to_numpy(), atol=1e-6, rtol=1e-14)

        if plot:
            f, ax = plt.subplots(2, 1)

            expected = TSCDataFrame.from_same_indices_as(actual, expected)

            expected.plot(ax=ax[0])
            actual.plot(ax=ax[1])

            plt.show()

    def test_expm_vs_spectral(self):
        # test if solution with matrix exponential matches the solution with spectral
        # decomposition

        ic = np.random.default_rng(1).uniform(size=self.generator_matrix.shape[0])
        time_values = np.linspace(0, 5, 10)

        expected = np.zeros([time_values.shape[0], ic.shape[0]])
        for i, t in enumerate(time_values):
            expected[i, :] = scipy.linalg.expm(self.generator_matrix * t) @ ic

        evals, evec = np.linalg.eig(self.generator_matrix)
        ic_adapted = np.linalg.lstsq(evec, ic, rcond=None)[0]

        actual = (
            LinearDynamicalSystem(sys_type="differential", sys_mode="spectral")
            .setup_spectral_system(eigenvectors_right=evec, eigenvalues=evals)
            .evolve_system(
                initial_conditions=ic_adapted,
                time_values=time_values,
            )
        )

        # errors can be introduced by the least square solution
        nptest.assert_allclose(actual.to_numpy(), expected, atol=1e-8, rtol=1e-13)

    def test_time_values(self):

        time_values = np.random.default_rng(1).uniform(size=100) * 100

        actual = (
            LinearDynamicalSystem(
                sys_type="flowmap", sys_mode="spectral", time_invariant=True
            )
            .setup_spectral_system(
                eigenvectors_right=self.eigvec_right_flowmap,
                eigenvalues=self.eigvals_flowmap,
            )
            .evolve_system(
                initial_conditions=self.eigvec_left @ np.ones(shape=[2, 1]),
                time_values=time_values,
                # to match up the flowmap system we have to assume a time delta of 1
                time_delta=1,
            )
        )

        nptest.assert_array_equal(actual.time_values(), np.sort(time_values))

    def test_multi_initial_conditions(self):

        n_timeseries = 10
        initial_conditions = np.random.default_rng(1).uniform(size=(2, n_timeseries))

        time_values = np.linspace(0, 20, 100)

        actual = (
            LinearDynamicalSystem(
                sys_type="flowmap", sys_mode="spectral", time_invariant=True
            )
            .setup_spectral_system(
                eigenvectors_right=self.eigvec_right_flowmap,
                eigenvalues=self.eigvals_flowmap,
            )
            .evolve_system(
                initial_conditions=self.eigvec_left @ initial_conditions,
                time_delta=1,
                time_values=time_values,
                feature_names_out=["A", "B"],
            )
        )

        self.assertEqual(actual.n_timesteps, len(time_values))
        self.assertEqual(actual.n_timeseries, n_timeseries)
        self.assertEqual(actual.columns.tolist(), ["A", "B"])
        nptest.assert_array_equal(actual.time_values(), time_values)

    def test_feature_columns(self):

        actual = (
            LinearDynamicalSystem(
                sys_type="flowmap", sys_mode="spectral", time_invariant=True
            )
            .setup_spectral_system(
                eigenvectors_right=self.eigvec_right_flowmap,
                eigenvalues=self.eigvals_flowmap,
            )
            .evolve_system(
                initial_conditions=self.eigvec_left @ np.ones(shape=[2, 1]),
                time_values=np.arange(4),
                # to match up the discrete system we have to assume a time delta of 1
                time_delta=1,
                feature_names_out=["expectedA", "expectedB"],
            )
        )

        self.assertEqual(actual.columns.tolist(), ["expectedA", "expectedB"])

        with self.assertRaises(ValueError):
            LinearDynamicalSystem(
                sys_type="flowmap", sys_mode="spectral"
            ).setup_spectral_system(
                eigenvectors_right=self.eigvec_right_flowmap,
                eigenvalues=self.eigvals_flowmap,
            ).evolve_system(
                initial_conditions=self.eigvec_left @ np.ones(shape=[2, 1]),
                time_values=np.arange(4),
                # to match up the discrete system we have to assume a time delta of 1
                time_delta=1,
                feature_names_out=[1, 2, 3],
            )

    def test_return_types(self):
        actual = (
            LinearDynamicalSystem(
                sys_type="flowmap", sys_mode="spectral", time_invariant=True
            )
            .setup_spectral_system(
                eigenvectors_right=self.eigvec_right_flowmap,
                eigenvalues=self.eigvals_flowmap,
            )
            .evolve_system(
                initial_conditions=self.eigvec_left @ np.ones(shape=[2, 2]),
                time_values=np.arange(1),
                # to match up the discrete system we have to assume a time delta of 1
                time_delta=1,
            )
        )

        # Is a TSCDataFrame, also for single time steps
        self.assertIsInstance(actual, TSCDataFrame)
        self.assertTrue(actual.has_degenerate())

        actual = (
            LinearDynamicalSystem(
                sys_type="flowmap", sys_mode="spectral", time_invariant=True
            )
            .setup_spectral_system(
                eigenvectors_right=self.eigvec_right_flowmap,
                eigenvalues=self.eigvals_flowmap,
            )
            .evolve_system(
                initial_conditions=self.eigvec_left @ np.ones(shape=[2, 2]),
                time_values=np.arange(2),
                # to match up the discrete system we have to assume a time delta of 1
                time_delta=1,
            )
        )

        # is a TSCDataFrame, because more than two time values are a time series
        self.assertIsInstance(actual, TSCDataFrame)


class DMDTest(unittest.TestCase):
    def _create_random_tsc(self, dim, n_samples):
        data = np.random.default_rng(1).normal(size=(n_samples, dim))
        data = pd.DataFrame(data, columns=np.arange(dim))
        return TSCDataFrame.from_single_timeseries(data)

    @classmethod
    def _create_harmonic_tsc(cls, n_samples, dim):
        x_eval = np.linspace(0, 2, n_samples)

        col_stacks = []

        counter = 1
        for i in range(dim):
            if np.mod(i, 2) == 0:
                col_stacks.append(np.cos(x_eval * 2 * np.pi / counter))
            else:
                col_stacks.append(np.sin(x_eval * 2 * np.pi / counter))
            counter += 1

        data = pd.DataFrame(np.column_stack(col_stacks))
        return TSCDataFrame.from_single_timeseries(data)

    def test_dmd_eigenpairs(self):
        # From
        # http://www.astronomia.edu.uy/progs/algebra/Linear_Algebra,_4th_Edition__(2009)Lipschutz-Lipson.pdf # noqa
        # page 297 Example 9.5

        dmd_model = DMDFull(is_diagonalize=True)

        mock_koopman_matrix = np.array([[3.0, 1], [2, 2]])
        dmd_model._compute_koopman_matrix = mock.MagicMock(
            return_value=mock_koopman_matrix
        )

        dmd_model = dmd_model.fit(self._create_random_tsc(n_samples=2, dim=2))

        expected_eigenvalues = np.array([4.0, 1.0])  # must be sorted descending
        nptest.assert_array_equal(expected_eigenvalues, dmd_model.eigenvalues_)

        expected_eigenvectors_right = np.array([[1, 1], [1, -2]])

        assert_equal_eigenvectors(
            expected_eigenvectors_right, dmd_model.eigenvectors_right_
        )

        expected_eigenvectors_left = np.array([[2 / 3, 1 / 3], [1 / 3, -1 / 3]])

        # NOTE: the left eigenvectors are transposed because they are
        # stored row-wise (whereas right eigenvectors are column-wise). The helper
        # function assert_equal_eigenvectors assumes column-wise orientation
        assert_equal_eigenvectors(
            expected_eigenvectors_left.T, dmd_model.eigenvectors_left_.T
        )

        # sanity check: diagonalization of mocked Koopman matrix
        nptest.assert_equal(
            expected_eigenvectors_right
            @ np.diag(expected_eigenvalues)
            @ expected_eigenvectors_left,
            mock_koopman_matrix,
        )

        actual = (
            dmd_model.eigenvectors_right_
            @ np.diag(dmd_model.eigenvalues_)
            @ dmd_model.eigenvectors_left_
        )

        nptest.assert_allclose(mock_koopman_matrix, actual, rtol=1e-15, atol=1e-15)

    def test_dmd_equivalence_generator_flowmap(self):
        test_data = self._create_random_tsc(n_samples=500, dim=30)

        generator_system = DMDFull(is_diagonalize=True, approx_generator=True).fit(
            test_data
        )
        flowmap_system = DMDFull(is_diagonalize=True, approx_generator=False).fit(
            test_data
        )

        self.assertEqual(generator_system.sys_type, "differential")
        self.assertEqual(flowmap_system.sys_type, "flowmap")

        time_values = np.linspace(0, 5, 20)

        generator_result = generator_system.predict(
            test_data.initial_states(), time_values
        )

        flowmap_result = flowmap_system.predict(test_data.initial_states(), time_values)

        pdtest.assert_frame_equal(generator_result, flowmap_result, rtol=0, atol=1e-15)

        # check that eigenvalues are actually different
        nptest.assert_allclose(
            np.exp(generator_system.eigenvalues_) * generator_system.dt_,
            flowmap_system.eigenvalues_,
            atol=1e-16,
        )

    def test_mode_equivalence_dmd(self):
        # test mode = matrix and mode = spectrum against
        # (both should obtain similar results)

        tsc_df = self._create_harmonic_tsc(100, 2)
        tsc_df = TSCTakensEmbedding(delays=1).fit_transform(tsc_df)

        # for flowmap case
        first = DMDFull(
            sys_mode="spectral", is_diagonalize=False, approx_generator=False
        ).fit_predict(tsc_df)

        second = DMDFull(sys_mode="matrix", approx_generator=False).fit_predict(tsc_df)

        pdtest.assert_frame_equal(first, second, rtol=1e-16, atol=1e-12)

        # for generator case
        first = DMDFull(
            sys_mode="spectral", is_diagonalize=False, approx_generator=True
        ).fit_predict(tsc_df)
        second = DMDFull(sys_mode="matrix", approx_generator=True).fit_predict(tsc_df)

        pdtest.assert_frame_equal(first, second, rtol=1e-16, atol=1e-11)

    def test_mode_equivalence_gdmd(self):
        # test mode = matrix and mode = spectrum against
        # (both should obtain similar results)

        tsc_df = self._create_harmonic_tsc(100, 2)
        tsc_df = TSCTakensEmbedding(delays=1).fit_transform(tsc_df)

        first = gDMDFull(sys_mode="spectral", is_diagonalize=False).fit_predict(tsc_df)
        second = gDMDFull(sys_mode="matrix").fit_predict(tsc_df)

        pdtest.assert_frame_equal(first, second, rtol=1e-16, atol=1e-12)

    def test_dmd_vs_gdmd(self, plot=False):
        # need to time embed "standing waves" to be able to reconstruct it
        tsc_df = self._create_harmonic_tsc(100, 2)
        tsc_df = TSCTakensEmbedding(delays=1).fit_transform(tsc_df)

        first = DMDFull(is_diagonalize=True, approx_generator=True).fit(
            tsc_df,
        )
        # extremely high accuracy to get to a similar error
        second = gDMDFull(
            is_diagonalize=True, kwargs_fd=dict(scheme="center", accuracy=16)
        ).fit(tsc_df)

        score_dmd = first.score(tsc_df)
        score_gdmd = second.score(tsc_df)

        # also fails if there are changes in the implementation that includes small
        # numerical noise
        self.assertLessEqual(np.abs(score_dmd - score_gdmd), 9.81e-11)

        if plot:
            print(score_dmd)
            print(score_gdmd)

            from datafold.utils.plot import plot_eigenvalues

            ax = plot_eigenvalues(first.eigenvalues_)
            plot_eigenvalues(second.eigenvalues_, ax=ax)

            f, ax = plt.subplots(nrows=3)
            tsc_df.plot(ax=ax[0])
            ax[0].set_title("original")
            first.reconstruct(tsc_df).plot(ax=ax[1])
            ax[1].set_title("reconstructed DMD")
            second.reconstruct(tsc_df).plot(ax=ax[2])
            ax[2].set_title("reconstructed gDMD")
            plt.show()

    def test_dmd_pydmd1(self):

        test_data = self._create_random_tsc(n_samples=500, dim=30)
        pydmd = PyDMDWrapper(
            method="dmd", svd_rank=1000, tlsq_rank=0, exact=True, opt=False
        ).fit(test_data)

        # datafold and PyDMD have a different way to order the eigenvalues. For
        # the test we sort both according to the complex eigenvalue
        expected_eigenvalues, expected_modes = sort_eigenpairs(
            pydmd.eigenvalues_, pydmd.dmd_modes
        )

        dmd = DMDFull().fit(test_data)
        actual = dmd.dmd_modes

        nptest.assert_allclose(
            dmd.eigenvalues_,
            expected_eigenvalues,
            atol=1e-4,
            rtol=0,
        )

        assert_equal_eigenvectors(expected_modes, actual, tol=1e-15)

    def test_dmd_pydmd2(self):
        test_data = self._create_random_tsc(n_samples=500, dim=100)

        pydmd = PyDMDWrapper(
            method="dmd", svd_rank=10, tlsq_rank=0, exact=True, opt=False
        ).fit(test_data)
        expected = pydmd.dmd_modes

        dmd = DMDEco(svd_rank=10).fit(test_data)
        actual = dmd.dmd_modes

        nptest.assert_allclose(dmd.eigenvalues_, pydmd.eigenvalues_, atol=1e-15, rtol=0)

        assert_equal_eigenvectors(expected, actual, tol=1e-15)

    def test_reconstruct_indices(self):

        expected_indices = self._create_random_tsc(n_samples=100, dim=10)
        actual_indices = DMDFull(is_diagonalize=False).fit_predict(expected_indices)

        is_df_same_index(
            actual_indices,
            expected_indices,
            check_column=True,
            check_index=True,
            handle="raise",
        )

    def test_predict_indices(self):
        tsc_df_fit = self._create_random_tsc(n_samples=100, dim=10)
        predict_ic = self._create_random_tsc(n_samples=20, dim=10)

        dmd = DMDFull(is_diagonalize=False).fit(tsc_df_fit)

        # predict with array
        predict_ic = predict_ic.to_numpy()
        actual = dmd.predict(predict_ic)

        self.assertIsInstance(actual, TSCDataFrame)
        self.assertEqual(actual.n_timeseries, predict_ic.shape[0])
        self.assertEqual(actual.n_timesteps, 1)
        nptest.assert_array_equal(actual.ids, np.arange(predict_ic.shape[0]))

        # provide own time series IDs in the initial condition and own time values
        expected_ids = np.arange(0, predict_ic.shape[0] * 2, 2)
        expected_time_values = np.arange(500)

        predict_ic = TSCDataFrame(
            predict_ic,
            index=pd.MultiIndex.from_arrays(
                [
                    expected_ids,
                    np.zeros(predict_ic.shape[0]),
                ]
            ),
            columns=tsc_df_fit.columns,
        )

        actual = dmd.predict(predict_ic, time_values=expected_time_values)

        self.assertIsInstance(actual, TSCDataFrame)
        self.assertEqual(actual.n_timeseries, predict_ic.shape[0])
        self.assertEqual(actual.n_timesteps, len(expected_time_values))
        nptest.assert_array_equal(actual.time_values(), expected_time_values)
        nptest.assert_array_equal(actual.ids, expected_ids)

    def test_invalid_time_values(self):
        tsc_df_fit = self._create_random_tsc(n_samples=100, dim=10)
        predict_ic = self._create_random_tsc(n_samples=20, dim=10).to_numpy()

        dmd = DMDFull(is_diagonalize=False).fit(tsc_df_fit)

        time_values = np.arange(500, dtype=float)

        with self.assertRaises(ValueError):
            _values = time_values.copy()
            _values[0] = np.nan
            dmd.predict(predict_ic, _values)

        with self.assertRaises(ValueError):
            _values = time_values.copy()
            _values[-1] = np.inf
            dmd.predict(predict_ic, _values)

        with self.assertRaises(ValueError):
            _values = time_values.copy()
            _values[0] = -1
            dmd.predict(predict_ic, _values)

        with self.assertRaises(ValueError):
            _values = time_values.copy()
            _values = _values[::-1]
            dmd.predict(predict_ic, _values)

        with self.assertRaises(TypeError):
            _values = time_values.copy().astype(complex)
            _values[-1] = _values[-1] + 1j
            dmd.predict(predict_ic, _values)

        with self.assertRaises(ValueError):
            _values = time_values.copy()
            _values[0] = _values[1]
            dmd.predict(predict_ic, _values)

        with self.assertRaises(ValueError):
            _values = time_values.copy()[np.newaxis, :]
            dmd.predict(predict_ic, _values)

    def test_invalid_feature_names(self):
        tsc_df_fit = self._create_random_tsc(n_samples=100, dim=10)
        predict_ic = self._create_random_tsc(n_samples=1, dim=10)

        dmd = DMDFull(is_diagonalize=False).fit(tsc_df_fit)

        predict_ic.columns = pd.Index(np.arange(2, 12))

        with self.assertRaises(ValueError):
            dmd.predict(predict_ic)


class StreamingDMDTest(unittest.TestCase):
    def _snapshots(self, n_states, n_snaps, noise_cov=0.0):

        # Code part for test taken from
        # https://github.com/cwrowley/dmdtools/blob/master/python/tests/test_dmd.py
        # See also LICENSE_bundeled file for the copy of the BSD 3-Clause license

        dt = 0.01  # timestep

        # Define the example system
        v1 = np.random.default_rng(1).normal(size=n_states)
        v2 = np.random.default_rng(2).normal(size=n_states)
        v3 = np.random.default_rng(3).normal(size=n_states)
        v4 = np.random.default_rng(4).normal(size=n_states)

        # characteristic frequencies
        f1 = 5.2
        f2 = 1.0

        X = np.zeros([n_snaps, n_states])

        for k in range(n_snaps):
            shot = (
                v1 * np.cos(2 * np.pi * f1 * dt * k)
                + v2 * np.cos(2 * np.pi * f2 * dt * k)
                + v3 * np.sin(2 * np.pi * f1 * dt * k)
                + v4 * np.sin(2 * np.pi * f2 * dt * k)
            )
            X[k, :] = shot + np.sqrt(noise_cov) * np.random.default_rng(5).normal(
                size=n_states
            )

        return TSCDataFrame.from_array(X)

    def _generate_delayed_sine_wave(self, sigma_noise=0):
        t_eval = np.linspace(0, 2 * 8 * np.pi, 1000)
        data = np.sin(t_eval)
        data += np.random.default_rng(1).normal(
            loc=data, scale=sigma_noise, size=len(data)
        )

        df = TSCDataFrame.from_array(
            data[:, np.newaxis], time_values=t_eval, feature_names=["sin"]
        )
        df = TSCTakensEmbedding(delays=4).fit_transform(df)
        return df

    def test_sine_curve(self, plot=False):

        df = self._generate_delayed_sine_wave()

        batchsize = 200

        stream_dmd_classes = [
            (StreamingDMD(max_rank=None, ngram=5), {}),
            (OnlineDMD(weighting=1.0), dict(batch_initialize=True)),
        ]

        batches = np.array_split(df, df.shape[0] // batchsize)
        predict = []
        predict_fulldmd = []
        train = []

        for dmd, fit_params in stream_dmd_classes:

            for i in range(len(batches) - 1):
                fit_batch = batches[i]
                predict_batch = batches[i + 1]

                dmd_full = DMDFull().fit(pd.concat(batches[: i + 1], axis=0))
                expected_ev = dmd_full.eigenvalues_[:2]

                predict_fulldmd.append(
                    dmd_full.reconstruct(predict_batch).loc[:, "sin"]
                )

                dmd = dmd.partial_fit(fit_batch, **fit_params)
                actual_ev = dmd.eigenvalues_[:2]

                train.append(dmd.reconstruct(fit_batch).loc[:, "sin"])
                predict.append(dmd.reconstruct(predict_batch).loc[:, "sin"])

                nptest.assert_allclose(expected_ev, actual_ev, rtol=1e-15, atol=1e-12)

            if plot:
                ax = plot_eigenvalues(
                    dmd_full.eigenvalues_,
                    plot_unit_circle=True,
                    plot_kwargs=dict(marker="o", alpha=0.3, label="full"),
                )
                plot_eigenvalues(
                    dmd.eigenvalues_, ax=ax, plot_kwargs=dict(label="streaming")
                )
                plt.legend()

                ax = df.loc[:, ["sin"]].plot()

                for i, p in enumerate(predict_fulldmd):
                    ax.plot(
                        p.time_values(),
                        p.to_numpy(),
                        c="orange",
                        linewidth=3,
                        label=f"{dmd_full=}" if i == 0 else None,
                    )

                for i, p in enumerate(train):
                    ax.plot(
                        p.time_values(),
                        p.to_numpy(),
                        "-",
                        c="green",
                        label=f"{dmd=} train" if i == 0 else None,
                    )

                for i, p in enumerate(predict):
                    ax.plot(
                        p.time_values(),
                        p.to_numpy(),
                        "--",
                        c="red",
                        label=f"{dmd=} test" if i == 0 else None,
                    )

                plt.legend()

        if plot:
            plt.show()

    def test_pod_compression(self):
        df = self._generate_delayed_sine_wave()

        dmd = StreamingDMD(max_rank=1)
        dmd.partial_fit(df)

        self.assertEqual(len(dmd.eigenvalues_), 1)

    def test_compare_methods(self, plot=False):

        n_samples = 500
        n_states = 1000
        noise_cov = 10  # measurement noise covariance

        X = self._snapshots(n_states, n_samples, noise_cov)

        dmd_full = DMDFull()
        dmd_full.fit(X)

        dmd_partial = StreamingDMD(max_rank=None)
        dmd_partial.partial_fit(X)

        # actual does not necessarily compute all possible eigenvalues
        # -> compare only leading eigenvalues from DMDFull
        ev_actual = dmd_partial.eigenvalues_
        ev_expected = dmd_full.eigenvalues_[: len(ev_actual)]

        nptest.assert_allclose(ev_expected, ev_actual, rtol=1e-14, atol=1e-15)

        if plot:
            print("standard:")
            print(ev_expected)
            print("\nstreaming:")
            print(ev_actual)

            ax = plot_eigenvalues(
                ev_expected,
                plot_unit_circle=True,
                plot_kwargs=dict(marker="+", markersize=10, c="red", label="streaming"),
            )
            ax = plot_eigenvalues(
                ev_actual,
                ax=ax,
                plot_kwargs=dict(marker="o", markersize=3, c="blue", label="full"),
            )

            ax.set_title("DMD eigenvalues")
            ax.set_xlabel(r"$\Re(\lambda)$")
            ax.set_ylabel(r"$\Im(\lambda)$")
            plt.legend()

            plt.show()


class TestOnlineDMD(unittest.TestCase):
    def test_online_dmd(self):
        """Test taken and adapted from
        https://github.com/haozhg/odmd/blob/master/tests/test_online.py. For license from
        "odmd" see LICENSE_bundeled file in datafold repository."""

        for n in range(2, 10):  # n -> state dimension
            T = 100 * n  # number of measurements

            # linear system matrix
            A = np.random.default_rng(2).normal(size=(n, n))

            now = np.random.default_rng(3).normal(size=(n, T))
            next = A.dot(now)

            X = TSCDataFrame.from_shift_matrices(now, next, snapshot_orientation="col")
            X.is_validate = False  # speeds up the iteration

            dmd = OnlineDMD(weighting=0.9)

            for tsc in X.itertimeseries(valid_tsc=True):
                dmd = dmd.partial_fit(tsc)
                if dmd.ready_:
                    self.assertLess(np.linalg.norm(dmd.A - A), 1e-6)

            actual = dmd.reconstruct(X)  # calls predict within
            pdtest.assert_frame_equal(X, actual, rtol=1e-13, atol=1e-10)

    def test_online_init_vs_full(self):
        X = DMDTest._create_harmonic_tsc(n_samples=500, dim=4)

        expected = DMDFull().fit(X)
        actual = OnlineDMD(weighting=1.0).partial_fit(X, batch_initialize=True)

        nptest.assert_allclose(
            expected.eigenvalues_, actual.eigenvalues_, rtol=1e-14, atol=1e-14
        )
        nptest.assert_allclose(
            expected.eigenvectors_right_,
            actual.eigenvectors_right_,
            rtol=1e-15,
            atol=1e-12,
        )

        # The weighing != 1.0 triggers a differen if case
        # (but should have essentially no effect)
        actual = OnlineDMD(weighting=1.0 - 1e-15).partial_fit(X, batch_initialize=True)

        nptest.assert_allclose(
            expected.eigenvectors_right_,
            actual.eigenvectors_right_,
            rtol=1e-15,
            atol=1e-12,
        )
        nptest.assert_allclose(
            expected.eigenvectors_right_,
            actual.eigenvectors_right_,
            rtol=1e-15,
            atol=1e-12,
        )
