#!/usr/bin/env python3

import unittest
import unittest.mock as mock

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest
import scipy.linalg
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from datafold.dynfold import TSCTakensEmbedding
from datafold.dynfold.dmd import (
    ControlledAffineDynamicalSystem,
    ControlledLinearDynamicalSystem,
    DMDControl,
    DMDEco,
    DMDFull,
    LinearDynamicalSystem,
    OnlineDMD,
    PyDMDWrapper,
    StreamingDMD,
    gDMDAffine,
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


class ControlledLinearDynamicalSystemTest(unittest.TestCase):
    def setUp(self) -> None:
        gen = np.random.default_rng(42)
        self.state_size = 4
        self.input_size = 2
        self.n_timesteps = 6
        self.A = gen.uniform(size=(self.state_size, self.state_size))
        self.x0 = gen.uniform(size=self.state_size)
        self.B = gen.uniform(size=(self.state_size, self.input_size))
        self.u = gen.uniform(size=(self.n_timesteps, self.input_size))
        self.t = np.linspace(0, self.n_timesteps - 1, self.n_timesteps)
        self.names = ["x" + str(i + 1) for i in range(self.state_size)]

        self.expected = np.zeros((self.n_timesteps, self.state_size))
        x = self.x0
        for idx, time in enumerate(self.t):
            self.expected[idx, :] = x
            x = self.A @ x + self.B @ self.u[idx, :]
        # do not write state past last since t starts at 0

    def test_controlled_system(self):
        actual = (
            ControlledLinearDynamicalSystem()
            .setup_matrix_system(self.A, self.B)
            .evolve_system(self.x0, self.u)
        )

        nptest.assert_allclose(actual.to_numpy(), self.expected, atol=1e-8, rtol=1e-13)

    def test_controlled_vs_linear(self):
        controlled = (
            ControlledLinearDynamicalSystem()
            .setup_matrix_system(self.A, self.B)
            .evolve_system(self.x0, np.zeros(self.u.shape))
        )
        linear = (
            LinearDynamicalSystem(sys_type="flowmap", sys_mode="matrix")
            .setup_matrix_system(self.A)
            .evolve_system(self.x0, self.t, time_delta=self.t[1])
        )

        nptest.assert_allclose(
            controlled.to_numpy(), linear.to_numpy(), atol=1e-8, rtol=1e-13
        )

    @unittest.skip(reason="Fractional timestep not implemented yet")
    def test_integer_vs_fractional(self):
        pass

    def test_time_delta(self):
        t = np.linspace(0, (self.n_timesteps - 1) / 2, self.n_timesteps)
        dt = 0.5

        actual = (
            ControlledLinearDynamicalSystem()
            .setup_matrix_system(self.A, self.B)
            .evolve_system(self.x0, self.u, t, time_delta=dt)
        )

        nptest.assert_allclose(actual.to_numpy(), self.expected, atol=1e-8, rtol=1e-13)

    def test_multi_initial_conditions(self):
        x0 = np.array([self.x0, self.x0]).T
        expected = np.vstack([self.expected, self.expected])
        ts_ids = np.array((10, 20))

        with self.assertRaises(ValueError):
            actual = ControlledLinearDynamicalSystem().evolve_system(
                x0, self.u, self.t, self.A, self.B, 1, ts_ids, self.names, True
            )
        u = np.stack([self.u, self.u])
        actual = ControlledLinearDynamicalSystem().evolve_system(
            x0, u, self.t, self.A, self.B, 1, ts_ids, self.names, False
        )

        nptest.assert_allclose(actual.to_numpy(), expected, atol=1e-8, rtol=1e-13)
        self.assertEqual(actual.n_timesteps, len(self.t))
        self.assertEqual(actual.n_timeseries, 2)
        self.assertEqual(actual.columns.tolist(), self.names)
        nptest.assert_array_equal(actual.time_values(), self.t)

    def test_feature_columns(self):

        actual = (
            ControlledLinearDynamicalSystem()
            .setup_matrix_system(self.A, self.B)
            .evolve_system(self.x0, self.u, feature_names_out=self.names)
        )

        self.assertEqual(actual.columns.tolist(), self.names)

        with self.assertRaises(ValueError):
            (
                ControlledLinearDynamicalSystem()
                .setup_matrix_system(self.A, self.B)
                .evolve_system(self.x0, self.u, feature_names_out=[1, 2, 3])
            )

    def test_return_types(self):
        actual = (
            ControlledLinearDynamicalSystem()
            .setup_matrix_system(self.A, self.B)
            .evolve_system(self.x0, self.u[:1], time_delta=1)
        )

        # Is a TSCDataFrame, also for single time steps
        self.assertIsInstance(actual, TSCDataFrame)
        self.assertTrue(actual.has_degenerate())

        actual = (
            ControlledLinearDynamicalSystem()
            .setup_matrix_system(self.A, self.B)
            .evolve_system(self.x0, self.u[:2])
        )

        # is a TSCDataFrame, because more than two time values are a time series
        self.assertIsInstance(actual, TSCDataFrame)

    def test_dimension_mismatch(self):
        # non-square system matrix
        with self.assertRaises(ValueError):
            ControlledLinearDynamicalSystem().setup_matrix_system(
                np.zeros((3, 4)), np.zeros((4, 4))
            )

        # mismatch between system and input matrix
        with self.assertRaises(ValueError):
            ControlledLinearDynamicalSystem().setup_matrix_system(
                np.zeros((4, 4)), np.zeros((3, 4))
            )

        # wrong initial condition
        with self.assertRaises(ValueError):
            ControlledLinearDynamicalSystem().setup_matrix_system(
                np.zeros((4, 4)), np.zeros((4, 3))
            ).evolve_system(np.zeros(3), np.zeros((4, 3)))

        # wrong control input
        with self.assertRaises(ValueError):
            ControlledLinearDynamicalSystem().setup_matrix_system(
                np.zeros((4, 4)), np.zeros((4, 3))
            ).evolve_system(np.zeros(4), np.zeros((3, 4)))

        # wrong time values
        with self.assertRaises(ValueError):
            ControlledLinearDynamicalSystem().setup_matrix_system(
                np.zeros((4, 4)), np.zeros((4, 3))
            ).evolve_system(np.zeros(3), np.zeros((4, 3)), np.zeros(3))


class ControlledAffineDynamicalSystemTest(unittest.TestCase):
    def setUp(self) -> None:
        gen = np.random.default_rng(42)
        self.state_size = 4
        self.input_size = 2
        self.n_timesteps = 6
        self.A = gen.uniform(-0.5, 0.5, size=(self.state_size, self.state_size))
        np.fill_diagonal(self.A, gen.uniform(-1.0, -0.5, size=self.state_size))
        self.x0 = gen.uniform(-1.0, 1.0, size=self.state_size)
        Bi = [
            gen.uniform(-0.5, 0.5, size=(self.state_size, self.state_size))
            for i in range(self.input_size)
        ]
        self.u = gen.uniform(-1.0, 1.0, size=(self.n_timesteps, self.input_size))
        self.t = np.linspace(0, self.n_timesteps - 1, self.n_timesteps)
        self.names = ["x" + str(i + 1) for i in range(self.state_size)]
        u_interp = interp1d(self.t, self.u, axis=0)

        def affine_sys_func(t, x):
            u = u_interp(t)
            B = np.zeros((self.state_size, self.state_size))
            for i in range(self.input_size):
                B += Bi[i] * u[i]
            x = self.A @ x + B @ x
            return x

        ivp_solution = solve_ivp(
            affine_sys_func, t_span=(self.t[0], self.t[-1]), y0=self.x0, t_eval=self.t
        )
        self.expected = ivp_solution.y.T
        self.Bi = np.stack(Bi, 2)

    def test_affine_system(self):
        actual = (
            ControlledAffineDynamicalSystem()
            .setup_matrix_system(self.A, self.Bi)
            .evolve_system(self.x0, self.u)
        )

        nptest.assert_allclose(actual.to_numpy(), self.expected, atol=0.1, rtol=0.1)

    def test_affine_vs_linear(self):

        controlled = (
            ControlledAffineDynamicalSystem()
            .setup_matrix_system(self.A, self.Bi)
            .evolve_system(self.x0, np.zeros(self.u.shape))
        )
        linear = (
            LinearDynamicalSystem(sys_type="differential", sys_mode="matrix")
            .setup_matrix_system(self.A)
            .evolve_system(self.x0, self.t, time_delta=self.t[1])
        )

        # use high tolerance since RK numerical integration compared to matrix exponent
        nptest.assert_allclose(
            controlled.to_numpy(), linear.to_numpy(), rtol=1e-3, atol=1e-3
        )


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
        self.assertEqual(actual.n_timesteps, 2)
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
            dmd.predict(predict_ic, time_values=_values)

        with self.assertRaises(ValueError):
            _values = time_values.copy()
            _values[-1] = np.inf
            dmd.predict(predict_ic, time_values=_values)

        with self.assertRaises(ValueError):
            _values = time_values.copy()
            _values[0] = -1
            dmd.predict(predict_ic, time_values=_values)

        with self.assertRaises(ValueError):
            _values = time_values.copy()
            _values = _values[::-1]
            dmd.predict(predict_ic, time_values=_values)

        with self.assertRaises(TypeError):
            _values = time_values.copy().astype(complex)
            _values[-1] = _values[-1] + 1j
            dmd.predict(predict_ic, time_values=_values)

        with self.assertRaises(ValueError):
            _values = time_values.copy()
            _values[0] = _values[1]
            dmd.predict(predict_ic, time_values=_values)

        with self.assertRaises(ValueError):
            _values = time_values.copy()[np.newaxis, :]
            dmd.predict(predict_ic, time_values=_values)

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


class DMDControlTest(unittest.TestCase):
    def _create_harmonic_tsc(self, n_samples, dim):
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

    def _create_control_tsc(self, state_size, input_size, n_timesteps) -> TSCDataFrame:
        gen = np.random.default_rng(42)

        A = gen.uniform(-1.0, 1.0, size=(state_size, state_size))
        x0 = gen.uniform(size=state_size)
        B = gen.uniform(-1.0, 1.0, size=(state_size, input_size))
        u = gen.uniform(size=(n_timesteps, input_size))
        names = ["x" + str(i + 1) for i in range(state_size)]

        tsc_df = (
            ControlledLinearDynamicalSystem()
            .setup_matrix_system(A, B)
            .evolve_system(x0, u, feature_names_out=names)
        )

        for i in range(input_size):
            tsc_df["u" + str(i + 1)] = u[:, i]

        return tsc_df

    def test_dmd_control(self):
        state_size = 4
        input_size = 2
        n_timesteps = 50
        n_predict = 5

        tsc_df = self._create_control_tsc(state_size, input_size, n_timesteps)

        state_cols = [f"x{i+1}" for i in range(state_size)]
        input_cols = [f"u{i+1}" for i in range(input_size)]

        u = tsc_df[input_cols].iloc[-n_predict:]
        t = tsc_df.index.get_level_values(1)[-n_predict:]
        expected = tsc_df[state_cols].iloc[-n_predict:]

        dmd = DMDControl().fit(expected, u)

        actual = dmd.predict(expected.initial_states(), U=u, time_values=t)

        pdtest.assert_frame_equal(actual, expected, rtol=1e-8, atol=1e-8)

    def test_dmd_control_free(self):
        tsc_df = self._create_harmonic_tsc(100, 2)
        tsc_df = TSCTakensEmbedding(delays=1).fit_transform(tsc_df)
        tsc_ic = tsc_df.initial_states()

        U = TSCDataFrame.from_same_indices_as(
            tsc_df, np.zeros([tsc_df.shape[0], 1]), except_columns=["u"]
        )

        dmd1 = DMDControl().fit(tsc_df, U=U)
        dmd2 = DMDFull(sys_mode="matrix", approx_generator=False).fit(tsc_df)

        U_pred = TSCDataFrame.from_array(
            np.zeros([10, 1]), time_values=np.arange(10), feature_names=["u"]
        )

        actual = dmd1.predict(tsc_ic, U=U_pred)  # control_input=np.zeros((10, 0)
        expected = dmd2.predict(tsc_ic, time_values=np.arange(10))

        pdtest.assert_frame_equal(actual, expected, rtol=1e-15, atol=1e-14)

    def test_dmd_control_multiple(self):
        state_size = 4
        input_size = 2
        n_timesteps = 50
        state_cols = [f"x{i+1}" for i in range(state_size)]
        input_cols = [f"u{i+1}" for i in range(input_size)]

        tsc_df_single = self._create_control_tsc(state_size, input_size, n_timesteps)
        df = pd.DataFrame(
            tsc_df_single.to_numpy(),
            index=np.arange(n_timesteps) * 0.1,
            columns=state_cols + input_cols,
        )
        tsc_df = TSCDataFrame.from_frame_list([df, df.copy(deep=True)])

        dmd = DMDControl()
        expected = tsc_df[state_cols]
        u = tsc_df[input_cols]
        actual = dmd.fit_predict(expected, U=u)

        pdtest.assert_frame_equal(actual, expected, rtol=1e-8, atol=1e-8)

    def test_dmd_control_reconstruct(self):
        state_size = 4
        input_size = 2
        n_timesteps = 50

        original = self._create_control_tsc(state_size, input_size, n_timesteps)

        state_cols = [f"x{i+1}" for i in range(state_size)]
        input_cols = [f"u{i+1}" for i in range(input_size)]

        reconstructed = DMDControl().fit_predict(
            X=original[state_cols],
            U=original[input_cols],
        )

        pdtest.assert_frame_equal(
            original[state_cols], reconstructed, check_exact=False
        )


class gDMDAffineTest(unittest.TestCase):
    def _generate_system(self, state_size, input_size):
        gen = np.random.default_rng(42)

        A = gen.uniform(-0.5, 0.5, size=(state_size, state_size))
        np.fill_diagonal(A, gen.uniform(-1.0, -0.5, size=state_size))
        Bi = np.stack(
            [
                gen.uniform(-0.5, 0.5, size=(state_size, state_size))
                for i in range(input_size)
            ],
            2,
        )

        return A, Bi

    def _create_control_tsc(
        self, state_size, n_timesteps, n_ic, input_size=2, random=False
    ):
        gen = np.random.default_rng(42)
        dt = 0.1

        A, Bi = self._generate_system(state_size, input_size)

        x0 = gen.uniform(-1.0, 1.0, size=(state_size, n_ic))
        x0 = np.hstack([x0, x0, x0, x0])
        if random:
            u = gen.uniform(-1.0, 1.0, size=(n_ic, n_timesteps, input_size))
        else:
            if input_size == 1:
                u = np.concatenate(
                    [
                        np.ones((n_ic, n_timesteps, input_size)),
                        -np.ones((n_ic, n_timesteps, input_size)),
                    ],
                    0,
                )
            elif input_size == 2:
                u = np.concatenate(
                    [
                        np.ones((n_ic, n_timesteps, input_size)),
                        -np.ones((n_ic, n_timesteps, input_size)),
                        np.stack(
                            [
                                np.ones((n_ic, n_timesteps)),
                                -np.ones((n_ic, n_timesteps)),
                            ],
                            2,
                        ),
                        np.stack(
                            [
                                -np.ones((n_ic, n_timesteps)),
                                np.ones((n_ic, n_timesteps)),
                            ],
                            2,
                        ),
                    ],
                    0,
                )
            else:
                raise NotImplementedError("Only available for input size 1 and 2")

        t = np.linspace(0, n_timesteps - 1, n_timesteps) * dt
        names = ["x" + str(i + 1) for i in range(state_size)]

        sys = ControlledAffineDynamicalSystem().setup_matrix_system(A, Bi)
        tsc_df = sys.evolve_system(
            x0, u, time_values=t, time_delta=0.1, feature_names_out=names
        )

        ureshaped = u.reshape((-1, input_size))
        for i in range(input_size):
            tsc_df["u" + str(i + 1)] = ureshaped[:, i]

        return tsc_df, sys

    def test_dmda_control_insample(self):
        state_size = 4
        input_size = 2
        n_timesteps = 50
        n_ic = 5

        tsc_df, sys = self._create_control_tsc(
            state_size, n_timesteps, n_ic, input_size
        )

        state_cols = [f"x{i+1}" for i in range(state_size)]
        input_cols = [f"u{i+1}" for i in range(input_size)]
        dmd = gDMDAffine()

        expected = tsc_df[state_cols]
        actual = dmd.fit_predict(tsc_df[state_cols], U=tsc_df[input_cols])

        pdtest.assert_frame_equal(actual, expected, rtol=5e-3, atol=0.01)

    def test_dmda_control_highorder(self):
        state_size = 4
        input_size = 2
        n_timesteps = 50
        n_ic = 5

        tsc_df, sys = self._create_control_tsc(
            state_size, n_timesteps, n_ic, input_size
        )

        state_cols = [f"x{i+1}" for i in range(state_size)]
        input_cols = [f"u{i+1}" for i in range(input_size)]
        dmd = gDMDAffine(diff_accuracy=6)

        expected = tsc_df[state_cols]
        actual = dmd.fit_predict(tsc_df[state_cols], U=tsc_df[input_cols])

        pdtest.assert_frame_equal(actual, expected, rtol=5e-3, atol=0.01)

    def test_dmda_control_outsample(self):
        state_size = 4
        input_size = 2
        n_timesteps = 50
        n_ic = 5

        tsc_df, sys = self._create_control_tsc(
            state_size, n_timesteps, n_ic, input_size
        )

        state_cols = [f"x{i+1}" for i in range(state_size)]
        input_cols = [f"u{i+1}" for i in range(input_size)]
        dmd = gDMDAffine().fit(tsc_df[state_cols], U=tsc_df[input_cols])

        t = tsc_df.index.get_level_values(1)
        t = np.linspace(t.min(), t.max() * 1.1, 2 * n_timesteps)
        U = TSCDataFrame.from_array(
            np.vstack([np.sin(0.2 * np.pi * t), np.cos(0.3 * np.pi * t)]).T,
            time_values=t,
            feature_names=["u1", "u2"],
        )
        x0 = np.random.default_rng(42).uniform(-1.0, 1.0, size=state_size)
        expected = sys.evolve_system(
            x0, U.to_numpy(), time_values=t, feature_names_out=state_cols
        )
        actual = dmd.predict(expected.initial_states()[state_cols], U=U)

        pdtest.assert_frame_equal(actual, expected, rtol=0, atol=1e-3)

    def test_dmda_control_random(self):
        state_size = 4
        input_size = 2
        n_timesteps = 50
        n_ic = 20

        tsc_df, sys = self._create_control_tsc(
            state_size, n_timesteps, n_ic, input_size
        )

        state_cols = [f"x{i+1}" for i in range(state_size)]
        input_cols = [f"u{i+1}" for i in range(input_size)]
        dmd = gDMDAffine()

        expected = tsc_df[state_cols]
        actual = dmd.fit_predict(tsc_df[state_cols], U=tsc_df[input_cols])

        pdtest.assert_frame_equal(actual, expected, rtol=0.01, atol=0.05)
