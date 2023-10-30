#!/usr/bin/env python3

import unittest
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest
import pytest
import scipy.linalg
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from datafold.dynfold import TSCTakensEmbedding
from datafold.dynfold.dmd import (
    DMDControl,
    DMDStandard,
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
    is_symmetric_matrix,
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
                sys_type="flowmap", sys_mode="spectral", is_time_invariant=True
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
                sys_type="flowmap", sys_mode="spectral", is_time_invariant=True
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
                sys_type="flowmap", sys_mode="spectral", is_time_invariant=True
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
                sys_type="flowmap", sys_mode="spectral", is_time_invariant=True
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
                sys_type="flowmap", sys_mode="spectral", is_time_invariant=True
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
                sys_type="flowmap", sys_mode="spectral", is_time_invariant=True
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
        self.u = gen.uniform(size=(self.n_timesteps - 1, self.input_size))
        self.t = np.linspace(0, self.n_timesteps - 1, self.n_timesteps)
        self.names = ["x" + str(i + 1) for i in range(self.state_size)]

        self.expected = np.zeros((self.n_timesteps, self.state_size))
        self.expected[0, :] = self.x0

        for idx, _time in enumerate(self.t[1:]):
            self.expected[idx + 1, :] = (
                self.A @ self.expected[idx, :] + self.B @ self.u[idx, :]
            )

        # do not write state past last since t starts at 0

    def test_controlled_system(self):
        actual = (
            LinearDynamicalSystem("flowmap", "matrix", is_controlled=True)
            .setup_matrix_system(self.A, control_matrix=self.B)
            .evolve_system(
                self.x0,
                time_values=np.arange(self.u.shape[0] + 1),
                control_input=self.u,
                time_delta=1,
            )
        )

        nptest.assert_allclose(actual.to_numpy(), self.expected, atol=1e-8, rtol=1e-13)

    def test_controlled_vs_linear(self):
        controlled = (
            LinearDynamicalSystem(
                sys_type="flowmap", sys_mode="matrix", is_controlled=True
            )
            .setup_matrix_system(self.A, control_matrix=self.B)
            .evolve_system(
                self.x0,
                time_values=self.t,
                control_input=np.zeros(self.u.shape),
                time_delta=self.t[1] - self.t[0],
            )
        )
        linear = (
            LinearDynamicalSystem(sys_type="flowmap", sys_mode="matrix")
            .setup_matrix_system(self.A)
            .evolve_system(
                self.x0, time_values=self.t, time_delta=self.t[1] - self.t[0]
            )
        )

        nptest.assert_allclose(
            controlled.to_numpy(), linear.to_numpy(), atol=1e-8, rtol=1e-13
        )

    def test_time_delta(self):
        t = np.linspace(0, (self.n_timesteps - 1) / 2, self.n_timesteps)
        dt = 0.5

        actual = (
            LinearDynamicalSystem(
                sys_type="flowmap", sys_mode="matrix", is_controlled=True
            )
            .setup_matrix_system(self.A, control_matrix=self.B)
            .evolve_system(self.x0, control_input=self.u, time_values=t, time_delta=dt)
        )

        nptest.assert_allclose(actual.to_numpy(), self.expected, atol=1e-8, rtol=1e-13)

    def test_multi_initial_conditions(self):
        x0 = np.array([self.x0, self.x0]).T
        expected = np.vstack([self.expected, self.expected])
        ts_ids = np.array((10, 20))

        u = np.stack([self.u, self.u])

        system = LinearDynamicalSystem(
            sys_type="flowmap", sys_mode="matrix", is_controlled=True
        ).setup_matrix_system(system_matrix=self.A, control_matrix=self.B)
        actual = system.evolve_system(
            x0,
            control_input=u,
            time_values=self.t,
            time_delta=1,
            time_series_ids=ts_ids,
            feature_names_out=self.names,
        )

        nptest.assert_allclose(actual.to_numpy(), expected, atol=1e-15, rtol=1e-15)
        self.assertEqual(actual.n_timesteps, len(self.t))
        self.assertEqual(actual.n_timeseries, 2)
        self.assertEqual(actual.columns.tolist(), self.names)
        nptest.assert_array_equal(actual.time_values(), self.t)

    def test_feature_columns(self):
        actual = (
            LinearDynamicalSystem(
                sys_type="flowmap", sys_mode="matrix", is_controlled=True
            )
            .setup_matrix_system(self.A, control_matrix=self.B)
            .evolve_system(
                self.x0,
                control_input=self.u,
                time_values=np.arange(self.u.shape[0] + 1),
                feature_names_out=self.names,
                time_delta=1,
            )
        )

        self.assertIsInstance(actual, TSCDataFrame)
        self.assertEqual(actual.columns.tolist(), self.names)
        nptest.assert_array_equal(actual.time_values(), np.arange(self.u.shape[0] + 1))

        # invalid feature_names_out
        with self.assertRaises(ValueError):
            (
                LinearDynamicalSystem(
                    sys_type="flowmap", sys_mode="matrix", is_controlled=True
                )
                .setup_matrix_system(self.A, control_matrix=self.B)
                .evolve_system(
                    self.x0,
                    control_input=self.u,
                    time_values=np.arange(self.u.shape[0]),
                    time_delta=1,
                    feature_names_out=[1, 2, 3],
                )
            )

    def test_return_types(self):
        actual = (
            LinearDynamicalSystem(
                sys_type="flowmap", sys_mode="matrix", is_controlled=True
            )
            .setup_matrix_system(self.A, control_matrix=self.B)
            .evolve_system(
                self.x0,
                control_input=self.u[:1],
                time_values=np.array([0, 1]),
                time_delta=1,
            )
        )

        # Is a TSCDataFrame, also for single time steps
        self.assertIsInstance(actual, TSCDataFrame)
        self.assertEqual(actual.n_timesteps, 2)
        self.assertEqual(actual.n_timeseries, 1)

        actual = (
            LinearDynamicalSystem(
                sys_type="flowmap", sys_mode="matrix", is_controlled=True
            )
            .setup_matrix_system(self.A, control_matrix=self.B)
            .evolve_system(
                self.x0,
                control_input=self.u[:2, :],
                time_values=[0, 1, 2],
                time_delta=1,
            )
        )

        # is a TSCDataFrame, because more than two time values are a time series
        self.assertIsInstance(actual, TSCDataFrame)

    def test_dimension_mismatch(self):
        setting = dict(sys_type="flowmap", sys_mode="matrix", is_controlled=True)

        # non-square system matrix
        with self.assertRaises(ValueError):
            LinearDynamicalSystem(**setting).setup_matrix_system(
                np.zeros([3, 4]), control_matrix=np.zeros([4, 4])
            )

        # mismatch between system and input matrix
        with self.assertRaises(ValueError):
            LinearDynamicalSystem(**setting).setup_matrix_system(
                np.zeros([4, 4]), control_matrix=np.zeros([3, 4])
            )

        # wrong initial condition
        with self.assertRaises(ValueError):
            LinearDynamicalSystem(**setting).setup_matrix_system(
                np.zeros((4, 4)), control_matrix=np.zeros([4, 3])
            ).evolve_system(
                np.zeros(3), control_input=np.zeros([4, 3]), time_values=np.arange(4)
            )

        # wrong control input
        with self.assertRaises(ValueError):
            LinearDynamicalSystem(**setting).setup_matrix_system(
                np.zeros((4, 4)), control_matrix=np.zeros((4, 3))
            ).evolve_system(
                np.zeros(4), control_input=np.zeros([3, 4]), time_values=np.arange(3)
            )

        # wrong time values -> only zeros
        with self.assertRaises(ValueError):
            LinearDynamicalSystem(**setting).setup_matrix_system(
                np.zeros((4, 4)), control_matrix=np.zeros([4, 3])
            ).evolve_system(
                np.zeros(3), control_input=np.zeros([4, 3]), time_values=np.zeros(3)
            )

        # wrong time values -> control input vs. time values mismatch
        with self.assertRaises(ValueError):
            LinearDynamicalSystem(**setting).setup_matrix_system(
                np.zeros((4, 4)), control_matrix=np.zeros([4, 3])
            ).evolve_system(
                np.zeros(3), control_input=np.zeros([4, 3]), time_values=np.arange(999)
            )


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
            for _ in range(self.input_size)
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
        system = LinearDynamicalSystem(
            sys_type="differential",
            sys_mode="matrix",
            is_controlled=True,
            is_control_affine=True,
        ).setup_matrix_system(self.A, control_matrix=self.Bi)

        system._requires_last_control_state = True

        actual = system.evolve_system(
            self.x0,
            control_input=self.u,
            time_values=np.arange(self.u.shape[0]),
            time_delta=1,
        )

        nptest.assert_allclose(actual.to_numpy(), self.expected, atol=0.2, rtol=0.1)

    def test_affine_vs_linear(self):
        controlled_system = LinearDynamicalSystem(
            sys_type="differential",
            sys_mode="matrix",
            is_controlled=True,
            is_control_affine=True,
        ).setup_matrix_system(self.A, control_matrix=self.Bi)

        controlled_system._requires_last_control_state = True

        controlled = controlled_system.evolve_system(
            self.x0,
            control_input=np.zeros(self.u.shape),
            time_values=self.t,
            time_delta=self.t[1] - self.t[0],
        )

        linear = (
            LinearDynamicalSystem(sys_type="differential", sys_mode="matrix")
            .setup_matrix_system(self.A)
            .evolve_system(
                self.x0, time_values=self.t, time_delta=self.t[1] - self.t[0]
            )
        )

        # use high tolerance since RK numerical integration compared to matrix exponent
        nptest.assert_allclose(
            controlled.to_numpy(), linear.to_numpy(), rtol=0, atol=1e-3
        )


class DMDTest(unittest.TestCase):
    @classmethod
    def _create_random_tsc(cls, dim, n_samples):
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

    def test_clone(self):
        from sklearn.base import clone

        data = self._create_harmonic_tsc(n_samples=50, dim=5)
        dmd = DMDStandard().fit(data)
        dmd2 = clone(dmd)

        assert id(dmd) != id(dmd2)

    def test_dmd_eigenpairs(self):
        # From
        # http://www.astronomia.edu.uy/progs/algebra/Linear_Algebra,_4th_Edition__(2009)Lipschutz-Lipson.pdf # noqa
        # page 297 Example 9.5

        dmd = DMDStandard(diagonalize=True)

        mock_system_matrix = np.array([[3.0, 1], [2, 2]])
        dmd._compute_full_system_matrix = mock.MagicMock(
            return_value=[mock_system_matrix, None, None, None, None]
        )

        dmd = dmd.fit(self._create_random_tsc(n_samples=2, dim=2))

        expected_eigenvalues = np.array([4.0, 1.0])  # must be sorted descending
        nptest.assert_array_equal(expected_eigenvalues, dmd.eigenvalues_)

        expected_eigenvectors_right = np.array([[1, 1], [1, -2]])

        assert_equal_eigenvectors(expected_eigenvectors_right, dmd.eigenvectors_right_)

        expected_eigenvectors_left = np.array([[2 / 3, 1 / 3], [1 / 3, -1 / 3]])

        # NOTE: the left eigenvectors are transposed because they are
        # stored row-wise (whereas right eigenvectors are column-wise). The helper
        # function assert_equal_eigenvectors assumes column-wise orientation
        assert_equal_eigenvectors(
            expected_eigenvectors_left.T, dmd.eigenvectors_left_.T
        )

        # sanity check: diagonalization of mocked Koopman matrix
        nptest.assert_equal(
            expected_eigenvectors_right
            @ np.diag(expected_eigenvalues)
            @ expected_eigenvectors_left,
            mock_system_matrix,
        )

        actual = (
            dmd.eigenvectors_right_ @ np.diag(dmd.eigenvalues_) @ dmd.eigenvectors_left_
        )

        nptest.assert_allclose(mock_system_matrix, actual, rtol=1e-15, atol=1e-15)

    def test_dmd_equivalence_generator_flowmap(self):
        test_data = self._create_random_tsc(n_samples=500, dim=30)

        generator_system = DMDStandard(diagonalize=True, approx_generator=True).fit(
            test_data
        )
        flowmap_system = DMDStandard(diagonalize=True, approx_generator=False).fit(
            test_data
        )

        self.assertEqual(generator_system.sys_type, "differential")
        self.assertEqual(flowmap_system.sys_type, "flowmap")

        time_values = np.linspace(0, 5, 20)

        generator_result = generator_system.predict(
            test_data.initial_states(), time_values=time_values
        )

        flowmap_result = flowmap_system.predict(
            test_data.initial_states(), time_values=time_values
        )

        pdtest.assert_frame_equal(generator_result, flowmap_result, rtol=0, atol=1e-15)

        # check the relationship of eigenvalues of the flowmap and differential system
        # respectively
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
        first = DMDStandard(
            sys_mode="spectral", diagonalize=False, approx_generator=False
        ).fit_predict(tsc_df)

        second = DMDStandard(sys_mode="matrix", approx_generator=False).fit_predict(
            tsc_df
        )

        pdtest.assert_frame_equal(first, second, rtol=1e-16, atol=1e-11)

        # for generator case
        first = DMDStandard(
            sys_mode="spectral", diagonalize=False, approx_generator=True
        ).fit_predict(tsc_df)
        second = DMDStandard(sys_mode="matrix", approx_generator=True).fit_predict(
            tsc_df
        )

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

        first = DMDStandard(diagonalize=True, approx_generator=True).fit(
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

        dmd = DMDStandard().fit(test_data)
        actual = dmd.dmd_modes

        nptest.assert_allclose(
            dmd.eigenvalues_,
            expected_eigenvalues,
            atol=1e-4,
            rtol=0,
        )

        assert_equal_eigenvectors(expected_modes, actual, tol=1.12e-15)

    def test_dmd_pydmd2(self):
        test_data = self._create_random_tsc(n_samples=500, dim=100)

        pydmd = PyDMDWrapper(
            method="dmd", svd_rank=10, tlsq_rank=0, exact=True, opt=False
        ).fit(test_data)

        expected_eigvals, expected_modes = sort_eigenpairs(
            pydmd.eigenvalues_, pydmd.dmd_modes
        )

        # dmd = DMDEco(svd_rank=10).fit(test_data)
        dmd = DMDStandard(rank=10).fit(test_data)
        actual_eigvals, actual_modes = dmd.eigenvalues_, dmd.dmd_modes

        nptest.assert_allclose(expected_eigvals, actual_eigvals, atol=1e-15, rtol=0)
        assert_equal_eigenvectors(expected_modes, actual_modes, tol=1e-15)

    def test_reconstruct_indices(self):
        expected_indices = self._create_random_tsc(n_samples=100, dim=10)
        actual_indices = DMDStandard(diagonalize=False).fit_predict(expected_indices)

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

        dmd = DMDStandard(diagonalize=False).fit(tsc_df_fit)

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

        dmd = DMDStandard(diagonalize=False).fit(tsc_df_fit)

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

        dmd = DMDStandard(diagonalize=False).fit(tsc_df_fit)

        predict_ic.columns = pd.Index(np.arange(2, 12))

        with self.assertRaises(ValueError):
            dmd.predict(predict_ic)

    def test_sample_weights(self):
        harmonic = self._create_harmonic_tsc(100, 2)
        random = self._create_random_tsc(2, 100)
        combined = TSCDataFrame.from_frame_list([harmonic, random])
        weights = np.hstack([np.ones(99), np.zeros(99)])

        expected = DMDStandard().fit(harmonic)
        actual = DMDStandard().fit(combined, sample_weights=weights)

        nptest.assert_array_equal(expected.eigenvalues_, actual.eigenvalues_)

    def test_resdmd(self):
        dim = 7
        test_data = self._create_harmonic_tsc(n_samples=128, dim=dim)

        dmd = DMDStandard().fit(test_data)
        expected_modes = dmd.dmd_modes
        expected_eigenvalues = dmd.eigenvalues_

        with self.assertRaises(ValueError):
            DMDStandard(residual_filter=0.0).fit(test_data)

        resdmd_high = DMDStandard(residual_filter=1e3).fit(test_data)
        actual_modes = resdmd_high.dmd_modes

        resdmd_med = DMDStandard(residual_filter=1e-1).fit(test_data)

        nptest.assert_allclose(
            np.abs(resdmd_high.eigenvalues_),
            np.abs(expected_eigenvalues),
            atol=1e-4,
            rtol=0,
        )

        # check eigendecomposition
        nptest.assert_almost_equal(
            np.linalg.norm(
                actual_modes
                @ np.diag(resdmd_high.eigenvalues_)
                @ np.linalg.inv(actual_modes),
                "fro",
            ),
            np.linalg.norm(
                expected_modes
                @ np.diag(expected_eigenvalues)
                @ np.linalg.inv(expected_modes),
                "fro",
            ),
        )

        assert len(resdmd_med.eigenvalues_) < dim
        assert resdmd_med.dmd_modes.shape[0] == dim

    @pytest.mark.skip(reason="Computing the pseudospectrum needs more work.")
    def test_pseudospectrum(self, plot=True):
        dim = 10
        eps = 1
        test_data = self._create_harmonic_tsc(n_samples=128, dim=dim)
        test_data = TSCTakensEmbedding(delays=1).fit_transform(test_data)

        dmd = DMDStandard().fit(test_data)
        resdmd = DMDStandard(residual_filter=eps, compute_pseudospectrum=True).fit(
            test_data
        )

        self.assertTrue(is_symmetric_matrix(resdmd._G))
        self.assertTrue(is_symmetric_matrix(resdmd._R))

        ax = plot_eigenvalues(dmd.eigenvalues_, plot_unit_circle=True)
        ax.set_title("DMD eigenvalues")
        ax = plot_eigenvalues(resdmd.eigenvalues_, plot_unit_circle=True)
        ax.set_title("ResDMD eigenvalues")

        reconstruct1 = dmd.reconstruct(test_data)
        reconstruct2 = resdmd.reconstruct(test_data)

        dmd.score(test_data)  # TODO make test that both reconstruct well!
        resdmd.score(test_data)

        for ev in resdmd.eigenvalues_:
            self.assertIn(ev, dmd.eigenvalues_)

        residuals, eigfunc = resdmd.pseudospectrum(
            dmd.eigenvalues_, return_eigfuncs=True
        )

        assert len(resdmd.eigenvalues_) == len(residuals[residuals <= eps])

        # TODO: this currently fails -- check if there is a bug and if it has to be true

        ax = test_data.plot(c="red")
        reconstruct1.plot(ax=ax, c="black", linestyle="--")
        ax.set_title("DMD")

        ax = test_data.plot(c="red")
        reconstruct2.plot(ax=ax, c="black", linestyle="--")
        ax.set_title("ResDMD")

        if plot:
            plt.show()


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
            (OnlineDMD(weighting=1.0), dict(batch_fit=True)),
        ]

        batches = np.array_split(df, df.shape[0] // batchsize)
        predict = []
        predict_fulldmd = []
        train = []

        for dmd, fit_params in stream_dmd_classes:
            for i in range(len(batches) - 1):
                fit_batch = batches[i]
                predict_batch = batches[i + 1]

                dmd_full = DMDStandard().fit(pd.concat(batches[: i + 1], axis=0))
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

        dmd_full = DMDStandard()
        dmd_full.fit(X)

        dmd_partial = StreamingDMD(max_rank=None)
        dmd_partial.partial_fit(X)

        # actual does not necessarily compute all possible eigenvalues
        # -> compare only leading eigenvalues from DMDStandard
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


class OnlineDMDTest(unittest.TestCase):
    def test_online_dmd(self):
        """Test taken and adapted from
        https://github.com/haozhg/odmd/blob/master/tests/test_online.py. For license from
        "odmd" see LICENSE_bundeled file in datafold repository.
        """
        for n in range(2, 10):  # n -> state dimension
            T = 100 * n  # number of measurements

            # linear system matrix
            A = np.random.default_rng(2).normal(size=(n, n))

            now_state = np.random.default_rng(3).normal(size=(n, T))
            next_state = A.dot(now_state)

            X = TSCDataFrame.from_shift_matrices(
                now_state, next_state, snapshot_orientation="col"
            )
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

        expected = DMDStandard().fit(X)
        actual = OnlineDMD(weighting=1.0).partial_fit(X, batch_fit=True)

        nptest.assert_allclose(
            expected.eigenvalues_, actual.eigenvalues_, rtol=1e-14, atol=1e-14
        )
        nptest.assert_allclose(
            expected.eigenvectors_right_,
            actual.eigenvectors_right_,
            rtol=1e-15,
            atol=1e-12,
        )

        # The weighing != 1.0 triggers a different if-case
        # (but should have essentially no effect)
        actual = OnlineDMD(weighting=1.0 - 1e-15).partial_fit(X, batch_fit=True)

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

    def test_invalid_feature_names(self):
        tsc_df_fit = DMDTest._create_random_tsc(n_samples=100, dim=10)
        predict_ic = DMDTest._create_random_tsc(n_samples=1, dim=10)

        dmd = DMDStandard(diagonalize=False).fit(tsc_df_fit)

        predict_ic.columns = pd.Index(np.arange(2, 12))

        with self.assertRaises(ValueError):
            dmd.predict(predict_ic)


class DMDControlTest(unittest.TestCase):
    def _create_control_tsc(
        self, state_size, control_size, n_timesteps, time_delta=1.0
    ):
        gen = np.random.default_rng(42)

        A = gen.uniform(-1.0, 1.0, size=(state_size, state_size))
        x0 = gen.uniform(size=state_size)
        B = gen.uniform(-1.0, 1.0, size=(state_size, control_size))
        U = gen.uniform(size=(n_timesteps - 1, control_size))

        time_values = np.arange(0, time_delta * n_timesteps, time_delta)

        X = (
            LinearDynamicalSystem(
                sys_type="flowmap", sys_mode="matrix", is_controlled=True
            )
            .setup_matrix_system(A, control_matrix=B)
            .evolve_system(
                x0,
                control_input=U,
                time_values=time_values,
                time_delta=time_delta,
                feature_names_out=[f"x {i + 1}" for i in range(state_size)],
            )
        )

        U = TSCDataFrame.from_array(
            U,
            time_values=time_values[:-1],
            feature_names=[f"u {i + 1}" for i in range(control_size)],
        )

        return X, U

    def test_dmd_control(self):
        state_size = 4
        input_size = 2
        n_timesteps = 50
        n_predict = 5

        X, U = self._create_control_tsc(state_size, input_size, n_timesteps)

        U_pred = U.tail(n_predict - 1)
        expected = X.tail(n_predict)

        t_eval = expected.time_values()

        dmd = DMDControl().fit(expected, U=U_pred)

        actual1 = dmd.predict(expected.initial_states(), U=U_pred)

        # it's discouraged to provide U and t_eval, but check that it works if it is identical
        actual2 = dmd.predict(expected.initial_states(), U=U_pred, time_values=t_eval)

        with self.assertRaises(ValueError):
            dmd.predict(expected.initial_states(), U=U_pred, time_values=t_eval[:-1])

        pdtest.assert_frame_equal(actual1, actual2, rtol=0, atol=0)
        pdtest.assert_frame_equal(actual1, expected, rtol=1e-11, atol=1e-15)

    def test_dmd_control_free(self):
        tsc_df = DMDTest._create_harmonic_tsc(n_samples=100, dim=2)
        tsc_df = TSCTakensEmbedding(delays=1).fit_transform(tsc_df)
        tsc_ic = tsc_df.initial_states()

        # mock control free by setting "U" to zero
        U = TSCDataFrame.from_same_indices_as(
            tsc_df.tsc.drop_last_n_samples(1),
            np.zeros([tsc_df.shape[0] - 1, 1]),
            except_columns=["u"],
        )

        dmd1 = DMDControl().fit(tsc_df, U=U)
        dmd2 = DMDStandard(sys_mode="matrix", approx_generator=False).fit(tsc_df)

        U_pred = TSCDataFrame.from_array(
            np.zeros([9, 1]), time_values=np.arange(1, 10), feature_names=["u"]
        )

        actual = dmd1.predict(tsc_ic, U=U_pred)  # control_input=np.zeros((10, 0)
        expected = dmd2.predict(tsc_ic, time_values=np.arange(1, 11))

        pdtest.assert_frame_equal(actual, expected, rtol=1e-14, atol=1e-13)

    def test_dmd_control_multiple(self):
        state_size = 4
        input_size = 2
        n_timesteps = 50

        X, U = self._create_control_tsc(state_size, input_size, n_timesteps, 0.1)
        X_expected = TSCDataFrame.from_frame_list([X, X])
        U = TSCDataFrame.from_frame_list([U, U])

        dmd = DMDControl()
        actual = dmd.fit_predict(X_expected, U=U)
        pdtest.assert_frame_equal(actual, X_expected, rtol=1e-8, atol=1e-8)

    def test_dmd_control_reconstruct(self):
        state_size = 4
        input_size = 2
        n_timesteps = 50

        X_tsc, U_tsc = self._create_control_tsc(state_size, input_size, n_timesteps)

        reconstructed = DMDControl().fit_predict(
            X=X_tsc,
            U=U_tsc,
        )

        pdtest.assert_frame_equal(
            X_tsc, reconstructed, check_exact=False, rtol=1e-11, atol=1e-10
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

        sys = LinearDynamicalSystem(
            sys_type="differential",
            sys_mode="matrix",
            is_controlled=True,
            is_control_affine=True,
        )
        sys.setup_matrix_system(A, control_matrix=Bi)
        sys._requires_last_control_state = True

        X = sys.evolve_system(
            x0, control_input=u, time_values=t, time_delta=0.1, feature_names_out=names
        )

        U = TSCDataFrame.from_same_indices_as(
            X,
            values=u.reshape((-1, input_size)),
            except_columns=["u" + str(i + 1) for i in range(input_size)],
        )

        return X, U, sys

    def test_dmda_control_insample(self):
        state_size = 4
        input_size = 2
        n_timesteps = 50
        n_ic = 5

        X, U, _ = self._create_control_tsc(state_size, n_timesteps, n_ic, input_size)

        dmd = gDMDAffine()

        expected = X
        actual = dmd.fit_predict(X, U=U)

        pdtest.assert_frame_equal(actual, expected, rtol=5e-3, atol=0.01)

    def test_dmda_control_highorder(self):
        state_size = 4
        input_size = 2
        n_timesteps = 50
        n_ic = 5

        X, U, _ = self._create_control_tsc(state_size, n_timesteps, n_ic, input_size)

        dmd = gDMDAffine(diff_accuracy=6)

        expected = X
        actual = dmd.fit_predict(X, U=U)

        pdtest.assert_frame_equal(actual, expected, rtol=5e-3, atol=0.01)

    def test_dmda_control_outsample(self):
        state_size = 4
        input_size = 2
        n_timesteps = 50
        n_ic = 5

        X, U, sys = self._create_control_tsc(state_size, n_timesteps, n_ic, input_size)

        dmd = gDMDAffine().fit(X=X, U=U)

        t_orig = X.index.get_level_values(1)
        t_oos = np.linspace(t_orig.min(), t_orig.max() * 1.1, 2 * n_timesteps)
        U = TSCDataFrame.from_array(
            np.column_stack([np.sin(0.2 * np.pi * t_oos), np.cos(0.3 * np.pi * t_oos)]),
            time_values=t_oos,
            feature_names=["u1", "u2"],
        )

        x0 = np.random.default_rng(42).uniform(-1.0, 1.0, size=state_size)
        expected = sys.evolve_system(
            x0,
            control_input=U.to_numpy(),
            time_values=t_oos,
            feature_names_out=X.columns,
        )

        actual = dmd.predict(expected.initial_states(), U=U)
        pdtest.assert_frame_equal(actual, expected, rtol=0, atol=1e-3)

    def test_dmda_control_random(self):
        state_size = 4
        input_size = 2
        n_timesteps = 50
        n_ic = 20

        X, U, _ = self._create_control_tsc(state_size, n_timesteps, n_ic, input_size)

        dmd = gDMDAffine()

        expected = X
        actual = dmd.fit_predict(X, U=U)

        pdtest.assert_frame_equal(actual, expected, rtol=0.01, atol=0.05)
