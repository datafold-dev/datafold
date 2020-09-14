#!/usr/bin/env python3

import unittest
import unittest.mock as mock

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nptest
import pandas as pd
import scipy.linalg

from datafold.dynfold.dmd import DMDEco, DMDFull, LinearDynamicalSystem, PyDMDWrapper
from datafold.pcfold import TSCDataFrame
from datafold.utils.general import (
    assert_equal_eigenvectors,
    is_df_same_index,
    sort_eigenpairs,
)


class LinearDynamicalSystemTest(unittest.TestCase):
    def _set_attrs_discrete_system(self):

        # A too small time_delta increases error (its only a first order finite diff
        # scheme). Smallest time_delta by testing 1e-8
        self.time_delta_approx = 1e-8

        # Is \dot{x} = A x (i.e., A = generator_matrix)
        self.generator_matrix = np.array([[1, -3], [-2, 1]])

        # This is basically, that the Koopman matrix is
        # A = (K - I) / time_delta, now we can simply use a small time_delta, because
        # we have the true generator matrix A given
        self.dyn_matrix_discrete = (
            np.eye(self.generator_matrix.shape[0])
            + self.time_delta_approx * self.generator_matrix
        )

        self.eigvals, self.eigvec_right = np.linalg.eig(self.dyn_matrix_discrete)
        self.eigvec_left = np.linalg.solve(
            self.eigvec_right * self.eigvals, self.dyn_matrix_discrete
        )

        # Check that diagonalization of the discrete matrix is correct
        nptest.assert_allclose(
            self.eigvec_right @ np.diag(self.eigvals) @ self.eigvec_left,
            self.dyn_matrix_discrete,
            atol=1e-15,
            rtol=0,
        )

    def setUp(self) -> None:
        self._set_attrs_discrete_system()

    def test_approx_continuous_linear_system(self, plot=False):

        n_timesteps = 200
        time_values = np.linspace(0, 1, n_timesteps)

        ic = np.array([[3], [2]])

        expected = np.zeros([n_timesteps, self.generator_matrix.shape[0]])
        for i, t in enumerate(time_values):
            expected[i, :] = np.real(
                (scipy.linalg.expm(t * self.generator_matrix) @ ic).ravel()
            )

        actual = LinearDynamicalSystem(
            system_type="discrete", time_invariant=True
        ).evolve_system_spectrum(
            dynmatrix=self.eigvec_right,
            eigenvalues=self.eigvals,
            time_delta=self.time_delta_approx,
            initial_conditions=self.eigvec_left @ ic,
            time_values=time_values,
        )

        nptest.assert_allclose(expected, actual.to_numpy(), atol=1e-6, rtol=1e-14)

        if plot:
            f, ax = plt.subplots(2, 1)

            expected = TSCDataFrame.from_same_indices_as(actual, expected)

            expected.plot(ax=ax[0])
            actual.plot(ax=ax[1])

            plt.show()

    def test_time_values(self):

        time_values = np.random.default_rng(1).uniform(size=(100)) * 100

        actual = LinearDynamicalSystem(
            system_type="discrete", time_invariant=True
        ).evolve_system_spectrum(
            dynmatrix=self.eigvec_right,
            eigenvalues=self.eigvals,
            # to match up the discrete system we have to assume a time delta of 1
            time_delta=1,
            initial_conditions=self.eigvec_left @ np.ones(shape=[2, 1]),
            time_values=time_values,
        )

        nptest.assert_array_equal(actual.time_values(), np.sort(time_values))

    def test_multi_initial_conditions(self):

        n_timeseries = 10
        initial_conditions = np.random.default_rng(1).uniform(size=(2, n_timeseries))

        time_values = np.linspace(0, 20, 100)

        actual = LinearDynamicalSystem(
            system_type="continuous", time_invariant=True
        ).evolve_system_spectrum(
            dynmatrix=self.eigvec_right,
            eigenvalues=self.eigvals,
            # to match up the discrete system we have to assume a time delta of 1
            time_delta=1,
            initial_conditions=self.eigvec_left @ initial_conditions,
            time_values=time_values,
            feature_columns=["A", "B"],
        )

        self.assertEqual(actual.n_timesteps, len(time_values))
        self.assertEqual(actual.n_timeseries, n_timeseries)
        self.assertEqual(actual.columns.tolist(), ["A", "B"])
        nptest.assert_array_equal(actual.time_values(), time_values)

    def test_feature_columns(self):

        actual = LinearDynamicalSystem(
            system_type="continuous", time_invariant=True
        ).evolve_system_spectrum(
            dynmatrix=self.eigvec_right,
            eigenvalues=self.eigvals,
            # to match up the discrete system we have to assume a time delta of 1
            time_delta=1,
            initial_conditions=self.eigvec_left @ np.ones(shape=[2, 1]),
            time_values=np.arange(4),
            feature_columns=["expectedA", "expectedB"],
        )

        self.assertEqual(actual.columns.tolist(), ["expectedA", "expectedB"])

        with self.assertRaises(ValueError):
            LinearDynamicalSystem().evolve_system_spectrum(
                dynmatrix=self.eigvec_right,
                eigenvalues=self.eigvals,
                # to match up the discrete system we have to assume a time delta of 1
                time_delta=1,
                initial_conditions=self.eigvec_left @ np.ones(shape=[2, 1]),
                time_values=np.arange(4),
                feature_columns=[1, 2, 3],
            )

    def test_return_types(self):
        actual = LinearDynamicalSystem(
            system_type="continuous", time_invariant=True
        ).evolve_system_spectrum(
            dynmatrix=self.eigvec_right,
            eigenvalues=self.eigvals,
            # to match up the discrete system we have to assume a time delta of 1
            time_delta=1,
            initial_conditions=self.eigvec_left @ np.ones(shape=[2, 2]),
            time_values=np.arange(1),
        )

        # Is a TSCDataFrame, also for single time steps
        self.assertIsInstance(actual, TSCDataFrame)
        self.assertTrue(actual.has_degenerate())

        actual = LinearDynamicalSystem(
            system_type="continuous", time_invariant=True
        ).evolve_system_spectrum(
            dynmatrix=self.eigvec_right,
            eigenvalues=self.eigvals,
            # to match up the discrete system we have to assume a time delta of 1
            time_delta=1,
            initial_conditions=self.eigvec_left @ np.ones(shape=[2, 2]),
            time_values=np.arange(2),
        )

        # is a TSCDataFrame, because more than two time values are a time series
        self.assertIsInstance(actual, TSCDataFrame)


class DMDTest(unittest.TestCase):
    def _create_random_tsc(self, dim, n_samples):
        data = np.random.default_rng(1).normal(size=(n_samples, dim))
        data = pd.DataFrame(data)
        return TSCDataFrame.from_single_timeseries(data)

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

    def test_dmd_eigenpairs(self):
        # From http://www.astronomia.edu.uy/progs/algebra/Linear_Algebra,_4th_Edition__(2009)Lipschutz-Lipson.pdf
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

        # NOTE: the left eigenvectors are transposed because the they are
        # stored row-wise (whereas right eigenvectors are column-wise). The helper
        # function assert_equal_eigenvectors assumes column-wise ordering
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

    def test_dmd_pydmd1(self):

        test_data = self._create_random_tsc(n_samples=500, dim=30)

        pydmd = PyDMDWrapper(
            method="dmd", svd_rank=1000, tlsq_rank=0, exact=True, opt=False
        ).fit(test_data)

        # datafold and PyDMD have a different way to order the eigenvalues. For
        # the test we sort both accoring to the complex eigenvalue
        expected_eigenvalues, expected_modes = sort_eigenpairs(
            pydmd.eigenvalues_, pydmd.dmd_modes
        )

        dmd = DMDFull().fit(test_data)
        actual = dmd.dmd_modes

        nptest.assert_allclose(
            dmd.eigenvalues_, expected_eigenvalues, atol=1e-4, rtol=0,
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
        self.assertEqual(actual.n_timesteps, dmd.time_values_in_.quantity)
        nptest.assert_array_equal(actual.ids, np.arange(predict_ic.shape[0]))

        # provide own time series IDs in the initial condition and own time values
        expected_ids = np.arange(0, predict_ic.shape[0] * 2, 2)
        expected_time_values = np.arange(500)

        predict_ic = TSCDataFrame(
            predict_ic,
            index=pd.MultiIndex.from_arrays(
                [expected_ids, np.zeros(predict_ic.shape[0]),]
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

        time_values = np.arange(500, dtype=np.float)

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
            _values = time_values.copy().astype(np.complex)
            _values[-1] = _values[-1] + 1j
            dmd.predict(predict_ic, _values)

        with self.assertRaises(ValueError):
            _values = time_values.copy()
            _values[0] = _values[1]
            dmd.predict(predict_ic, _values)

        with self.assertRaises(ValueError):
            _values = time_values.copy()[np.newaxis, :]
            dmd.predict(predict_ic, _values)
