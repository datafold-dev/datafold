import unittest
from copy import deepcopy

import diffusion_maps as legacy_dmap
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nptest
import pandas as pd
import pytest
import scipy.sparse
from scipy.stats import norm
from sklearn.datasets import make_swiss_roll
from sklearn.metrics import mean_squared_error

from datafold.dynfold import DiffusionMaps, LocalRegressionSelection
from datafold.dynfold.dmap import DiffusionMapsVariable
from datafold.dynfold.tests.test_helper import (
    assert_equal_eigenvectors,
    circle_data,
    cmp_dmap_legacy,
    cmp_eigenpairs,
    cmp_kernel_matrix,
    make_strip,
)
from datafold.pcfold import ContinuousNNKernel, GaussianKernel, TSCDataFrame
from datafold.pcfold.distance import SklearnKNN
from datafold.pcfold.kernels import ConeKernel
from datafold.utils.general import is_symmetric_matrix, random_subsample
from datafold.utils.plot import plot_pairwise_eigenvector

try:
    import rdist  # noqa
except ImportError:
    IMPORTED_RDIST = False
else:
    IMPORTED_RDIST = True


class DiffusionMapsTest(unittest.TestCase):
    def setUp(self):
        self.xmin = 0.0
        self.ymin = 0.0
        self.width = 1.0
        self.height = 1e-1
        self.num_samples = 50000
        self.data = make_strip(
            self.xmin, self.ymin, self.width, self.height, self.num_samples
        )

    @staticmethod
    def _compute_rayleigh_quotients(matrix, eigenvectors):
        """Compute Rayleigh quotients."""
        n = eigenvectors.shape[1]
        rayleigh_quotients = np.zeros(n)
        for i in range(n):
            v = eigenvectors[:, i]
            rayleigh_quotients[i] = np.dot(v, matrix @ v) / np.dot(v, v)
        rayleigh_quotients = np.sort(np.abs(rayleigh_quotients))
        return rayleigh_quotients[::-1]

    @staticmethod
    def mock_eigensolver_call():
        """This code is executed before each test.

        The purpose is to overwrite the argument "validate_matrix" in
        "compute_kernel_eigenpairs", which enables checks that are disabled by default
        """
        from datafold.dynfold import dmap
        from datafold.pcfold import eigsolver

        def mock_compute_kernel_eigenpairs(*args, **kwargs):
            kwargs["validate_matrix"] = True  # always validate matrix
            return eigsolver.compute_kernel_eigenpairs(*args, **kwargs)

        dmap.compute_kernel_eigenpairs = mock_compute_kernel_eigenpairs

    @pytest.fixture(autouse=True)
    def run_before_each_test(self):
        """This runs before each test."""
        DiffusionMapsTest.mock_eigensolver_call()
        yield

    def test_accuracy(self):
        n_samples = 5000
        n_eigenpairs = 10
        epsilon = 5e-1

        downsampled_data, _ = random_subsample(self.data, n_samples)

        # symmetrize_kernel=False, because the rayleigh_quotient requires the
        # kernel_matrix_
        dm = DiffusionMaps(
            GaussianKernel(epsilon=epsilon),
            symmetrize_kernel=False,
            n_eigenpairs=n_eigenpairs,
        ).fit(downsampled_data, store_kernel_matrix=True)

        actual_eigvals = dm.eigenvalues_
        expected_eigvals = self._compute_rayleigh_quotients(
            dm.kernel_matrix_, dm.eigenvectors_
        )

        nptest.assert_allclose(np.abs(actual_eigvals), np.abs(expected_eigvals))

    def test_set_param(self):
        dmap = DiffusionMaps(GaussianKernel(epsilon=1))
        dmap.set_params(**dict(kernel=GaussianKernel(epsilon=2)))

        self.assertEqual(dmap.kernel.epsilon, 2)

    def test_is_valid_sklearn_estimator(self):
        from sklearn.utils.estimator_checks import check_estimator

        for estimator, check in check_estimator(
            DiffusionMaps(GaussianKernel(epsilon=1.0), n_eigenpairs=3),
            generate_only=True,
        ):
            check(estimator)

    def test_feature_names_out(self):
        X_swiss, _ = make_swiss_roll(n_samples=100, noise=0, random_state=5)
        X_swiss = TSCDataFrame.from_array(X_swiss)

        dmap = DiffusionMaps(kernel=GaussianKernel(2), n_eigenpairs=5)
        dmap.fit(X_swiss)
        expected = dmap.get_feature_names_out()
        actual = dmap.transform(X_swiss).columns

        self.assertTrue(np.all(expected == actual))

    def test_multiple_epsilon_values(self, plot=False):
        n_samples = 5000
        n_maps = 10
        n_eigenpairs = 10
        epsilon_min, epsilon_max = 1e-1, 1e1
        epsilons = np.logspace(np.log10(epsilon_min), np.log10(epsilon_max), n_maps)

        downsampled_data, _ = random_subsample(self.data, n_samples)

        eigvects = np.zeros((n_maps, downsampled_data.shape[0], n_eigenpairs))
        eigvals = np.zeros((n_maps, n_eigenpairs))

        for i, epsilon in enumerate(reversed(epsilons)):
            dm = DiffusionMaps(
                GaussianKernel(epsilon),
                n_eigenpairs=n_eigenpairs,
                symmetrize_kernel=False,
            ).fit(downsampled_data, store_kernel_matrix=True)

            eigvals[i, :] = dm.eigenvalues_
            eigvects[i, :, :] = dm.eigenvectors_

            ew = dm.eigenvalues_
            rq = self._compute_rayleigh_quotients(dm.kernel_matrix_, dm.eigenvectors_)
            nptest.assert_allclose(np.abs(ew), np.abs(rq), atol=1e-16)

            if plot:
                plt.figure()
                plt.title(f"$\\epsilon$ = {epsilon:.3f}")
                for k in range(1, 10):
                    plt.subplot(2, 5, k)
                    plt.scatter(
                        downsampled_data[:, 0],
                        downsampled_data[:, 1],
                        c=eigvects[i, :, k],
                    )
                    plt.xlim([self.xmin, self.xmin + self.width])
                    plt.ylim([self.ymin, self.ymin + self.height])
                    plt.tight_layout()
                    plt.gca().set_title(f"$\\psi_{k}$")
                plt.subplot(2, 5, 10)
                plt.step(range(eigvals[i, :].shape[0]), np.abs(eigvals[i, :]))
                plt.title(f"epsilon = {epsilon:.2f}")

        if plot:
            plt.show()

    def test_compute_all_eigenpairs(self):
        # check that all eigenpairs can be computed
        X_swiss_all, _ = make_swiss_roll(n_samples=100, noise=0, random_state=5)
        actual1 = DiffusionMaps(kernel=GaussianKernel(epsilon=2), n_eigenpairs=100).fit(
            X_swiss_all
        )

        actual2 = DiffusionMaps(
            kernel=GaussianKernel(epsilon=2), n_eigenpairs=100, symmetrize_kernel=False
        ).fit(X_swiss_all)

        self.assertEqual(actual1.eigenvectors_.shape[1], 100)
        self.assertEqual(actual1.eigenvalues_.shape[0], 100)

        self.assertEqual(actual2.eigenvectors_.shape[1], 100)
        self.assertEqual(actual2.eigenvalues_.shape[0], 100)

    def test_sanity_dense_sparse(self):
        data, _ = make_swiss_roll(1000, random_state=1)

        dense_case = DiffusionMaps(GaussianKernel(epsilon=1.25), n_eigenpairs=11).fit(
            data, store_kernel_matrix=True
        )
        sparse_case = DiffusionMaps(
            GaussianKernel(epsilon=1.25, distance=dict(cut_off=1e100)),
            n_eigenpairs=11,
        ).fit(data, store_kernel_matrix=True)

        nptest.assert_allclose(
            dense_case.kernel_matrix_,
            sparse_case.kernel_matrix_.toarray(),
            rtol=1e-13,
            atol=1e-14,
        )
        nptest.assert_allclose(
            dense_case.eigenvalues_, sparse_case.eigenvalues_, rtol=1e-13, atol=1e-14
        )

        assert_equal_eigenvectors(dense_case.eigenvectors_, sparse_case.eigenvectors_)

    def test_time_exponent(self):
        data, _ = make_swiss_roll(2000, random_state=1)

        actual1 = DiffusionMaps(
            GaussianKernel(epsilon=1.5), n_eigenpairs=5, time_exponent=0
        ).fit_transform(data)

        # With small positive time_exponent goes into a different routine, but has to
        # be approximately the same.
        actual2 = DiffusionMaps(
            GaussianKernel(epsilon=1.5), n_eigenpairs=5, time_exponent=1e-12
        ).fit_transform(data)

        nptest.assert_allclose(actual1, actual2, rtol=0, atol=1e-15)

    def test_symmetric_dense(self):
        data, _ = make_swiss_roll(2000, random_state=1)

        dmap1 = DiffusionMaps(
            GaussianKernel(epsilon=1.5), n_eigenpairs=5, symmetrize_kernel=True
        ).fit(data)

        dmap2 = DiffusionMaps(
            GaussianKernel(epsilon=1.5), n_eigenpairs=5, symmetrize_kernel=False
        ).fit(data)

        # make sure that the symmetric transformation is really used
        self.assertTrue(dmap1._dmap_kernel.is_conjugate)

        # Note: cannot compare kernel matrices, because they are only similar (sharing
        # same eigenvalues and eigenvectors [after transformation] not equal
        nptest.assert_allclose(
            dmap1.eigenvalues_, dmap2.eigenvalues_, rtol=1e-14, atol=1e-14
        )

        assert_equal_eigenvectors(dmap1.eigenvectors_, dmap2.eigenvectors_, tol=1e-13)

    def test_symmetric_sparse(self):
        data, _ = make_swiss_roll(1500, random_state=2)

        dmap1 = DiffusionMaps(
            GaussianKernel(epsilon=3, distance=dict(cut_off=1e100)),
            n_eigenpairs=5,
            symmetrize_kernel=True,
        ).fit(data)

        dmap2 = DiffusionMaps(
            GaussianKernel(epsilon=3, distance=dict(cut_off=1e100)),
            n_eigenpairs=5,
            symmetrize_kernel=False,
        ).fit(data)

        # make sure that the symmetric transformation is really used
        self.assertTrue(dmap1._dmap_kernel.is_conjugate)

        # Note: cannot compare kernel matrices, because they are only similar (sharing
        # same eigenvalues and eigenvectors [after transformation] not equal
        nptest.assert_allclose(
            dmap1.eigenvalues_, dmap2.eigenvalues_, rtol=1e-14, atol=1e-14
        )

        assert_equal_eigenvectors(dmap1.eigenvectors_, dmap2.eigenvectors_, tol=1e-13)

    def test_set_target_coords1(self):
        X_swiss_all, _ = make_swiss_roll(n_samples=2000, noise=0, random_state=5)

        actual_dmap = DiffusionMaps(
            GaussianKernel(epsilon=2.0), n_eigenpairs=6
        ).set_target_coords([1, 5])
        actual = actual_dmap.fit_transform(X_swiss_all)

        actual_dmap2 = DiffusionMaps(GaussianKernel(epsilon=2.0), n_eigenpairs=6)
        actual2 = (
            actual_dmap2.fit(X_swiss_all)
            .set_target_coords([1, 5])
            .transform(X_swiss_all)
        )

        expected = DiffusionMaps(
            GaussianKernel(epsilon=2.0), n_eigenpairs=6
        ).fit_transform(X_swiss_all)

        nptest.assert_array_equal(actual, expected[:, [1, 5]])
        # because of the additional computation in transform the results are slightly
        # different
        nptest.assert_allclose(actual2, expected[:, [1, 5]], rtol=0, atol=1e-16)
        nptest.assert_array_equal(actual_dmap.eigenvectors_, expected)
        nptest.assert_array_equal(actual_dmap2.eigenvectors_, expected)

        self.assertEqual(actual_dmap.n_features_out_, 2)
        self.assertEqual(actual_dmap2.n_features_out_, 2)

    def test_set_target_coords2(self):
        X_swiss_all, _ = make_swiss_roll(n_samples=2000, noise=0, random_state=5)

        actual_dmap = (
            DiffusionMaps(GaussianKernel(epsilon=2.0), n_eigenpairs=6)
            .set_target_coords([1, 5])
            .fit(X_swiss_all)
        )
        actual = actual_dmap.inverse_transform(actual_dmap.eigenvectors_[:, [1, 5]])
        self.assertEqual(actual.shape[1], actual_dmap.n_features_in_)

        with self.assertRaises(ValueError):
            # It is expected that only only reduced points map back
            actual_dmap.inverse_transform(actual_dmap.eigenvectors_)

    def test_set_target_coords3(self):
        X_swiss_all, _ = make_swiss_roll(n_samples=500, noise=0, random_state=5)

        actual_dmap = DiffusionMaps(GaussianKernel(epsilon=2.0), n_eigenpairs=6).fit(
            X_swiss_all
        )
        actual_dmap.set_target_coords(indices=[0, 1])

        with self.assertRaises(TypeError):
            actual_dmap.set_target_coords(indices=[0.0, 1.5])

        with self.assertRaises(ValueError):
            actual_dmap.set_target_coords(indices=[-1, 2])

        actual_dmap.set_target_coords(indices=[0, 5])
        with self.assertRaises(ValueError):
            actual_dmap.set_target_coords(indices=[0, 6])

    def test_nystrom_out_of_sample_swiss_roll(self, plot=False):
        X_swiss_all, color_all = make_swiss_roll(
            n_samples=4000, noise=0, random_state=5
        )

        setting = {
            "kernel": GaussianKernel(epsilon=1.7),
            "n_eigenpairs": 7,
            "is_stochastic": True,
            "alpha": 1,
            "symmetrize_kernel": True,
        }

        dmap_embed = DiffusionMaps(**setting).fit(X_swiss_all)

        if plot:
            plot_pairwise_eigenvector(
                eigenvectors=dmap_embed.transform(X_swiss_all),
                n=1,
                scatter_params=dict(c=color_all),
            )

        dmap_embed_eval_expected = dmap_embed.eigenvectors_[:, [1, 5]]
        dmap_embed_eval_actual = dmap_embed.set_target_coords(indices=[1, 5]).transform(
            X=X_swiss_all
        )

        # even though the target_coords were set, still all eigenvectors must be
        # accessible
        self.assertEqual(dmap_embed.eigenvectors_.shape[1], 7)

        nptest.assert_allclose(
            dmap_embed_eval_actual, dmap_embed_eval_expected, atol=1e-15
        )

        if plot:
            X_swiss_oos, color_oos = make_swiss_roll(
                n_samples=30000, noise=0, random_state=5
            )

            f, ax = plt.subplots(2, 3, figsize=(10, 8))
            marker = "."
            markersize = 0.2
            ax[0][0].scatter(
                dmap_embed_eval_expected[:, 0],
                dmap_embed_eval_expected[:, 1],
                s=markersize,
                marker=marker,
                c=color_all,
            )
            ax[0][0].set_title("expected (DMAP eigenvector)")

            ax[0][1].scatter(
                dmap_embed_eval_actual[:, 0],
                dmap_embed_eval_actual[:, 1],
                s=markersize,
                marker=marker,
                c=color_all,
            )
            ax[0][1].set_title("actual (DMAP Nyström on training data)")

            absdiff = np.abs(dmap_embed_eval_expected - dmap_embed_eval_actual)
            abs_error_norm = np.linalg.norm(absdiff, axis=1)

            error_scatter = ax[0][2].scatter(
                dmap_embed_eval_expected[:, 0],
                dmap_embed_eval_expected[:, 1],
                s=markersize,
                marker=marker,
                c=abs_error_norm,
                cmap=plt.get_cmap("Reds"),
            )

            f.colorbar(error_scatter, ax=ax[0][2])
            ax[0][2].set_title("abs. difference")

            gh_embed_eval_oos = dmap_embed.transform(X_swiss_oos)
            ax[1][0].scatter(
                gh_embed_eval_oos[:, 0],
                gh_embed_eval_oos[:, 1],
                s=markersize,
                marker=marker,
                c=color_oos,
            )

            ax[1][0].set_title(
                f"DMAP Nyström out-of-sample \n ({gh_embed_eval_oos.shape[0]} points) "
            )

            ax[1][2].text(
                0.01,
                0.5,
                f"both have same setting \n epsilon="
                f"{dmap_embed.kernel.epsilon}, symmetrize_kernel="
                f"{setting['symmetrize_kernel']}, "
                f"chosen_eigenvectors={[1, 5]}",
            )

            plt.show()

    def test_nystrom_out_of_sample_1dspiral(self, plot=False):
        def sample_1d_spiral(phis):
            c1 = phis * np.cos(phis)
            c2 = phis * np.sin(phis)
            return np.vstack([c1, c2]).T

        phis = np.linspace(0, np.pi * 4, 50)
        phis_oos = np.linspace(0, np.pi * 4, 50) - ((phis[1] - phis[0]) / 2)

        # remove first so that they are all between the phis-samples
        phis_oos = phis_oos[1:]

        X_all = sample_1d_spiral(phis)
        X_oos = sample_1d_spiral(phis_oos)

        # for variation use sparse code
        dmap_embed = DiffusionMaps(
            GaussianKernel(epsilon=0.9, distance=dict(cut_off=1e100)), n_eigenpairs=2
        ).fit(X_all)

        expected_oos = (
            dmap_embed.eigenvectors_[:-1, 1] + dmap_embed.eigenvectors_[1:, 1]
        ) / 2

        actual_oos = dmap_embed.set_target_coords(indices=[1]).transform(X_oos)

        self.assertLessEqual(
            mean_squared_error(expected_oos, actual_oos.ravel()), 6.559405995567413e-09
        )

        if plot:
            plt.plot(X_all[:, 0], X_all[:, 1], "-*")
            plt.plot(X_oos[:, 0], X_oos[:, 1], ".")
            plt.axis("equal")

            plt.figure()
            plt.plot(expected_oos, np.zeros(49), "+")
            plt.plot(dmap_embed.transform(X_all), np.zeros(50), "-*")
            plt.plot(dmap_embed.transform(X_oos), np.zeros(49), ".")

            plt.show()

    def test_out_of_sample_property(self, plot=False):
        # NOTE it is quite hard to compare a train dataset versus a "ground truth"
        # solution for kernel methods. This is because subsampling a training set from
        # the entire dataset changes the density in the dataset. Therefore, different
        # epsilon values are needed, which ultimately change the embedding itself.

        # Therefore, only reference solutions can be tested here.

        X_swiss_train, color_train = make_swiss_roll(2700, random_state=1)
        X_swiss_test, color_test = make_swiss_roll(1300, random_state=1)

        setting = {
            "kernel": GaussianKernel(epsilon=1.9),
            "n_eigenpairs": 7,
            "is_stochastic": True,
            "alpha": 1,
            "symmetrize_kernel": True,
        }

        dmap_embed = DiffusionMaps(**setting).fit(X_swiss_train)

        dmap_embed_test_eval = dmap_embed.set_target_coords(indices=[1, 5]).transform(
            X_swiss_test
        )

        if plot:
            plot_pairwise_eigenvector(
                eigenvectors=dmap_embed.eigenvectors_,
                n=1,
                scatter_params=dict(c=color_train),
            )

            plot_pairwise_eigenvector(
                eigenvectors=dmap_embed_test_eval,
                n=1,
                scatter_params=dict(c=color_test),
            )
            plt.show()

        # NOTE: These tests are only to detect potentially unwanted changes in computation
        # NOTE: For some reason the remote computer produces other results. Therefore,
        # it is only checked with "allclose"
        np.set_printoptions(precision=17)
        print(dmap_embed_test_eval.sum(axis=0))

        nptest.assert_allclose(
            dmap_embed_test_eval.sum(axis=0),
            (6.0898767014414625, 0.08601754715746428),
            atol=1e-15,
        )

        print(dmap_embed_test_eval.min(axis=0))
        nptest.assert_allclose(
            dmap_embed_test_eval.min(axis=0),
            (-0.0273443078497527, -0.03623258738512025),
            atol=1e-15,
        )

        print(dmap_embed_test_eval.max(axis=0))
        nptest.assert_allclose(
            np.abs(dmap_embed_test_eval.max(axis=0)),
            (0.02598525783966298, 0.03529902485787183),
            atol=1e-15,
        )

    def test_cknn_kernel(self):
        # Check that no errors are raised, for non-floating point kernels

        data = np.random.default_rng(1).random(size=(100, 100))

        for alpha, is_stochastic in zip([0, 0.5, 1], [True, False]):
            dmap = DiffusionMaps(
                ContinuousNNKernel(k_neighbor=4, delta=1.11),
                alpha=alpha,
                is_stochastic=is_stochastic,
            )
            dmap = dmap.fit(data, store_kernel_matrix=True)

            self.assertIsInstance(dmap.kernel_matrix_, scipy.sparse.csr_matrix)
            if is_stochastic:
                self.assertEqual(dmap.kernel_matrix_.dtype, float)
            else:
                self.assertEqual(dmap.kernel_matrix_.dtype, bool)

            self.assertIsInstance(dmap.eigenvectors_, np.ndarray)
            self.assertIsInstance(dmap.eigenvalues_, np.ndarray)

    def test_dynamic_kernel(self):
        _x = np.linspace(0, 2 * np.pi, 20)
        df = pd.DataFrame(
            np.column_stack([np.sin(_x), np.cos(_x)]), columns=["sin", "cos"]
        )
        tsc_data = TSCDataFrame.from_single_timeseries(df=df)

        dmap = DiffusionMaps(kernel=ConeKernel()).fit(tsc_data)

        self.assertIsInstance(dmap.eigenvectors_, TSCDataFrame)

        actual_forward = dmap.transform(tsc_data.iloc[:10])
        self.assertIsInstance(actual_forward, TSCDataFrame)

        actual_inverse = dmap.inverse_transform(actual_forward)
        self.assertIsInstance(actual_inverse, TSCDataFrame)

        with self.assertRaises(TypeError):
            dmap.transform(tsc_data.iloc[:10].to_numpy())

    def test_distance(self):
        _x = np.linspace(0, 2 * np.pi, 20)
        df = pd.DataFrame(
            np.column_stack([np.sin(_x), np.cos(_x)]), columns=["sin", "cos"]
        )
        tsc_data = TSCDataFrame.from_single_timeseries(df=df)

        dmap = DiffusionMaps(kernel=GaussianKernel(distance=dict(cut_off=2))).fit(
            tsc_data, store_kernel_matrix=True
        )

        # cut-off is squared because it is squared euclidean (while the cut-off is given
        # in Euclidean metric)
        self.assertEqual(dmap._dmap_kernel.distance.cut_off, 4)
        self.assertIsInstance(dmap.kernel_matrix_, scipy.sparse.csr_matrix)

    def test_knn_kernel_matrix(self, plot=False):
        X_swiss_train, color_train = make_swiss_roll(2700, random_state=1)
        X_swiss_oos, color_oos = make_swiss_roll(1300, random_state=1)

        setting = {
            "kernel": GaussianKernel(
                epsilon=2.1, distance=SklearnKNN(metric="sqeuclidean", k=100)
            ),
            "n_eigenpairs": 7,
            "is_stochastic": True,
            "alpha": 1,
        }

        dmap_list = [None, None]

        for i, symmetrize_kernel in enumerate([True, False]):
            setting["symmetrize_kernel"] = symmetrize_kernel
            dmap_list[i] = DiffusionMaps(**setting).fit(
                X_swiss_train, store_kernel_matrix=True
            )
            psi_oos = dmap_list[i].transform(X_swiss_oos)

            self.assertFalse(is_symmetric_matrix(dmap_list[i].kernel_matrix_))

            reconst_oos = dmap_list[i].inverse_transform(psi_oos)

            # Test for linear reconstruction to identify future changes
            self.assertLessEqual(np.linalg.norm(X_swiss_oos - reconst_oos), 72.441387)

        nptest.assert_array_equal(
            dmap_list[0].kernel_matrix_.toarray(), dmap_list[1].kernel_matrix_.toarray()
        )
        nptest.assert_array_equal(
            dmap_list[0].eigenvectors_, dmap_list[1].eigenvectors_
        )
        nptest.assert_array_equal(dmap_list[0].eigenvalues_, dmap_list[1].eigenvalues_)

        if plot:
            plot_pairwise_eigenvector(
                eigenvectors=dmap_list[0].eigenvectors_,
                n=1,
                scatter_params=dict(c=color_train),
            )

            plot_pairwise_eigenvector(
                eigenvectors=psi_oos,
                n=1,
                scatter_params=dict(c=color_oos),
            )
            plt.show()

    def test_kernel_symmetric_conjugate(self):
        X = make_swiss_roll(1000)[0]

        # The expected kernel is the one where no conjugate transform is performed
        expected = DiffusionMaps(
            kernel=GaussianKernel(epsilon=lambda K: np.median(K)),
            n_eigenpairs=3,
            symmetrize_kernel=False,
        ).fit(X, store_kernel_matrix=True)

        # It is checked if the true kernel matrix is recovered
        actual = DiffusionMaps(
            kernel=GaussianKernel(lambda K: np.median(K)),
            n_eigenpairs=3,
            symmetrize_kernel=True,
        ).fit(X, store_kernel_matrix=True)

        nptest.assert_allclose(
            expected.kernel_matrix_, actual.kernel_matrix_, atol=1e-15, rtol=1e-15
        )

    def test_types_tsc(self):
        # fit=TSCDataFrame
        _x = np.linspace(0, 2 * np.pi, 20)
        df = pd.DataFrame(
            np.column_stack([np.sin(_x), np.cos(_x)]), columns=["sin", "cos"]
        )

        tsc_data = TSCDataFrame.from_single_timeseries(df=df)

        dmap = DiffusionMaps(kernel=GaussianKernel(epsilon=0.4)).fit(
            tsc_data, store_kernel_matrix=True
        )

        self.assertIsInstance(dmap.eigenvectors_, TSCDataFrame)

        # insert TSCDataFrame -> output TSCDataFrame
        actual_tsc = dmap.transform(tsc_data.iloc[:10, :])
        self.assertIsInstance(actual_tsc, TSCDataFrame)

        # insert np.ndarray -> output np.ndarray
        actual_nd = dmap.transform(tsc_data.iloc[:10, :].to_numpy())
        self.assertIsInstance(actual_nd, np.ndarray)

        # check that compuation it exactly the same
        nptest.assert_array_equal(actual_tsc.to_numpy(), actual_nd)

    def test_types_pcm(self):
        # fit=TSCDataFrame
        _x = np.linspace(0, 2 * np.pi, 20)
        pcm_data = np.column_stack([np.sin(_x), np.cos(_x)])
        tsc_data = TSCDataFrame.from_single_timeseries(
            pd.DataFrame(pcm_data, columns=["sin", "cos"])
        )

        dmap = DiffusionMaps(kernel=GaussianKernel(epsilon=0.4)).fit(
            pcm_data, store_kernel_matrix=True
        )

        self.assertIsInstance(dmap.eigenvectors_, np.ndarray)
        self.assertIsInstance(dmap.kernel_matrix_, np.ndarray)

        # insert np.ndarray -> output np.ndarray
        actual_nd = dmap.transform(pcm_data[:10, :])
        self.assertIsInstance(actual_nd, np.ndarray)

        # insert TSCDataFrame -> time information is returned, even when during fit no
        # time series data was returned
        actual_tsc = dmap.transform(tsc_data.iloc[:10, :])
        self.assertIsInstance(actual_tsc, TSCDataFrame)

        nptest.assert_array_equal(actual_nd, actual_tsc)

        single_sample = tsc_data.iloc[[0], :]
        actual = dmap.transform(single_sample)
        self.assertIsInstance(actual, TSCDataFrame)

    def test_sparse_time_series_collection(self):
        X1 = pd.DataFrame(make_swiss_roll(n_samples=250)[0])
        X2 = pd.DataFrame(make_swiss_roll(n_samples=250)[0])

        X = TSCDataFrame.from_frame_list([X1, X2])

        actual_dmap = DiffusionMaps(
            kernel=GaussianKernel(epsilon=1.25, distance=dict(cut_off=10)),
            n_eigenpairs=6,
        )
        actual_result = actual_dmap.fit_transform(X, store_kernel_matrix=True)

        expected_dmap = DiffusionMaps(
            kernel=GaussianKernel(epsilon=1.25, distance=dict(cut_off=10)),
            n_eigenpairs=6,
        )
        expected_result = expected_dmap.fit_transform(
            X.to_numpy(), store_kernel_matrix=True
        )

        self.assertIsInstance(actual_dmap.kernel_matrix_, scipy.sparse.csr_matrix)
        self.assertIsInstance(expected_dmap.kernel_matrix_, scipy.sparse.csr_matrix)
        self.assertIsInstance(actual_result, TSCDataFrame)
        self.assertIsInstance(expected_result, np.ndarray)

        nptest.assert_equal(actual_result.to_numpy(), expected_result)

    @unittest.skipIf(not IMPORTED_RDIST, reason="rdist not installed")
    def test_cknn_kernel2(self):
        from time import time

        import datafold.pcfold as pfold

        k_neighbor = 15
        delta = 1

        num_samples = 500
        xmin, ymin = -2, -1
        width, height = 4, 2

        data = make_strip(xmin, ymin, width, height, num_samples)

        t0 = time()
        pcm = pfold.PCManifold(data)
        pcm.optimize_parameters()

        t1 = time()
        cknn_kernel = pfold.kernels.ContinuousNNKernel(
            k_neighbor=k_neighbor,
            delta=delta,
            distance=dict(cut_off=pcm.cut_off, backend="rdist"),
        )
        k, distance = cknn_kernel(
            pcm,
        )
        t2 = time()

        dmap = DiffusionMaps(n_eigenpairs=10)
        dmap._dmap_kernel = cknn_kernel
        dmap.fit(pcm)

        t3 = time()

        print(f"kernel has {k.nnz/k.shape[0]} neighbors per row, on {k.shape[0]} rows")
        print(f"pcm: {t1-t0}, cknn kernel: {t2-t1}, dmap: {t3-t2}")

    @unittest.skip(reason="Speed test without any asssertions, use if required")
    def test_speed(self):
        from time import time

        import datafold.pcfold as pfold

        num_samples = 15000
        xmin, ymin = -2, -1
        width, height = 4, 2
        random_state = 1

        data = make_strip(xmin, ymin, width, height, num_samples)
        rng = np.random.default_rng(random_state)

        n_large_points = int(15000)
        L1, L2 = 4, 3  # width and height of the rectangle
        data = rng.uniform(
            low=(-L1 / 2, -L2 / 2), high=(L1 / 2, L2 / 2), size=(n_large_points, 2)
        )

        pcm = pfold.PCManifold(data)
        pcm.optimize_parameters()

        setting = {
            "kernel": GaussianKernel(
                pcm.kernel.epsilon,
                distance={"cut_off": pcm.cut_off, "backend": "scipy.kdtree"},
            ),
            "n_eigenpairs": 5,
            "is_stochastic": True,
            "alpha": 1,
            "symmetrize_kernel": True,
        }

        dmap_embed = DiffusionMaps(**setting)
        kernel = deepcopy(dmap_embed.kernel)

        t1 = time()
        dmap_embed.fit(data, store_kernel_matrix=True)
        t2 = time()
        # compute kernel
        kernel(data)
        t22 = time()

        solver_kwargs = {
            "k": setting["n_eigenpairs"],
            "which": "LM",
            "v0": np.ones(data.shape[0]),
            "tol": 1e-14,
        }
        _, _ = scipy.sparse.linalg.eigsh(dmap_embed.kernel_matrix_, **solver_kwargs)
        t3 = time()

        print(
            f"kernel+eigsh: {t22-t2+t3-t2}, fit: {t2-t1}, "
            f"kernel only: {t22-t2}, eigsh: {t3-t2}"
        )

        return 1


class DiffusionMapsLegacyTest(unittest.TestCase):
    """We want to produce exactly the same results as the forked DMAP repository. These
    are test to make sure this is the case. All dmaps have symmetrize_kernel=False to
    be able to compare the kernel.
    """

    @pytest.fixture(autouse=True)
    def run_before_each_test(self):
        """This runs before each test."""
        DiffusionMapsTest.mock_eigensolver_call()
        yield

    def test_simple_dataset(self):
        """Taken from method_examples(/diffusion_maps/diffusion_maps.ipynb) repository."""
        data, epsilon = circle_data()

        actual = DiffusionMaps(
            kernel=GaussianKernel(epsilon=epsilon),
            n_eigenpairs=11,
            symmetrize_kernel=False,
        ).fit(data)
        expected = legacy_dmap.DiffusionMaps(points=data, epsilon=epsilon)

        cmp_eigenpairs(actual, expected)

    def test_kernel_matrix_simple_dense(self):
        data, epsilon = circle_data()

        actual = DiffusionMaps(
            GaussianKernel(epsilon=epsilon), n_eigenpairs=11, symmetrize_kernel=False
        ).fit(data, store_kernel_matrix=True)
        expected = legacy_dmap.DenseDiffusionMaps(points=data, epsilon=epsilon)

        cmp_kernel_matrix(actual, expected, rtol=1e-14, atol=1e-15)

    def test_kernel_matrix_simple_sparse(self):
        data, epsilon = circle_data(nsamples=1000)

        actual = DiffusionMaps(
            GaussianKernel(epsilon=epsilon, distance=dict(cut_off=1e100)),
            n_eigenpairs=11,
            symmetrize_kernel=False,
        ).fit(data, store_kernel_matrix=True)
        expected = legacy_dmap.SparseDiffusionMaps(points=data, epsilon=epsilon)

        nptest.assert_allclose(
            actual.kernel_matrix_.toarray(),
            expected.kernel_matrix.toarray(),
            rtol=1e-14,
            atol=1e-15,
        )

    def test_swiss_roll_dataset(self):
        """Taken from method_examples(/diffusion_maps/diffusion_maps.ipynb) repository."""
        data, _ = make_swiss_roll(n_samples=1000, noise=0.01, random_state=1)

        actual = DiffusionMaps(
            GaussianKernel(epsilon=1.25), n_eigenpairs=11, symmetrize_kernel=False
        ).fit(data)
        expected = legacy_dmap.DenseDiffusionMaps(points=data, epsilon=1.25)

        cmp_eigenpairs(actual, expected)

    def test_multiple_cutoff(self):
        data1, _ = make_swiss_roll(1000, random_state=0)
        data1 *= 0.01  # scale data down to allow for smaller cut_offs

        data2, epsilon2 = circle_data()

        all_cut_offs = np.append(np.linspace(0.1, 1, 5), np.linspace(1, 10, 5))

        ne = 5
        for cut_off in all_cut_offs:
            actual1 = DiffusionMaps(
                GaussianKernel(epsilon=1e-3, distance=dict(cut_off=cut_off)),
                n_eigenpairs=ne,
                symmetrize_kernel=False,
            ).fit(data1)
            expected1 = legacy_dmap.DiffusionMaps(
                points=data1, epsilon=1e-3, num_eigenpairs=ne, cut_off=cut_off
            )

            cmp_eigenpairs(actual1, expected1)

            actual2 = DiffusionMaps(
                GaussianKernel(epsilon=epsilon2, distance=dict(cut_off=cut_off)),
                n_eigenpairs=ne,
                symmetrize_kernel=False,
            ).fit(data2)
            expected2 = legacy_dmap.DiffusionMaps(
                points=data2, epsilon=epsilon2, num_eigenpairs=ne, cut_off=cut_off
            )

            cmp_eigenpairs(actual2, expected2)

    def test_normalized_kernel(self):
        data, _ = make_swiss_roll(1000, random_state=123)
        epsilon = 1.25

        # actual = DiffusionMaps(
        #     epsilon=epsilon, n_eigenpairs=11, is_stochastic=False
        # ).fit(data)
        # expected = legacy_dmap.DenseDiffusionMaps(
        #     data, epsilon=1.25, normalize_kernel=False
        # )
        # cmp_kernel_matrix(actual, expected, rtol=1e-14, atol=1e-14)
        #
        # actual = DiffusionMaps(
        #     epsilon=epsilon, n_eigenpairs=11, is_stochastic=True
        # ).fit(data)
        # expected = legacy_dmap.DenseDiffusionMaps(
        #     data, epsilon=1.25, normalize_kernel=True
        # )
        # cmp_kernel_matrix(actual, expected, rtol=1e-15, atol=1e-15)

        # Sparse case
        actual = DiffusionMaps(
            GaussianKernel(epsilon=epsilon, distance=dict(cut_off=3)),
            n_eigenpairs=11,
            is_stochastic=False,
        ).fit(data, store_kernel_matrix=True)

        expected = legacy_dmap.SparseDiffusionMaps(
            data, epsilon=1.25, num_eigenpairs=11, cut_off=3, normalize_kernel=False
        )

        cmp_kernel_matrix(actual, expected, rtol=1e-15, atol=1e-15)

        actual = DiffusionMaps(
            GaussianKernel(epsilon=epsilon, distance=dict(cut_off=3)),
            n_eigenpairs=11,
            is_stochastic=True,
            symmetrize_kernel=False,
        ).fit(data, store_kernel_matrix=True)
        expected = legacy_dmap.SparseDiffusionMaps(
            data, epsilon=1.25, num_eigenpairs=11, cut_off=3, normalize_kernel=True
        )
        cmp_kernel_matrix(actual, expected, rtol=1e-15, atol=1e-15)

    def test_renormalization_factor(self):
        data, _ = make_swiss_roll(1000, random_state=1)
        nfactor = np.linspace(0, 1, 5)

        for factor in nfactor:
            actual = DiffusionMaps(
                GaussianKernel(epsilon=1.25),
                n_eigenpairs=11,
                symmetrize_kernel=False,
                alpha=factor,
            ).fit(data, store_kernel_matrix=True)
            expected = legacy_dmap.DenseDiffusionMaps(
                data, epsilon=1.25, renormalization=factor
            )
            cmp_dmap_legacy(actual, expected, rtol=1e-15, atol=1e-15)

            actual = DiffusionMaps(
                GaussianKernel(epsilon=1.25, distance=dict(cut_off=3)),
                n_eigenpairs=11,
                symmetrize_kernel=False,
                alpha=factor,
            ).fit(data, store_kernel_matrix=True)

            expected = legacy_dmap.SparseDiffusionMaps(
                data, epsilon=1.25, cut_off=3, renormalization=factor
            )
            cmp_dmap_legacy(actual, expected, rtol=1e-15, atol=1e-15)

    def test_multiple_epsilon(self):
        data, _ = make_swiss_roll(1000, random_state=123)
        epsilons = np.linspace(1.2, 1.7, 5)[1:]
        n_eigenpairs = 5
        for eps in epsilons:
            actual_dense = DiffusionMaps(
                GaussianKernel(epsilon=eps),
                n_eigenpairs=n_eigenpairs,
                symmetrize_kernel=False,
            ).fit(data, store_kernel_matrix=True)
            expected_dense = legacy_dmap.DenseDiffusionMaps(
                points=data, num_eigenpairs=n_eigenpairs, epsilon=eps
            )

            actual_sparse = DiffusionMaps(
                GaussianKernel(epsilon=eps, distance=dict(cut_off=3)),
                n_eigenpairs=n_eigenpairs,
                symmetrize_kernel=False,
            ).fit(data, store_kernel_matrix=True)
            expected_sparse = legacy_dmap.SparseDiffusionMaps(
                points=data, epsilon=eps, num_eigenpairs=n_eigenpairs, cut_off=3
            )

            cmp_dmap_legacy(actual_dense, expected_dense, rtol=1e-15, atol=1e-15)
            cmp_dmap_legacy(actual_sparse, expected_sparse, rtol=1e-14, atol=1e-14)

    def test_num_eigenpairs(self):
        data, _ = make_swiss_roll(1000)
        all_n_eigenpairs = np.linspace(10, 50, 5).astype(int)

        for n_eigenpairs in all_n_eigenpairs:
            actual = DiffusionMaps(
                GaussianKernel(epsilon=1.25),
                n_eigenpairs=n_eigenpairs,
                symmetrize_kernel=False,
            ).fit(data, store_kernel_matrix=True)
            expected = legacy_dmap.DenseDiffusionMaps(
                data, epsilon=1.25, num_eigenpairs=n_eigenpairs
            )

            cmp_dmap_legacy(actual, expected, rtol=1e-15, atol=1e-15)

            actual = DiffusionMaps(
                GaussianKernel(epsilon=1.25, distance=dict(cut_off=3)),
                n_eigenpairs=n_eigenpairs,
                symmetrize_kernel=False,
            ).fit(data, store_kernel_matrix=True)
            expected = legacy_dmap.SparseDiffusionMaps(
                data, epsilon=1.25, cut_off=3, num_eigenpairs=n_eigenpairs
            )

            cmp_dmap_legacy(actual, expected, rtol=1e-15, atol=1e-15)


class LocalRegressionSelectionTest(unittest.TestCase):
    def test_n_subsample(self):
        X = np.random.default_rng(1).uniform(size=(100, 10))
        dmaps = DiffusionMaps(
            GaussianKernel(epsilon=2.1), n_eigenpairs=6
        ).fit_transform(X)

        # no error
        LocalRegressionSelection(n_subsample=np.inf).fit_transform(dmaps)
        LocalRegressionSelection(n_subsample=20).fit_transform(dmaps)

        with self.assertRaises(ValueError):
            LocalRegressionSelection(n_subsample=1000).fit_transform(dmaps)

    def test_automatic_eigendirection_selection_swiss_roll(self):
        X, color = make_swiss_roll(n_samples=5000, noise=0.01, random_state=1)
        dm = DiffusionMaps(GaussianKernel(epsilon=2.1), n_eigenpairs=6).fit(X)

        loc_regress = LocalRegressionSelection(n_subsample=1000)
        loc_regress = loc_regress.fit(dm.eigenvectors_)

        self.assertTrue(np.isnan(loc_regress.residuals_[0]))
        self.assertTrue(loc_regress.residuals_[1] == 1.0)

        # only starting from 2 because the first two values are trivial
        self.assertTrue(np.argmax(loc_regress.residuals_[2:]) == 3)

    def test_automatic_eigendirection_selection_rectangle(self):
        """Test taken from:
        Paper: Parsimonious Representation of Nonlinear Dynamical Systems Through
        Manifold Learning: A Chemotaxis Case Study, Dsila et al., page 7
        https://arxiv.org/abs/1505.06118v1.
        """
        n_samples = 5000
        n_subsample = 500

        # lengths 2, 4, 8 are from paper, added .3 to have it more clear on which index
        # the next independent eigenfunction should appear
        x_length_values = [1, 2.3, 4.3, 8.3]

        rng = np.random.default_rng(1)

        for xlen in x_length_values:
            x_direction = rng.uniform(0, xlen, size=(n_samples, 1))
            y_direction = rng.uniform(size=(n_samples, 1))
            X = np.hstack([x_direction, y_direction])

            dmap = DiffusionMaps(kernel=GaussianKernel(0.1), n_eigenpairs=10).fit(X)

            loc_regress = LocalRegressionSelection(n_subsample=n_subsample)
            loc_regress.fit(dmap.eigenvectors_)

            # Trivial first two values:
            self.assertTrue(np.isnan(loc_regress.residuals_[0]))
            self.assertTrue(loc_regress.residuals_[1] == 1.0)  # always first directions

            loc_regress.residuals_[0:2] = 0  # setting to zero for easier checking

            # Ignoring the first two trivial cases:
            # From the paper-example we know the position of the next independent
            # eigendirection
            self.assertEqual(int(xlen + 1), np.argmax(loc_regress.residuals_))

    def test_api_automatic_parametrization(self):
        # Same test as test_choose_automatic_parametrization, just using the proper
        # sklean-like API
        n_samples = 5000
        n_subsample = 500

        x_length_values = [2.3, 4.3, 8.3]

        rng = np.random.default_rng(1)

        for xlen in x_length_values:
            x_direction = rng.uniform(0, xlen, size=(n_samples, 1))
            y_direction = rng.uniform(size=(n_samples, 1))

            data = np.hstack([x_direction, y_direction])
            dmap = DiffusionMaps(kernel=GaussianKernel(0.1), n_eigenpairs=10).fit(data)

            # -----------------------------------
            # Streategy 1: choose by dimension
            for s, kwargs in [
                ("dim", dict(intrinsic_dim=2)),
                ("threshold", dict(regress_threshold=0.5)),
            ]:
                loc_regress_dim = LocalRegressionSelection(
                    n_subsample=n_subsample, strategy=s, random_state=5, **kwargs
                )
                actual = loc_regress_dim.fit_transform(dmap.eigenvectors_)

                actual_indices = loc_regress_dim.evec_indices_
                expected_indices = np.array([1, int(xlen + 1)])

                nptest.assert_equal(actual_indices, expected_indices)

                expected = dmap.eigenvectors_[:, actual_indices]
                nptest.assert_array_equal(actual, expected)


class DiffusionMapsVariableTest(unittest.TestCase):
    @staticmethod
    def eig_neg_factor(exact, approx):
        # chooses the right "direction" of eigenvector to compare with exact solution
        n1 = exact.flatten() @ approx.flatten()
        n2 = exact.flatten() @ (-1 * approx.flatten())

        if n2 > n1:
            return -1
        else:
            return 1

    @staticmethod
    def plot_quantities(data, dmap):
        h3 = lambda x: 1 / np.sqrt(6) * (x**3 - 3 * x)  # 3rd Hermetian polynomial
        assert data.ndim == 2 and data.shape[1] == 1

        f, ax = plt.subplots(ncols=3, nrows=3)
        f.suptitle(
            f"N={data.shape[0]}, "
            f"eps={dmap.epsilon}, "
            f"beta={dmap.beta}, "
            f"expected_dim={dmap.expected_dim}, "
            f"nn_bandwidth={dmap.nn_bandwidth}"
        )

        ax[0][0].plot(data, dmap.rho0_, "-")
        ax[0][0].set_title("rho0 - ad hoc bandwidth function")

        ax[0][1].hist(data, density=True, bins=100, color="grey", edgecolor="black")
        ax[0][1].plot(data, dmap.peq_est_, "*", color="#1f77b4", label="estimate")
        ax[0][1].set_title("hist distribution data")

        factor = DiffusionMapsVariableTest.eig_neg_factor(
            h3(data), dmap.eigenvectors_[:, 3]
        )
        ax[1][0].plot(
            np.linspace(-3, 3, 200), h3(np.linspace(-3, 3, 200)), label="exact, H3"
        )
        ax[1][0].plot(
            data[:, 0],
            factor * dmap.eigenvectors_[:, 3],
            "-",
            label="dmap_variable_kernel, ev_idx=3",
        )

        ax[1][0].legend()

        ax[0][2].plot(data, dmap.rho_, "*")
        ax[0][2].set_title("rho - bandwidth function")
        ax[1][1].plot(data, dmap.q0_, "*")
        ax[1][1].set_title("q0 - sampling density - estimate")
        ax[1][2].plot(data, dmap.peq_est_, "*", label="estimate")

        ax[1][2].plot(
            np.linspace(-3, 3, 200),
            norm.pdf(np.linspace(-3, 3, 200), 0, 1),
            label="exact",
        )
        ax[1][2].legend()
        ax[1][2].set_title("peq - invariant measure, estimate")

        M = dmap.eigenvalues_.shape[0]
        ax[2][0].plot(np.arange(M), dmap.eigenvalues_, "*-")
        ax[2][0].set_xlabel("idx")
        ax[2][0].set_ylabel("eigval")

        im = ax[2][1].imshow(
            np.abs(dmap.eigenvectors_.T @ dmap.eigenvectors_)
            / dmap.eigenvectors_.shape[0]
        )
        ax[2][1].set_title("inner products of EV (abs and rel)")
        f.colorbar(im, ax=ax[2][1])

    def test_ornstein_uhlenbeck(self, plot=False):
        from scipy.special import erfinv

        nr_samples = 5000
        n_eigenpairs = 20

        def compute_nice_ou(N):
            # non-random sampling
            delta = 1 / (N + 1)
            xtilde = delta * np.arange(1, N + 1)
            x = np.sqrt(2) * erfinv(2 * xtilde - 1)

            # bool_idx = np.logical_and(x >= -3, x <=3)
            return x[:, np.newaxis]

        X = compute_nice_ou(nr_samples)

        dmap = DiffusionMapsVariable(
            epsilon=0.001,
            n_eigenpairs=n_eigenpairs,
            nn_bandwidth=100,
            expected_dim=1,
            beta=-0.5,
            symmetrize_kernel=True,
        ).fit(X)

        if plot:
            DiffusionMapsVariableTest.plot_quantities(X, dmap)
            plt.show()

        # TESTS:
        h3 = lambda x: 1 / np.sqrt(6) * (x**3 - 3 * x)  # 3rd Hermitian polynomial
        factor = DiffusionMapsVariableTest.eig_neg_factor(
            h3(X), dmap.eigenvectors_[:, 3]
        )

        actual = factor * dmap.eigenvectors_[:, 3]
        expected = h3(X)

        # using only a reference computation (fails if quality gets worse)
        self.assertLessEqual(np.abs(actual - expected.ravel()).max(), 1.5943698803387)

        actual = dmap.peq_est_
        expected = norm.pdf(X, 0, 1)

        # using only a reference computation (fails if quality gets worse)
        nptest.assert_allclose(
            actual, expected.ravel(), atol=0.0002519, rtol=0.29684159
        )
