"""62-make-diffusionmaps-and-geometricharmonicsinterpolator-compatible-with-scikit-learn-api
Unit test for the Geometric Harmonics module.
"""

import unittest

import diffusion_maps as legacy_dmap
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nptest
from scipy.sparse import csr_matrix
from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import ParameterGrid
from sklearn.utils.estimator_checks import check_estimator

from datafold.dynfold import DiffusionMaps
from datafold.dynfold.outofsample import (
    GeometricHarmonicsInterpolator,
    LaplacianPyramidsInterpolator,
    MultiScaleGeometricHarmonicsInterpolator,
)
from datafold.dynfold.tests.test_helper import (
    assert_equal_eigenvectors,
    make_points,
    make_strip,
)
from datafold.pcfold.distance import IS_IMPORTED_RDIST
from datafold.pcfold.kernels import GaussianKernel


def plot_scatter(points: np.ndarray, values: np.ndarray, **kwargs) -> None:
    title = kwargs.pop("title", None)
    if title:
        plt.title(title)
    plt.scatter(
        points[:, 0],
        points[:, 1],
        c=values,
        marker="o",
        rasterized=True,
        s=2.5,
        **kwargs,
    )
    cb = plt.colorbar()
    cb.set_clim([np.min(values), np.max(values)])
    cb.set_ticks(np.linspace(np.min(values), np.max(values), 5))
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.gca().set_aspect("equal")


def f(points: np.ndarray) -> np.ndarray:
    """Function to interpolate."""
    # return np.ones(points.shape[0])
    # return np.arange(points.shape[0])
    return np.sin(np.linalg.norm(points, axis=-1))


class GeometricHarmonicsTest(unittest.TestCase):
    # TODO: not tested yet:
    #  * error measurements (kfold, etc.), also with nD interpolation

    def setUp(self):
        # self.num_points = 1000
        # self.points = downsample(np.load('data.npy'), self.num_points)
        # self.values = np.ones(self.num_points)

        # np.save('actual-data.npy', self.points)

        # self.points = np.load('actual-data.npy')
        # self.num_points = self.points.shape[0]
        # self.values = np.ones(self.num_points)

        self.points = make_points(23, -4, -4, 4, 4)
        self.num_points = self.points.shape[0]
        self.values = f(self.points)

    def test_valid_sklearn_estimator(self):
        # disable check on  boston housing dataset
        # see: https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        estimator = GeometricHarmonicsInterpolator(n_eigenpairs=1)

        for e, check in check_estimator(estimator, generate_only=True):
            check(e)

        self.assertTrue(estimator._get_tags()["multioutput"])
        self.assertTrue(estimator._get_tags()["requires_y"])

    def test_geometric_harmonics_interpolator(self, plot=False):
        eps = 1e-1

        ghi = GeometricHarmonicsInterpolator(
            GaussianKernel(epsilon=eps, distance=dict(cut_off=1e1 * eps)),
            n_eigenpairs=self.num_points - 3,
        )
        ghi = ghi.fit(self.points, self.values)

        points = make_points(100, -4, -4, 4, 4)

        values = ghi.predict(points)

        residual = values - f(points)
        self.assertLess(np.max(np.abs(residual)), 7.5e-2)

        print(f"Original function={f(points)}")
        print(f"Sampled points={self.values}")
        print(f"Reconstructed function={values}")
        print(f"Residual={residual}")

        if plot:
            plt.subplot(2, 2, 1)
            plot_scatter(points, f(points), title="Original function")

            plt.subplot(2, 2, 2)
            plot_scatter(self.points, self.values, title="Sampled function")

            plt.subplot(2, 2, 4)
            plot_scatter(points, values, title="Reconstructed function")

            plt.subplot(2, 2, 3)
            plot_scatter(points, residual, title="Residual", cmap="RdBu_r")

            plt.tight_layout()
            plt.show()

    def test_eigenfunctions(self, plot=False):
        eps = 1e1
        cut_off = 1e1 * eps
        n_eigenpairs = 3

        points = make_strip(0, 0, 1, 1e-1, 3000)

        dm = DiffusionMaps(
            GaussianKernel(epsilon=eps, distance=dict(cut_off=1e100)),
            n_eigenpairs=n_eigenpairs,
        ).fit(points)

        setting = {
            "kernel": GaussianKernel(eps, distance=dict(cut_off=cut_off)),
            "n_eigenpairs": n_eigenpairs,
            "is_stochastic": False,
        }

        ev1 = GeometricHarmonicsInterpolator(**setting).fit(
            points, dm.eigenvectors_[:, 1]
        )
        ev2 = GeometricHarmonicsInterpolator(**setting).fit(
            points, dm.eigenvectors_[:, 2]
        )

        rel_err1 = np.linalg.norm(
            dm.eigenvectors_[:, 1] - ev1.predict(points), np.inf
        ) / np.linalg.norm(dm.eigenvectors_[:, 1], np.inf)
        self.assertAlmostEqual(rel_err1, 0, places=1)

        rel_err2 = np.linalg.norm(
            dm.eigenvectors_[:, 2] - ev2.predict(points), np.inf
        ) / np.linalg.norm(dm.eigenvectors_[:, 2], np.inf)
        self.assertAlmostEqual(rel_err2, 0, places=1)

        if plot:
            new_points = make_points(50, 0, 0, 1, 1e-1)
            ev1i = ev1.predict(new_points)
            ev2i = ev2.predict(new_points)
            plt.subplot(1, 2, 1)
            plt.scatter(new_points[:, 0], new_points[:, 1], c=ev1i, cmap="RdBu_r")
            plt.subplot(1, 2, 2)
            plt.scatter(new_points[:, 0], new_points[:, 1], c=ev2i, cmap="RdBu_r")
            plt.show()

    def test_dense_sparse(self):
        data, _ = make_swiss_roll(n_samples=1000, noise=0, random_state=1)
        dim_red_eps = 1.25

        dense_setting = {
            "kernel": GaussianKernel(dim_red_eps, distance=dict(cut_off=np.inf)),
            "n_eigenpairs": 6,
            "is_stochastic": False,
        }

        sparse_setting = {
            "kernel": GaussianKernel(dim_red_eps, distance=dict(cut_off=1e100)),
            "n_eigenpairs": 6,
            "is_stochastic": False,
        }

        dmap_dense = DiffusionMaps(**dense_setting).fit(data)
        values = dmap_dense.eigenvectors_[:, 1]

        # The parameters are set equal to the previously generated DMAP, therefore both
        # have to be equal.
        gh_dense_cmp = GeometricHarmonicsInterpolator(**dense_setting).fit(
            data, values, store_kernel_matrix=True
        )
        gh_sparse_cmp = GeometricHarmonicsInterpolator(**sparse_setting).fit(
            data, values, store_kernel_matrix=True
        )

        # Check the the correct format is set
        self.assertTrue(isinstance(gh_dense_cmp.kernel_matrix_, np.ndarray))
        self.assertTrue(isinstance(gh_sparse_cmp.kernel_matrix_, csr_matrix))

        gh_dense_cmp.predict(data)
        gh_sparse_cmp.predict(data)

        # Check if sparse (without cutoff) and dense case give close results
        nptest.assert_allclose(
            gh_sparse_cmp.predict(data),
            gh_dense_cmp.predict(data),
            rtol=1e-14,
            atol=1e-15,
        )
        nptest.assert_allclose(
            gh_sparse_cmp.gradient(data),
            gh_dense_cmp.gradient(data),
            rtol=1e-14,
            atol=1e-15,
        )

    def test_variable_number_of_points(self):
        # Simply check if something fails

        rng = np.random.default_rng(1)

        data = rng.normal(size=(100, 5))
        values = rng.normal(size=100)

        parameter_grid = ParameterGrid(
            {
                "is_stochastic": [False],
                "alpha": [0, 1],
                "distance": [
                    dict(cut_off=10),
                    dict(cut_off=100),
                    dict(cut_off=np.inf),
                ],
            }
        )

        for setting in parameter_grid:
            _dkwg = setting.pop("distance")

            gh = GeometricHarmonicsInterpolator(
                GaussianKernel(epsilon=0.01, distance=_dkwg),
                n_eigenpairs=3,
                **setting,
            ).fit(data, values)

            # larger number of samples than original data
            oos_data = rng.normal(size=(200, 5))

            gh.predict(oos_data)
            gh.gradient(oos_data)

            oos_data = rng.normal(size=(100, 5))  # same size as original data
            gh.predict(oos_data)
            gh.gradient(oos_data)

            oos_data = rng.normal(size=(50, 5))  # less than original data
            gh.predict(oos_data)
            gh.gradient(oos_data)

            oos_data = rng.normal(size=(1, 5))  # single sample
            gh.predict(oos_data)
            gh.gradient(oos_data)

    @unittest.skip(reason="functionality and testing not finished")
    def test_multiscale(self):
        x_lims_train = (0, 10)
        y_lims_train = (0, 10)

        x_lims_test = (-2, 12)
        y_lims_test = (-2, 12)

        nr_sample_x_train = 30
        nr_sample_y_train = 30

        nr_sample_x_test = 200
        nr_sample_y_test = 200

        xx, yy = np.meshgrid(
            np.linspace(*x_lims_train, nr_sample_x_train),
            np.linspace(*y_lims_train, nr_sample_y_train),
        )
        zz = np.sin(yy) * np.sin(xx)

        X_train = np.vstack(
            [xx.reshape(np.prod(xx.shape)), yy.reshape(np.prod(yy.shape))]
        ).T
        y_train = zz.reshape(np.prod(zz.shape))

        xx_oos, yy_oos = np.meshgrid(
            np.linspace(*x_lims_test, nr_sample_x_test),
            np.linspace(*y_lims_test, nr_sample_y_test),
        )
        zz_oos = np.sin(yy_oos) * np.sin(xx_oos)

        X_oos = np.vstack(
            [
                xx_oos.reshape(np.prod(xx_oos.shape)),
                yy_oos.reshape(np.prod(yy_oos.shape)),
            ]
        ).T
        y_test = zz_oos.reshape(np.prod(zz_oos.shape))

        gh_single_interp = GeometricHarmonicsInterpolator(
            epsilon=13.0,
            n_eigenpairs=130,
            alpha=0,
            is_stochastic=False
            # condition=1.0,
            # admissible_error=1,
            # initial_scale=5,
        ).fit(X_train, y_train)

        gh_multi_interp = MultiScaleGeometricHarmonicsInterpolator(
            initial_scale=50, n_eigenpairs=11, condition=50, admissible_error=0.4
        ).fit(X_train, y_train)

        print("-----------------")
        print("Residuum (train error):")
        score_single_train = gh_single_interp.score(X_train, y_train)
        score_multi_train = gh_multi_interp.score(X_train, y_train)
        print(f"gh single = {score_single_train}")
        print(f"gh multi = {score_multi_train}")

        print("---")
        print("Test error:")
        score_single_test = gh_single_interp.score(X_oos, y_test)
        score_multi_test = gh_multi_interp.score(X_oos, y_test)
        print(f"gh single = {score_single_test}")
        print(f"gh multi = {score_multi_test}")
        print("----------------- \n")

        #################################################################################
        #################################################################################
        #################################################################################
        # TRAIN DATA
        f, ax = plt.subplots(2, 3, sharex=True, sharey=True)

        cur_row = ax[0]

        cur_row[0].contourf(xx, yy, zz)
        vlim = (np.min(zz), np.max(zz))
        cur_row[0].plot(xx, yy, ".", c="k")
        cur_row[0].set_title("Original")

        # plt.figure("Single-scale geometric harmonics")
        cur_row[1].plot(xx, yy, ".", c="k")

        cur_row[1].contourf(
            xx,
            yy,
            gh_single_interp.predict(X_train).reshape(
                nr_sample_x_train, nr_sample_y_train
            ),
            vmin=vlim[0],
            vmax=vlim[1],
        )

        cur_row[1].set_title("Single geometric harmonics")

        cur_row[2].plot(xx, yy, ".", c="k")

        cur_row[2].contourf(
            xx,
            yy,
            gh_multi_interp(X_train).reshape(nr_sample_x_train, nr_sample_y_train),
            vmin=vlim[0],
            vmax=vlim[1],
        )
        cur_row[2].set_title("Multi-scale geometric")

        cur_row = ax[1]

        abs_diff_single_train = np.abs(
            zz
            - gh_single_interp.predict(X_train).reshape(
                nr_sample_x_train, nr_sample_y_train
            )
        )
        abs_diff_multi_train = np.abs(
            zz - gh_multi_interp(X_train).reshape(nr_sample_x_train, nr_sample_y_train)
        )

        vmin = np.min([abs_diff_single_train.min(), abs_diff_multi_train.min()])
        vmax = np.max([abs_diff_single_train.max(), abs_diff_multi_train.max()])

        cur_row[1].set_title("abs difference single scale")
        cnf = cur_row[1].contourf(
            xx, yy, abs_diff_single_train, cmap="Reds", vmin=vmin, vmax=vmax
        )
        f.colorbar(cnf)
        cur_row[1].plot(xx, yy, ".", c="k")

        cur_row[2].set_title("abs difference multi scale")
        cnf = cur_row[2].contourf(
            xx, yy, abs_diff_multi_train, cmap="Reds", vmin=vmin, vmax=vmax
        )
        f.colorbar(cnf)

        cur_row[2].plot(xx, yy, ".", c="k")

        #################################################################################
        #################################################################################
        #################################################################################
        # OOS DATA
        f, ax = plt.subplots(2, 3, sharex=True, sharey=True)

        cur_row = ax[0]

        cur_row[0].contourf(xx_oos, yy_oos, zz_oos)
        vlim = (np.min(zz_oos), np.max(zz_oos))
        cur_row[0].set_title("Original")

        cur_row[1].set_title("Single geometric harmonics")
        cur_row[1].contourf(
            xx_oos,
            yy_oos,
            gh_single_interp.predict(X_oos).reshape(nr_sample_x_test, nr_sample_y_test),
            vmin=vlim[0],
            vmax=vlim[1],
        )

        cur_row[2].set_title("Multi scale geometric harmonics")
        cur_row[2].contourf(
            xx_oos,
            yy_oos,
            gh_multi_interp(X_oos).reshape(nr_sample_x_test, nr_sample_y_test),
            vmin=vlim[0],
            vmax=vlim[1],
        )

        cur_row = ax[1]

        abs_diff_single_train = np.abs(
            zz_oos
            - gh_single_interp.predict(X_oos).reshape(
                nr_sample_x_test, nr_sample_y_test
            )
        )
        abs_diff_multi_train = np.abs(
            zz_oos - gh_multi_interp(X_oos).reshape(nr_sample_x_test, nr_sample_y_test)
        )

        vmin = np.min([abs_diff_single_train.min(), abs_diff_multi_train.min()])
        vmax = np.max([abs_diff_single_train.max(), abs_diff_multi_train.max()])

        cur_row[1].set_title("abs difference single scale")
        cnf = cur_row[1].contourf(
            xx_oos, yy_oos, abs_diff_single_train, cmap="Reds", vmin=vmin, vmax=vmax
        )
        f.colorbar(cnf)

        cur_row[2].set_title("abs difference multi scale")
        cnf = cur_row[2].contourf(
            xx_oos, yy_oos, abs_diff_multi_train, cmap="Reds", vmin=vmin, vmax=vmax
        )
        f.colorbar(cnf)

        plt.show()

    @unittest.skipIf(not IS_IMPORTED_RDIST, "rdist is not available")
    def test_different_backends(self):
        data, _ = make_swiss_roll(1000, random_state=1)

        eps_interp = 100  # in this case much larger compared to 1.25 for dim. reduction
        n_eigenpairs = 50

        setting = {
            "kernel": GaussianKernel(eps_interp),
            "n_eigenpairs": n_eigenpairs,
            "dist_kwargs": dict(cut_off=1e100, backend="scipy.kdtree"),
        }
        setting2 = {
            "kernel": GaussianKernel(eps_interp),
            "n_eigenpairs": n_eigenpairs,
            "dist_kwargs": dict(cut_off=1e100, backend="scipy.kdtree"),
        }

        actual_phi_rdist = GeometricHarmonicsInterpolator(**setting).fit(
            data, data[:, 0]
        )
        actual_phi_kdtree = GeometricHarmonicsInterpolator(**setting2).fit(
            data, data[:, 0]
        )

        nptest.assert_allclose(
            actual_phi_rdist.eigenvalues_,
            actual_phi_kdtree.eigenvalues_,
            atol=9e-14,
            rtol=1e-14,
        )

        assert_equal_eigenvectors(
            actual_phi_rdist.eigenvectors_, actual_phi_kdtree.eigenvectors_
        )

        result_rdist = actual_phi_rdist.predict(data)
        result_kdtree = actual_phi_kdtree.predict(data)

        # TODO: it is not clear why relative large tolerances are required... (also see
        #  further below).
        nptest.assert_allclose(result_rdist, result_kdtree, atol=1e-12, rtol=1e-13)

    # def test_gradient(self):
    #     xx, yy = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 100, 20))
    #     zz = xx + np.sin(yy)
    #
    #     data_points = np.vstack(
    #         [xx.reshape(np.product(xx.shape)), yy.reshape(np.product(yy.shape))]
    #     ).T
    #     target_values = zz.reshape(np.product(zz.shape))
    #
    #     gh_interp = GeometricHarmonicsInterpolator(epsilon=100, n_eigenpairs=50)
    #     gh_interp = gh_interp.fit(data_points, target_values)
    #     score = gh_interp.score(data_points, target_values)
    #     print(f"score={score}")
    #
    #     plt.figure()
    #     plt.contourf(xx, yy, zz)
    #     plt.figure()
    #     plt.contourf(xx, yy, gh_interp(data_points).reshape(20, 20))
    #
    #     grad_x = xx
    #     grad_y = np.cos(yy)
    #     grad = np.vstack(
    #         [
    #             grad_x.reshape(np.product(grad_x.shape)),
    #             grad_y.reshape(np.product(grad_y.shape)),
    #         ]
    #     ).T
    #
    #     print(np.linalg.norm(gh_interp.gradient(data_points) - grad))

    def test_stochastic_kernel(self):
        # Currently, only check if it runs through (with is_stochastic=True

        data = np.linspace(0, 2 * np.pi, 40)[:, np.newaxis]
        values = np.sin(data)

        gh_interp = GeometricHarmonicsInterpolator(
            kernel=GaussianKernel(epsilon=0.5, distance=dict(cut_off=np.inf)),
            n_eigenpairs=30,
            is_stochastic=True,
            alpha=0,
            symmetrize_kernel=False,
        ).fit(data, values)

        score = gh_interp.score(data, values)

        # NOTE: if is_stochastic=True and alpha =0, the GH is not able to reproduce the
        # sin curve exactly.

        # To identify changes in the implementation, this checks against a reference
        # solution
        print(score)

        # Somehow, the remote computer produces a slightly different result...
        reference = 0.04836717878208042
        self.assertLessEqual(score, reference)

    def test_renormalization_kernel(self, plot=False):
        # Currently, only check if it runs through (with is_stochastic=True)

        data = np.linspace(0, 2 * np.pi, 100)[:, np.newaxis]
        values = np.sin(data)

        gh_interp = GeometricHarmonicsInterpolator(
            GaussianKernel(epsilon=2, distance=dict(cut_off=np.inf)),
            n_eigenpairs=30,
            is_stochastic=True,
            alpha=1,
            symmetrize_kernel=True,
        ).fit(data, values)

        data_interp = np.linspace(0, 2 * np.pi, 100)[:, np.newaxis]

        predicted_partial = gh_interp.predict(data[:10, :])
        predicted_all = gh_interp.predict(data_interp)
        score = gh_interp.score(data, values)

        # NOTE: if is_stochastic=True and alpha=1 the GH is able to reproduce the
        # sin curve more accurately.
        # self.assertEqual(score, 0.0005576927798107333)

        if plot:
            # To identify changes in the implementation, this checks against a reference
            # solution
            print(score)
            plt.plot(data, values, "-*")
            plt.plot(data_interp, predicted_all, "-*")
            plt.plot(data[:10, :], predicted_partial, "-*")
            plt.show()


class GeometricHarmonicsLegacyTest(unittest.TestCase):
    # We want to produce exactly the same results as the forked DMAP repository. These
    # are test to make sure this is the case.

    def setUp(self):
        self.data, _ = make_swiss_roll(n_samples=1000, noise=0, random_state=1)

        dim_red_eps = 1.25

        dmap = DiffusionMaps(
            GaussianKernel(epsilon=dim_red_eps, distance=dict(cut_off=1e100)),
            n_eigenpairs=6,
        ).fit(self.data)

        self.phi_all = dmap.eigenvectors_[:, [1, 5]]  # column wise like X_all

        train_idx_stop = int(self.data.shape[0] * 2 / 3)
        self.data_train = self.data[:train_idx_stop, :]
        self.data_test = self.data[train_idx_stop:, :]
        self.phi_train = self.phi_all[:train_idx_stop, :]
        self.phi_test = self.phi_all[train_idx_stop:, :]

    def test_method_example1(self):
        # Example from method_examples/diffusion_maps/geometric_harmonics --
        # out-of-samples case.

        eps_interp = 100  # in this case much larger compared to 1.25 for dim. reduction
        n_eigenpairs = 50

        # Because the distances were changed (to consistently squared) the
        # interpolation DMAP has to be computed again for the legacy case.
        legacy_dmap_interp = legacy_dmap.SparseDiffusionMaps(
            points=self.data_train,  # use part of data
            epsilon=eps_interp,  # eps. for interpolation
            num_eigenpairs=n_eigenpairs,  # number of basis functions
            cut_off=np.inf,
            normalize_kernel=False,
        )

        setting = {
            "kernel": GaussianKernel(epsilon=eps_interp, distance=dict(cut_off=1e100)),
            "n_eigenpairs": n_eigenpairs,
        }

        actual_phi0 = GeometricHarmonicsInterpolator(**setting).fit(
            self.data_train, self.phi_train[:, 0]
        )
        actual_phi1 = GeometricHarmonicsInterpolator(**setting).fit(
            self.data_train, self.phi_train[:, 1]
        )
        actual_phi2d = GeometricHarmonicsInterpolator(**setting).fit(
            self.data_train, self.phi_train
        )

        expected_phi0 = legacy_dmap.GeometricHarmonicsInterpolator(
            points=self.data_train,
            values=self.phi_train[:, 0],
            # legacy code requires to set epsilon even in the case when
            # "diffusion_maps" is handled
            epsilon=-1,
            diffusion_maps=legacy_dmap_interp,
        )

        expected_phi1 = legacy_dmap.GeometricHarmonicsInterpolator(
            points=self.data_train,
            values=self.phi_train[:, 1],
            epsilon=-1,
            diffusion_maps=legacy_dmap_interp,
        )

        # The reason why there is a relatively large atol is because we changed the way
        # to compute an internal parameter in the GeometricHarmonicsInterpolator (from
        # n**3 to n**2) -- this introduced some numerical differences.
        nptest.assert_allclose(
            actual_phi0.predict(self.data),
            expected_phi0(self.data),
            rtol=1e-10,
            atol=1e-14,
        )
        nptest.assert_allclose(
            actual_phi1.predict(self.data),
            expected_phi1(self.data),
            rtol=1e-10,
            atol=1e-14,
        )

        # only phi_test because the computation is quite expensive
        nptest.assert_allclose(
            actual_phi0.gradient(self.data_test),
            expected_phi0.gradient(self.data_test),
            rtol=1e-13,
            atol=1e-14,
        )
        nptest.assert_allclose(
            actual_phi1.gradient(self.data_test),
            expected_phi1.gradient(self.data_test),
            rtol=1e-13,
            atol=1e-14,
        )

        # nD case
        nptest.assert_allclose(
            actual_phi2d.predict(self.data)[:, 0],
            expected_phi0(self.data),
            rtol=1e-11,
            atol=1e-12,
        )
        nptest.assert_allclose(
            actual_phi2d.predict(self.data)[:, 1],
            expected_phi1(self.data),
            rtol=1e-11,
            atol=1e-12,
        )

        nptest.assert_allclose(
            actual_phi2d.gradient(self.data_test, vcol=0),
            expected_phi0.gradient(self.data_test),
            rtol=1e-13,
            atol=1e-14,
        )
        nptest.assert_allclose(
            actual_phi2d.gradient(self.data_test, vcol=1),
            expected_phi1.gradient(self.data_test),
            rtol=1e-13,
            atol=1e-14,
        )

    def test_method_example2(self):
        # Example from method_examples/diffusion_maps/geometric_harmonics -- inverse case.

        eps_interp = 0.0005
        # in this case much smaller compared to 1.25 for dim. reduction or 100 for the
        # forward map
        n_eigenpairs = 100

        legacy_dmap_interp = legacy_dmap.SparseDiffusionMaps(
            points=self.phi_train,  # (!!) we use phi now
            epsilon=eps_interp,  # new eps. for interpolation
            num_eigenpairs=n_eigenpairs,
            cut_off=1e100,
            normalize_kernel=False,
        )

        setting = {
            "kernel": GaussianKernel(epsilon=eps_interp, distance=dict(cut_off=1e100)),
            "n_eigenpairs": n_eigenpairs,
            "is_stochastic": False,
        }

        actual_x0 = GeometricHarmonicsInterpolator(**setting).fit(
            self.phi_train, self.data_train[:, 0]
        )
        actual_x1 = GeometricHarmonicsInterpolator(**setting).fit(
            self.phi_train, self.data_train[:, 1]
        )
        actual_x2 = GeometricHarmonicsInterpolator(**setting).fit(
            self.phi_train, self.data_train[:, 2]
        )

        # interpolate both values at once (new feature)
        actual_2values = GeometricHarmonicsInterpolator(**setting).fit(
            self.phi_train, self.data_train
        )

        # compare to legacy GH
        expected_x0 = legacy_dmap.GeometricHarmonicsInterpolator(
            points=self.phi_train,
            values=self.data_train[:, 0],
            epsilon=-1,
            diffusion_maps=legacy_dmap_interp,
        )

        expected_x1 = legacy_dmap.GeometricHarmonicsInterpolator(
            points=self.phi_train,
            values=self.data_train[:, 1],
            epsilon=-1,
            diffusion_maps=legacy_dmap_interp,
        )

        expected_x2 = legacy_dmap.GeometricHarmonicsInterpolator(
            points=self.phi_train,
            values=self.data_train[:, 2],
            epsilon=-1,
            diffusion_maps=legacy_dmap_interp,
        )

        nptest.assert_allclose(
            actual_x0.predict(self.phi_all),
            expected_x0(self.phi_all),
            rtol=1e-4,
            atol=1e-6,
        )
        nptest.assert_allclose(
            actual_x1.predict(self.phi_all),
            expected_x1(self.phi_all),
            rtol=1e-4,
            atol=1e-6,
        )
        nptest.assert_allclose(
            actual_x2.predict(self.phi_all),
            expected_x2(self.phi_all),
            rtol=1e-4,
            atol=1e-6,
        )

        # only phi_test because the computation is quite expensive
        nptest.assert_allclose(
            actual_x0.gradient(self.phi_test),
            expected_x0.gradient(self.phi_test),
            rtol=1e-13,
            atol=1e-14,
        )
        nptest.assert_allclose(
            actual_x1.gradient(self.phi_test),
            expected_x1.gradient(self.phi_test),
            rtol=1e-13,
            atol=1e-14,
        )
        nptest.assert_allclose(
            actual_x2.gradient(self.phi_test),
            expected_x2.gradient(self.phi_test),
            rtol=1e-13,
            atol=1e-14,
        )

        nptest.assert_allclose(
            actual_2values.predict(self.phi_all)[:, 0],
            expected_x0(self.phi_all),
            rtol=1e-5,
            atol=1e-7,
        )
        nptest.assert_allclose(
            actual_2values.predict(self.phi_all)[:, 1],
            expected_x1(self.phi_all),
            rtol=1e-5,
            atol=1e-7,
        )
        nptest.assert_allclose(
            actual_2values.predict(self.phi_all)[:, 2],
            expected_x2(self.phi_all),
            rtol=1e-5,
            atol=1e-7,
        )

        nptest.assert_allclose(
            actual_2values.gradient(self.phi_test, vcol=0),
            expected_x0.gradient(self.phi_test),
            rtol=1e-13,
            atol=1e-14,
        )
        nptest.assert_allclose(
            actual_2values.gradient(self.phi_test, vcol=1),
            expected_x1.gradient(self.phi_test),
            rtol=1e-13,
            atol=1e-14,
        )
        nptest.assert_allclose(
            actual_2values.gradient(self.phi_test, vcol=2),
            expected_x2.gradient(self.phi_test),
            rtol=1e-13,
            atol=1e-14,
        )

    def test_same_underlying_kernel(self):
        # Actually not a legacy test, but uses the setup.

        from datafold.pcfold.distance import BruteForceDist

        eps_interp = 0.0005

        # GH must be trained before to set kernel
        gh = GeometricHarmonicsInterpolator(
            kernel=GaussianKernel(eps_interp),
            n_eigenpairs=1,
            alpha=0.5,
            is_stochastic=True,
            symmetrize_kernel=True,
        ).fit(self.data_train, self.phi_train)

        self.assertTrue(gh._dmap_kernel.is_stochastic)
        self.assertEqual(gh._dmap_kernel.alpha, 0.5)
        self.assertTrue(gh._dmap_kernel.is_symmetric)
        self.assertIsInstance(gh._dmap_kernel.distance, BruteForceDist)


class LaplacianPyramidsTest(unittest.TestCase):
    def setUpSyntheticFernandez(self) -> None:
        rng = np.random.default_rng(2)

        self.X_fern = np.linspace(0, 10 * np.pi, 2000)[:, np.newaxis]
        self.X_fern_test = np.sort(rng.uniform(0, 10 * np.pi, 500))[:, np.newaxis]

        delta = 0.05

        # EVALUATE TRAIN DATA
        indicator_range2 = np.logical_and(
            self.X_fern > 10 * np.pi / 3, self.X_fern <= 10 * np.pi
        )
        indicator_range3 = np.logical_and(
            self.X_fern > 2 * 10 * np.pi / 2, self.X_fern <= 10 * np.pi
        )

        noise = rng.uniform(low=-delta, high=delta, size=self.X_fern.shape[0])
        noise = noise[:, np.newaxis]

        self.y_fern = (
            np.sin(self.X_fern)
            + 0.5 * np.sin(3 * self.X_fern) * indicator_range2
            + 0.25 * np.sin(9 * self.X_fern) * indicator_range3
            + noise
        )

        # EVALUATE TEST DATA
        indicator_range2 = np.logical_and(
            self.X_fern_test > 10 * np.pi / 3, self.X_fern_test <= 10 * np.pi
        )
        indicator_range3 = np.logical_and(
            self.X_fern_test > 2 * 10 * np.pi / 2, self.X_fern_test <= 10 * np.pi
        )

        noise = rng.uniform(low=-delta, high=delta, size=self.X_fern_test.shape[0])
        noise = noise[:, np.newaxis]

        self.y_fern_test = (
            np.sin(self.X_fern_test)
            + 0.5 * np.sin(3 * self.X_fern_test) * indicator_range2
            + 0.25 * np.sin(9 * self.X_fern_test) * indicator_range3
            + noise
        )

    def setUpSyntheticFernandezAddFunc(self):
        # add simple function on same X
        # primarily to test multiple target functions

        self.y_fern2 = np.sin(self.X_fern)
        self.y_fern2_test = np.sin(self.X_fern_test)

    def setUpRabin(self):
        rng = np.random.default_rng(1)

        self.X_rabin = rng.uniform(0, np.pi / 4, 3500)[:, np.newaxis]
        self.X_rabin = np.sort(self.X_rabin, axis=0)

        self.y_rabin = np.sin(1 / (self.X_rabin + 0.01))

        self.X_rabin_test = rng.uniform(0, np.pi / 4, 165)[:, np.newaxis]
        self.X_rabin_test = np.sort(self.X_rabin_test, axis=0)
        self.y_rabin_test = np.sin(1 / (self.X_rabin_test + 0.01))

    def setUp(self) -> None:
        self.setUpSyntheticFernandez()
        self.setUpSyntheticFernandezAddFunc()
        self.setUpRabin()

    def _plot(self, lp, train_X, train_y, train_y_eval, test_X, test_y, test_y_eval):
        plt.figure()
        plt.plot(train_X, train_y, ".", label="train")
        plt.plot(train_X, train_y_eval, ".", label="train_eval")
        plt.plot(test_X, test_y, "o", label="test")
        plt.plot(test_X, test_y_eval, "-+", label="test_eval")
        plt.legend()

        if lp is not None:
            lp.plot_eps_vs_residual()

    def test_valid_sklearn_estimator(self):
        for estimator, check in check_estimator(
            LaplacianPyramidsInterpolator(initial_epsilon=100, auto_adaptive=True),
            generate_only=True,
        ):
            try:
                check(estimator)
            except Exception as e:
                print(check)
                print(estimator)
                raise e

    def test_synthetic_example_rabin(self, plot=False):
        # TODO: currently, there is a robustness issue. For very small scales,
        #  some cdist row-sums get zero -- the recisprocal therefore inf. Therefore,
        #  the residual_tol is currently larger than in the paper (section 3.2.1.)
        lp = LaplacianPyramidsInterpolator(
            initial_epsilon=0.5, mu=2, residual_tol=1e-10
        )
        lp = lp.fit(self.X_rabin, self.y_rabin)

        train_score = lp.score(self.X_rabin, self.y_rabin)
        test_score = lp.score(self.X_rabin_test, self.y_rabin_test)
        self.assertEqual(train_score, 1.0)
        self.assertLess(0.99999, test_score)

        if plot:
            train_eval = lp.predict(self.X_rabin)
            test_eval = lp.predict(self.X_rabin_test)

            self._plot(
                lp,
                self.X_rabin,
                self.y_rabin,
                train_eval,
                self.X_rabin_test,
                self.y_rabin_test,
                test_eval,
            )
            plt.show()

    def test_synthetic_example_rabin_adaptive(self, plot=False):
        # TODO: currently, there is a robustness issue. For very small scales,
        #  some cdist row-sums get zero -- the recisprocal therefore inf. Therefore,
        #  the residual_tol is currently larger than in the paper (section 3.2.1.)
        lp = LaplacianPyramidsInterpolator(
            initial_epsilon=0.5, mu=2, residual_tol=None, auto_adaptive=True
        )
        lp = lp.fit(self.X_rabin, self.y_rabin)

        train_score = lp.score(self.X_rabin, self.y_rabin)
        test_score = lp.score(self.X_rabin_test, self.y_rabin_test)

        self.assertLessEqual(0.98779668, train_score)
        self.assertLessEqual(0.99182011, test_score)

        if plot:
            train_eval = lp.predict(self.X_rabin)
            test_eval = lp.predict(self.X_rabin_test)

            self._plot(
                lp,
                self.X_rabin,
                self.y_rabin,
                train_eval,
                self.X_rabin_test,
                self.y_rabin_test,
                test_eval,
            )

            plt.show()

    def test_synthetic_example_fernandez(self, plot=False):
        lp = LaplacianPyramidsInterpolator(
            initial_epsilon=10 * np.pi,
            mu=2,
            auto_adaptive=True,
        )
        lp = lp.fit(self.X_fern, self.y_fern)

        train_score = lp.score(self.X_fern, self.y_fern)
        test_score = lp.score(self.X_fern_test, self.y_fern_test)

        self.assertLessEqual(0.9977929, train_score)
        self.assertLessEqual(0.9975171, test_score)

        if plot:
            train_eval = lp.predict(self.X_fern)
            test_eval = lp.predict(self.X_fern_test)

            self._plot(
                lp,
                self.X_fern,
                self.y_fern,
                train_eval,
                self.X_fern_test,
                self.y_fern_test,
                test_eval,
            )
            plt.show()

    def test_synthetic_example_fernandez_residualtol(self, plot=False):
        lp = LaplacianPyramidsInterpolator(
            initial_epsilon=10 * np.pi,
            mu=2,
            residual_tol=1e-1,
            auto_adaptive=False,
        )
        lp = lp.fit(self.X_fern, self.y_fern)

        train_score = lp.score(self.X_fern, self.y_fern)
        test_score = lp.score(self.X_fern_test, self.y_fern_test)

        self.assertLessEqual(0.9999998, train_score)
        self.assertLessEqual(0.9971422, test_score)

        if plot:
            train_eval = lp.predict(self.X_fern)
            test_eval = lp.predict(self.X_fern_test)
            self._plot(
                lp,
                self.X_fern,
                self.y_fern,
                train_eval,
                self.X_fern_test,
                self.y_fern_test,
                test_eval,
            )
            plt.show()

    def test_synthetic_example_fernandez_multifunc(self, plot=False):
        lp = LaplacianPyramidsInterpolator(
            initial_epsilon=10 * np.pi,
            mu=2,
            residual_tol=1e-1,
            auto_adaptive=True,
        )

        y_train = np.hstack([self.y_fern, self.y_fern2])
        y_test = np.hstack([self.y_fern_test, self.y_fern2_test])

        lp = lp.fit(self.X_fern, y_train)

        train_score = lp.score(self.X_fern, y_train)
        test_score = lp.score(self.X_fern_test, y_test)

        self.assertLessEqual(0.9986398, train_score)
        self.assertLessEqual(0.9984894, test_score)

        if plot:
            train_eval = lp.predict(self.X_fern)
            test_eval = lp.predict(self.X_fern_test)

            self._plot(
                None,
                self.X_fern,
                self.y_fern,
                train_eval[:, 0],
                self.X_fern_test,
                self.y_fern_test,
                test_eval[:, 0],
            )

            self._plot(
                lp,
                self.X_fern,
                self.y_fern2,
                train_eval[:, 1],
                self.X_fern_test,
                self.y_fern2_test,
                test_eval[:, 1],
            )

            plt.show()


if __name__ == "__main__":
    t = LaplacianPyramidsTest()
    t.setUp()
    t.test_synthetic_example_fernandez_multifunc(plot=False)
    # t.test_synthetic_example_fernandez_residualtol(plot=False)
    # t.test_synthetic_example_rabin(plot=False)
    # t.test_synthetic_example_rabin_adaptive(plot=False)
