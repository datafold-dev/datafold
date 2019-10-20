"""62-make-diffusionmaps-and-geometricharmonicsinterpolator-compatible-with-scikit-learn-api
Unit test for the Geometric Harmonics module.
"""

import unittest

import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import ParameterGrid, train_test_split

from datafold.dynfold import GeometricHarmonicsFunctionBasis, GeometricHarmonicsInterpolator
from datafold.dynfold.kernel import DmapKernelFixed
from datafold.dynfold.tests.helper import *


def plot_scatter(points: np.ndarray, values: np.ndarray, **kwargs) -> None:
    title = kwargs.pop('title', None)
    if title:
        plt.title(title)
    plt.scatter(points[:, 0], points[:, 1], c=values, marker='o', rasterized=True, s=2.5, **kwargs)
    cb = plt.colorbar()
    cb.set_clim([np.min(values), np.max(values)])
    cb.set_ticks(np.linspace(np.min(values), np.max(values), 5))
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.gca().set_aspect('equal')


def f(points: np.ndarray) -> np.ndarray:
    """Function to interpolate.

    """
    # return np.ones(points.shape[0])
    # return np.arange(points.shape[0])
    return np.sin(np.linalg.norm(points, axis=-1))


def show_info(title: str, values: np.ndarray) -> None:
    """Log relevant information.

    """
    logging.info('{}: mean = {:g}, std. = {:g}, max. abs. = {:g}'
                 .format(title, np.mean(values), np.std(values),
                         np.max(np.abs(values))))


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

    def test_geometric_harmonics_interpolator(self):
        logging.basicConfig(level=logging.DEBUG)

        eps = 1e-1

        ghi = GeometricHarmonicsInterpolator(epsilon=eps, num_eigenpairs=self.num_points-3, cut_off=1e1 * eps)
        ghi = ghi.fit(self.points, self.values)

        points = make_points(100, -4, -4, 4, 4)

        values = ghi(points)

        residual = values - f(points)
        self.assertLess(np.max(np.abs(residual)), 7.5e-2)

        show_info('Original function', f(points))
        show_info('Sampled points', self.values)
        show_info('Reconstructed function', values)
        show_info('Residual', residual)

        # plt.subplot(2, 2, 1)
        # plot(points, f(points), title='Original function')
        #
        # plt.subplot(2, 2, 2)
        # plot(self.points, self.values, title='Sampled function')
        #
        # plt.subplot(2, 2, 4)
        # plot(points, values, title='Reconstructed function')
        #
        # plt.subplot(2, 2, 3)
        # plot(points, residual, title='Residual', cmap='RdBu_r')
        #
        # plt.tight_layout()
        # plt.show()

    def test_eigenfunctions(self):
        logging.basicConfig(level=logging.DEBUG)

        eps = 1e1
        cut_off = 1e1 * eps
        num_eigenpairs = 3

        points = make_strip(0, 0, 1, 1e-1, 3000)

        dm = DiffusionMaps(epsilon=eps, num_eigenpairs=num_eigenpairs, cut_off=1E100).fit(points)
        ev = dm.eigenvectors_

        # plt.subplot(1, 2, 1)
        # plt.scatter(points[:, 0], points[:, 1], c=ev[1, :], cmap='RdBu_r')
        # plt.subplot(1, 2, 2)
        # plt.scatter(points[:, 0], points[:, 1], c=ev[2, :], cmap='RdBu_r')
        # plt.show()

        setting = {"epsilon": eps, "num_eigenpairs": num_eigenpairs, "cut_off": cut_off, "is_stochastic": False}

        ev1 = GeometricHarmonicsInterpolator(**setting).fit(points, ev[1, :])
        ev2 = GeometricHarmonicsInterpolator(**setting).fit(points, ev[2, :])

        # new_points = make_points(50, 0, 0, 1, 1e-1)
        # ev1i = ev1(new_points)
        # ev2i = ev2(new_points)
        # plt.subplot(1, 2, 1)
        # plt.scatter(new_points[:, 0], new_points[:, 1], c=ev1i,
        #             cmap='RdBu_r')
        # plt.subplot(1, 2, 2)
        # plt.scatter(new_points[:, 0], new_points[:, 1], c=ev2i,
        #             cmap='RdBu_r')
        # plt.show()

        rel_err1 = (np.linalg.norm(ev[1, :] - ev1(points), np.inf) /
                    np.linalg.norm(ev[1, :], np.inf))
        self.assertAlmostEqual(rel_err1, 0, places=1)

        rel_err2 = (np.linalg.norm(ev[2, :] - ev2(points), np.inf) /
                    np.linalg.norm(ev[2, :], np.inf))
        self.assertAlmostEqual(rel_err2, 0, places=1)

    def test_dense_sparse(self):
        data, _ = make_swiss_roll(n_samples=1000, noise=0, random_state=1)
        dim_red_eps = 1.25

        dense_setting = {"epsilon": dim_red_eps, "num_eigenpairs": 6, "cut_off": np.inf, "is_stochastic": False}
        sparse_setting = {"epsilon": dim_red_eps, "num_eigenpairs": 6, "cut_off": 1E100, "is_stochastic": False}

        dmap_dense = DiffusionMaps(**dense_setting).fit(data)
        values = dmap_dense.eigenvectors_[1, :]

        dmap_sparse = DiffusionMaps(**sparse_setting).fit(data)

        # Check if any error occurs (functional test) and whether the provided DMAP is changed in any way.
        gh_dense = GeometricHarmonicsInterpolator(**dense_setting).fit(data, values)
        gh_sparse = GeometricHarmonicsInterpolator(**sparse_setting).fit(data, values)

        self.assertEqual(gh_dense.kernel_, dmap_dense.kernel_)
        self.assertEqual(gh_sparse.kernel_, dmap_sparse.kernel_)

        # The parameters are set equal to the previously generated DMAP, therefore both have to be equal.
        gh_dense_cmp = GeometricHarmonicsInterpolator(**dense_setting).fit(data, values)
        gh_sparse_cmp = GeometricHarmonicsInterpolator(**sparse_setting).fit(data, values)

        self.assertEqual(gh_dense_cmp.kernel_, dmap_dense.kernel_)
        self.assertEqual(gh_sparse_cmp.kernel_, dmap_sparse.kernel_)

        # Check the the correct format is set
        self.assertTrue(isinstance(gh_dense_cmp.kernel_matrix_, np.ndarray))
        self.assertTrue(isinstance(gh_sparse_cmp.kernel_matrix_, csr_matrix))

        gh_dense_cmp(data)
        gh_sparse_cmp(data)

        # Check if sparse (without cutoff) and dense case give close results
        nptest.assert_allclose(gh_sparse_cmp(data), gh_dense_cmp(data), rtol=1E-14, atol=1E-15)
        nptest.assert_allclose(gh_sparse_cmp.gradient(data), gh_dense_cmp.gradient(data), rtol=1E-14, atol=1E-15)

    def test_variable_number_of_points(self):

        # Simply check if something fails

        np.random.seed(1)

        data = np.random.randn(100, 5)
        values = np.random.randn(100)

        # TODO: not sure how to proceed with "is_stochastic": True and cdist (the normalization may fail)  #65
        parameter_grid = ParameterGrid({"is_stochastic": [False], "alpha": [0, 1], "cut_off": [10, 100, np.inf]})

        for setting in parameter_grid:
            gh = GeometricHarmonicsInterpolator(epsilon=0.01, num_eigenpairs=3, **setting).fit(data, values)

            oos_data = np.random.randn(200, 5)  # larger number of samples than original data

            gh(oos_data)
            gh.gradient(oos_data)

            oos_data = np.random.randn(100, 5)  # same size as original data
            gh(oos_data)
            gh.gradient(oos_data)

            oos_data = np.random.randn(50, 5)   # less than original data
            gh(oos_data)
            gh.gradient(oos_data)

            oos_data = np.random.randn(1, 5)  # single sample
            gh(oos_data)
            gh.gradient(oos_data)

    # def test_gradient(self):
    #     logging.basicConfig(level=logging.DEBUG)
    #
    #     eps = 1e1
    #     cut_off = np.inf
    #     num_eigenpairs = 65
    #
    #     points = make_strip(0, 0, 1, 1, 5000)
    #
    #     dm = DiffusionMaps(points, eps, cut_off=cut_off,
    #                        num_eigenpairs=num_eigenpairs,
    #                        use_cuda=False)
    #     ev = dm.eigenvectors[4, :]
    #
    #     dmaps_opts = {'num_eigenpairs': num_eigenpairs,
    #                   'cut_off': cut_off, 'use_cuda': False}
    #     u = GeometricHarmonicsInterpolator(points, ev, eps,
    #                                        diffusion_maps=dm,
    #                                        diffusion_maps_options=dmaps_opts)
    #
    #     rel_err = (np.linalg.norm(ev - u(points), np.inf) /
    #                np.linalg.norm(ev, np.inf))
    #     self.assertAlmostEqual(rel_err, 0, places=1)
    #
    #     # new_points = make_points(10, 0, 0, 1, 1)
    #     # # ui = u(new_points)
    #     # dui = u.gradient(new_points)
    #
    #     # plt.scatter(points[:, 0], points[:, 1], c=u(points),
    #     #             cmap='RdBu_r')
    #     # plt.colorbar()
    #     # plt.quiver(new_points[:, 0], new_points[:, 1], dui[:, 0], dui[:, 1],
    #     #            units='xy', scale=2.5)
    #     # plt.gca().set_aspect('equal')
    #
    #     # plt.subplot(2, 2, 3)
    #     # plt.scatter(new_points[:, 0], new_points[:, 1], c=dui[:, 0],
    #     #             cmap='RdBu_r')
    #     # plt.gca().set_aspect('equal')
    #     # plt.colorbar()
    #     # plt.subplot(2, 2, 4)
    #     # plt.scatter(new_points[:, 0], new_points[:, 1], c=dui[:, 1],
    #     #             cmap='RdBu_r')
    #     # plt.gca().set_aspect('equal')
    #     # plt.colorbar()
    #     # # plt.tight_layout()
    #     plt.show()

    def test_geometric_harmonics_function_basis(self):
        data, _ = make_swiss_roll(3000, noise=0, random_state=0)
        dmap = DiffusionMaps(epsilon=0.3, num_eigenpairs=50, is_stochastic=False).fit(data)

        actual_interp = GeometricHarmonicsFunctionBasis(epsilon=0.3, num_eigenpairs=50).fit(data)
        # TODO: issue #44
        expected_interp = GeometricHarmonicsInterpolator(epsilon=0.3, num_eigenpairs=50).fit(data, dmap.eigenvectors_.T)

        nptest.assert_array_equal(actual_interp.kernel_matrix_, expected_interp.kernel_matrix_)

        # TODO: it is not quite clear why they are not exactly the same
        #   * when changing sigma=1 in the eigenvalue solver, the relative error improves.
        #   * the tolerance in the eigenvector solver also has an small effect
        #   * the previous assert makes sure that the kernel_matrix is *exactly* the same
        nptest.assert_allclose(actual_interp(data), expected_interp(data), rtol=1E-5, atol=1E-15)

    def test_gradient(self):
        xx, yy = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 100, 20))
        zz = xx + np.sin(yy)

        data_points = np.vstack([xx.reshape(np.product(xx.shape)), yy.reshape(np.product(yy.shape))]).T
        target_values = zz.reshape(np.product(zz.shape))

        gh_interp = GeometricHarmonicsInterpolator(epsilon=100, num_eigenpairs=50)
        gh_interp = gh_interp.fit(data_points, target_values)
        score = gh_interp.score(data_points, target_values)
        print(f"score={score}")

        plt.figure()
        plt.contourf(xx, yy, zz)
        plt.figure()
        plt.contourf(xx, yy, gh_interp(data_points).reshape(20, 20))

        grad_x = xx
        grad_y = np.cos(yy)
        grad = np.vstack([grad_x.reshape(np.product(grad_x.shape)), grad_y.reshape(np.product(grad_y.shape))]).T

        print(np.linalg.norm(gh_interp.gradient(data_points) - grad))


class GeometricHarmonicsLegacyTest(unittest.TestCase):
    # We want to produce exactly the same results as the forked DMAP repository. These are test to make sure this is
    # the case.

    def setUp(self):
        np.random.seed(1)
        self.data, _ = make_swiss_roll(n_samples=1000, noise=0, random_state=1)

        dim_red_eps = 1.25

        dmap = DiffusionMaps(epsilon=dim_red_eps, num_eigenpairs=6, cut_off=1E100).fit(self.data)

        self.phi_all = dmap.eigenvectors_[[1, 5], :].T  # column wise like X_all

        self.data_train, self.data_test, self.phi_train, self.phi_test = \
            train_test_split(self.data, self.phi_all, test_size=1/3, train_size=2/3)

    def test_method_example1(self):
        # Example from method_examples/diffusion_maps/geometric_harmonics -- out-of-samples case.

        eps_interp = 100  # in this case much larger compared to 1.25 for dim. reduction
        num_eigenpairs = 50

        # Because the distances were changed (to consistently squared) the interpolation DMAP has to be computed again
        #  for the legacy case.
        legacy_dmap_interp = \
            legacy_dmap.SparseDiffusionMaps(points=self.data_train,         # use part of data
                                            epsilon=eps_interp,             # eps. for interpolation
                                            num_eigenpairs=num_eigenpairs,  # number of basis functions
                                            cut_off=np.inf,
                                            normalize_kernel=False)

        setting = {"epsilon": eps_interp, "num_eigenpairs": num_eigenpairs, "cut_off": 1E100}

        actual_phi0 = GeometricHarmonicsInterpolator(**setting).fit(self.data_train, self.phi_train[:, 0])
        actual_phi1 = GeometricHarmonicsInterpolator(**setting).fit(self.data_train, self.phi_train[:, 1])
        actual_phi2d = GeometricHarmonicsInterpolator(**setting).fit(self.data_train, self.phi_train)

        expected_phi0 = legacy_dmap.GeometricHarmonicsInterpolator(
            points=self.data_train,
            values=self.phi_train[:, 0],
            epsilon=-1,  # legacy code requires to set epsilon even in the case when "diffusion_maps" is handled
            diffusion_maps=legacy_dmap_interp)

        expected_phi1 = legacy_dmap.GeometricHarmonicsInterpolator(
            points=self.data_train,
            values=self.phi_train[:, 1],
            epsilon=-1,
            diffusion_maps=legacy_dmap_interp)

        # The reason why there is a relatively large atol is because we changed the way to compute an internal parameter
        # in the GeometricHarmonicsInterpolator (from n**3 to n**2) -- this introduced some numerical differences.
        nptest.assert_allclose(actual_phi0(self.data), expected_phi0(self.data), rtol=1E-10, atol=1E-14)
        nptest.assert_allclose(actual_phi1(self.data), expected_phi1(self.data), rtol=1E-10, atol=1E-14)

        # only phi_test because the computation is quite expensive
        nptest.assert_allclose(
            actual_phi0.gradient(self.data_test), expected_phi0.gradient(self.data_test), rtol=1E-13, atol=1E-14)
        nptest.assert_allclose(
            actual_phi1.gradient(self.data_test), expected_phi1.gradient(self.data_test), rtol=1E-13, atol=1E-14)

        # nD case
        nptest.assert_allclose(actual_phi2d(self.data)[:, 0], expected_phi0(self.data), rtol=1E-11, atol=1E-12)
        nptest.assert_allclose(actual_phi2d(self.data)[:, 1], expected_phi1(self.data), rtol=1E-11, atol=1E-12)

        nptest.assert_allclose(actual_phi2d.gradient(self.data_test, vcol=0),
                               expected_phi0.gradient(self.data_test), rtol=1E-13, atol=1E-14)
        nptest.assert_allclose(actual_phi2d.gradient(self.data_test, vcol=1),
                               expected_phi1.gradient(self.data_test), rtol=1E-13, atol=1E-14)

    def test_method_example2(self):
        # Example from method_examples/diffusion_maps/geometric_harmonics -- inverse case.
        np.random.seed(1)

        eps_interp = 0.0005  # in this case much smaller compared to 1.25 for dim. reduction or 100 for the forward map
        num_eigenpairs = 100

        legacy_dmap_interp = legacy_dmap.SparseDiffusionMaps(points=self.phi_train,  # (!!) we use phi now
                                                             epsilon=eps_interp,  # new eps. for interpolation
                                                             num_eigenpairs=num_eigenpairs,
                                                             cut_off=1E100,
                                                             normalize_kernel=False)

        setting = {"epsilon": eps_interp, "num_eigenpairs": num_eigenpairs, "is_stochastic": False, "cut_off": 1E100}

        actual_x0 = GeometricHarmonicsInterpolator(**setting).fit(self.phi_train, self.data_train[:, 0])
        actual_x1 = GeometricHarmonicsInterpolator(**setting).fit(self.phi_train, self.data_train[:, 1])
        actual_x2 = GeometricHarmonicsInterpolator(**setting).fit(self.phi_train, self.data_train[:, 2])

        # interpolate both values at once (new feature)
        actual_2values = GeometricHarmonicsInterpolator(**setting).fit(self.phi_train, self.data_train)

        # compare to legacy GH
        expected_x0 = legacy_dmap.GeometricHarmonicsInterpolator(
            points=self.phi_train,
            values=self.data_train[:, 0],
            epsilon=-1,
            diffusion_maps=legacy_dmap_interp)

        expected_x1 = legacy_dmap.GeometricHarmonicsInterpolator(
            points=self.phi_train,
            values=self.data_train[:, 1],
            epsilon=-1,
            diffusion_maps=legacy_dmap_interp)

        expected_x2 = legacy_dmap.GeometricHarmonicsInterpolator(
            points=self.phi_train,
            values=self.data_train[:, 2],
            epsilon=-1,
            diffusion_maps=legacy_dmap_interp)

        nptest.assert_allclose(actual_x0(self.phi_all), expected_x0(self.phi_all), rtol=1E-4, atol=1E-6)
        nptest.assert_allclose(actual_x1(self.phi_all), expected_x1(self.phi_all), rtol=1E-4, atol=1E-6)
        nptest.assert_allclose(actual_x2(self.phi_all), expected_x2(self.phi_all), rtol=1E-4, atol=1E-6)

        # only phi_test because the computation is quite expensive
        nptest.assert_allclose(actual_x0.gradient(self.phi_test),
                               expected_x0.gradient(self.phi_test), rtol=1E-13, atol=1E-14)
        nptest.assert_allclose(actual_x1.gradient(self.phi_test),
                               expected_x1.gradient(self.phi_test), rtol=1E-13, atol=1E-14)
        nptest.assert_allclose(actual_x2.gradient(self.phi_test),
                               expected_x2.gradient(self.phi_test), rtol=1E-13, atol=1E-14)

        nptest.assert_allclose(actual_2values(self.phi_all)[:, 0], expected_x0(self.phi_all), rtol=1E-5, atol=1E-7)
        nptest.assert_allclose(actual_2values(self.phi_all)[:, 1], expected_x1(self.phi_all), rtol=1E-5, atol=1E-7)
        nptest.assert_allclose(actual_2values(self.phi_all)[:, 2], expected_x2(self.phi_all), rtol=1E-5, atol=1E-7)

        nptest.assert_allclose(actual_2values.gradient(self.phi_test, vcol=0),
                               expected_x0.gradient(self.phi_test), rtol=1E-13, atol=1E-14)
        nptest.assert_allclose(actual_2values.gradient(self.phi_test, vcol=1),
                               expected_x1.gradient(self.phi_test), rtol=1E-13, atol=1E-14)
        nptest.assert_allclose(actual_2values.gradient(self.phi_test, vcol=2),
                               expected_x2.gradient(self.phi_test), rtol=1E-13, atol=1E-14)

    def test_same_underlying_kernel(self):
        # Actually not a legacy test, but uses the set up.

        eps_interp = 0.0005
        actual = DmapKernelFixed(epsilon=eps_interp, is_stochastic=False, alpha=1)

        # diffusion map as argument
        gh = GeometricHarmonicsInterpolator(epsilon=eps_interp, num_eigenpairs=1, is_stochastic=False)

        self.assertEqual(gh.kernel_, actual)

    def test_backend_rdist_kdtree(self):

        eps_interp = 100  # in this case much larger compared to 1.25 for dim. reduction
        num_eigenpairs = 50

        setting = {"epsilon": eps_interp, "num_eigenpairs": num_eigenpairs, "cut_off": 1E100, "dist_backend": "rdist"}  #
        setting2 = {"epsilon": eps_interp, "num_eigenpairs": num_eigenpairs, "cut_off": 1E100, "dist_backend": "scipy.kdtree"}

        actual_phi_rdist = GeometricHarmonicsInterpolator(**setting).fit(self.data_train, self.phi_train[:, 0])
        actual_phi_kdtree = GeometricHarmonicsInterpolator(**setting2).fit(self.data_train, self.phi_train[:, 0])

        nptest.assert_allclose(actual_phi_rdist.eigenvalues_, actual_phi_kdtree.eigenvalues_, atol=1E-14, rtol=1E-14)
        cmp_eigenvectors(actual_phi_rdist.eigenvectors_, actual_phi_kdtree.eigenvectors_)

        result_rdist = actual_phi_rdist(self.data)
        result_kdtree = actual_phi_kdtree(self.data)
        nptest.assert_allclose(result_rdist, result_kdtree, atol=1E-14, rtol=1E-14)


if __name__ == '__main__':

    import os
    verbose = os.getenv('VERBOSE')
    if verbose is not None:
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    else:
        logging.basicConfig(level=logging.ERROR, format='%(message)s')

    # unittest.main()

    t = GeometricHarmonicsTest()
    t.setUp()
    t.test_gradient()

    # t = GeometricHarmonicsLegacyTest()
    # t.setUp()
    # t.test_backend_rdist_kdtree()
