#!/usr/bin/env python

import unittest

import numpy as np
import numpy.testing as nptest

from datafold.pcfold import *


class TestPCMEstimation(unittest.TestCase):
    def test_optimize_parameters_default(self):
        random_state = 1
        rng = np.random.default_rng(random_state)
        n_data = 1000

        result = []
        for n_dim in [1, 2, 3, 4]:
            rdata = rng.uniform(size=(n_data, n_dim))

            pcm = PCManifold(rdata)
            pcm.optimize_parameters(random_state=random_state)

            result.append([pcm.cut_off, pcm.kernel.epsilon])

        result_expected = [
            [1.21461718e-02, 8.00890537e-06],
            [8.94210281e-02, 4.34083863e-04],
            [1.90532972e-01, 1.97076394e-03],
            [2.94194881e-01, 4.69855750e-03],
        ]

        # reference test:needs update when changing behavior
        nptest.assert_almost_equal(result_expected, result, decimal=9)

    def test_optimize_parameters_subsample(self):
        random_state = 1
        rng = np.random.default_rng(random_state)
        n_data = 1000
        n_dim = 2
        rdata = rng.uniform(size=(n_data, n_dim))
        # compute the "best" approximation by not subsampling
        pcm = PCManifold(rdata)
        pcm.optimize_parameters(random_state=random_state, n_subsample=n_data)
        cut_off_best = pcm.cut_off
        epsilon_best = pcm.kernel.epsilon

        result = []
        for n_subsample in [100, 150, 250, 500]:

            pcm = PCManifold(rdata)
            pcm.optimize_parameters(random_state=random_state, n_subsample=n_subsample)

            result.append([pcm.cut_off, pcm.kernel.epsilon])

        # test if the approximated values for epsilon and the cutoff are within a good bound from the best value
        _zero = np.zeros((len(result),))
        nptest.assert_almost_equal(
            np.array(result)[:, 0] - np.array([cut_off_best]), _zero, decimal=3
        )
        nptest.assert_almost_equal(
            np.array(result)[:, 1] - np.array([epsilon_best]), _zero, decimal=5
        )

    def test_optimize_parameters_scaling(self):
        random_state = 1
        np.random.seed(random_state)
        n_data = 100

        result = []
        for n_dim in [1, 2, 3, 4]:
            rdata = np.random.rand(n_data, n_dim)

            pcm = PCManifold(rdata)
            pcm.optimize_parameters(random_state=random_state, result_scaling=2)

            result.append([pcm.cut_off, pcm.kernel.epsilon])
        result_expected = [
            [0.25836304904325946, 0.003623724119579275],
            [0.6454852768168744, 0.022618666941727785],
            [0.9544613110485779, 0.049455088384160635],
            [1.1736192756542319, 0.07477368634377791],
        ]

        # reference test:needs update when changing behavior
        nptest.assert_almost_equal(result_expected, result, decimal=14)

    def test_optimize_parameters_below_tolerance(self):
        random_state = 1
        np.random.seed(random_state)
        n_data = 100

        result = []
        for tol in [1e-6, 1e-8, 1e-10]:
            for n_dim in [1, 2, 3, 4]:
                rdata = np.random.rand(n_data, n_dim)

                pcm = PCManifold(rdata)
                pcm.optimize_parameters(random_state=random_state, tol=tol)

                result.append([np.exp(-pcm.cut_off ** 2 / pcm.kernel.epsilon) - tol])
            result_expected = np.zeros(len(result),)

            nptest.assert_allclose(
                result_expected, np.asarray(result).ravel(), rtol=tol, atol=1e-15
            )

    @staticmethod
    def generate_mushroom(n_points=500):

        NX = int(np.sqrt(n_points))
        space = np.linspace(0, 1, NX)

        x, y = np.meshgrid(space, 2 * space)

        data = np.vstack([x.flatten(), y.flatten()]).T
        data = np.random.rand(NX * NX, 2)
        data[:, 1] = data[:, 1] * 1.0

        def transform(x, y):
            return x + y ** 3, y - x ** 3

        xt, yt = transform(data[:, 0], data[:, 1])
        data_mushroom = np.vstack([xt.flatten(), yt.flatten()]).T
        data_rectangle = data

        return data_mushroom, data_rectangle

    @unittest.skip(reason="Legacy, needs refactoring.")
    def test_llr1(self):
        # create a set of dependent, and one with independent, vectors
        np.random.seed(1)
        n_data = 100
        rdata = np.random.randn(n_data, n_data)
        u, s, v = np.linalg.svd(rdata)

        dependent_vectors = np.column_stack(
            [u[:, 0], u[:, 0] * 2, u[:, 0] ** 2 + u[:, 0]]
        )
        independent_vectors = np.column_stack(
            [u[:, 1], u[:, 2] * 2, u[:, 2] ** 2 + u[:, 3]]
        )

        all_vectors = np.column_stack([dependent_vectors, independent_vectors])

        pcme = PCMEstimation
        eps = 1
        result = pcme.compute_residuals(
            all_vectors,
            eps_scale=eps,
            progressBar=False,
            skipFirst=False,
            bandwidth_type="median",
        )

        allutils._assert_eq_matrices_tol(  # <- use nptest
            np.array([1, 0, 0, 1, 1, 1]), result["Residuals"], tol=1e-1
        )


if __name__ == "__main__":
    unittest.main()
