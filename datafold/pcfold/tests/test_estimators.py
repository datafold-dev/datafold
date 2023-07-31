#!/usr/bin/env python

import unittest

import numpy as np
import numpy.testing as nptest

from datafold.pcfold import PCManifold


class TestPCMEstimation(unittest.TestCase):
    def test_optimize_parameters_default(self):
        random_state = 2
        gen = np.random.default_rng(random_state)
        n_data = 100

        result = []
        for n_dim in [1, 2, 3, 4]:
            rdata = gen.uniform(size=(n_data, n_dim))

            pcm = PCManifold(rdata)
            pcm.optimize_parameters(random_state=random_state)

            result.append([pcm.cut_off, pcm.kernel.epsilon])

        result_expected = [
            [0.288124501, 0.004506659],
            [0.591118486, 0.018968955],
            [0.703665523, 0.026879852],
            [0.815311632, 0.036086237],
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

        # test if the approximated values for epsilon and the cutoff are within a good
        # bound from the best value
        _zero = np.zeros((len(result),))
        nptest.assert_almost_equal(
            (np.array(result)[:, 0] - np.array([cut_off_best])), _zero, decimal=1
        )
        nptest.assert_almost_equal(
            np.array(result)[:, 1] - np.array([epsilon_best]), _zero, decimal=3
        )

    def test_optimize_parameters_scaling(self):
        random_state = 1
        gen = np.random.default_rng(random_state)
        n_data = 100

        result = []
        for n_dim in [1, 2, 3, 4]:
            rdata = gen.uniform(size=(n_data, n_dim))

            pcm = PCManifold(rdata)
            pcm.optimize_parameters(random_state=random_state, result_scaling=2)

            result.append([pcm.cut_off, pcm.kernel.epsilon])

        result_expected = [
            [0.51297749066774, 0.0142853518602],
            [1.07758224784906, 0.06303694836363],
            [1.37926934745955, 0.10327435556181],
            [1.67525618948012, 0.1523550263642],
        ]

        # reference test:needs update when changing behavior
        nptest.assert_almost_equal(result_expected, result, decimal=8)

    def test_optimize_parameters_below_tolerance(self):
        random_state = 1
        gen = np.random.default_rng(random_state)
        n_data = 100

        result = []
        for tol in [1e-6, 1e-8, 1e-10]:
            for n_dim in [1, 2, 3, 4]:
                rdata = gen.uniform(size=(n_data, n_dim))

                pcm = PCManifold(rdata)
                pcm.optimize_parameters(random_state=random_state, tol=tol)

                result.append([np.exp(-pcm.cut_off**2 / pcm.kernel.epsilon) - tol])
            result_expected = np.zeros(
                len(result),
            )

            nptest.assert_allclose(
                result_expected, np.asarray(result).ravel(), rtol=tol, atol=1e-15
            )

    @staticmethod
    def generate_mushroom(n_points=500):
        NX = int(np.sqrt(n_points))

        rng = np.random.default_rng(1)
        data = rng.random((NX * NX, 2))
        data[:, 1] = data[:, 1] * 1.0

        def transform(x, y):
            return x + y**3, y - x**3

        xt, yt = transform(data[:, 0], data[:, 1])
        data_mushroom = np.vstack([xt.flatten(), yt.flatten()]).T
        data_rectangle = data

        return data_mushroom, data_rectangle


if __name__ == "__main__":
    unittest.main()
