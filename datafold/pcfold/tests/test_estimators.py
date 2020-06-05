#!/usr/bin/env python

import unittest

import numpy as np
import numpy.testing as nptest

from datafold.pcfold import *


class TestPCMEstimation(unittest.TestCase):
    def test_optimize_parameters_default(self):
        random_state = 2
        gen = np.random.default_rng(random_state)
        n_data = 100

        result = []
        for n_dim in [1, 2, 3, 4]:
            rdata = gen.random(size=(n_data, n_dim))

            pcm = PCManifold(rdata)
            pcm.optimize_parameters(random_state=random_state)

            result.append([pcm.cut_off, pcm.kernel.epsilon])

        # print(result)
        result_expected = [
            [0.12610897125265086, 0.0008633488008103683],
            [0.31424866149403663, 0.005360943095613588],
            [0.49445482177635525, 0.013272341786725417],
            [0.5855076721372056, 0.018610562709202767],
        ]

        # reference test:needs update when changing behavior
        nptest.assert_almost_equal(result_expected, result, decimal=14)

    def test_optimize_parameters_scaling(self):
        random_state = 1
        gen = np.random.default_rng(random_state)
        n_data = 100

        result = []
        for n_dim in [1, 2, 3, 4]:
            rdata = gen.random(size=(n_data, n_dim))

            pcm = PCManifold(rdata)
            pcm.optimize_parameters(random_state=random_state, result_scaling=2)

            # print([pcm.cut_off, pcm.kernel.epsilon])
            result.append([pcm.cut_off, pcm.kernel.epsilon])

        # print(result)
        result_expected = [
            [0.25263361232467707, 0.003464787374764583],
            [0.6251912756694166, 0.02121876691780111],
            [0.88057031318024, 0.042094213956176206],
            [1.2497638171770942, 0.08479108999475204],
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
