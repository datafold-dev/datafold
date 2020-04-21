#!/usr/bin/env python

import unittest

import numpy as np

import datafold.pcfold.tests.allutils
from datafold.pcfold import *


class PCMEstimationUnitTests(unittest.TestCase):
    def test_optimize_parameters_default(self):
        random_state = 1
        np.random.seed(random_state)
        n_data = 100

        result = []
        for n_dim in [1, 2, 3, 4]:
            rdata = np.random.rand(n_data, n_dim)

            pcm = PCManifold(rdata)
            pcm.optimize_parameters(random_state=random_state)

            result.append([pcm.cut_off, pcm.kernel.epsilon])
        result_expected = [
            [0.13242882262406347, 0.0009520491292024215],
            [0.3348229248212359, 0.006085898373905224],
            [0.4876941887863767, 0.012911880135270698],
            [0.5963298590812559, 0.01930489463309435],
        ]
        datafold.pcfold.tests.allutils._assert_eq_matrices_tol(
            np.array(result_expected), np.array(result), tol=1e-8
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
            [0.26485764524812694, 0.003808196516809686],
            [0.6696458496424718, 0.024343593495620895],
            [0.9753883775727534, 0.05164752054108279],
            [1.1926597181625118, 0.0772195785323774],
        ]
        datafold.pcfold.tests.allutils._assert_eq_matrices_tol(
            np.array(result_expected), np.array(result), tol=1e-8
        )

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
            result_expected = np.zeros(4,)

            datafold.pcfold.tests.allutils._assert_eq_matrices_tol(
                np.array(result_expected), np.array(result), tol=tol
            )

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

        allutils._assert_eq_matrices_tol(
            np.array([1, 0, 0, 1, 1, 1]), result["Residuals"], tol=1e-1
        )


if __name__ == "__main__":
    unittest.main()
