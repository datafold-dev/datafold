#!/usr/bin/env python

import pickle
import unittest

import numpy as np
import sklearn.datasets

import datafold.pcfold.tests.allutils
from datafold.pcfold import *


class TestPCManifold(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_pickleable(self):

        data = np.array([[1, 2, 3]])
        pcm = PCManifold(data)

        pickled_estimator = pickle.dumps(pcm)
        unpickled_estimator = pickle.loads(pickled_estimator)

        # Check if after pickling all attributes are recovered:
        self.assertTrue(hasattr(unpickled_estimator, "kernel"))
        self.assertTrue(hasattr(unpickled_estimator, "_cut_off"))
        self.assertTrue(hasattr(unpickled_estimator, "_dist_backend"))
        self.assertTrue(hasattr(unpickled_estimator, "_dist_params"))


@unittest.skip(reason="Legacy code, requires update.")
class PCManifoldUnitTests(unittest.TestCase):
    def test_init(self):
        pcm = PCManifold(points=np.array([[]]))

    def test_pdist1_parallel(self):

        expected = np.array([[1]])
        actual = np.array([[1]])

        datafold.pcfold.tests.allutils._assert_eq_matrices_tol(
            expected, actual, tol=1e-10
        )

    def test_basic_setup(self):
        # test manifold geometry class

        metric = np.identity(3)
        metric[0, 0] = 1
        metric[0, 1] = 0

        n_points = 500
        noise = 1e-2
        seed = 1
        metric_parameters = {
            "method": "raydist",
            "options": {"cut_off": 100, "kmin": 10, "metric_matrix": metric},
        }

        kernel_parameters = {
            "kernel": "gaussian",
            "options": {"epsilon": 10, "kmin": 10, "cnn": False},
        }

        points, tt = sklearn.datasets.make_swiss_roll(
            n_points, noise=noise, random_state=seed
        )

        np.random.seed(1)
        # points = np.random.randn(n_points,2)
        # points = np.column_stack([np.cos(tt), np.sin(tt)])

        mg = PCManifold(
            points=points,
            verbosity_level=0,
            kernel_parameters=kernel_parameters,
            metric_parameters=metric_parameters,
        )

        mg.analyze(analysis_level=0)
        evecs, evals = mg.diffusion_maps(n_evecs=5)

        points = mg.points

        # prev_distances = mg.sparse_distance_matrix()

    @unittest.skip(reason="Legacy code, requires update.")
    def test_mahalanobis(self):
        """
        test the mahalanobis-distance DMAPS with the mushroom to square example
        """

        d_mushroom, d_rectangle = allutils.generate_mushroom()

        distmatX = distance_matrix(d_rectangle)
        mahalanobisindices = np.array(np.argsort(distmatX.todense(), axis=1))

        MHNOW = 10
        tol = 1e-8

        cov_matrices = np.zeros(
            (d_mushroom.shape[0], d_mushroom.shape[1], d_mushroom.shape[1])
        )
        for i in range(d_mushroom.shape[0]):
            xdata = d_mushroom[mahalanobisindices[i, 0:MHNOW].astype(int), :]
            cov_matrices[i, :, :] = np.linalg.pinv(np.cov(xdata.T), rcond=tol)

        # first, compute mahalanobis distance matrix
        def mahalanobis_distance(ix, iy):
            x = dataset[ix, :]
            y = dataset[iy, :]

            ci = cov_matrices[ix, :, :]
            cj = cov_matrices[iy, :, :]
            return np.sqrt(1 / 2 * (x - y) @ ((ci + cj) @ (x - y)))

        Dmat = np.zeros((dataset.shape[0], dataset.shape[0]))
        for i in range(dataset.shape[0]):
            if np.mod(i, dataset.shape[0] // 10) == 0:
                print(str(int(i / dataset.shape[0] * 100)) + ", ", end="")
            for k in range(i, dataset.shape[0]):
                Dmat[i, k] = mahalanobis_distance(i, k)
                Dmat[k, i] = Dmat[i, k]

        Dmat = scipy.sparse.csr_matrix(Dmat)


if __name__ == "__main__":

    unittest.main()
