#!/usr/bin/env python3

import logging
import os
import unittest

import numpy as np
import numpy.testing as nptest
from sklearn.datasets import make_swiss_roll

from datafold.dynfold.diffusion_maps import DiffusionMaps, LocalRegressionSelection


class LocalRegressionSelectionTest(unittest.TestCase):
    def test_automatic_eigendirection_selection_swiss_roll(self):
        points, color = make_swiss_roll(n_samples=5000, noise=0.01, random_state=1)
        dm = DiffusionMaps(epsilon=2.1, num_eigenpairs=6).fit(points)

        # import matplotlib.pyplot as plt
        # f, ax = plt.subplots(figsize=[9, 9])
        # ax.scatter(dm.eigenvectors[1, :], dm.eigenvectors[5, :], c=color,
        #            cmap=plt.cm.Spectral)  # NOTE: eigenvectors are row-wise in matrix
        # plot_eigenvectors_n_vs_all(dm.eigenvectors, 1, color)

        loc_regress = LocalRegressionSelection(n_subsample=1000)
        loc_regress = loc_regress.fit(dm.eigenvectors_)

        self.assertTrue(np.isnan(loc_regress.residuals_[0]))
        self.assertTrue(loc_regress.residuals_[1] == 1.0)
        # only starting from 2 because the first two values are trivial
        self.assertTrue(np.argmax(loc_regress.residuals_[2:]) == 3)

    def test_automatic_eigendirection_selection_rectangle(self):
        nsamples = 5000

        # lengths 2, 4, 8 are from paper, added .3 to have it more clear on which index the next independent
        # eigenfunction should appear
        x_length_values = [1, 2.3, 4.3, 8.3]

        for xlen in x_length_values:
            x_direction = np.random.uniform(0, xlen, size=(nsamples, 1))
            y_direction = np.random.uniform(0, 1, size=(nsamples, 1))
            data = np.hstack([x_direction, y_direction])

            dmap = DiffusionMaps(0.1, num_eigenpairs=10).fit(data)

            loc_regress = LocalRegressionSelection(n_subsample=1000)
            loc_regress.fit(dmap.eigenvectors_)

            self.assertTrue(np.isnan(loc_regress.residuals_[0]))
            self.assertTrue(loc_regress.residuals_[1] == 1.0)  # always first directions

            loc_regress.residuals_[0:2] = 0  # setting to zero for easier checking

            # from the paper-example we know the position of the next independent eigendirection
            # Paper: Parsimonious Representation of Nonlinear Dynamical Systems Through Manifold Learning: A
            # Chemotaxis Case Study, Dsila et al., page 7     https://arxiv.org/abs/1505.06118v1
            self.assertEqual(
                int(xlen + 1), np.argmax(loc_regress.residuals_)
            )  # ignoring the first two trivial cases

            # from pydmap.plot import plot_eigenvectors_n_vs_all
            # import matplotlib.pyplot as plt
            # plot_eigenvectors_n_vs_all(dmap.eigenvectors, 1)
            # plt.show()

    def test_choose_automatic_parametrization(self):
        # For explanation see "test_automatic_eigendirection_selection_rectangle"
        # redoing this test here, for choosing the correct parametrization automatically

        nsamples = 5000

        x_length_values = [2.3, 4.3, 8.3]

        np.random.seed(1)

        for xlen in x_length_values:
            x_direction = np.random.uniform(0, xlen, size=(nsamples, 1))
            y_direction = np.random.uniform(0, 1, size=(nsamples, 1))
            data = np.hstack([x_direction, y_direction])

            dmap = DiffusionMaps(0.1, num_eigenpairs=10).fit(data)

            loc_regress_dim = LocalRegressionSelection(
                n_subsample=1000, strategy="dim", intrinsic_dim=2
            )
            loc_regress_threshold = LocalRegressionSelection(
                n_subsample=1000, strategy="threshold", regress_threshold=0.9
            )

            actual_dim = loc_regress_dim.fit_transform(dmap.eigenvectors_)
            actual_thresh = loc_regress_threshold.fit_transform(dmap.eigenvectors_)

            indices1 = loc_regress_dim.evec_indices_
            indices2 = loc_regress_threshold.evec_indices_

            nptest.assert_equal(np.sort(indices1), np.array([1, int(xlen + 1)]))
            nptest.assert_equal(np.sort(indices2), np.array([1, int(xlen + 1)]))

            expected = dmap.eigenvectors_[[1, int(xlen + 1)], :]

            nptest.assert_array_equal(actual_dim, expected)
            nptest.assert_array_equal(actual_thresh, expected)

    def test_api_automatic_parametrization(self):
        # Same test as test_choose_automatic_parametrization, just using the proper sklean-like API
        nsamples = 5000

        x_length_values = [2.3, 4.3, 8.3]

        np.random.seed(1)

        for xlen in x_length_values:
            x_direction = np.random.uniform(0, xlen, size=(nsamples, 1))
            y_direction = np.random.uniform(0, 1, size=(nsamples, 1))

            data = np.hstack([x_direction, y_direction])

            # dmap1 = DiffusionMaps(0.1, num_eigenpairs=10, parametrization_strategy="locregress_intrinsic_dim",
            #                       locregress_intrinsic_dim=2).fit(data)

            dmap1 = DiffusionMaps(0.1, num_eigenpairs=10).fit(data)

            loc_regress_dim = LocalRegressionSelection(
                n_subsample=1000, strategy="dim", intrinsic_dim=2
            )
            actual = loc_regress_dim.fit_transform(dmap1.eigenvectors_)

            indices_dim = loc_regress_dim.evec_indices_

            nptest.assert_equal(np.sort(indices_dim), np.array([1, int(xlen + 1)]))

            expected = dmap1.eigenvectors_[indices_dim, :]
            nptest.assert_array_equal(actual, expected)

            # -----------------------------------

            dmap2 = DiffusionMaps(0.1, num_eigenpairs=10).fit(data)
            loc_regress_thresh = LocalRegressionSelection(
                n_subsample=1000, strategy="threshold", regress_threshold=0.9
            )

            actual = loc_regress_thresh.fit_transform(dmap2.eigenvectors_)
            indices_thresh = loc_regress_thresh.evec_indices_

            nptest.assert_equal(np.sort(indices_thresh), np.array([1, int(xlen + 1)]))

            expected = dmap2.eigenvectors_[[1, int(xlen + 1)], :]
            nptest.assert_array_equal(actual, expected)


if __name__ == "__main__":
    verbose = os.getenv("VERBOSE")
    if verbose is not None:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    else:
        logging.basicConfig(level=logging.ERROR, format="%(message)s")

    # Comment in to run/debug specific tests

    t = LocalRegressionSelectionTest()
    t.setUp()
    t.test_api_automatic_parametrization()
    exit()

    # DiffusionMapsLegacyTest().test_sanity_dense_sparse()
    # exit()
    unittest.main()
