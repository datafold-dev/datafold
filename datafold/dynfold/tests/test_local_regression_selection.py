#!/usr/bin/env python3

import logging
import os
import unittest

import numpy as np
import numpy.testing as nptest
from sklearn.datasets import make_swiss_roll

from datafold.dynfold.dmap import DiffusionMaps, LocalRegressionSelection


class LocalRegressionSelectionTest(unittest.TestCase):
    def test_automatic_eigendirection_selection_swiss_roll(self):
        points, color = make_swiss_roll(n_samples=5000, noise=0.01, random_state=1)
        dm = DiffusionMaps(epsilon=2.1, n_eigenpairs=6).fit(points)

        loc_regress = LocalRegressionSelection(n_subsample=1000)
        loc_regress = loc_regress.fit(dm.eigenvectors_)

        self.assertTrue(np.isnan(loc_regress.residuals_[0]))
        self.assertTrue(loc_regress.residuals_[1] == 1.0)

        # only starting from 2 because the first two values are trivial
        self.assertTrue(np.argmax(loc_regress.residuals_[2:]) == 3)

    def test_automatic_eigendirection_selection_rectangle(self):
        """
        from
        Paper: Parsimonious Representation of Nonlinear Dynamical Systems Through
        Manifold Learning: A Chemotaxis Case Study, Dsila et al., page 7
        https://arxiv.org/abs/1505.06118v1
        """

        n_samples = 5000
        n_subsample = 500

        # lengths 2, 4, 8 are from paper, added .3 to have it more clear on which index
        # the next independent eigenfunction should appear
        x_length_values = [1, 2.3, 4.3, 8.3]

        for xlen in x_length_values:
            x_direction = np.random.uniform(0, xlen, size=(n_samples, 1))
            y_direction = np.random.uniform(0, 1, size=(n_samples, 1))
            data = np.hstack([x_direction, y_direction])

            dmap = DiffusionMaps(0.1, n_eigenpairs=10).fit(data)

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
        n_subssample = 500

        x_length_values = [2.3, 4.3, 8.3]

        np.random.seed(1)

        for xlen in x_length_values:
            x_direction = np.random.uniform(0, xlen, size=(n_samples, 1))
            y_direction = np.random.uniform(0, 1, size=(n_samples, 1))

            data = np.hstack([x_direction, y_direction])
            dmap = DiffusionMaps(0.1, n_eigenpairs=10).fit(data)

            # -----------------------------------
            # Streategy 1: choose by dimension

            loc_regress_dim = LocalRegressionSelection(
                n_subsample=n_subssample, strategy="dim", intrinsic_dim=2
            )
            actual = loc_regress_dim.fit_transform(dmap.eigenvectors_)

            actual_indices = loc_regress_dim.evec_indices_
            expected_indices = np.array([1, int(xlen + 1)])

            nptest.assert_equal(actual_indices, expected_indices)

            expected = dmap.eigenvectors_[:, actual_indices]
            nptest.assert_array_equal(actual, expected)

            # -----------------------------------
            # Streategy 2: choose by threshold

            loc_regress_thresh = LocalRegressionSelection(
                n_subsample=n_subssample, strategy="threshold", regress_threshold=0.9
            )

            actual = loc_regress_thresh.fit_transform(dmap.eigenvectors_)

            actual_indices = loc_regress_thresh.evec_indices_
            expected_indices = np.array([1, int(xlen + 1)])

            nptest.assert_equal(actual_indices, expected_indices)

            expected = dmap.eigenvectors_[:, expected_indices]
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
