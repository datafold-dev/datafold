#!/usr/bin/env python

import pickle
import unittest

import numpy as np

from datafold.pcfold import PCManifold


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
        self.assertTrue(hasattr(unpickled_estimator, "dist_kwargs"))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            PCManifold(np.linspace(0, 10, 20))

        with self.assertRaises(ValueError):
            data = np.linspace(0, 10, 20).reshape(10, 2).astype(np.str_)
            PCManifold(data)

        with self.assertRaises(TypeError):
            data = np.linspace(0, 10, 20).reshape(10, 2)
            PCManifold(data, dist_kwargs=1)

        with self.assertRaises(ValueError):
            PCManifold(np.array([[]]))

        with self.assertRaises(ValueError):
            data = np.linspace(0, 10, 20).reshape(10, 2)
            data[0, 0] = np.nan
            PCManifold(data)

        with self.assertRaises(ValueError):
            data = np.linspace(0, 10, 20).reshape(10, 2)
            data[0, 0] = np.inf
            PCManifold(data)


if __name__ == "__main__":
    unittest.main()
