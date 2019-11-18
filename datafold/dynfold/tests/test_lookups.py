import logging
import unittest

from datafold.dynfold.plot import LookupDmapsEpsilon
from datafold.dynfold.tests.helper import make_strip


class LookupsTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        self.xmin = 0.0
        self.ymin = 0.0
        self.width = 1.0
        self.height = 1e-1
        self.num_samples = 500
        self.data = make_strip(
            self.xmin, self.ymin, self.width, self.height, self.num_samples
        )

    def test_lookup_dmaps_epsilon(self):
        epsilons = [0.1, 0.4, 0.6, 0.8]
        lut_dm = LookupDmapsEpsilon(self.data, epsilons)

        for i, e in enumerate(epsilons):
            self.assertEqual(lut_dm._dmaps[i].epsilon, e)


if __name__ == "__main__":

    # comment in to run/debug single runs:
    # t = LookupsTest()
    # t.setUp()
    # t.test_lookup_kernel_mats_epsilon()
    # exit()

    import os

    verbose = os.getenv("VERBOSE")
    if verbose is not None:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    else:
        logging.basicConfig(level=logging.ERROR, format="%(message)s")
    unittest.main()
