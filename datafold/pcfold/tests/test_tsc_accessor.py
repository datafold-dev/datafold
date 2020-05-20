import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest

from datafold.pcfold import TSCDataFrame


class TestTscAccessor(unittest.TestCase):
    def setUp(self) -> None:
        # The last two elements are used
        idx = pd.MultiIndex.from_arrays(
            [[0, 0, 1, 1, 15, 15, 45, 45, 45], [0, 1, 0, 1, 0, 1, 17, 18, 19]]
        )
        col = ["A", "B"]
        self.simple_df = pd.DataFrame(np.random.rand(9, 2), index=idx, columns=col)

    def test_normalize_time1(self):
        # NOTE: more tests are included in test_tsc_data_structre/test_is_normalize_time()
        to_convert = TSCDataFrame(self.simple_df)

        actual = to_convert.tsc.normalize_time()
        expected = to_convert

        pdtest.assert_frame_equal(actual, expected)

    def test_normalize_time2(self):
        simple_df = self.simple_df.copy()
        simple_df.index = pd.MultiIndex.from_arrays(
            [
                simple_df.index.get_level_values(0),
                simple_df.index.get_level_values(1) + 1,
            ]
        )

        to_convert = TSCDataFrame(simple_df)

        actual = to_convert.tsc.normalize_time()
        expected = TSCDataFrame(self.simple_df)  # undo the +1

        pdtest.assert_frame_equal(actual, expected)

    def test_normalize_time3(self):
        simple_df = self.simple_df.copy()
        simple_df.index = pd.MultiIndex.from_arrays(
            [
                simple_df.index.get_level_values(0),
                simple_df.index.get_level_values(1) + 0.5,
            ]
        )  # shift by float

        to_convert = TSCDataFrame(simple_df)

        actual = to_convert.tsc.normalize_time()
        expected = TSCDataFrame(self.simple_df)  # undo the +1

        pdtest.assert_frame_equal(actual, expected)

    def test_shift_matrices(self):
        # TODO: potentially do more tests (esp. with uneven number of time series,
        #  this is a quite important functionality!)

        tc = TSCDataFrame(self.simple_df)
        actual_left, actual_right = tc.tsc.compute_shift_matrices()

        original_values = self.simple_df.to_numpy()

        expected_left = np.zeros([2, 5])
        expected_left[:, 0] = original_values[0, :]
        expected_left[:, 1] = original_values[2, :]
        expected_left[:, 2] = original_values[4, :]
        expected_left[:, 3] = original_values[6, :]
        expected_left[:, 4] = original_values[7, :]

        expected_right = np.zeros_like(expected_left)
        expected_right[:, 0] = original_values[1, :]
        expected_right[:, 1] = original_values[3, :]
        expected_right[:, 2] = original_values[5, :]
        expected_right[:, 3] = original_values[7, :]
        expected_right[:, 4] = original_values[8, :]

        nptest.assert_equal(actual_left, expected_left)
        nptest.assert_equal(actual_right, expected_right)

        actual_left, actual_right = tc.tsc.compute_shift_matrices(
            snapshot_orientation="row"
        )

        nptest.assert_equal(actual_left, expected_left.T)
        nptest.assert_equal(actual_right, expected_right.T)

    def test_shift_matrices2(self):
        simple_df = self.simple_df.copy()
        simple_df = simple_df.drop(labels=[45])

        tc = TSCDataFrame(simple_df)

        actual_left, actual_right = tc.tsc.compute_shift_matrices()

        original_values = simple_df.to_numpy()

        expected_left = np.zeros([2, 3])
        expected_left[:, 0] = original_values[0, :]
        expected_left[:, 1] = original_values[2, :]
        expected_left[:, 2] = original_values[4, :]

        expected_right = np.zeros_like(expected_left)
        expected_right[:, 0] = original_values[1, :]
        expected_right[:, 1] = original_values[3, :]
        expected_right[:, 2] = original_values[5, :]

        nptest.assert_equal(actual_left, expected_left)
        nptest.assert_equal(actual_right, expected_right)

    def test_shift_time1(self):
        tsc_df = TSCDataFrame(self.simple_df)

        tsc_df.tsc.shift_time(5)
        nptest.assert_array_equal(
            tsc_df.index.get_level_values(1),
            self.simple_df.index.get_level_values(1) + 5,
        )

    def test_shift_time2(self):
        tsc_df = TSCDataFrame(self.simple_df)

        with self.assertRaises(AttributeError):
            # time is not allowed to be negative
            tsc_df.tsc.shift_time(-5)


if __name__ == "__main__":
    # test = TestErrorTimeSeries()
    # test.setUp()
    # test.test_error_per_timestep3()
    # exit()

    unittest.main()
