import unittest
from datetime import datetime

import numpy as np
import numpy.testing as nptest

from datafold import TSCDataFrame
from datafold.dynfold.base import TSCPredictMixin


class TestTSCBase(unittest.TestCase):
    def test_predict_mixin01(self):
        # test with only time values
        mixin = TSCPredictMixin()
        mixin.dt_ = 1
        expected = np.array([0, 1, 2])
        actual = mixin._validate_and_set_time_values_predict(
            time_values=expected, X=None, U=None
        )
        nptest.assert_array_equal(actual, expected)

    def test_predict_mixin02(self):
        # Test with time information in X
        mixin = TSCPredictMixin()
        mixin.dt_ = 1

        expected = np.array([0, 1, 2])
        X = TSCDataFrame.from_array(np.arange(2), time_values=expected[0])
        actual = mixin._validate_and_set_time_values_predict(
            time_values=expected, X=X, U=None
        )
        nptest.assert_array_equal(actual, expected)

        with self.assertRaises(ValueError):
            # raise error because not all values in time_values are greater or equal to
            # initial condition reference
            X = TSCDataFrame.from_array(np.arange(2), time_values=expected[1])
            mixin._validate_and_set_time_values_predict(
                time_values=expected, X=X, U=None
            )

        # because reference value in the initial condition is not contained in time_values,
        # the time value is prepended
        expected = expected + 2
        X = TSCDataFrame.from_array(np.arange(2), time_values=0)
        actual = mixin._validate_and_set_time_values_predict(
            time_values=expected, X=X, U=None
        )
        nptest.assert_array_equal(actual, np.append(0, expected))

        X = TSCDataFrame.from_array(
            np.arange(6).reshape(3, 2), time_values=np.array([0, 1, 2])
        )
        actual = mixin._validate_and_set_time_values_predict(
            time_values=expected, X=X, U=None
        )
        nptest.assert_array_equal(actual, expected)

    def test_predict_mixin03(self):
        # Test with time information in control input U
        mixin = TSCPredictMixin()
        mixin.dt_ = 1

        expected = np.array([0, 1, 2, 3])

        X = TSCDataFrame.from_array(np.arange(2), time_values=expected[0])
        U = TSCDataFrame.from_array(
            np.arange(6).reshape(3, 2), time_values=expected[:-1]
        )

        actual = mixin._validate_and_set_time_values_predict(
            time_values=expected, X=X, U=U
        )
        nptest.assert_array_equal(actual, expected)

        # no time information in X
        actual = mixin._validate_and_set_time_values_predict(
            time_values=expected, X=np.array([0, 1]), U=U
        )
        nptest.assert_array_equal(actual, expected)

        with self.assertRaises(ValueError):
            X = TSCDataFrame.from_array(np.arange(2), time_values=expected[1])
            mixin._validate_and_set_time_values_predict(time_values=expected, X=X, U=U)

    def test_predict_mixin04(self):
        # Test with U as Numpy array
        mixin = TSCPredictMixin()
        mixin.dt_ = 0.1

        expected = np.array([0.0, 0.1, 0.2])

        X = np.arange(2)
        U = np.arange(6).reshape(3, 2)

        actual = mixin._validate_and_set_time_values_predict(
            time_values=expected, X=X, U=U
        )
        nptest.assert_array_equal(actual, expected)

        # no time information in X
        actual = mixin._validate_and_set_time_values_predict(
            time_values=expected, X=np.array([0, 1]), U=U
        )
        nptest.assert_array_equal(actual, expected)

    def test_predict_mixin05(self):
        # Test with U as datetime
        mixin = TSCPredictMixin()
        mixin.dt_ = np.timedelta64(1, "h")
        dt_now = np.datetime64(datetime.now())

        expected = np.array(
            [dt_now, dt_now + mixin.dt_, dt_now + 2 * mixin.dt_, dt_now + 3 * mixin.dt_]
        )

        X = np.arange(2)
        U = np.arange(6).reshape(3, 2)

        actual = mixin._validate_and_set_time_values_predict(time_values=None, X=X, U=U)

        # this should not take more than 5000 us (microseconds)
        self.assertTrue(((actual - expected).astype(int) < 1000).all())

    def test_validate_time_values(self):
        mixin = TSCPredictMixin()

        expected = np.array([0, 1, 2, 3])
        actual = mixin._validate_time_values_format(time_values=expected)

        nptest.assert_array_equal(actual, expected)

        with self.assertRaises(ValueError):
            mixin._validate_time_values_format(time_values=np.atleast_2d(expected))

        with self.assertRaises(TypeError):
            mixin._validate_time_values_format(time_values=expected.astype(str))

        with self.assertRaises(ValueError):
            mixin._validate_time_values_format(time_values="Invalid")

        with self.assertRaises(ValueError):
            # no support of negative values
            mixin._validate_time_values_format(time_values=np.array([-1, 0, 1]))

        with self.assertRaises(ValueError):
            mixin._validate_time_values_format(time_values=np.array([np.nan, 0, 1]))

        with self.assertRaises(ValueError):
            mixin._validate_time_values_format(time_values=np.array([np.inf, 0, 1]))

        with self.assertRaises(ValueError):
            mixin._validate_time_values_format(time_values=np.array([0, 0, 0]))

        with self.assertRaises(ValueError):
            mixin._validate_time_values_format(time_values=np.array([2, 1, 0]))
