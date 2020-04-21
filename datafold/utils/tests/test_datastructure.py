#!/usr/bin/env python3
import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest

from datafold.utils.general import *


class TestDataStructureUtils(unittest.TestCase):
    def setUp(self) -> None:
        self._create_random_series()
        self._create_random_dataframe()

    def _create_random_series(self):
        self.series1 = pd.Series(np.random.rand(10), index=np.arange(10))

    def _create_random_dataframe(self):
        self.df1 = pd.DataFrame(
            np.random.rand(10, 3), index=np.arange(10), columns=["A", "B", "C"]
        )
        self.df2 = pd.DataFrame(
            np.random.rand(10, 1), index=np.arange(10), columns=["A"]
        )

    def test_series_if_applicable1(self):
        actual = series_if_applicable(self.series1)
        pdtest.assert_series_equal(actual, self.series1)

    def test_series_if_applicable2(self):
        actual = series_if_applicable(self.df1)
        pdtest.assert_frame_equal(actual, self.df1)

    def test_series_if_applicable3(self):
        actual = series_if_applicable(self.df2)
        pdtest.assert_series_equal(actual, self.df2.iloc[:, 0])

    def test_is_df_same_index_columns1(self):
        with self.assertRaises(AssertionError):
            is_df_same_index(self.df1, self.df2, handle="raise")

    def test_is_df_same_index_columns2(self):
        df1_local = self.df1.copy()
        is_df_same_index(self.df1, df1_local, handle="raise")

    def test_is_df_same_index_columns3(self):
        df1_local = self.df1.copy()
        df1_local.columns = ["One", "Two", "Three"]

        with self.assertRaises(AssertionError):
            is_df_same_index(self.df1, df1_local)

    def test_is_integer1(self):
        self.assertTrue(is_integer(6.0))

    def test_is_integer2(self):
        self.assertTrue(is_integer(np.float64(5)))

    def test_is_integer3(self):
        self.assertFalse(is_integer(1.3))

    def test_is_integer4(self):
        self.assertFalse(is_integer(np.array([1, 2])))

    def test_is_float1(self):
        self.assertTrue(is_float(1.0))

    def test_is_float2(self):
        self.assertFalse(is_float(1))

    def test_is_float3(self):
        self.assertTrue(is_float(np.float16(1.0)))

    def test_is_float4(self):
        self.assertFalse(is_float(np.int8(1.0)))

    def test_if1dim_colvec1(self):
        actual = if1dim_colvec(np.array([1, 2, 3]))

        self.assertIsInstance(actual, np.ndarray)
        self.assertEqual(actual.ndim, 2)
        self.assertEqual(actual.shape[0], 3)

    def test_if1dim_colvec2(self):
        vec = np.array([1, 2, 3])[np.newaxis, :]
        actual = if1dim_colvec(vec)

        nptest.assert_array_equal(actual, vec)

    def test_if1dim_colvec3(self):
        vec = np.array([1, 2, 3])[:, np.newaxis]
        actual = if1dim_colvec(vec)

        nptest.assert_array_equal(actual, vec)

    def test_if1dim_rowvec1(self):
        actual = if1dim_rowvec(np.array([1, 2, 3]))

        self.assertIsInstance(actual, np.ndarray)
        self.assertEqual(actual.ndim, 2)
        self.assertEqual(actual.shape[1], 3)

    def test_if1dim_rowvec2(self):
        vec = np.array([1, 2, 3])[:, np.newaxis]
        actual = if1dim_colvec(vec)
        nptest.assert_array_equal(actual, vec)

    def test_if1dim_rowvec3(self):
        vec = np.array([1, 2, 3])[np.newaxis, :]
        actual = if1dim_colvec(vec)
        nptest.assert_array_equal(actual, vec)


# if __name__ == "__main__":
#     TestDataStructureUtils().test_if1dim_colvec2()
#     actual = if1dim_rowvec(np.array([1, 2, 3]))
