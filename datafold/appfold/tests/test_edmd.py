#!/usr/bin/env python3

import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.testing as pdtest

from datafold.appfold.edmd import EDMDDict
from datafold.pcfold.timeseries import TSCDataFrame
from datafold.pcfold.timeseries.transform import (
    TSCPrincipalComponent,
    TSCQoiScale,
    TSCTakensEmbedding,
)


class EDMDTest(unittest.TestCase):
    def setUp(self) -> None:
        time = np.linspace(0, 2 * np.pi, 100)
        df = pd.DataFrame(np.sin(time) + 10, index=time, columns=["sin"])
        self.sine_wave_tsc = TSCDataFrame.from_single_timeseries(df)

    def test_id_dict(self):
        _edmd_dict = EDMDDict().fit(self.sine_wave_tsc)

        pdtest.assert_frame_equal(
            _edmd_dict.transform(self.sine_wave_tsc), self.sine_wave_tsc
        )

        pdtest.assert_frame_equal(
            _edmd_dict.inverse_transform(self.sine_wave_tsc), self.sine_wave_tsc
        )

    def test_simple_sine_wave(self, plot=False):
        _edmd_dict = EDMDDict(
            steps=(
                ("scale", TSCQoiScale(name="min-max")),
                ("delays", TSCTakensEmbedding(delays=10)),
                ("pca", TSCPrincipalComponent(n_components=2)),
            )
        )

        forward_dict = _edmd_dict.fit_transform(X_ts=self.sine_wave_tsc)
        inverse_dict = _edmd_dict.inverse_transform(X_ts=forward_dict)

        # index not the same because of Takens, so only check column
        pdtest.assert_index_equal(
            self.sine_wave_tsc.columns, inverse_dict.columns,
        )

        diff = inverse_dict - self.sine_wave_tsc
        # sort out the removed rows from Takens (NaN values)
        self.assertTrue((diff.dropna() < 1e-14).all().all())

        if plot:
            ax = self.sine_wave_tsc.plot()
            inverse_dict.plot(ax=ax)
            plt.show()
