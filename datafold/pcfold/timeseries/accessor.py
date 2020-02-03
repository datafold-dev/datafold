#!/usr/bin/env python3

import itertools
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from datafold.pcfold.timeseries.collection import (
    TimeSeriesCollectionError,
    TSCDataFrame,
)
from datafold.utils.datastructure import is_integer


@pd.api.extensions.register_dataframe_accessor("tsc")
class TSCollectionMethods(object):
    def __init__(self, tsc_df):

        # NOTE: cannot call TSCDataFrame(tsc_df) here to transform in case it is a normal
        # DataFrame. This is because the accessor has to know when updating this object.
        if not isinstance(tsc_df, TSCDataFrame):
            raise ValueError(
                "Can use 'tsc' extension only for type TSCDataFrame (convert before)."
            )

        self._tsc_df = tsc_df

    def shift_time(self, shift_t):

        if shift_t == 0:
            return self._tsc_df

        convert_times = self._tsc_df.index.get_level_values(1) + shift_t
        convert_times = pd.Index(convert_times, self._tsc_df.index.names[1])

        new_tsc_index = pd.MultiIndex.from_arrays(
            [self._tsc_df.index.get_level_values(0), convert_times]
        )

        self._tsc_df.index = new_tsc_index
        self._tsc_df._validate()

        return self._tsc_df

    def normalize_time(self):

        convert_times = self._tsc_df.index.get_level_values(1)
        min_time, _ = self._tsc_df.time_interval()

        if not self._tsc_df.is_const_dt():
            raise TimeSeriesCollectionError(
                "To normalize the time index it is required that the time time delta is "
                "constant."
            )

        convert_times = np.array(
            (convert_times / self._tsc_df.dt) - min_time, dtype=np.int
        )
        convert_times = pd.Index(convert_times, name=self._tsc_df.index.names[1])

        new_tsc_index = pd.MultiIndex.from_arrays(
            [self._tsc_df.index.get_level_values(0), convert_times]
        )
        self._tsc_df.index = new_tsc_index
        self._tsc_df._validate()

        return self._tsc_df

    def shift_matrices(
        self, snapshot_orientation="column"
    ) -> Tuple[np.ndarray, np.ndarray]:

        if not self._tsc_df.is_const_dt():
            raise TimeSeriesCollectionError(
                "Cannot compute shift matrices: Time series are required to have the "
                "same time delta."
            )

        ts_counts = self._tsc_df.lengths_time_series

        if is_integer(ts_counts):
            ts_counts = pd.Series(
                np.ones(self._tsc_df.nr_timeseries, dtype=np.int) * ts_counts,
                index=self._tsc_df.ids,
            )

        nr_shift_snapshots = (ts_counts.subtract(1)).sum()
        insert_indices = np.append(0, (ts_counts.subtract(1)).cumsum().to_numpy())

        assert len(insert_indices) == self._tsc_df.nr_timeseries + 1

        if snapshot_orientation == "column":
            shift_left = np.zeros([self._tsc_df.nr_qoi, nr_shift_snapshots])
        elif snapshot_orientation == "row":
            shift_left = np.zeros([nr_shift_snapshots, self._tsc_df.nr_qoi])
        else:
            raise ValueError(f"snapshot_orientation={snapshot_orientation} not known")

        shift_right = np.zeros_like(shift_left)

        # NOTE: if this has performance issues or memory issues, then it may be beneficial
        # to do the whole thing with boolean indexing

        for i, (id_, ts_df) in enumerate(self._tsc_df.itertimeseries()):
            # transpose because snapshots are column-wise by convention, whereas here they
            # are row-wise

            if snapshot_orientation == "column":
                # start from 0 and exclude last snapshot
                shift_left[:, insert_indices[i] : insert_indices[i + 1]] = ts_df.iloc[
                    :-1, :
                ].T
                # exclude 0 and go to last snapshot
                shift_right[:, insert_indices[i] : insert_indices[i + 1]] = ts_df.iloc[
                    1:, :
                ].T
            else:  # "row"
                shift_left[insert_indices[i] : insert_indices[i + 1], :] = ts_df.iloc[
                    :-1, :
                ]
                shift_right[insert_indices[i] : insert_indices[i + 1], :] = ts_df.iloc[
                    1:, :
                ]

        return shift_left, shift_right

    def kfold_cv_reassign_ids(self, train_indices, test_indices):

        # mark train samples as zero and test samples with 1
        mask_train_test = np.zeros(self._tsc_df.shape[0])
        mask_train_test[test_indices] = 1

        change_fold_indicator = np.append(0, np.diff(mask_train_test)).astype(np.bool)
        change_id_indicator = np.append(
            0, np.diff(self._tsc_df.index.get_level_values("ID"))
        ).astype(np.bool)

        id_cum_sum_mask = np.logical_or(change_fold_indicator, change_id_indicator)
        new_ids = np.cumsum(id_cum_sum_mask)

        new_idx = pd.MultiIndex.from_arrays(
            arrays=(new_ids, self._tsc_df.index.get_level_values("time"))
        )

        splitted_tsc = TSCDataFrame.from_same_indices_as(
            self._tsc_df, values=self._tsc_df, except_index=new_idx
        )

        train_tsc = splitted_tsc.iloc[train_indices, :]
        test_tsc = splitted_tsc.iloc[test_indices, :]

        # asserts also assumption made in the algorithm (in hindsight)
        assert isinstance(train_tsc, TSCDataFrame)
        assert isinstance(test_tsc, TSCDataFrame)

        return train_tsc, test_tsc

    def plot_density2d(self, time, xresolution: int, yresolution: int, covariance=None):
        """
        Plot the density for a given time_step. For this:
          - Take the first two columns of the underlying data frame and interpret them as
            x and y coordinates.
          - Then, place Gaussian bells onto these coordinates and sum up the values of the
            corresponding probability density functions (PDF).
          - The PDF must be evaluated on a fine-granular grid.

          Returns axis handle.
        """

        if len(self._tsc_df.columns) != 2:
            raise ValueError("Density can only be plotted for 2D time series.")

        if covariance is None:
            covariance = np.eye(2)

        df = self._tsc_df.select_times(time_points=time)

        xmin = df.iloc[:, 0].min()
        xmax = df.iloc[:, 0].max()
        ymin = df.iloc[:, 1].min()
        ymax = df.iloc[:, 1].max()

        xdim = (xmin, xmax, xresolution)
        ydim = (ymin, ymax, yresolution)

        grid = self._generate_2d_meshgrid(xdim, ydim)

        coordinates = list(list(item) for item in df.itertuples(index=False))
        summed_density_values_as_vector = self._sum_up_density_values_on_grid(
            grid, coordinates, covariance
        )

        reshaped_density_values = summed_density_values_as_vector.reshape(
            xresolution, yresolution
        )

        gridsize = (1, 1)
        ax1 = plt.subplot2grid(gridsize, (0, 0))

        heatmap_plot = ax1.imshow(
            reshaped_density_values, origin="lower", cmap="seismic"
        )

        ax1.set_xlabel(df.columns[0])
        ax1.set_ylabel(df.columns[1])

        # The plot contains [xy]resolution pixels starting from 0 to [xy]resolution.
        # Therefore, place ticks at first and last pixel.
        ax1.set_xticks([0, xresolution])
        ax1.set_yticks([0, yresolution])

        float_precision = 4
        ax1.set_xticklabels(
            [round(xmin, float_precision), round(xmax, float_precision)]
        )
        ax1.set_yticklabels(
            [round(ymin, float_precision), round(ymax, float_precision)]
        )

        return heatmap_plot

    def _generate_2d_meshgrid(self, xdim, ydim):
        """
        Both parameters must be a tuple consisting of three values (min, max, resolution
        Return a n x 2 vector representing the grid.
        """
        x = np.linspace(*xdim)
        y = np.linspace(*ydim)

        xx, yy = np.meshgrid(x, y)
        grid = np.c_[xx.ravel(), yy.ravel()]

        return grid

    def _sum_up_density_values_on_grid(self, grid, coordinates, covariance):
        # Evaluate probability density functions of Gaussian bells on grid.
        normal_distributions_at_given_coordinates = [
            multivariate_normal(coordinate, cov=covariance)
            for coordinate in coordinates
        ]
        evaluated_density_functions_on_grid = [
            distribution.pdf(grid)
            for distribution in normal_distributions_at_given_coordinates
        ]

        # Stack all 1D-vectors and sum them up vertically.
        stacked_density_values_on_grid = np.vstack(evaluated_density_functions_on_grid)
        summed_density_values_as_vector = np.sum(stacked_density_values_on_grid, axis=0)

        return summed_density_values_as_vector


if __name__ == "__main__":
    pass
