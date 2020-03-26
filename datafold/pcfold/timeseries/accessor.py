#!/usr/bin/env python3

import itertools
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from datafold.pcfold.timeseries.collection import TSCDataFrame, TSCException
from datafold.utils.datastructure import is_float, is_integer


@pd.api.extensions.register_dataframe_accessor("tsc")
class TSCAccessor(object):
    def __init__(self, tsc_df: TSCDataFrame):

        # NOTE: cannot call TSCDataFrame(tsc_df) here to transform in case it is a normal
        # DataFrame. This is because the accessor has to know when updating this object.
        if not isinstance(tsc_df, TSCDataFrame):
            raise TypeError(
                "Can use 'tsc' extension only for type TSCDataFrame (convert before)"
            )

        self._tsc_df: TSCDataFrame = tsc_df

    def check_tsc(
        self,
        ensure_all_finite=True,
        ensure_same_length=False,
        ensure_const_delta_time=True,
        ensure_delta_time=None,
        ensure_same_time_values=False,
        ensure_normalized_time=False,
        ensure_n_timeseries=None,
        ensure_min_n_timesteps=None,
    ) -> TSCDataFrame:

        # TODO: allow handle_fail="raise | warn | return"?

        if ensure_same_length and not self._tsc_df.is_equal_length():
            raise TSCException.not_same_length(
                actual_lengths=self._tsc_df.is_equal_length()
            )

        if ensure_const_delta_time or ensure_delta_time is not None:
            # save only once, as it can be expensive...
            actual_time_delta = self._tsc_df.delta_time
        else:
            actual_time_delta = None

        if ensure_const_delta_time and isinstance(actual_time_delta, pd.Series):
            raise TSCException.not_const_delta_time(actual_time_delta)

        if ensure_delta_time is not None:

            if (
                isinstance(actual_time_delta, pd.Series)
                and is_float(ensure_delta_time)
                or is_float(actual_time_delta)
                and isinstance(ensure_delta_time, pd.Series)
            ):
                raise TSCException.not_required_delta_time(
                    required_delta_time=ensure_delta_time,
                    actual_delta_time=actual_time_delta,
                )

            if isinstance(actual_time_delta, pd.Series) and isinstance(
                ensure_delta_time, pd.Series
            ):
                if not (actual_time_delta == ensure_delta_time).all():
                    raise TSCException.not_required_delta_time(
                        required_delta_time=ensure_delta_time,
                        actual_delta_time=actual_time_delta,
                    )

        if ensure_all_finite and not self._tsc_df.is_finite():
            raise TSCException.not_finite()

        if ensure_same_time_values and not self._tsc_df.is_same_time_values():
            raise TSCException.not_same_time_values()

        if ensure_normalized_time and self._tsc_df.is_normalized_time():
            raise TSCException.not_normalized_time()

        if (
            ensure_n_timeseries is not None
            and self._tsc_df.n_timeseries != ensure_n_timeseries
        ):
            raise TSCException.not_required_n_timeseries(
                required_n_timeseries=ensure_n_timeseries,
                actual_n_timeseries=self._tsc_df.n_timeseries,
            )

        if ensure_min_n_timesteps is not None:
            _n_timesteps = self._tsc_df.n_timesteps

            if is_integer(_n_timesteps) and _n_timesteps < ensure_min_n_timesteps:
                raise TSCException.not_min_timesteps(
                    _n_timesteps, actual_n_timesteps=_n_timesteps
                )

            if (
                isinstance(_n_timesteps, pd.Series)
                and (_n_timesteps < ensure_min_n_timesteps).any()
            ):
                raise TSCException.not_min_timesteps(
                    _n_timesteps, actual_n_timesteps=_n_timesteps
                )

        return self._tsc_df

    def shift_time(self, shift_t):

        if shift_t == 0:
            return self._tsc_df

        convert_times = self._tsc_df.index.get_level_values(1) + shift_t
        convert_times = pd.Index(convert_times, name=TSCDataFrame.IDX_TIME_NAME)

        new_tsc_index = pd.MultiIndex.from_arrays(
            [self._tsc_df.index.get_level_values(0), convert_times]
        )

        self._tsc_df.index = new_tsc_index
        self._tsc_df._validate()

        return self._tsc_df

    def normalize_time(self):

        convert_times = self._tsc_df.index.get_level_values(TSCDataFrame.IDX_TIME_NAME)
        min_time, _ = self._tsc_df.time_interval()
        delta_time = self._tsc_df.delta_time

        if not self._tsc_df.is_const_delta_time():
            raise TSCException.not_const_delta_time()

        if self._tsc_df.is_datetime_index():
            convert_times = convert_times.astype(np.int64)
            min_time = min_time.astype(np.int64)
            delta_time = delta_time.astype(np.int64)

        convert_times = np.array(
            (convert_times - min_time) / delta_time, dtype=np.int64
        )
        convert_times = pd.Index(convert_times, name=TSCDataFrame.IDX_TIME_NAME)

        new_tsc_index = pd.MultiIndex.from_arrays(
            [
                self._tsc_df.index.get_level_values(TSCDataFrame.IDX_ID_NAME),
                convert_times,
            ]
        )
        self._tsc_df.index = new_tsc_index
        self._tsc_df._validate()

        assert self._tsc_df.is_normalized_time()

        return self._tsc_df

    def shift_matrices(
        self, snapshot_orientation="column"
    ) -> Tuple[np.ndarray, np.ndarray]:

        if not self._tsc_df.is_const_delta_time():
            raise TSCException.not_const_delta_time()

        ts_counts = self._tsc_df.n_timesteps

        if is_integer(ts_counts):
            ts_counts = pd.Series(
                np.ones(self._tsc_df.n_timeseries, dtype=np.int) * ts_counts,
                index=self._tsc_df.ids,
            )

        assert isinstance(ts_counts, pd.Series)  # for mypy

        nr_shift_snapshots = (ts_counts.subtract(1)).sum()
        insert_indices = np.append(0, (ts_counts.subtract(1)).cumsum().to_numpy())

        assert len(insert_indices) == self._tsc_df.n_timeseries + 1

        if snapshot_orientation == "column":
            shift_left = np.zeros([self._tsc_df.n_features, nr_shift_snapshots])
        elif snapshot_orientation == "row":
            shift_left = np.zeros([nr_shift_snapshots, self._tsc_df.n_features])
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

    def time_values_overview(self):

        table = pd.DataFrame(
            [self._tsc_df.time_interval(_id) for _id in self._tsc_df.ids],
            index=self._tsc_df.ids,
            columns=["start", "end"],
        )

        table["delta_time"] = self._tsc_df.delta_time
        return table

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

        df = self._tsc_df.select_time_values(time_values=time)

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
