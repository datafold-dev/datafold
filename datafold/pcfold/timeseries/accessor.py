#!/usr/bin/env python3

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from datafold.pcfold.timeseries.collection import TimeSeriesCollectionError, TSCDataFrame


@pd.api.extensions.register_dataframe_accessor("tsc")
class TSCollectionMethods(object):

    def __init__(self, tsc_df):

        # NOTE: cannot call TSCDataFrame(tsc_df) here to transform in case it is a normal DataFrame.
        # This is because the accessor has to know when updating this object.
        if not isinstance(tsc_df, TSCDataFrame):
            raise ValueError("Can use 'tsc' extension only for type TSCDataFrame (convert before).")

        self._tsc_df = tsc_df

    def normalize_time(self):

        convert_times = self._tsc_df.index.get_level_values(1)
        min_time, _ = self._tsc_df.time_interval()

        if not self._tsc_df.is_const_dt():
            raise TimeSeriesCollectionError("To normalize the time index it is required that the time time delta is "
                                            "constant.")

        convert_times = np.array((convert_times / self._tsc_df.dt) - min_time, dtype=np.int)
        convert_times = pd.Index(convert_times, name=self._tsc_df.index.names[1])

        converted_time_multi_index = pd.MultiIndex.from_arrays([self._tsc_df.index.get_level_values(0), convert_times])
        self._tsc_df.index = converted_time_multi_index

        return self._tsc_df

    def shift_matrices(self, snapshot_orientation="column"):

        if not self._tsc_df.is_const_dt():
            raise TimeSeriesCollectionError("Cannot compute shift matrices: Time series are required to have the same "
                                            "time delta.")

        ts_counts = self._tsc_df.lengths_time_series
        if isinstance(ts_counts, int):
            ts_counts = pd.Series(np.ones(self._tsc_df.nr_timeseries, dtype=np.int) * ts_counts, index=self._tsc_df.ids)

        nr_shift_snapshots = (ts_counts - 1).sum()
        insert_indices = np.append(0, (ts_counts - 1).cumsum().to_numpy())

        assert len(insert_indices) == self._tsc_df.nr_timeseries + 1

        if snapshot_orientation == "column":
            shift_left = np.zeros([self._tsc_df.nr_qoi, nr_shift_snapshots])
        elif snapshot_orientation == "row":
            shift_left = np.zeros([nr_shift_snapshots, self._tsc_df.nr_qoi])
        else:
            raise ValueError(f"snapshot_orientation={snapshot_orientation} not known")

        shift_right = np.zeros_like(shift_left)

        # NOTE: if this has performance issues or memory issues, then it may be beneficial to do the whole thing with
        #  boolean indexing

        for i, (id_, ts_df) in enumerate(self._tsc_df.itertimeseries()):
            # transpose because snapshots are column-wise by convention, whereas here they are row-wise

            if snapshot_orientation == "column":
                # start from 0 and exclude last snapshot
                shift_left[:, insert_indices[i]:insert_indices[i+1]] = ts_df.iloc[:-1, :].T
                # exclude 0 and go to last snapshot
                shift_right[:, insert_indices[i]:insert_indices[i+1]] = ts_df.iloc[1:, :].T
            else:  # "row"
                shift_left[insert_indices[i]:insert_indices[i + 1], :] = ts_df.iloc[:-1, :]
                shift_right[insert_indices[i]:insert_indices[i+1], :] = ts_df.iloc[1:, :]

        return shift_left, shift_right

    def takens_embedding(self, lag, delays, frequency, time_direction):
        """Wrapper function for class TakensEmbedding"""
        takens = TakensEmbedding(lag, delays, frequency, time_direction)
        return takens.apply(self._tsc_df)

    def plot_density2d(self, time, xresolution: int, yresolution: int, covariance=None):
        """
        Plot the density for a given time_step. For this:
          - Take the first two columns of the underlying data frame and interpret them as x and y coordinates.
          - Then, place Gaussian bells onto these coordinates and sum up the values of the corresponding probability density functions (PDF).
          - The PDF must be evaluated on a fine-granular grid.

          Returns axis handle.
        """

        if len(self._tsc_df.columns) != 2:
            raise ValueError("Density can only be plotted for 2D time series.")

        if covariance is None:
            covariance = np.eye(2)

        df = self._tsc_df.single_time_df(time=time)

        xmin = df.iloc[:, 0].min()
        xmax = df.iloc[:, 0].max()
        ymin = df.iloc[:, 1].min()
        ymax = df.iloc[:, 1].max()

        xdim = (xmin, xmax, xresolution)
        ydim = (ymin, ymax, yresolution)

        grid = self._generate_2d_meshgrid(xdim, ydim)

        coordinates = list(list(item) for item in df.itertuples(index=False))
        summed_density_values_as_vector = self._sum_up_density_values_on_grid(grid, coordinates, covariance)

        reshaped_density_values = summed_density_values_as_vector.reshape(xresolution, yresolution)

        gridsize = (1, 1)
        ax1 = plt.subplot2grid(gridsize, (0, 0))

        heatmap_plot = ax1.imshow(reshaped_density_values, origin="lower", cmap="seismic")

        ax1.set_xlabel(df.columns[0])
        ax1.set_ylabel(df.columns[1])

        # The plot contains [xy]resolution pixels starting from 0 to [xy]resolution.
        # Therefore, place ticks at first and last pixel.
        ax1.set_xticks([0, xresolution])
        ax1.set_yticks([0, yresolution])

        float_precision = 4
        ax1.set_xticklabels([round(xmin, float_precision), round(xmax, float_precision)])
        ax1.set_yticklabels([round(ymin, float_precision), round(ymax, float_precision)])

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
        normal_distributions_at_given_coordinates = [multivariate_normal(coordinate, cov=covariance) for coordinate in
                                                     coordinates]
        evaluated_density_functions_on_grid = [distribution.pdf(grid) for distribution in
                                               normal_distributions_at_given_coordinates]

        # Stack all 1D-vectors and sum them up vertically.
        stacked_density_values_on_grid = np.vstack(evaluated_density_functions_on_grid)
        summed_density_values_as_vector = np.sum(stacked_density_values_on_grid, axis=0)

        return summed_density_values_as_vector


class TakensEmbedding(object):

    def __init__(self, lag: int, delays: int, frequency: int, time_direction="backward"):
        # TODO: provide fill strategy for edge cases, or remove rows that have a nan?

        if lag < 0:
            raise ValueError(f"Lag has to be non-negative. Got lag={lag}")

        if delays < 1:
            raise ValueError(f"delays has to be an integer larger than zero. Got delays={frequency}")

        if frequency < 1:
            raise ValueError(f"frequency has to be an integer larger than zero. Got frequency={frequency}")

        if time_direction not in ["backward", "forward"]:
            raise ValueError(f"time_direction={time_direction} invalid. Valid choices: {['backward', 'forward']}")

        self.lag = lag
        self.delays = delays
        self.frequency = frequency
        self.time_direction = time_direction

        self.delay_indices = self._precompute_delay_indices()

    def _precompute_delay_indices(self):
        # zero delay (original data) is not treated
        return self.lag + (np.arange(1, (self.delays+1)*self.frequency, self.frequency))

    def _expand_all_delay_columns(self, cols):
        def expand():
            delayed_columns = list()
            for didx in self.delay_indices:
                delayed_columns.append(self._expand_single_dela_column(cols, didx))
            return delayed_columns

        # the name of the original indices is not changed, therefore append the delay indices to
        return cols.tolist() + list(itertools.chain(*expand()))

    def _expand_single_dela_column(self, cols, delay_idx):
            return list(map(lambda q: "d".join([q, str(delay_idx)]), cols))

    def _setup_delayed_timeseries_collection(self, tsc):
        data = np.zeros([tsc.shape[0], tsc.shape[1] * (self.delays + 1)])
        columns = self._expand_all_delay_columns(tsc.columns)

        delayed_timeseries = TSCDataFrame(pd.DataFrame(data, tsc.index, columns),
                                            qoi_name="time_delayed_qoi")

        delayed_timeseries.loc[:, tsc.columns] = tsc
        return delayed_timeseries

    def _shift_timeseries(self, timeseries, delay_idx):
        if self.time_direction == "backward":
            shifted_timeseries = timeseries.shift(delay_idx, fill_value=np.nan).copy()
        elif self.time_direction == "forward":
            shifted_timeseries = timeseries.shift(-1 * delay_idx, fill_value=np.nan).copy()
        else:
            raise ValueError(f"time_direction={self.time_direction} not known.")

        columns = self._expand_single_dela_column(timeseries.columns, delay_idx)
        shifted_timeseries.columns = columns

        return shifted_timeseries

    def apply(self, tsc: TSCDataFrame):

        if (tsc.lengths_time_series <= self.delay_indices.max()).any():
            raise TimeSeriesCollectionError(
                f"Mismatch of delay and time series length. Shortest time series has length "
                f"{np.array(tsc.lengths_time_series).min()} and maximum delay is {self.delay_indices.max()}")

        delayed_tsc = self._setup_delayed_timeseries_collection(tsc)

        for i, ts in tsc.itertimeseries():
            for delay_idx in self.delay_indices:
                shifted_timeseries = self._shift_timeseries(ts, delay_idx)
                delayed_tsc.loc[i, shifted_timeseries.columns] = shifted_timeseries.values

        #delayed_tsc = delayed_tsc.sort_index(axis=1)
        return delayed_tsc

@DeprecationWarning  # TODO: implement if required...
class TimeFiniteDifference(object):

    # TODO: provide longer shifts? This could give some average of slow and fast variables...

    def __init__(self, scheme="centered"):
        self.scheme = scheme

    def _get_shift_negative(self):
        pass

    def _get_shift_positive(self):
        pass

    def _finite_difference_backward(self):
        pass

    def _finite_difference_forward(self, tsc):
        pass

    def _finite_difference_centered(self):
        pass

    def apply(self, tsc: TSCDataFrame):

        for i, traj in tsc:
            pass


if __name__ == "__main__":
    pass
