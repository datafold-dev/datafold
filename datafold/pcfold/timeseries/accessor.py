#!/usr/bin/env python3

import itertools
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nptest
import pandas as pd
from scipy.stats import multivariate_normal

from datafold.pcfold.timeseries.collection import TSCDataFrame, TSCException
from datafold.utils.general import is_float, is_integer


@pd.api.extensions.register_dataframe_accessor("tsc")
class TSCAccessor(object):
    """Accessor functions operatoring on TSCDataFrame.

    See `documentation <https://pandas.pydata.org/pandas-docs/stable/development/extending.html?highlight=accessor>`_
    for regular pandas accessors.

    Parameters
    ----------

    tsc_df
        time series collection data to carry out accessor functions on
    """

    def __init__(self, tsc_df: TSCDataFrame):

        # NOTE: cannot call TSCDataFrame(tsc_df) here to transform in case it is a normal
        # DataFrame. This is because the accessor has to know when updating this object.
        if not isinstance(tsc_df, TSCDataFrame):
            raise TypeError(
                "Can use 'tsc' extension only for type TSCDataFrame (convert before)"
            )

        self._tsc_df = tsc_df

    def check_tsc(
        self,
        ensure_all_finite: bool = True,
        ensure_same_length: bool = False,
        ensure_const_delta_time: bool = True,
        ensure_delta_time: Optional[float] = None,
        ensure_same_time_values: bool = False,
        ensure_normalized_time: bool = False,
        ensure_n_timeseries: Optional[int] = None,
        ensure_min_timesteps: Optional[int] = None,
    ) -> TSCDataFrame:
        """Validate time series properties.

        This summarises the single check functions also contained in `TSCAccessor`.

        Parameters
        ----------
        ensure_all_finite
            If True, check if all values are finite (no 'nan' or 'inf' values).

        ensure_same_length
            If True, check if all time series have the same length.

        ensure_const_delta_time
            If True, check that all time series have the same time-delta.

        ensure_delta_time
            If provided, check that time series have required time-delta.

        ensure_same_time_values
            If True, check that all time series share the same time values.

        ensure_normalized_time
            If True, check if the time values are normalized.

        ensure_n_timeseries
            If provided, check if the required number time series are present.
            
        ensure_min_timesteps
            If provided, check if every time series has the required minimum of time
            steps.

        Returns
        -------
        TSCDataFrame
            validated time series collection (without changes)
        """

        # TODO: allow handle_fail="raise | warn | return"?

        if ensure_all_finite:
            self.check_finite()

        if ensure_same_length:
            self.check_timeseries_same_length()

        if ensure_const_delta_time:
            self.check_const_time_delta()

        if ensure_delta_time is not None:
            self.check_required_time_delta(required_time_delta=ensure_delta_time)

        if ensure_same_time_values:
            self.check_timeseries_same_timevalues()

        if ensure_normalized_time:
            self.check_normalized_time()

        if ensure_n_timeseries is not None:
            self.check_required_n_timeseries(required_n_timeseries=ensure_n_timeseries)

        if ensure_min_timesteps is not None:
            self.check_required_min_timesteps(ensure_min_timesteps)

        return self._tsc_df

    def check_finite(self) -> None:
        """Check if all values are finite (i.e. does not contain `nan` or `inf`).
        """
        if not self._tsc_df.is_finite():
            raise TSCException.not_finite()

    def check_timeseries_same_length(self) -> None:
        """Check if time series in the collection have the same length.
        """
        if not self._tsc_df.is_equal_length():
            raise TSCException.not_same_length(
                actual_lengths=self._tsc_df.is_equal_length()
            )

    def check_const_time_delta(self) -> None:
        """Check if all time series have the same time-delta.
        """
        if not self._tsc_df.is_const_delta_time():
            raise TSCException.not_const_delta_time(self._tsc_df.delta_time)

    def check_timeseries_same_timevalues(self) -> None:
        """Check if all time series in the collection share the same time values.
        """
        if not self._tsc_df.is_same_time_values():
            raise TSCException.not_same_time_values()

    def check_normalized_time(self) -> None:
        """Check if time series collection has normalized time.

        See Also
        --------

        :py:meth:`TSCAccessor.normalize_time`

        """
        if not self._tsc_df.is_normalized_time():
            raise TSCException.not_normalized_time()

    def check_required_time_delta(
        self, required_time_delta: Union[pd.Series, float, int]
    ) -> None:
        """Check if time series collection has required time-delta.

        Parameters
        ----------
        required_time_delta
            single value or per time series
        """

        try:
            # this is a better variant than
            # np.asarray(self._tsc_df.delta_time) == np.asarray(required_time_delta)
            # because the shapes can also mismatch
            nptest.assert_array_equal(
                np.asarray(self._tsc_df.delta_time), np.asarray(required_time_delta)
            )
        except AssertionError:
            raise TSCException.not_required_delta_time(
                required_delta_time=required_time_delta,
                actual_delta_time=self._tsc_df.delta_time,
            )

    def check_required_n_timeseries(self, required_n_timeseries: int) -> None:
        """Check if in the collection are exactly the required number of time series.

        Parameters
        ----------
        required_n_timeseries
            value
        """
        if self._tsc_df.n_timeseries != required_n_timeseries:
            raise TSCException.not_required_n_timeseries(
                required_n_timeseries=required_n_timeseries,
                actual_n_timeseries=self._tsc_df.n_timeseries,
            )

    def check_required_min_timesteps(self, required_min_timesteps: int) -> None:
        """Check if all time series in the collection have a minimum number of time steps.

        Parameters
        ----------
        required_min_timesteps
            value
        """
        _n_timesteps = self._tsc_df.n_timesteps
        if (np.asarray(_n_timesteps) < required_min_timesteps).any():
            raise TSCException.not_min_timesteps(
                required_n_timesteps=required_min_timesteps,
                actual_n_timesteps=_n_timesteps,
            )

    def iter_timevalue_window(self, blocksize, offset):

        if not is_integer(blocksize):
            raise TypeError("'blocksize must be of type integer'")

        if not is_integer(offset):
            raise TypeError("'offset must be of type integer'")

        if blocksize <= 0:
            raise ValueError("'blocksize must be positive")

        if offset <= 0:
            raise ValueError("'offset must be positive")

        time_values = self._tsc_df.time_values()
        start = 0
        end = start + blocksize

        while end <= time_values.shape[0]:
            selected_time_values = time_values[start:end]

            start = start + offset
            end = start + blocksize

            yield self._tsc_df.select_time_values(selected_time_values)

    def shift_time(self, shift_t: float):
        """Shift all time values from the time series by a constant value.

        Parameters
        ----------
        shift_t
            positive or negative time shift value

        Returns
        -------
        TSCDataFrame
            same shape as input
        """

        if shift_t == 0:
            return self._tsc_df

        convert_times = self._tsc_df.index.get_level_values(1) + shift_t
        convert_times = pd.Index(convert_times, name=TSCDataFrame.tsc_time_idx_name)

        new_tsc_index = pd.MultiIndex.from_arrays(
            [self._tsc_df.index.get_level_values(0), convert_times]
        )

        self._tsc_df.index = new_tsc_index
        self._tsc_df._validate()

        return self._tsc_df

    def normalize_time(self):
        """Normalize time for time series in the collection.

        Normalized time has the following properties:

        * global time starts at zero (at least one time series has time value 0)
        * time delta is constant 1

        Returns
        -------
        TSCDataFrame
            normalized data with same shape as input

        Raises
        ------
        TSCException
            If time delta between all time series is not constant.

        """

        convert_times = self._tsc_df.index.get_level_values(
            TSCDataFrame.tsc_time_idx_name
        )
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
        convert_times = pd.Index(convert_times, name=TSCDataFrame.tsc_time_idx_name)

        new_tsc_index = pd.MultiIndex.from_arrays(
            [
                self._tsc_df.index.get_level_values(TSCDataFrame.tsc_id_idx_name),
                convert_times,
            ]
        )
        self._tsc_df.index = new_tsc_index
        self._tsc_df._validate()

        assert self._tsc_df.is_normalized_time()

        return self._tsc_df

    def compute_shift_matrices(
        self, snapshot_orientation: str = "col"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes shift matrices from time series data.

        Both shift matrices have same shape with `(n_features, n_snapshots-1)` or
        `(n_snapshots-1, n_features)`, depending on `snapshot_orientation`.

        Parameters
        ----------
        snapshot_orientation
            Orientation of snapshots (system states at time) either in rows ("row") or
            column-wise ("col")

        Returns
        -------

        :class:`numpy.ndarray`
            shift matrix for time steps `(0,1,2,...,N-1)`

        :class:`numpy.ndarray`
            shift matrix for time steps `(1,2,...,N)`

        Raises
        ------
        TSCException
            If time series have no constant time delta.

        See Also
        --------
        :py:class:`DMDFull`

        """

        self.check_const_time_delta()

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

        if snapshot_orientation == "col":
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

            if snapshot_orientation == "col":
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

    def time_values_overview(self) -> pd.DataFrame:
        """Generate table with overview of time values.

        Example of how the table looks:
        .. Comment: generated with https://truben.no/table/

        +----------------+------------+----------+----+
        | Time series ID | start time | end time | dt |
        +================+============+==========+====+
        | 1              | 1          | 10       | 2  |
        +----------------+------------+----------+----+
        | 2              | 1          | 10       | 1  |
        +----------------+------------+----------+----+
        | 3              | 3          | 13       | 3  |
        +----------------+------------+----------+----+

        Returns
        -------
        pandas.DataFrame
            overview
        """

        table = pd.DataFrame(
            [self._tsc_df.time_interval(_id) for _id in self._tsc_df.ids],
            index=self._tsc_df.ids,
            columns=["start", "end"],
        )

        table["delta_time"] = self._tsc_df.delta_time
        return table

    def plot_density2d(
        self,
        time,
        xresolution: int,
        yresolution: int,
        covariance: Optional[np.ndarray] = None,
    ):
        """Plot the density for a given time.

        For this:

          * Take the first two columns of the underlying data frame and interpret them as
            `x` and `y` coordinates.
          * Place Gaussian bells onto these coordinates and sum up the values of the
            corresponding probability density functions (PDF).
          * The PDF must be evaluated on a fine-granular grid.

        Parameters
        ----------
        time
            time value at which to draw the density

        xresolution
            resolution in `x` direction

        yresolution
            resolution in `y` direction

        covariance
            covariance of Gaussian bells

        Returns
        -------
        matplotlib object
            axis handle
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
    for i in dir(TSCAccessor):
        if not i.startswith("_"):
            print(f"   .. automethod:: {i}")
