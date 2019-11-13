#!/usr/bin/env python3

import numbers
from typing import Tuple, Union

import numpy as np
import pandas as pd


class TimeSeriesCollectionError(Exception):
    """Error raised if data structure is not correct."""


class TSCDataFrame(pd.DataFrame):

    ID_NAME = "ID"  # name used in index of (unique) time series

    def __init__(self, *args, **kwargs):
        # TODO: at the moment a sliced object is a Series, not a TSCSeries. Therefore, a single qoi, is also represented
        #  as a DataFrame

        time_name = kwargs.pop("time_name", None)
        qoi_name = kwargs.pop("qoi_name", None)

        # NOTE: do not move this call after other setters "self.attribute = ...". Otherwise, there is an infinite
        # recursion because pandas handles the __getattr__ magic function.
        super(TSCDataFrame, self).__init__(*args, **kwargs)

        if time_name is None:
            self.time_name = "time"
        else:
            self.time_name = time_name

        if time_name is None:
            self.qoi_name = "qoi"
        else:
            self.qoi_name = qoi_name

        self._validate()
        self._insert_missing_index_names()
        self.sort_index(level=[self.ID_NAME, self.time_name], inplace=True)

    @classmethod
    def from_tensor(cls, tensor: np.ndarray, columns, time_index=None):

        # TODO: need to make sure, that if pandas implements from_tensor in pd.DataFrame,
        #   then this needs to be renamed!

        if tensor.ndim != 3:
            raise ValueError("Input tensor has to be of dimension 3. Index (1) denotes the time series, (2) time and "
                             f"(3) the quantity of interest. Got tensor.ndim={tensor.ndim}")

        nr_timeseries, nr_timesteps, nr_qoi = tensor.shape  # depth=time series, row = time, col = qoi

        time_series_ids = np.arange(nr_timeseries).repeat(nr_timesteps)

        if time_index is None:
            time_index = np.arange(nr_timesteps)

        full_time_index = np.resize(time_index, nr_timeseries * nr_timesteps)

        df_index = pd.MultiIndex.from_arrays([time_series_ids, full_time_index])
        data = tensor.reshape(nr_timeseries * nr_timesteps, nr_qoi)
        return cls(pd.DataFrame(data=data, index=df_index, columns=columns))

    @classmethod
    def from_same_indices_as(cls, indices_from: "TSCDataFrame", values: np.ndarray, except_index=None,
                             except_columns=None):

        if except_index is not None and except_columns is not None:
            raise ValueError("'except_index' and 'except_columns' are both given. "
                             "Cannot copy index or column from existing TSCDataFrame if both is excluded.")

        if except_index is None:
            index = indices_from.index
        else:
            index = except_index

        if except_columns is None:
            columns = indices_from.columns
        else:
            columns = except_columns

        return cls(pd.DataFrame(data=values, index=index, columns=columns))

    @classmethod
    def from_single_timeseries(cls, df):
        """Requires only 1-dim index (time). The time series gets the ID=0."""

        if df.index.ndim != 1:
            raise ValueError("Only single time index (without ID) are allowed.")

        df[cls.ID_NAME] = 0
        df.set_index(cls.ID_NAME, append=True, inplace=True)
        df = df.reorder_levels([cls.ID_NAME, df.index.names[0]])

        return cls(df)

    @property
    def _constructor(self):
        return TSCDataFrame

    def _validate(self):
        if self.index.nlevels != 2:
            # must exactly have two levels [ID, time]
            raise AttributeError("index.nlevels =! 1. Index has to be a pd.MultiIndex with two levels. "
                                 "First level: time series ID. "
                                 f"Second level: time. Got: {self.index.nlevels}")

        ids_index = self.index.get_level_values(0)
        time_index = self.index.get_level_values(1)

        if self.columns.nlevels != 1:
            # must exactly have two levels [ID, time]
            raise AttributeError(f"columns.nlevels =! 1. Columns has to be single level. "
                                 f"Got: Columns.nlevels={self.columns.nlevels}")

        if ids_index.dtype != np.int:
            # The ids have to be integer values
            raise AttributeError("Time series IDs must be of integer value. Got "
                                 f"self.index.get_level_values(0).dtype={self.index.get_level_values(0).dtype}")

        if (ids_index < 0).any():
            unique_ids = np.unique(ids_index)
            unique_negative_ids = unique_ids[unique_ids < 0]
            raise AttributeError(f"All time series IDs have to be positive integer values. Got time series ids:"
                                 f"{unique_negative_ids}")

        if time_index.dtype.kind in 'OSU':
            raise AttributeError(f"time_index has not a numeric dype. Got time_index.dtype={time_index.dtype}")

        if (time_index < 0).any():
            raise AttributeError("All time values have to be non-negative. Found values "
                                 f"{(time_index[time_index < 0])}")

        if not (isinstance(self.time_name, str) and isinstance(self.qoi_name, str)):
            raise AttributeError("Attribute time_name and qoi_name have to be of type str. "
                                 f"type(self.time_name)={type(self.time_name)} and "
                                 f"type(self.qoi_name)={type(self.qoi_name)}")

        #if pd.isnull(self.values).any():
        #    raise AttributeError("Contains invalid values (nan or inf).")

        if self.index.names is not None and not (isinstance(self.index.names, list) and len(self.index.names) == 2):
            raise AttributeError("Provided self.index.names is invalid. Has to be a list of length 2. "
                                 f"Got self.index.names={self.index.names}")

        if (ids_index.value_counts() <= 1).any():
            unique_ids, ids_time_count = np.unique(ids_index, return_counts=True)
            mask_short_ids = ids_time_count <= 1
            raise AttributeError("The minimum length of a time series is 2. Some IDs have less entries:"
                                 f"time series ids={unique_ids[mask_short_ids]} with respective lengths "
                                 f"{ids_time_count[mask_short_ids]}")

        if self.index.duplicated().any():
            raise AttributeError(f"Duplicated indices found: {self.index[self.index.duplicated()].to_numpy()}")

        return True

    def _insert_missing_index_names(self):

        if self.index.names is None or self.index.names[1] != self.time_name:
            self.index.names = [self.ID_NAME, self.time_name]

        if self.columns.name is None or self.columns.name != self.qoi_name:
            self.columns.name = self.qoi_name

    @property
    def nr_timeseries(self) -> int:
        return len(self.ids)

    @property
    def nr_qoi(self) -> int:
        return self.shape[1]

    @property
    def ids(self) -> np.ndarray:
        self.index = self.index.remove_unused_levels()
        return self.index.levels[0]

    @property
    def lengths_time_series(self) -> Union[pd.Series, int]:
        nr_time_elements = self.index.get_level_values(0).value_counts()
        nr_unique_elements = len(np.unique(nr_time_elements))

        if nr_unique_elements == 1:
            return int(nr_time_elements.iloc[0])
        else:
            nr_time_elements.index.name = self.ID_NAME
            nr_time_elements.sort_index(inplace=True)  # seems not to be sorted in the first place.
            nr_time_elements.name = "counts"
            return nr_time_elements

    @property
    def loc(self):
        class LocHandler:
            def __init__(self, tsc):
                self.tsc_as_df = pd.DataFrame(tsc)  # cast for the request back to a DataFrame

            def __getitem__(self, item):
                sliced = self.tsc_as_df.loc[item]
                _type = type(sliced)

                try:
                    # TODO: at the moment there is no TSCSeries, so always use TSCDataFrame, even when sliced is a
                    #  pd.Series
                    return TSCDataFrame(sliced)
                except AttributeError:
                    # returns to pd.Series or pd.DataFrame (depending on what the sliced object of an pd.DataFrame is)
                    return _type(sliced)

            def __setitem__(self, key, value):
                self.tsc_as_df.loc[key] = value
                # may raise AttributeError, when after the insertion it is not a valid TSCDataFrame
                return TSCDataFrame(self.tsc_as_df)

        return LocHandler(self)

    @property
    def dt(self) -> Union[pd.Series, float]:
        """Returns float if all time deltas are identical, else returns Series."""

        dt = pd.Series(np.nan, index=self.ids, name="dt")

        for i, ts in self.itertimeseries():
            if ts.shape[0] == 1:
                raise TimeSeriesCollectionError("Cannot compute time delta because time series of length 1 exist.")

            # the rounding to 15 decimals (~ double precision) is required as diff can create numerical noise which
            # can result in a larger set of "unique values"
            time_diffs = np.unique(np.around(np.diff(ts.index), decimals=15))
            if len(time_diffs) == 1:
                dt.loc[i] = time_diffs[0]

        nr_different_dt = len(np.unique(dt))

        if nr_different_dt == 1:
            return float(dt.iloc[0])
        else:
            return dt

    def itertimeseries(self):
        for i, ts in self.groupby(level=self.ID_NAME):
            # cast single time series back to DataFrame
            yield (i, pd.DataFrame(ts.loc[i, :]))

    def to_pcmanifold(self):
        raise NotImplementedError("To implement")

    def is_equal_length(self) -> bool:
        return len(np.unique(self.lengths_time_series)) == 1

    def is_const_dt(self) -> bool:
        return isinstance(self.dt, float)

    def is_same_ts_length(self):
        return isinstance(self.lengths_time_series, int)

    def is_equal_time_index(self) -> bool:
        check_series = self.reset_index(level=1).loc[:, self.time_name]

        if not self.is_equal_length():
            return False

        time_values = None

        for i, current_time_values in check_series.groupby(self.ID_NAME):
            if time_values is None:
                time_values = current_time_values.to_numpy()
            else:
                if not np.array_equal(current_time_values.to_numpy(), time_values):
                    return False
                time_values = current_time_values.to_numpy()
        return True

    def is_normalized_time(self):
        """Normalized time is defined as:
        * first time record is zero, in any of the time series
        * constant time delta of 1"""
        if not self.is_const_dt():
            return False
        return self.time_interval()[0] == 0 and self.dt == 1

    def is_contain_nans(self):
        return np.any(np.isnan(self))

    def insert_ts(self, df, ts_id=None):
        if ts_id is None:
            ts_id = self.ids.max() + 1  # is unique and positive

        if ts_id in self.ids:
            raise ValueError(f"ID {ts_id} already present.")

        if not isinstance(ts_id, numbers.Integral):
            raise ValueError("ts_id has to be an integer not present in collection already.")

        if self.nr_qoi != df.shape[1]:
            raise ValueError("TODO: Write error")  # TODO

        if not (self.columns == df.columns).all():
            raise ValueError("TODO: Write error")  # TODO

        # Add the id to the first level of the MultiIndex
        df.index = pd.MultiIndex.from_arrays([np.ones(df.shape[0], dtype=np.int)*ts_id, df.index])

        # This call also if everything else is valid.
        self = pd.concat([self, df], sort=False, axis=0)  # self has to appear first to keep TSCDataFrame type.
        return self

    def time_interval(self, ts_id=None) -> Tuple[int, int]:
        """ts_id if None get global (over all time series) min/max value."""

        if ts_id is None:
            time_values = self.time_indices()
        else:
            time_values = self.loc[ts_id, :].index

        return time_values.min(), time_values.max()

    def time_indices(self, require_const_dt=False, unique_values=False):

        time_indices = self.index.levels[1].to_numpy()

        if require_const_dt:
            if not self.is_const_dt():
                raise TimeSeriesCollectionError

        if unique_values:
            return np.unique(time_indices)
        else:
            return time_indices

    def time_index_fill(self):
        """Creates an time array over the entire interval and also fills potential gaps. Requires const time delta. """

        if not self.is_const_dt():
            raise TimeSeriesCollectionError("Time delta is required to be constant.")

        start, end = self.time_interval()
        return np.arange(start, np.nextafter(end, np.finfo(np.float).max), self.dt)

    def qoi_to_ndarray(self, qoi: str):

        if not self.is_equal_time_index():
            raise TimeSeriesCollectionError("The time series' time index are not the same for all time series.")

        return np.reshape(self.loc[:, qoi].to_numpy(), (self.nr_timeseries, self.lengths_time_series))

    def single_timeindex_df(self, time_index: int):
        """Extract from each time series a single time series index."""

        # TODO: if required: do also for 'time' instead of 'index'

        points_df = pd.DataFrame(np.nan, index=self.ids, columns=self.columns)

        # later set for index, but here is more convenient to have it as regular column
        time_column = self.index.names[1]

        # # Illegal value for now is the smallest (negative) integer possible, The problem with using "nan" is that the
        # # time index would internally be set float (while time is integer).
        points_df[time_column] = np.iinfo(np.int).min

        for i, ts in self.itertimeseries():
            points_df.loc[i, self.columns] = ts.iloc[time_index, :]
            points_df.loc[i, time_column] = ts.index[time_index]

        assert np.any(points_df[time_column]) != np.iinfo(np.int).min

        points_df = points_df.set_index(keys=time_column, append=True)
        return points_df

    def single_time_df(self, time):
        """Extract from each time series the row for time. If there is no corresponding entry for 'time', then the
         time series is skipped. If no time series has an entry for time, then an KeyError is raised."""
        idx = pd.IndexSlice

        # cast to DataFrame first, because this time series has to be at least of length 2
        return pd.DataFrame(self).loc[idx[:, time], :]

    def initial_states_df(self):
        df = self.single_timeindex_df(0)
        df.index.names = [self.ID_NAME, "_".join(["initial", self.time_name])]
        return df

    def final_states_df(self):
        df = self.single_timeindex_df(-1)
        df.index.names = [self.ID_NAME, "_".join(["final", self.time_name])]
        return df


if __name__ == "__main__":

    idx = pd.MultiIndex.from_arrays([[0, 0, 1, 1, 55], [0, 1, 0, 1, 99]])
    idx.name = "time"
    col = ["A", "B"]
    df = pd.DataFrame(np.random.rand(5, 2), index=idx, columns=col)

    ts = TSCDataFrame(df)
    print(ts)
