#!/usr/bin/env python3

import numbers
from typing import Generator, List, Optional, Tuple, Union

import matplotlib.colors as mclrs
import numpy as np
import pandas as pd

from datafold.utils.datastructure import is_integer

PD_IDX_TYPE = Union[pd.Index, List[str]]


class TSCException(Exception):
    """Error raised if TSC is not correct."""

    def __init__(self, message):
        super(TSCException, self).__init__(message)

    @classmethod
    def not_finite(cls):
        return cls(f"values are not finite (nan or inf values present)")

    @classmethod
    def not_same_length(cls, actual_lengths):
        return cls(f"time series have not the same length. Got {actual_lengths}")

    @classmethod
    def not_required_length(cls, required_length, actual_length):
        return cls(
            f"time series have not required length {required_length}. Got: {actual_length}"
        )

    @classmethod
    def not_const_delta_time(cls, actual_delta_time):
        return cls(f"not const delta time \n {actual_delta_time}")

    @classmethod
    def not_required_delta_time(cls, required_delta_time, actual_delta_time):
        return cls(
            f"required delta_time (={required_delta_time}) does not met actual delta "
            f"time {actual_delta_time}"
        )

    @classmethod
    def not_same_time_values(cls):
        return cls("time series have not the same time values")

    @classmethod
    def not_normalized_time(cls):
        return cls("time is not normalized")

    @classmethod
    def not_required_n_timeseries(cls, required_n_timeseries, actual_n_timeseries):
        return cls(
            f"there are {actual_n_timeseries} time series present. "
            f"Required: {required_n_timeseries}"
        )


class TSCDataFrame(pd.DataFrame):

    IDX_ID_NAME = "ID"  # name used in index of (unique) time series
    IDX_TIME_NAME = "time"
    IDX_QOI_NAME = "qoi"

    FLOAT64_TIME_PRECISION_DECIMALS = 15

    def __init__(self, *args, **kwargs):
        # TODO: at the moment a sliced object is a Series, not a TSCSeries. Therefore,
        #  a single qoi, is also represented as a DataFrame

        # NOTE: do not move this call after other setters "self.attribute = ...".
        # Otherwise, there is an infinite recursion because pandas handles the
        # __getattr__ magic function.
        super(TSCDataFrame, self).__init__(*args, **kwargs)

        self._validate()
        # self._set_time_precision_float64()
        self.sort_index(level=[self.IDX_ID_NAME, self.IDX_TIME_NAME], inplace=True)

    @classmethod
    def from_tensor(
        cls, tensor: np.ndarray, time_series_ids=None, columns=None, time_index=None
    ):

        # TODO: [minor] need to make sure, that if pandas implements from_tensor in
        #  pd.DataFrame, then this needs to be renamed!

        if tensor.ndim != 3:
            raise ValueError(
                "Input tensor has to be of dimension 3. Index (1) denotes the time "
                "series, (2) time and (3) the quantity of interest. "
                f"Got tensor.ndim={tensor.ndim}"
            )

        # depth=time series, row = time, col = qoi
        (n_timeseries, n_timesteps, n_qoi,) = tensor.shape

        if time_series_ids is None:
            time_series_ids = np.arange(n_timeseries).repeat(n_timesteps)
        else:
            if time_series_ids.ndim > 1:
                raise ValueError("parameter time_series_ids has to be 1-dim.")
            time_series_ids = time_series_ids.repeat(n_timesteps)

        if columns is None:
            columns = [f"qoi_{i}" for i in range(n_qoi)]

        if time_index is None:
            time_index = np.arange(n_timesteps)

        full_time_index = np.resize(time_index, n_timeseries * n_timesteps)

        df_index = pd.MultiIndex.from_arrays([time_series_ids, full_time_index])
        data = tensor.reshape(n_timeseries * n_timesteps, n_qoi)
        return cls(data=data, index=df_index, columns=columns)

    @classmethod
    def from_same_indices_as(
        cls,
        indices_from: "TSCDataFrame",
        values,
        except_index: Optional[PD_IDX_TYPE] = None,
        except_columns: Optional[PD_IDX_TYPE] = None,
    ):

        if except_index is not None and except_columns is not None:
            raise ValueError(
                "'except_index' and 'except_columns' are both given. "
                "Cannot copy index or column from existing TSCDataFrame if both is "
                "excluded."
            )

        # view input as array (allows for different input, which is
        # compatible with numpy.ndarray
        values = np.asarray(values)

        if except_index is None:
            index = indices_from.index  # type: ignore  # mypy cannot infer type here
        else:
            index = except_index

        if except_columns is None:
            columns = indices_from.columns
        else:
            columns = except_columns

        return cls(data=values, index=index, columns=columns)

    @classmethod
    def from_single_timeseries(cls, df: pd.DataFrame, ts_id=None):
        """Requires only 1-dim index (time). The time series gets the ID=0."""

        if df.index.ndim != 1:
            raise ValueError("Only single time index (without ID) are allowed.")

        if ts_id is None:
            ts_id = 0

        if isinstance(df, pd.Series):
            if df.name is None:
                df.name = "qoi1"
            df = pd.DataFrame(df)

        df[cls.IDX_ID_NAME] = ts_id
        df.set_index(cls.IDX_ID_NAME, append=True, inplace=True)
        df = df.reorder_levels([cls.IDX_ID_NAME, df.index.names[0]])

        return cls(df)

    @classmethod
    def from_csv(cls, filepath, **kwargs):
        # NOTE: Overwrites the super class method (which is deprecated since version 0.21
        # Here the csv is read from a csv file that was a
        # TSCDataFrame, i.e. tscdf.to_csv("filename.csv") and therefore should be valid.
        df = pd.read_csv(filepath, index_col=[0, 1], header=[0], **kwargs)
        return cls(df)

    @property
    def _constructor(self):
        return TSCDataFrame

    @property
    def _constructor_expanddim(self):
        raise NotImplementedError(
            "Currently, there is no intension to expand the "
            "dimension of a TSCDataFrame"
        )

    def _validate(self):
        if self.index.nlevels != 2:
            # must exactly have two levels [ID, time]
            raise AttributeError(
                "index.nlevels =! 1. Index has to be a pd.MultiIndex with two levels. "
                "First level: time series ID. "
                f"Second level: time. Got: {self.index.nlevels}"
            )

        self._insert_index_names()
        ids_index = self.index.get_level_values(self.IDX_ID_NAME)
        time_index = self.index.get_level_values(self.IDX_TIME_NAME)

        if self.columns.nlevels != 1:
            # must exactly have two levels [ID, time]
            raise AttributeError(
                f"columns.nlevels =! 1. Columns has to be single level. "
                f"Got: Columns.nlevels={self.columns.nlevels}"
            )

        if ids_index.dtype != np.int:
            # The ids have to be integer values
            raise AttributeError(
                "Time series IDs must be of integer value. Got "
                f"self.index.get_level_values(0). "
                f"dtype={self.index.get_level_values(0).dtype}"
            )

        if (ids_index < 0).any():
            unique_ids = np.unique(ids_index)
            unique_negative_ids = unique_ids[unique_ids < 0]
            raise AttributeError(
                f"All time series IDs have to be positive integer values. "
                f"Got time series ids: {unique_negative_ids}"
            )

        if time_index.dtype.kind in "bcmOSUV":
            # See further info for 'kind'-codes:
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.kind.html?highlight=kind#numpy.dtype.kind
            raise AttributeError(
                f"Time values have to be numeric. Got dtype={time_index.dtype}"
            )

        if time_index.dtype.kind in "if" and (time_index < 0).any():
            raise AttributeError(
                "Time values have to be non-negative. Found values "
                f"{(time_index[time_index < 0])}"
            )

        if self.to_numpy().dtype.kind not in "biufc":
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.kind.html?highlight=kind#numpy.dtype.kind
            raise AttributeError("data dtype must be numeric")

        if (ids_index.value_counts() <= 1).any():
            unique_ids, ids_time_count = np.unique(ids_index, return_counts=True)
            mask_short_ids = ids_time_count <= 1
            raise AttributeError(
                "The minimum length of a time series is 2. Some IDs have less entries:"
                f"time series ids={unique_ids[mask_short_ids]} with respective lengths "
                f"{ids_time_count[mask_short_ids]}"
            )

        if self.index.duplicated().any():
            raise AttributeError(
                f"Duplicated indices found: "
                f"{self.index[self.index.duplicated()].to_numpy()}"
            )

        if self.columns.duplicated().any():
            raise AttributeError(
                f"Duplicated columns found: "
                f"{self.columns[self.columns.duplicated()].to_numpy()}"
            )

        return True

    def _set_time_precision_float64(self):
        time_level = 1

        # 1 is time, the call does not work by str here
        current_time_values = self.index.levels[time_level]

        if np.issubdtype(current_time_values, np.floating):
            adapted_time_values = np.around(
                current_time_values, decimals=self.FLOAT64_TIME_PRECISION_DECIMALS
            )
            self.index = self.index.set_levels(adapted_time_values, level=time_level)

    def _insert_index_names(self):
        self.index.names = [self.IDX_ID_NAME, self.IDX_TIME_NAME]
        self.columns.name = self.IDX_QOI_NAME

    @property
    def n_timeseries(self) -> int:
        return len(self.ids)

    @property
    def n_features(self) -> int:
        return self.shape[1]

    @property
    def ids(self) -> np.ndarray:
        self.index = self.index.remove_unused_levels()  # type: ignore
        return self.index.levels[0]

    @property
    def delta_time(self) -> Union[pd.Series, float]:
        """Returns float if all time deltas are identical, else returns Series."""

        INDEX_NAME = "delta_time"

        if self._is_datetime_index():
            # TODO: are there ways to better deal with timedeltas?
            #  E.g. could cast internally to float64
            # NaT = Not a Time (cmp. to NaN)
            dt_series = pd.Series(
                np.timedelta64("NaT"), index=self.ids, name=INDEX_NAME
            )
            dt_series = dt_series.astype("timedelta64[ns]")
        else:
            dt_series = pd.Series(np.nan, index=self.ids, name=INDEX_NAME)

        diff_times = np.diff(self.index.get_level_values(self.IDX_TIME_NAME))
        id_indexer = self.index.get_level_values(self.IDX_ID_NAME)
        for id_ in self.ids:
            id_diff_times = diff_times[id_indexer.get_indexer_for([id_])[:-1]]

            if not self._is_datetime_index():
                id_diff_times = np.around(id_diff_times, decimals=14)
            id_dt = np.unique(id_diff_times)

            if len(id_dt) == 1:
                dt_series[id_] = id_dt[0]

        nr_different_dt = len(np.unique(dt_series))

        if nr_different_dt == 1:
            return dt_series.iloc[0]
        else:
            return dt_series

    @property
    def lengths_time_series(self) -> Union[pd.Series, int]:
        n_time_elements = self.index.get_level_values(0).value_counts()
        nr_unique_elements = len(np.unique(n_time_elements))

        if nr_unique_elements == 1:
            return int(n_time_elements.iloc[0])
        else:
            n_time_elements.index.name = self.IDX_ID_NAME
            n_time_elements.sort_index(
                inplace=True
            )  # seems not to be sorted in the first place.
            n_time_elements.name = "counts"
            return n_time_elements

    @property
    def loc(self):
        class LocHandler:
            def __init__(self, tsc):
                # cast for the request back to a DataFrame
                self.tsc_as_df = pd.DataFrame(tsc)

            # TODO: best would be to wrap all attributes through getattr
            #  unfortunately this seems not to work with magic functions as __call__ when
            #  they depending on how they are used:
            #    self.loc.__call__ --> prints "got here" and returns the right attribute
            #    self.loc() --> Error cannot be called, has no __call__ function
            # def __getattr__(self, attr, *args, **kwargs):
            #     print("got here")
            #     return getattr(self.tsc_as_df.loc, attr)

            def __call__(self, axis):
                return self.tsc_as_df.loc(axis=axis)

            def __getitem__(self, item):
                sliced = self.tsc_as_df.loc[item]
                _type = type(sliced)

                try:
                    # TODO: at the moment there is no TSCSeries, so always use
                    #  TSCDataFrame, even when sliced is a pd.Series
                    return TSCDataFrame(sliced)
                except AttributeError:
                    # Fallback if the sliced is not a valid TSC anymore
                    # returns to pd.Series or pd.DataFrame (depending on what the
                    # sliced object of a standard pd.DataFrame is).
                    return _type(sliced)

            def __setitem__(self, key, value):
                self.tsc_as_df.loc[key] = value
                # may raise AttributeError, when after the insertion it is not a valid
                # TSCDataFrame
                return TSCDataFrame(self.tsc_as_df)

        return LocHandler(self)

    def _is_datetime_index(self):
        return self.index.get_level_values(1).dtype.kind == "M"

    def itertimeseries(self) -> Generator[Tuple[int, pd.DataFrame], None, None]:
        for i, ts in self.groupby(level=self.IDX_ID_NAME):
            # cast single time series back to DataFrame
            yield (i, pd.DataFrame(ts.loc[i, :]))

    def to_pcmanifold(self):
        raise NotImplementedError("To implement")

    def is_equal_length(self) -> bool:
        return len(np.unique(self.lengths_time_series)) == 1

    def is_const_dt(self) -> bool:
        # If dt is a Series it means it shows "dt per ID" (because it is not constant).
        return not isinstance(self.delta_time, pd.Series)

    def is_same_ts_length(self):
        return isinstance(self.lengths_time_series, int)

    def is_same_time_values(self) -> bool:

        length_time_series = self.lengths_time_series

        if isinstance(length_time_series, pd.Series):
            return False
        else:
            # Check:
            # If every time series is as long as all (unique) index levels, then they
            # are all the same.

            # This call is important, as the levels are usually not updated (even if
            # they not appear in in the index).
            # See: https://stackoverflow.com/a/43824296
            self.index = self.index.remove_unused_levels()
            n_time_level_values = len(self.index.levels[1])
            return length_time_series == n_time_level_values

    def is_normalized_time(self):
        """Normalized time is defined as:
        * first time record is zero, in any of the time series
        * constant time delta of 1"""
        if not self.is_const_dt():
            return False
        return self.time_interval()[0] == 0 and self.delta_time == 1

    def is_finite(self):
        return np.isfinite(self).all().all()

    def insert_ts(self, df: pd.DataFrame, ts_id=None):
        if ts_id is None:
            ts_id = self.ids.max() + 1  # is unique and positive

        if ts_id in self.ids:
            raise ValueError(f"ID {ts_id} already present.")

        if not is_integer(ts_id):
            raise ValueError(f"ts_id has to be an integer type. Got={type(ts_id)}.")

        if self.n_features != df.shape[1]:
            raise ValueError("TODO: Write error")  # TODO

        if not (self.columns == df.columns).all():
            raise ValueError("TODO: Write error")  # TODO

        # Add the id to the first level of the MultiIndex
        df.index = pd.MultiIndex.from_arrays(
            [np.ones(df.shape[0], dtype=np.int) * ts_id, df.index]
        )

        # 'self' has to appear first to keep TSCDataFrame type.
        return pd.concat([self, df], sort=False, axis=0)

    def time_interval(self, ts_id=None) -> Tuple[int, int]:
        """ts_id if None get global (over all time series) min/max value."""

        if ts_id is None:
            time_values = self.time_values()
        else:
            time_values = self.loc[ts_id, :].index

        return time_values.min(), time_values.max()

    def time_values(self, unique_values=False):

        if unique_values:
            self.index = self.index.remove_unused_levels()
            return self.index.levels[1]
        else:
            return self.index.get_level_values(self.IDX_TIME_NAME)

    def time_index_fill(self):
        """Creates an time array over the entire interval and also fills potential
        gaps. Requires const time delta. """

        if not self.is_const_dt():
            raise TSCException("Time delta is required to be constant.")

        start, end = self.time_interval()
        return np.arange(
            start, np.nextafter(end, np.finfo(np.float).max), self.delta_time
        )

    def qoi_to_ndarray(self, qoi: str):

        if not self.is_same_time_values():
            raise TSCException(
                "The time series' time index are not the same for all time series."
            )

        return np.reshape(
            self.loc[:, qoi].to_numpy(), (self.n_timeseries, self.lengths_time_series)
        )

    def single_timeindex_df(self, time_index: int):
        """Extract from each time series a single time series index."""

        (_id_codes, _idx_codes) = self.index.codes
        unique_codes = np.unique(_id_codes)

        idx_pos_first = np.zeros(len(self.ids))

        for i, code_ in enumerate(unique_codes):
            idx_pos_first[i] = np.argwhere(_id_codes == code_)[time_index]

        points_df = pd.DataFrame(self).iloc[idx_pos_first, :].copy()

        return points_df

    def select_times_values(self, time_values) -> Union[pd.DataFrame, "TSCDataFrame"]:
        """Returns pd.DataFrame if it is not a legal definition of TSC anymore (e.g.
        only one point for an ID)"""
        idx = pd.IndexSlice
        return self.loc[idx[:, time_values], :]

    def initial_states_df(self) -> pd.DataFrame:
        """Returns the initial condition (first state) for each time series as a
        pd.DataFrame (because it no longer a time series).
        """
        return self.single_timeindex_df(0)

    def final_states_df(self):
        return self.single_timeindex_df(-1)

    def plot(self, **kwargs):
        ax = kwargs.pop("ax", None)
        legend = kwargs.pop("legend", True)
        qoi_colors = None

        first = True

        for i, ts in self.itertimeseries():
            kwargs["ax"] = ax

            if first:
                ax = ts.plot(legend=legend, **kwargs)
                qoi_colors = [
                    mclrs.to_rgba(ax.lines[j].get_c()) for j in range(self.n_features)
                ]
                first = False
            else:
                ax = ts.plot(color=qoi_colors, legend=False, **kwargs)

        return ax


def allocate_time_series_tensor(nr_time_series, nr_timesteps, nr_qoi):
    """
    Allocate time series tensor that complies with TSCDataFrame.from_tensor(...)

    This indexing is for C-aligned arrays index order for "tensor[depth, row, column]"
       1. depth = timeseries (i.e. for respective initial condition)
       2. row = time step [k]
       3. column = qoi

    :param nr_time_series: number of time series
    :param nr_timesteps: number of time steps per time series
    :param nr_qoi: nr of quantity of interest values
    :return: zero-allocated tensor
    """
    return np.zeros([nr_time_series, nr_timesteps, nr_qoi])


if __name__ == "__main__":
    pass
