#!/usr/bin/env python3

from typing import Generator, List, Optional, Tuple, Union

import matplotlib.colors as mclrs
import numpy as np
import pandas as pd

from datafold.utils.datastructure import is_integer

PD_IDX_TYPE = Union[pd.Index, List[str]]
TSC_TIME_TYPE = Union[int, float, np.datetime64]


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
    def not_required_n_timesteps(cls, required_length, actual_length):
        return cls(
            "One or more time series do not meet the required number of samples"
            f" (={required_length}). Got: \n{actual_length}"
        )

    @classmethod
    def not_const_delta_time(cls, actual_delta_time=None):
        msg = "not const delta time"

        if actual_delta_time is not None:
            msg += f"\n {actual_delta_time}"

        return cls(msg)

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

    @classmethod
    def not_min_timesteps(cls, required_n_timesteps, actual_n_timesteps):
        return cls(
            f"the minimum number of required timesteps (={required_n_timesteps}) is not "
            f"met. Got: \n{actual_n_timesteps}"
        )


class _LocHandler(object):
    """Required for overwriting the behavior of TSCDataFrame.loc.
    iloc is currently not supported, see #61"""

    def __init__(self, tsc, method):
        # cast for the request back to a DataFrame
        self.tsc_as_df = pd.DataFrame(tsc)
        self._method = method

    # TODO: best would be to wrap all attributes through getattr
    #  unfortunately this seems not to work with magic functions as __call__ when
    #  they depending on how they are used:
    #    self.loc.__call__ --> prints "got here" and returns the right attribute
    #    self.loc() --> Error cannot be called, has no __call__ function
    # def __getattr__(self, attr, *args, **kwargs):
    #     print("got here")
    #     return getattr(self.tsc_as_df.loc, attr)

    def __call__(self, axis):
        if self._method == "loc":
            return self.tsc_as_df.loc(axis=axis)
        else:
            return self.tsc_as_df.iloc(axis=axis)

    def __getitem__(self, item):
        if self._method == "loc":
            sliced = self.tsc_as_df.loc[item]
        else:
            sliced = self.tsc_as_df.iloc[item]

        _type = type(sliced)

        try:
            # there is no "TSCSeries", so always use TSCDataFrame, even when
            # sliced has only 1 column and is a pd.Series.
            return TSCDataFrame(sliced)
        except AttributeError:
            # Fallback if the sliced is not a valid TSC anymore
            # returns to pd.Series or pd.DataFrame (depending on what the
            # sliced object of a standard pd.DataFrame is).
            return _type(sliced)

    def __setitem__(self, key, value):
        if self._method == "loc":
            self.tsc_as_df.loc[key] = value
        else:
            self.tsc_as_df.iloc[key] = value

        # may raise AttributeError, when after the insertion it is not a valid
        # TSCDataFrame
        return TSCDataFrame(self.tsc_as_df)


class TSCDataFrame(pd.DataFrame):

    IDX_ID_NAME = "ID"  # name used in index of (unique) time series
    IDX_TIME_NAME = "time"
    IDX_QOI_NAME = "qoi"

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
        cls, tensor: np.ndarray, time_series_ids=None, columns=None, time_values=None
    ):

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

        if time_values is None:
            time_values = np.arange(n_timesteps)

        # entire column, repeating for every time series
        col_time_values = np.resize(time_values, n_timeseries * n_timesteps)
        idx = pd.MultiIndex.from_arrays([time_series_ids, col_time_values])

        data = tensor.reshape(n_timeseries * n_timesteps, n_qoi)
        return cls(data=data, index=idx, columns=columns)

    @classmethod
    def from_shift_matrices(
        cls, left_matrix, right_matrix, snapshot_orientation="col", columns=None
    ):
        """Convenience function for case of shift matices (i.e. timeseries with only two
        time values."""

        if snapshot_orientation == "col":
            left_matrix, right_matrix = left_matrix.T, right_matrix.T
        elif snapshot_orientation == "row":
            pass
        else:
            raise ValueError("")

        if left_matrix.shape != right_matrix.shape:
            raise ValueError("")

        n_timeseries, n_qois = left_matrix.shape

        # allocate memory
        # (n_timeseries, n_timesteps, n_qoi,)
        zipped_values = np.zeros([n_timeseries * 2, n_qois])
        zipped_values[0::2, :] = left_matrix.copy()
        zipped_values[1::2, :] = right_matrix.copy()

        id_idx = np.repeat(np.arange(n_timeseries), 2)
        time_values = np.array([0, 1] * n_timeseries)
        index = pd.MultiIndex.from_arrays([id_idx, time_values])

        tsc = cls(data=zipped_values, index=index, columns=columns)
        tsc._validate()
        return tsc

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
        raise NotImplementedError("expanddim is not required for TSCDataFrame")

    def _validate(self) -> bool:

        if self.index.nlevels != 2:
            # must exactly have two levels [ID, time]
            raise AttributeError(
                "index.nlevels =! 1. Index has to be a pd.MultiIndex with two levels. "
                "First level: time series ID. "
                f"Second level: time. Got: {self.index.nlevels}"
            )

        # Insert required index names:
        #  -- Note: this overwrites potential previous names.
        self.index.names = [self.IDX_ID_NAME, self.IDX_TIME_NAME]
        self.columns.name = self.IDX_QOI_NAME

        ids_index = self.index.get_level_values(self.IDX_ID_NAME)
        time_index = self.index.get_level_values(self.IDX_TIME_NAME)

        if self.columns.nlevels != 1:
            # must exactly have two levels [ID, time]
            raise AttributeError(
                f"columns.nlevels =! 1. Columns has to be single level. "
                f"Got: Columns.nlevels={self.columns.nlevels}"
            )

        # just for security:
        self.index: pd.MultiIndex = self.index.remove_unused_levels()

        if ids_index.dtype != np.int:
            # convert to int64 if it is possible to transform without loss
            # (e.g. from 4.0 to 4) else raise Attribute error
            if (ids_index.astype(np.int64) == ids_index).all():
                self.index.set_levels(
                    self.index.levels[0].astype(np.int64),
                    level=self.IDX_ID_NAME,
                    inplace=True,
                )
            else:
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

    @property
    def n_timeseries(self) -> int:
        return len(self.ids)

    @property
    def n_features(self) -> int:
        return self.shape[1]

    @property
    def ids(self) -> pd.Index:
        # update index by removing potentially unused levels
        self.index = self.index.remove_unused_levels()
        return self.index.levels[0]

    @property
    def delta_time(self) -> Union[pd.Series, float]:
        """Returns float if all time deltas are identical, else returns Series."""

        INDEX_NAME = "delta_time"

        if self.is_datetime_index():
            # TODO: are there ways to better deal with timedeltas?
            #  E.g. could cast internally to float64
            # NaT = Not a Time (cmp. to NaN)
            dt_result_series = pd.Series(
                np.timedelta64("NaT"), index=self.ids, name=INDEX_NAME
            )
            dt_result_series = dt_result_series.astype("timedelta64[ns]")
        else:
            dt_result_series = pd.Series(np.nan, index=self.ids, name=INDEX_NAME)

        diff_times = np.diff(self.index.get_level_values(self.IDX_TIME_NAME))
        id_indexer = self.index.get_level_values(self.IDX_ID_NAME)

        for timeseries_id in self.ids:
            deltatimes_id = diff_times[id_indexer.get_indexer_for([timeseries_id])[:-1]]

            if not self.is_datetime_index():
                deltatimes_id = np.around(deltatimes_id, decimals=14)

            unique_deltatimes = np.unique(deltatimes_id)

            if len(unique_deltatimes) == 1:
                dt_result_series[timeseries_id] = unique_deltatimes[0]

        n_different_dts = len(np.unique(dt_result_series))

        if self.n_timeseries == 1 or n_different_dts == 1:
            single_value = dt_result_series.iloc[0]

            if isinstance(single_value, pd.Timedelta):
                # Somehow single_value gets turned into pd.Timedelta when calling .iloc[0]
                single_value = single_value.to_timedelta64()

            return single_value
        else:
            # return series listing delta_time per time series
            return dt_result_series

    @property
    def n_timesteps(self) -> Union[pd.Series, int]:
        n_time_elements = self.index.get_level_values(0).value_counts()
        n_unique_elements = len(np.unique(n_time_elements))

        if self.n_timeseries == 1 or n_unique_elements == 1:
            return int(n_time_elements.iloc[0])
        else:
            n_time_elements.index.name = self.IDX_ID_NAME
            # seems not to be sorted in the first place.
            n_time_elements.sort_index(inplace=True)
            n_time_elements.name = "counts"
            return n_time_elements

    @property
    def loc(self):
        return _LocHandler(self, method="loc")

    # @property
    # def iloc(self):
    #     return LocHandler(self, method="iloc")

    def is_datetime_index(self):
        return self.index.get_level_values(self.IDX_TIME_NAME).dtype.kind == "M"

    def itertimeseries(self) -> Generator[Tuple[int, pd.DataFrame], None, None]:
        for i, ts in self.groupby(level=self.IDX_ID_NAME):
            # cast single time series back to DataFrame
            yield i, pd.DataFrame(ts.loc[i, :])

    def is_equal_length(self) -> bool:
        return len(np.unique(self.n_timesteps)) == 1

    def is_const_delta_time(self) -> bool:
        # If dt is a Series it means it shows "dt per ID" (because it is not constant).
        _dt = self.delta_time

        if isinstance(_dt, pd.Series):
            return False

        return np.isfinite(_dt)

    def is_same_ts_length(self):
        return isinstance(self.n_timesteps, int)

    def is_same_time_values(self) -> bool:

        if self.n_timeseries == 1:
            # trivial case early
            return True

        length_time_series = self.n_timesteps

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
        if not self.is_const_delta_time():
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
            raise ValueError("number of features do not match")

        if not (self.columns == df.columns).all():
            raise ValueError("columns do not match")

        # Add the id to the first level of the MultiIndex
        df.index = pd.MultiIndex.from_arrays(
            [np.ones(df.shape[0], dtype=np.int) * ts_id, df.index]
        )

        # 'self' has to appear first to keep TSCDataFrame type.
        return pd.concat([self, df], sort=False, axis=0)

    def time_interval(self, ts_id=None) -> Tuple[TSC_TIME_TYPE, TSC_TIME_TYPE]:
        """ts_id if None get global (over all time series) min/max value."""

        if ts_id is None:
            time_values = self.time_values()
        else:
            time_values = self.loc[ts_id, :].index

        return time_values.min(), time_values.max()

    def time_values(self, unique_values=False):

        if unique_values:
            self.index = self.index.remove_unused_levels()
            return np.asarray(self.index.levels[1])
        else:
            return np.asarray(self.index.get_level_values(self.IDX_TIME_NAME))

    def time_index_fill(self):
        """Creates an time array over the entire interval and also fills potential
        gaps. Requires const time delta. """

        if not self.is_const_delta_time():
            raise TSCException.not_const_delta_time()

        start, end = self.time_interval()
        return np.arange(
            start, np.nextafter(end, np.finfo(np.float).max), self.delta_time
        )

    def qoi_to_array(self, qoi: str):

        if not self.is_same_time_values():
            raise TSCException.not_same_time_values()

        return np.reshape(
            self.loc[:, qoi].to_numpy(), (self.n_timeseries, self.n_timesteps)
        )

    def select_time_values(self, time_values) -> Union[pd.DataFrame, "TSCDataFrame"]:
        """Returns pd.DataFrame if it is not a legal definition of TSC anymore (e.g.
        only one point for an ID)"""
        idx = pd.IndexSlice
        return self.loc[idx[:, time_values], :]

    def initial_states(self, n_samples=1) -> pd.DataFrame:
        """Returns the initial condition (first state) for each time series as a
        pd.DataFrame (because it no longer a time series).
        """

        if not is_integer(n_samples) or n_samples < 1:
            raise ValueError("n_samples must be an integer and greater or equal to 1.")

        # only larger than 2 because by definition each time series
        # has a minimum of 2 time samples
        if n_samples > 2:
            self.tsc.check_tsc(ensure_min_n_timesteps=n_samples)
        if n_samples == 1:
            _df = pd.DataFrame(self)
        else:
            _df = self

        return _df.groupby(by="ID", axis=0, level=0).head(n=n_samples)

    def final_states(self, n_samples=1):
        if not is_integer(n_samples) or n_samples < 1:
            raise ValueError("")

        # only larger than 2 because by definition each time series
        # has a minimum of 2 time samples
        if n_samples > 2 and not (self.n_timesteps <= n_samples).all():
            raise TSCException.not_required_n_timesteps(
                required_length=n_samples, actual_length=self.n_timesteps
            )

        if n_samples == 1:
            _df = pd.DataFrame(self)
        else:
            _df = self

        return _df.groupby(by="ID", axis=0, level=0).tail(n=n_samples)

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


class InitialCondition(object):
    @classmethod
    def from_array(cls, X: np.ndarray, columns: Union[pd.Index, List[str]]):

        if isinstance(columns, list):
            # qoi name is not enforced for initial conditions
            columns = pd.Index(columns, name=TSCDataFrame.IDX_QOI_NAME)

        if X.ndim == 1:
            # make a "row-matrix"
            X = X[np.newaxis, :]

        index = pd.Index(np.arange(X.shape[0]), name=TSCDataFrame.IDX_ID_NAME)
        ic_df = pd.DataFrame(X, index=index, columns=columns)
        return ic_df

    @classmethod
    def from_tsc(cls, X, n_samples_ic=1):
        """Docu note: It simply extracts the initial states (ignoring the actual start
        times). If the start times and respective times values are important, use
        iter_reconstruct_ic"""

        ic_df = X.initial_states(n_samples=n_samples_ic)

        if n_samples_ic == 1:
            # drop the time column
            ic_df.index = ic_df.index.droplevel(TSCDataFrame.IDX_TIME_NAME)

        return ic_df

    @classmethod
    def iter_reconstruct_ic(cls, X, n_samples_ic=1):
        """Extract initial conditions of time series with same time values.

        This allows to iterate through initial conditions and reconstruct the time
        series."""

        table = X.tsc.time_values_overview()

        if np.isnan(table["delta_time"]).any():
            raise NotImplementedError(
                "Currently, only constant delta times are " "implemented."
            )

        for (_, _, _), df in table.groupby(by=["start", "end", "delta_time"], axis=0):
            grouped_ids = df.index

            grouped_tsc: TSCDataFrame = X.loc[grouped_ids, :]

            initial_states = grouped_tsc.initial_states(n_samples_ic)

            if n_samples_ic == 1:
                # the time index plays no role in for single IC (compared to time
                # series IC)
                initial_states.index = initial_states.index.droplevel(
                    TSCDataFrame.IDX_TIME_NAME
                )

            time_values = grouped_tsc.time_values(unique_values=True)

            if n_samples_ic > 1:
                # adapt the time values to include only the last time sample of the
                # initial_states and all following (used for prediction)
                time_values = time_values[n_samples_ic - 1 :]

            yield initial_states, time_values

    @classmethod
    def _validate_frame(cls, X_ic: pd.DataFrame):

        # INDEX
        # in index.names (n-D) or name (1-D) must be name=ID
        if X_ic.index.nlevels == 1:
            if X_ic.index.name != TSCDataFrame.IDX_ID_NAME:
                raise ValueError(
                    f"The index.name is not '{TSCDataFrame.IDX_ID_NAME}'. "
                    f"Got {X_ic.index.name}"
                )

            _id_index: pd.Index = X_ic.index

        else:  # X_ic.index.nlevels >= 1:
            if TSCDataFrame.IDX_ID_NAME not in X_ic.index.names:
                raise ValueError(
                    f"No index name has required "
                    f"index.name='{TSCDataFrame.IDX_ID_NAME}'."
                    f"Got {X_ic.index.names}"
                )

            _id_index = X_ic.index.get_level_values(TSCDataFrame.IDX_ID_NAME)

        if not _id_index.is_integer():
            raise ValueError(
                f"The index '{TSCDataFrame.IDX_ID_NAME}' must be of type integer. "
                f"Got type {_id_index.dtype}."
            )

        if _id_index.has_duplicates:
            raise ValueError(
                f"The index '{TSCDataFrame.IDX_ID_NAME}' must be unique. "
                f"Duplicates found \n{_id_index.duplicated()}"
            )

        # COLUMNS
        if X_ic.columns.nlevels != 1:
            raise ValueError(
                f"The columns must be single indexed. "
                f"Got columns.nlevels={X_ic.columns.nlevels}"
            )

        if X_ic.columns.has_duplicates:
            raise ValueError(
                f"The columns must be unique. Duplicates found: \n"
                f"{X_ic.columns.duplicated()}"
            )

    @classmethod
    def _validate_tsc(cls, X_ic: TSCDataFrame):

        # all the usual restrictions for TSCDataFrame apply
        assert X_ic._validate()

        X_ic.tsc.check_tsc(
            ensure_same_length=True,
            ensure_const_delta_time=True,
            ensure_same_time_values=True,
        )

    @classmethod
    def validate(cls, X_ic):

        # apply for both
        if X_ic.isnull().any().any():
            raise ValueError(
                "Initial conditions must be finite (i.e. no nan or inf " "values)"
            )

        if isinstance(X_ic, TSCDataFrame):
            # important to have this first (TSCDataFrame is also pd.DataFrame)
            cls._validate_tsc(X_ic)
        elif isinstance(X_ic, pd.DataFrame):
            cls._validate_frame(X_ic)
        else:
            raise TypeError(f"Type {type(X_ic)} not supported for initial conditions.")

        return True


def allocate_time_series_tensor(n_time_series, n_timesteps, n_qoi):
    """
    Allocate time series tensor that complies with TSCDataFrame.from_tensor(...)

    This indexing is for C-aligned arrays index order for "tensor[depth, row, column]"
       1. depth = timeseries (i.e. for respective initial condition)
       2. row = time step [k]
       3. column = qoi

    :param n_time_series: number of time series
    :param n_timesteps: number of time steps per time series
    :param n_qoi: nr of quantity of interest values
    :return: zero-allocated tensor
    """
    return np.zeros([n_time_series, n_timesteps, n_qoi])
