from typing import Generator, List, Optional, Tuple, Union

import matplotlib.colors as mclrs
import numpy as np
import pandas as pd
import scipy.sparse
from pandas.core.indexing import _iLocIndexer, _LocIndexer

from datafold.pcfold.distance import compute_distance_matrix
from datafold.utils.general import (
    df_type_and_indices_from,
    is_df_same_index,
    is_integer,
)

ColumnType = Union[pd.Index, List[str]]
NumericalTimeType = Union[int, float, np.datetime64]


class TSCException(Exception):
    """Error raised if TSC is not correct."""

    def __init__(self, message):
        super(TSCException, self).__init__(message)

    @classmethod
    def not_finite(cls):
        return cls(f"values are not finite (nan or inf values present)")

    @classmethod
    def not_same_length(cls, actual_lengths):
        return cls(f"Time series have not the same length. Got {actual_lengths}")

    @classmethod
    def not_required_n_timesteps(cls, required_length, actual_length):
        return cls(
            "One or more time series do not meet the minimum number of required time "
            f"samples. Required {required_length} time values but got \n{actual_length}."
        )

    @classmethod
    def not_const_delta_time(cls, actual_delta_time=None):
        msg = "The time sampling ('delta_time') is not constant."

        if actual_delta_time is not None:
            msg += f"\n {actual_delta_time}"

        return cls(msg)

    @classmethod
    def not_required_delta_time(cls, required_delta_time, actual_delta_time):
        return cls(
            f"The required delta_time (={required_delta_time}) does not match the "
            f"actual delta time {actual_delta_time}."
        )

    @classmethod
    def not_same_time_values(cls):
        return cls("The time series have not the same time values.")

    @classmethod
    def not_normalized_time(cls):
        return cls("The time values are not normalized.")

    @classmethod
    def not_required_n_timeseries(cls, required_n_timeseries, actual_n_timeseries):
        return cls(
            f"There are {actual_n_timeseries} time series present. "
            f"Required: {required_n_timeseries}"
        )

    @classmethod
    def not_min_timesteps(cls, required_n_timesteps, actual_n_timesteps):
        return cls(
            f"The minimum number of required timesteps (={required_n_timesteps}) is not "
            f"met. Got: \n{actual_n_timesteps}"
        )

    @classmethod
    def not_n_timesteps(cls, required: int):
        return cls(
            f"Invalid format: The time series collection has the requirement of"
            f" {required} samples per time series."
        )

    @classmethod
    def has_degenerate_ts(cls):
        return cls(
            "One or more time series are degenerated (i.e. have only a single sample)."
        )

    @classmethod
    def no_kernel(cls):
        return cls("No kernel is set.")


class _LocTSCIndexer(_LocIndexer):
    """Required for overwriting the behavior of :meth:`TSCDataFrame.loc`.
    `iloc` is currently not supported, see #61"""

    def __getitem__(self, item):
        sliced = super(_LocTSCIndexer, self).__getitem__(item)
        _type = type(sliced)

        try:
            # there is no "TSCSeries", so always use TSCDataFrame, even when
            # sliced has only 1 column and is a pd.Series.
            if isinstance(sliced, pd.Series):
                # raises attribute error if not possible
                sliced = TSCDataFrame(sliced)
            else:
                sliced._validate()

        except AttributeError:
            # Fallback if the sliced is not a valid TSC anymore
            # returns to pd.Series or pd.DataFrame (depending on what the
            # sliced object of a standard pd.DataFrame is).
            if _type == TSCDataFrame:
                sliced = pd.DataFrame(sliced)

        return sliced

    def __setitem__(self, key, value):
        set_obj = super(_LocTSCIndexer, self).__setitem__(key, value)

        try:
            # there is no "TSCSeries", so always use TSCDataFrame, even when
            # sliced has only 1 column and is a pd.Series.
            set_obj._validate()
        except AttributeError:
            # Fallback if the sliced is not a valid TSC anymore
            # returns to pd.Series or pd.DataFrame (depending on what the
            # sliced object of a standard pd.DataFrame is).
            set_obj = pd.DataFrame(set_obj)
        return set_obj


class _iLocTSCIndexer(_iLocIndexer):
    def __getitem__(self, item):
        sliced = super(_iLocIndexer, self).__getitem__(item)
        _type = type(sliced)

        try:
            # NOTE: this is different to loc -- here a pd.Series remains a pd.Series!
            # This is because of pandas internal testing
            if isinstance(sliced, pd.Series):
                # TODO: see gitlab issue #61
                #  https://gitlab.com/datafold-dev/datafold/-/issues/61
                raise AttributeError
            else:
                sliced._validate()

        except AttributeError:
            # Fallback if the sliced is not a valid TSC anymore
            # returns to pd.Series or pd.DataFrame (depending on what the
            # sliced object of a standard pd.DataFrame is).
            if _type == TSCDataFrame:
                sliced = pd.DataFrame(sliced)

        return sliced


class TSCDataFrame(pd.DataFrame):
    """Data frame to represent collections of time series data.

    The class inherits from pandas' data structure :class:`pandas.DataFrame` and provides
    additional methods to manipulate or analyse the time series collection. The main
    restrictions on a more general ``pandas.DataFrame`` are:

    * two-dimensional index, where the first index indicates the time series ID (
      integer), and the second the time (non-negative numerical values)
    * one-dimensional columns for feature names
    * neither the index nor in column allows duplicates

    A time series are usually two or more samples. However, the data structure can also
    store degenerated time series with only one time sample. This is useful to define
    initial conditions.

    Please visit the Pandas
    `documentation <https://pandas.pydata.org/pandas-docs/stable/index.html>`__ for
    inherited attributes, methods and other algorithms that can in general act on data
    frames.

    .. note::
        Because Pandas provides a large variety of functionality of its data
        structures, not all methods are tested to check if they comply with
        `TSCDataFrame`. Please report unexpected behavior, bugs or inconsistencies by
        `opening an issue <https://gitlab.com/datafold-dev/datafold/-/issues>`__.

    .. warning::
        Currently, there is no `TSCSeries`, which results in inconsistencies when
        using `iloc`. Even if the result is still a valid `TSCDataFrame` the type
        changes to `pandas.DataFrame` or `pandas.Series`.

    Examples
    --------

    A data frame structure with four time series with two features (columns). The first
    three share the same time values (rows). The fourth time series is a so-called
    degenerated time series because it only consists of a single sample.

    +-------------+---------------+-----------+-----------+
    | feature     |               | sin       | cos       |
    +=============+===============+===========+===========+
    | **ID**      | **time**      |           |           |
    +-------------+---------------+-----------+-----------+
    | 0           | 0.000000      | 0.000000  | 1.000000  |
    +-------------+---------------+-----------+-----------+
    |             | 0.063467      | 0.095056  | 0.995472  |
    +-------------+---------------+-----------+-----------+
    |             | 0.126933      | 0.189251  | 0.981929  |
    +-------------+---------------+-----------+-----------+
    | 1           | 0.000000      | 0.000000  | 1.000000  |
    +-------------+---------------+-----------+-----------+
    |             | 0.063467      | 0.189251  |  0.981929 |
    +-------------+---------------+-----------+-----------+
    |             | 0.126933      | 0.371662  | 0.928368  |
    +-------------+---------------+-----------+-----------+
    | 2           | 0.000000      | 0.000000  | 1.000000  |
    +-------------+---------------+-----------+-----------+
    |             | 0.063467      | 0.281733  |  0.959493 |
    +-------------+---------------+-----------+-----------+
    |             | 0.126933      | 0.540641  |  0.841254 |
    +-------------+---------------+-----------+-----------+
    | 3           | 0.000000      | 0.000000  | 1.000000  |
    +-------------+---------------+-----------+-----------+

    Parameters
    ----------
    *args
    **kwargs
        All parameters (`*args` and `**kwargs`) are passed to the superclass
        ``pandas.DataFrame``, except:

        * ``kernel`` :py:class:`.BaseManifoldKernel` - A kernel to describe the
           manifold  of the time series samples. Defaults to ``None``
        * ``dist_kwargs`` :class:`dict` - Keyword arguments passed to
          :py:meth:`.compute_distance_matrix`.

    Attributes
    ----------
    tsc_id_idx_name
        The index name of first index to select a time series.

    tsc_time_idx_name
        The index name of second index to select time.
    
    tsc_feature_col_name
        The name of feature axis (columns).

    kernel
        The kernel to describe the locality between samples. A
        :py:class:`.PCManifoldKernel` takes the samples independently, whereas a
        :py:class:`.TSCManifoldKernel` can include the time information from the
        collection in a kernel to describe the proximity.
    """

    tsc_id_idx_name = "ID"  # name used in index of (unique) time series
    tsc_time_idx_name = "time"
    tsc_feature_col_name = "feature"

    def __init__(self, *args, **kwargs):

        # remove specific attributes from kwargs first into local attributes
        kernel = kwargs.pop("kernel", None)
        dist_kwargs = kwargs.pop("dist_kwargs", dict())

        # NOTE: do not move this call after other setters "self.attribute = ...".
        # Otherwise, there is an infinite recursion because pandas handles the
        # __getattr__ magic function.
        super(TSCDataFrame, self).__init__(*args, **kwargs)

        self.kernel = kernel

        dist_kwargs.setdefault("cut_off", np.inf)
        dist_kwargs.setdefault("kmin", 0)
        dist_kwargs.setdefault("backend", "guess_optimal")
        self.attrs["dist_kwargs"] = dist_kwargs

        self._validate()

    @classmethod
    def from_tensor(
        cls,
        tensor: np.ndarray,
        time_series_ids: Optional[np.ndarray] = None,
        columns: Optional[Union[pd.Index, list]] = None,
        time_values: Optional[np.ndarray] = None,
    ) -> "TSCDataFrame":
        """Initialize time series collection from a three dimensional tensor.

        Parameters
        ----------
        tensor
            The time series data of shape `(n_timeseries, n_timesteps, n_feature,)`.

        time_series_ids
            The IDs of shape `(n_timeseries,)` to assign to respective time
            series. Defaults to `(0,1,2,..., n_timeseries)`.

        columns
            The feature names of shape `(n_feature,)`. Defaults to
            string `feature[0,1,2,..., n_feature]`.

        time_values
            Time values of the time series in the tensor. Defaults to
            `(0,1,2, ..., n_timesteps)`.

        Returns
        -------
        TSCDataFrame
            new instance
        """

        if tensor.ndim != 3:
            raise ValueError(
                "Input tensor has to be of dimension 3. Index (1) denotes the time "
                "series, (2) time and (3) the quantity of interest. "
                f"Got tensor.ndim={tensor.ndim}"
            )

        # depth=time series, row = time, col = feature
        (n_timeseries, n_timesteps, n_feature,) = tensor.shape

        if time_series_ids is None:
            time_series_ids = np.arange(n_timeseries).repeat(n_timesteps)
        else:
            if time_series_ids.ndim != 1:
                raise ValueError("parameter time_series_ids has to be 1-dim.")
            if time_series_ids.shape[0] != n_timeseries:
                raise ValueError(
                    f"len(time_series_ids)={len(time_series_ids)} must be "
                    f"n_timeseries={n_timeseries}."
                )
            if len(np.unique(time_series_ids)) != len(time_series_ids):
                raise ValueError("time_series_ids must be unique")

            time_series_ids = time_series_ids.repeat(n_timesteps)

        if columns is None:
            columns = pd.Index(
                [f"feature{i}" for i in range(n_feature)],
                name=TSCDataFrame.tsc_feature_col_name,
            )

        if time_values is None:
            time_values = np.arange(n_timesteps)

        # entire column, repeating for every time series
        col_time_values = np.resize(time_values, n_timeseries * n_timesteps)
        idx = pd.MultiIndex.from_arrays([time_series_ids, col_time_values])

        data = tensor.reshape(n_timeseries * n_timesteps, n_feature)

        # Sorting index here, handles cases where time values are not sorted in the input.
        data = pd.DataFrame(data=data, index=idx, columns=columns)
        data.sort_index(axis=0, level=0, inplace=True)
        return cls(data)

    @classmethod
    def from_shift_matrices(
        cls,
        left_matrix,
        right_matrix,
        snapshot_orientation: str = "col",
        columns: Optional[Union[pd.Index, list]] = None,
    ) -> "TSCDataFrame":
        """Initialize time series collection from shift matrices.

        Parameters
        ----------
        left_matrix
            Time series values at time "now".

        right_matrix
            Time series values at time "next".

        snapshot_orientation
            Indicate whether the snapshots (states) are in rows ("row") or columns
            ("col").
            
        columns
            Feature names of shape `(n_feature,)`. Defaults to
            `feature[0,1,2,..., n_feature]`.

        Returns
        -------
        TSCDataFrame
            new instance
        """

        snapshot_orientation = snapshot_orientation.lower()

        if snapshot_orientation == "col":
            left_matrix, right_matrix = left_matrix.T, right_matrix.T
        elif snapshot_orientation == "row":
            pass
        else:
            raise ValueError(
                f"snapshot_orientation={snapshot_orientation} not known. "
                f"Choose either 'row' or 'col'"
            )

        if left_matrix.shape != right_matrix.shape:
            raise ValueError(
                "The matrix shapes are not the same. "
                f"left_matrix.shape={left_matrix.shape}"
                f"right_matrix.shape={right_matrix.shape}"
            )

        n_timeseries, n_features = left_matrix.shape

        # allocate memory
        zipped_values = np.zeros([n_timeseries * 2, n_features])
        # copy for safety
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
        values: Union[np.ndarray, pd.DataFrame, "TSCDataFrame"],
        except_index: Optional[ColumnType] = None,
        except_columns: Optional[ColumnType] = None,
    ) -> "TSCDataFrame":
        """Initialize time series collection by using same index or columns from
        another `TSCDataFrame`.

        Parameters
        ----------
        indices_from
            Existing object to copy index and/or columns from.

        values
            Values for new time series collection.

        except_index
            Index for new time series collection (only copy columns). Should not be
            set together with `except_columns`.

        except_columns
            Columns for this time series collection (only copy index). Should not be
            set together with `except_index`.

        Returns
        -------
        TSCDataFrame
            new instance
        """

        if not isinstance(indices_from, TSCDataFrame):
            raise TypeError("'indices_from' must be of type TSCDataFrame")

        return df_type_and_indices_from(
            indices_from=indices_from,
            values=values,
            except_index=except_index,
            except_columns=except_columns,
        )

    @classmethod
    def from_single_timeseries(
        cls, df: Union[pd.Series, pd.DataFrame], ts_id: Optional[int] = None
    ) -> "TSCDataFrame":
        """Initialize a time series collection with single time series.

        This can be used to iteratively extent the collection (e.g. in a loop using
        :py:meth:`TSCDataFrame.insert_ts`).

        Parameters
        ----------
        df
            Time series data with time values in index and features in columns.

        ts_id
            ID of initial time series.

        Returns
        -------
        TSCDataFrame
            new instance
        """

        if df.index.ndim != 1:
            raise ValueError("Only single time index (without ID) are allowed.")

        if ts_id is None:
            ts_id = 0

        if isinstance(df, pd.Series):
            if df.name is None:
                df.name = cls.tsc_feature_col_name
            df = pd.DataFrame(df)

        # The time values must be sorted.
        df.sort_index(axis=0, inplace=True)

        df[cls.tsc_id_idx_name] = ts_id
        df.set_index(cls.tsc_id_idx_name, append=True, inplace=True)
        df = df.reorder_levels([cls.tsc_id_idx_name, df.index.names[0]])

        return cls(df)

    @classmethod
    def from_frame_list(
        cls,
        frame_list: List[pd.DataFrame],
        ts_ids: Optional[Union[np.ndarray, List[int]]] = None,
    ) -> "TSCDataFrame":
        """Initialize a time series collection from a list of time series.

        Parameters
        ----------
        frame_list
            Data frames with :code:`index=time_values` and :code:`columns=feature_names`.

        ts_ids
            Time series IDs for each corresponding time series in the ``frame_list``.

        Returns
        -------
        TSCDataFrame
            new instance
        """

        ref_df = frame_list[0]
        for _df in frame_list[1:]:
            is_df_same_index(
                ref_df, _df, check_index=False, check_column=True, handle="raise"
            )

        if ts_ids is None:
            ts_ids = np.arange(len(frame_list)).astype(np.int)
        else:
            ts_ids = np.asarray(ts_ids)

        assert isinstance(ts_ids, np.ndarray)  # mypy check

        if (
            ts_ids.ndim != 1
            or ts_ids.shape[0] != len(frame_list)
            or len(np.unique(ts_ids)) != len(frame_list)
        ):
            raise ValueError(
                "ts_ids must be unique time series IDs of same length as " "frame_list"
            )

        tsc_list = list()

        for _id, df in zip(ts_ids, frame_list):
            df.index = pd.MultiIndex.from_product([[_id], df.index.to_numpy()])
            tsc_list.append(TSCDataFrame(df))

        tsc = pd.concat(tsc_list, axis=0)

        return cls(tsc)

    @classmethod
    def from_csv(cls, filepath, **kwargs) -> "TSCDataFrame":
        """Initialize time series collection from csv file.

        Parameters
        ----------
        filepath
            The file path to the csv file.

        **kwargs
            keyword arguments handled to
            `pandas.read_csv <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html>`_

        Returns
        -------
        TSCDataFrame
            new instance
        """
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

    def _sort_tsc_index(self):
        self.sort_index(
            level=[self.tsc_id_idx_name, self.tsc_time_idx_name], inplace=True
        )

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
        self.index.names = [self.tsc_id_idx_name, self.tsc_time_idx_name]
        self.columns.name = self.tsc_feature_col_name

        ids_index = self.index.get_level_values(self.tsc_id_idx_name)
        time_index = self.index.get_level_values(self.tsc_time_idx_name)

        if self.columns.nlevels != 1:
            # must exactly have two levels [ID, time]
            raise AttributeError(
                f"Columns must be single level (columns.nlevels == 1).  "
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
                    level=self.tsc_id_idx_name,
                    inplace=True,
                )
            else:
                # The ids have to be integer values
                raise AttributeError(
                    "Time series IDs must be of integer value. Got "
                    f"dtype={self.index.get_level_values(0).dtype}"
                )

        if (ids_index < 0).any():
            unique_ids = np.unique(ids_index)
            unique_negative_ids = unique_ids[unique_ids < 0]
            raise AttributeError(
                f"All time series IDs have to be non-negative integer values. "
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
            raise AttributeError("Data type of time series must be numeric.")

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

        # bool index to the start of new IDs
        bool_new_id = np.append(1, np.diff(self.index.codes[0])).astype(np.bool)
        _ids = self.index.get_level_values(self.tsc_id_idx_name)[bool_new_id]

        if len(np.unique(_ids)) != len(_ids):
            raise AttributeError("Time series IDs appear multiple times.")

        ts_internal_codes = np.append(0, np.diff(self.index.codes[1]))
        if np.any(ts_internal_codes[~bool_new_id] <= 0):
            raise AttributeError("The time values of each time series must be sorted.")

        return True

    @property
    def n_timeseries(self) -> int:
        """Number of time series in the collection.
        """
        return len(self.ids)

    @property
    def n_features(self) -> int:
        """Number of features in the collection.
        """
        return self.shape[1]

    @property
    def ids(self) -> pd.Index:
        """The time series IDs in the collection.
        """
        # update index by removing potentially unused levels
        self.index = self.index.remove_unused_levels()
        return self.index.levels[0]

    @property
    def delta_time(self) -> Union[pd.Series, float]:
        """Time sampling frequency.

        Collects for each time series the time delta. Irregular frequencies are marked
        with `nan`. If all time series are consistent (i.e., all have same time delta,
        including `nan`), then a single float is returned.

        Returns
        -------
        """

        _index_name_series = "delta_time"

        if self.is_datetime_index():
            # TODO: are there ways to better deal with timedeltas?
            #  E.g. could cast internally to float64
            # NaT = Not a Time (cmp. to NaN)
            dt_result_series = pd.Series(
                np.timedelta64("NaT"), index=self.ids, name=_index_name_series
            )
            dt_result_series = dt_result_series.astype("timedelta64[ns]")
        else:
            # initialize all with nan values (which essentially is np.floag64.
            # return back to same dtype than time_values if all values are finite,
            # otherwise the type stays as float.
            dt_result_series = pd.Series(
                np.nan, index=self.ids, name=_index_name_series
            )

        diff_times = np.diff(self.index.get_level_values(self.tsc_time_idx_name))
        id_indexer = self.index.get_level_values(self.tsc_id_idx_name)

        for timeseries_id in self.ids:
            deltatimes_id = diff_times[id_indexer.get_indexer_for([timeseries_id])[:-1]]

            if not self.is_datetime_index():
                # TODO: see gitlab issue #85
                # round differences to machine 1 order below machine precision
                deltatimes_id = np.around(deltatimes_id, decimals=14)

            unique_deltatimes = np.unique(deltatimes_id)

            if len(unique_deltatimes) == 1:
                dt_result_series[timeseries_id] = unique_deltatimes[0]

        if not np.isnan(dt_result_series).all():
            n_different_dts = len(np.unique(dt_result_series))

            if not np.isnan(dt_result_series).any():
                dt_result_series = dt_result_series.astype(diff_times.dtype)

        else:  # all nan (i.e. irregular sampling), treat as all identical sampled
            n_different_dts = 1

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
    def n_timesteps(self) -> Union[int, pd.Series]:
        """Number of time steps per time series.

        Collects for each time series the number of time steps. If all time series are
        consistent (all have same time steps), then a single float is returned.

        Returns
        -------
        """
        n_time_elements = self.index.get_level_values(0).value_counts()
        n_unique_elements = len(np.unique(n_time_elements))

        if self.n_timeseries == 1 or n_unique_elements == 1:
            return int(n_time_elements.iloc[0])
        else:
            n_time_elements.index.name = self.tsc_id_idx_name
            # seems not to be sorted in the first place.
            n_time_elements.sort_index(inplace=True)
            n_time_elements.name = "counts"
            return n_time_elements

    @property
    def kernel(self):
        """The kernel to describe the proximity between samples.

        Returns
        -------

        :py:class:`.BaseManifoldKernel`
        """
        if "kernel" not in self.attrs:
            self.attrs["kernel"] = None

        return self.attrs["kernel"]

    @kernel.setter
    def kernel(self, kernel) -> None:
        """Set a new kernel.

        Parameters
        ----------
        kernel
            The new kernel to be set.

        Returns
        -------

        """

        from datafold.pcfold.kernels import BaseManifoldKernel

        if kernel is not None and not isinstance(kernel, BaseManifoldKernel):
            raise TypeError(
                f"Kernel must be a subclass of type BaseManifoldKernel. Got "
                f"type {type(kernel)}."
            )
        self.attrs["kernel"] = kernel

    @property
    def dist_kwargs(self) -> dict:
        """Keyword arguments passed to the internal distance matrix computation.

        See :py:meth:`datafold.pcfold.compute_distance_matrix` for parameter arguments.

        Returns
        -------

        """
        if "dist_kwargs" not in self.attrs:
            self.attrs["dist_kwargs"] = dict()

        return self.attrs["dist_kwargs"]

    @dist_kwargs.setter
    def dist_kwargs(self, dist_kwargs: Optional[dict]) -> None:
        """Set new keyword arguments for the distance matrix computation.

        Parameters
        ----------
        dist_kwargs
            The new arguments passed to the distance matrix backend. To set the default
            keywords pass ``None``.

        Returns
        -------

        """

        if dist_kwargs is not None and not isinstance(dist_kwargs, dict):
            raise TypeError(
                f"Keyword arguments for distance computation must be a dictionary or "
                f"None. Got type {type(dist_kwargs)}."
            )

        dist_kwargs = {} or dist_kwargs

        assert dist_kwargs is not None  # for mypy

        dist_kwargs.setdefault("cut_off", np.inf)
        dist_kwargs.setdefault("kmin", 0)
        dist_kwargs.setdefault("backend", "guess_optimal")

        self.attrs["dist_kwargs"] = dist_kwargs

    @property
    def loc(self):
        """Label-based indexing.

        Please visit
        `pd.DataFrame.loc <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html#pandas.DataFrame.loc>`_
        for full documentation.

        Returns
        -------

        """
        return _LocTSCIndexer("loc", self)

    @property
    def iloc(self):
        """Index-based indexing.

        Please visit
        `pd.DataFrame.iloc <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html#pandas.DataFrame.iloc>`_
        for full documentation.

        .. warning::
            For single column slices, currently, the type changes to ``pandas.Series``.

        Returns
        -------

        """
        return _iLocTSCIndexer("iloc", self)

    def set_index(
        self, keys, drop=True, append=False, inplace=False, verify_integrity=False,
    ):
        result = super(TSCDataFrame, self).set_index(
            keys=keys,
            drop=drop,
            append=append,
            inplace=inplace,
            verify_integrity=verify_integrity,
        )

        try:
            if inplace:
                self._validate()
            else:
                # calls validate and sorts result
                result = TSCDataFrame(result)
        except AttributeError as e:
            raise e

        return result

    def xs(
        self, key, axis=0, level=None, drop_level: bool = True
    ) -> Union[pd.DataFrame, pd.Series]:
        """Overwrites cross section to provide fall back solution in case the result
        is not a valid ``TSCDataFrame`` anymore.

        Parameters
        ----------
        args
            see docu
            `DataFrame.xs <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.xs.html#pandas.DataFrame.xs>`_

        Returns
        -------
        Union[TSCDataFrame, pandas.DataFrame, pandas.Series]
            Cross section of ``TSCDataFrame``. The type is determined by the resulting
            slice.
        """

        _slice = super(TSCDataFrame, self).xs(
            key, axis=axis, level=level, drop_level=drop_level
        )

        try:
            _slice._validate()
        except AttributeError:
            if isinstance(_slice, TSCDataFrame):
                _slice = pd.DataFrame(_slice)

        return _slice

    def __getitem__(self, key):

        _slice = super(TSCDataFrame, self).__getitem__(key=key)

        try:
            if isinstance(_slice, pd.Series):
                # try to convert to TSCDataFrame
                _slice = TSCDataFrame(_slice)
            else:
                _slice._validate()
        except AttributeError:
            if isinstance(_slice, TSCDataFrame):
                _slice = pd.DataFrame(_slice)

        return _slice

    def is_datetime_index(self) -> bool:
        """Indicates whether 'time' index is datetime format.
        """
        return self.index.get_level_values(self.tsc_time_idx_name).dtype.kind == "M"

    def itertimeseries(self) -> Generator[Tuple[int, pd.DataFrame], None, None]:
        """Generator over time series (id, pandas.DataFrame).

        .. note::
            Each time series is a ``pandas.DataFrame`` (not a ``TSCDataFrame``).

        Yields
        ------
        Tuple[int, pandas.DataFrame]
            Time series ID and corresponding time series.
        """
        for i, ts in self.groupby(level=self.tsc_id_idx_name):
            # cast single time series back to DataFrame
            yield i, pd.DataFrame(ts.loc[i, :])

    def is_equal_length(self) -> bool:
        """Indicates if all time series in the collection have the same number of
        timesteps.
        """
        return len(np.unique(self.n_timesteps)) == 1

    def is_const_delta_time(self) -> bool:
        """Indicates if all time series in the collection have the same time delta.
        """

        # If dt is a Series it means it shows "dt per ID" (because it is not constant).
        _dt = self.delta_time

        if isinstance(_dt, pd.Series):
            return False

        # pd.isnull is better than np.finite because it also checks for
        # NaT (Not a Time[-delta])
        return not pd.isnull(_dt)

    def is_same_time_values(self) -> bool:
        """Indicates if all time series in the collection share the same time values.
        """

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

    def is_normalized_time(self) -> bool:
        """Indicates if the time values are normalized.

        A normalized time has the following properties:

            * the first time record is zero (not necessarily for all time series), and
            * `delta_time` is constant 1 for all time series
        """
        if not self.is_const_delta_time():
            return False
        return self.time_interval()[0] == 0 and self.delta_time == 1

    def is_finite(self) -> bool:
        """Indicates if all feature values are finite (i.e. neither NaN nor inf).
        """
        return np.isfinite(self).all().all()

    def degenerate_ids(self) -> Optional[pd.Index]:
        """Return the degenerate time series IDs.

        Degenerate time series consist only of a single sample.

        Returns
        -------
        """

        n_timesteps = self.n_timesteps
        _ids = None

        if isinstance(n_timesteps, pd.Series):
            _ids = n_timesteps[n_timesteps == 1].index
            if len(_ids) == 0:
                _ids = None
        else:
            if is_integer(n_timesteps) and n_timesteps == 1:
                _ids = self.ids

        return _ids

    def has_degenerate(self) -> bool:
        """Indicates whether degenerate time series are present in the collection.

        Returns
        -------
        """
        return self.degenerate_ids() is not None

    def compute_distance_matrix(
        self, Y: Optional[Union["TSCDataFrame", np.ndarray]] = None, metric="euclidean"
    ) -> Union["TSCDataFrame", np.ndarray]:
        """Compute distance matrix on time series collection.

        Internally calls :py:meth:`datafold.pcfold.distance.compute_distance_matrix`
        and adds time information if possible.

        No time information is added when

           * the query matrix `Y` is a numpy matrix
           * the distance matrix is sparse due to ``dist_kwargs`` settings.

        Parameters
        ----------
        Y
            Query point cloud of shape (n_samples_Y, n_features). If provided, compute
            the distance matrix component-wise, else `Y=self` (pair-wise). For further
            details see also :class:`.DistanceAlgorithm`.

        metric
            Distance metric. The backend algorithm set in ``dist_kwargs`` must support
            the metric.

        Returns
        -------
        Union[np.ndarray, scipy.sparse.csr_matrix, TSCDataFrame]
            Distance matrix.
        """

        if Y is not None:
            is_attach_time = isinstance(Y, pd.DataFrame)
        else:
            is_attach_time = True

        distance_matrix = compute_distance_matrix(
            X=self, Y=Y, metric=metric, **self.dist_kwargs,
        )

        if is_attach_time and not isinstance(distance_matrix, scipy.sparse.spmatrix):
            time_idx_from = self if Y is None else Y

            distance_matrix = df_type_and_indices_from(
                time_idx_from,
                distance_matrix,
                except_columns=[f"X{i}" for i in np.arange(self.shape[0])],
            )

        return distance_matrix

    def compute_kernel_matrix(
        self, Y: Optional[Union[np.ndarray, "TSCDataFrame"]] = None, **kernel_kwargs
    ) -> pd.DataFrame:
        """Compute the kernel matrix with the specified kernel.

        Parameters
        ----------
        Y
            Query point cloud or other time series collection of shape
            `(n_samples_Y, n_features)`. If provided, it computes
            the kernel matrix component-wise, else `Y=self` for a pair-wise kernel
            matrix.


        kernel_kwargs
            Keyword arguments passed passed to the kernel.

        Returns
        -------
        Union[numpy.ndarray, pandas.DataFrame, TSCDataFrame]
            If the kernel is of type :class:`PCManifoldKernel` (i.e. acts on
            point clouds), the time information is added in this routine. If the time
            information. For :class:`.TSCManifoldKernel` (i.e. kernels natively act on
            time series data), the result is directly returned from the kernel.

        Optional
            The specified kernel can return further objects, which are all forwarded
            by this function. See :meth:`PCManifoldKernel.__call__`,
            :meth:`TSCManifoldKernel.__call__` or the respective kernel for details.
        """

        if self.kernel is None:
            raise TSCException.no_kernel()

        if Y is not None:
            is_attach_time = isinstance(Y, pd.DataFrame)
        else:
            is_attach_time = True

        kernel_output = self.kernel(
            X=self, Y=Y, dist_kwargs=self.dist_kwargs, **kernel_kwargs,
        )

        if isinstance(kernel_output, (tuple, list)):
            ty_kernel_matrix = type(kernel_output[0])
        else:
            ty_kernel_matrix = type(kernel_output)

        def _pcmkernel2tsc(kernel_matrix, indices_from):
            return df_type_and_indices_from(
                indices_from,
                kernel_matrix,
                except_columns=[f"X{i}" for i in np.arange(self.shape[0])],
            )

        if is_attach_time and ty_kernel_matrix == np.ndarray:
            time_idx_from = self if Y is None else Y

            if isinstance(kernel_output, (tuple, list)):
                kernel_output = list(kernel_output)
                kernel_output[0] = _pcmkernel2tsc(kernel_output[0], time_idx_from)
                kernel_output = tuple(kernel_output)
            else:
                kernel_output = _pcmkernel2tsc(kernel_output, time_idx_from)

        return kernel_output

    def insert_ts(
        self, df: pd.DataFrame, ts_id: Optional[int] = None
    ) -> "TSCDataFrame":
        """Inserts new time series to current collection.

        Parameters
        ----------
        df
            new time series, with same features (column names)
        ts_id
            Dedicated ID of new time series (must not be present in current collection).

        Returns
        -------
        TSCDataFrame
            self
        """
        if ts_id is None:
            ts_id = self.ids.max() + 1  # is unique and positive

        if ts_id in self.ids:
            raise ValueError(f"ID {ts_id} already present.")

        if not is_integer(ts_id):
            raise ValueError(f"ts_id has to be an integer type. Got={type(ts_id)}.")

        if self.n_features != df.shape[1] or not (self.columns == df.columns).all():
            raise ValueError(
                "Column names do not match. "
                f"Expected \n{self.columns} "
                f"but got \n{df.columns} "
            )

        # Add the id to the first level of the MultiIndex
        df.index = pd.MultiIndex.from_arrays(
            [np.ones(df.shape[0], dtype=np.int) * ts_id, df.index]
        )

        # 'self' has to appear first to keep TSCDataFrame type.
        return pd.concat([self, df.copy(deep=True)], sort=False, axis=0)

    def time_interval(
        self, ts_id: Optional[int] = None
    ) -> Tuple[NumericalTimeType, NumericalTimeType]:
        """Time interval (start, end) for all or single time series in
        the collection.

        Parameters
        ----------
        ts_id
            Time series ID. If not provided, the interval is over all time series
            in the collection.
        """

        if ts_id is None:
            time_values = self.time_values()
        else:
            time_values = self.loc[ts_id, :].index

        return np.min(time_values), np.max(time_values)

    def time_values(self) -> np.ndarray:
        """All time values that appear in at least one time series of the collection.

        Returns
        -------

        """

        self.index = self.index.remove_unused_levels()
        return np.asarray(self.index.levels[1])

    def time_values_delta_time(self) -> np.ndarray:
        """All time values between `(start, end)` interval with constant time
        delta.

        Potential gaps between time series are closed so that a constant delta time is
        maintained in returned time array.

        Returns
        -------
        numpy.ndarray
            time value array

        Raises
        ------
        TSCException
            if `delta_time` is not constant

        """

        if not self.is_const_delta_time():
            raise TSCException.not_const_delta_time()

        start, end = self.time_interval()
        return np.arange(
            start, np.nextafter(end, np.finfo(np.float64).max), self.delta_time
        )

    def feature_to_array(
        self, feature: Optional[str] = None, as_frame: bool = False
    ) -> np.ndarray:
        """Turns a single feature column into a matrix.

        Parameters
        ----------
        feature
            Name of feature to turn into array. The feature name must be provided if
            multiple features are present.
        
        as_frame
            If True, return pandas with time series IDs as index and time values as
            column indices.
        
        Returns
        -------
        numpy.ndarray
            feature values of shape `(n_timeseries, n_timesteps)`

        Raises
        ------
        TSCException
            If time series in the collection have not identical time values.

        """

        if feature is None and self.shape[1] > 1:
            raise ValueError(
                "parameter 'feature' must be provided if there are " "multiple features"
            )
        elif feature is None:
            feature = self.columns[0]

        if not self.is_same_time_values():
            raise TSCException.not_same_time_values()

        array = np.reshape(
            self.loc[:, feature].to_numpy(), (self.n_timeseries, self.n_timesteps)
        )

        if as_frame:
            array = pd.DataFrame(array, index=self.ids, columns=self.time_values())

        return array

    def select_time_values(
        self, time_values: Union[int, float, np.ndarray]
    ) -> Union[pd.DataFrame, "TSCDataFrame"]:
        """Select time values of all time series with a matching time value.

        Parameters
        ----------
        time_values:
            selected values

        Returns
        -------
        Union[pd.DataFrame, "TSCDataFrame"]
            If sliced data frame is not a valid `TSCDataFrame`, the fallback is to
            a ``pandas.DataFrame``.

        """
        idx = pd.IndexSlice
        return self.loc[idx[:, time_values], :]

    def initial_states(self, n_samples: int = 1) -> "TSCDataFrame":
        """Get initial state of each time series in the collection.

        Parameters
        ----------

        n_samples
            Number of samples required for an initial state.

        Returns
        -------
        TSCDataFrame
            Initial states of shape `(n_samples * n_timeseries, n_features)`.

        Raises
        ------
        TSCException
            If there is a time series in the collection that has less time values
            than than the required number of samples.
        """

        if not is_integer(n_samples) or n_samples < 1:
            raise ValueError(
                "The parameter 'n_samples' must be a positive integer value."
            )

        self.tsc.check_required_min_timesteps(required_min_timesteps=n_samples)

        return self.groupby(by="ID", axis=0, level=0).head(n=n_samples)

    def final_states(self, n_samples: int = 1) -> "TSCDataFrame":
        """Get the final states of each time series in the collection.

        Parameters
        ----------
        n_samples
            Number of samples required for the final state.

        Returns
        -------
        TSCDataFrame
            The final states of shape `(n_samples, n_features)`.

        Raises
        ------
        TSCException
            If there is a time series with less than the required number of samples.

        """

        if not is_integer(n_samples) or n_samples < 1:
            raise ValueError("Parameter 'n_samples' must be a positive integer.")

        self.tsc.check_required_min_timesteps(required_min_timesteps=n_samples)

        return self.groupby(by="ID", axis=0, level=0).tail(n=n_samples)

    def plot(self, **kwargs):
        """Plots time series.

        Parameters
        ----------
        **kwargs
            Key word arguments handled to each time series
            `pandas.DataFrame.plot() <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html?highlight=plot#pandas.DataFrame.plot>`_
            call.

        Returns
        -------
        matplotlib object
            axes handle
        """
        ax = kwargs.pop("ax", None)
        legend = kwargs.pop("legend", True)
        color = kwargs.pop("color", None)

        first = True

        for i, ts in self.itertimeseries():
            kwargs["ax"] = ax

            if first:
                ax = ts.plot(legend=legend, **kwargs)
                if color is None:
                    color = [
                        mclrs.to_rgba(ax.lines[j].get_c())
                        for j in range(self.n_features)
                    ]
                first = False
            else:
                ax = ts.plot(color=color, legend=False, **kwargs)

        return ax


class InitialCondition(object):
    """Helper functions to create and validate initial conditions for time series
    predictions.

    Initial conditions are internally described with ``TSCDataFrame`` objects.

    In general, initial conditions are required in models that train on time series
    data, see for example in :py:meth:`EDMD.predict`. An initial condition can consist of
    single states (e.g. a vector at time zero), or a time series itself. The latter is
    the case if model transformations require multiple time values to define the
    transformed state (such as :py:class:`.TSCTakensEmbedding` or
    :py:class:`TSCFiniteDifference`).
    """

    @classmethod
    def from_array(
        cls, X: np.ndarray, columns: Union[pd.Index, List[str]]
    ) -> TSCDataFrame:
        """Build initial conditions object from a NumPy array.

        Note that the assumption is that each row is a new initial condition. All time
        values are set to zero.

        Parameters
        ----------
        X
            Initial condition of shape `(n_ic, n_features)`.

        columns
            Feature names in model during fit (they can be accessed with
            :code:`model_obj.features_in_[1]`.

        Returns
        -------
        TSCDataFrame
            initial condition
        """

        if isinstance(columns, list):
            # feature name is not enforced for initial conditions
            columns = pd.Index(columns, name=TSCDataFrame.tsc_feature_col_name)

        if X.ndim == 1:
            # make a "row-matrix"
            X = X[np.newaxis, :]

        if X.ndim > 2:
            raise ValueError(
                "Cannot convert arrays with dimension larger than 2. Got "
                f"X.ndim={X.ndim}"
            )

        n_ic = X.shape[0]
        index = pd.MultiIndex.from_arrays([np.arange(n_ic), np.zeros(n_ic)])

        ic_df = TSCDataFrame(X, index=index, columns=columns)
        InitialCondition.validate(ic_df, n_samples_ic=1, dt=None)
        return ic_df

    @classmethod
    def from_tsc(cls, X: TSCDataFrame, n_samples_ic: int = 1) -> pd.DataFrame:
        """Extract initial states from a ``TSCDataFrame``.

        Parameters
        ----------
        X
            The time series data to extract initial states from.

        n_samples_ic
            The number of time steps per initial condition.

        Returns
        -------
        TSCDataFrame
            initial condition
        """

        ic_df = X.initial_states(n_samples=n_samples_ic)
        InitialCondition.validate(
            ic_df,
            n_samples_ic=n_samples_ic,
            dt=X.delta_time if n_samples_ic > 1 else None,
        )
        return ic_df

    @classmethod
    def iter_reconstruct_ic(
        cls, X: TSCDataFrame, n_samples_ic: int = 1
    ) -> Generator[Tuple[TSCDataFrame, np.ndarray], None, None]:
        """Extract and iterate over initial conditions over groups of time series that
        have identical time values.

        This iterator is particulary usefule to reconstruct time series.
        
        Parameters
        ----------
        X
            The time series collection to extract initial states from.

        n_samples_ic
            The number of time steps per initial condition.

        Returns
        -------
        iterator Tuple[TSCDataFrame, numpy.ndarray]
            Each iteration returns the initial states for each time series of the group
            and the associate time values.
        """

        X.tsc.check_required_min_timesteps(n_samples_ic)
        time_series_table = X.tsc.time_values_overview()

        if np.isnan(time_series_table["delta_time"]).any():
            raise NotImplementedError(
                "Currently, only constant delta times are " "implemented."
            )

        for (_, _, _), df in time_series_table.groupby(
            by=["start", "end", "delta_time"], axis=0
        ):
            grouped_ids = df.index

            grouped_tsc: TSCDataFrame = X.loc[grouped_ids, :]

            initial_states = grouped_tsc.initial_states(n_samples_ic)
            time_values = grouped_tsc.time_values()

            if n_samples_ic > 1:
                # adapt the time values to include only the last time sample of the
                # initial_states and all following (used for prediction)
                time_values = time_values[n_samples_ic - 1 :]

            yield initial_states, time_values

    @classmethod
    def validate(
        cls,
        X_ic: TSCDataFrame,
        n_samples_ic: Optional[int] = None,
        dt: Optional[float] = None,
    ) -> bool:
        """Validate the initial condition format of a :py:class:`-TSCDataFrame`.

        Parameters
        ----------
        X_ic
            The initial condition to validate.

        n_samples_ic
            If provided, then validate that each initial condition has the set number of
            samples.

        dt
            If provided, then validate that the time series have the set constant delta
            time sampling. The parameter ``n_samples_ic`` must be given at the same time.

        Returns
        -------
        """

        if not isinstance(X_ic, TSCDataFrame):
            raise TypeError(
                "The initial condition to be validated must be of type TSCDataFrame."
            )

        if n_samples_ic is None and dt is not None:
            raise ValueError(
                "If validating delta time, then the parameter 'n_samples_ic' "
                "must be given."
            )

        if n_samples_ic == 1 and dt is not None:
            raise ValueError(
                "Cannot check the time sampling rate, when at the same "
                "time only one sample is required per initial condition."
            )

        # all the usual restrictions for TSCDataFrame apply
        assert X_ic._validate()

        X_ic.tsc.check_tsc(
            ensure_const_delta_time=np.array(X_ic.n_timesteps > 1).any(),
            ensure_no_degenerate_ts=False,
            ensure_same_length=True,
            ensure_all_finite=True,
            ensure_same_time_values=True,
            ensure_n_timesteps=n_samples_ic,
            ensure_delta_time=dt,
        )

        return True


def allocate_time_series_tensor(n_time_series, n_timesteps, n_feature):
    """Allocate a time series tensor that complies with
    :py:meth:`TSCDataFrame.from_tensor()`.

    Allocated three-dimensional ``numpy.ndarray`` is C-aligned and can be accessed with

    .. code::

        tensor[timeseries, time step, feature]

    Parameters
    ----------
    n_time_series
        The number of time series.

    n_timesteps
        The number of time values of a single time series.

    n_feature
        The number of features for each time series.

    Returns
    -------
    numpy.ndarray
        zero allocated tensor

    See Also
    --------
    :py:meth:`TSCDataFrame.from_tensor`

    """
    return np.zeros(
        [n_time_series, n_timesteps, n_feature], order="C", dtype=np.float64
    )
