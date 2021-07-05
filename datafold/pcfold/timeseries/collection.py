from numbers import Number
from typing import Generator, List, Optional, Tuple, Union

import matplotlib.colors as mclrs
import numpy as np
import pandas as pd
import scipy.sparse
from pandas.api.types import is_datetime64_dtype, is_numeric_dtype
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
        return cls(f"Numeric values are not finite (nan or inf values present).")

    @classmethod
    def not_min_samples(cls, min_samples):
        return cls(f"A minimum number of {min_samples} samples is required.")

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
        msg = (
            "The time sampling ('delta_time') is not constant. \n"
            "Note: If the time values are numerical floating points and were generated "
            "with numpy.linspace, then this can introduce numerical noise breaking "
            "the equal spacing."
        )

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


def _is_numeric_dtype(obj):
    if isinstance(obj, Number):
        # this includes np.nan and np.inf
        return True
    elif isinstance(obj, pd.Series) or isinstance(obj, np.ndarray):
        # call is_numeric_dtype from pandas
        return is_numeric_dtype(obj)
    elif isinstance(obj, pd.DataFrame):
        return np.all([is_numeric_dtype(ty) for ty in obj.dtypes])
    else:
        return False


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
        if not _is_numeric_dtype(value):
            raise AttributeError("Data in TSCDataFrame must be numeric.")
        return super(_LocTSCIndexer, self).__setitem__(key, value)


class _iLocTSCIndexer(_iLocIndexer):
    def __getitem__(self, item):
        sliced = super(_iLocIndexer, self).__getitem__(item)
        _type = type(sliced)

        try:
            # NOTE: this is different to loc -- here a pd.Series remains a pd.Series!
            # This is because pandas internal testing requires this
            if isinstance(sliced, pd.Series):
                # TODO: see issue #61
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

    def __setitem__(self, key, value):
        if not _is_numeric_dtype(value):
            raise AttributeError("Data in TSCDataFrame must be numeric.")
        return super(_iLocTSCIndexer, self).__setitem__(key, value)


class TSCDataFrame(pd.DataFrame):
    """Data frame to store time series collections.

    The class inherits from pandas' data structure :class:`pandas.DataFrame` and provides
    additional functionality to manipulate and analyze a time series collection. The
    main restrictions compared to the more general DataFrame are

    * The row index must be a multi index of two levels, where the first indicates
      the time series ID (integer), and the second contains the time values of the time
      series (non-negative and finite numerical values)
    * The column must be a one-dimensional column index that contains the feature
      names (accounting to the spatial axis)
    * There are no duplicates in both row and column index allowed (the flag
      `allows_duplicate_labels <https://pandas.pydata.org/docs/reference/api/pandas.Flags.allows_duplicate_labels.html>`__
      is set to True). Note, that this disables inplace operations on the labels (e.g.
      :code:`tsc.set_index(new_index, inplace=True` raises an error).
    * All time series values must be of a numeric dtype (`nan` or `inf` are allowed).

    A single time series typically consists of two or more samples with unequal
    time values. However, it is also possible to store "degenerated time series",
    which only contain a single sample. In some situations this is useful, such as when
    setting up initial conditions.

    Extended algorithms that act on ``TSCDataFrame`` are provided in
    :py:class:`.TSCAccessor`. Furthermore, all functionality provided by pandas can
    also be used, see
    `documentation <https://pandas.pydata.org/pandas-docs/stable/index.html>`__.

    .. note::
        Pandas provides a large range of functionality. Not all available methods are
        tested if they work properly with ``TSCDataFrame``. Please report unexpected
        behavior, bugs or inconsistencies by opening a new
        `issue <https://gitlab.com/datafold-dev/datafold/-/issues>`__.

    .. warning::
        Currently, there is no corresponding ``TSCSeries``, which results in
        inconsistencies in `iloc` indexing. For example, ``tsc.iloc[:, 0]`` returns a
        ``pandas.Series`` instead of a ``TSCDataFrame``. It is encouraged to use slices
        that return a data frame, i.e. ``tsc.iloc[:, [0]]``.

    Examples
    --------

    An example ``TSCDataFrame`` with three time series and two features (columns). The
    first two time series share the same time values (rows). The third time series
    contains only a single sample (degenerated time series).

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

    Parameters
    ----------
    *args
    **kwargs
        All parameters (`*args` and `**kwargs`) are passed to the superclass
        ``pandas.DataFrame``, except the keyword arguments:

        * ``kernel`` :py:class:`.BaseManifoldKernel` - A kernel to describe the
          point similarity. Defaults to ``None``.
        * ``dist_kwargs`` :class:`dict` - Keyword arguments passed to
          :py:meth:`.compute_distance_matrix`.

    Attributes
    ----------
    tsc_id_idx_name
        The index name of the first level in the row index.

    tsc_time_idx_name
        The index name of the second level in the row index.

    tsc_feature_col_name
        The name of of the columns index.

    kernel
        The kernel to describe the proximity between samples. A
        :py:class:`.PCManifoldKernel` ignores the time information and
        :py:class:`.TSCManifoldKernel` can utilize include temporal information in the
        metric.

    dist_kwargs
        Keyword arguments passed to the internal distance matrix computation. See
        :py:meth:`datafold.pcfold.distance.compute_distance_matrix` for parameter
        arguments.
    """

    tsc_id_idx_name = "ID"
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

        try:
            # In validate an AttributeError is raised, follow-up changes to the data
            # are dealt with the flag (introduced in pandas v. 1.2.0)
            self.flags.allows_duplicate_labels = False
            self._validate()
        except AttributeError as e:
            # This is a special case of input, where the fallback is a cast to
            # pd.DataFrame.
            # Internally of pandas the __init__ is called like this:
            # df._constructor(res)
            # --> df._constructor is TSCDataFrame
            # --> res is a BlockManager (pandas internal type)
            # If a BlockManager slices the index, then no AttributeError is raised,
            # but instead self casted to DataFrame
            # (case was necessary with changes introduced in pandas==1.2.0)
            _is_blockmanager_input = len(args) > 0 and isinstance(
                args[0], pd.core.internals.managers.BlockManager
            )

            if _is_blockmanager_input and self.index.nlevels != 2:
                # TODO: this seems dangerous... maybe there are better ways to deal
                #  with this?
                self = pd.DataFrame(self)
            else:
                raise e

    def __setattr__(self, key, value):
        if key == "index":
            value = self._validate_index(value)
        elif key == "columns":
            value = self._validate_columns(value)

        # Note: validation of data is not necessary here because this is done
        # in _LocTSCIndexer and _iLocTSCIndexer
        return super(TSCDataFrame, self).__setattr__(key, value)

    @classmethod
    def from_tensor(
        cls,
        tensor: np.ndarray,
        time_series_ids: Optional[np.ndarray] = None,
        columns: Optional[Union[pd.Index, list]] = None,
        time_values: Optional[np.ndarray] = None,
    ) -> "TSCDataFrame":
        """Create a ``TSCDataFrame`` from a three dimensional tensor.

        Parameters
        ----------
        tensor
            The time series data of shape `(n_timeseries, n_timesteps, n_feature,)`.

        time_series_ids
            The IDs of shape `(n_timeseries,)` to assign for the respective time
            series. Defaults to `(0,1,2,..., n_timeseries-1)`.

        columns
            The feature names of shape `(n_feature,)`. Defaults to
            string `feature[0,1,2,..., n_feature-1]`.

        time_values
            Time values of the time series in the tensor. Defaults to
            `(0,1,2, ..., n_timesteps-1)`.

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
        (
            n_timeseries,
            n_timesteps,
            n_feature,
        ) = tensor.shape

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
        """Create ``TSCDataFrame`` from shift matrices.

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
            `feature[0,1,2,..., n_feature-1]`.

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
        """Create ``TSCDataFrame`` by using same index or columns from other
        ``TSCDataFrame``.

        Parameters
        ----------
        indices_from
            Object to copy index and/or columns from.

        values
            Values for new ``TSCDataFrame``.

        except_index
            Index for new object (only copy columns). Should not be
            set together with `except_columns`.

        except_columns
            Columns for new object (only copy index). Should not be
            set together with `except_index`.

        Returns
        -------
        TSCDataFrame
            new instance
        """

        if not isinstance(indices_from, TSCDataFrame):
            raise TypeError("Parameter 'indices_from' must be of type 'TSCDataFrame'")

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
        """Create ``TSCDataFrame`` from single time series.

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

        if isinstance(df, TSCDataFrame) or df.index.nlevels > 1:
            # Handling of TSCDataFrame can be implemented if required.
            raise TypeError("The row index must only contain time values.")

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
        ts_ids: Optional[Union[np.ndarray, pd.Index, List[int]]] = None,
    ) -> "TSCDataFrame":
        """Create ``TSCDataFrame`` from a list of time series.

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
        try:
            frame_list = list(frame_list)
        except TypeError:
            raise TypeError("Parameter 'frame_list' must be iterable")

        if len(frame_list) == 0:
            raise ValueError("'frame_list' must contain at least one object")

        ref_df = frame_list[0]
        for _df in frame_list[1:]:
            if not isinstance(_df, pd.DataFrame):
                raise TypeError("All elements in list must be of type DataFrame.")

            is_df_same_index(
                ref_df,
                _df,
                check_index=False,
                check_column=True,
                check_names=False,
                handle="raise",
            )

        # prepare list
        final_list: List[pd.DataFrame] = []

        for df in frame_list:
            # >= 2 to raise error for invalid DataFrames
            if df.index.nlevels >= 2 and not isinstance(df, TSCDataFrame):
                try:
                    df = TSCDataFrame(df)
                except AttributeError:
                    raise TypeError(
                        f"Cannot process entry check that the format is correct. \n {df}"
                    )

            if isinstance(df, TSCDataFrame):
                if len(df.ids) > 1:
                    df = [e[1] for e in list(df.itertimeseries())]
                else:
                    df = pd.DataFrame(df)
                    df.index = df.index.get_level_values(TSCDataFrame.tsc_time_idx_name)

            if isinstance(df, list):
                final_list += df
            else:
                final_list.append(df)

        if ts_ids is None:
            ts_ids = np.arange(len(final_list)).astype(int)
        else:
            ts_ids = np.asarray(ts_ids)

        assert isinstance(ts_ids, np.ndarray)  # mypy check

        if (
            ts_ids.ndim != 1
            or ts_ids.shape[0] != len(final_list)
            or len(np.unique(ts_ids)) != len(final_list)
        ):
            raise ValueError(
                "Parameter 'ts_ids' must contain unique time series IDs with same "
                "length than 'frame_list'"
            )

        tsc_list = list()
        for _id, df in zip(ts_ids, final_list):
            # TODO: can be done without loop and would be more efficient
            df.index = pd.MultiIndex.from_product([[_id], df.index.to_numpy()])
            tsc_list.append(TSCDataFrame(df))

        return cls(pd.concat(tsc_list, axis=0))

    @staticmethod
    def unique_delta_times(
        delta_times: Union[np.ndarray, pd.Series], rtol=1e-11, atol=1e-15
    ) -> np.ndarray:
        """Returns unique (within tolerances) delta time values.

        The tolerances are required for floating point time values. For example,

        .. code::

            np.unique(np.diff(np.linspace(0,2, 4)))

        prints :code:`array([0.6666666666666666, 0.6666666666666667])`.

        The default tolerance values are adjusted in test_linspace_unique_delta_times so
        that a wide range of time_values generated with ``np.linspace`` are considered
        equally spaced (despite the numerical noise that breaks the equality).

        Parameters
        ----------
        delta_times
            Array of delta times between samples of a time series.

        rtol
            Relative tolerance of the largest versus the smallest delta time.

        atol
            Absolute tolerance between the delta times to be considered unique.

        Returns
        -------
        numpy.ndarray
            Unique (in tolerance levels) delta times.
        """

        unique_delta_times = np.unique(np.asarray(delta_times))

        if unique_delta_times.dtype == float and len(unique_delta_times) > 1:
            # Note: unique_delta_times is already sorted by np.unique
            max_step = np.diff(unique_delta_times)
            rel_step = max_step / unique_delta_times[1:]

            # print(f"max_step={max_step} | rel_step={rel_step}")

            abs_groups = max_step <= atol
            rel_groups = rel_step <= rtol

            groups = np.logical_not(np.logical_or(abs_groups, rel_groups)).cumsum()
            groups = np.append(0, groups)

            n_groups = groups[-1] + 1  # +1 because 0 is also a group
            _unique_dt = np.zeros(n_groups)

            for group in range(n_groups):
                _unique_dt[group] = np.mean(unique_delta_times[groups == group])

            unique_delta_times = _unique_dt

        return unique_delta_times

    def to_csv(self, *args, **kwargs) -> Optional[str]:
        """Write object to a comma-separated values (csv) file.

        Internaly, the method casts the object (self) to pd.Dataframe before the csv is
        written. The reason is that for larger files it could be observed that the
        internals of ``to_csv`` could lead to invalid TSCDataFrame objects (which
        raise an error).

        Parameters
        ----------
        **kwargs
            All keyword arguments are passed to the super class. See
            `docu <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html>`__

        Returns
        -------
        None or str
            If path is None, returns the resulting csv format as a
            string. Otherwise returns None.

        """

        return pd.DataFrame(self).to_csv(*args, **kwargs)

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
        raise NotImplementedError("expanddim is not supported for TSCDataFrame")

    def _validate_index(self, index: Union[pd.MultiIndex, pd.Index]) -> pd.MultiIndex:

        if not isinstance(index, pd.MultiIndex):
            raise AttributeError("index must be of type pd.MultiIndex")

        if index.nlevels != 2:
            # must exactly have two levels [ID, time]
            raise AttributeError(
                "Index has to be a 'MultiIndex' with two levels (index.nlevels == 2). "
                "First level for the time series ID, "
                f"second level for the time values. Got: {index.nlevels}"
            )

        if index.duplicated().any():
            raise AttributeError(
                f"Duplicated indices found: " f"{index[index.duplicated()].to_numpy()}"
            )

        # Insert required index names:
        #  -- Note: this overwrites potential previous names.
        index.names = [self.tsc_id_idx_name, self.tsc_time_idx_name]

        ids_index = index.get_level_values(self.tsc_id_idx_name)
        time_index = index.get_level_values(self.tsc_time_idx_name)

        if ids_index.dtype != int:
            # convert to int if it is possible to transform without loss
            # (e.g. from 5.0 to 5); else raise AttributeError
            if (ids_index.astype(int) == ids_index).all():
                index = index.set_levels(
                    index.levels[0].astype(int),
                    level=self.tsc_id_idx_name,
                )
            else:
                raise AttributeError(
                    "Time series IDs must be of type integer. "
                    f"Got dtype={ids_index.dtype}"
                )

        if (ids_index < 0).any():
            unique_negative_ids = np.unique(ids_index < 0)
            raise AttributeError(
                f"All time series IDs have to be non-negative integer values. "
                f"Got invalid time series IDs: {unique_negative_ids}"
            )

        if time_index.dtype.kind in "bcmOSUV":
            # See further info for 'kind'-codes:
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.kind.html
            raise AttributeError(
                f"Time values have to be numeric and real-valued. "
                f"Got dtype={time_index.dtype}"
            )
        else:
            assert (
                time_index.dtype.kind in "uifM"
            ), f"Numpy dtype.kind={time_index.dtype.kind} not handled, please report bug."

        # u - unsigned integer | i - signed integer | f - floating point
        # do not perform this check for:
        # M - datetime (bc. the second condition raises an error)
        if time_index.dtype.kind in "uif" and (time_index < 0).any():
            raise AttributeError(
                "Time values have to be non-negative. "
                f"Found invalid time values {(time_index[time_index < 0])}"
            )

        # bool index to the start of new IDs
        bool_new_id = np.append(1, np.diff(index.codes[0])).astype(bool)
        _ids = ids_index[bool_new_id]

        if len(np.unique(_ids)) != len(_ids):
            raise AttributeError("Time series IDs appear multiple times.")

        ts_internal_codes = np.append(0, np.diff(index.codes[1]))
        if np.any(ts_internal_codes[~bool_new_id] <= 0):
            raise AttributeError("The time values of each time series must be sorted.")

        return index

    def _validate_columns(self, columns: pd.Index) -> pd.Index:

        if not isinstance(columns, pd.Index):
            raise AttributeError("columns must be of type pd.Index")

        if columns.nlevels != 1:
            # must exactly have two levels [ID, time]
            raise AttributeError(
                f"Columns must be single level (columns.nlevels == 1).  "
                f"Got: columns.nlevels={columns.nlevels}"
            )

        if columns.duplicated().any():
            raise AttributeError(
                f"Duplicated columns found: "
                f"{columns[columns.duplicated()].to_numpy()}"
            )

        columns.name = self.tsc_feature_col_name
        return columns

    def _validate_data(self) -> None:
        if not _is_numeric_dtype(self):
            raise AttributeError("All feature columns must have a numeric dtype")

    def _validate(self) -> bool:

        # just for security remove unused levels --
        # disable validate to avoid infinite recursion
        self.index: pd.Index = self._validate_index(self.index)
        self.index = self.index.remove_unused_levels()

        self.columns: pd.Index = self._validate_columns(self.columns)
        self._validate_data()
        return True

    @property
    def n_timeseries(self) -> int:
        """Number of time series in the collection."""
        return len(self.ids)

    @property
    def n_features(self) -> int:
        """Number of features in the collection."""
        return self.shape[1]

    @property
    def ids(self) -> pd.Index:
        """The time series IDs in the collection."""
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
            _id_dt = diff_times[id_indexer.get_indexer_for([timeseries_id])[:-1]]
            _id_unique_dt = self.unique_delta_times(_id_dt)

            if len(_id_unique_dt) == 1:
                dt_result_series[timeseries_id] = _id_unique_dt[0]

        if not np.isnan(dt_result_series).all():

            _unique_result_series = self.unique_delta_times(
                dt_result_series, rtol=0, atol=1e-15
            )

            n_different_dts = len(_unique_result_series)

            if not np.isnan(dt_result_series).any():
                # TODO: here it may be interesting to check the new "Null" types of
                #  pandas. They allow to also have nan by still keeping other dtypes
                #  than float (--> only float has a nan representation in Numpy types)
                dt_result_series = dt_result_series.astype(diff_times.dtype)

        else:
            # all nan (i.e. irregular sampling), treat as all "identical
            # irregular sampled"
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
            n_time_elements = n_time_elements.sort_index()
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

        Visit
        `pd.DataFrame.iloc <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html#pandas.DataFrame.iloc>`_
        for documentation of how to index DataFrame.

        .. warning::
            For single column slices (e.g. ``tsc.iloc[:, 0]``), the type
            changes to ``pandas.Series``, use ``tsc.iloc[:, [0]]`` instead to maintain a
            ``TSCDataFrame``.

        Returns
        -------

        """
        return _iLocTSCIndexer("iloc", self)

    def set_index(
        self,
        keys,
        drop=True,
        append=False,
        inplace=False,
        verify_integrity=False,
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

        _slice = pd.DataFrame(self).xs(
            key, axis=axis, level=level, drop_level=drop_level
        )

        try:
            _slice = TSCDataFrame(_slice)
        except AttributeError:
            pass

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
        """Indicates whether 'time' index is datetime format."""
        return is_datetime64_dtype(self.index.get_level_values(self.tsc_time_idx_name))

    def itertimeseries(self) -> Generator[Tuple[int, pd.DataFrame], None, None]:
        """Generator of contained time series.

        Each iteration returns the time series ID and the corresponding
        time series (a :class:`pd.DataFrame` instead of a ``TSCDataFrame``).

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
        time steps.
        """
        return len(np.unique(self.n_timesteps)) == 1

    def is_const_delta_time(self) -> bool:
        """Indicates if all time series in the collection have the same time delta."""

        # If dt is a Series it means it shows "dt per ID" (because it is not constant).
        _dt = self.delta_time

        if isinstance(_dt, pd.Series):
            return False

        # pd.isnull is better than np.finite because it also checks for
        # NaT (Not a Time[-delta])
        return not pd.isnull(_dt)

    def is_same_time_values(self) -> bool:
        """Indicates if all time series in the collection share the same time values."""

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
        """Indicates if all feature values are finite (i.e. neither NaN nor inf)."""
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
            X=self,
            Y=Y,
            metric=metric,
            **self.dist_kwargs,
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
            X=self,
            Y=Y,
            dist_kwargs=self.dist_kwargs,
            **kernel_kwargs,
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
            New time series, with same features (column names) as the existing.
            If ``df`` is of type ``pd.DataFrame``, the row index must contain time
            values.
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

        if df.index.nlevels == 1:
            _index = df.index
        elif df.index.nlevels == 2 and isinstance(df, TSCDataFrame):
            _index = df.index.get_level_values(self.tsc_time_idx_name)
        else:
            raise ValueError("The input parameter 'df' is of wrong format.")

        if self.is_datetime_index() ^ np.issubdtype(_index.dtype, np.datetime64):
            raise ValueError(
                "If the existing index is datetime64, but the new time "
                f"series has dype={_index.dtype} for time values."
            )

        # Add the id to the first level of the MultiIndex
        df.index = pd.MultiIndex.from_arrays(
            [np.ones(df.shape[0], dtype=int) * ts_id, _index],
            names=[self.tsc_id_idx_name, self.tsc_time_idx_name],
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

        if self.is_datetime_index():
            return np.arange(start, end + np.timedelta64(1, "ns"), self.delta_time)
        else:
            # maintain actual time dtype
            _dtype = self.index.get_level_values(self.tsc_time_idx_name).dtype

            return np.arange(
                start, np.nextafter(end, np.finfo(float).max), self.delta_time
            ).astype(_dtype)

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

        color, c = kwargs.pop("color", None), kwargs.pop("c", None)

        if color is not None and c is not None:
            raise TypeError(
                "Got both 'color' and 'c', which are aliases of one another."
            )
        elif c is not None:
            color = c

        first = True

        for i, ts in self.itertimeseries():
            kwargs["ax"] = ax

            if first:
                # there may be already lines in it, set color of *new* lines
                exist_lines = len(ax.lines) if ax is not None else 0

                ax = ts.plot(color=color, legend=legend, **kwargs)

                if color is None:
                    color = [
                        mclrs.to_rgba(ax.lines[j].get_c())
                        for j in range(exist_lines, exist_lines + self.n_features)
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
    return np.zeros([n_time_series, n_timesteps, n_feature], order="C", dtype=float)
