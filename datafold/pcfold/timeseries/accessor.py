#!/usr/bin/env python3

import warnings
from typing import Generator, Optional, Tuple, Union

import findiff.diff
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest
from scipy.stats import multivariate_normal

from datafold.pcfold.timeseries.collection import TSCDataFrame, TSCException
from datafold.utils.general import is_integer


@pd.api.extensions.register_dataframe_accessor("tsc")
class TSCAccessor(object):
    """Extension functions for TSCDataFrame.

    See `documentation <https://pandas.pydata.org/pandas-docs/stable/development/extending.html?highlight=accessor>`_
    for regular pandas accessors.

    The functions are available through the accessor `tsc`, for example,

    .. code::

            tsc_object.tsc.normalize_time()

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
                "The 'tsc' extension only works for type TSCDataFrame (convert before)."
            )

        self._tsc_df = tsc_df

    def check_tsc(
        self,
        *,
        ensure_all_finite: bool = True,
        ensure_min_samples: Optional[int] = None,
        ensure_same_length: bool = False,
        ensure_const_delta_time: bool = True,
        ensure_delta_time: Optional[float] = None,
        ensure_same_time_values: bool = False,
        ensure_normalized_time: bool = False,
        ensure_n_timeseries: Optional[int] = None,
        ensure_min_timesteps: Optional[int] = None,
        ensure_n_timesteps: Optional[int] = None,
        ensure_no_degenerate_ts: bool = True,
    ) -> TSCDataFrame:
        """Validate time series properties.

        This summarises the single check functions also contained in `TSCAccessor`.

        Parameters
        ----------
        ensure_all_finite
            If True, check if all values are finite (no 'nan' or 'inf' values).

        ensure_min_samples
            If provided, check that the frame has at least required samples.

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

        ensure_n_timesteps
            If provded, check that all time series have exactly the number the
            timesteps spectifed.

        ensure_min_timesteps
            If provided, check if every time series has the required minimum of time
            steps.

        ensure_no_degenerate_ts
            If True, make sure that no degenerate (single sampled) time series are
            present.

        Returns
        -------
        TSCDataFrame
            validated time series collection (without changes)
        """

        # TODO: allow handle_fail="raise | warn | return"?

        if ensure_all_finite:
            self.check_finite()

        if ensure_min_samples is not None:
            self.check_min_samples(min_samples=ensure_min_samples)

        if ensure_same_length:
            self.check_timeseries_same_length()

        if ensure_const_delta_time:
            self.check_const_time_delta()

        if ensure_delta_time is not None:
            self.check_required_time_delta(required_time_delta=ensure_delta_time)

        if ensure_same_time_values:
            self.check_equal_timevalues()

        if ensure_normalized_time:
            self.check_normalized_time()

        if ensure_n_timeseries is not None:
            self.check_required_n_timeseries(required_n_timeseries=ensure_n_timeseries)

        if ensure_n_timesteps is not None:
            self.check_required_n_timesteps(ensure_n_timesteps)

        if ensure_min_timesteps is not None:
            self.check_required_min_timesteps(ensure_min_timesteps)

        if ensure_no_degenerate_ts:
            self.check_no_degenerate_ts()

        return self._tsc_df

    def check_finite(self) -> None:
        """Check if all values are finite (i.e. does not contain `nan` or `inf`)."""
        if not self._tsc_df.is_finite():
            raise TSCException.not_finite()

    def check_min_samples(self, min_samples) -> None:
        """Check if there is a minimum number of samples included."""
        if self._tsc_df.shape[0] < min_samples:
            raise TSCException.not_min_samples(min_samples=min_samples)

    def check_timeseries_same_length(self) -> None:
        """Check if time series in the collection have the same length."""
        if not self._tsc_df.is_equal_length():
            raise TSCException.not_same_length(
                actual_lengths=self._tsc_df.is_equal_length()
            )

    def check_const_time_delta(self) -> Union[pd.Series, float]:
        """Check if all time series have the same time-delta."""
        delta_time = self._tsc_df.delta_time
        if not self._tsc_df.is_const_delta_time():
            raise TSCException.not_const_delta_time(self._tsc_df.delta_time)
        return delta_time

    def check_equal_timevalues(self) -> None:
        """Check if all time series in the collection share the same time values."""
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
            delta_times = np.asarray(self._tsc_df.delta_time)

            if self._tsc_df.is_datetime_index():
                if (delta_times != required_time_delta).any():
                    raise AttributeError
            else:
                # this is a better variant than
                # np.asarray(self._tsc_df.delta_time) == np.asarray(required_time_delta)
                # because the shapes can also mismatch
                nptest.assert_allclose(
                    delta_times,
                    np.asarray(required_time_delta),
                    rtol=1e-12,
                    atol=1e-15,
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

    def check_required_n_timesteps(self, required_n_timesteps: int) -> None:
        n_timesteps = self._tsc_df.n_timesteps

        if isinstance(n_timesteps, pd.Series):
            raise TSCException.not_n_timesteps(required=required_n_timesteps)
        else:
            assert isinstance(n_timesteps, int)
            if n_timesteps != required_n_timesteps:
                raise TSCException.not_n_timesteps(required=required_n_timesteps)

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

    def check_no_degenerate_ts(self):
        if self._tsc_df.has_degenerate():
            raise TSCException.has_degenerate_ts()

    def check_non_overlapping_timeseries(self) -> None:
        """Check if all time series have disjoint time values (do not overlap).

        Returns
        -------

        """
        _, counts = np.unique(
            self._tsc_df.index.get_level_values(TSCDataFrame.tsc_time_idx_name),
            return_counts=True,
        )
        if (counts > 1).any():
            raise TSCException("time series are required to be non-overlapping")

    @classmethod
    def check_equal_delta_time(
        cls, X: TSCDataFrame, Y: TSCDataFrame, atol=1e-15, require_const=False
    ) -> Tuple[Union[float, pd.Series], Union[float, pd.Series]]:
        """Check if two time series collections have the same delta times.

        Parameters
        ----------
        X
            First time series collection.
        Y
            Second time series collection.
        atol
            Tolerance passed to :py:meth:`.equal_const_delta_time`

        require_const
            If True, both `X` and `Y` must have constant delta times.

        Raises
        ------
        :py:class:`TSCException` - if time_delta not equal or if either `X` or `Y` is not
        constant with ``require_const=True``.

        Returns
        -------

        """
        X_dt = X.delta_time
        Y_dt = Y.delta_time

        equal = True
        if isinstance(X_dt, pd.Series) and not require_const:
            if not isinstance(Y_dt, pd.Series):
                equal = False
            else:
                try:
                    pdtest.assert_series_equal(X_dt, Y_dt, atol=atol)
                except AssertionError:
                    equal = False

        elif (
            isinstance(X_dt, pd.Series) or isinstance(Y_dt, pd.Series)
        ) and require_const:
            raise TSCException.not_const_delta_time()
        else:
            if not cls.equal_const_delta_time(X_dt, Y_dt, atol=atol):
                equal = False

        if not equal:
            raise TSCException.not_required_delta_time(X_dt, Y_dt)

        return X_dt, Y_dt

    @classmethod
    def equal_const_delta_time(cls, dt1: float, dt2: float, atol=1e-15) -> bool:
        """Returns True, if the time deltas should be treated equally.

        Parameters
        ----------
        dt1
            First delta time.

        dt2
            Second delta time.

        atol
            Acceptable absolute tolerance between the two delta times. This is
            relevant for floating delta times that have "numerical noise" when
            equally spaced.

        Returns
        -------
        bool
        """
        return np.abs(dt1 - dt2) <= atol

    def iter_timevalue_window(
        self,
        window_size: int,
        offset: int,
        per_time_series: bool = False,
        strictly_sequential: bool = True,
    ) -> Generator[TSCDataFrame, None, None]:
        """Iterator over time series intervals (window).

        Parameters
        ----------
        window_size
            The number of samples for each window. Note that the `blocksize` is not
            guaranteed and is usually shorter in last iterations if the number of samples
            are not a multiple of `blocksize`.

        offset
            A positive integer value that indicates by how much the next window should be
            shifted. If ``offset=blocksize``, then the windows are non-overlapping.

        per_time_series
            If True, the windows are generated for each time series separately. This is
            recommended for time series a collection consists where the time series are
            disjoint (i.e. non-overlapping time intervals).

        strictly_sequential
            TODO

        Returns
        -------
        Generator[TSCDataFrame]
            An iterator for the windowed time series data.
        """

        self.check_const_time_delta()

        if not is_integer(window_size):
            raise TypeError(
                f"The parameter 'window_size={window_size}' must be of type integer. "
                f"Got {type(window_size)}"
            )

        if not is_integer(offset):
            raise TypeError(
                f"The parameter 'offset={offset}' must be of type integer."
                f"Got {type(window_size)}"
            )

        if window_size <= 0:
            raise ValueError(
                f"The parameter 'window_size={window_size}' must be positive."
                f"Got {type(window_size)}"
            )

        if offset <= 0:
            raise ValueError(
                f"The parameter 'offset={offset}' must be positive."
                f"Got {type(window_size)}"
            )

        if per_time_series:
            _iter_timeseries_collection = self._tsc_df.groupby(
                by=self._tsc_df.tsc_id_idx_name
            )
        else:
            # This mimics a single element of the "groupby" function -- the first element
            # is the time series id, but not required here and therefore None
            _iter_timeseries_collection = [(None, self._tsc_df)]

        for _, current_tsc in _iter_timeseries_collection:
            time_values = current_tsc.time_values()
            start = 0
            end = start + window_size

            while end <= time_values.shape[0]:
                selected_time_values = time_values[start:end]

                start = start + offset
                end = start + window_size

                _ret = current_tsc.select_time_values(selected_time_values)

                if strictly_sequential and isinstance(_ret.n_timesteps, pd.Series):
                    pass
                else:
                    yield _ret

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
        """Normalize time in time series collection.

        A :py:class:`TSCDataFrame` with normalized time has the following properties:

        * the global time starts at zero
        * delta_time is constant one

        Note, that at least one time series starts at time zero, but other can

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
            convert_times = convert_times.astype(int)
            min_time = min_time.astype(int)
            delta_time = delta_time.astype(int)

        convert_times = np.array((convert_times - min_time) / delta_time, dtype=int)
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

    def time_derivative(
        self,
        scheme="center",
        diff_order: int = 1,
        accuracy: int = 2,
        shift_index: bool = False,
    ) -> Union[pd.DataFrame, TSCDataFrame]:
        """Compute finite differences in time for each time series.

        .. note::
            The boundary samples are dropped at which no finite difference scheme of
            the set accuracy is possible. To apply lower accuracy schemes requires
            implementation.

        Parameters
        ----------
        scheme
            The finite difference scheme 'backward', 'center' or 'forward'.

        diff_order
            The order of the derivative.

        accuracy
            The accuracy (even positive integer) of the derivative scheme.

        shift_index
            If True, then the time is shifted such that no future samples are included.
            For example, for the coefficients` [-1,0,1]`, the computed time derivative
            for time 1 is then shifted to time 2. The option is inteded for
            `scheme='center'`. The parameter has no effect for `scheme=backward` and is
            discouraged for `scheme=forward`.

        Returns
        -------
        Union[pd.DataFrame, TSCDataFrame]
            The finite difference time series. The boundary samples are removed,
            i.e. the number of samples decrease accordingly.
        """

        class InternalDiff(findiff.diff.Diff):
            """Overwrites the behaviour of the findiff superclass."""

            def diff(
                self,
                data: Union[np.ndarray, pd.DataFrame],
                spacing: float,
                accuracy: int,
            ):
                n_samples = data.shape[self.axis]
                coeff_dict = findiff.coefficients(self.order, accuracy)

                weights = coeff_dict[scheme]["coefficients"]
                offsets = coeff_dict[scheme]["offsets"]

                # only select samples where we can compute the centered difference
                # scheme (i.e. we drop samples at the time series boundary)
                start_sample = np.abs(offsets.min())
                end_sample = n_samples - np.abs(offsets.max())

                ref_slice = slice(start_sample, end_sample, 1)
                off_slices = [
                    self._shift_slice(ref_slice, offsets[k], n_samples)
                    for k in range(len(offsets))
                ]

                data_dt = np.zeros_like(data, dtype=float)

                if isinstance(data, pd.DataFrame):
                    data_numpy = data.to_numpy()
                else:
                    data_numpy = data.view()

                self._apply_to_array(
                    data_dt, data_numpy, weights, off_slices, ref_slice, self.axis
                )

                data_dt = data_dt[start_sample:end_sample, :]

                h_inv = 1.0 / spacing ** self.order
                data_dt *= h_inv

                if scheme in ["center", "forward"] and shift_index:
                    # NOTE: Only the first samples of the time values are dropped. This
                    # means that the time is shifted to the finite difference offset that
                    # lies furthest in the future.
                    lost_samples = data.shape[0] - data_dt.shape[0]
                    return pd.DataFrame(
                        data_dt, index=data.index[lost_samples:], columns=data.columns
                    )
                else:
                    return pd.DataFrame(
                        data_dt,
                        index=data.index[start_sample:end_sample],
                        columns=data.columns,
                    )

        if scheme not in ["backward", "center", "forward"]:
            raise ValueError(f"scheme={scheme} must be 'center' or 'backward'")

        self.check_const_time_delta()
        spacing = self._tsc_df.delta_time

        min_samples = np.inf
        time_derivative = list()

        dt_func = InternalDiff(axis=0, order=diff_order)

        for ts_id, time_series in self._tsc_df.itertimeseries():
            time_series_dt = dt_func.diff(
                data=time_series,
                spacing=spacing,
                accuracy=accuracy,
            )
            min_samples = min(min_samples, time_series_dt.shape[0])

            time_derivative.append(time_series_dt)

        time_derivative = TSCDataFrame.from_frame_list(
            time_derivative, ts_ids=self._tsc_df.ids
        )
        return time_derivative

    def assign_ids_sequential(self) -> TSCDataFrame:
        """Assign time series IDs sequentially starting from zero in a collection.

        Note, that this operation is inplace and overwrites the existing time series IDs.

        Returns
        -------
        TSCDataFrame
            The data with time series IDs in sequential order.
        """

        self._tsc_df.index = self._tsc_df.index.remove_unused_levels()
        # levels[0] = IDs, levels[1] = time
        n_timeseries = len(self._tsc_df.index.levels[0])

        self._tsc_df.index = self._tsc_df.index.set_levels(
            np.arange(n_timeseries), level=0
        )
        return self._tsc_df

    def assign_ids_train_test(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        return_dropped: bool = False,
    ):
        """Split and assign time series IDs based on training and test indices.

        Note, that the indices included in train_indices and test_indices must be
        disjoint (i.e. a sample cannot be both in ``train_indices`` and
        ``test_indices``). Indices that are not included in either training or
        testing are dropped samples.

        Parameters
        ----------
        train_indices
            The indices to indicate which samples are included in the training set.

        test_indices
            The indices to indicate which samples are included in the test set.

        return_dropped
            If True, a DataFrame is returned in the third return value which includes
            all samples that were neigher included in the training indices nor the test
            indices.

        Returns
        -------
        TSCDataFrame
            The time series collection for training.

        TSCDataFrame
            The time series collection for testing.

        pandas.DataFrame
            The dropped samples; only returned if ``return_dropped_samples=True``.
        """

        def _test_array(_array):
            bool_dim = _array.ndim == 1
            bool_positive = np.all(_array >= 0)
            bool_sorted = np.all(_array[:-1] < _array[1:])
            bool_type = _array.dtype == int

            if not (bool_dim and bool_positive and bool_sorted and bool_type):
                raise ValueError(
                    "The arrays 'train_indices' and 'test_indices' must be sorted 1-dim. "
                    "array of non-negative and unique integer values."
                )

        _test_array(train_indices)
        _test_array(test_indices)

        all_indices = np.append(train_indices, test_indices)
        if len(np.unique(all_indices)) != len(all_indices):
            raise ValueError(
                "Some indices are both included in 'train_indices' and 'test_indices'."
            )

        # mark train samples with 0 and test samples with 1
        mask_test = np.zeros(self._tsc_df.shape[0])
        mask_test[test_indices] = 1

        # usage of np.diff -> detect changes in
        # i) new fold or ii) new ID (i.e. time series)
        # -- both change detections are required to reassign new IDs

        # i) detect switch between test / train
        change_fold_indicator = np.append(0, np.diff(mask_test)).astype(bool)

        # ii) detect switch of new ID
        change_id_indicator = np.append(
            0,
            np.diff(self._tsc_df.index.get_level_values(TSCDataFrame.tsc_id_idx_name)),
        ).astype(bool)

        # iii) detect switch of dropped indices
        mask_dropped = np.ones(self._tsc_df.shape[0], dtype=bool)
        mask_dropped[train_indices] = False
        mask_dropped[test_indices] = False

        # cumulative sum of on or the other change and reassign IDs
        id_cum_sum_mask = np.logical_or(
            np.logical_or(change_fold_indicator, change_id_indicator),
            mask_dropped,
        )
        new_ids = np.cumsum(id_cum_sum_mask)

        reassigned_ids_idx = pd.MultiIndex.from_arrays(
            arrays=(
                new_ids,
                self._tsc_df.index.get_level_values(TSCDataFrame.tsc_time_idx_name),
            )
        )

        # See also gitlab issue #105 (if addressed, can use TSCDataFrame)
        splitted_df = pd.DataFrame(
            data=self._tsc_df.to_numpy(),
            index=reassigned_ids_idx,
            columns=self._tsc_df.columns,
        )

        train_tsc = splitted_df.iloc[train_indices, :]
        test_tsc = splitted_df.iloc[test_indices, :]

        try:
            # return TSCDataFrame if legal
            train_tsc = TSCDataFrame(train_tsc)
        except AttributeError:
            pass  # return pd.DataFrame

        try:
            # return TSCDataFrame if legal
            test_tsc = TSCDataFrame(test_tsc)
        except AttributeError:
            pass  # return pd.DataFrame

        if return_dropped:
            dropped_tsc = pd.DataFrame(splitted_df).loc[mask_dropped]
            return train_tsc, test_tsc, dropped_tsc
        else:
            return train_tsc, test_tsc

    def assign_ids_const_delta(self, drop_samples=False) -> Optional[TSCDataFrame]:
        """Split time series with irregular time sampling frequencies in new time
        series of intervals with constant time sampling.

        This function only considers time series with irregular ``delta_time`` and aims
        to split these into sub time series of finite ``delta_time``. Time series with a
        finite ``delta_time`` may only receive a new ID, but the samples remain the same.

        The detection of constant sampling intervals is carried out with the second time
        differences. For the detection of a new sub time series the sampling rate must
        be at constant for three samples (i.e. two time differences). This means the
        function does not assign sample pairs (time series of length two) when dealing
        with completely irregular time series. Instead the samples of irregular
        intervals are dropped. The main reason for this is that the assignment is not
        unique.

        Parameters
        ----------
        drop_samples
            If True, the function drops samples from irregular sampled intervals (up
            to entire time series). If dropping samples is required for assignment and
            the parameter is set to False, a ``ValueError`` is raised.

        Returns
        -------
        Optional[TSCDataFrame]
            Time series collection with re-allocated time series if irregular time
            series are present in the ``TSCDataFrame``. It returns ``None`` if all time
            series have a completely irregular time sampling.
        """

        def split_irregular_time_series(local_tsc_df: TSCDataFrame, min_id: int):
            assert min_id >= 0

            if local_tsc_df.shape[0] == 1 and drop_samples:
                # degenerated time series are dropped
                return None
            elif local_tsc_df.shape[0] == 1 and not drop_samples:
                raise ValueError(
                    "There is a single-sampled time series present and at "
                    "the same time 'drop_samples=False'."
                )

            if local_tsc_df.shape[0] == 2:
                # return early of special case of only 2 samples
                return local_tsc_df

            # time difference
            first_diff = np.diff(
                local_tsc_df.index.get_level_values(TSCDataFrame.tsc_time_idx_name)
            )

            if local_tsc_df.is_datetime_index():
                first_diff = first_diff.astype(int)

            first_diff = np.append(np.inf, first_diff)

            # change in time difference
            second_diff = np.diff(first_diff)
            second_diff = np.append(second_diff, 0)

            # Indicator for first case:
            # There is a gap in the sampling, e.g.
            # 1,2,3,10,11,12
            # This results into
            # first diff   1,1,7,1,1
            # second diff   0,6,-6,0
            # To identify this case, neighboring non-zero (with respect to the first
            # sample) are identified (the "6" identifies a new start of an ID)
            indicator = np.logical_and(second_diff[:-1], second_diff[1:])

            # remove the indentifications of the first kind from the second diff
            # from the example above remove the 6 and the neighboring -6
            second_diff[np.append(0, indicator).astype(bool)] = 0
            second_diff[np.append(indicator, 0).astype(bool)] = 0

            indicator = np.append(0, indicator)

            # Indicator for the second case:
            # There is a new sampling frequency
            # 1,2,3,5,7,9
            # This results into
            # first diff   1,1,2,2,2
            # second diff   0,1,0,0
            # I.e. there is a single difference (without a neighboring)
            # We simply take the second_diff (after removals of the first case) as
            # indicator for the start of a new time series ID).
            indicator = np.logical_or(indicator, second_diff.astype(bool))

            new_ids = np.cumsum(indicator)
            new_ids += min_id

            unique_ids, counts = np.unique(new_ids, return_counts=True)

            if drop_samples:
                remove_ids = unique_ids[counts == 1]

                mask_keep_elements = ~np.isin(new_ids, remove_ids)
                new_ids = new_ids[mask_keep_elements]
                local_tsc_df = local_tsc_df.loc[mask_keep_elements, :]
            else:
                if np.array(counts == 1).any():
                    raise ValueError(
                        "The new time series collection is invalid because there are "
                        "intervals of irregular time sampling frequency. Consider "
                        "setting 'drop_samples=True'."
                    )

            if local_tsc_df.shape[0] == 2 and drop_samples:
                return None
            else:
                # prepare df and assign new ids -
                # >> this creates new sub time series
                reassigned_ids_idx = pd.MultiIndex.from_arrays(
                    arrays=(
                        new_ids,
                        local_tsc_df.index.get_level_values(
                            TSCDataFrame.tsc_time_idx_name
                        ),
                    )
                )

                return local_tsc_df.set_index(reassigned_ids_idx)

        result_dfs = list()

        min_id = 0
        for _id, timeseries_df in self._tsc_df.groupby(by=TSCDataFrame.tsc_id_idx_name):

            if pd.isnull(timeseries_df.delta_time):
                new_df = split_irregular_time_series(timeseries_df, min_id=min_id)
            else:
                # reset time series ID
                new_df = timeseries_df
                new_df.index = new_df.index.set_levels([min_id], 0)

            if new_df is not None:
                min_id = max(new_df.ids) + 1
                result_dfs.append(new_df)
            else:
                if not drop_samples:
                    raise RuntimeError(
                        "BUG: DataFrame is None while drop_samples=False. Please report."
                    )

        if result_dfs:
            self._tsc_df = pd.concat(result_dfs, axis=0)
            self._tsc_df = self._tsc_df.tsc.assign_ids_sequential()

            if np.isnan(np.asarray(self._tsc_df.delta_time)).any():
                warnings.warn(
                    "The function 'assign_ids_const_delta' was unsuccessful "
                    "to remove all irregular time series. Please "
                    "consider to report case."
                )

            return self._tsc_df
        else:
            return None

    def shift_matrices(
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
            If time series collection has no constant time delta.

        See Also
        --------
        :py:class:`DMDFull`

        """

        self.check_const_time_delta()

        # can be implemented if required:
        self.check_required_min_timesteps(required_min_timesteps=2)

        ts_counts = self._tsc_df.n_timesteps

        if is_integer(ts_counts):
            ts_counts = pd.Series(
                np.ones(self._tsc_df.n_timeseries, dtype=int) * ts_counts,
                index=self._tsc_df.ids,
            )

        assert isinstance(ts_counts, pd.Series)  # for mypy

        n_shift_snapshots = (ts_counts.subtract(1)).sum()
        insert_indices = np.append(0, (ts_counts.subtract(1)).cumsum().to_numpy())

        assert len(insert_indices) == self._tsc_df.n_timeseries + 1

        if snapshot_orientation == "col":
            shift_left = np.zeros([self._tsc_df.n_features, n_shift_snapshots])
        elif snapshot_orientation == "row":
            shift_left = np.zeros([n_shift_snapshots, self._tsc_df.n_features])
        else:
            raise ValueError(f"snapshot_orientation={snapshot_orientation} not known")

        shift_right = np.zeros_like(shift_left)

        # NOTE: if this has performance issues or memory issues, then it may be beneficial
        # to do the whole thing with boolean indexing

        # TODO: maybe three is a better readable code using pandas' functionatlity?
        #  e.g. shift https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html
        #  could also be more efficient than cmp to itertimeseries()
        for i, (id_, ts_df) in enumerate(self._tsc_df.itertimeseries()):
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
    # copy content to doc/source/devapi.rst
    for i in dir(TSCAccessor):
        if not i.startswith("_"):
            print(f"   .. automethod:: {i}")
