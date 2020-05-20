#!/usr/bin/env python3

from typing import Generator, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

from datafold.pcfold import TSCDataFrame
from datafold.utils.general import is_df_same_index, series_if_applicable


class TSCMetric(object):
    """Compute metrics for time series collection data.

    Parameters
    ----------

    metrics

        * "rmse" - root mean squared error,
        * "rrmse" - relative root mean squared error
        * "mse" - mean squared error,
        * "mae" - mean absolute error,
        * "max" maximum error,
        * "l2" - Eucledian norm

    mode
        compute metric per "timeseries", "timestep" or "feature"

    scaling
        Prior scaling (useful for heterogeneous time series features).

        * "id" - no scaling,
        * "min-max" - each feature is scaled into (0, 1) range,
        * "standard" - remove mean and scale to unit variance for each feature,
        * "l2_normalize" - divide each feature by Euclidean norm

    References
    ----------

    "rrmse" is taken from :cite:`le_clainche_higher_2017`

    """

    _cls_valid_modes = ["timeseries", "timestep", "feature"]
    _cls_valid_metrics = ["rmse", "rrmse", "mse", "mae", "max", "l2"]
    _cls_valid_scaling = ["id", "min-max", "standard", "l2_normalize"]

    def __init__(self, metric: str, mode: str, scaling: str = "id"):

        mode = mode.lower()
        metric = metric.lower()

        if metric in self._cls_valid_metrics:
            self.metric = self._metric_from_str_input(metric)
        else:
            raise ValueError(
                f"Invalid metric={mode}. Choose from {self._cls_valid_metrics}"
            )

        if mode in self._cls_valid_modes:
            self.mode = mode
        else:
            raise ValueError(
                f"Invalid mode={mode}. Choose from {self._cls_valid_modes}"
            )

        self.scaling = self._select_scaling(name=scaling)

    def _select_scaling(self, name):

        if name == "id":
            return None
        elif name == "min-max":
            return MinMaxScaler()
        elif name == "standard":
            return StandardScaler()
        elif name == "l2_normalize":
            return Normalizer(norm="l2")
        else:
            raise ValueError(
                f"scaling={name} is not known. Choose from {self._cls_valid_scaling}"
            )

    def _scaling(self, y_true: TSCDataFrame, y_pred: TSCDataFrame):

        # it is checked before that y_true and y_pred indices/columns are identical
        index, columns = y_true.index, y_true.columns

        # first normalize y_true, afterwards (with the same factors from y_true!) y_pred
        if self.scaling is not None:  # is None if scaling is identity
            y_true = self.scaling.fit_transform(y_true)
            y_pred = self.scaling.transform(y_pred.to_numpy())

            y_true = TSCDataFrame(y_true, index=index, columns=columns)
            y_pred = TSCDataFrame(y_pred, index=index, columns=columns)

        return y_true, y_pred

    def _l2_metric(
        self, y_true, y_pred, sample_weight=None, multioutput="uniform_average"
    ):

        diff = y_true - y_pred

        if sample_weight is not None:
            diff = sample_weight[:, np.newaxis] * diff

        l2_norm = np.linalg.norm(diff, axis=0)

        if multioutput == "uniform_average":
            l2_norm = np.mean(l2_norm)

        return l2_norm

    def _rmse_metric(
        self, y_true, y_pred, sample_weight=None, multioutput="uniform_average"
    ):
        # TODO: [minor] when upgrading to scikit-learn 0.22 mean_squared error has a
        #  keyword "squared", if set to False, this computes the RMSE directly

        mse_error = metrics.mean_squared_error(
            y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
        )
        return np.sqrt(mse_error)

    def _rrmse_metric(
        self, y_true, y_pred, sample_weight=None, multioutput="uniform_average"
    ):
        """Metric from :cite:`le_clainche_higher_2017`
        """

        if multioutput == "uniform_average":
            norm_ = np.sum(np.square(np.linalg.norm(y_true, axis=1)))
        else:  # multioutput == "raw_values":
            norm_ = np.sum(np.square(y_true), axis=0)

        if (np.asarray(norm_) <= 1e-14).any():
            raise RuntimeError(
                f"norm factor(s) are too small for rrmse \n norm_factor = {norm_}"
            )

        mse_error = metrics.mean_squared_error(
            y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
        )

        mse_error_relative = np.divide(mse_error, norm_)
        return np.sqrt(mse_error_relative)

    def _max_error(
        self, y_true, y_pred, sample_weight=None, multioutput="uniform_average"
    ):
        """Wrapper for :class:`sklean.metrics.max_error` to allow `sample_weight` and
        `multioutput` arguments (both have not effect).
        """

        # fails if y is multioutput
        return metrics.max_error(y_true=y_true, y_pred=y_pred)

    def _metric_from_str_input(self, error_metric: str):

        error_metric = error_metric.lower()

        if error_metric == "rmse":  # root mean squared error
            error_metric_handle = self._rmse_metric
        elif error_metric == "rrmse":
            error_metric_handle = self._rrmse_metric
        elif error_metric == "mse":
            error_metric_handle = metrics.mean_squared_error
        elif error_metric == "mae":
            error_metric_handle = metrics.mean_absolute_error
        elif error_metric == "max":
            error_metric_handle = self._max_error
        elif error_metric == "l2":
            error_metric_handle = self._l2_metric
        else:
            raise ValueError(f"Metric {error_metric} not known. Please report bug.")

        return error_metric_handle

    def _is_scalar_multioutput(self, multioutput) -> bool:
        # Return True if there is only one column (because features are averaged)
        if (
            isinstance(multioutput, str) and multioutput == "uniform_average"
        ) or isinstance(multioutput, np.ndarray):
            # array -> average with weights
            scalar_score = True

        elif multioutput == "raw_values":
            scalar_score = False

        else:
            raise ValueError(f"Illegal argument multioutput='{multioutput}'")
        return scalar_score

    def _single_column_name(self, multioutput) -> list:

        assert self._is_scalar_multioutput(multioutput)

        if isinstance(multioutput, str) and multioutput == "uniform_average":
            column = ["metric_uniform_average"]
        elif isinstance(multioutput, np.ndarray):
            column = ["metric_user_weights"]
        else:
            raise ValueError(f"Illegal argument of multioutput={multioutput}")

        return column

    def _metric_per_timeseries(
        self,
        y_true: TSCDataFrame,
        y_pred: TSCDataFrame,
        sample_weight=None,
        multioutput="uniform_average",
    ) -> Union[pd.Series, pd.DataFrame]:

        if sample_weight is not None:
            # same length of time series to have mapping
            # sample_weight -> time step of time series (can be a different time value)
            y_true.tsc.check_timeseries_same_length()

            if sample_weight.shape[0] != y_true.n_timesteps:
                raise ValueError(
                    f"'sample_weight' length (={len(sample_weight)}) "
                    f"does not match the number of time steps (={y_true.n_timesteps})"
                )

        if self._is_scalar_multioutput(multioutput=multioutput):

            column = self._single_column_name(multioutput=multioutput)

            # Make in both cases a DataFrame and later convert to Series in the scalar
            # case this allows to use .loc[i, :] in the loop
            error_per_timeseries = pd.DataFrame(
                np.nan, index=y_true.ids, columns=column
            )
        else:
            error_per_timeseries = pd.DataFrame(
                np.nan, index=y_true.ids, columns=y_true.columns.to_list(),
            )

        for i, y_true_single in y_true.itertimeseries():
            y_pred_single = y_pred.loc[i, :]

            error_per_timeseries.loc[i, :] = self.metric(
                y_true_single,
                y_pred_single,
                sample_weight=sample_weight,
                multioutput=multioutput,
            )

        return series_if_applicable(error_per_timeseries)

    def _metric_per_feature(
        self,
        y_true: TSCDataFrame,
        y_pred: TSCDataFrame,
        sample_weight=None,
        multioutput="raw_values",
    ):
        # Note: score per feature is never a multioutput-average, because a feature is
        # seen as a scalar quantity

        if sample_weight is not None:
            if sample_weight.shape[0] != y_true.shape[0]:
                raise ValueError(
                    f"'sample_weight' length (={sample_weight.shape[0]}) "
                    f"does not match the number of feature values "
                    f"(y.shape[0]={y_true.shape[0]})"
                )

        metric_per_feature = self.metric(
            y_true.to_numpy(),
            y_pred.to_numpy(),
            sample_weight=sample_weight,
            multioutput="raw_values",  # raw_values to tread every feature separately
        )

        metric_per_feature = pd.Series(metric_per_feature, index=y_true.columns,)
        return metric_per_feature

    def _metric_per_timestep(
        self,
        y_true: TSCDataFrame,
        y_pred: TSCDataFrame,
        sample_weight=None,
        multioutput="uniform_average",
    ):

        if sample_weight is not None:
            # sample weights -> each time series has a different weight

            # Currently, all time series must have the same time values to have the same
            # length for each time step
            y_true.tsc.check_timeseries_same_length()

            # the weight, must be as long as the time series
            if sample_weight.shape[0] != y_true.n_timeseries:
                raise ValueError(
                    f"'sample_weight' shape (={sample_weight.shape[0]}) "
                    f"does not match the number of time series (={y_true.n_timeseries})."
                )

        time_indices = pd.Index(y_true.time_values(), name="time")

        if self._is_scalar_multioutput(multioutput=multioutput):
            column = self._single_column_name(multioutput=multioutput)

            # Make in both cases a DataFrame and later convert to Series in the scalar
            # case this allows to use .loc[i, :] in the loop
            metric_per_time = pd.DataFrame(np.nan, index=time_indices, columns=column)

        else:
            metric_per_time = pd.DataFrame(
                np.nan, index=time_indices, columns=y_true.columns.to_list()
            )

        metric_per_time.index = metric_per_time.index.set_names(
            TSCDataFrame.tsc_time_idx_name
        )

        idx_slice = pd.IndexSlice
        for t in time_indices:

            y_true_t = pd.DataFrame(y_true.loc[idx_slice[:, t], :])
            y_pred_t = pd.DataFrame(y_pred.loc[idx_slice[:, t], :])

            metric_per_time.loc[t, :] = self.metric(
                y_true_t,
                y_pred_t,
                sample_weight=sample_weight,
                multioutput=multioutput,
            )

        return series_if_applicable(metric_per_time)

    def __call__(
        self,
        y_true: TSCDataFrame,
        y_pred: TSCDataFrame,
        sample_weight: Optional[np.ndarray] = None,
        multioutput: Union[str, np.ndarray] = "raw_values",
    ) -> Union[pd.Series, pd.DataFrame]:
        """Compute metric between two time series collections.
        
        Parameters
        ----------
        y_true
            Ground truth time series collection (basis for scaling), of shape
            `(n_samples, n_features)`.

        y_pred
            Predicted time series (the same scaling as for `y_true` will be applied),
            with exact same index (`ID` and `time` and columns as `y_true`).

        sample_weight
            Gives samples individual weights depending on the `mode`.

            * `mode=timeseries` array of shape `(n_timesteps,)`. Each time step has a \
               different weight (note that time values can be different).
            * `mode=feature` array of shape `(n_samples,)`. Each feature sample has a \
               different weight.
            * `mode=timestep` array of shape `(n_timeseries,)`. Each time series has \
               a different weight.

        multioutput
            Handling of metric, if evaluated over multiple features (columns), specify
            how to weight each feature. The parameter is ignored for `mode=feature`,
            because then each feature is a single output.

            * "raw_values" - returns metric per feature (i.e., the metric is not reduced)
            * "uniform_average" - returns metric averaged over all features averaged with
              uniform weight
            * ``numpy.ndarray`` of shape `(n_features,)` - returns metric for all \
               features averaged with specified weights

        Returns
        -------
        Union[pd.Series, pandas.DataFrame]
            metric evaluations, `pandas.DataFrame` for `multioutput=raw_values`

        Raises
        ------
        TSCException
            If not all values are finite in `y_true` or `y_pred` or if \
            :class:`TSCDataFrame` properties do not allow for a `sample_weight` argument.
        """

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)

            if sample_weight.ndim != 1:
                raise ValueError("'sample_weight' must be an 1-dim. array")

        # checks:
        y_true.tsc.check_finite()
        y_pred.tsc.check_finite()
        if not is_df_same_index(
            y_true, y_pred, check_index=True, check_column=True, handle="return"
        ):
            raise ValueError("Indices between 'y_pred' and 'y_true' must be equal.")

        # scaling:
        y_true, y_pred = self._scaling(y_true=y_true, y_pred=y_pred)

        # compute metric depending on mode:
        if self.mode == "timeseries":
            metric_result = self._metric_per_timeseries(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multioutput=multioutput,
            )

        elif self.mode == "timestep":
            metric_result = self._metric_per_timestep(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multioutput=multioutput,
            )

        elif self.mode == "feature":
            metric_result = self._metric_per_feature(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multioutput="raw_values",
            )
        else:
            raise ValueError(f"Invalid mode={self.mode}. Please report bug.")

        if isinstance(metric_result, pd.Series):
            assert not metric_result.isnull().any()
        elif isinstance(metric_result, pd.DataFrame):
            assert not metric_result.isnull().any().any()
        else:
            raise RuntimeError(
                f"Unknown return type {type(metric_result)}. Please report bug."
            )

        return metric_result


class TSCScoring(object):
    """Create scoring function from :class:`.TSCMetric`.

    Parameters
    ----------
    tsc_metric
        Time series collections metric.

    greater_is_better
        If True, the metric measures accuracy, else the metric measures the error.

    **metric_kwargs
        keyword arguments "sample_weight" and "multioutput" for
        :py:meth:`TSCMetric.__call__`

    Notes
    -----
    According to scikit-learn a `score` is a scalar value where higher values are \
    better than lower return values \
    (`ref <https://scikit-learn.org/stable/modules/model_evaluation.html>`_). This means:

        * Usually :class:`.TSCMetric` returns a vector metric with multiple components
          (metric per time series, timestep or feature). Therefore, the metric values
          must be "compressed" again to obtain a single score value.
        * Currently, all metrics measure the error, to comply with "higher score values
          are better" the metric values are negated.
    """

    def __init__(self, tsc_metric: TSCMetric, greater_is_better=False, **metric_kwargs):
        self.tsc_metric = tsc_metric
        self.metric_kwargs = metric_kwargs
        self.greater_is_better = greater_is_better

    def __call__(
        self,
        y_true: TSCDataFrame,
        y_pred: TSCDataFrame,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """Computes score between two time series collections.

        Parameters
        ----------
        y_true
            Ground truth time series data.

        y_pred
            Predicted time series data.

        sample_weight
            Not to be confused with parameter `samples_weight` in
            :py:meth:`TSCMetric.__call__`.

            The metric values (usually multiple values, depending on mode) can be weighted
            for the score:

            * `TSCMetric.mode=feature` - weight array of shape `(n_feature,)`
            * `TSCMetric.mode=timeseries` - weight array of shape `(n_timeseries,)`
            * `TSCMetric.mode=time` - weight array of shape `(n_timesteps,)`

        Returns
        -------
        :class:`float`
            score
        """

        eval_tsc_metric: pd.Series = self.tsc_metric(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=self.metric_kwargs.get("sample_weight", None),
            multioutput=self.metric_kwargs.get("multioutput", "uniform_average"),
        )

        eval_tsc_metric = series_if_applicable(eval_tsc_metric)

        if isinstance(eval_tsc_metric, pd.DataFrame):
            raise ValueError(
                "The TSCMetric must be configured that multioutputs (multiple feature "
                "columns) are weighted. Provide in 'multioutput' a string 'uniform' or "
                "an array with individual weights. "
            )

        if sample_weight is None:
            score = np.mean(eval_tsc_metric.to_numpy())
        elif isinstance(sample_weight, np.ndarray):
            assert len(sample_weight) == len(eval_tsc_metric)
            score = np.average(eval_tsc_metric.to_numpy(), weights=sample_weight)
        else:
            raise TypeError(f"sample_weight={sample_weight} is invalid.")

        if self.greater_is_better:
            factor = 1
        else:
            factor = -1

        return factor * float(score)


class TSCKfoldSeries(object):
    """K-fold splits on entire time series.

    Both the training and the test set consist of time series in its original length.
    Therefore, to perform the split, the time series collection must consist of
    multiple time series.

    Parameters
    ----------
    n_splits
        The number of splits.

    shuffle
        If True, the time series are shuffled.

    random_state
        Use fixed seed if `shuffle=True`.
    """

    def __init__(
        self, n_splits=3, shuffle: bool = False, random_state: Optional[int] = None
    ):
        self.kfold_splitter = KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def split(
        self, X: TSCDataFrame, y=None, groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yields k-folds of training and test indices of time series collection.

        Parameters
        ----------
        X
            The time series collection to split.

        y: None
            ignored

        groups: None
            ignored

        Yields
        ------
        numpy.ndarray
            train indices

        numpy.ndarray
            test indices

        Raises
        ------
        NotImplementedError
            If time series have not equal length.
        """
        if not X.is_equal_length():
            raise NotImplementedError(
                "Currently, all time series are required to have the same length for "
                "this method. This can be generalized, contributions welcome."
            )

        n_time_series = X.n_timeseries
        len_time_series = X.n_timesteps
        n_samples = X.shape[0]

        indices_matrix = np.arange(n_samples).reshape([n_time_series, len_time_series])

        # uses the indices as samples and splits along the time series
        # the indices (rows) are then collected and can be used to select from X
        for train, test in self.kfold_splitter.split(indices_matrix):
            train_indices = indices_matrix[train].flatten()
            test_indices = indices_matrix[test].flatten()

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of splits, which are also the number of cross-validation iterations.

        All parameter are ignored to align with scikit-learn's function.

        Parameters
        ----------
        X
            ignored

        y
            ignored

        groups
            ignored

        Returns
        -------
        """
        return self.kfold_splitter.get_n_splits(X=X, y=y, groups=groups)


class TSCKFoldTime(object):
    """K-fold splits on time values.

    The splits are along the time axis. This means that the time series collection can
    also consist of only a single time series. Note that if a block is taken from
    testing, then this results in more training series as in the original time series
    collection. For example, for a single time series this would result in two training
    time series and one test time series.

    Parameters
    ----------

    n_splits
        The number of splits.
    """

    def __init__(self, n_splits: int = 3):
        self.kfold_splitter = KFold(n_splits=n_splits, shuffle=False, random_state=None)

    def split(self, X: TSCDataFrame, y=None, groups=None):
        """Yields k-folds of training and test indices of time series collection.

        Parameters
        ----------
        X
            data to split

        y: None
            ignored

        groups: None
            ignored

        Yields
        ------
        numpy.ndarray
            train indices

        numpy.ndarray
            test indices
        """
        if not X.is_same_time_values():
            raise NotImplementedError(
                "Currently, each time series must have the same " "time indices."
            )

        n_samples = X.shape[0]
        indices_matrix = np.arange(n_samples).reshape(
            [X.n_timesteps, X.n_timeseries], order="F"
        )

        for train, test in self.kfold_splitter.split(indices_matrix):
            train_indices = indices_matrix[train].flatten()
            test_indices = indices_matrix[test].flatten()
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Number of splits, which are also the number of cross-validation iterations.

        All parameter are ignored to align with scikit-learn's function.

        Parameters
        ----------
        X
            ignored

        y
            ignored

        groups
            ignored

        Returns
        -------
        """
        return self.kfold_splitter.get_n_splits(X=X, y=y, groups=groups)


def kfold_cv_reassign_ids(X: TSCDataFrame, train_indices, test_indices):
    """Re-assigns time series ids based on training and test indices.

    Returns
    -------
    TSCDataFrame
        The training time series collection.

    TSCDataFrame
        The test time series collection.
    """
    # mark train samples with 0 and test samples with 1
    mask_train_test = np.zeros(X.shape[0])
    mask_train_test[test_indices] = 1

    # usage of np.diff -> dectect changes in
    # i) new fold or ii) new ID (i.e. time series)
    # -- both change detections are required to reassign new IDs

    # i) detect switch between test / train
    change_fold_indicator = np.append(0, np.diff(mask_train_test)).astype(np.bool)

    # ii) detect switch of new ID
    change_id_indicator = np.append(
        0, np.diff(X.index.get_level_values(TSCDataFrame.tsc_id_idx_name))
    ).astype(np.bool)

    # cumulative sum of on or the other change and reassign IDs
    id_cum_sum_mask = np.logical_or(change_fold_indicator, change_id_indicator)
    new_ids = np.cumsum(id_cum_sum_mask)

    reassigned_ids_idx = pd.MultiIndex.from_arrays(
        arrays=(new_ids, X.index.get_level_values(TSCDataFrame.tsc_time_idx_name),)
    )

    splitted_tsc = TSCDataFrame.from_same_indices_as(
        X, values=X, except_index=reassigned_ids_idx
    )

    train_tsc = splitted_tsc.iloc[train_indices, :]
    test_tsc = splitted_tsc.iloc[test_indices, :]

    # asserts also assumption made in the algorithm (in hindsight)
    assert isinstance(train_tsc, TSCDataFrame)
    assert isinstance(test_tsc, TSCDataFrame)

    return train_tsc, test_tsc
