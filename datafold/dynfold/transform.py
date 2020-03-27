#!/usr/bin/env python3

import itertools
from typing import Union

import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from datafold.dynfold.base import FEATURE_NAME_TYPES, TRANF_TYPES, TSCTransformerMixIn
from datafold.pcfold.kernels import MultiquadricKernel
from datafold.pcfold.timeseries import TSCDataFrame
from datafold.pcfold.timeseries.collection import TSCException


class TSCQoiPreprocess(BaseEstimator, TSCTransformerMixIn):
    VALID_SCALE_NAMES = ["min-max", "standard"]

    def __init__(self, sklearn_transformer):
        """
        See
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
        for a list of preprocessing classes that can be used. A mandatory requirement
        (for now) is that it supports an inverse mapping.
        """

        self.sklearn_transformer = sklearn_transformer

        if not hasattr(self.sklearn_transformer, "transform") or not hasattr(
            sklearn_transformer, "inverse_transform"
        ):
            raise AttributeError(
                f"transform cls {self.sklearn_transformer} must provide a 'transform' "
                f"and 'inverse_transform' attribute"
            )

    @classmethod
    def from_name(cls, name):

        if name == "min-max":
            return cls(MinMaxScaler(feature_range=(0, 1), copy=True))
        elif name == "standard":
            return cls(StandardScaler(copy=True, with_mean=True, with_std=True))
        else:
            raise ValueError(
                f"name='{name}' is not known. Choose from {cls.VALID_SCALE_NAMES}"
            )

    def fit(self, X: TRANF_TYPES, y=None, **fit_params):

        X = self._validate_data(X)
        self._setup_features_fit(X, features_out="like_features_in")

        self.sklearn_transformer_fit_ = clone(
            estimator=self.sklearn_transformer, safe=True
        )

        X_intern = self._X_to_numpy(X)
        self.sklearn_transformer_fit_.fit(X_intern)

        return self

    def transform(self, X: TRANF_TYPES):
        check_is_fitted(self, "sklearn_transformer_fit_")

        X = self._validate_data(X)
        self._validate_features_transform(X)

        X_intern = self._X_to_numpy(X)
        values = self.sklearn_transformer_fit_.transform(X_intern)
        return self._same_type_X(X=X, values=values, set_columns=self.features_out_[1])

    def fit_transform(self, X: TRANF_TYPES, y=None, **fit_params):

        X = self._validate_data(X)

        self._setup_features_fit(X, features_out="like_features_in")

        self.sklearn_transformer_fit_ = clone(self.sklearn_transformer)
        values = self.sklearn_transformer_fit_.fit_transform(X)

        return self._same_type_X(X=X, values=values, set_columns=self.features_out_[1])

    def inverse_transform(self, X: TRANF_TYPES):
        X_intern = self._X_to_numpy(X)
        values = self.sklearn_transformer_fit_.inverse_transform(X_intern)
        return self._same_type_X(X=X, values=values, set_columns=self.features_in_[1])


class TSCIdentity(BaseEstimator, TSCTransformerMixIn):
    def __init__(self):
        pass

    def fit(self, X: TRANF_TYPES, y=None, **fit_params):

        X = self._validate_data(X)
        self._setup_features_fit(X, features_out="like_features_in")

        # Dummy attribute to indicate that fit was called
        self.is_fit_ = True

        return self

    def transform(self, X: TRANF_TYPES):

        check_is_fitted(self, "is_fit_")

        X = self._validate_data(X)
        self._validate_features_transform(X)

        return X

    def inverse_transform(self, X: TRANF_TYPES):

        check_is_fitted(self, "is_fit_")
        X = self._validate_data(X)
        self._validate_features_inverse_transform(X)
        return X


class TSCPrincipalComponent(PCA, TSCTransformerMixIn):
    def __init__(
        self,
        n_components=2,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
    ):
        super(TSCPrincipalComponent, self).__init__(
            n_components=n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state,
        )

    def fit(self, X: TRANF_TYPES, y=None, **fit_params):

        X = self._validate_data(X)

        self._setup_features_fit(
            X, features_out=[f"pca{i}" for i in range(self.n_components)]
        )

        X_intern = self._X_to_numpy(X)

        # validate happens here:
        return super(TSCPrincipalComponent, self).fit(X_intern, y=y)

    def transform(self, X: TRANF_TYPES):
        check_is_fitted(self)
        X = self._validate_data(X)

        self._validate_features_transform(X)

        X_intern = self._X_to_numpy(X)
        pca_data = super(TSCPrincipalComponent, self).transform(X_intern)
        return self._same_type_X(X, values=pca_data, set_columns=self.features_out_[1])

    def fit_transform(self, X, y=None):

        X = self._validate_data(X)

        self._setup_features_fit(
            X, features_out=[f"pca{i}" for i in range(self.n_components)]
        )

        X_intern = self._X_to_numpy(X)
        pca_values = super(TSCPrincipalComponent, self).fit_transform(X_intern, y=y)
        return self._same_type_X(
            X, values=pca_values, set_columns=self.features_out_[1]
        )

    def inverse_transform(self, X: TRANF_TYPES):
        self._validate_features_inverse_transform(X)

        X_intern = self._X_to_numpy(X)
        data_orig_space = super(TSCPrincipalComponent, self).inverse_transform(X_intern)

        return self._same_type_X(
            X, values=data_orig_space, set_columns=self.features_in_[1]
        )


class TSCTakensEmbedding(BaseEstimator, TSCTransformerMixIn):
    def __init__(
        self, lag: int = 0, delays: int = 10, frequency: int = 1,
    ):
        """
        fillin_handle:
            If 'value': all fill-ins will be set to this value
            If 'remove': all rows that are affected of fill-ins are removed.
        """

        self.lag = lag
        self.delays = delays
        self.frequency = frequency

    def _validate_parameter(self):
        from sklearn.utils.validation import check_scalar

        check_scalar(
            self.lag, name="lag", target_type=(np.integer, int), min_val=0, max_val=None
        )

        check_scalar(
            self.delays,
            name="delays",
            target_type=(np.integer, int),
            min_val=1,
            max_val=None,
        )

        check_scalar(
            self.frequency,
            name="delays",
            target_type=(np.integer, int),
            min_val=1,
            max_val=None,
        )

        if self.frequency > 1 and self.delays <= 1:
            raise ValueError(
                f"If frequency (={self.frequency} is larger than 1, "
                f"then number for delays (={self.delays}) has to be larger "
                "than 1)."
            )

    def _setup_delay_indices_array(self):
        # zero delay (original data) is not contained as an index
        # This makes it easier to just delay through the indices (instead of computing
        # the indices during the delay.
        return self.lag + (
            np.arange(1, (self.delays * self.frequency) + 1, self.frequency)
        )

    def _columns_to_type_str(self, X):
        # in case the column in not string it is important to transform it here to
        # string. Otherwise, There are mixed types (e.g. int and str), because the
        # delayed columns are all strings to indicate the delay number.
        X.columns = X.columns.astype(np.str)
        return X

    def _expand_all_delay_columns(self, cols):
        def expand():
            delayed_columns = list()
            for delay_idx in self.delay_indices_:
                _cur_delay_columns = list(
                    map(lambda q: ":d".join([q, str(delay_idx)]), cols.astype(str))
                )
                delayed_columns.append(_cur_delay_columns)
            return delayed_columns

        # the name of the original indices is not changed, therefore append the delay
        # indices to
        columns_names = cols.tolist() + list(itertools.chain(*expand()))

        return pd.Index(
            columns_names, dtype=np.str, copy=False, name=TSCDataFrame.IDX_QOI_NAME
        )

    def _validate_takens_properties(self, X):

        if not isinstance(X, TSCDataFrame):
            if X.index.nlevels == 1:
                # X is required to be TSC here
                X = TSCDataFrame.from_single_timeseries(X)
            else:
                X = TSCDataFrame(X)

        if not X.is_const_delta_time():
            raise TSCException.not_const_delta_time()

        return X

    def fit(self, X: FEATURE_NAME_TYPES, y=None, **fit_params):

        self._validate_parameter()
        X = self._validate_data(X, ensure_feature_name_type=True,)

        X = self._columns_to_type_str(X)
        X = self._validate_takens_properties(X)

        self.delay_indices_ = self._setup_delay_indices_array()
        features_out = self._expand_all_delay_columns(X.columns)

        # only TSCDataFrame works here
        self._setup_frame_input_fit(
            features_in=X.columns, features_out=features_out,
        )
        return self

    def transform(self, X: FEATURE_NAME_TYPES):

        X = self._validate_data(X, ensure_feature_name_type=True)
        X = self._validate_takens_properties(X)

        if (X.n_timesteps <= self.delay_indices_.max()).any():
            raise TSCException(
                f"Mismatch of delay and time series length. Shortest time series has "
                f"length {np.array(X.n_timesteps).min()} and maximum delay is "
                f"{self.delay_indices_.max()}."
            )

        X = self._columns_to_type_str(X)
        self._validate_features_transform(X)

        #################################
        ### Implementation staying in pandas using shift()
        ### This implementation is for many cases similarly fast as the numpy version
        ### below, but has a performance drop for high-dimensions (dim>500)
        # id_groupby = X.groupby(TSCDataFrame.IDX_ID_NAME)
        # concat_dfs = [X]
        #
        # for delay_idx in self.delay_indices_:
        #     shifted_data = id_groupby.shift(delay_idx, fill_value=np.nan)
        #     shifted_data = shifted_data.add_suffix(f":d{delay_idx}")
        #     concat_dfs.append(shifted_data)
        #
        # X = pd.concat(concat_dfs, axis=1)

        # if self.fillin_handle == "remove":
        #     # _TODO: use pandas.dropna()
        #     bool_idx = np.logical_not(np.sum(pd.isnull(X), axis=1).astype(np.bool))
        #     X = X.loc[bool_idx]

        # Implementation using numpy functions.

        # pre-allocate list
        delayed_timeseries = [pd.DataFrame([])] * len(X.ids)

        max_delay = max(self.delay_indices_)
        for idx, (_, df) in enumerate(X.groupby(TSCDataFrame.IDX_ID_NAME)):

            # use time series numpy block
            time_series_numpy = df.to_numpy()

            # max_delay determines the earliest sample that has no fill-in
            original_data = time_series_numpy[max_delay:, :]

            # select the data (row_wise) for each delay block
            # in last iteration "max_delay - delay == 0"
            delayed_data = np.hstack(
                [
                    time_series_numpy[max_delay - delay : -delay, :]
                    for delay in self.delay_indices_
                ]
            )

            # go back to DataFrame, and adapt the index be excluding removed indices
            df = pd.DataFrame(
                np.hstack([original_data, delayed_data]),
                index=df.index[max_delay:],
                columns=self.features_out_[1],
            )

            delayed_timeseries[idx] = df

        X = pd.concat(delayed_timeseries, axis=0)

        try:
            X = TSCDataFrame(X)
        except AttributeError:
            # simply return the pandas DataFrame then
            pass

        return X

    def inverse_transform(self, X: TRANF_TYPES):
        check_is_fitted(self)
        X = self._validate_data(X, ensure_feature_name_type=True)
        X = self._validate_takens_properties(X)
        self._validate_features_inverse_transform(X)

        return X.loc[:, self.features_in_[1]]


class TSCRadialBasis(BaseEstimator, TSCTransformerMixIn):
    def __init__(self, kernel=MultiquadricKernel(epsilon=1.0), exact_distance=True):
        """
        Parameters
        ----------
        kernel
        exact_distance
        """
        self.kernel = kernel
        self.exact_distance = exact_distance

    def fit(self, X: TRANF_TYPES, y=None, **fit_kwargs):
        X = self._validate_data(X)

        self.centers_ = self._X_to_numpy(X)
        n_centers = self.centers_.shape[0]
        self._setup_features_fit(X, [f"rbf{i}" for i in range(n_centers)])

        return self

    def transform(self, X: TRANF_TYPES):
        check_is_fitted(self, attributes=["centers_"])
        X = self._validate_data(X)
        self._validate_features_transform(X)

        X_intern = self._X_to_numpy(X)

        rbf_coeff = self.kernel(
            self.centers_,
            X_intern,
            dist_backend="brute",
            dist_backend_kwargs={"exact_numeric": self.exact_distance},
        )

        return self._same_type_X(X, values=rbf_coeff, set_columns=self.features_out_[1])

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        X_intern = self._X_to_numpy(X)

        # pdist case (more efficient than fit().transform() using cdist on self data
        rbf_coeff = self.kernel(
            X_intern,
            dist_backend="brute",
            dist_backend_kwargs={"exact_numeric": self.exact_distance},
        )

        return self._same_type_X(
            X=X, values=rbf_coeff, set_columns=self.features_out_[1]
        )

    def inverse_transform(self, X: TRANF_TYPES):
        self._validate_features_inverse_transform(X)

        if self._has_feature_names(X):
            rbf_coeff = X.to_numpy()
        else:
            rbf_coeff = X

        if not hasattr(self, "inv_coeff_matrix_"):
            # save inv_coeff_matrix_
            center_kernel = self.kernel(
                self.centers_,
                dist_backend="brute",
                dist_backend_kwargs={"exact_numeric": self.exact_distance},
            )
            self.inv_coeff_matrix_ = np.linalg.lstsq(
                center_kernel, self.centers_, rcond=None
            )[0]

        X_inverse = rbf_coeff @ self.inv_coeff_matrix_
        return self._same_type_X(X, values=X_inverse, set_columns=self.features_in_[1])


@NotImplementedError  # TODO: implement if required...
class __TSCFiniteDifference(object):

    # TODO: provide longer shifts? This could give some average of slow and fast
    #  variables...

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
