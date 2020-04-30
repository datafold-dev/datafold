#!/usr/bin/env python3

import itertools
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.utils.validation import check_is_fitted, check_scalar

from datafold.decorators import warn_experimental_class
from datafold.dynfold.base import DataFrameType, TransformType, TSCTransformerMixIn
from datafold.pcfold import MultiquadricKernel, TSCDataFrame
from datafold.pcfold.kernels import PCManifoldKernel


class TSCFeaturePreprocess(BaseEstimator, TSCTransformerMixIn):
    """Apply a scikit-learn preprocess algorithms but generalize to time series
    collections.

    Often scikit-learn performs "pandas.DataFrame in -> numpy.ndarray out". This wrapper
    generalizes this to "pandas.DataFrame in -> pandas.DataFrame out".

    Parameters
    ----------

    sklearn_transformer
        See `here <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing>`_
        for a list of possible preprocessing algorithms.
    """

    _cls_valid_scale_names = ("min-max", "standard")

    def __init__(self, sklearn_transformer):
        self.sklearn_transformer = sklearn_transformer

    @classmethod
    def from_name(cls, name: str) -> "TSCFeaturePreprocess":
        """Select common transform algorithms by name.

        Parameters
        ----------
        name
            - "min-max" - :class:`sklearn.preprocessing.MinMaxScaler`
            - "standard" - :class:`sklearn.preprocessing.StandardScaler`

        Returns
        -------
        TSCFeaturePreprocess
            new instance
        """
        if name == "min-max":
            return cls(MinMaxScaler(feature_range=(0, 1), copy=True))
        elif name == "standard":
            return cls(StandardScaler(copy=True, with_mean=True, with_std=True))
        else:
            raise ValueError(
                f"name='{name}' is not known. Choose from {cls._cls_valid_scale_names}"
            )

    def fit(self, X: TransformType, y=None, **fit_params) -> "TSCFeaturePreprocess":
        """Handle fit to internal transform sklearn object.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Training data with shape `(n_samples, n_features)`.

        y: None
            ignored

        Returns
        -------
        TSCFeaturePreprocess
            self
        """

        if not hasattr(self.sklearn_transformer, "transform"):
            raise TypeError("sklearn object has to transform attribute")

        X = self._validate_data(X)
        self._setup_features_fit(X, features_out="like_features_in")

        self.sklearn_transformer_fit_ = clone(
            estimator=self.sklearn_transformer, safe=True
        )

        X_intern = self._X_to_numpy(X)
        self.sklearn_transformer_fit_.fit(X_intern)

        return self

    def transform(self, X: TransformType):
        """Handle transform to internal sklearn object.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data to transform with shape `(n_samples, n_features)`.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type and shape as `X`
        """

        check_is_fitted(self, "sklearn_transformer_fit_")

        X = self._validate_data(X)
        self._validate_feature_input(X, direction="transform")

        X_intern = self._X_to_numpy(X)
        values = self.sklearn_transformer_fit_.transform(X_intern)
        return self._same_type_X(X=X, values=values, set_columns=self.features_out_[1])

    def fit_transform(self, X: TransformType, y=None, **fit_params):
        """Handle fit_transform to internal sklearn object.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Training data to transform with shape `(n_samples, n_features)`.

        y: None
            ignored

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type and shape as `X`
        """

        X = self._validate_data(X)

        self._setup_features_fit(X, features_out="like_features_in")

        self.sklearn_transformer_fit_ = clone(self.sklearn_transformer)
        values = self.sklearn_transformer_fit_.fit_transform(X)

        return self._same_type_X(X=X, values=values, set_columns=self.features_out_[1])

    def inverse_transform(self, X: TransformType):
        """Handle inverse_transform to internal sklearn object.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data to map back with shape `(n_samples, n_features)`.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type and shape as `X`
        """

        if not hasattr(self.sklearn_transformer, "inverse_transform"):
            raise TypeError(
                "sklearn object does not provide inverse_transform function"
            )

        X_intern = self._X_to_numpy(X)
        values = self.sklearn_transformer_fit_.inverse_transform(X_intern)
        return self._same_type_X(X=X, values=values, set_columns=self.features_in_[1])


class TSCIdentity(BaseEstimator, TSCTransformerMixIn):
    """Dummy transformer for testing or  as a "passthrough" placeholder.

    Attributes
    ----------
    is_fit_ : bool
        True if fit has been called.
    """

    def __init__(self):
        pass

    def fit(self, X: TransformType, y=None, **fit_params):
        """Passthrough data and set internals for validation.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data with shape `(n_samples, n_features)`.

        y: None
            ignored

        Returns
        -------
        TSCIdentity
            self
        """
        X = self._validate_data(X)
        self._setup_features_fit(X, features_out="like_features_in")

        # Dummy attribute to indicate that fit was called
        self.is_fit_ = True

        return self

    def transform(self, X: TransformType) -> TransformType:
        """Passthrough data and validate features shape.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data with shape `(n_samples, n_features)` to passthrough.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type and hape as `X`
        """
        check_is_fitted(self, "is_fit_")

        X = self._validate_data(X)
        self._validate_feature_input(X, direction="transform")

        return X

    def inverse_transform(self, X: TransformType):
        """Passthrough data and validate features shape.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data to passthrough with shape `(n_samples, n_features)`.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type and shape as `X`
        """
        check_is_fitted(self, "is_fit_")
        X = self._validate_data(X)
        self._validate_feature_input(X, direction="inverse_transform")
        return X


class TSCPolynomialFeatures(PolynomialFeatures, TSCTransformerMixIn):
    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        order: str = "C",
    ):
        super(TSCPolynomialFeatures, self).__init__(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
            order=order,
        )

    def fit(self, X: TransformType, y=None, **fit_params):
        X = self._validate_data(X)
        # TODO: need to remove the ID state powers_ otherwise they are in twice... (or
        #  make as option...)

        super(TSCPolynomialFeatures, self).fit(self._X_to_numpy(X), y=y)

        # Note1: the n_output_features_ is quite similar to TSCTransformerMixIn,
        # but actually is set inside PolynomialFeatures
        # Note3: the n_output_features_ are corrected by X.shape[1] because we can skip
        self._setup_features_fit(
            X, features_out=[f"poly{i}" for i in range(self.n_output_features_)],
        )
        return self

    def transform(self, X: TransformType):
        check_is_fitted(self)
        X = self._validate_data(X)
        self._validate_feature_input(X, direction="transform")

        poly_data = super(TSCPolynomialFeatures, self).transform(self._X_to_numpy(X))
        return self._same_type_X(X, values=poly_data, set_columns=self.features_out_[1])

    def fit_transform(self, X, y=None, **fit_params):
        X = self._validate_data(X)

        pca_values = super(TSCPolynomialFeatures, self).fit_transform(
            self._X_to_numpy(X), y=y
        )

        self._setup_features_fit(
            X, features_out=[f"poly{i}" for i in range(self.n_output_features_)]
        )

        return self._same_type_X(
            X, values=pca_values, set_columns=self.features_out_[1]
        )


class TSCApplyLambdas(BaseEstimator, TSCTransformerMixIn):
    def __init__(self, lambda_list):
        self.lambda_list = lambda_list

    def fit(self, X: TransformType, y=None):

        if isinstance(X, np.ndarray):
            raise NotImplementedError(
                "Currently not implemented for numpy.ndraay. If required please open "
                "a Gitlab issue."
            )

        X = self._validate_data(X, ensure_feature_name_type=True)

        features_out = [
            f"{feature_name}_lambda{i}"
            for feature_name in X.columns
            for i in range(len(self.lambda_list))
        ]

        self._setup_features_fit(X, features_out=features_out)
        return self

    def transform(self, X: TransformType):
        check_is_fitted(self)
        X = self._validate_data(X, ensure_feature_name_type=True)
        self._validate_feature_input(X, direction="transform")

        lambdas_applied = list()

        for i, _lambda in enumerate(self.lambda_list):

            lambda_result = X.apply(func=_lambda, axis=0, raw=True)
            lambda_result.columns = pd.Index(
                [f"{feature_name}_lambda{i}" for feature_name in X.columns]
            )

            lambdas_applied.append(lambda_result)

        X_transformed = pd.concat(lambdas_applied, axis=1)
        X_transformed.columns.name = TSCDataFrame.tsc_feature_col_name

        if isinstance(X, TSCDataFrame):
            return TSCDataFrame(X_transformed)
        else:
            return X_transformed


class TSCPrincipalComponent(PCA, TSCTransformerMixIn):
    """Compute principal components from data and also allow time series collection data.

    ``TSCPrincipalComponent`` subclasses ``PCA`` from scikit-learn to generalize the
    input and output of pandas DataFrames (including TSCDataFrame). All input
    parameters remain the same. For documentation please visit:

    * `PCA docu <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_ # noqa
    * `PCA user guide <https://scikit-learn.org/stable/modules/decomposition.html#pca>`_
    """

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

    def fit(self, X: TransformType, y=None, **fit_params) -> "PCA":
        """Compute the principal component vectors from training data.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Training data with shape `(n_samples, n_features)`.

        y: None
            ignored

        Returns
        -------
        TSCPrincipalComponent
            self
        """

        X = self._validate_data(X)
        self._setup_features_fit(
            X, features_out=[f"pca{i}" for i in range(self.n_components)]
        )

        # validation happens here:
        super(TSCPrincipalComponent, self).fit(self._X_to_numpy(X), y=y)
        return self

    def transform(self, X: TransformType):
        """Apply dimension reduction by projecting the data on principal components.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Out-of-sample points with shape `(n_samples, n_features)` to perform dimension
            reduction on.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` with shape `(n_samples, n_components_)`
        """

        check_is_fitted(self)
        X = self._validate_data(X)

        self._validate_feature_input(X, direction="transform")
        pca_data = super(TSCPrincipalComponent, self).transform(self._X_to_numpy(X))
        return self._same_type_X(X, values=pca_data, set_columns=self.features_out_[1])

    def fit_transform(self, X: TransformType, y=None) -> TransformType:
        """Compute principal components from data and reduce dimension on same data.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Training data with shape `(n_samples, n_features)`.
            
        y: None
            ignored

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` with shape `(n_samples, n_components_)`
        """

        X = self._validate_data(X)
        self._setup_features_fit(
            X, features_out=[f"pca{i}" for i in range(self.n_components)]
        )

        pca_values = super(TSCPrincipalComponent, self).fit_transform(
            self._X_to_numpy(X), y=y
        )
        return self._same_type_X(
            X, values=pca_values, set_columns=self.features_out_[1]
        )

    def inverse_transform(self, X: TransformType):
        """Reconstruct the data to the original space.

        Parameters
        ----------
        X:
            Out-of-sample points with shape `(n_samples, n_components_)` to map back to
            original space.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` with shape `(n_samples, n_features)`
        """

        self._validate_feature_input(X, direction="inverse_transform")

        X_intern = self._X_to_numpy(X)
        data_orig_space = super(TSCPrincipalComponent, self).inverse_transform(X_intern)

        return self._same_type_X(
            X, values=data_orig_space, set_columns=self.features_in_[1]
        )


class TSCTakensEmbedding(BaseEstimator, TSCTransformerMixIn):
    """Perform Takens time delay embedding on time series collection data.

    Parameters
    ----------

    lag
        Number of time steps to lag before embedding past data.

    delays
        Number for delays to embed.

    frequency
        Time step frequency (every time step, every second, ...)

    Attributes
    ----------
    delay_indices_ : numpy.ndarray
        Delay indices (backwards in time) assuming a fixed time delta in the time series.

    min_timesteps_: int
        Minimum required time steps for each time series to have a single embedding
        vector.

    delta_time_fit_
        Time delta during measured during fit. This is primarily used to check that
        `transform` or `inverse_transform` data still have the same time delta for
        consistency.

    References
    ----------

    :cite:`rand_detecting_1981`
    """

    def __init__(
        self, lag: int = 0, delays: int = 10, frequency: int = 1,
    ):
        self.lag = lag
        self.delays = delays
        self.frequency = frequency

    def _validate_parameter(self):

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
            columns_names,
            dtype=np.str,
            copy=False,
            name=TSCDataFrame.tsc_feature_col_name,
        )

    def fit(self, X: DataFrameType, y=None, **fit_params) -> "TSCTakensEmbedding":
        """Compute delay indices based on settings and check if they work for the time
        series provided.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame
            Time series collection to check `delay_indices_` with.

        y: None
            ignored

        Returns
        -------
        TSCTakensEmbedding
            self

        Raises
        ------
        TSCException
            Time series collection restrictions in **X**: (1) time delta must be constant
            (2) all time series must have the minimum number of time samples to obtain
            one sample in the time delay embedding.

        """
        self._validate_parameter()

        self.delay_indices_ = self._setup_delay_indices_array()
        self.min_timesteps_ = max(self.delay_indices_) + 1

        X = self._validate_data(
            X,
            validate_tsc_kwargs={
                "ensure_const_delta_time": True,
                "ensure_min_timesteps": self.min_timesteps_,
            },
            ensure_feature_name_type=True,
        )

        X = self._columns_to_type_str(X)

        # save delta time during fit to check that time series collections in
        # transform have the same delta time
        self.delta_time_fit_ = X.delta_time

        features_out = self._expand_all_delay_columns(X.columns)

        self._setup_frame_input_fit(
            features_in=X.columns, features_out=features_out,
        )
        return self

    def transform(self, X: DataFrameType) -> DataFrameType:
        """Perform Takens time delay embedding for each time series in the collection.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame
            Time series (collection).

        Returns
        -------
        TSCDataFrame, pandas.DataFrame
            Each time series is shortend by the number of samples required for the
            delays. Type falls back to `pandas.DataFrame` if not a valid TSCDataFrame.

        Raises
        ------
        TSCException
            Time series collection restrictions in `X`: (1) time delta must be constant
            (2) all time series must have the minimum number of time samples to obtain
            one sample in the time delay embedding.
        """

        X = self._validate_data(
            X,
            validate_tsc_kwargs={
                # must be same time delta as during fit
                "ensure_delta_time": self.delta_time_fit_,
                "ensure_min_timesteps": self.min_timesteps_,
            },
            ensure_feature_name_type=True,
        )

        X = self._columns_to_type_str(X)
        self._validate_feature_input(X, direction="transform")

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

        for idx, (_, df) in enumerate(X.groupby(TSCDataFrame.tsc_id_idx_name)):

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

    def inverse_transform(self, X: TransformType) -> TransformType:
        """Remove time delayed features from columns of time delay embedded time
        series collection.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame
            time delayed data

        Returns
        -------
        TSCDataFrame, pandas.DataFrame
            same type and shape as `X`
        """
        check_is_fitted(self)

        X = self._validate_data(
            X,
            ensure_feature_name_type=True,
            # will only be checked if TSCDataFrame (pandas DataFrame is also legal and
            # won't be checked for delta time)
            validate_tsc_kwargs=dict(ensure_delta_time=self.delta_time_fit_),
        )
        self._validate_feature_input(X, direction="inverse_transform")

        return X.loc[:, self.features_in_[1]]


class TSCRadialBasis(BaseEstimator, TSCTransformerMixIn):
    """Represent data in terms coefficients of a set of radial basis functions.

    Parameters
    ----------

    kernel
        Radial basis kernel to compute the coeffients with.

    center_type
        Selection of what to take as centers during fit. 

        * `all_data` - all data are centers
        * `initial_condition` - take initial conditions as centers. Note in for this
           case the data `X` during fit must be a ``TSCDataFrame``.

    exact_distance
        An inexact distance computation increases the performance at the cost of
        numerical inaccuracies (~1e-7 for Euclidean distance, and ~1 e-14 for squared
        Eucledian distance).
    
    Attributes
    ----------

    centers_: numpy.ndarray
        The center points of the radial basis functions.

    inv_coeff_matrix_: np.ndarray
        Matrix to map RBF coefficients to original space. Computation is delayed
        until `inverse_transform` is called for the first time.
    """

    _cls_valid_center_types = ["all_data", "initial_condition"]

    def __init__(
        self,
        kernel: PCManifoldKernel = MultiquadricKernel(epsilon=1.0),
        center_type="all_data",
        exact_distance=True,
    ):
        self.kernel = kernel
        self.center_type = center_type
        self.exact_distance = exact_distance

    def _validate_center_type(self, center_type):
        if center_type not in self._cls_valid_center_types:
            raise ValueError(
                f"center_type={center_type} not valid. Choose from "
                f"{self._cls_valid_center_types} "
            )

    def fit(self, X: TransformType, y=None, **fit_kwargs) -> "TSCRadialBasis":
        """Set the point centers of the radial basis functions.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Point centers with shape (n_centers, n_features). Must be `TSCDataFrame` if
            center type is `initial_condition`.

        y: None
            ignored

        Returns
        -------
        TSCRadialBasis
            self
        """

        from datafold.pcfold import PCManifold

        X = self._validate_data(X)
        self._validate_center_type(center_type=self.center_type)

        from datafold.pcfold import PCManifold

        if self.center_type == "all_data":
            self.centers_ = self._X_to_numpy(X)
        else:  # self.center_type == "initial_condition":
            if not isinstance(X, TSCDataFrame):
                raise TypeError("Data 'X' must be TSCDataFrame")
            self.centers_ = X.initial_states().to_numpy()

        self.centers_ = PCManifold(
            self.centers_,
            kernel=self.kernel,
            dist_backend="brute",
            **dict(exact_numeric=self.exact_distance),
        )

        n_centers = self.centers_.shape[0]
        self._setup_features_fit(X, [f"rbf{i}" for i in range(n_centers)])

        return self

    def transform(self, X: TransformType) -> TransformType:
        """Transform data to coefficients of the radial basis functions.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data with shape `(n_samples, n_features)`.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` with shape `(n_samples, n_centers)`
        """
        check_is_fitted(self, attributes=["centers_"])
        X = self._validate_data(X)
        self._validate_feature_input(X, direction="transform")

        X_intern = self._X_to_numpy(X)
        rbf_coeff = self.centers_.compute_kernel_matrix(Y=X_intern)

        return self._same_type_X(X, values=rbf_coeff, set_columns=self.features_out_[1])

    def fit_transform(self, X, y=None, **fit_params):
        """Simultaneuously set the data to transform also to the point centers .

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Radial basis center points and data to transform with shape \
            `(n_samples, n_features)`

        y: None
            ignored

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` with shape `(n_samples, n_centers)`
        """
        self.fit(X)
        X_intern = self._X_to_numpy(X)
        self._validate_center_type(center_type=self.center_type)

        if self.center_type == "all_data":
            # compute pdist distance matrix, which is often more efficient
            rbf_coeff = self.centers_.compute_kernel_matrix()
        else:  # self.center_type == "initial_condition":
            rbf_coeff = self.centers_.compute_kernel_matrix(Y=X_intern)

        return self._same_type_X(
            X=X, values=rbf_coeff, set_columns=self.features_out_[1]
        )

    def inverse_transform(self, X: TransformType):
        """Transform radial basis coefficients back to the original function values.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Coefficient representation of the radial basis functions with shape \
            `(n_samples, n_center)`.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` with shape `(n_samples, n_features)`
        """
        self._validate_feature_input(X, direction="inverse_transform")

        if self._has_feature_names(X):
            rbf_coeff = X.to_numpy()
        else:
            rbf_coeff = X

        if not hasattr(self, "inv_coeff_matrix_"):
            # save inv_coeff_matrix_
            center_kernel = self.centers_.compute_kernel_matrix()
            self.inv_coeff_matrix_ = np.linalg.lstsq(
                center_kernel, self.centers_, rcond=None
            )[0]

        X_inverse = rbf_coeff @ self.inv_coeff_matrix_
        return self._same_type_X(X, values=X_inverse, set_columns=self.features_in_[1])


@warn_experimental_class
class TSCFiniteDifference(BaseEstimator, TSCTransformerMixIn):
    def __init__(self, spacing: Union[str, float] = "dt", diff_order=1, accuracy=2):
        self.spacing = spacing
        self.diff_order = diff_order
        self.accuracy = accuracy

    def fit(self, X: TransformType, y=None, **fit_params):

        X = self._validate_data(X, ensure_feature_name_type=False)

        if self._has_feature_names(X):
            features_out = [f"{col}_dt" for col in X.columns]
        else:
            features_out = [f"dt{i}" for i in np.arange(X.shape[1])]

        self._setup_features_fit(X=X, features_out=features_out)

        if self.spacing == "dt":

            if not isinstance(X, TSCDataFrame):
                raise TypeError("'spacing=dt' only works for time series collections")

            self.spacing_ = X.delta_time

            if isinstance(self.spacing_, pd.Series) or np.isnan(self.spacing_):
                raise ValueError(
                    "delta time (=spacing) of TSCDataFrame must be constant"
                )
        else:
            self.spacing_ = self.spacing

        check_scalar(
            self.spacing_,
            "spacing",
            target_type=(int, float, np.integer, np.floating),
            min_val=np.finfo(np.float64).eps,
            max_val=None,
        )
        self.spacing_ = float(self.spacing_)

        check_scalar(
            self.diff_order,
            "diff_order",
            target_type=(int, np.integer),
            min_val=1,
            max_val=None,
        )

        check_scalar(
            self.accuracy,
            name="accuracy",
            target_type=(int, np.integer),
            min_val=1,
            max_val=None,
        )

        return self

    def transform(self, X: TransformType) -> TransformType:

        from findiff import FinDiff

        check_is_fitted(self)
        X = self._validate_data(X)
        self._validate_feature_input(X=X, direction="transform")

        # first parameter is the axis along which to take the derivative
        # second parameter is the grid spacing
        # third parameter the derivative order
        # acc = order of accuracy (defaults to 2)
        dt_func = FinDiff(0, self.spacing_, self.diff_order, acc=self.accuracy)

        if isinstance(X, TSCDataFrame):
            time_derivative = X
            for tsid, time_series in X.itertimeseries():
                time_series_dt = dt_func(time_series.to_numpy())
                time_derivative.loc[tsid, :] = time_series_dt
        else:
            time_derivative = dt_func(np.asarray(X))

        return self._same_type_X(
            X=X, values=time_derivative, set_columns=self.features_out_[1]
        )
