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
from datafold.pcfold import MultiquadricKernel, PCManifold, TSCDataFrame
from datafold.pcfold.kernels import PCManifoldKernel
from datafold.pcfold.timeseries.collection import TSCException

try:
    from findiff import FinDiff
except ImportError:
    IMPORTED_FINDIFF = False
else:
    IMPORTED_FINDIFF = True


class TSCFeaturePreprocess(BaseEstimator, TSCTransformerMixIn):
    """Wrapper of a scikit-learn preprocess algorithms to allow time series
    collections as input and output.

    Often scikit-learn performs "pandas.DataFrame in -> numpy.ndarray out". This wrapper
    makes sure to have "pandas.DataFrame in -> pandas.DataFrame out".

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
        """Calls fit of internal transform ``sklearn`` object.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Training data of shape `(n_samples, n_features)`.

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
        """Calls transform of internal transform ``sklearn`` object.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data to transform of shape `(n_samples, n_features)`.

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
        return self._same_type_X(
            X=X, values=values, feature_names=self.features_out_[1]
        )

    def fit_transform(self, X: TransformType, y=None, **fit_params):
        """Calls fit_transform of internal transform ``sklearn`` object..

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Training data to transform of shape `(n_samples, n_features)`.

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

        return self._same_type_X(
            X=X, values=values, feature_names=self.features_out_[1]
        )

    def inverse_transform(self, X: TransformType):
        """Calls `inverse_transform` of internal transform ``sklearn`` object.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data to map back of shape `(n_samples, n_features)`.

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
        return self._same_type_X(X=X, values=values, feature_names=self.features_in_[1])


class TSCIdentity(BaseEstimator, TSCTransformerMixIn):
    """Dummy transformer for testing or as a "passthrough" placeholder.

    Parameters
    ----------
    include_const
        If True, a constant (all ones) column is attached to the data.

    Attributes
    ----------
    is_fit_ : bool
        True if fit has been called.
    """

    def __init__(self, include_const: bool = False):
        self.include_const = include_const

    def fit(self, X: TransformType, y=None, **fit_params):
        """Passthrough data and set internals for validation.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data of shape `(n_samples, n_features)`.

        y: None
            ignored

        Returns
        -------
        TSCIdentity
            self
        """
        X = self._validate_data(X)

        if self.include_const and self._has_feature_names(X):
            features_out = np.append(X.columns, ["const"])
        else:
            features_out = "like_features_in"

        self._setup_features_fit(X, features_out=features_out)

        # Dummy attribute to indicate that fit was called
        self.is_fit_ = True

        return self

    def transform(self, X: TransformType) -> TransformType:
        """Passthrough data and validate feature.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data of shape `(n_samples, n_features)` to passthrough.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type and shape as `X`
        """
        check_is_fitted(self, "is_fit_")

        X = self._validate_data(X)
        self._validate_feature_input(X, direction="transform")

        if self.include_const:
            if self._has_feature_names(X):
                X["const"] = 1
            else:
                X = np.column_stack([X, np.ones(X.shape[0])])

        return X

    def inverse_transform(self, X: TransformType):
        """Passthrough data and validate features shape.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data to passthrough of shape `(n_samples, n_features)`.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type and shape as `X`
        """
        check_is_fitted(self, "is_fit_")
        X = self._validate_data(X)
        self._validate_feature_input(X, direction="inverse_transform")

        if self.include_const:
            if self._has_feature_names(X):
                X = X.drop("const", axis=1)
            else:
                X = X[:, :-1]

        return X


class TSCPrincipalComponent(PCA, TSCTransformerMixIn):
    """Compute principal components from data.

    This is a subclass of scikit-learn's ``PCA``  to generalize the
    input and output of :class:`pandas.DataFrames` and :class:`.TSCDataFrame`. All input
    parameters remain the same. For documentation please visit:

    * `PCA docu <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_
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
        """Compute the principal components from training data.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Training data of shape `(n_samples, n_features)`.

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
            Out-of-sample points of shape `(n_samples, n_features)` to perform dimension
            reduction on.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` of shape `(n_samples, n_components_)`
        """

        check_is_fitted(self)
        X = self._validate_data(X)

        self._validate_feature_input(X, direction="transform")
        pca_data = super(TSCPrincipalComponent, self).transform(self._X_to_numpy(X))
        return self._same_type_X(
            X, values=pca_data, feature_names=self.features_out_[1]
        )

    def fit_transform(self, X: TransformType, y=None, **fit_params) -> TransformType:
        """Compute principal components from data and reduce dimension on same data.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Training data of shape `(n_samples, n_features)`.
            
        y: None
            ignored

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` of shape `(n_samples, n_components_)`
        """

        X = self._validate_data(X)
        self._setup_features_fit(
            X, features_out=[f"pca{i}" for i in range(self.n_components)]
        )

        pca_values = super(TSCPrincipalComponent, self).fit_transform(
            self._X_to_numpy(X), y=y
        )
        return self._same_type_X(
            X, values=pca_values, feature_names=self.features_out_[1]
        )

    def inverse_transform(self, X: TransformType):
        """Map data from the reduced space back to the original space.

        Parameters
        ----------
        X:
            Out-of-sample points of shape `(n_samples, n_components_)` to map back to
            original space.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` of shape `(n_samples, n_features)`
        """

        self._validate_feature_input(X, direction="inverse_transform")

        X_intern = self._X_to_numpy(X)
        data_orig_space = super(TSCPrincipalComponent, self).inverse_transform(X_intern)

        return self._same_type_X(
            X, values=data_orig_space, feature_names=self.features_in_[1]
        )


class TSCTakensEmbedding(BaseEstimator, TSCTransformerMixIn):
    """Perform Takens time delay embedding on time series collection data.

    Parameters
    ----------

    lag
        Number of time steps to lag before embedding starts.

    delays
        Number for time delays to embed.

    frequency
        Time step frequency to emebd (e.g. to embed every sample or only every second
        or third).

    kappa
        Weight of exponential factor in delayed coordinates
        :math:`e^{-d \cdot \kappa}(x_{-d})` with :math:`d = 0, \ldots delays` being the
        delay index. Adapted from :cite:`berry_time-scale_2013`, Eq. 2.1).

    Attributes
    ----------

    delay_indices_ : numpy.ndarray
        Delay indices (backwards in time) assuming a fixed time delta in the time series.

    min_timesteps_: int
        Minimum required time steps for each time series to have a single embedding
        vector.

    delta_time_fit_
        Time delta measured during model fit. This is primarily used to check that
        `transform` or `inverse_transform` data still have the same time delta for
        consistency.

    References
    ----------

    * Original paper from Takens :cite:`rand_detecting_1981`
    * Takens delay embedding in the context of time series data
      :cite:`champion_discovery_2019` (the time delay embedding is then a transform
      function of :py:class:`.EDMD` dictionary).
    """

    def __init__(
        self, lag: int = 0, delays: int = 10, frequency: int = 1, kappa: float = 0
    ):
        self.lag = lag
        self.delays = delays
        self.frequency = frequency
        self.kappa = kappa

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

        check_scalar(
            self.kappa,
            name="kappa",
            target_type=(np.integer, int, np.floating, float),
            min_val=0.0,
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
                # rename columns: [column_name]:d[delay_index]
                _cur_delay_columns = [f"{col}:d{delay_idx}" for col in cols.astype(str)]
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
        """Compute delay indices based on settings and validate input with setting.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame
            Time series collection to validate for time delay embedding.

        y: None
            ignored

        Returns
        -------
        TSCTakensEmbedding
            self

        Raises
        ------
        TSCException
            Time series collection requirements in `X`: (1) time delta must be constant
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
            Time series collection.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame
            Each time series is shortend by the number of samples required for the
            delays. The type can fall back to `pandas.DataFrame` if the result is not
            not a valid :class:`.TSCDataFrame` anymore (this is a typical scenario for
            time series initial conditions).

        Raises
        ------
        TSCException
            Time series collection requirements in `X`: (1) time delta must be constant
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
        ### Implementation using pandas by using shift()
        ### This implementation is better readable, and is for many cases similarly
        # fast to the numpy version (below), but has a performance drop for
        # high-dimensions (dim>500)
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

        if self.kappa > 0:
            # only the delayed coordinates are multiplied with the exp factor
            kappa_vec = np.exp(-self.kappa * np.arange(1, self.delays + 1))
        else:
            kappa_vec = None

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

            if self.kappa > 0:
                delayed_data = delayed_data.astype(np.float)
                delayed_data *= kappa_vec

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
        """Remove time delayed feature columns of time delay embedded time series
        collection.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame
            Time delayed data of shape `(n_samples, n_features_embedded)`

        Returns
        -------
        TSCDataFrame, pandas.DataFrame
            same type as `X` of shape `(n_samples, n_features_original)`
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
    """Represent data in coefficients of radial basis functions.

    Parameters
    ----------

    kernel
        Radial basis kernel to compute the coefficients with.

    center_type
        Selection of what to take as centers during fit. 

        * `all_data` - all data points during fit are used as centers
        * `initial_condition` - take the initial condition states as centers.
           Note for this option the data `X` during fit must be of
           type :class:`.TSCDataFrame`.

    exact_distance
        An inexact distance computation increases the performance at the cost of
        numerical inaccuracies (~1e-7 for Euclidean distance, and ~1 e-14 for squared
        Eucledian distance).
    
    Attributes
    ----------

    centers_: numpy.ndarray
        The center points of the radial basis functions.

    inv_coeff_matrix_: numpy.ndarray
        Matrix to map radial basis coefficients to original space. Computation is
        delayed until `inverse_transform` is called for the first time.
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
            Data of shape (n_centers, n_features) to extract point centers from. Must be
            of type :class:`TSCDataFrame` if center type is `initial_condition`.

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

        if self.center_type == "all_data":
            self.centers_ = self._X_to_numpy(X)
        else:  # self.center_type == "initial_condition":
            if not isinstance(X, TSCDataFrame):
                raise TypeError("Data 'X' must be TSCDataFrame")
            self.centers_ = X.initial_states().to_numpy()

        self.centers_ = PCManifold(
            self.centers_,
            kernel=self.kernel,
            dist_kwargs=dict(backend="brute", exact_numeric=self.exact_distance),
        )

        n_centers = self.centers_.shape[0]
        self._setup_features_fit(X, [f"rbf{i}" for i in range(n_centers)])

        return self

    def transform(self, X: TransformType) -> TransformType:
        """Transform data to radial basis functions coefficients.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data of shape `(n_samples, n_features)`.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` of shape `(n_samples, n_centers)`
        """
        check_is_fitted(self, attributes=["centers_"])
        X = self._validate_data(X)
        self._validate_feature_input(X, direction="transform")

        X_intern = self._X_to_numpy(X)
        rbf_coeff = self.centers_.compute_kernel_matrix(Y=X_intern)

        return self._same_type_X(
            X, values=rbf_coeff, feature_names=self.features_out_[1]
        )

    def fit_transform(self, X, y=None, **fit_params):
        """Set the data as centers and transform to radial basis coefficients.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Radial basis center points and data to transform of shape \
            `(n_samples, n_features)`

        y: None
            ignored

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` of shape `(n_samples, n_centers)`
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
            X=X, values=rbf_coeff, feature_names=self.features_out_[1]
        )

    def inverse_transform(self, X: TransformType):
        """Transform radial basis coefficients back to the original function values.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Coefficient representation of the radial basis functions of shape \
            `(n_samples, n_center)`.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` of shape `(n_samples, n_features)`
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
        return self._same_type_X(
            X, values=X_inverse, feature_names=self.features_in_[1]
        )


class TSCPolynomialFeatures(PolynomialFeatures, TSCTransformerMixIn):
    """Compute polynomial features from data.

    This is a subclass of ``PolynomialFeatures`` from scikit-learn to generalize the
    input and output of :class:`pandas.DataFrames` and :class:`.TSCDataFrame`.

    This class adds the parameter `include_first_order` to choose whether to include the
    identity states. For all other parameters please visit the super class
    documentation of
    `PolynomialFeatures <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html>`_.
    """

    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        include_first_order=False,
    ):
        self.include_first_order = include_first_order

        super(TSCPolynomialFeatures, self).__init__(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
            order="C",
        )

    @property
    def powers_(self):
        powers = super(TSCPolynomialFeatures, self).powers_
        if self.include_first_order:
            return powers
        else:
            return powers[powers.sum(axis=1) != 1, :]

    def _get_poly_feature_names(self, X, input_features=None):
        # Note: get_feature_names function is already provided by super class
        if self._has_feature_names(X):
            feature_names = self.get_feature_names(input_features=X.columns)
        else:
            feature_names = self.get_feature_names()
        return feature_names

    def _non_id_state_mask(self):
        powers = super(TSCPolynomialFeatures, self).powers_
        return powers.sum(axis=1) != 1

    def fit(self, X: TransformType, y=None, **fit_params) -> "TSCPolynomialFeatures":
        """Compute number of output features.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data of shape `(n_samples, n_features)`.

        Returns
        -------
        TSCPolynomialFeatures
            self
        """
        X = self._validate_data(X)

        super(TSCPolynomialFeatures, self).fit(X, y=y)

        self._setup_features_fit(
            X, features_out=self._get_poly_feature_names(X),
        )

        return self

    def transform(self, X: TransformType) -> TransformType:
        """Transform data to polynomial features.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            The data of shape `(n_samples, n_features)` to transform.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Transformed data of shape `(n_samples, n_polynomials)` and with same type
            as `X`.
        """
        check_is_fitted(self)
        X = self._validate_data(X)
        self._validate_feature_input(X, direction="transform")

        poly_data = super(TSCPolynomialFeatures, self).transform(X)

        if not self.include_first_order:
            poly_data = poly_data[:, self._non_id_state_mask()]

        poly_data = self._same_type_X(
            X, values=poly_data, feature_names=self._get_poly_feature_names(X)
        )

        return poly_data


class TSCApplyLambdas(BaseEstimator, TSCTransformerMixIn):
    """Transform data in an element-by-element fashion with lambda functions.

    Each function is called on every column in the data (i.e. the number of samples
    remains the same).

    Two examples using a Python lambda expression and a NumPy's
    `ufunc <https://numpy.org/devdocs/reference/ufuncs.html>`_:

    .. code-block:: python

        TSCApplyLambdas(lambdas=[lambda x: x**3])
        TSCApplyLambdas(lambdas=[np.sin, np.cos])

    Parameters
    ----------
    lambdas
        List of `lambda` or `ufunc` functions (`ufunc` should not be reducing the data).
        Each column `X_col` is passed to the function and the returned `X_transformed`
        data must be of the same shape as `X_col`, i.e.
        :code:`X_transformed = func(X_col)`
    """

    def __init__(self, lambdas):
        self.lambdas = lambdas

    def _not_implemented_numpy_arrays(self, X):
        if isinstance(X, np.ndarray):
            raise NotImplementedError(
                "Currently not implemented for numpy.ndarray. If this is required please "
                "open an issue on Gitlab."
            )

    def fit(self, X: TransformType, y=None, **fit_params) -> "TSCApplyLambdas":
        """Set internal feature information.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Training data.
        y: None
            ignored

        Returns
        -------
        TSCApplyLambdas
            self
        """
        self._not_implemented_numpy_arrays(X)
        X = self._validate_data(X, ensure_feature_name_type=True)

        features_out = [
            f"{feature_name}_lambda{i}"
            for feature_name in X.columns
            for i in range(len(self.lambdas))
        ]

        self._setup_features_fit(X, features_out=features_out)
        return self

    def transform(self, X: TransformType) -> TransformType:
        """Transform data with specified lambda functions.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data of shape `(n_samples, n_features)` to transform.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            The transformed data of same type as `X` and of shape
            `(n_samples, n_lambdas * n_features)`
        """
        self._not_implemented_numpy_arrays(X)

        check_is_fitted(self)
        X = self._validate_data(X, ensure_feature_name_type=True)
        self._validate_feature_input(X, direction="transform")

        lambdas_applied = list()

        for i, _lambda in enumerate(self.lambdas):

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


class TSCFiniteDifference(BaseEstimator, TSCTransformerMixIn):
    """Compute time derivative with finite difference scheme.

    .. note::
        The class internally uses the Python package findiff, which currently is
        optional in *datafold*. The class raises an `ImportError` if findiff is not
        installed.

    Parameters
    ----------

    spacing: Union[str, float]
        The time difference between samples. If "dt" (str) then the time sampling
        frequency of a :meth:`.TSCDataFrame.delta_time` is used during fit.

    diff_order
        The derivative order.

    accuracy
        The convergence order of the finite difference scheme.

    Attributes
    ----------

    spacing_
        The resolved time difference between samples. Equals the parameter
        input if it was of type :class`float`.

    See Also
    --------
    `findiff documentation <https://findiff.readthedocs.io/en/latest/>`_
    """

    def __init__(
        self, spacing: Union[str, float] = "dt", diff_order: int = 1, accuracy: int = 2
    ):

        if not IMPORTED_FINDIFF:
            # TODO: Currently, findiff is an optional package and listed in the
            #  dev-requirements. If the class is more used internally of datafold,
            #  make it a proper dependency.
            raise ImportError(
                "TSCFiniteDifference requires the Python package "
                "'findiff' installed."
            )

        self.spacing = spacing
        self.diff_order = diff_order
        self.accuracy = accuracy

    def fit(self, X: TransformType, y=None, **fit_params) -> "TSCFiniteDifference":
        """Set and validate time spacing between samples.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data of shape `(n_samples, n_features)`.

        y: None
            ignored

        Returns
        -------
        TSCFiniteDifference
            self

        Raises
        ------
        TSCException
            If time series data has not a constant time delta or the input `X` has not
            the same value as specified in `spacing` during initialization.

        """
        X = self._validate_data(
            X,
            ensure_feature_name_type=False,
            validate_tsc_kwargs=dict(
                ensure_delta_time=self.spacing
                if isinstance(self.spacing, float)
                else None
            ),
        )

        if self._has_feature_names(X):
            features_out = [f"{col}_dot" for col in X.columns]
        else:
            features_out = [f"dot{i}" for i in np.arange(X.shape[1])]

        self._setup_features_fit(X=X, features_out=features_out)

        if self.spacing == "dt":

            if not isinstance(X, TSCDataFrame):
                raise TypeError(
                    "For input 'spacing=dt' a time series collections is required."
                )

            self.spacing_ = X.delta_time

            if isinstance(self.spacing_, pd.Series) or np.isnan(self.spacing_):
                raise TSCException.not_const_delta_time(actual_delta_time=self.spacing_)
        else:
            self.spacing_ = self.spacing

            if (
                isinstance(X, TSCDataFrame)
                and np.asarray(self.spacing_ != X.delta_time).all()
            ):
                raise ValueError(
                    f"A spacing of {self.spacing} was specified, but the time series "
                    f"collection has a time delta of {X.delta_time}"
                )

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
        """Compute the finite difference values.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data of shape `(n_samples, n_features)`.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Transformed data of same shape and type as `X`.

        Raises
        ------
        TSCException
            If input `X` has a different time delta than data during `fit`.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X, validate_tsc_kwargs=dict(ensure_delta_time=self.spacing_)
        )
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
            X=X, values=time_derivative, feature_names=self.features_out_[1]
        )
