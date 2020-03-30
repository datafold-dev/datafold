#!/usr/bin/env python3

from typing import List, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils import _print_elapsed_time  # NOTE: internal from sklearn!
from sklearn.utils.validation import check_is_fitted

from datafold.appfold._edmd import EDMDCV  # do not remove, even if not used here
from datafold.dynfold.base import (
    PRE_FIT_TYPES,
    PRE_IC_TYPES,
    TRANF_TYPES,
    TSCPredictMixIn,
    TSCTransformerMixIn,
)
from datafold.dynfold.dmd import DMDBase, DMDFull
from datafold.pcfold import TSCDataFrame
from datafold.pcfold.timeseries.collection import InitialCondition, TSCException


class EDMD(Pipeline, TSCPredictMixIn):
    """Extended Dynamic Mode Decomposition model to approximate the Koopman operator,
    which defines a dynamical system learnt form data.

    ...

    Parameters
    ----------
    dict_steps
        List of (name, transform) tuples (implementing fit/transform and
        inverse_transform if necessary, see ``include_id_state``) that are
        chained (in the order in which they are in the list). All transform must be
        able to deal with time series collection data.

    dmd_model
        The Dynamic Model Decomposition (DMD) model as the final estimator of the
        pipeline. Approximates the Koopman matrix.

    include_id_state
        Indicates if the identity states are included in the dictionary.

    memory
        see superclass `Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline>`_

    verbose
        see superclass `Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline>`_

    Attributes
    ----------
    named_steps
        see superclass `Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline>`_

    See Also
    --------
    :class:`EDMDCV`

    References
    ----------
    .. todo::
        Add reference to Williams et al.
    """

    def __init__(
        self,
        dict_steps: List[Tuple],
        dmd_model: DMDBase = DMDFull(),
        include_id_state: bool = True,
        memory=None,
        verbose=False,
    ):

        self.dict_steps = dict_steps
        self.dmd_model = dmd_model
        self.include_id_state = include_id_state

        # TODO: if necessary provide option to give user defined metric
        self._setup_default_tsc_metric_and_score()

        all_steps = self.dict_steps + [("dmd", self.dmd_model)]
        super(EDMD, self).__init__(steps=all_steps, memory=memory, verbose=verbose)

    @property
    def _dmd_model(self) -> DMDBase:
        # Improves (internal) code readability when using  attribute
        # '_dmd_model' instead of general '_final_estimator'
        return self._final_estimator

    def _validate_dictionary(self):
        # Check that all are TSCTransformer
        for (_, trans_str, transformer) in self._iter(with_final=False):
            if not isinstance(transformer, TSCTransformerMixIn):
                raise TypeError(
                    "Currently, in the pipeline only supports transformers "
                    "that can handle indexed data structures (pd.DataFrame "
                    "and TSCDataFrame)"
                )

    @property
    def transform(self):
        """Apply dictionary on time series data.

        Parameters
        ----------
        X : TSCDataFrame with (n_samples_per_timeseries, n_features)
           Time series collection to transform in the dictionary. Must fulfill input
           requirements of first step of the pipeline.

        Returns
        -------
        Xt : TSCDataFrame (n_samples, n_transformed_features)
        """
        return self._transform

    def _transform(self, X: TRANF_TYPES) -> TRANF_TYPES:
        """Forward transformation of dictionary."""

        if self.include_id_state:
            # copy required to properly attach X later on
            X_dict = X.copy(deep=True)
        else:
            X_dict = X

        # carry out dictionary transformations:
        for _, name, tsc_transform in self._iter(with_final=False):
            X_dict = tsc_transform.transform(X_dict)

        if self.include_id_state:
            X_dict = self._attach_id_state(X=X, X_dict=X_dict)

        return X_dict

    @property
    def inverse_transform(self):
        """Apply inverse transformations in reverse order of the dictionary.

        All transform steps in the pipeline must support ``inverse_transform``.

        Parameters
        ----------
        X : TSCDataFrame of shape (n_samples per time series, n_transformed_features)
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features. Must fulfill
            input requirements of last step of pipeline's
            ``inverse_transform`` method.

        Returns
        -------
        Xt : TSCDataFrame of shape (n_samples, n_features)
        """
        return self._inverse_transform

    def _inverse_transform(self, X: TRANF_TYPES) -> TRANF_TYPES:
        """Inverse transformation. """

        if self.include_id_state:
            # simply select columns from attached id state:
            X_ts = X.loc[:, self.features_in_[1]]
        else:
            # it is required to inverse_transform, because the initial states are not
            # available:
            X_ts = X
            reverse_iter = reversed(list(self._iter(with_final=False)))
            for _, _, tsc_transform in reverse_iter:
                X_ts = tsc_transform.inverse_transform(X_ts)

        return X_ts

    def _compute_n_samples_ic(self, X, X_dict):
        diff = X.n_timesteps - X_dict.n_timesteps

        if isinstance(diff, pd.Series):
            # time series can have different number of time values (in which case it is
            # a Series), however, the the difference has to be the same for all time
            # series
            assert (diff.iloc[0] == diff).all()
            diff = diff.iloc[0]

        # +1 because the diff indicates how many samples were removed -- we want the
        # number that is required for the initial condition
        return int(diff) + 1

    def _validate_type_and_n_samples_ic(self, X_ic):

        if self.n_samples_ic_ == 1:
            if isinstance(X_ic, TSCDataFrame):
                raise TypeError(
                    "The n_samples to define an inital condition ("
                    "n_samples_ic_={}) is incorrect. Got a time series "
                    "collection with minimum 2 samples per time series."
                )
        else:  # self.n_samples_ic_ > 1
            if not isinstance(X_ic, TSCDataFrame):
                raise TypeError(
                    "For the initial condition a TSCDataFrame is required, "
                    f"with {self.n_samples_ic_} (n_samples_ic_) samples per initial "
                    f"condition. Got type={type(X_ic)}."
                )

            if not (X_ic.n_timesteps > self.n_samples_ic_).all():

                raise TSCException(
                    f"For each initial condition exactly {self.n_samples_ic_} samples "
                    f"(attribute n_samples_ic_) are required. Got: \n {X_ic.n_timesteps}"
                )

    def _attach_id_state(self, X, X_dict):
        # remove states from X (the id-states) that are also removed during dictionary
        # transformations
        X = X.loc[X_dict.index, :]
        try:
            X = pd.concat([X, X_dict], axis=1)
        except AttributeError as e:
            all_columns = X_dict.columns.append(X.columns)
            duplicates = all_columns[all_columns.duplicated()]
            raise ValueError(
                "The ID state could not be attached, because the columns\n"
                f"{duplicates}\n"
                f"are already present in the dictionary."
            )

        return X

    def fit(self, X: PRE_FIT_TYPES, y=None, **fit_params) -> "EDMD":
        """Fit the EDMD model

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the DMD model.

        Parameters
        ----------
        X : TSCDataFrame
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : None
            Parameter without use. Only there to fulfill the general parameter of
            scikit-learn

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        """
        self._validate_data(
            X,
            ensure_feature_name_type=True,
            validate_tsc_kwargs={"ensure_const_delta_time": True},
        )
        self._setup_features_and_time_fit(X)

        # calls internally fit_transform (!!), and stores results into cache if
        # "self.memory is not None" (see docu)
        X_dict, fit_params = self._fit(X, y, **fit_params)

        self.n_samples_ic_ = self._compute_n_samples_ic(X, X_dict)

        if self.include_id_state:
            X_dict = self._attach_id_state(X=X, X_dict=X_dict)

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            self._dmd_model.fit(X=X_dict, y=y, **fit_params)

        return self

    def predict(self, X: PRE_IC_TYPES, time_values=None, **predict_params):
        """Apply dictionary to the initial condition data, and predict with the dmd
        model.

        Parameters
        ----------
        X of shape (n_initial_conditions * n_samples, n_features)
            Initial condition for prediction. The ``n_samples`` is determined by the
            dictionary and available in ``n_samples_ic_`` after calling ``fit``. If
            ``n_samples_ic_ = 1`` a DataFrame is required (per row one initial
            condition), and if``n_samples_ic_ > 1`` a TSCDataFrame is required where
            each time series is an initial condition. Must fulfill input requirements
            of first step of the pipeline.
        time_values
            Time values to evaluate the dynamical system, for each initial condition.
            
        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline.

        Returns
        -------
        X_ts : TSCDataFrame
            Predicted time series collection.
        """

        check_is_fitted(self)
        self._validate_data(
            X,
            ensure_feature_name_type=True,
            validate_tsc_kwargs={"ensure_const_delta_time": True},
        )

        InitialCondition.validate(X)

        X, time_values = self._validate_features_and_time_values(
            X=X, time_values=time_values
        )

        self._validate_type_and_n_samples_ic(X_ic=X)

        X_dict = self.transform(X)

        # this needs to always hold if the checks _validate_type_and_n_samples_ic are
        #  correct
        assert isinstance(X_dict, pd.DataFrame)

        # now we compute the time series in "dictionary space":
        X_latent_ts = self._dmd_model.predict(
            X_dict, time_values=time_values, **predict_params
        )

        # transform from "dictionary space" to "user space"
        X_ts = self.inverse_transform(X_latent_ts)

        return X_ts

    def reconstruct(self, X: TSCDataFrame) -> TSCDataFrame:
        """
        Reconstruct an existing time series collection: For each time series of the
        input collection the initial condition and time values will used to
        ``predict`` with the EDMD model.


        Parameters
        ----------
        X: TSCDataFrame
            Time series collection to reconstruct.

        Returns
        -------
        X_reconstruct : TSCDataFrame
            Reconstructed time series collection. The shape may be different to ``X``,
            if for the initial condition requires more than one sample.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X,
            ensure_feature_name_type=True,
            validate_tsc_kwargs={"ensure_const_delta_time": True},
        )
        self._validate_feature_names(X)

        X_reconstruct: List[TSCDataFrame] = []
        for X_ic, time_values in InitialCondition.iter_reconstruct_ic(
            X, n_samples_ic=self.n_samples_ic_
        ):
            # transform initial condition to dictionary space
            X_dict_ic = self.transform(X_ic)

            # evolve state with dmd model
            X_dict_ts = self._dmd_model.predict(X=X_dict_ic, time_values=time_values)

            # transform back to user space
            X_est_ts = self.inverse_transform(X_dict_ts)

            X_reconstruct.append(X_est_ts)

        X_reconstruct = pd.concat(X_reconstruct, axis=0)
        assert isinstance(X_reconstruct, TSCDataFrame)

        # NOTE: time series contained in X_reconstruct can be shorter in length than
        # the original time series (i.e. no full reconstruction), because some transfom
        # models drop samples (e.g. Takens)
        return X_reconstruct

    def fit_predict(self, X, y=None, **fit_params):
        """Applies fit_predict of last step in pipeline after transforms.

        Applies fit_transforms of a pipeline to the data, followed by the
        fit_predict method of the DMD model in the pipeline. Valid
        only if the DMD model implements fit_predict.

        Parameters
        ----------
        X : TSCDataFrame
            Training data. Must fulfill input requirements of first step of
            the pipeline.

        y : None
            Parameter without use. Only there to fulfill the general parameter of
            scikit-learn

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        X_reconstruct : TSCDataFrame
            Reconstructed time series collection. The shape may be different to ``X``,
            if for the initial condition requires more than one sample.
        """
        return self.fit(X=X, y=y, **fit_params).reconstruct(X=X)

    def fit_transform(self, X: TSCDataFrame, y=None, **fit_params):
        """
        Fit the dictionary and the DMD model and return the transformed data.

        Parameters
        ----------
        X: TSCDataFrame
            Time series collection data to fit the model with and return the transformed
            data.

        y : None
            Parameter without use. Only there to fulfill the general parameter of
            scikit-learn.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        X_dict: TSCDataFrame
            Time series collection after applying all transforms in the pipeline.
        """
        # NOTE: could be improved, but this function is probably not required very often.
        return self.fit(X=X, y=y, **fit_params).transform(X)

    def score(self, X: TSCDataFrame, y=None, sample_weight=None):
        """Apply transforms, and score with the final estimator

        Parameters
        ----------
        X : TSCDataFrame
            Time series collection to reconstruct. Must fulfill input requirements of
            first step of the pipeline.

        y : None
            Parameter without use. Only there to fulfill the general parameter of
            scikit-learn.

        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method set up.

        Returns
        -------
        score : float
        """

        assert y is None
        self._check_attributes_set_up(check_attributes=["_score_eval"])

        # does all the checks:
        X_est_ts = self.reconstruct(X)

        # Important note for getting initial states in dictionary space:
        # during .transform() samples can be discarded (e.g. when applying Takens)
        # This means that in the latent space there can be less samples than in the
        # "physical" space and this is corrected:
        if X.shape[0] > X_est_ts.shape[0]:
            X = X.loc[X_est_ts.index, :]

        return self._score_eval(X, X_est_ts, sample_weight)
