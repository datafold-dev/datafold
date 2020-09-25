"""This file contains code that is copied and modified from

scikit-learn
version 0.22.1.
repository: https://github.com/scikit-learn/scikit-learn/
project homepage: https://scikit-learn.org/stable/

Specifically, this applies to the following files:

*  sklearn.model_selection._validation.py, _fit_and_score
*  sklearn.model_selection._search.py, BaseSearchCV.fit
*  sklearn.sklearn.pipeline.pipeline.py Pipeline

For the datafold module "edmd.py" (this file) the following license from the
scikit-learn project is added in addition to the datafold license (see LICENSE file)

-- scikit-learn license and copyright notice START

New BSD License

Copyright (c) 2007–2019 The scikit-learn developers.
All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the Scikit-learn Developers  nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

-- scikit-learn license and copyright notice END
"""


import numbers
import time
import warnings
from itertools import product
from traceback import format_exception_only
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection import GridSearchCV, check_cv
from sklearn.model_selection._validation import is_classifier
from sklearn.pipeline import Pipeline
from sklearn.utils import _message_with_time, _print_elapsed_time
from sklearn.utils.validation import _check_fit_params, check_is_fitted, indexable

from datafold.dynfold import DMDBase, DMDFull
from datafold.dynfold.base import (
    InitialConditionType,
    TimePredictType,
    TransformType,
    TSCPredictMixin,
    TSCTransformerMixin,
)
from datafold.pcfold import InitialCondition, TSCDataFrame, TSCKfoldSeries, TSCKFoldTime
from datafold.pcfold.timeseries.collection import TSCException
from datafold.pcfold.timeseries.metric import TSCCrossValidationSplits
from datafold.utils.general import (
    df_type_and_indices_from,
    is_integer,
    projection_matrix_from_features,
)


class EDMD(Pipeline, TSCPredictMixin):
    """Extended Dynamic Mode Decomposition (EDMD) model to approximate the Koopman
    operator with a matrix.

    The model is trained with time series collection data to identify a dynamical system.
    The implementation is a special case of :class:`sklearn.pipeline.Pipeline`,
    where the EDMD dictionary contains a flexible number of transform models and the
    final estimator is a :class:`.DMDBase` model, which decomposes the time series data
    into spatio-temporal components.

    * *original* data refers to the original time series measurements (used to `fit`
      the model)
    * *dictionary* data refers to the data after it was transformed by the dictionary
      and before it is processed by the DMD model

    The two options ``include_id_state`` and ``compute_koopman_modes`` set the
    attribute ``koopman_modes`` after calling `fit`. If both options are disabled the
    ``inverse_transform`` is performed via the dictionary transformations functions in
    reverse order (Note, that this can include non-linear inverse mappings).

    ...

    Parameters
    ----------
    dict_steps
        List with `(string_identifier, model)` of models to transform the data. The
        list defines the transformation pipeline and order of execution. All models in
        the list must be able to handle :class:`.TSCDataFrame` as input and output.

    dmd_model
        A DMD variant that represents the final estimator. The DMD approximates either
        the Koopman operator or the generator of it with a matrix. The DMD model performs
        time series predictions in the dictionary space that are then ultimately
        transformed back to the original space.

    include_id_state
        If True, the original time series samples are added to the dictionary (without
        any transformations), after the time series are transformed to dictionary space.
        The mapping from the dictionary space back to the full-state original space
        is then much easier, and accessible through the attribute ``koopman_modes``.
        This comes at the cost of a larger dictionary dimension, and can become
        inveasible for large feature dimensions.

        .. note::
            The final dictionary :py:class:`.TSCDataFrame` must not contain any feature
            names of the original feature names to avoid duplicates.

    compute_koopman_modes
        If True, a matrix that contains the Koopman modes is computed. It linearly maps
        from the dictionary space to the full-state original space. The matrix is
        accessible via the attribute ``koopman_modes``. If ``include_id_state=True``
        at the same time, this parameter has no effect. 

    memory: :class:`Optional[None, str, object]`, :class:`object` with the \
    `joblib.Memory` interface
        Used to cache the fitted transformers of the pipeline. By default, no caching is
        performed. If a string is given, it is the path to the caching directory.
        Enabling caching triggers a clone of the transformers before fitting. Therefore,
        the transformer instance given to the pipeline cannot be inspected directly.
        Use the attribute ``named_steps`` or ``steps`` to inspect estimators within the
        pipeline. Caching the transformers is advantageous when fitting is time consuming.

    verbose: :class:`bool`
        If True, the time elapsed while fitting each step will be printed as it is
        completed.

    Attributes
    ----------

    named_steps: :class:`Dict[str, object]`
        Read-only attribute to access any step parameter by user given name. Keys are
        step names and values are steps parameters.

    koopman_modes: Optional[pandas.DataFrame]
        A matrix of shape `(n_features_original_space, n_features_dict_space)` which
        stores the weights to map from the dictionary states to the original full-state
        (see Eq. 16 in :cite:`williams_datadriven_2015`). The attribute remains
        ``None`` if both ``include_id_state`` and ``compute_koopman_modes`` are False.

    koopman_eigenvalues: pandas.Series
        The eigenvalues of the Koopman matrix or the Koopman generator matrix of
        shape `(n_features_dict,)`. The attribute is not available if the set DMD model
        does only compute the system matrix and not the its spectral components.

    n_samples_ic_: int
        The number of time samples required for an initial condition. If the value is
        larger than 1, then a time series is required with the same sampling interval
        of the time series during fit.

    See Also
    --------

    :py:class:`EDMDCV`

    References
    ----------

    :cite:`williams_datadriven_2015`
    """

    def __init__(
        self,
        dict_steps: List[Tuple[str, object]],
        dmd_model: Optional[DMDBase] = None,
        *,
        include_id_state: bool = True,
        compute_koopman_modes: bool = True,
        memory: Optional[Union[str, object]] = None,
        verbose: bool = False,
    ):

        self.dict_steps = dict_steps
        self.dmd_model = dmd_model if dmd_model is not None else DMDFull()
        self.include_id_state = include_id_state
        self.compute_koopman_modes = compute_koopman_modes

        # TODO: if necessary provide option to give user defined metric
        self._setup_default_tsc_metric_and_score()

        all_steps = self.dict_steps + [("dmd", self.dmd_model)]
        super(EDMD, self).__init__(steps=all_steps, memory=memory, verbose=verbose)

    @property
    def _dmd_model(self) -> DMDBase:
        # Improves (internal) code readability when using  attribute
        # '_dmd_model' instead of general '_final_estimator'
        return self._final_estimator

    @property
    def koopman_modes(self):
        check_is_fitted(self)

        if self._koopman_modes is None:
            return None
        else:
            # return pandas object to properly indicate what modes are corresponding to
            # which feature
            modes = pd.DataFrame(
                self._koopman_modes,
                index=self.feature_names_in_,
                columns=[f"evec{i}" for i in range(self._koopman_modes.shape[1])],
            )
            return modes

    @property
    def koopman_eigenvalues(self):
        check_is_fitted(self)
        if not self._dmd_model.is_spectral_mode():
            raise AttributeError(
                "The DMD model was not configured to provide spectral "
                "components for the Koopman matrix."
            )

        return pd.Series(self._dmd_model.eigenvalues_, name="evals")

    def koopman_eigenfunction(self, X: TransformType) -> TransformType:
        """Evaluate the Koopman eigenfunctions.

        The operation is equivalent to Eq. 18 in :cite:`williams_datadriven_2015`.

        Parameters
        ----------

        X : TSCDataFrame, pandas.DataFrame
            The points of the original space at which to evaluate the Koopman
            eigenfunctions. If `n_samples_ic_ > 1`, then the input must be a
            :py:class:`.TSCDataFrame` where each time series must have at least
            ``n_samples_ic_`` samples, with the same time delta as during fit. The input
            must fulfill the first step in the pipeline.

        Returns
        -------
        Union[TSCDataFrame, pandas.DataFrame]
            The evaluated Koopman eigenfunctions. The number of samples are reduced
            accordingly if :code:`n_samples_ic_ > 1` with fallback to
            ``pandas.DataFrame`` if the it is not not a legal :py:class:`.TSCDataFrame`.
        """
        check_is_fitted(self)
        if not self._dmd_model.is_spectral_mode():
            raise AttributeError(
                "The DMD model was not configured to provide spectral "
                "components for the Koopman matrix."
            )

        X_dict = self.transform(X)

        # transform of X_dict matrix
        #   -> note that in the DMD model, there are column-oriented features
        eval_eigenfunction = self._dmd_model.compute_spectral_system_states(
            X_dict.to_numpy().T
        )

        columns = [f"koop_eigfunc{i}" for i in range(eval_eigenfunction.shape[0])]
        eval_eigenfunction = df_type_and_indices_from(
            indices_from=X_dict, values=X_dict, except_columns=columns
        )

        return eval_eigenfunction

    def _validate_dictionary(self):
        # Check that all are TSCTransformer
        for (_, trans_str, transformer) in self._iter(with_final=False):
            if not isinstance(transformer, TSCTransformerMixin):
                raise TypeError(
                    "Currently, in the pipeline only supports transformers "
                    "that can handle indexed data structures (pd.DataFrame "
                    "and TSCDataFrame)"
                )

    @property
    def feature_names_in_(self):
        # delegate to first step (which will call _check_is_fitted)
        # NOTE: n_features_in_ is also delegated, but already included in the super
        # class Pipeline (implementation by sklearn)
        return self.steps[0][1].feature_names_in_

    def transform(self, X: TransformType) -> TransformType:
        """Perform dictionary transformations on time series data (original space).

        Parameters
        ----------
        X : TSCDataFrame, pandas.DataFrame
           States to transform. Must fulfill the input requirements of first step of
           the pipeline. Some transformers may require a minimum number of time steps,
           in this case a TSCDataFrame is mandatory that fulfills this requirement.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame
            The transformed samples. The number of samples are reduced accordingly if
            `n_samples_ic_ > 1` with fallback to `pandas.DataFrame` if not a legal
            `TSCDataFrame`.
        """
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

    def inverse_transform(self, X: TransformType) -> TransformType:
        """Perform inverse dictionary transformations on time series data (dictionary
        space).

        The actual performed inverse transformation depends on the parameter settings
        ``include_id_state`` and ``compute_koopman_modes``.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame
            Time series or single samples in observable space. Must contain the
            original state feature names or fulfill input requirements of last step of
            pipeline's ``inverse_transform`` method.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame
            Same type as `X` with original space features.
        """

        if self._inverse_map is not None:
            # Note, here the samples are row-wise
            values = X.to_numpy() @ self._inverse_map

            X_ts = df_type_and_indices_from(
                indices_from=X, values=values, except_columns=self.feature_names_in_
            )

        else:
            # inverse_transform the pipeline because an inverse linear map is not
            # available.
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

    def _compute_inverse_map(self, X: TSCDataFrame, X_dict: TSCDataFrame):
        """Compute matrix that linearly maps from dictionary space to original feature
        space.

        This is equivalent to matrix :math:`B`, Eq. 16 in
        :cite:`williams_datadriven_2015`.

        See also `_compute_koopman_modes` for further details.

        Parameters
        ----------
        X_dict
            Dictionary data.

        X
            Original full-state data.

        Returns
        -------
        numpy.ndarray

        """

        if self.include_id_state:
            # trivial case: we just need a projection matrix to select the
            # original full-states from the dictionary functions
            inverse_map = projection_matrix_from_features(
                X_dict.columns, self.feature_names_in_
            )

        elif self.compute_koopman_modes:
            # Compute the matrix in a least squares sense
            # inverse_map = "B" in Williams et al., Eq. 16
            inverse_map = scipy.linalg.lstsq(
                X_dict.to_numpy(), X.loc[X_dict.index, :].to_numpy(), cond=None
            )[0]

        else:
            # use inverse_transform of dictionary
            inverse_map = None

        return inverse_map

    def _compute_koopman_modes(self) -> Optional[np.ndarray]:
        """Compute the Koopman modes based on the user settings.

        The Koopman modes :math:`V` are a computed with

        .. math::
            V = B \cdot \Psi_{DMD}

        where :math:`B` linearly maps the dictionary states from the dictionary space
        to the original full-state space. See :cite:`williams_datadriven_2015` Eq. 20.

        There are three possible ways:

        1. The full-state original states are included in the dictionary. The matric
           :math:`B` is a projection matrix on the corresponding original state
           features.

        2. The matrix :math:`B` was computed in a least squares sense during `fit` (
           option `compute_koopman_modes=True`)
           .. math::
                B = \Psi^{\dagger} \cdot D
            where :math:`\Psi` are the dictionary states and `D` the original states.

        3. The matrix :math:`B` was not computed during fit. The the Koopman modes
           are set to ``None``. Instead the `inverse_transform` of the dictionary pipeline
           must be called to get back to the original space.

        Returns
        -------
        Optional[numpy.ndarray]
            The computed Koopman modes.
        """

        if not hasattr(self, "_inverse_map"):
            raise NotImplementedError("_inverse_map not available. Please report bug. ")

        if self._inverse_map is not None and self._dmd_model.is_spectral_mode():
            koopman_modes = self._inverse_map.T @ self._dmd_model.eigenvectors_right_
        else:
            # Set koopman_modes to None if there are no spectral components of the
            # Koopman operator or its generator.
            koopman_modes = None

        return koopman_modes

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

    def fit(self, X: TimePredictType, y=None, **fit_params) -> "EDMD":
        r"""Fit the EDMD model.

        Internally calls `fit_transform` of all transform methods in the dictionary (in
        same order) and then fits the DMD model (final estimator).

        Parameters
        ----------
        X
            Training time series data. Must fulfill input requirements of first
            `dict_step` in the dictionary pipeline.

        y : None
            ignored

        **fit_params: Dict[str, object]
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``. To add parameters for the  DMD model use
            ``s=dmd``, e.g. ``dmd__param``.
            
        Returns
        -------
        EDMD
            self

        Raises
        ------
        TSCException
            Time series collection restrictions in `X`: (1) time delta must be constant
            (2) all values must be finite (no `NaN` or `inf`)
        """
        self._validate_datafold_data(
            X, ensure_tsc=True, validate_tsc_kwargs={"ensure_const_delta_time": True},
        )
        # NOTE: self._setup_features_and_time_fit(X) is not called here, because the
        # n_features_in_ and n_feature_names_in_ is delegated to the first instance in
        # the pipeline. The time values are set separately here:
        time_values = self._validate_time_values(time_values=X.time_values())
        self.time_values_in_ = time_values
        self.dt_ = X.delta_time

        # '_fit' calls internally fit_transform (!!), and stores results into cache if
        # "self.memory is not None" (see docu):
        fit_params = self._check_fit_params(**fit_params or {})
        X_dict = self._fit(X, y, **fit_params)

        self.n_samples_ic_ = self._compute_n_samples_ic(X, X_dict)

        if self.include_id_state:
            X_dict = self._attach_id_state(X=X, X_dict=X_dict)

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            self._dmd_model.fit(X=X_dict, y=y, **fit_params["dmd"])

        self._inverse_map = self._compute_inverse_map(X=X, X_dict=X_dict)
        self._koopman_modes = self._compute_koopman_modes()

        return self

    def _predict_ic(self, X_dict: TSCDataFrame, time_values, qois) -> TSCDataFrame:
        """Prediction with initial condition.

        Parameters
        ----------
        X_dict
            The initial condition in dictionary space.

        time_values
            The future time values to evaluate the system at.

        qois
            A subselection of quantities of interest (must be part of the dictionary).

        Returns
        -------

        """

        if qois is None:
            feature_columns = self.feature_names_in_
        else:
            feature_columns = qois

        if self.koopman_modes is not None:
            if qois is None:
                modes = self.koopman_modes.to_numpy()
            else:
                project_matrix = projection_matrix_from_features(
                    self.feature_names_in_, qois
                )
                modes = project_matrix.T @ self.koopman_modes.to_numpy()

            # compute the time series in original space directly by adapting the modes
            X_ts = self._dmd_model.predict(
                X_dict,
                time_values=time_values,
                **{"modes": modes, "feature_columns": feature_columns},
            )
        else:
            # predict all dictionary time series
            X_ts = self._dmd_model.predict(X_dict, time_values=time_values)

            # transform from dictionary space by pipeline inverse_transform
            X_ts = self.inverse_transform(X_ts)
            X_ts = X_ts.loc[:, feature_columns]

        return X_ts

    def predict(
        self,
        X: InitialConditionType,
        time_values: Optional[np.ndarray] = None,
        qois: Optional[Union[pd.Index, List[str]]] = None,
        **predict_params,
    ):
        """Time series predictions for one or many initial conditions.

        The internal prediction steps:

        1. Perform dictionary transformations on initial condition input (`X`).
        2. Use the transformed initial conditions and predict the time series in
           dictionary space with the DMD model.
        3. Map the dictionary space time series back to the original space.

        Parameters
        ----------
        X: TSCDataFrame, numpy.ndarray
            Initial conditions states for prediction. The preferred input type is
            :py:class:`TSCDataFrame`. If an initial condition only requires a
            single samples (``n_samples_ic_ = 1``), then an :class:`numpy.ndarray`
            is also sufficient (row-wise ordered). Each initial condition must
            fulfill the input requirements of first step of the pipeline.

        time_values
            Time values to evaluate the model for each initial condition.
            Defaults to time values contained in the data available during ``fit``. The
            values should be ascending and non-negative numeric values.

        qois
            A list of feature names of interest to be include in the returned
            predictions. If ``include_id_state=True``, the time series are only
            computed for the selected features in the dictionary space (via Koopman
            modes), which decreases the memory requirements. Otherwise, the features
            are selected afterwards.

        **predict_params: Dict[str, object]
            Keyword arguments passed to the ``predict`` method of the DMD model.

        Returns
        -------
        TSCDataFrame
            Predicted time series, each evaluated at the specified time values.

        Raises
        ------
        TSCException
            Time series collection requirements in `X`: (1) same (constant) time delta as
            during fit (2) all time series must have identical
            time values (3) all values must be finite (no `NaN` or `inf`)
        """

        check_is_fitted(self)

        if isinstance(X, np.ndarray):
            # work internally only with TSCDataFrame
            X = InitialCondition.from_array(X, columns=self.feature_names_in_)
        else:
            InitialCondition.validate(
                X,
                n_samples_ic=self.n_samples_ic_,
                dt=self.dt_ if self.n_samples_ic_ > 1 else None,
            )

        X, time_values = self._validate_features_and_time_values(
            X=X, time_values=time_values
        )

        self._validate_datafold_data(
            X, ensure_tsc=True,
        )

        X_dict = self.transform(X)
        X_ts = self._predict_ic(X_dict=X_dict, time_values=time_values, qois=qois)

        return X_ts

    def reconstruct(
        self, X: TSCDataFrame, qois: Optional[Union[pd.Index, List[str]]] = None,
    ) -> TSCDataFrame:
        """Reconstruct existing time series collection.

        Internal steps to reconstruct a time series collection:

        1. Extract the initial states from each time series in the collection (
           this can also be multiple time samples, see attribute ``n_samples_ic_``).
        2. Predict the remaining states of each time series with the EDMD model at the
           same time values.

        Parameters
        ----------
        X
            The time series collection to reconstruct.

        qois
            A list of feature names of interest to be include in the returned
            predictions. Passed to :py:meth:`.predict`.

        Returns
        -------
        TSCDataFrame
            Reconstructed time series collection. If `n_samples_ic_ > 1` the number
            of samples for each time series decrease accordingly.

        Raises
        ------
        TSCException
            Time series collection requirements in `X`: (1) time delta must be constant
            (2) all values must be finite (no `NaN` or `inf`)
        """
        check_is_fitted(self)

        X = self._validate_datafold_data(
            X,
            ensure_tsc=True,
            # Note: no const_delta_time required here. The required const samples for
            # time series initial conditions is included in the predict method.
        )
        self._validate_feature_names(X)

        X_reconstruct: List[TSCDataFrame] = []
        for X_ic, time_values in InitialCondition.iter_reconstruct_ic(
            X, n_samples_ic=self.n_samples_ic_
        ):
            # transform initial condition to dictionary space
            X_dict_ic = self.transform(X_ic)

            X_est_ts = self._predict_ic(
                X_dict=X_dict_ic, time_values=time_values, qois=qois
            )

            X_reconstruct.append(X_est_ts)

        X_reconstruct = pd.concat(X_reconstruct, axis=0)
        assert isinstance(X_reconstruct, TSCDataFrame)

        # NOTE: time series contained in X_reconstruct can be shorter in length than
        # the original time series (i.e. no full reconstruction), because some transfom
        # models drop samples (e.g. Takens)
        return X_reconstruct

    def fit_predict(
        self,
        X: TSCDataFrame,
        y=None,
        qois: Optional[Union[pd.Index, List[str]]] = None,
        **fit_params,
    ):
        """Fit the model and reconstruct the training data.

        Parameters
        ----------
        X
            Training time series data. Must fulfill input requirements of first step of
            the pipeline.

        y: None
            ignored

        qois
            A list of feature names of interest to be include in the returned
            predictions. Passed to :py:meth:`.predict`.

        **fit_params: Dict[str, object]
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        TSCDataFrame
            Reconstructed time series collection. If `n_samples_ic_ > 1` the number
            of samples for each time series decreases accordingly.

        Raises
        ------
        TSCException
            Time series collection restrictions in **X**: (1) time delta must be constant
            (2) all values must be finite (no `NaN` or `inf`)
        """
        return self.fit(X=X, y=y, **fit_params).reconstruct(X=X, qois=qois)

    def fit_transform(self, X: TSCDataFrame, y=None, **fit_params):
        """Fit the dictionary and the DMD model and return the transformed time series.

        Parameters
        ----------
        X
            Time series collection data to fit the model.

        y: None
            ignored

        **fit_params: Dict[str, object]
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        TSCDataFrame
             Transformed time series data in dictionary space.

        Raises
        ------
        TSCException
            Time series collection restrictions in **X**: (1) time delta must be constant
            (2) all values must be finite (no `NaN` or `inf`)
        """
        # NOTE: could be improved, but this function is probably not required very often.
        return self.fit(X=X, y=y, **fit_params).transform(X)

    def score(
        self, X: TSCDataFrame, y=None, sample_weight: Optional[np.ndarray] = None
    ):
        """Reconstruct the time series collection and score the recunstructed time
        series with the original time series.

        Parameters
        ----------
        X
            Time series collection to reconstruct and score. Must fulfill input
            requirements of all dictionary models.

        y: None
            ignored

        sample_weight
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the internal ``score`` method.

        Returns
        -------
        float
            score

        Raises
        ------
        TSCException
            Time series collection restrictions in `X`: (1) time delta must be constant
            (2) all values must be finite (no `NaN` or `inf`)
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


def _split_X_edmd(X: TSCDataFrame, y, train_indices, test_indices):
    X_train, X_test = X.tsc.assign_ids_train_test(
        train_indices=train_indices, test_indices=test_indices
    )

    if not isinstance(X_train, TSCDataFrame) or not isinstance(X_test, TSCDataFrame):
        raise RuntimeError(
            "X_train or X_test is not a TSCDataFrame anymore. "
            "Potential reason is too small folds."
        )

    return X_train, X_test


def _fit_and_score_edmd(
    edmd: EDMD,
    X: TSCDataFrame,
    y,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    error_score=np.nan,
):
    if verbose > 1:
        if parameters is None:
            msg = ""
        else:
            msg = "%s" % (", ".join("%s=%s" % (k, v) for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * "."))
    else:
        msg = ""

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    train_scores: Dict[str, numbers.Number] = {}
    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        edmd = edmd.set_params(**cloned_parameters)

    start_time = time.time()

    X_train, X_test = _split_X_edmd(X, y, train_indices=train, test_indices=test)

    try:
        edmd = edmd.fit(X=X_train, y=y, **fit_params)
    except Exception as e:
        # Handle all exception, to not waste other working or complete results
        fit_time = time.time() - start_time  # Note fit time as time until error
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = {"score": error_score}
                if return_train_score:
                    train_scores = {"score": error_score}
            warnings.warn(
                "Estimator fit failed. The score on this train-test"
                f" partition for these parameters will be set to {error_score}. "
                f"Details: \n{format_exception_only(type(e), e)[0]}",
                FitFailedWarning,
            )
        else:
            raise ValueError(
                "error_score must be the string 'raise' or a"
                " numeric value. (Hint: if using 'raise', please"
                " make sure that it has been spelled correctly.)"
            )

    else:
        fit_time = time.time() - start_time
        test_scores = {"score": edmd.score(X_test, y=None)}

        score_time = time.time() - start_time - fit_time

        if return_train_score:
            train_scores = {"score": edmd.score(X_train, y=None)}

    if verbose > 2:
        if isinstance(test_scores, dict):
            for scorer_name in sorted(test_scores):
                msg += f", {scorer_name}="
                if return_train_score:
                    msg += f"(train={train_scores[scorer_name]:.3f},"
                    msg += f" test={test_scores[scorer_name]:.3f})"
                else:
                    msg += f"{test_scores[scorer_name]:.3f}"
        else:
            msg += ", score="
            msg += (
                f"{test_scores:.3f}"
                if not return_train_score
                else f"(train={train_scores:.3f}, test={test_scores:.3f})"
            )

    if verbose > 1:
        total_time = score_time + fit_time
        print(_message_with_time("CV", msg, int(total_time)))

    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_n_test_samples:
        ret.append(X_test.shape[0])
    if return_times:
        assert isinstance(score_time, numbers.Number)
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    if return_estimator:
        ret.append(edmd)
    return ret


class EDMDCV(TSCPredictMixin, GridSearchCV):
    """Exhaustive parameter search over specified grid for a :class:`EDMD` model with
    cross-validation.

    ...

    Parameters
    ----------
    estimator
        Model to be optimized. Either use the default ``score``
        function, or ``scoring`` must be passed.

    param_grid
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : string, callable, list/tuple, dict or None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        Note: that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's score method is used.

        .. warning::
            The multi-metric optimization is experimental. Please use with care.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    cv : cross-validation generator
        Determines the cross-validation splitting strategy. Possible inputs for cv are:

        - :class:`.TSCKfoldSeries` splits `k` folds across time series (useful when
            many time series are in a collection)
        - :class:`.TSCKFoldTime` splits `k` folds across time

    refit : bool, string, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a string denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``. In that
        case, the ``best_estimator_`` and ``best_parameters_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    verbose : :class:`int`
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        `FitFailedWarning` is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is ``np.nan``.

    return_train_score : :class:`bool`
        If ``False``, the ``cv_results_`` attribute will not include training
        scores. Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    Attributes
    ----------

    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a ``pandas.DataFrame``.
        For instance the below given table

        +------------+-----------+------------+-----------------+---+---------+
        |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
        +============+===========+============+=================+===+=========+
        |  'poly'    |     --    |      2     |       0.80      |...|    2    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'poly'    |     --    |      3     |       0.70      |...|    4    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.1   |     --     |       0.80      |...|    3    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.2   |     --     |       0.93      |...|    1    |
        +------------+-----------+------------+-----------------+---+---------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
            'split1_test_score'  : [0.82, 0.50, 0.70, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
            'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.80, 0.92, 0.70, 0.93],
            'split1_train_score' : [0.82, 0.55, 0.70, 0.87],
            'mean_train_score'   : [0.81, 0.74, 0.70, 0.90],
            'std_train_score'    : [0.01, 0.19, 0.00, 0.03],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00, 0.01],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }

        .. note::

            The key ``'params'`` is used to store a list of parameter
            settings dicts for all the parameter candidates.
            The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
            ``std_score_time`` are all in seconds. For multi-metric evaluation,
            the scores for all the scorers are available in the ``cv_results_``
            dict at the keys ending with that scorer's name (``'_<scorer_name>'``)
            instead of ``'_score'`` shown above. ('split0_test_precision',
            'mean_train_precision' etc.)

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.
        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator
        For multi-metric evaluation, this is present only if ``refit`` is
        specified.
        This attribute is not available if ``refit`` is a function.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.
        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).
        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.
        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.
        This is present only if ``refit`` is True.

    Notes
    -----

    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.
    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    """

    def __init__(
        self,
        estimator: EDMD,
        *,
        param_grid: Union[Dict, List[Dict]],
        cv: TSCCrossValidationSplits,
        **kwargs,
    ):

        super(EDMDCV, self).__init__(
            estimator=estimator, param_grid=param_grid, cv=cv, **kwargs
        )

    def _validate_settings_edmd(self):
        # leave import here to avoid circular imports

        if not isinstance(self.estimator, EDMD):
            raise TypeError("EDMDCV only supports EDMD estimators.")

        if not isinstance(self.cv, TSCCrossValidationSplits):
            raise TypeError(f"cv must be of type {(TSCKfoldSeries, TSCKFoldTime)}")

    def _check_multiscore(self):
        scorers, self.multimetric_ = _check_multimetric_scoring(
            self.estimator, scoring=self.scoring
        )

        if self.multimetric_:
            if (
                self.refit is not False
                and (
                    not isinstance(self.refit, str)
                    or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers
                )
                and not callable(self.refit)
            ):
                raise ValueError(
                    "For multi-metric scoring, the parameter "
                    "refit must be set to a scorer key or a "
                    "callable to refit an estimator with the "
                    "best parameter setting on the whole "
                    "data and make the best_* attributes "
                    "available for that metric. If this is "
                    "not needed, refit should be set to "
                    "False explicitly. %r was passed." % self.refit
                )
            else:
                refit_metric = self.refit
        else:
            refit_metric = "score"

        return scorers, refit_metric

    def fit(self, X: TSCDataFrame, y=None, **fit_params):
        """Run fit with all sets of parameter.

        Parameters
        ----------

        X
            Training time series data.

        y: None
            ignored

        **fit_params : Dict[str, object]
            Parameters passed to the ``fit`` method of the estimator.
        """
        self._validate_settings_edmd()
        X = self._validate_datafold_data(X)

        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))

        scorers, refit_metric = self._check_multiscore()

        X, y = indexable(X, y)
        fit_params = _check_fit_params(X, fit_params)

        n_splits = cv.get_n_splits(X, y, groups=None)

        base_estimator = clone(self.estimator)

        parallel = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch
        )

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )

        results: Dict[str, Any] = {}
        with parallel:
            all_candidate_params: List[List[Dict[str, Any]]] = []
            all_out: List[Any] = []

            def evaluate_candidates(candidate_params):
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        f"Fitting {n_splits} folds for each of {n_candidates} candidates,"
                        f" totalling {n_candidates * n_splits} fits"
                    )

                out = parallel(
                    delayed(_fit_and_score_edmd)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        **fit_and_score_kwargs,
                    )
                    for parameters, (train, test) in product(
                        candidate_params, cv.split(X, y, groups=None)
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        f"inconsistent results. Expected {n_splits} "
                        f"splits, got {len(out) // n_candidates}"
                    )

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, scorers, n_splits, all_out
                )
                return results

            self._run_search(evaluate_candidates)

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)

                if not is_integer(self.best_index_):
                    raise TypeError("best_index_ returned is not an integer")
                if self.best_index_ < 0 or self.best_index_ >= len(results["params"]):
                    raise IndexError("best_index_ index out of range")
            else:
                self.best_index_ = results[f"rank_test_{refit_metric}"].argmin()
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers["score"]

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    def __getattr__(self, name, *args, **kwargs):

        check_is_fitted(self)

        if hasattr(self.best_estimator_, name):
            return getattr(self.best_estimator_, name)(*args, **kwargs)
        else:
            raise AttributeError(f"attribute {name} now known")

    def reconstruct(
        self, X: TSCDataFrame, qois: Optional[Union[pd.Index, List[str]]] = None
    ):
        """Delegated to :py:meth:`.EDMD.reconstruct` of ``best_estimator_``.
        """
        return self.__getattr__(name="reconstruct", X=X)

    def predict(
        self,
        X: InitialConditionType,
        time_values: Optional[np.ndarray] = None,
        qois: Optional[Union[np.ndarray, pd.Index, List[str]]] = None,
        **predict_params,
    ):
        """Delegated to :py:meth:`.EDMD.predict` of ``best_estimator_``.
        """
        return self.__getattr__(
            name="predict", X=X, time_values=time_values, qois=qois, **predict_params
        )
