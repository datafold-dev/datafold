"""This file contains code that is copied and modified from

scikit-learn
version 0.24.1.
repository: https://github.com/scikit-learn/scikit-learn/
project homepage: https://scikit-learn.org/stable/

Specifically, this applies to the following files and functions:

*  sklearn.model_selection._validation.py, _fit_and_score
*  sklearn.model_selection._search.py, BaseSearchCV.fit
*  sklearn.sklearn.pipeline.pipeline.py Pipeline

For the datafold module "edmd.py" (this file) the following license from the
scikit-learn project is in addition to the datafold license (see LICENSE file).

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
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import product
from traceback import format_exc
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse
from joblib import Parallel, delayed, logger
from sklearn.base import clone
from sklearn.exceptions import FitFailedWarning, NotFittedError
from sklearn.model_selection import GridSearchCV, check_cv
from sklearn.model_selection._validation import _num_samples, _score, is_classifier
from sklearn.pipeline import Pipeline
from sklearn.utils import _print_elapsed_time, check_scalar
from sklearn.utils.validation import _check_fit_params, check_is_fitted, indexable

from datafold._decorators import warn_experimental_class
from datafold.dynfold import DMDBase, DMDFull
from datafold.dynfold.base import (
    InitialConditionType,
    TimePredictType,
    TransformType,
    TSCPredictMixin,
    TSCTransformerMixin,
)
from datafold.pcfold import InitialCondition, TSCDataFrame, TSCKfoldSeries, TSCKFoldTime
from datafold.pcfold.timeseries.metric import TSCCrossValidationSplit
from datafold.utils.general import (
    df_type_and_indices_from,
    if1dim_rowvec,
    projection_matrix_from_feature_names,
)


class EDMD(
    Pipeline,
    TSCTransformerMixin,
    TSCPredictMixin,
):
    r"""Extended Dynamic Mode Decomposition.

    A data-driven model to approximate the Koopman operator from time series collection
    data (:py:class:`.TSCDataFrame`) with a finite function basis, specified by the
    data transformations in the EDMD-dictionary. The model is similar to the superclass
    :class:`sklearn.pipeline.Pipeline`, in that the EDMD-dictionary corresponds to the
    transformations in the pipeline and a :class:`.DMDBase` model as the final estimator
    of the pipeline to approximate the Koopman operator (using the EDMD-dictionary time
    series). A key difference to a Pipeline as of scikit-learn is that the states are not
    only mapped forward (to the EDMD-dictionary) but also reconstructed to the original
    full-state time series (typically via Koopman modes).

    If the internal DMD model computes the eigenpairs of the Koopman matrix, then the
    EDMD model provides the Koopman triplet (modes :math:`V`, eigenvalues :math:`\Lambda` and
    eigenfunctions :math:`\xi(\mathbf{x})`)

    .. math::

        \mathbf{x}_{k} = V \Lambda^k \xi(\mathbf{x}_0)

    ...

    Parameters
    ----------
    dict_steps
        List with `(string_identifier, model)` of models to transform the data. The
        list defines the transformation pipeline and order of execution. All models in
        the list must be able to accept :class:`.TSCDataFrame` as input in `fit` and
        output in `transform`.

    dmd_model
        A :py:class:`.DMDBase` as the final estimator. The DMD model either
        approximates the Koopman operator or generator based on the EDMD-dictionary data.

    include_id_state
        If True, the original time series states are added to the EDMD dictionary. The
        mapping from the EDMD dictionary states back to the full-state is then only a
        projection and the cost of an increased EDMD-dictionary dimension. The parameter has
        no effect if all elements in the pipeline preserve the original states in their output,
        as this would duplicate data and lead to conflicts in the feature names (see also
        parameter ``dict_preserves_id_state_``).

    dict_preserves_id_state: str, bool, defaults to "infer"
        The parameter indicates whether the final dictionary state includes the original full
        state data. If

        * True, then this simplifies the inverse map to a projection on the full
          state coordinates. In this case the parameter `include_id_state` is ignored because
          the full state si already contained. An error is raised if not all features can be
          matched).
        * False, then the dictionary does not contain the features of the original full state.
          If also `include_id_state=False`, then the inverse map from dictionary to full state
          is performed in a least squares sense.
        * "infer", then there is a routine that checks the feature names of whether the full
          state is contained in the dictionary, i.e. ``feature_names_in_`` are contained in
          ``feature_names_out_``.

           .. warning::
                If the full state names are contained in the dictionary but are actually
                altered within the dictionary pipeline, then this leads to a wrong state
                reconstruction. If unsure it is safer to explicitly set a boolean.

    sort_koopman_triplets
        Sort the Koopman triplets by the mean absolute value of the initial
        Koopman eigenfunctions of the data passed during fit. Ignored if the Koopman
        triplet ist not available. Sorting criteria is adapted from :cite:t:`manojlovic-2020`.

    use_transform_inverse
        If True, the mapping from EDMD-dictionary states to the full-state is
        performed with ``inverse_transform`` by the models included the dictionary in
        reverse order. Note that all models need to provide ``inverse_transform``.

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

    n_features_in_
        The number of features in the full-state time series (data passed to `fit`).

    feature_names_in_
        The features names in the full-state time series (data passed to `fit`).

    n_features_out_
        The number of features in the EDMD-dictionary time series.

    feature_names_out_
        The feature names in the EDMD-dictionary time series.

    feature_names_pred_
        The feature names in time series predictions.

    named_steps: :class:`Dict[str, object]`
        Read-only attribute to access the EDMD-dictionary models by given name.

    dict_preserves_id_state_: bool
        Boolean flag of whether the full state is contained in the dictionary state. The
        attribute depends on the initialization parameters `include_id_state` and
        `dict_preserves_id_state`.

    is_partial_fit_: bool
        Boolean flag to indicate whether the model is fit with `partial_fit`. If True, then
        only the `partial_fit` method can be used and `fit` raises an error and vice versa.

    koopman_modes: Optional[pandas.DataFrame]
        A ``DataFrame`` of shape `(n_features_original_space, n_features_dict_space)`
        with the modes to map Koopman eigenfunctions to the full-state. The attribute is
        ``None`` if ``use_inverse_transform=True`` or if the DMD model does not provide
        the right eigenvectors of the Koopman matrix.

    koopman_eigenvalues: pandas.Series
        The approximate eigenvalues of the Koopman operator or generator of
        shape `(n_features_dict,)`. The attribute is ``None`` if the DMD model does not
        provide the right eigenvectors of the Koopman matrix.

    n_samples_ic_: int
        The number of time samples required for an initial condition. If the value is
        larger than 1, then for an initial condition a time series is required with the
        same sampling interval of the time series during fit.

    See Also
    --------

    :py:class:`EDMDCV`

    References
    ----------
    * original EDMD description :cite:`williams-2015`
    * for sorting the Koopman components :cite:`manojlovic-2020`

    """

    def __init__(
        self,
        dict_steps: List[Tuple[str, object]],
        dmd_model: Optional[DMDBase] = None,
        *,
        include_id_state: bool = True,
        dict_preserves_id_state: Union[str, bool] = "infer",
        use_transform_inverse: bool = False,
        sort_koopman_triplets: bool = False,
        memory: Optional[Union[str, object]] = None,
        verbose: bool = False,
    ):
        self.dict_steps = dict_steps
        self.dmd_model = dmd_model if dmd_model is not None else DMDFull()
        self.include_id_state = include_id_state
        self.dict_preserves_id_state = dict_preserves_id_state
        self.use_transform_inverse = use_transform_inverse
        self.sort_koopman_triplets = sort_koopman_triplets

        # TODO: if necessary provide option for user defined metric
        self._setup_default_tsc_metric_and_score()

        # type information for some attr set during model construction
        self.dict_preserves_id_states_: bool
        self.n_samples_ic_: int
        self.is_partial_fit_: bool

        all_steps = self.dict_steps + [("dmd", self.dmd_model)]
        super(EDMD, self).__init__(steps=all_steps, memory=memory, verbose=verbose)

    @property
    def _dmd_model(self) -> DMDBase:
        # Improves (internal) code readability when using attribute
        # '_dmd_model' instead of general '_final_estimator'
        return self._final_estimator

    @property
    def koopman_modes(self):
        check_is_fitted(self, attributes=["_koopman_modes"])

        if self._koopman_modes is None:
            return None
        else:
            # pandas object to properly indicate what modes are
            # corresponding to which feature
            modes = pd.DataFrame(
                self._koopman_modes,
                index=self.feature_names_pred_,
                columns=[f"evec{i}" for i in range(self._koopman_modes.shape[1])],
            )
            return modes

    @property
    def koopman_eigenvalues(self):
        check_is_fitted(self)
        if not self._dmd_model.is_spectral_mode:
            raise AttributeError(
                "The underlying DMD model was not configured to provide spectral "
                "components for the Koopman matrix."
            )

        return pd.Series(self._dmd_model.eigenvalues_, name="evals")

    def koopman_eigenfunction(self, X: TransformType) -> TransformType:
        r"""Evaluate the Koopman eigenfunctions.

        The Koopman eigenfunctions (:math:`\xi`) are spectrally aligned states of the
        EDMD-dictionary (:math:`g(x)`).

        * If the internal DMD model provides the left eigenvectors (:math:`\Psi^{-1}`)

            .. math::
                \xi(x) = \Psi^{-1} g(x)

        * If the DMD model only provides the right eigenvectors (:math:`\Psi`)

            .. math::
                \xi(x) = \Psi^{\dagger} g(x)

        See also Eq. 18 in :cite:t:`williams-2015`.

        Parameters
        ----------

        X : TSCDataFrame, pandas.DataFrame
            The samples of the original space at which to evaluate the Koopman
            eigenfunctions. If `n_samples_ic_ > 1`, then the input must be a
            :py:class:`.TSCDataFrame` where each time series must have at least
            ``n_samples_ic_`` samples, with the same time delta as during fit. The input
            must fulfill the first step in the pipeline.

        Returns
        -------
        Union[TSCDataFrame, pandas.DataFrame]
            The evaluated Koopman eigenfunctions. The number of samples are reduced
            accordingly if :code:`n_samples_ic_ > 1`.
        """
        check_is_fitted(self)
        if not self._dmd_model.is_spectral_mode:
            raise AttributeError(
                "The DMD model was not configured to provide spectral "
                "components for the Koopman matrix."
            )

        X_dict = self.transform(X)

        # transform of X_dict matrix
        #   -> note that the transpose is required because in the DMD model the
        #      features are column oriented
        eval_eigenfunction = self._dmd_model.compute_spectral_system_states(
            X_dict.to_numpy().T
        )

        columns = [f"koop_eigfunc{i}" for i in range(eval_eigenfunction.shape[0])]
        eval_eigenfunction = df_type_and_indices_from(
            indices_from=X_dict, values=eval_eigenfunction.T, except_columns=columns
        )

        return eval_eigenfunction

    @property
    def n_features_out_(self):
        # Note: this returns the number of features by the dictionary transformation,
        # NOT from a EDMD prediction
        # TODO: raise NotFittedError if trying to access before .fit? (relevant for all
        #  following attrs)
        return self._dmd_model.n_features_in_

    @property
    def feature_names_out_(self):
        # Note: this returns the feature names of the dictionary
        # transformation NOT from a EDMD prediction (see feature_names_pred_)
        return self._dmd_model.feature_names_in_

    @property
    def control_names_in_(self):
        if self.is_controlled_:
            return self.dmd_model.control_names_in_
        else:
            return None

    @property
    def n_control_in_(self):
        if self.is_controlled_:
            return self.dmd_model.n_control_in_
        else:
            return None

    @property
    def feature_names_pred_(self):
        # TODO: should TSCPredictMixin include feature names for prediction?
        return self._feature_names_pred

    def _validate_dictionary(self) -> bool:
        """Validates that all elements in the EDMD dictionary.

        During the validation the methods evaluates a flag to indicate whether the original
        states are still the dictionary transformation.

        Returns
        -------
        bool
            boolean flag if dictionary preserves the original states
        """

        for (_, trans_str, transformer) in self._iter(with_final=False):
            if not isinstance(transformer, TSCTransformerMixin):
                raise TypeError(
                    "The EDMD dictionary only supports datafold transformers that handle the "
                    "data structure 'TSCDataFrame'."
                )

        return True

    def _set_dict_preserves_id_state(self, X_dict: TSCDataFrame) -> bool:
        if self.dict_preserves_id_state == "infer":
            return np.isin(self.feature_names_in_, X_dict.columns).all()
        elif isinstance(self.dict_preserves_id_state, bool):
            if self.dict_preserves_id_state and self.include_id_state:
                warnings.warn(
                    f"setting {self.dict_preserves_id_state=} and {self.include_id_state=} "
                    f"duplicates the original features in the dictionary"
                )

            return self.dict_preserves_id_state
        else:
            raise ValueError(
                f"Could not read {self.dict_preserves_id_state=}. Set to string "
                f"'infer' or a boolean value"
            )

    def _compute_n_samples_ic(self, X, X_dict):
        diff = X.n_timesteps - X_dict.n_timesteps

        if isinstance(diff, pd.Series):
            # time series can have different number of time values (in which case it is
            # a Series), however, the difference has to be the same for all time series
            assert (diff.iloc[0] == diff).all()
            diff = diff.iloc[0]

        # +1 because the diff indicates how many samples were removed -- we want the
        # number that is required for the initial condition
        return int(diff) + 1

    def _least_squares_inverse_map(self, X, X_dict, U):
        if isinstance(X, pd.DataFrame):
            if U is not None:
                X = X.loc[
                    U.index,
                ].to_numpy()
            else:
                X = X.to_numpy()

        if isinstance(X_dict, pd.DataFrame):
            if U is not None:
                X_dict = X_dict.loc[
                    U.index,
                ].to_numpy()
                U = U.to_numpy()
            else:
                X_dict = X_dict.to_numpy()

        if False and U is not None:
            matrix = scipy.linalg.lstsq(np.column_stack([X_dict, U]), X, cond=None)[0]
            self._inverse_map_control: np.ndarray = matrix[X_dict.shape[1] :, :]
            return matrix[: X_dict.shape[1], :]
        else:
            return scipy.linalg.lstsq(X_dict, X, cond=None)[0]

    def _compute_inverse_map(
        self, X: TSCDataFrame, X_dict: TSCDataFrame, U
    ) -> Optional[Union[scipy.sparse.csr_matrix, np.ndarray]]:
        """Compute matrix that linearly maps from dictionary space to original feature
        space.

        # TODO: If EDMD is controlled there is currently only a map from dictionary states to
            the full-state representation.
            - i.e.
            X_dict B = X
            -
            However, the control states can also be included in the mapping, which can improve
            the mapping
            -
            [X_dict, U] [B_dict, B_u] = X
            -
            This is also captured in a typical state space representation (matrix "D")
            see https://en.wikipedia.org/wiki/State-space_representation#Linear_systems

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
        numpy.ndarray, scipy.sparse.csr_matrix, None

        """

        if self.include_id_state or self.dict_preserves_id_state_:
            # trivial case: we just need a projection matrix to select the
            # original full-states from the dictionary functions
            try:
                inverse_map = projection_matrix_from_feature_names(
                    X_dict.columns, self.feature_names_in_
                )
            except ValueError as e:
                # here it is assumed that the the error is not raised if
                # self.include_id_state=True because we have the control over it
                raise ValueError(
                    f"{self.dict_preserves_id_state_=} but not all features "
                    f"names could be found in the dictionary's feature names"
                ) from e
        else:
            # Compute the matrix in a least squares sense
            # inverse_map = "B" in Williams et al., Eq. 16
            inverse_map = self._least_squares_inverse_map(
                X=X.loc[X_dict.index, :], X_dict=X_dict, U=U
            )

        return inverse_map

    def _compute_koopman_modes(self, inverse_map: np.ndarray) -> np.ndarray:
        r"""Compute Koopman modes.

        The Koopman modes :math:`V` are a computed with

        .. math::
            V = B \cdot \Psi_{DMD}

        where :math:`B` is matrix that maps the EDMD-dictionary states to the full-space
        features and :math:`\Psi_{DMD}` are the right eigenvectors of the Koopman
        matrix, computed in the internal DMD model.

        See :cite:`williams_datadriven_2015` Eq. 20.

        Parameters
        ----------

        inverse_map
            Matrix :math:`B`, as computed in :py:meth:`._compute_inverse_map`

        Returns
        -------
        numpy.ndarray
            Koopman modes
        """
        koopman_modes = inverse_map.T @ self._dmd_model.eigenvectors_right_
        return koopman_modes

    def _sort_koopman_triplets(self, X_dict_ic: TSCDataFrame) -> None:
        r"""The ranking and sorting of Koopman triplets, adapted from
        https://arxiv.org/pdf/2006.11765.pdf

        Given the Koopman mode decomposition

        .. math::

                x_{j+1} = V \Lambda \xi(x_j} = \sum_{p=1}^P \vec{v}_p \lambda_p \xi_p(x_j)

        where :math:`x` is the full-spate vector, :math:`V` the Koopman modes,
        :math:`\Lambda` the diagonal matrix with the eigenvalues and :math:`\xi` the
        Koopman eigenfunctions.

        We assume the Koopman modes are normed to length 1, which can then be rewritten as
        .. math::

                x_{j+1} = V F^{-1} \Lambda F \xi(x_j)

        where :math:`F` is a diagonal matrix containing the norms for normalization.
        The ranking is then computed for each triplet :math:`p` with the absolute value
        of the initial Koopman eigenfunction

        .. math::

            \operatorname{importance}(p) = \vert F_p \xi_p(x_0) \vert

        If there are multiple initial conditions in the data present, then the mean is
        taken over all initial conditions.

        Finally, the triplets (modes, eigenvalues, eigenfunctions) are sorted
        from high importance to low importance according to their computed value.

        Parameters
        ----------
        X_dict_ic
            The initial states of the EDMD-dictionary time series.
        """

        ic_koop_eigenfunc = self.dmd_model.compute_spectral_system_states(
            X_dict_ic.to_numpy().T
        )

        # the importance ranking is with the assumption that the modes are
        # normalized to 1 -- the factor corrects the ic_koop_eigenfunc accordingly
        # (1/factor * modes) * eigvals * (factor * eigfunc)
        factor = np.linalg.norm(self._koopman_modes, axis=0)
        triplet_importance = np.abs((factor[:, np.newaxis] * ic_koop_eigenfunc))
        # take the mean over all
        triplet_importance = np.mean(triplet_importance, axis=1)

        argsort_importance = np.argsort(triplet_importance.ravel())[::-1]

        # Sort everything:
        self._koopman_modes: np.ndarray = self._koopman_modes[:, argsort_importance]
        self._dmd_model.eigenvalues_ = self._dmd_model.eigenvalues_[argsort_importance]
        self._dmd_model.eigenvectors_right_ = self._dmd_model.eigenvectors_right_[
            :, argsort_importance
        ]

        if (
            hasattr(self._dmd_model, "eigenvectors_left_")
            and self._dmd_model.eigenvectors_left_ is not None
        ):
            # Note the left eigenvectors are in the rows of the matrix by convention
            self._dmd_model.eigenvectors_left_ = self._dmd_model.eigenvectors_left_[
                argsort_importance, :
            ]

    def _attach_id_state(self, X, X_dict):
        # remove states from X (the id-states) that are also removed during dictionary
        # transformations
        X = X.loc[X_dict.index, :]
        try:
            X = pd.concat([X, X_dict], axis=1)
        except AttributeError as e:
            all_columns = X_dict.columns.append(X.columns)
            duplicates = all_columns[all_columns.duplicated()]

            if len(duplicates) > 0:
                raise ValueError(
                    "The ID state could not be attached, because the columns\n"
                    f"{duplicates}\n"
                    f"are also present in the dictionary."
                ) from e
            else:
                raise e

        return X

    def _partial_fit(self, X, y=None, **fit_params_steps):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()

        if self.memory is not None:
            raise ValueError(f"{self.memory=} is not supported for partial fit")

        for (step_idx, name, transformer) in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            # Fit the current transformer
            fitted_transformer = transformer.partial_fit(X, y, **fit_params_steps[name])
            X = transformer.transform(X)

            self.steps[step_idx] = (name, fitted_transformer)
        return X

    def _reconstruct(self, X: TSCDataFrame, U, qois):

        X_reconstruct = []
        for X_ic, time_values in InitialCondition.iter_reconstruct_ic(
            X, n_samples_ic=self.n_samples_ic_
        ):
            # transform initial condition to EDMD-dictionary space
            X_dict_ic = self.transform(X_ic)

            X_est_ts = self._predict_ic(
                X_dict=X_dict_ic, U=U, time_values=time_values, qois=qois
            )

            X_reconstruct.append(X_est_ts)

        X_reconstruct = pd.concat(X_reconstruct, axis=0)
        assert isinstance(X_reconstruct, TSCDataFrame)

        # NOTE: time series contained in X_reconstruct can be shorter than
        # the original time series (i.e. no full reconstruction), because some transform
        # models drop samples (e.g. time delay embeddings or finite differences)
        return X_reconstruct

    def _predict_ic(self, X_dict: TSCDataFrame, U, time_values, qois) -> TSCDataFrame:
        """Prediction with initial condition.

        Parameters
        ----------
        X_dict
            The initial condition in EDMD dictionary space.

        U
            The control time series (None if not needed)

        time_values
            The future time values to evaluate the system at.

        qois
            A subselection of quantities of interest (must be in feature_names_pred_).

        Returns
        -------

        """

        if qois is None:
            feature_columns = self.feature_names_pred_
        else:
            feature_columns = qois

        if self._inverse_map is not None:
            if self._koopman_modes is not None:
                # The DMD model computed spectral components and the
                # Koopman modes are available.

                if qois is None:
                    modes = self.koopman_modes.to_numpy()
                else:
                    project_matrix = projection_matrix_from_feature_names(
                        self.feature_names_pred_, qois
                    )
                    modes = project_matrix.T @ self._koopman_modes

                # compute the time series in original space directly by adapting the modes

                dmd_params = dict(
                    U=U,
                    time_values=time_values,
                    modes=modes,
                    feature_columns=feature_columns,
                )
                dmd_params = {k: v for k, v in dmd_params.items() if v is not None}

                X_ts = self._dmd_model.predict(X_dict, **dmd_params)
            else:
                # The DMD model does not provide the spectral components of the
                # Koopman matrix. The inverse_map needs to be applied afterwards because
                # the DMD model requires to maintain a square matrix to forward the
                # system.

                # computes system in EDMD-dictionary space
                dmd_params = dict(U=U, time_values=time_values)
                dmd_params = {k: v for k, v in dmd_params.items() if v is not None}

                X_ts = self._dmd_model.predict(X_dict, **dmd_params)

                # map back to original space and select qois

                if hasattr(self, "_inverse_map_control"):
                    vals = (X_ts.to_numpy() @ self._inverse_map)[
                        :-1, :
                    ] + U.to_numpy() @ self._inverse_map_control
                    idx = X_ts.index[:-1]
                else:
                    vals = X_ts.to_numpy() @ self._inverse_map
                    idx = X_ts.index

                X_ts = TSCDataFrame(
                    vals,
                    columns=self.feature_names_pred_,
                    index=idx,
                ).loc[:, feature_columns]

        else:
            # predict all EDMD-dictionary time series
            X_ts = self._dmd_model.predict(X_dict, time_values=time_values)

            # transform from EDMD-dictionary space by pipeline inverse_transform
            X_ts = self.inverse_transform(X_ts)
            X_ts = X_ts.loc[:, feature_columns]

        return X_ts

    def fit(
        self,
        X: TimePredictType,
        U: Optional[TimePredictType] = None,
        y=None,
        **fit_params,
    ) -> "EDMD":
        r"""Fit the model.

        Internally the method calls `fit_transform` of all models contained in the
        EDMD dictionary (in given order) and then approximates the Koopman operator in the
        DMD model as the final estimator of the pipeline. Finally, the inverse map from
        dictionary states to the original states is set up.

        Parameters
        ----------
        X
            Training time series data. Must fulfill input requirements of first
            `dict_step` in the EDMD-dictionary pipeline.

        U
            Time series with control states acting on the system. The states are passed to the
            DMD model, at which point the time indices must be identical to the states in `X`.

        y : None
            ignored

        **fit_params: Dict[str, object]
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``. To add parameters for the internal DMD model use
            ``s=dmd``, e.g. ``dmd__param``.

        Returns
        -------
        EDMD
            self

        Raises
        ------
        TSCException
            Time series collection restrictions in `X`: (1) time delta must be constant
            (2) all time series values must be finite (no `NaN` or `inf`)
        """

        # Currently, validation of U is only performed in the final DMD estimator
        self._validate_datafold_data(
            X,
            ensure_tsc=True,
            tsc_kwargs=dict(ensure_const_delta_time=True),
        )

        if hasattr(self, "is_partial_fit_") and self.is_partial_fit_:
            raise ValueError(
                f"Fit cannot be called if the model was set up with {self.is_partial_fit_=}"
            )
        else:
            self.is_partial_fit_ = False

        self.is_controlled_ = False if U is None else True

        # 1) first get the EDMD fit_params, 2) validate the fit_params for the pipeline,
        # 2) separate the DMD fit_params as the dmd is called later
        fit_params = self._check_fit_params(**fit_params or {})
        dmd_fit_params = fit_params.pop("dmd", None)

        self._validate_dictionary()

        # NOTE: self._setup_features_and_time_fit(X) is not called here, because the
        # n_features_in_ and n_feature_names_in_ is delegated to the first instance in
        # the pipeline. The time values are set separately here:
        self._validate_time_values_format(time_values=X.time_values())
        self.dt_ = X.delta_time
        self._feature_names_pred = X.columns

        # '_fit' calls internally fit_transform (!!), and stores results into cache if
        # "self.memory is not None" (see docu):
        X_dict = self._fit(X, y, **fit_params)
        self.n_samples_ic_ = self._compute_n_samples_ic(X, X_dict)

        # Either automatically detected of enforced in dict_preserves_id_states
        self.dict_preserves_id_state_ = self._set_dict_preserves_id_state(X_dict)

        if self.include_id_state and not self.dict_preserves_id_state_:
            # only attach original states if they are not preserved
            X_dict = self._attach_id_state(X=X, X_dict=X_dict)

        if self.is_controlled_ and self.n_samples_ic_ > 1:
            # intersection of indices, because often U does not require *all* indices in X
            # (mainly the last control state should not be included)
            assert U is not None  # for mypy
            inters_keys = X_dict.index.intersection(U.index)
            U = U.loc[inters_keys, :]  # type: ignore

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self.is_controlled_:
                self._dmd_model.fit(X=X_dict, U=U, y=y, **dmd_fit_params)
            else:
                self._dmd_model.fit(X=X_dict, y=y, **dmd_fit_params)

        if not self.use_transform_inverse:
            self._inverse_map = self._compute_inverse_map(X=X, X_dict=X_dict, U=U)

            if self.dmd_model.is_spectral_mode:

                if self.is_controlled_:
                    raise NotImplementedError(
                        "currently the inverse mapping from a spectral "
                        "system representation and control is not captured in datafold."
                    )

                self._koopman_modes = self._compute_koopman_modes(
                    inverse_map=self._inverse_map,
                )

                if self.sort_koopman_triplets:
                    self._sort_koopman_triplets(X_dict_ic=X_dict.initial_states())

            else:
                self._koopman_modes = None
        else:
            # Indicator to use `ìnverse_transform` of dictionary and not linear map of
            # Koopman modes.
            self._inverse_map, self._koopman_modes = None, None

        return self

    def predict(
        self,
        X: InitialConditionType,
        *,
        U: Optional[InitialConditionType] = None,
        time_values: Optional[np.ndarray] = None,
        qois: Optional[Union[pd.Index, List[str]]] = None,
        **predict_params,
    ):
        """Evaluate dynamical system for one or many initial conditions.

        The internal prediction steps are:

        1. Transform initial conditions to EDMD-dictionary states.
        2. Predict EDMD-dictionary time series with the DMD model.
        3. Map the EDMD-dictionary time series back to the full-state time series.

        Depending on the configuration, step 2 and 3 can be carried out in a single linear
        system.

        Parameters
        ----------
        X: TSCDataFrame, numpy.ndarray
            The initial conditions, where initial condition must have exactly
            ``n_samples_ic_`` samples (mapped to a single EDMD-dictionary state). The
            preferred input type is :py:class:`TSCDataFrame`. If only a single state
            is required (``n_samples_ic_ = 1``), then a :class:`numpy.ndarray` is also
            accepted, where each row corresponds to an initial condition.

        U
            If the model was fit with control input, then this argument is required.
            For each initial condition in `X` there must be a corresponding time series with
            the control states over the prediction horizon in `U`. Each time series in `U`
            must have the same time values. The time horizon is taken from `U` (i.e.
            ``time_values`` has to be either identical or ``None``).

        time_values
            The time values to evaluate the model at for each initial condition. The values
            should be ascending and non-negative numeric values. Default parameter

            * uncontrolled system: time series of lengths two, where the first state is the
              initial condition and the second state the evaluation after `self.dt_`
            * controlled system: the time values are taken from the parameter ``U``.

        qois
            A list of feature names of interest to be included in the returned
            predictions. Must be a subset of ``feature_names_pred_``.

        **predict_params: Dict[str, object]
            ignored

        Returns
        -------
        TSCDataFrame
            Predicted time series collection, where each time series is evaluated at the
            specified time values.

        Raises
        ------
        TSCException
            Time series collection requirements in `X`: (1) same (constant) time delta as
            during fit (2) all time series must have identical
            time values (3) all values must be finite (i.e. no `NaN` or `inf` values).
        """
        check_is_fitted(self)

        time_values = self._validate_and_set_time_values_predict(
            time_values=time_values, X=X, U=U
        )

        if isinstance(X, np.ndarray):
            # work only with TSCDataFrame internally
            X = InitialCondition.from_array(
                if1dim_rowvec(X),
                time_value=time_values[0],
                feature_names=self.feature_names_in_,
                ts_ids=U.ids if isinstance(U, TSCDataFrame) else None,
            )
        else:
            InitialCondition.validate(
                X,
                n_samples_ic=self.n_samples_ic_,
                dt=self.dt_ if self.n_samples_ic_ > 1 else None,
            )

        if self.is_controlled_:
            if isinstance(U, np.ndarray):

                if X.n_timeseries > 1:
                    raise NotImplementedError(
                        "If U is a numpy array, then only a prediction with "
                        "a single initial condition is allowed. "
                        f"Got {X.n_timeseries}"
                    )

                U = InitialCondition.from_array_control(
                    U,
                    control_names=self.control_names_in_,
                    dt=self.dt_,
                    time_values=time_values[:-1],
                    ts_id=int(X.ids[0]) if isinstance(X, TSCDataFrame) else None,
                )
            elif isinstance(U, TSCDataFrame):
                InitialCondition.validate_control(X_ic=X, U=U)
            else:
                raise TypeError(f"U has invalid type (got {type(U)}.")

        self._validate_feature_names(X, U)

        qois = self._validate_qois(
            qois=qois, valid_feature_names=self._feature_names_pred
        )

        self._validate_datafold_data(
            X,
            ensure_tsc=True,
        )

        X_dict = self.transform(X)

        if self.is_controlled_:
            assert isinstance(U, TSCDataFrame)
            U = U.loc[
                U.index.get_level_values(TSCDataFrame.tsc_time_idx_name)
                >= time_values[0],
                :,
            ]

        X_ts = self._predict_ic(X_dict=X_dict, U=U, time_values=time_values, qois=qois)
        return X_ts

    def fit_predict(
        self,
        X: TSCDataFrame,
        U: Optional[TSCDataFrame] = None,
        y=None,
        qois: Optional[Union[pd.Index, List[str]]] = None,
        **fit_params,
    ):
        """Fit the model and reconstruct the training data.

        Parameters
        ----------
        X
            Training time series data. Must fulfill input requirements of first
            `dict_step` in the EDMD-dictionary pipeline.

        U
            Control time series passed to the DMD model. At this point the index of `U` must
            be identical to `X_dict`.

        y: None
            ignored

        qois
            A list of feature names of interest to be included in the returned
            predictions. Must be a subset of ``feature_names_pred_``. Passed to
            :py:meth:`.predict`.

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
        return self.fit(X=X, U=U, y=y, **fit_params).reconstruct(X=X, U=U, qois=qois)

    def partial_fit(self, X: TimePredictType, U=None, y=None, **fit_params) -> "EDMD":
        """Incremental fit of the model.

        The partial fit call is forwarded to all transformers in the dictionary and the set
        DMD model. The update fails if one model does not support incremental updates.

        Parameters
        ----------
        X
            The new batch of training time series.

        U
            Currently, there is no implementation that supports both an online/streaming
            setting with control.

        y
            ignored

        fit_params
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``. To add parameters for the set DMD model use
            ``s=dmd``, e.g. ``dmd__param``.

        Returns
        -------
        self
            updated model
        """
        self._validate_datafold_data(
            X,
            ensure_tsc=True,
            tsc_kwargs=dict(ensure_const_delta_time=True),
        )

        if U is not None:
            raise NotImplementedError(
                "Currently there are no DMD models that that support both streaming "
                "and control."
            )

        try:
            check_is_fitted(self)
            initial_fit = False
            assert self.is_partial_fit_
        except NotFittedError:
            initial_fit = True

            if hasattr(self, "is_partial_fit_") and not self.is_partial_fit_:
                raise ValueError(
                    "The model is already build with the 'fit' method. "
                    "Please use `partial_fit` also for the initial fit."
                )
            else:
                self.is_partial_fit_ = True
            self.dt_ = X.delta_time
            self._feature_names_pred = X.columns

        # '_fit' calls internally fit_transform (!!), and stores results into cache if
        # "self.memory is not None" (see docu):
        fit_params = self._check_fit_params(**fit_params or {})
        dmd_fit_params = fit_params.pop("dmd", None)

        X_dict = self._partial_fit(X, y, **fit_params)

        if initial_fit:
            self.dict_preserves_id_state_ = self._validate_dictionary()

        if initial_fit:
            self.n_samples_ic_ = self._compute_n_samples_ic(X, X_dict)

        if self.include_id_state and not self.dict_preserves_id_state_:
            X_dict = self._attach_id_state(X=X, X_dict=X_dict)
        elif not self.include_id_state and not not self.dict_preserves_id_state_:
            raise NotImplementedError(
                "Currently, there is no implementation to partial_fit non-trivial update of "
                "the modes. Either the dictionary must preserve the id states or "
                "`include_is_state=True`."
            )

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            self._dmd_model.partial_fit(X=X_dict, y=y, **dmd_fit_params)

        if (
            not self.use_transform_inverse
        ):  # TODO: avoid duplicated code with fit() here!

            if not self.use_transform_inverse:
                self._inverse_map = self._compute_inverse_map(X=X, X_dict=X_dict, U=U)

            if self.dmd_model.is_spectral_mode:
                self._koopman_modes = self._compute_koopman_modes(
                    inverse_map=self._inverse_map,
                )

                if self.sort_koopman_triplets:
                    self._sort_koopman_triplets(X_dict_ic=X_dict.initial_states())

            else:
                self._koopman_modes = None
        else:
            # Indicator to use `ìnverse_transform` of dictionary and not linear map of
            # Koopman modes.
            self._inverse_map, self._koopman_modes = None, None

        return self

    def transform(self, X: TransformType) -> TransformType:
        """Map the original states to the dictionary representation.

        Parameters
        ----------
        X : TSCDataFrame, pandas.DataFrame
           Time series to transform. Must fulfill the input requirements of
           first step of the pipeline. Each time series must have a minimum of
           ``n_samples_ic_`` samples.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame
            The transformed time series. The number of samples of each time series are reduced
            if `n_samples_ic_ > 1`.
        """
        if self.include_id_state:
            # copy required to properly attach X later on
            X_dict = X.copy(deep=True)
        else:
            X_dict = X

        # TODO: need to validate data here?

        # carry out dictionary transformations:
        for _, name, tsc_transform in self._iter(with_final=False):
            X_dict = tsc_transform.transform(X_dict)

        if self.include_id_state and not self.dict_preserves_id_state_:
            X_dict = self._attach_id_state(X=X, X_dict=X_dict)

        return X_dict

    def fit_transform(  # type: ignore[override]
        self, X: TSCDataFrame, *, U: Optional[TSCDataFrame] = None, y=None, **fit_params
    ):
        """Fit the model and return the EDMD dictionary time series.

        Parameters
        ----------
        X
            Time series collection data to fit the model.

        U
            Control time series passed to :py:meth:`.fit`.

        y: None
            ignored

        **fit_params: Dict[str, object]
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        TSCDataFrame
             EDMD-dictionary time series data.

        Raises
        ------
        TSCException
            Time series collection restrictions in `X`: (1) time delta must be constant
            (2) all values must be finite (no `NaN` or `inf`)
        """
        return self.fit(X=X, U=U, y=y, **fit_params).transform(X)

    def inverse_transform(self, X: TransformType) -> TransformType:
        """Perform inverse dictionary transformations on dictionary time series.

        The actual performed inverse transformation depends on the parameter settings
        ``include_id_state`` and ``use_transform_inverse``.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame
            Time series to map back to the full-state time series. The feature names
            must match the ones in attribute ``feature_names_out_``.

        Returns
        -------
        TSCDataFrame
            full-state time series
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

    def reconstruct(
        self,
        X: TSCDataFrame,
        *,
        U: Optional[TSCDataFrame] = None,
        qois: Optional[Union[pd.Index, List[str]]] = None,
    ) -> TSCDataFrame:
        """Reconstruct existing time series collection.

        Internal steps to reconstruct a time series collection:

        1. Extract the initial conditions (first ``n_samples_ic_`` samples) from each
           time series in the collection.
        2. Predict the remaining states of each time series with the built EDMD model
           at the same time values.

        Parameters
        ----------
        X
            The time series collection to reconstruct. The first ``n_samples_ic_`` of
            each time series must fulfill the requirements of an initial condition.

        U
            Control time series for the data. Only required if model was fit with control.

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
            Time series collection requirements in `X`: (1) all values must be
            finite (no `NaN` or `inf`)
        """
        check_is_fitted(self)

        X = self._validate_datafold_data(
            X,
            ensure_tsc=True,
            # Note: no const_delta_time required here. The required const samples for
            # time series initial conditions is included in the predict method.
        )
        self._validate_feature_names(X=X, U=U)
        self._validate_qois(qois=qois, valid_feature_names=self.feature_names_pred_)

        return self._reconstruct(X=X, U=U, qois=qois)

    def score(
        self,
        X: TSCDataFrame,
        U=None,
        y=None,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """Score between given time series and its model reconstruction.

        Parameters
        ----------
        X
            The time series collection to reconstruct. The first ``n_samples_ic_`` of
            each time series must fulfill the requirements of an initial condition.

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
        X_reconstruct = self.reconstruct(X=X, U=U)

        if self.n_samples_ic_ > 1:
            # Note that during `reconstruct` samples can be discarded (e.g. when
            # applying time delay embedding). In the latent space there are then less
            # samples than in the full-state, which is corrected here:
            X = X.loc[X_reconstruct.index, :]

        return self._score_eval(X, X_reconstruct, sample_weight)


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
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
):
    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0] + 1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += f"; {candidate_progress[0] + 1}/ {candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        edmd = edmd.set_params(**cloned_parameters)

    start_time = time.time()

    # NOTE: this deviates from the _fit_and_score in sklearn, which uses a
    #  _safe_split function (defined in metaestimators.py)
    X_train, X_test = _split_X_edmd(X, y, train_indices=train, test_indices=test)

    result = {}
    try:
        edmd.fit(X_train, y=None, **fit_params)
    except Exception:
        # Handle all exception, to not waste other working or complete results
        fit_time = time.time() - start_time  # Note fit time as time until error
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            # Note fit time as time until error
            fit_time = time.time() - start_time
            score_time = 0.0
            if error_score == "raise":
                raise
            elif isinstance(error_score, numbers.Number):
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score

                warnings.warn(
                    "Estimator fit failed. The score on this train-test"
                    f" partition for these parameters will be set to {error_score}. "
                    "Details: \n%s" % (format_exc()),
                    FitFailedWarning,
                )
            result["fit_failed"] = True
    else:
        result["fit_failed"] = False

        fit_time = time.time() - start_time
        test_scores = _score(edmd, X_test, None, scorer, error_score)
        score_time = time.time() - start_time - fit_time

        if return_train_score:
            train_scores = _score(edmd, X_train, None, scorer, error_score)

    if verbose > 1:

        sorted_keys = sorted(parameters)  # Ensure deterministic o/p
        params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)

        total_time = score_time + fit_time
        end_msg = "[CV] END "
        result_msg = params_msg + (";" if params_msg else "")
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    # mypy says: # FIXME
    #   error: Incompatible types in assignment (expression has type "Number",
    #   target has type "bool")
    # I am not sure why target (the assignment to key in dict) has type bool -
    # to avoid the error I ignored the type
    result["test_scores"] = test_scores  # type: ignore
    if return_train_score:
        result["train_scores"] = train_scores  # type: ignore
    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X_test)  # type: ignore
    if return_times:
        result["fit_time"] = fit_time  # type: ignore
        result["score_time"] = score_time  # type: ignore
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = edmd
    return result


class EDMDCV(GridSearchCV):
    """Exhaustive parameter search over specified grid for a :class:`EDMD` model with
    cross-validation.

    .. note::
        EDMDCV sublasses from :py:class:`sklearn.GridSearchCV`. However, it does not
        support the parameter ``scoring`` which enables multi metric evaluations.
        Furthermore, ``refit`` is restricted to a bool.

    ...

    Parameters
    ----------
    estimator
        Model to be optimized. Uses the ``score`` function set in the model.

    param_grid
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    n_jobs
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    pre_dispatch
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - A string 'all', in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    cv
        Determines the cross-validation splitting strategy. Possible inputs are:

        - :class:`.TSCKfoldSeries` splits `k` folds across time series (useful when
            many time series are in a collection)
        - :class:`.TSCKFoldTime` splits `k` folds across time

    refit
        Refit an estimator using the best found parameters on the whole
        dataset.

    verbose : :class:`int`
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        `FitFailedWarning` is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is ``np.nan``.

    return_train_score
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
        imported into a ``pandas.DataFrame``. See documentation in super class.

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.
        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator
        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

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
    data. If `n_jobs` was set to a value higher than one, the data is copied for each
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
        cv: TSCCrossValidationSplit,
        n_jobs: Optional[int] = None,
        pre_dispatch: Union[int, str] = "2*n_jobs",
        refit: bool = True,
        verbose: int = 1,
        error_score: Union[str, numbers.Number] = "raise",
        return_train_score: bool = True,
    ):

        super(EDMDCV, self).__init__(
            estimator=estimator,
            param_grid=param_grid,
            scoring=None,
            n_jobs=n_jobs,
            cv=cv,
            refit=refit,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _validate_settings_edmd(self):
        if not isinstance(self.estimator, EDMD):
            raise TypeError("EDMDCV only supports EDMD estimators.")

        if not isinstance(self.cv, TSCCrossValidationSplit):
            raise TypeError(f"cv must be of type {(TSCKfoldSeries, TSCKFoldTime)}")

    def fit(self, X: TSCDataFrame, y=None, **fit_params):
        """Fit and score the model for all parameter candidates.

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

        refit_metric = "score"

        def scorers(estimator, X, y=None):
            return estimator.score(X)

        X, y = indexable(X, y)

        fit_params = _check_fit_params(X, fit_params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        n_splits = cv_orig.get_n_splits(X, y)

        base_estimator = deepcopy(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

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
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):

                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        f"Fitting {n_splits} folds for each of {n_candidates} candidates,"
                        f" totalling {n_candidates * n_splits} fits."
                    )

                out = parallel(
                    delayed(_fit_and_score_edmd)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(
                        enumerate(candidate_params), enumerate(cv.split(X, y))
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
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)
                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

        self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
        self.best_score_ = results["mean_test_%s" % refit_metric][self.best_index_]
        self.best_params_ = results["params"][self.best_index_]

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = deepcopy(
                deepcopy(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()

            self.best_estimator_.fit(X, **fit_params)

            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self


class EDMDWindowPrediction(object):
    """Adapt EDMD model to perform reconstruct and score time series of same length.

    The adaptation of the EDMD model is useful if a fixed prediction horizon is
    tested. Instead of the default behaviour of EDMD to reconstruct the full time series
    found in a :py:class`.TSCDataFrame`, time series of equal length are extracted from
    the data and then reconstructed.

    Parameters
    ----------

    window_size
        An integer value indicating the time steps to include in a window. The value
        must be greater than the the attribute ``edmd.n_samples_ic_``, because a
        window also contains the samples dedicated for the initial condition.

    offset
        An integer value to indicate the offset between two windows. When setting
        `offset=window_size-edmd.n_samples_ic_`, then the test samples do not overlap
        between windows and samples are not dropped.

    """

    def __init__(self, window_size: int = 10, offset: int = 10):
        self.window_size = window_size
        self.offset = offset

    def _validate(self):

        if self.window_size is not None and self.offset is not None:
            check_scalar(
                self.window_size,
                name="time_horizon",
                target_type=(np.integer, int),
                min_val=1,
            )

            check_scalar(
                self.offset, name="offset", target_type=(np.integer, int), min_val=1
            )
        elif self.window_size is not None or self.offset is not None:
            raise ValueError("'window_size' and 'offset' must be provided together")

    def _window_reconstruct(
        self,
        X: TSCDataFrame,
        edmd: EDMD,
        offset: int,
        y=None,
        qois=None,
        return_X_windows: bool = False,
    ):
        """Reconstruct existing time series of equal length.

        This method is used to overwrite the default score method of an :py:class:`.EDMD`
        model, which in contrast to this model scores on the time series found in `X`.

        In this method, the time series are subdivided into smaller time series of
        equal length (windows). Each window contains the initial condition and the
        samples to score the model against. This therefore corresponds to a more
        systematic approach.

        Parameters
        ----------
        X
            The time series collection to reconstruct. From each time series of the
            collection windows are extracted and separately reconstructed.

        qois
            A list of feature names of interest to be include in the returned
            predictions. Passed to :py:meth:`.predict`.

        return_X_windows
            If True, then an additional time series collection is returned,
            which contains extracted windows from `X`.

        Returns
        -------
        TSCDataFrame, Optional[TSCDataFrame]
            The reconstructed time series collection and if `return_X_windows=True`
            also the extracted windows from `X`.

        """

        if not hasattr(edmd, "window_size"):
            raise AttributeError(
                "The EDMD object requires the attribute 'window_size' "
                "to perform windowed reconstruction in data."
            )

        if not isinstance(edmd.window_size, int):
            raise TypeError("'window_size' must be of type int")

        if edmd.window_size <= edmd.n_samples_ic_:
            raise ValueError(
                f"edmd.window_size={edmd.window_size} must be larger than the number of "
                "samples required to make an initial condition ("
                f"edmd.n_samples_ic_={edmd.n_samples_ic_})"
            )

        X = edmd._validate_datafold_data(
            X,
            ensure_tsc=True,
            tsc_kwargs=dict(
                ensure_const_delta_time=True, ensure_min_timesteps=edmd.window_size
            ),
        )
        qois = edmd._validate_qois(
            qois=qois, valid_feature_names=edmd.feature_names_pred_
        )

        X_windows = TSCDataFrame.from_frame_list(
            list(
                X.tsc.iter_timevalue_window(
                    window_size=edmd.window_size,
                    offset=offset,
                    per_time_series=True,
                    strictly_sequential=True,
                )
            )
        )

        final_index_windows = X_windows.index.copy()

        n_timesteps = X_windows.n_timesteps
        assert isinstance(n_timesteps, int) and n_timesteps == edmd.window_size

        final_index_reconstruct = (
            X_windows.groupby(TSCDataFrame.tsc_id_idx_name)
            .tail(edmd.window_size - edmd.n_samples_ic_ + 1)
            .index
        )

        first_id_time_values = X_windows.loc[X_windows.ids[0]].index
        X_windows.index = pd.MultiIndex.from_product(
            [X_windows.ids, first_id_time_values],
            names=[TSCDataFrame.tsc_id_idx_name, TSCDataFrame.tsc_time_idx_name],
        )

        X_reconstruct = edmd._reconstruct(X=X_windows, U=None, qois=qois)

        # recover true index:
        X_windows.index = final_index_windows
        X_reconstruct.index = final_index_reconstruct

        if return_X_windows:
            return X_reconstruct, X_windows
        else:
            return X_reconstruct

    def _window_score(self, X, y=None, sample_weight=None, qois=None, edmd=None):
        """Score of reconstructed windowed time series collection.

        This method can overwrite the default score method of an EDMD model. In this
        method, the time series in `X` are again subdivided into smaller time series of
        equal length (the windows). Each window contains the initial condition and the
        samples to score the model against. This therefore corresponds to a more
        systematic approach to analyze the error over a prediciton horizon.

        Parameters
        ----------
        X
            Time series to reconstruct in windowed fashion.
        y
            ignored

        sample_weight
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the internal ``score`` method.

        edmd
            The EDMD model to apply the score function to.

        Returns
        -------
        float
            score

        """

        # does all the checks:
        X_reconstruct, X = edmd.reconstruct(X=X, y=y, qois=qois, return_X_windows=True)

        if qois is None:
            X_reconstruct = X_reconstruct.loc[:, X.columns]
            X = X.loc[X_reconstruct.index, :]
        else:
            X = X.loc[X_reconstruct.index, qois]

        return edmd._score_eval(X, X_reconstruct, sample_weight)

    def adapt_model(self, estimator: EDMD):
        """Adapts the EDMD model.

        Attaches the attribute `window_size` to the EDMD model and overwrites the
        `score` and `reconstruct` methods.

        Parameters
        ----------
        estimator
            The model to adapt. The model must be already fit.

        Returns
        -------
        EDMD
            The adapted model.

        """
        estimator.window_size = self.window_size

        # overwrite the two methods with new "windowed" methods
        # ignored types for mypy
        estimator.reconstruct = partial(  # type: ignore
            self._window_reconstruct, edmd=estimator, offset=self.offset
        )
        estimator.score = partial(self._window_score, edmd=estimator)  # type: ignore
        return estimator


@warn_experimental_class
class EDMDPostObservable(object):  # pragma: no cover
    """# TODO

    # TODO: Alternative? EDMDCVErrorObservable?
    # TODO: Testing & Docu
    # TODO: compute mean of error time series if offset < blocksize?

    Parameters
    ----------
    estimator
        # TODO

    cv
        - the best estimator is the final
        - the observable is only evaluated on the validation set
        - # TODO:

    observables
        factor https://en.wikipedia.org/wiki/Half-normal_distribution

    time_horizon
        - assumed to be unbiased

    offset
        - by how much to move the prediction window

    n_jobs
        - the CV splits can be computed in parallel

    pre-dispatch
        - check docu in EDMDCV
    """

    _valid_observables = ["std", "abserr"]

    def __init__(
        self,
        estimator,
        cv: Optional[TSCCrossValidationSplit] = None,
        observable="abserr",  # std or abserr
        # --> Change default to std (but change parameter in scripts first)
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        pre_dispatch: str = "2*n_jobs",
    ):
        self.estimator = estimator
        self.cv = cv
        self.observable = observable
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch

    def _adapt_edmd_model(self, edmd, X_validate, abserr_timeseries):

        # 1. get original data
        X_dict = edmd.transform(X_validate)

        if self.observable == "std":
            target_timeseries = abserr_timeseries * np.sqrt(np.pi / 2)
        else:
            target_timeseries = abserr_timeseries

        # 3. compute inverse map
        inverse_map2error_values = edmd._least_squares_inverse_map(
            X=target_timeseries, X_dict=X_dict.loc[abserr_timeseries.index, :]
        )

        # 4. compute Koopman modes for error observables
        modes_error_values = edmd._compute_koopman_modes(inverse_map2error_values)

        # 5. attach Koopman modes to existing
        edmd._koopman_modes = np.row_stack([edmd._koopman_modes, modes_error_values])

        # 6. change feature_names_out_pred_ by attaching to existing
        edmd._feature_names_pred = pd.Index(
            np.append(
                edmd.feature_names_pred_,
                [f"{self.observable}_{col}" for col in edmd.feature_names_pred_],
            ),
            name=TSCDataFrame.tsc_feature_col_name,
        )

        return edmd

    def _compute_err_timeseries(self, edmd, X_test):
        """

        Parameters
        ----------
        edmd
        X_test
        qois

        Returns
        -------

        """
        # TODO: here is a distinction that maybe is better to solve via a new parameter
        #  in reconstruct (e.g. return_X to return the samples in X that are actually
        #  reconstructed)

        try:
            X_reconstruct, X_test = edmd.reconstruct(
                X_test, qois=None, return_X_windows=True
            )
        except TypeError:
            X_reconstruct = edmd.reconstruct(X_test)

        # remove initial states, because they are often almost exact
        X_reconstruct = X_reconstruct.drop(labels=X_reconstruct.head(1).index)

        err_timeseries = (X_test.loc[X_reconstruct.index, :] - X_reconstruct).abs()

        return X_test, err_timeseries

    def _fit_and_create_error_timeseries(
        self, edmd: EDMD, X: TSCDataFrame, y, split_nr, train, test, fit_params, verbose
    ):

        if verbose:
            msg = f"split: {split_nr}"
            print("[CV] {} {}".format(msg, (64 - len(msg)) * "."), flush=True)

        X_train, X_test = _split_X_edmd(X, y, train_indices=train, test_indices=test)

        edmd = edmd.fit(X=X_train, y=y, **fit_params)

        test_score = edmd.score(X_test, y=None)

        if verbose:
            print(f"test_score = {test_score}")

        X_test, err_timeseries = self._compute_err_timeseries(edmd, X_test)

        return {
            "test_score": test_score,
            "edmd": edmd,
            "X_test": X_test,
            "err_timeseries": err_timeseries,
        }

    def _validate(self):
        if not isinstance(self.estimator, EDMD):
            raise TypeError("estimator must be of type EDMD")

        try:
            check_is_fitted(self.estimator)
            # NOTE: can be implemented to deal with an already-fitted EDMD
            raise ValueError("EDMD estimator must not be fitted already")
        except NotFittedError:
            pass

        if self.observable not in self._valid_observables:
            raise ValueError(
                f"observable={self.observable} is invalid. "
                f"Choose from {self._valid_observables}"
            )

    def _run_cv(self, X, y, **fit_params):
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        # n_splits = self.cv.get_n_splits(X, y, groups=None)

        parallel = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch
        )

        results: Dict[str, Any] = {}
        X_validate: Optional[TSCDataFrame] = None
        err_timeseries: Optional[TSCDataFrame] = None
        best_estimator: Optional[EDMD] = None
        with parallel:

            def evaluate_cv_splits():
                nonlocal X_validate
                nonlocal err_timeseries
                nonlocal best_estimator

                ret = parallel(
                    delayed(self._fit_and_create_error_timeseries)(
                        edmd=deepcopy(self.estimator),
                        X=X,
                        y=y,
                        split_nr=i,
                        train=train,
                        test=test,
                        fit_params=fit_params,
                        verbose=self.verbose,
                    )
                    for i, (train, test) in enumerate(cv.split(X, y, groups=None))
                )

                if len(ret) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )

                scores: np.ndarray = np.asarray([r["test_score"] for r in ret])
                best_estimator_idx = np.argmax(scores)
                best_estimator = ret[best_estimator_idx]["edmd"]

                X_validate = TSCDataFrame.from_frame_list([r["X_test"] for r in ret])
                err_timeseries = TSCDataFrame.from_frame_list(
                    [r["err_timeseries"] for r in ret]
                )

                return ret, X_validate, err_timeseries

            evaluate_cv_splits()

        self.cv_results_ = results

        return X_validate, err_timeseries, best_estimator

    def fit_transform(self, X: TSCDataFrame, y=None, **fit_params):
        """Computes the post observables and alters the EDMD estimator.

        Parameters
        ----------

        X
            Training time series data.

        y: None
            ignored

        **fit_params : Dict[str, object]
            Parameters passed to the ``fit`` method of the estimator.
        """
        self._validate()

        # scorers, refit_metric = self._check_multiscore()

        X, y = indexable(X, y)
        fit_params = _check_fit_params(X, fit_params)

        if self.cv is None:
            # Need to use deepcopy here, bc. sklearn.clone does not clone methods but
            # re-initializes the object
            self.final_estimator_ = deepcopy(self.estimator)
            self.final_estimator_ = self.final_estimator_.fit(X, y, **fit_params)
            X_validate, abserr_timeseries = self._compute_err_timeseries(
                self.final_estimator_, X_test=X
            )
        else:
            X_validate, abserr_timeseries, self.final_estimator_ = self._run_cv(
                X, y, **fit_params
            )

        self.final_estimator_ = self._adapt_edmd_model(
            edmd=self.final_estimator_,
            X_validate=X_validate,
            abserr_timeseries=abserr_timeseries,
        )

        return self.final_estimator_


# class EDMDControl(
#     TSCTransformerMixin,
#     TSCPredictMixin,
# ):
#     r"""Adapt :class:`EDMD` to controlled systems.
#
#     This class provides a wrapper around :class:`EDMD` which allows usage of
#     the main functionality (transform, fit, predict). It approximates Koopman
#     operator for a controlled system from time series collection data
#     (:py:class:`.TSCDataFrame`), using a finite function basis to represent the
#     state. The underlying DMD model is restricted to :class:`DMDControl`.
#
#     ...
#
#     Parameters
#     ----------
#     dict_steps : List[Tuple[str, object]]
#         List with `(string_identifier, model)` of models to transform the data. The
#         list defines the transformation pipeline and order of execution. All models in
#         the list must be able to accept :class:`.TSCDataFrame` as input in `fit` and
#         output in `transform`.
#
#     dmd_model: Union[DMDControl, gDMDAffine]
#         The class to use for evaluating the Dynamic Mode Decomposition given the lifting.
#         :py:class:`DMDControl` is flowmap Koopman operator representation of
#         :math:`x^+ = Ax + Bu`.
#         :py:class:`gDMDAffine` is differential Koopman generator representation of
#         :math:\dot{x} = Ax + Bux`.
#
#     include_id_state
#         If True, the original time series data are added to the EDMD-dictionary. The
#         mapping from the EDMD-dictionary states back to the full-state is then only a
#         projection and the cost of an increased EDMD-dictionary dimension.
#
#     **kwargs
#         passed to :py:class:`EDMD`
#
#
#     Attributes
#     ----------
#     sys_matrix : np.ndarray
#         Koopman approximation of the lifted state matrix
#
#     control_matrix : np.ndarray
#         Koopman approximation of the lifted control matrix
#
#     See Also
#     --------
#
#     :py:class:`EDMD`
#     :py:class:`DMDControl`
#
#     """
#
#     def __init__(
#         self,
#         dict_steps: List[Tuple[str, object]],
#         dmd_model: Optional[Union[DMDControl, gDMDAffine]] = None,
#         include_id_state: bool = True,
#         **kwargs,
#     ):
#         self.dict_steps = dict_steps
#         self._dmd_model = dmd_model if dmd_model is not None else DMDControl()
#         self.include_id_state = include_id_state
#         self._edmd = EDMD(
#             self.dict_steps,
#             include_id_state=include_id_state,
#             use_transform_inverse=False,
#             **kwargs,
#         )
#
#     def transform(self, X: TSCDataFrame) -> TSCDataFrame:
#         """Perform dictionary transformations on time series.
#
#         Parameters
#         ----------
#         X : TSCDataFrame, pandas.DataFrame
#            Time series to transform. Must fulfill the input requirements of
#            first step of the pipeline.
#
#         Returns
#         -------
#         TSCDataFrame, pandas.DataFrame
#             The transformed time series.
#         """
#         return self._edmd.transform(X)
#
#     def inverse_transform(self, X: TransformType) -> TransformType:
#         """Perform inverse dictionary transformations on dictionary time series.
#
#         Parameters
#         ----------
#         X: TSCDataFrame, pandas.DataFrame
#             Time series to map back to the full-state time series.
#
#         Returns
#         -------
#         TSCDataFrame
#             full-state time series
#         """
#         return self._edmd.inverse_transform(X)
#
#     @property
#     def feature_names_in_(self):  # formerly state_columns
#         return self._edmd.feature_names_in_
#
#     def fit(
#         self,
#         X: TimePredictType,
#         U: Optional[TimePredictType] = None,
#         y=None,
#         **fit_params,
#     ) -> "EDMDControl":
#         """Compute Koopman approximation of lifted state and control matrices
#
#         Parameters
#         ----------
#         X : TSCDataFrame
#             State input data
#
#         U : TSCDataFrame
#             Control input data
#
#         **fit_params: Dict[str, object]
#             Parameters passed to the ``fit`` method of each step, where
#             each parameter name is prefixed such that parameter ``p`` for step
#             ``s`` has key ``s__p``. To add parameters for the set DMD model use
#             ``s=dmd``, e.g. ``dmd__param``.
#
#         Returns
#         -------
#         self
#         """
#         self._validate_datafold_data(
#             X,
#             ensure_tsc=True,
#             tsc_kwargs={"ensure_const_delta_time": True},
#         )
#         fit_params = self._edmd._check_fit_params(**fit_params or {})
#         dmd_fit_params = fit_params.pop("dmd", {})
#         _ = fit_params.pop("edmd", None)
#         Xlift = self._edmd._fit(X, **fit_params)
#         self.n_samples_ic_ = self._edmd._compute_n_samples_ic(X, Xlift)
#
#         if self._edmd.include_id_state:
#             Xlift = self._edmd._attach_id_state(X=X, X_dict=Xlift)
#         self._edmd._inverse_map = self._edmd._compute_inverse_map(X=X, X_dict=Xlift)
#
#         Ulift = TSCDataFrame(index=Xlift.index)
#         if not (U is None):
#             cols = U.columns if isinstance(U, pd.DataFrame) else range(U.shape[1])
#             Ulift[cols] = U  # do not apply dictionary, only trim for delay transforms
#         self._dmd_model.fit(Xlift, Ulift, **dmd_fit_params)
#         self.sys_matrix = self._dmd_model.sys_matrix_
#         self.control_matrix = self._dmd_model.control_matrix_
#         return self
#
#     def predict(
#         self,
#         X: InitialConditionType,
#         time_values: Optional[np.ndarray] = None,
#         U: Optional[Union[TSCDataFrame, np.ndarray]] = None,
#         check_inputs: bool = True,
#         lifted_state: bool = False,
#         **predict_params,
#     ) -> TSCDataFrame:
#         """Predict time series data for each initial condition at
#         specified time values given the specified control input.
#
#         Parameters
#         ----------
#         X : InitialConditionType
#             Single initial condition of shape `(n_features,)` or multiple initial
#             conditions of shape `(n_features, n_initial_conditions)`.
#
#         time_values : np.ndarray
#             Time series at which to evaluate the system. Must be equally spaced
#             and use the same timestep as the training data. If U is
#             a TSCDataFrame time_values can be skipped and inferred from the index
#
#         U : np.ndarray | TSCDataFrame
#             The control input at the provided time values with shape
#             `(n_timesteps, n_control_dimensions)`
#
#         check_inputs : bool, optional, default True
#             Allows skipping input checks and assignments to improve performance.
#
#             .. warning::
#                 Use with caution - May result in silent errors.
#
#         lifted_state : bool, optional, default False
#             If true, the output includes the predictions for the lifting dimension
#
#         Returns
#         -------
#         TSCDataFrame
#         """
#         check_is_fitted(self, ["sys_matrix", "control_matrix"])
#         if isinstance(X, np.ndarray):
#             X = if1dim_rowvec(X)
#             X = InitialCondition.from_array(
#                 X, columns=self._dmd_model.feature_names_in_
#             )
#
#         X0lift = self._edmd.transform(X)
#         Xlift_tsc = self._dmd_model.predict(
#             X0lift,
#             time_values=time_values,
#             U=U,
#             check_inputs=check_inputs,
#             **predict_params,
#         )
#         if lifted_state:
#             return Xlift_tsc
#         X_tsc = self.inverse_transform(Xlift_tsc)
#         return X_tsc
#
#     def reconstruct(
#         self, X: TSCDataFrame, qois=None, U: Optional[TSCDataFrame] = None, *kwargs
#     ) -> TSCDataFrame:
#         """Reconstruct existing time series collection.
#
#         Internal steps to reconstruct a time series collection:
#
#         1. Extract the initial conditions from each time series in the collection.
#         2. Predict the remaining states of each time series with the built EDMD model
#            at the same time values.
#
#         Parameters
#         ----------
#         X
#             The time series states to reconstruct.
#
#         U
#             The time series control input to reconstruct.
#
#         qois
#             ignored
#
#         Returns
#         -------
#         TSCDataFrame
#             Reconstructed time series collection.
#         """
#         check_is_fitted(self, ["sys_matrix", "control_matrix"])
#         self._validate_datafold_data(
#             X,
#             ensure_tsc=True,
#             tsc_kwargs={"ensure_const_delta_time": True},
#         )
#         X_reconstruct = []
#         for X_ic, time_values in InitialCondition.iter_reconstruct_ic(
#             X, n_samples_ic=self.n_samples_ic_
#         ):
#             X_est_ts = self.predict(
#                 X_ic,
#                 time_values=time_values,
#                 U=U
#                 if U is None
#                 else U[X.index.get_level_values("time") >= time_values[0]],
#             )
#             X_reconstruct.append(X_est_ts)
#
#         X_reconstruct = pd.concat(X_reconstruct, axis=0)
#         assert isinstance(X_reconstruct, TSCDataFrame)
#
#         return X_reconstruct
#
#     def fit_predict(
#         self,
#         X: TSCDataFrame,
#         y=None,
#         U: Optional[TSCDataFrame] = None,
#         **fit_params,
#     ):
#         """Fit the model and reconstruct the training data.
#
#         Parameters
#         ----------
#         X
#             Training time series state data. Must fulfill input requirements of first
#             `dict_step` in the EDMD-dictionary pipeline.
#
#         U
#             Time series control input data for both training and prediction. Must fulfill
#             input requirements of first `dict_step` in the EDMD-dictionary pipeline.
#
#         **fit_params: Dict[str, object]
#             Parameters passed to the ``fit`` method of each step, where
#             each parameter name is prefixed such that parameter ``p`` for step
#             ``s`` has key ``s__p``.
#
#         Returns
#         -------
#         TSCDataFrame
#             Reconstructed time series collection.
#         """
#         return self.fit(X=X, U=U, y=y, **fit_params).reconstruct(X=X, U=U)
#
#     def fit_transform(
#         self, X: TSCDataFrame, y=None, U: Optional[TSCDataFrame] = None, **fit_params
#     ):
#         """Fit the model and return the EDMD-dictionary time series.
#
#         Parameters
#         ----------
#         X
#             The time series states to reconstruct.
#         U
#             The time series control input to reconstruct.
#
#         y: None
#             ignored
#
#         **fit_params: Dict[str, object]
#             Parameters passed to the ``fit`` method of each step, where
#             each parameter name is prefixed such that parameter ``p`` for step
#             ``s`` has key ``s__p``.
#
#         Returns
#         -------
#         TSCDataFrame
#              EDMD-dictionary time series data.
#         """
#         return self.fit(X=X, U=U, y=y, **fit_params).transform(X)
