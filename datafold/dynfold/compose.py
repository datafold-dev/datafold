#!/usr/bin/env python3

from typing import List, Tuple

import numpy as np
import pandas as pd
import sklearn.compose as compose
import sklearn.pipeline as pipeline
from sklearn.utils.validation import check_is_fitted

from datafold.dynfold.base import TransformType, TSCTransformerMixin


class TSCPipeline(pipeline.Pipeline, TSCTransformerMixin):  # pragma: no cover
    @property
    def feature_names_in_(self):
        return self.steps[0][1].feature_names_in_

    @property
    def n_features_out_(self):
        return self.steps[-1][1].n_features_out_

    @property
    def feature_names_out_(self):
        return self.steps[-1][1].feature_names_out_


class TSCColumnTransformer(compose.ColumnTransformer, TSCTransformerMixin):
    """A column transformer for time series data.

    This class is a wrapper class of scikit-learn's ColumnTransformer and adopted for
    :py:class:`TSCDataFrame` in the pipeline.

    For the undocumented attributes please go to the base class documentation
    `ColumnTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>`__

    .. note::
        The parameter ``sparse_threshold`` of the super class is not supported.

    Parameters
    ----------
    transformers
        All transformers included in the list must be able to process
        :py:class:`TSCDataFrame`. See base class for the detailed specification of
        the tuple.
    """

    def __init__(
        self,
        transformers: List[Tuple],
        *,
        remainder="drop",
        n_jobs=None,
        transformer_weights=None,
        verbose=False
    ):
        super(TSCColumnTransformer, self).__init__(
            transformers=transformers,
            remainder=remainder,
            sparse_threshold=0.3,  # default value, will be ignored
            transformer_weights=transformer_weights,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    # @property
    # def n_features_in_(self):
    #     # this is only to make it explicit that this is already handled by the super class
    #     return super(TSCColumnTransformer, self).n_features_in_

    @property
    def feature_names_in_(self):
        check_is_fitted(self, "transformers_")
        return self.transformers_[0][1].feature_names_in_

    @property
    def n_features_out_(self):
        check_is_fitted(self, "transformers_")
        return sum([tr.n_features_out_ for _, tr, _ in self.transformers_])

    @property
    def feature_names_out_(self):
        check_is_fitted(self, "transformers_")
        indices = [tr.feature_names_out_ for _, tr, _ in self.transformers_]

        for i in range(1, len(indices)):
            indices[0] = indices[0].append(indices[i])
            indices[i] = None

        return indices[0]

    def _hstack(self, Xs):
        if self.sparse_output_:
            raise NotImplementedError(
                "Currently there is no support for sparse output in TSCColumnsTransformer"
            )

        all_columns = pd.Index(np.hstack([df.columns.to_numpy() for df in Xs]))

        if all_columns.has_duplicates:
            # handle feature names in case of conflict
            for i in range(len(self.transformers_)):
                Xs[i] = Xs[i].add_prefix(self.transformers_[i][0] + "__")

        # dropna(axis=0) removes all rows that were dropped during transform
        # (i.e. transformations that require multiple timesteps).
        return pd.concat(Xs, axis=1).dropna(axis=0)

    def fit(self, X: TransformType, y=None, **fit_params):
        X = self._validate_datafold_data(X)
        self._read_fit_params(attrs=None, fit_params=fit_params)
        return super(TSCColumnTransformer, self).fit(X)
