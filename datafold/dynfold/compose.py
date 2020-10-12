#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils.validation import check_is_fitted

from datafold.dynfold.base import TransformType, TSCTransformerMixin


class TSCColumnTransformer(TSCTransformerMixin, ColumnTransformer):
    def _hstack(self, Xs):
        if self.sparse_output_:
            raise NotImplementedError(
                "Currently there is only dense output for " "TSCColumnsTransformer"
            )
        # dropna(axis=0) removes all rows that were dropped during transform
        # (i.e. transformations that require multiple timesteps).
        return pd.concat(Xs, axis=1).dropna(axis=0)

    # @property  # NOTE this one is handled by the super class
    # def n_features_in_(self):
    #     return self.transformers_[0][1].n_features_in_

    @property
    def feature_names_in_(self):
        check_is_fitted(self, "named_transformers_")
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

    def fit(self, X: TransformType, y=None, **fit_params):
        X = self._validate_datafold_data(X)
        self._read_fit_params(attrs=None, fit_params=fit_params)
        return super(TSCColumnTransformer, self).fit(X)


@NotImplementedError
class TSCFeatureUnion(TSCTransformerMixin, FeatureUnion):
    def n_features_in_(self):
        pass

    @property
    def n_features_out_(self):
        # TODO: this is wrong, need to collect all feature names...
        return self.transformer_list[0][1].n_features_out_

    @property
    def feature_names_in_(self):
        return self.transformer_list[0][1].feature_names_in_

    @property
    def feature_names_out_(self):
        # TODO: this is wrong, need to collect all feature names...
        return self.transformer_list[0][1].feature_names_out_

    def fit(self, X: TransformType, y=None, **fit_params):
        X = self._validate_datafold_data(X)
        self._read_fit_params(attrs=None, fit_params=fit_params)

        super(TSCFeatureUnion, self).fit(X, y=y)

        self._setup_features_fit(X, features_out=[])

        return self

    def transform(self, X: TransformType):
        pass

    def fit_transform(self, X: TransformType, y=None, **fit_params) -> TransformType:
        pass
