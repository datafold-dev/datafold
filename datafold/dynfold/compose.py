#!/usr/bin/env python3


import pandas as pd
from sklearn import compose, pipeline
from sklearn.utils.validation import check_is_fitted

from datafold.dynfold.base import TransformType, TSCTransformerMixin


class TSCPipeline(pipeline.Pipeline, TSCTransformerMixin):  # pragma: no cover
    # @property # Note: this property is handled in the super class
    # def feature_names_in_(self):
    #    return self.steps[0][1].feature_names_in_

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
    `ColumnTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>`__.

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
        transformers: list[tuple],
        *,
        remainder="drop",
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=True
    ):
        super().__init__(
            transformers=transformers,
            remainder=remainder,
            sparse_threshold=0.3,  # default value, parameter ignored in TSCColumnTransformer
            transformer_weights=transformer_weights,
            n_jobs=n_jobs,
            verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out,
        )

    @property
    def n_features_out_(self):
        check_is_fitted(self, "transformers_")
        return sum(tr.n_features_out_ for _, tr, _ in self.transformers_)

    def _hstack(self, Xs):
        if self.sparse_output_:
            raise NotImplementedError(
                "Currently there is no support for sparse output in TSCColumnTransformer."
            )

        # dropna(axis=0) removes all rows that were dropped during transform
        # (i.e. transformations that require multiple timesteps).
        Xs = pd.concat(Xs, axis=1, sort=True, ignore_index=True).dropna(axis=0)
        Xs.columns = pd.Index(self.get_feature_names_out())
        return Xs

    def fit(self, X: TransformType, y=None, **fit_params):
        X = self._validate_datafold_data(X)
        self._read_fit_params(attrs=None, fit_params=fit_params)
        return super().fit(X)

    def partial_fit(self, X, y=None, **fit_params):
        if not hasattr(self, "transformers_"):
            return self.fit(X, y, **fit_params)
        else:
            return self
