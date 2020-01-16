#!/usr/bin/env python3

from typing import Union

import numpy as np
import pandas as pd

from datafold.pcfold.timeseries import TSCDataFrame


class TimeSeriesTransformMixIn:
    from datafold.pcfold.timeseries import TSCDataFrame

    def fit_transform(self, X_ts: TSCDataFrame, **fit_params):
        raise NotImplementedError

    def transform(self, X_ts: TSCDataFrame):
        raise NotImplementedError

    def inverse_transform(self, X_ts: TSCDataFrame):
        raise NotImplementedError


class TSCQoiTransform(TimeSeriesTransformMixIn):

    VALID_NAMES = ["min-max", "standard"]

    def __init__(self, cls, **kwargs):
        """
        See
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
        for scaling classes
        """

        if not hasattr(cls, "transform") or not hasattr(cls, "inverse_transform"):
            raise AttributeError(
                f"transform cls {cls} must provide a 'transform' "
                f"and 'inverse_transform' attribute"
            )
        self.transform_cls_ = cls(**kwargs)

    @classmethod
    def from_name(cls, name):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        if name == "min-max":
            return cls(cls=MinMaxScaler, feature_range=(0, 1))
        elif name == "mean":
            return cls(cls=StandardScaler, with_mean=True, with_std=False)
        elif name == "standard":
            return cls(cls=StandardScaler, with_mean=True, with_std=True)
        else:
            raise ValueError(f"name={name} is not known. Choose from {cls.VALID_NAMES}")

    def fit_transform(self, X_ts: TSCDataFrame, **fit_params):
        data = self.transform_cls_.fit_transform(X_ts.to_numpy())
        return TSCDataFrame.from_same_indices_as(X_ts, data)

    def transform(self, X_ts: TSCDataFrame):
        data = self.transform_cls_.transform(X_ts.to_numpy())
        return TSCDataFrame.from_same_indices_as(X_ts, data)

    def inverse_transform(self, X_ts: TSCDataFrame):
        data = self.transform_cls_.inverse_transform(X_ts.to_numpy())
        return TSCDataFrame.from_same_indices_as(X_ts, data)


@DeprecationWarning
class NormalizeQoi(object):
    VALID_STRATEGIES = ["id", "min-max", "mean", "standard"]

    def __init__(self, normalize_strategy: Union[str, dict], undo: bool):

        if undo and not isinstance(normalize_strategy, dict):
            raise ValueError(
                "If revert=True, the parameter normalize_strategy has to be a dict, "
                "containing the corresponding normalization factors"
            )

        # if the normalize_strategy is a str, then compute based on the handled
        # TSCDataFrame else if a dict (as returned from normalize()) it uses the
        # information there
        self.compute_norm_factor = isinstance(normalize_strategy, str)

        normalize_strategy = self.check_normalize_qoi_strategy(normalize_strategy)

        if isinstance(normalize_strategy, str):
            # make a new dict with consisting information
            self.normalize_strategy = {"strategy": normalize_strategy}
        else:
            self.normalize_strategy = normalize_strategy

        self.undo = undo

    @staticmethod
    def check_normalize_qoi_strategy(strategy: Union[str, dict]):

        if isinstance(strategy, str) and strategy not in NormalizeQoi.VALID_STRATEGIES:
            raise ValueError(
                f"strategy={strategy} not valid. Choose from "
                f" {NormalizeQoi.VALID_STRATEGIES}"
            )

        elif isinstance(strategy, dict):
            if "strategy" not in strategy.keys():
                raise ValueError("Not a valid dict to describe normalization strategy.")

        return strategy

    def _id(self, df):
        return df

    def _min_max(self, df):

        if self.compute_norm_factor:
            self.normalize_strategy["min"], self.normalize_strategy["max"] = (
                df.min(),
                df.max(),
            )

        if not self.undo:

            return (df - self.normalize_strategy["min"]) / (
                self.normalize_strategy["max"] - self.normalize_strategy["min"]
            )
        else:
            self._check_columns_present(df.columns, ["min", "max"])

            return (
                df * (self.normalize_strategy["max"] - self.normalize_strategy["min"])
                + self.normalize_strategy["min"]
            )

    def _mean(self, df):

        if self.compute_norm_factor:
            (
                self.normalize_strategy["mean"],
                self.normalize_strategy["min"],
                self.normalize_strategy["max"],
            ) = (
                df.mean(),
                df.min(),
                df.max(),
            )

        if not self.undo:
            return (df - self.normalize_strategy["mean"]) / (
                self.normalize_strategy["max"] - self.normalize_strategy["min"]
            )
        else:
            self._check_columns_present(df.columns, ["min", "max", "mean"])
            return (
                df * (self.normalize_strategy["max"] - self.normalize_strategy["min"])
                + self.normalize_strategy["mean"]
            )

    def _standard(self, df):

        if self.compute_norm_factor:
            self.normalize_strategy["mean"], self.normalize_strategy["std"] = (
                df.mean(),
                df.std(),
            )

        if not self.undo:
            df = df - self.normalize_strategy["mean"]
            return df / self.normalize_strategy["std"]
        else:
            self._check_columns_present(df.columns, ["std", "mean"])
            return df * self.normalize_strategy["std"] + self.normalize_strategy["mean"]

    def _check_columns_present(self, df_columns, strategy_fields):
        for field in strategy_fields:
            element_columns = self.normalize_strategy[field].index
            if not np.isin(df_columns, element_columns).all():
                raise ValueError("TODO")

    def transform(self, df: pd.DataFrame):

        strategy = self.normalize_strategy["strategy"]

        if strategy == "id":
            norm_handle = self._id
        elif strategy == "min-max":
            norm_handle = self._min_max
        elif strategy == "mean":
            norm_handle = self._mean
        elif strategy == "standard":
            norm_handle = self._standard
        else:
            raise ValueError(f"strategy={self.normalize_strategy} not known")

        return norm_handle(df), self.normalize_strategy
