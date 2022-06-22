"""The highest level of *datafold* accommodates models that capture complex processing
pipelines. Essentially, the models are "meta-models" because they provide a single point
of access to multiple sub-models captured in the class. The models at this layer are
at the end of the machine learning process and are intended to solve complex
data-driven use-cases or analysis tasks.

The modularization in *datafold*'s software of the first and
the second layer becomes profitable for the meta-models: the data process pipeline can be
combined with greater flexibility, which makes it easier to test model
configurations and model accuracies in a parameter space.

**Relevant base classes from scikit-learn:**

* :class:`sklearn.pipeline.Pipeline`
  The pipeline (meta-estimator) is a base class of :py:class:`.EDMD` (Extended Dynamic
  Mode Decomposition), along the mixins :py:class:`TSCPredictMixin` (time series
  predictions) and :py:class:`TSCTransformMixin` (for the EDMD dictionary).

* :class:`sklearn.model_selection.GridSearchCV`
  Building upon the meta-model approach, an :class:`EDMD` instance can be integrated
  into a cross-validation pipeline :class:`EDMDCV`
  with base class :class:`sklearn.model_selection.GridSearchCV`. The cross-validation
  provides an exhaustive search over a user-specified parameter space that can include
  parameters from all sub-models and includes data-splitting schemes of time series data.
"""

from datafold.appfold.edmd import EDMD, EDMDCV
