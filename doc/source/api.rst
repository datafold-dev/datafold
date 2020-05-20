Software documentation
======================

*datafold's* software architecture consists of three package layers

* :py:class:`datafold.appfold` (high) - for meta-models
* :py:class:`datafold.dynfold` (medium) - for manifold learning, data transformations and
  time series decompositions
* :py:class:`datafold.pcfold` (low) - for data structures and associated objects and
  algorithms

The three layers encapsulate a workflow hierarchy to maintain a high degree of modularity
in *datafold*. Each model is intended to be used on their own or internally in other
model implementations. Dependencies between the layers are
unidirectional, where models can depend on the functionality of lower levels and in some
cases at the same layer.

Each model has a clear scope, which is indicated by the associated base classes. A base
class is either directly from scikit-learn or *datafold* provides own
specifications that align to the scikit-learn API in a duck-typing fashion

* :class:`sklearn.BaseEstimator`
   All models inherit from this base class
* :class:`.TSCTransformerMixIn`
  The mixin for transformer is aligned to `scikit-learn`'s
  `TransformerMixIn <https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html>`_
  but also allows passing time series data as input.
* :class:`.TSCPredictMixin`
  The mixin is intended for models that fit time series data and are capable to
  model dynamical systems (cf. system identification). For model evaluation, a fitted
  model requires an initial condition and time specifications.
* :class:`sklearn.RegressorMixin` and `sklearn.MultiOutputMixin`
  The scikit-learn mixins are used in *datafold* models that interpolate or regress
  function values on manifold point cloud.

.. automodapi:: datafold.appfold
.. automodapi:: datafold.dynfold
.. disable inherited members for pcfold because TSCDataFrame and PCManifold inherit
   from very rich data structures that have *many* functions and attributes.
.. automodapi:: datafold.pcfold
    :no-inherited-members: