Software documentation
======================

**datafold** 's software architecture consists of three package layers

* high level for meta models: :py:class:`datafold.appfold`
* medium level for models for manifold learning and data trasnformation
  :py:class:`datafold.dynfold`
* low level for data structures and associated algorithms :py:class:`datafold.pcfold`

The three layers encapsulate a workflow hierarchy intending to maintain a high degree of
modularity in *datafold* that allows ongoing contribution in the active research of
data-driven models with an explicit manifold context. The models or data structures on
each layer can, therefore, also be used on their own and be part of new model
implementations. Dependencies between the layers are unidirectional, where models can
depend on the functionality of lower levels and in some cases also of the same layer.

Each package layer contains model classes with clear scope, which is indicated by its
base class(es).  All *datafold* models integrate with base classes from scikit-learn's
API or in the case of time series data generalize the API in a conformant (duck typing)
fashion.

* :class:`sklean.BaseEstimator`
  The application interface was templated from the scikit-learn
  :cite:`pedregosa_scikit-learn_2011`: project.
* :class:`TSCTransformerMixIn`
  MixIn that is aligned to `scikit-learn`'s
  `TransformerMixIn <https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html>`_
  but generalizes the input and output to time series data.
* :class:`TSCPredictMixin`
  MixIn that structures models for system identification. The model is fitted with time
  series data, requires initial conditions and time specifications for evaluation.
* :class:`sklean.RegressorMixin` and `sklean.MultiOutputMixin`
  MixIns for models that interpolate or regress function values on point cloud
  manifolds.

.. automodapi:: datafold.appfold
.. automodapi:: datafold.dynfold
.. disable inherited members for pcfold because TSCDataFrame and PCManifold inherit
   from very rich data structures that have *many* functions and attributes.
.. automodapi:: datafold.pcfold
    :no-inherited-members: