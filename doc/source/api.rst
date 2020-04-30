Software documentation
======================

**datafold** 's software architecture consists of three package layers (from high level to
low level: :module:`datafold.appfold`, :module:`datafold.dynfold` and
:module:`datafold.pcfold`). The three layers encapsualte a workflow hierarchy, from low
level data structures and algorithms, data transformations on the medium layer and high
level meta-models to solve complex machine learning tasks. The datafold software
architecture aims to ensure a high degree of modularity to allow ongoing contribution in
the active research of manifold aware data-driven models.

Each of the layer contains model sclasses with clear isolated scope. The base class(es)
of each model class indicate it's purpose and scope. The application interface was
taken by example from the scikit-learn :cite:`pedregosa_scikit-learn_2011`: All
data-driven models integrate with base classes from scikit-learn's API or in the case
of time series data generalize the API in a conformant (duck typing) fashion. The
models and data structures on each layer can, therefore, also be used on their own.
Dependencies between the layers are unidirectional, where models can require
functionality of lower levels and only in some cases of the same layer.

datafold integrates with other projects of the
`Python's scientific computing stack <https://scipy.org/about.html>`_.
This makes it easy to reuse well tested and widely used algorithms and data structures
and provides a familiar handling to new datafold users that are already familiar with
Python's scientific computing stack.

.. automodapi:: datafold.appfold
.. automodapi:: datafold.dynfold
.. disable inherited members for pcfold because TSCDataFrame and PCManifold inherit
   from very rich data structures that have *many* functions and attributes.
.. automodapi:: datafold.pcfold
    :no-inherited-members: