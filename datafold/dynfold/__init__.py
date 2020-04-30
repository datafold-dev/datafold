"""
The second package layer in datafold consists of data-driven models to deal with data
manifold structure. The naming scheme indicates it's purpose of extracting dynamics
from time series data.

There are three type of models on this layer.

1. Models subclassing :class:`TSCTransformMixIn`:

    * Computing a coordinate basis for (time series) data manifold. These models can be
      used on their own for (static) point cloud data or
      time series data. An important model is the :class:`DiffusionMaps` for which datafold
      provides
      an implementation that through the internal use of :class:`PCManifold` can scale to
      larger datasets with sparse kernel matrices. The model can be used for manifold
      learning to reduce the dimension in the dataset or it can approximate a functional
      basis of the Laplace-Beltrami operator (and others).
    * Models to transform time series collections data specifically (indicated with the
      prefix `TSC` in the classname). This includes time delay embedding
      :class:`TSCTakensEmbedding` but also scikit-learn wrappers like
      :class:`TSCPrincipalComponent` which generalize the input and output to time series
      data.

2. Models subclassing :class:`sklearn.base.RegressorMixin`:
    * The methods can interpolate general function values on manifolds, this includes
      the important case of the non-linear mappings (image and pre-image) for manifold
      learning models. An example is the :class:`LaplacianPyramidsInterpolator`.

2. Models subclassing :class:`TSCPredictMixIn`:

    * DMD based models that compute a Koopman matrix based on the time series input (
      require :class:`TSCDataFrame`. These models define a dynamical system and are
      capable to carry out future predictions (:class:`DMDBase`).
"""

from datafold.dynfold.dmap import DiffusionMaps, LocalRegressionSelection
from datafold.dynfold.dmd import DMDBase, DMDEco, DMDFull
from datafold.dynfold.outofsample import (
    GeometricHarmonicsInterpolator,
    LaplacianPyramidsInterpolator,
)
from datafold.dynfold.transform import (
    TSCFeaturePreprocess,
    TSCIdentity,
    TSCPrincipalComponent,
    TSCRadialBasis,
    TSCTakensEmbedding,
)
