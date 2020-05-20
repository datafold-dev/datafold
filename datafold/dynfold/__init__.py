"""The second layer package in *datafold* consists of data-driven models dealing directly
with the point cloud manifold or the dynamics of time series data.

The models implemented at this layer can be used in meta-models in the
:class:`datafold.appfold` layer, but can also be used on their own for analysis tasks.
There are three types of models in this layer:

1. Models subclassing :class:`.TSCTransformMixIn`:
   Broadly speaking these models transform data into another representation. However, the
   complexity of and reason for a data transformation can be quite different. The `TSC`
   prefix in the mix-in indicates that the model can also handle time series data  in
   :class:`TSCDataFrame`.

    * One type of data transformation is to compute a new coordinate basis of either point
      cloud or time series data. In this category is the :class:`.DiffusionMaps` model,
      which can be used for manifold learning to non-linearly reduce the dimension of a
      dataset or to approximate the eigenfunctions of the Laplace-Beltrami operator
      (among others).

    * Another type of model transforms a collection of time series. Examples include
      time delay embedding :class:`.TSCTakensEmbedding`,
      scikit-learn wrappers for feature scaling or :class:`TSCPrincipalComponent` which
      are contained in *datafold* to generalize the processing to time series data.

2. Models subclassing :class:`sklearn.base.RegressorMixin`:
   These models interpolate/regress general function values on manifolds. An important
   scenario to use the methods is to provide a mapping (image and pre-image) for
   non-linear manifold learning methods. An example model is the
   :class:`LaplacianPyramidsInterpolator`.

2. Models subclassing :class:`.TSCPredictMixIn`:
   On this level this tpye of model are mainly variants of the Dynamic Mode
   Decomposition algorithm (:class:`.DMDBase`). These models fit time series data,
   meaning the input is restricted to :class:`TSCDataFrame` input. A fitted model
   defines a linear dynamical system which can be used to predict time series.
"""

from datafold.dynfold.dmap import DiffusionMaps, LocalRegressionSelection
from datafold.dynfold.dmd import DMDBase, DMDEco, DMDFull
from datafold.dynfold.outofsample import (
    GeometricHarmonicsInterpolator,
    LaplacianPyramidsInterpolator,
)
from datafold.dynfold.transform import (
    TSCApplyLambdas,
    TSCFeaturePreprocess,
    TSCFiniteDifference,
    TSCIdentity,
    TSCPolynomialFeatures,
    TSCPrincipalComponent,
    TSCRadialBasis,
    TSCTakensEmbedding,
)
