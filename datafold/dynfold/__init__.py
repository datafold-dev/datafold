""" The middle sub-package layer in datafold consists of data-driven models dealing
directly with a data manifold and dynamics of observations.

The models implemented on this layer can be used in meta-models in the
:class:`datafold.appfold` layer, however, the models can also be used on their own. There
arethree type of models in this layer:

1. Models subclassing :class:`.TSCTransformMixIn`:
    Broadly speaking these models transform data into another representation. However,
    the reason and complexity for a data transformation can be quite different. The
    `TSC` prefix in the mix-in indicates that the model can also handle
    time series data (`TSCDataFrame`).

    * An important data transformation is to extract a new coordinate basis (either
      point cloud or time series data). An important model in this category is the
      :class:`.DiffusionMaps`. The models can be used for manifold
      learning to reduce the dimension in the dataset or it can compute an (approximate)
      function basis of the Laplace-Beltrami operator (and others).
    * Other transformations aim to transform time series (or collections
      thereof). These methods have the prefix `TSC` in the classname. The
      transformations include time delay embedding
      :class:`.TSCTakensEmbedding`, but also scikit-learn wrappers like feature scaling
      or :class:`TSCPrincipalComponent` which generalize the input and output to time
      series data.

2. Models subclassing :class:`sklearn.base.RegressorMixin`:
    These models interpolate/regress general function values on manifolds. An
    important scenario to use the methods is to provide a mapping (image
    and pre-image) for non-linear manifold learning methods. An example model is the
    :class:`LaplacianPyramidsInterpolator`.

2. Models subclassing :class:`.TSCPredictMixIn`:
    :class:`.DMDBase` models that linearly decompose time series data to define a linear
    dynamical system with respect to the input data and can be used in a :class:`.EDMD`
    model to compute the Koopman matrix. Because the models only work with time series
    data, the input is restricted to :class:`TSCDataFrame`.
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
