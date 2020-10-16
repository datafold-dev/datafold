"""The lowest level of *datafold* provides data structures, objects directly associated
with data (e.g., kernels) and fundamental algorithms on data (e.g., distance matrix and
eigen-solver). There are two data structures provided in *datafold*:

* :py:class:`.PCManifold` for point cloud data with manifold assumption. The data
  structure is derived from :class:`numpy.ndarray` and attaches a kernel to describe
  local proximity between points. All kernels implemented in *datafold* have the base
  class :py:class:`.PCManifoldKernel`. The data structure encapsulates the complexity of
  recurring routines in kernel methods. For example, it computes sparse/dense kernel
  matrices of different distance metrics and eigenpairs with different backends.

* :py:class:`.TSCDataFrame` is a collections of time series data. The data structure's
  base class is Pandas' :class:`pandas.DataFrame` and can index multiple time series and
  time values of potentially multi-dimensional time series in a single object. The data
  structure is mainly required for system identification models. The time series can have
  different properties, e.g. different time series lengths or time values. For this
  reason. there are many methods available to validate the model's assumptions.

There are other low-level machine learning algorithms or tasks directly connected
to the data structures. For estimating the scale of a :class:`GaussianKernel` the
algorithms in :py:meth:`PCManifold.optimize_parameters` are suitable. For
:py:class:`TSCDataframe` this includes time series splits into training/test sets (
(:py:class:`TSCKfoldSeries` or :py:class:`TSCKFoldTime`) or measuring error metrics
between predicted and true time series (:py:class:`TSCMetric`).
"""


import datafold.pcfold.timeseries.accessor
from datafold.pcfold.kernels import (
    ConeKernel,
    ContinuousNNKernel,
    CubicKernel,
    DmapKernelFixed,
    GaussianKernel,
    InverseMultiquadricKernel,
    MultiquadricKernel,
    PCManifoldKernel,
    QuinticKernel,
)
from datafold.pcfold.pointcloud import (
    PCManifold,
    estimate_cutoff,
    estimate_scale,
    pcm_remove_outlier,
    pcm_subsample,
)
from datafold.pcfold.timeseries.collection import (
    InitialCondition,
    TSCDataFrame,
    allocate_time_series_tensor,
)
from datafold.pcfold.timeseries.metric import (
    TSCKfoldSeries,
    TSCKFoldTime,
    TSCMetric,
    TSCScoring,
    TSCWindowFoldTime,
)
