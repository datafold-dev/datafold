"""
First and lowest level in datafold. It provides data structures, objects directly
associated to data (e.g., kernel) and fundamental algorithms on data (e.g., distance
matrix and eigen-solver). There are two data structure provided by datafold:

* Point cloud data with manifold assumption :class:`PCManifold` (subclasses
  :class:`numpy.ndarray`). It attaches a kernel and distance metric information to data
  to describe local proximity between points. The data structure encapsulates complexity
  of re-ocurring routines in kernel methods (e.g., computing sparse/dense kernel
  matricies and eigenpairs with different backends).

* Collections of time series data :class:`TSCDataFrame`  (subclasses Pandas'
  ``pandas.DataFrame``). The data structure indexes different time series and time
  values of potentially multi-dimensional time series in a data frame. The time series
  can have different properties, e.g. different lengths or time values. This allows all
  available data to be included in one data object. The data structure is required
  for system identification models. Furthermore, the data structure provides many
  functions that allow to validate time series assumptions of a model (e.g. do time
  series have the same length, time values or constant time sampling?).

On this level there is also more functionality directly connected to the data
structures. For example, performing time series splits into training and test sets or
measuring error metrics and scoring between predicted and true time series.
"""


import datafold.pcfold.timeseries.accessor
from datafold.pcfold.kernels import (
    CubicKernel,
    DmapKernelFixed,
    GaussianKernel,
    InverseMultiquadricKernel,
    MultiquadricKernel,
    PCManifoldKernel,
    QuinticKernel,
    ThinPlateKernel,
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
)
