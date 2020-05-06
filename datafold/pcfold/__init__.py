"""
The first and lowest level of *datafold* provides data structures, objects directly
associated to data (e.g., kernel) and fundamental algorithms on data (e.g., distance
matrix and eigen-solver). There are two data structure provided in *datafold*:

* Point cloud data with manifold assumption :class:`.PCManifold` (subclasses
  :class:`numpy.ndarray`). It attaches a kernel and distance metric information to data
  to describe local proximity between points. The data structure encapsulates complexity
  of recurring routines in kernel methods (e.g., computing sparse/dense kernel
  matrices and eigenpairs with different backends).

* Collections of time series data :class:`.TSCDataFrame` (subclasses Pandas'
  :class:`pandas.DataFrame`). The data structure indexes multiple time series and time
  values of potentially multi-dimensional time series in a single data frame object. The
  time series can have different properties, e.g. different lengths or time values.
  This allows all available data to be included in one data object. The data structure
  is required for system identification models. Furthermore, the data structure
  provides many functions that allow to validate time series assumptions of a model (
  e.g. do time series have the same length, time values or constant time sampling?).

There is also functionality directly connected to the data structures. For example,
optimizing the parameters of a Gaussian kernel with
:py:meth:`PCManifold.optimize_parameters`  parameters, performing time
series splits into training/test sets
measuring error metrics between predicted and true time series.
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
