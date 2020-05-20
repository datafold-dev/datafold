Documented internals
====================

Functions
~~~~~~~~~

.. autofunction:: datafold.pcfold.timeseries.metric.kfold_cv_reassign_ids


Classes
~~~~~~~

.. currentmodule:: datafold.dynfold.dmd

.. autoclass:: datafold.pcfold.kernels.RadialBasisKernel

.. autoclass:: datafold.dynfold.dmd.LinearDynamicalSystem

   .. rubric:: Methods Documentation
   .. automethod:: evolve_system_spectrum

.. autoclass:: datafold.pcfold.timeseries.accessor.TSCAccessor

   .. rubric:: Methods Documentation
   .. automethod:: check_const_time_delta
   .. automethod:: check_finite
   .. automethod:: check_normalized_time
   .. automethod:: check_required_min_timesteps
   .. automethod:: check_required_n_timeseries
   .. automethod:: check_required_time_delta
   .. automethod:: check_timeseries_same_length
   .. automethod:: check_timeseries_same_timevalues
   .. automethod:: check_tsc
   .. automethod:: normalize_time
   .. automethod:: plot_density2d
   .. automethod:: compute_shift_matrices
   .. automethod:: shift_time
   .. automethod:: time_values_overview


Modules
~~~~~~~

.. automodapi:: datafold.dynfold.base
    :skip: BaseEstimator,DmapKernelFixed,List,NotFittedError,NumericalMathError,TSCDataFrame,TSCMetric,TSCScoring,TimePredictType,TransformerMixin,Tuple,DataFrameType

.. automodapi:: datafold.pcfold.distance
    :skip: NearestNeighbors,apply_continuous_nearest_neighbor,cdist,if1dim_colvec,pairwise_distances,pdist,squareform,warn_experimental_function,warn_experimental_function,Sequence,Type,BallTree,is_symmetric_matrix,if1dim_rowvec

.. automodapi:: datafold.pcfold.eigsolver
    :skip: Callable,sort_eigenpairs,is_symmetric_matrix,Tuple,Dict
