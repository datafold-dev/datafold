Documented internals
====================

Classes
~~~~~~~

.. currentmodule:: datafold.dynfold.dmd

.. autoclass:: datafold.pcfold.kernels.RadialBasisKernel

.. autoclass:: datafold.dynfold.dmd.LinearDynamicalSystem

   .. rubric:: Methods Documentation
   .. automethod:: evolve_system_spectrum

.. autoclass:: datafold.pcfold.timeseries.accessor.TSCAccessor

   .. rubric:: Methods Documentation
   .. automethod:: assign_ids_const_delta
   .. automethod:: assign_ids_sequential
   .. automethod:: assign_ids_train_test
   .. automethod:: check_const_time_delta
   .. automethod:: check_finite
   .. automethod:: check_non_overlapping_timeseries
   .. automethod:: check_normalized_time
   .. automethod:: check_required_min_timesteps
   .. automethod:: check_required_n_timeseries
   .. automethod:: check_required_time_delta
   .. automethod:: check_timeseries_same_length
   .. automethod:: check_timeseries_same_timevalues
   .. automethod:: check_tsc
   .. automethod:: compute_shift_matrices
   .. automethod:: iter_timevalue_window
   .. automethod:: normalize_time
   .. automethod:: plot_density2d
   .. automethod:: shift_time
   .. automethod:: time_derivative
   .. automethod:: time_values_overview


.. autoclass:: datafold.pcfold.timeseries.collection.TSCException


Modules
~~~~~~~

.. Fill in classes that have 
       WARNING: autosummary: stub file not found 'XXXXX'. Check your autosummary_generate setting.

.. automodapi:: datafold.dynfold.base
    :skip: List,NotFittedError,TSCDataFrame,TSCMetric,TSCScoring,TimePredictType,TransformerMixin,Tuple,TSCException,NamedTuple

.. automodapi:: datafold.pcfold.distance
    :skip: NearestNeighbors,cdist,if1dim_colvec,pairwise_distances,pdist,squareform,Sequence,Type,BallTree,if1dim_rowvec

.. automodapi:: datafold.pcfold.eigsolver
    :skip: Callable,sort_eigenpairs,is_symmetric_matrix,Tuple,Dict




