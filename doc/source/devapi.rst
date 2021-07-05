Documented internals
====================

Classes
~~~~~~~

.. currentmodule:: datafold.dynfold.dmd

.. autoclass:: datafold.pcfold.kernels.RadialBasisKernel

.. autoclass:: datafold.dynfold.dmd.LinearDynamicalSystem

   .. rubric:: Methods Documentation
   .. automethod:: is_matrix_mode
   .. automethod:: is_spectral_mode
   .. automethod:: is_differential_system
   .. automethod:: is_flowmap_system
   .. automethod:: is_linear_system_setup
   .. automethod:: setup_spectral_system
   .. automethod:: setup_matrix_system
   .. automethod:: evolve_system

.. autoclass:: datafold.pcfold.timeseries.accessor.TSCAccessor

   .. rubric:: Methods Documentation
   .. automethod:: assign_ids_const_delta
   .. automethod:: assign_ids_sequential
   .. automethod:: assign_ids_train_test
   .. automethod:: check_const_time_delta
   .. automethod:: check_equal_delta_time
   .. automethod:: check_equal_timevalues
   .. automethod:: check_finite
   .. automethod:: check_min_samples
   .. automethod:: check_no_degenerate_ts
   .. automethod:: check_non_overlapping_timeseries
   .. automethod:: check_normalized_time
   .. automethod:: check_required_min_timesteps
   .. automethod:: check_required_n_timeseries
   .. automethod:: check_required_n_timesteps
   .. automethod:: check_required_time_delta
   .. automethod:: check_timeseries_same_length
   .. automethod:: check_tsc
   .. automethod:: shift_matrices
   .. automethod:: equal_const_delta_time
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
    :skip: List,NotFittedError,TSCDataFrame,TSCMetric,TSCScoring,TimePredictType,TransformerMixin,Tuple,TSCException

.. automodapi:: datafold.pcfold.distance
    :skip: NearestNeighbors,cdist,if1dim_colvec,pairwise_distances,pdist,squareform,Sequence,Type,BallTree,if1dim_rowvec,is_integer

.. automodapi:: datafold.pcfold.eigsolver
    :skip: Callable,sort_eigenpairs,is_symmetric_matrix,Tuple,Dict
