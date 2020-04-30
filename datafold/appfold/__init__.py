"""
The highest level includes models that capture complex processing pipelines. This type
of model can be described as "meta model" because it provides a single point of
access to usually multiple sub-models included in a model. The models on this layer are
therefore end of the Machine Learning pipeline and are intended to solve complex
data-driven usecases or analysis tasks.

The high degree of modularization in datafold's software of the first and second layer
become profitable for the meta-models: The data process pipeline can be combined with a
high degree of flexibility, which makes it easier to test model configurations and
model accuracies in a parameter space.

datafold provides an model implementation of the "Extended Dynamic Mode Decomposition".
Creating an instance of :class:`EDMD` allows a flexible choice of dictionary to be set
together with a final DMD based model (:class:`DMDBase`) to approximate the Koopman
operator in the functional representation of the dictionary. Following up on the
meta-model approach `EDMD` can be integrated in a cross-validation pipeline.
:class:`EDMDCV` (subclassing :class:`sklearn.model_selection.GridSearchCV`)
provides an exhaustive search over a user specified parameter space, including
cross-validation splitting of time series data.
"""

from datafold.appfold.edmd import EDMD, EDMDCV
