---
title: 'datafold: efficient algorithms for data close to non-linear manifolds'
tags:
  - data driven models
  - manifold 
  - time series
  - extended dynamic mode decomposition 
  - dynamical systems
  - Python
 
authors:
  - name: Daniel Lehmberg
    orcid: 0000-0002-4012-5014
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Felix Dietrich
    orcid: 0000-0002-4012-5014
    affiliation: 2
  - name: Gerta Köster 
    orcid: 0000-0002-4012-5014
    affiliation: 1
  - name: Hans-Joachim Bungartz
    orcid: 0000-0002-4012-5014
    affiliation: 2
  
affiliations:
 - name: Munich University of Applied Sciences
   index: 1
 - name: Technical University of Munich
   index: 2

date: 06 May 2020
bibliography: paper.bib
---

# TODO
Alt. titles:

* datafold: data-driven models with explicit parametrization of non-linear manifolds   

# Summary

Ever increasing data availability has changed the way of analysing and interpreting data in many scientific fields and applications. While the (hidden) complex systems being analyzed remain the same, data measurements increase both in quantity and dimension. The main drivers of increasing data availability are larger (computer) simulation capabilities and increasingly versatile and available sensors. Contrasting with an equation-driven workflow, a scientist can use data-driven models to analyze a wider range of systems that may also include systems with unknown or untractable equations. The models can be used in a variety of data-driven scenarios, such as enriching the analysis of unknown systems or merely serve as an equation-free surrogate by providing fast, albeit approximate, responses for unseen data. 

However, expanding datasets also create challanges throughout the entire analysis workflow from processing, extracting, to interpreting the data. On the other hand, new data often does not provide completely new and uncorrelated information to existing data. One way to handle this contradiction is to understand and parametrize the intrinsic data geometry. This structure is often of much lower dimension than the ambient data space, and finding a suitable set of coordinates allows the dataset to be reduced to its intrinsic data geometry. We refer to this geometric structure encoded in the data as a "manifold". In mathematical terms, a manifold is a topological space that is locally homeomorphic to the Eucledian space. Because of the *local* property, in order to find a *global* parametrization of a (smooth) manifold, it is therefore required to account for non-linearity (curvature). The well-known manifold hypothesis states that such manifolds underlie many observations and processes, including time-dependent systems.

In general, all data-driven models presume some pattern or structure in the available data. Many successful machine learning algorithms adapt to this underlying structure to solve tasks like regression or classification. Models can be distinguished and classified with many criteria, for example, with respect to data assumptions or application context (e.g. `[@bishop_pattern_2006:2006]`). In this work, we separate models based on whether the model includes a parametrization of the learnt data manifold. 

*datafold* is a Python package providing **data**-driven models with an *explicit* mani-**fold** parametrization. The explicit parametrization allows prior knowledge of a system and its problem specific domain to be included, such as the (partially) known governing equation terms of a system `@williams_datadriven_2015:2015, @brunton_discovering_2016:2016]` or the proximity between points in the dataset `[@coifman_diffusion_2006:2006]`. The software provides a set of models and data structures, which are all integrated in a software architecture with clear modularization (the model API is used as a template from the scikit-learn project, `@pedregosa_scikit-learn_2011:2011`). The software design of datafold can accomodate models that range from higher level tasks (e.g., system identification, `@williams_datadriven_2015:2015`) to lower level algorithms (e.g., encoding proximity information on a manifold `[@coifman_diffusion_2006:2006]`). We want to support the active research in the scope of data-driven analysis with manifold context and target students, researchers and experienced practitioners from different fields for dataset analysis. 

In *datafold* we address data static points clouds and temporal ordered time series data.

![(Left) Point cloud of embedded hand written digits between 0 and 5. Each point has 64 dimensions with each dimension being a pixel of an an 8 x 8 image. (Right) Conceptual illustration of a three dimensional time series forming a phase space with geometrical structure. The time series start in the `(x,y)` plane and end in the `z`-axis \label{fig:manifold}](manifold_figure.png)

## 1. Point cloud data

High-dimensional and unordered point clouds are often directly connected to the "manifold assumption", which states that the data is assumed to lie close to an intrinsic lower dimensional manifold. Our software is aimed towards finding a low dimensional parametrization (embedding) of the manifold. In a machine learning context this is also referred to as "non-linear unsupervised learning" or shorter "manifold learning". Often the models are endowed with a kernel which encodes the proximity between data with the aim to preserve local structures. Examples are the general "Kernel Principal Component Analysis" `[@bengio_learning_2004]`, "Local Linear Embedding" `[@belkin_laplacian_2003:2003]`, or "Hessian Eigenmaps" `[@donoho_hessian_2003:2003]`. A variety of manifold learning models already exist in the `scikit-learn` Python package. In addition to these, *datafold* provides an efficient implementation of the "Diffusion Maps" model `[@coifman_diffusion_2006:2006]`. The model includes an optional sparse kernel matrix representation that allows to scale with datasets of increasing number of points. In addition to dimension reduction, with "Diffusion Maps", a user can approximate the Laplace-Beltrami operator, Fokker-Plank operator or the graph Laplacian. *datafold* also provides functionality for follow up aspects of non-linear manifold learning. Important issues are estimating the kernel scale parameters to describe the locality of points in a dataset and extending the embedding to useen data. The latter is referred the image and/or pre-image mapping between the original and latent space (e.g., see analysis in `[@chiavazzo_reduced_2014:2014]`). These so-called "out-of-sample" entensions interpolate general function values on manifold point clouds and, therefore, have to handle large input data dimensions `[@coifman_geometric_2006:2006; @fernandez_auto-adaptative_2014:2014; @rabin_heterogeneous_2012:2012]`.


## 2. Time series data

As a second type of data, *datafold* targets time series (and collections of thereof). In this case a data-driven model aims to fit and generalize the underlying dynamics to perform prediction or regression. Usually, the phase space of the dynamical system, underlying the time series observations, is usually assumed to be a manifold (see a conceptual illustration in \autoref{fig:manifold}). *datafold* focuses on the "Dynamic Mode Decomposition" (DMD) `[@schmid_dynamic_2010:2010; @kutz_dynamic_2016:2016]` and the "Extended Dynamic Mode Decomposition" (E-DMD) `[@williams_datadriven_2015:2015]`. DMD linearly decomposes the available time series data into spatio-temporal components, which then define a linear dynamical system. Many DMD based variants address the generally non-linear underlying dynamical system. This is usually done by changing the time series coordinates in a step before DMD is applied `[@williams_datadriven_2015:2015; @champion_discovery_2019:2019; @le_clainche_higher_2017:2017; @giannakis_data-driven_2019:2019]`. The justification of this workflow is covered by operator theory and functional analysis, specifically the Koopman operator. In practice, the E-DMD approximates the Koopman operator with a matrix, based on a finite set of functions evaluated on the available data, the so-called "dictionary". Finding a good choice of dictionary, however, is comparable to the machine learning task of "model selection". One main objective in the model implementation of E-DMD and DMD variants included *datafold* is therefore a great flexibility in setting up a data processing pipeline (the E-DMD implementation subclasses the scikit-learn Pipeline). *datafold* provides an open-source software platform with a design that reflects a workflow hierarchy: From low level data structures and algorithms to high level meta models intended to solve complex machine learning tasks. The modularity in datafold mirrors both a high flexibility to test model configurations and openness to new model implementations with a clear and isolated scope.


# Acknowledgements

Daniel Lehmberg (DL) is supported by the German Research Foundation (DFG), grant no. KO 5257/3-1. DL thanks the research office (FORWIN) of Munich University of Applied Sciences and CeDoSIA of TUM Graduate School at the Technical University of Munich for their support.

TODO: Others!


