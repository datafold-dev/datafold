---
title: 'datafold: data-driven models with manifold structure for (non-) temporal data'
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

date: 29 April 2020
bibliography: paper.bib
---

# Overview

Ever increasing data availability has changed the way of analysing and interpreting data in many scientific fields and applications. While the (hidden) complex systems being analyzed remain the same, data measurements increase both in quantity and dimension. The main drivers of increasing data availability are larger (computer) simulation capabilities and increasingly versatile and available sensors. 

Contrasting with an equation-driven workflow, a scientist can use data-driven models to analyze a wider range of systems that may also include systems with unknown or untractable equations. The models can be used in a variety of data-driven scenarios, such as amplifying [TODO: find better word] the analysis of unknown systems or merely serve as an equation-free surrogate by providing fast, albeit approximate, responses for unseen data. 

However, expanding datasets also create challanges throughout the entire analysis workflow from processing, extracting, to interpreting the data. On the other hand, new data often does not provide completely new and uncorrelated information to existing data. One way to handle this contradiction is to understand and parametrize the intrinsic data geometry. This structure is often of much lower dimension than the ambient data space, and finding a suitable set of coordinates allows the dataset to be reduced to its intrinsic data geometry. We refer to this geometric structure encoded in the data as a "manifold". In mathematical terms, a manifold is a topological space that is locally homeomorphic to the Eucledian space. Because of the *local* property, in order to find a *global* parametrization of a (smooth) manifold, it is therefore required to account for non-linearity (curvature). The well-known manifold hypothesis states that such manifolds underlie many observations and processes, including time-dependent systems.

In general, all data-driven models presume some pattern or structure in the available data. Many successful machine learning algorithms adapt to this underlying structure to solve tasks like regression or classification. Models can be distinguished and classified with many criteria, for example, with respect to data assumptions or application context (e.g. `[@bishop_pattern_2006:2006]`). In this work, we separate models based on whether the model includes a parametrization of the learnt data manifold. 

The first type of model does not include any manifold parametrization or only does so  *implicitly*. This means that the model parameters can adapt to the intrinsic data structure, but it is impossible to directly access or diagnose what this structure is. A canonical example is the feedforward neural network, which can provide great predictive power as it adapts to general non-linear manifolds, but the trained model is then a black-box that does not provide a link between the parameter weights ("neurons") and the underlying data structure. 

In contrast, the second type of model includes an *explicit* parametrization of the data manifold, which can be profitable for the data analysis. The explicit parametrization allows prior knowledge of a system and its problem specific domain to be included, such as the (partially) known governing equation terms of a system `@williams_datadriven_2015:2015, @brunton_discovering_2016:2016]` or the proximity between points in the dataset `[@coifman_diffusion_2006:2006]`. The parametrization of (non-linear) manifolds is often rooted in the rich theory of functional analysis and differential geometry. Therefore, models commonly include a coordinate change of data into a functional vector basis. However, the explicit manifold parametrization (as part of a model) can also be again a black-box model (e.g. an auto-encoder). The key point is that the choice of how to handle the underlying manifold structure is part of a modelling decision and can be selected according to the problem.

# datafold

*datafold* is a Python package providing **data**-driven models with an explicit mani-**fold** parametrization. The software provides a set of models and data structures, which are all integrated in a software architecture with clear modularization (the model API is used as a template from the scikit-learn project, `@pedregosa_scikit-learn_2011:2011`). The software design of datafold can accomodate models that range from higher level tasks (e.g., system identification, `@williams_datadriven_2015:2015`) to lower level algorithms (e.g. encoding proximity information on a manifold `[@coifman_diffusion_2006:2006]`). In *datafold* we address data with and without temporal context, which is further discussed in the following sections. We want to support the active research in the scope of *datafold* and target students, researchers and experienced practitioners from different fields for dataset analysis.

![(Left) Point cloud of embedded hand written digits between 0 and 5. Each point has 64 dimensions with each dimension being a pixel of an an 8 x 8 image. (Right) Conceptual illustration of a three dimensional time series forming a phase space with geometrical structure. The time series start in the `(x,y)` plane and end in the `z`-axis \label{fig:manifold}](manifold_figure.png)

## 1. Point cloud data

The first type of data are unordered samples in high-dimensional point clouds. These datasets are often directly connected to the "manifold assumption", which states that the data is assumed to lie close to an intrinsic lower dimensional manifold. Our software is aimed towards finding a low dimensional parametrization (embedding) of the manifold. In a machine learning context this is also referred to as "non-linear unsupervised learning" or shorter "manifold learning". A variety of manifold learning models exist in the scikit-learn package. Often the models are endowed with a kernel which encodes the proximity between data with the aim to preserve local structures. Examples are the general "Kernel Principal Component Analysis" `[@bengio_learning_2004]`, "Local Linear Embedding" `[@belkin_laplacian_2003:2003]`, or "Hessian Eigenmaps" `[@donoho_hessian_2003:2003]`. In addition to these, *datafold* provides an efficient implementation of the "Diffusion Maps" model `[@coifman_diffusion_2006:2006]`. The model includes an optional sparse kernel representation that allows to scale with datasets of increasing number of points. With the "Diffusion Map" model, in addition to dimension reduction, a user can approximate the Laplace-Beltrami operator, Fokker-Plank operator or the graph Laplacian. 

[TODO: It would also be good to mention that there are several heuristics in datafold, e.g. to estimate the kernel scale parameter for Diffusion Maps and to choose a good distance matrix computation algorithm.]
[TODO: why do we need this paragraph here? it would be important to at least add a sentence like "The datafold software efficiently implements several of these extension methods".]

An important issue in many applications of manifold learning is extending the image and/or pre-image mapping between the original and latent space to unseen data. This mapping is generally non-linear and therefore often requires a separate model to perform the mapping (see for an example analysis of different models `[@chiavazzo_reduced_2014:2014]`). These so-called "out-of-sample" models interpolate general function values of the  manifold point cloud and, therefore, have to handle a large input data dimensions. In order to extend the function values to manifold regions in the vicinity of the available data samples, models often follow a multi-scale approach. This allows the function to be extended to a larger region when the manifold is "simple" (i.e. close to constant) or to further restricted regions where the manifold is "complicated" (i.e. highly non-linear). Two examples of out-of-sample methods are the "geometric harmonics interpolation" with multiscale extension `[@coifman_geometric_2006:2006]` and the the Laplacian Pyramids `[@fernandez_auto-adaptative_2014:2014; @rabin_heterogeneous_2012:2012]`.

## 2. Time series data

*datafold* can also address data with temporal context sampled from a dynamical system. In this case a data-driven model aims to fit and generalize the underlying dynamics, also known as "system identification" or "time series prediction". The formulation of a dynamical system includes a phase space (i.e. set of possible states) and a rule of how to evolve a given state to a future state. The phase space is usually assumed to be a manifold (see a conceptual illustration in \autoref{fig:manifold}). The temporal context and inherent order of time series data require a more specialized data structure compared to general point clouds. This includes that the usual assumption of idependent and identically distributed (i.i.d.) samples no longer holds. However, to describe the phase space manifold, models for point cloud data become relevant again and can improve the accuracy to identify a dynamical system from data.

*datafold* focuses on the "Dynamic Mode Decomposition" (DMD) `[@schmid_dynamic_2010:2010; @kutz_dynamic_2016:2016]` and the "Extended Dynamic Mode Decomposition" (E-DMD) `[@williams_datadriven_2015:2015]`. DMD linearly decomposes the available time series data into spatio-temporal components, which then define a linear dynamical system. Many DMD based variants address the generally non-linear underlying dynamical system. This is usually done by changing the time series coordinates in a step before DMD is applied `[@williams_datadriven_2015:2015; @champion_discovery_2019:2019; @le_clainche_higher_2017:2017; @giannakis_data-driven_2019:2019]`. The justification of this workflow is covered by operator theory and functional analysis, specifically the Koopman operator. In contrast to a non-linear flow operator in a typical dynamical system form, the Koopman operator acts linearly on a function space (the so-called observable space). The Koopman view on a dynamical system is exact and typically the space of observable functions of a system is infinite dimensional. In practice, the E-DMD approximates the Koopman operator with a matrix, based on a finite set of functions evaluated on the available data, the so-called "dictionary". The functional representation of the dictionary defines a change of coordinates, which in an optimal case linearize the dynamics `[@kutz_dynamic_2016:2016]`. In other words, in an optimal setting the dictionary contains observable functions that linearize the system's dynamics and allows the Koopman matrix to describe a non-linear dynamical systems in this new functional coordinate system. 

However, finding a good choice of dictionary is comparable to the machine learning task of "model selection". In addition to linearizing the dynamics, other reasons often make processing the time series data necessary. For example, heterogenous data can make feature scaling necessary. Another important issue is that the given time samples may actually be only partial observations of the phase space. In this case it is possible to exploit the time ordering and perform a time delay embedding to reconstruct a diffeomorphic copy of the phase space manifold (compare Takens theorem `[@rand_detecting_1981:1981]`). 

[TODO: Again, at least one sentence is missing that states "With datafold, several DMD and EDMD algorithms are implemented and usable in the sklearn pipeline. For example, it is possible to create a pipeline that (A) creates a PCA embedding of the origianl data (B) applies DMAPs on the PCA embedding, and (C) uses this new space as a dictionary of EDMD.]

## Summary

datafold provides an open-source software platform with a design that reflects a workflow hierarchy: From low level data structures and algorithms to high level meta models intended to solve complex machine learning tasks. Setting up a data-driven model to solve complex tasks (such as the E-DMD model) can include a flexible number of data transformations in a processing pipeline. The modularity in datafold mirrors both a high flexibility to test model configurations and openness to new model implementations with a clear and isolated scope.

# Acknowledgements

Daniel Lehmberg (DL) is supported by the German Research Foundation (DFG), grant no. KO 5257/3-1. DL thanks the research office (FORWIN) of Munich University of Applied Sciences and CeDoSIA of TUM Graduate School at the Technical University of Munich for their support.

TODO: Others!


