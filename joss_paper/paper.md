---
title: 'datafold: data-driven models for point clouds and time series on manifolds'
tags:
  - data-driven models
  - manifold learning
  - point cloud
  - time series
  - dynamic mode decomposition
  - dynamical system
  - Python
authors:
  - name: Daniel Lehmberg
    orcid: 0000-0002-4012-5014
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Felix Dietrich
    orcid: 0000-0002-2906-1769
    affiliation: 2
  - name: Gerta Köster
    orcid: 0000-0002-3369-6206
    affiliation: 1
  - name: Hans-Joachim Bungartz
    affiliation: 2
affiliations:
 - name: Munich University of Applied Sciences, Lothstr. 64, 80333 Munich, Germany
   index: 1
 - name: Technical University of Munich, Boltzmannstr. 3, 85747 Garching, Germany
   index: 2
date: 18 May 2020
bibliography: paper.bib
---

# Summary
Ever increasing data availability has changed the way how data is analyzed and interpreted in many scientific fields. While the underlying complex systems remain the same, data measurements increase in both quantity and dimension. The main drivers are larger computer simulation capabilities and increasingly versatile sensors. In contrast to an equation-driven workflow, a scientist can use data-driven models to analyze a wider range of systems, including those with unknown or intractable equations. The models can be applied to a variety of data-driven scenarios, such as enriching the analysis of unknown systems or merely serve as an equation-free surrogate by providing fast, albeit approximate, responses to unseen data.

However, expanding datasets create challenges throughout the analysis workflow from extracting and processing to interpreting the data. This includes the fact that new data does not always provide completely new and uncorrelated information to existing data. One way to extract the essential information is to understand and parametrize the intrinsic data geometry. An intrinsic geometry is what most data-driven models assume implicitly or explicitly in the available data, and successful machine learning algorithms adapt to this underlying structure for tasks like regression or classification [e.g., @bishop16]. This geometry is often of much lower dimension than the ambient data space, and finding a suitable set of coordinates can reduce the complexity of the dataset. We refer to this geometric structure encoded in the data as a "manifold". In mathematical terms, a manifold is a topological space that is locally homeomorphic to Euclidean space. Typically, manifold learning attempts to construct a global parametrization (embedding) of this manifold, in a space of much lower dimension than the original ambient space. The well-known manifold hypothesis states that such manifolds underlie many observations and processes, including time-dependent systems.


*datafold* is a Python package that provides **data**-driven models for point clouds to find an *explicit* mani-**fold** parametrization and to identify non-linear dynamical systems on these manifolds. The explicit data manifold treatment allows prior knowledge of a system and its problem-specific domain to be included. This can be the proximity between points in the dataset [@coifman8] or functions defined on the phase space manifold of a dynamical system, such as (partially) known governing equation terms [@williams4; @brunton18].

*datafold* is open-source software with a design that reflects a workflow hierarchy: from low-level data structures and algorithms to high-level meta-models intended to solve complex machine learning tasks. The key benefit of *datafold* is that it accommodates and integrates models on the different workflow levels. Each model has been investigated and tested individually and found to be useful by the scientific community. In *datafold* these models can be used in a single processing pipeline. Our integrated workflow facilitates the application of data-driven analysis and thus has the potential to boost widespread utilization. The implemented models are integrated into a software architecture with a clear modularization and an API that is templated from the `scikit-learn` project, which can be used as part of its processing pipeline [@pedregosa17]. The data structures are subclasses from common objects of the Python scientific computing stack, allowing models to generalize for both static point clouds and temporally ordered time series collection data. The software design and modularity in *datafold* reflects two requirements: high flexibility to test model configurations, and openness to new model implementations with clear and well-defined scope. We want to support active research in data-driven analysis with manifold context and thus target students, researchers and experienced practitioners from different fields of dataset analysis.

![(Left) Point cloud of embedded handwritten digits between 0 and 5 with the "Diffusion Map" model. Each point originally has 64 dimensions where each dimension represents a pixel of an 8 x 8 image. (Right) Conceptual illustration of a three-dimensional time series forming a phase space with geometrical structure. The time series start on the `(x,y)` plane and end on the `z`-axis. \label{fig:manifold}](manifold_figure.png)

## 1. Point cloud data

High-dimensional and unordered point clouds are often directly connected to the "manifold assumption", i.e. that the data lies close to a lower-dimensional manifold. Our software aims to find a low-dimensional parametrization (embedding) of this manifold. In a machine learning context, this is also referred to as "non-linear unsupervised learning" or shorter "manifold learning". Often the models are endowed with a kernel which encodes the proximity between data to preserve local structures. Examples are the general "Kernel Principal Component Analysis" [@bengio9], "Local Linear Embedding" [@belkin14], or "Hessian Eigenmaps" [@donoho15]. A variety of manifold learning models already exist in the `scikit-learn` Python package. In addition to these, *datafold* provides an efficient implementation of the "Diffusion Maps" model [@coifman8]. The model includes an optional sparse kernel matrix representation with which the model can scale to larger datasets. In addition to non-linear dimension reduction, "Diffusion Maps" allow the user to approximate mathematically meaningful objects on manifold data, such as the Laplace-Beltrami operator [@coifman8]. *datafold* also supplies functionality for follow-up aspects of non-linear manifold learning, such as estimating the kernel scale parameters to describe the locality of points in a dataset and extending the embedding to unseen data. The latter refers to the image and pre-image mapping between the original and latent space [e.g., see analysis in @chiavazzo19]. This so-called "out-of-sample" extension interpolates general function values on manifold point clouds and, therefore, has to handle large input data dimensions [@coifman7; @fernandez10; @rabin6]. In *datafold*, out-of-sample extensions are implemented efficiently, so that interpolated function values for millions of points can be computed in seconds on a standard desktop computer.

## 2. Time series data
A special kind of point cloud type targeted by *datafold* are time series and collections thereof. In this case, a data-driven model can fit and generalize the underlying dynamics to perform prediction or regression. Usually, the phase space of the dynamical system, underlying the time series observations, is assumed to be a manifold (see a conceptual illustration in \autoref{fig:manifold}). *datafold* focuses on the algorithms "Dynamic Mode Decomposition" (DMD) [@schmid13; @tu3; @kutz12] and "Extended Dynamic Mode Decomposition" [E-DMD, @williams4]. DMD linearly decomposes the available time series data into spatio-temporal components, which then define a linear dynamical system. Many DMD based variants address even more general, non-linear underlying dynamical systems. This is usually done by changing the time series coordinates in a step before DMD is applied [@williams4; @champion2; @le0; @giannakis1]. The justification of this workflow is covered by operator theory and functional analysis, specifically the Koopman operator. In practice, the E-DMD algorithm approximates the Koopman operator with a matrix, based on a finite set of functions evaluated on the available data, the so-called "dictionary". Finding a good choice for the dictionary is comparable to the machine learning task of "model selection" and requires great flexibility in setting up the data processing pipeline. The flexibility of setting an arbitrary dictionary combined with a selection of the provided DMD variants is a core feature of *datafold's* implementation of E-DMD.

## Comparison to other software projects

The Python package *statsmodels* [@seabold22] includes statistical models for time series analysis, such as the *AutoRegressive Integrated Moving Average* (ARIMA) and variations thereof. These models usually assume an underlying stochastical process and aim to approximate autocorrelations in time series data. Another type of popular data-driven models are deep neural networks; widely adopted packages are the frameworks *tensorflow* [@abadi24; @abadi26] and *PyTorch* [@paszke21]. Deep learning models generalize well in several problem domains, but common drawbacks are a non-deterministic construction process, the requirement of large datasets, and only limited options to include prior knowledge about the underlying system. For manifold learning, one can use (variational) autoencoders, while recurrent architectures [e.g., LSTM networks @hochreiter25] are often used for system identification. Other packages in *datafold*’s scope are the Python packages *PyDMD* [@demo23] and *PySINDy* [@desilva20]. The *PyDMD* project includes numerous variations of DMD, such as the "Higher Order DMD" [@le0]. The *PySINDy* software focuses on "Sparse Identification of Non-linear Dynamics" (SINDy) [@brunton18]. *datafold*'s EDMD implementation allows an arbitrary DMD variant to be used in a processing pipeline to regress the dynamics, which means the models from both *PyDMD* and *PySINDy* could be used in combination with *datafold*.

# Acknowledgements

DL is supported by the German Research Foundation (DFG), grant no. KO 5257/3-1 and thanks the research office (FORWIN) of Munich University of Applied Sciences and CeDoSIA of TUM Graduate School at the Technical University of Munich for their support. DL and FD thank Yannis Kevrekidis from the Johns Hopkins University, Baltimore, USA, for his support during a research stay of DL in 2018, which paved the way for the *datafold* software project.

# References
