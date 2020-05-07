---
title: 'datafold: efficient algorithms for data lying close to non-linear manifolds'
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
* datafold: efficient data-driven models for (points/samples) lying near non-linear
  manifolds

# Gerta's comments

* I took out most of the brackets (), because they interrupt the flow of a sentence. If it's important, you don't need a bracket, if it's not, drop it entirely.
* I took out a "contradiction" where I saw none. And a "on the one hand" where there was no "other hand".
* There are two or three spots where I do not follow you completely, but mostly I find it very clear.
* *Can we bragg more?* If this is supposed to be a teaser, where is the eniticing news? If you tell me where *datafold* is unique, I can help with finding the words to sell it.

# Summary
Ever increasing data availability has changed the way data is analyzed and interpreted in many scientific fields. While the underlying complex systems remain the same, data measurements increase in both quantity and dimension. The main drivers are larger computer simulation capabilities and increasingly versatile sensors. In contrast to an equation-driven workflow, a scientist can use data-driven models to analyze a wider range of systems that may also involve systems with unknown or intractable equations. The models can be applied to a variety of data-driven scenarios, such as enriching the analysis of unknown systems, or merely serve as an equation-free surrogate by providing fast, albeit approximate, responses to unseen data. 

However, expanding datasets create challenges throughout the analysis workflow from processing, extracting, to interpreting data. One is that while there is more data, it often does not provide completely new, uncorrelated information to existing data. One way to handle this is to understand and parametrize the intrinsic data geometry. This structure is often of much lower dimension than the ambient data space, and finding a suitable set of coordinates can reduce the dataset to its intrinsic data geometry. We refer to this geometric structure encoded in the data as a "manifold". In mathematical terms, a manifold is a topological space that is locally homeomorphic to the Euclidian space. Because of the *local* property, finding a *global* parametrization of a (smooth) manifold, requires accounting for non-linearity, that is, curvature. The well-known manifold hypothesis states that such manifolds underlie many observations and processes, including time-dependent systems.

In general, all data-driven models presume some pattern or structure in the available data. Many successful machine learning algorithms adapt to this underlying structure when solving tasks like regression or classification. Models can be distinguished and classified with many criteria, for example, with respect to data assumptions or application context (e.g., `[@bishop16:2006]`). In this work, we separate models based on whether the model includes a parametrization of the learned data manifold. 

*datafold* is a Python package that provides **data**-driven models with an *explicit* mani-**fold** parametrization. The explicit parametrization allows prior knowledge of a system and its problem-specific domain to be included, such as the (partially) known governing equation terms of a system `@williams4:2015, @brunton18:2016]` or the proximity between points in the dataset `[@coifman8:2006]`.  In *datafold* we address data static point clouds and temporal ordered time series data. The software bundles a set of models and data structures, which are all integrated in an architecture with clear modularization (the model API is used as a template from the `scikit-learn` project, `@pedregosa17:2011`). The software design of datafold can accomodate models that range from higher level tasks, such as system identification, `@williams4:2015`) to lower level algorithms, e.g. encoding proximity information on a manifold `[@coifman8:2006]. *
In *datafold* we address data static point clouds and temporal ordered time series data.datafold* is an open-source software platform with a design that reflects a workflow hierarchy: from low level data structures and algorithms to high level meta models intended to solve complex machine learning tasks. The modularity in *datafold* reflects two requirements: high flexibility to test model configurations, and openness to new model implementations with a clear and isolated scope. We want to support active research on data-driven analysis in a manifold context. Thus we target students, researchers and experienced practitioners from different fields of dataset analysis.

![(Left) Point cloud of embedded hand written digits between 0 and 5. Each point has 64 dimensions with each dimension being a pixel of an an 8 x 8 image. (Right) Conceptual illustration of a three dimensional time series forming a phase space with geometrical structure. The time series start in the `(x,y)` plane and end in the `z`-axis \label{fig:manifold}](manifold_figure.png)

## 1. Point cloud data

High-dimensional and unordered point clouds are often directly connected to the "manifold assumption", that the data lies close to an intrinsic lower dimensional manifold. Our software aims to find a low dimensional parametrization (embedding) of this manifold. In a machine learning context this is also referred to as "non-linear unsupervised learning" or shorter "manifold learning". Often the models are endowed with a kernel which encodes the proximity between data to preserve local structures. Examples are the general "Kernel Principal Component Analysis" `[@bengio9:2004]`, "Local Linear Embedding" `[@belkin14:2003]`, or "Hessian Eigenmaps" `[@donoho15:2003]`. A variety of manifold learning models already exist in the `scikit-learn` Python package. In addition to these, *datafold* provides an efficient implementation of the "Diffusion Maps" model `[@coifman8:2006]`. The model includes an optional sparse kernel matrix representation so that datasets  that comprise an increasing number of points can be scaled. In addition to dimension reduction, "Diffusion Maps" allow the user to approximate the Laplace-Beltrami operator, Fokker-Plank operator or the graph Laplacian. *datafold* also supplies functionality for follow up aspects of non-linear manifold learning, such as estimating the kernel scale parameters to describe the locality of points in a dataset and extending the embedding to unseen data. The latter refers to the image and/or pre-image mapping between the original and latent space (e.g., see analysis in `[@chiavazzo19:2014]`). These so-called "out-of-sample" extension interpolate general function values on manifold point clouds and, therefore, have to handle large input data dimensions `[@coifman7:2006; @fernandez10:2014; @rabin6:2012]`.

## 2. Time series data

As a second type of data, *datafold* targets time series and collections thereof. In this case, a data-driven model aims to fit and generalize the underlying dynamics to perform prediction or regression. Usually, the phase space of the dynamical system, underlying the time series observations, is assumed to be a manifold (see a conceptual illustration in \autoref{fig:manifold}). *datafold* focuses on the "Dynamic Mode Decomposition" (DMD) `[@schmid13:2010; @tu3:2014; @kutz12:2016]` and the "Extended Dynamic Mode Decomposition" (E-DMD) `[@williams4:2015]`. DMD linearly decomposes the available time series data into spatio-temporal components, which then define a linear dynamical system. Many DMD based variants address the generally non-linear underlying dynamical system. This is usually done by changing the time series coordinates in a step before DMD is applied `[@williams4:2015; @champion2:2019; @le0:2017; @giannakis1:2019]`. The justification of this workflow is covered by operator theory and functional analysis, specifically the Koopman operator. In practice, the E-DMD approximates the Koopman operator with a matrix, based on a finite set of functions evaluated on the available data, the so-called "dictionary". Finding a good choice for the dictionary is comparable to the machine learning task of "model selection" and requires great  flexibility in setting up the data processing pipeline. This flexibiltiy is the core of *datafold's* implementation of E-DMD.


# Acknowledgements

Daniel Lehmberg (DL) is supported by the German Research Foundation (DFG), grant no. KO 5257/3-1. DL thanks the research office (FORWIN) of Munich University of Applied Sciences and CeDoSIA of TUM Graduate School at the Technical University of Munich for their support.

TODO: Others! (GK: I suggest you mention your stay with Yannis' group.)


