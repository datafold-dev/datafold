<!---
From:
https://joss.readthedocs.io/en/latest/submitting.html#what-should-my-paper-contain

Your paper should include:

* A list of the authors of the software and their affiliations, using the correct
 format (see the example below).
* A summary describing the high-level functionality and purpose of the software for a
 diverse, non-specialist audience.
* A clear Statement of Need that illustrates the research purpose of the software.
* A list of key references, including to other software addressing related needs.
* Mention (if applicable) of any past or ongoing research projects using the software
 and recent scholarly publications enabled by it.
* Acknowledgement of any financial support.

Check compilation of paper here:
https://whedon.theoj.org/

List of potential reviewers (check out who matches!):
https://docs.google.com/spreadsheets/d/1PAPRJ63yq9aPC1COLjaQp8mHmEq3rZUzwUYxTulyu78/edit#gid=856801822

-->

---
title: 'datafold: data-driven manifold structure in (non-) temporal data'
tags:
  - data driven models
  - manifold 
  - time series
  - extended dynamic mode decomposition 
  - dynamical systems
  -  Python
 
authors:
  - name: Daniel Lehmberg
    orcid: 0000-0002-4012-5014
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Felix Dietrich
    orcid: 0000-0002-4012-5014
    affiliation: 2
  - name Gerta Köster 
    orcid: 0000-0002-4012-5014
    affiliation: 1
  - Hans
  
affiliations:
 - name: Munich University of Applied Sciences
   index: 1
 - name: Technical University of Munich
   index: 2
date: DD MMMM YYYY
bibliography: paper.bib
---

# TODO
* [ ] A statement of need: Do the authors clearly state what problems the software is designed to solve and who the target audience is?
* [ ] Installation instructions: Is there a clearly-stated list of dependencies? Ideally these should be handled with an automated package management solution.
* [ ] Example usage: Do the authors include examples of how to use the software (ideally to solve real-world analysis problems).
* [ ] Functionality documentation: Is the core functionality of the software documented to a satisfactory level (e.g., API method documentation)?
* [ ] Automated tests: Are there automated tests or manual steps described so that the functionality of the software can be verified?
* [ ] Community guidelines: Are there clear guidelines for third parties wishing to 1) Contribute to the software 2) Report issues or problems with the software 3) Seek support

# Introduction 

!-- the data may be on multiple scales both spatial and temporal
!-- model reduction, surrogate models, predictions
!-- Why are we using data-driven models -> for systems where governing equations are unknown or untractable --> example ist climate or weather
!-- do not argue fully with interpretability - instead argue it opens many possibilities of what to include -- e.g. we can insert expert knowledge, prior assumption of a metric, or simply use mathematical theory to approximate manifolds. 


Ever increasing data availability has changed the ways of analysing and interpreting data in many scientific fields and applications. While the (hidden) complex systems being analyzed, such as a physical systems remain the same, the number of data measurements increase both in quantity and dimension. Main reasons are larger (computer) simulation capabilities and a versatile availability of sensors. 

Contrasting an equation-driven workflow, data-driven models allow a wider range of systems with also unknown or untractable equations to be analyzed. The models can be used in a variety of data-driven usecases, such as amplifying the analysis of unknown systems |cite| or merely serve as an equation-free surrogate by providing fast and approximate responses for unseen data |cites|. 

However, increasing datasets also create challanges in the entire analysis workflow from processing, extracting, to interpreting the data. This is often described with the "curse of dimensionality". On the other hand, new data often does not provide completely new and uncorrelated information to existing data. One way to handle this contradiction is to understand and parametrize the intrinsic data geometry. This is often of much lower dimension than the ambient data space. Finding a suitable set of coordinated then allows to reduce the dataset to its intrinsic data geometry. We refer to this geometric structure encoded in the data as "manifold". In mathematical terms, a manifold is a topological space that locally is homeomorphic to the Eucledian space. Manifolds arise in many physical applications. In point clouds that instead sample in the entire Eucledian space sample on the restricted manifold object or in dynamical systems where the states evolve on a manifold phase space over time.

In general, all data-driven models pressume some pattern or structure in the available data. Machine learning is then the process of building numerical models adapting to this underyling structure to solve tasks like regression or classification. There are a large number of machine learning models, with a variety of different assumptions about the data structure and applicable in different task context (see e.g. |Bishop|). There are many criteria to distinguish between model families. We like to separate models in whether they include a parametrization of the learnt data manifold. 

The first type of model does not include any manifold parametrization or does so only *implicitly*. This means that the model parameters can adapt to the intrinsic data structre, but it is impossible to directly access or diagnose of what this structure is. A canonical example is the feedforward neural network, which can provide great predictive power as it adapts to general non-linear manifolds, but the trained model is then a black-box in that it does not provide a link between the weights ("neurons") and the underlying data structure. 

In contrast, the second type of model includes an *explicit* parametrization of the (hidden) data manifold. This model type can be profitable for the data analysis because it provides variabilty in how to set up a data-driven model. It allows prior knowledge of a system and it's problem specific domain  to be included, such as (partially) known governing equation terms of a system |pysindy| or the proximity between points in the dataset |cone kernels|. The parametrization of (non-)linear (data) manifolds is often rooted in the rich theory of functional analysis and differential geometry. It therefore often includes a coordinate change of the data into a functional vector basis. Not in all cases does a choice of parametrization provide more insight, in fact it can again come from a black-box model. The key point is that the choice of parametrization is part of a modelling decision. 

In a coordinate basis that parametrizes the manifold it becomes then possible, for example, to find a new parsiomeous data representation to mitigate the burden of high data dimension or to (approximately) linearize the manifold which allows dynamical processes to be composed into its spatio-temporal components |cite|. These aspects are further discussed below. 

# datafold

datafold is a Python package providing **data**-driven models with an explicit mani**fold** parametrization. The software provides a set of model implementation and data structures, which is all integrated in a software architecture with clear modularization (the model API is taken by example from the scikit-learn project |cite|). The software design of datafold addresses the circumstance that models can range from higher level tasks (e.g., system identification |EDMD| or building surrogate models |FD|) to lower level algorithms (e.g. encoding proximity information on a manifold |cite Diffusion Maps|). With datafold we want to address the active research in the field and welcome contributions in the scope of datafold. The software targets students, researchers and experienced practitioners from different fields for dataset analysis. 

In datafold we address data with and without temporal context, which are further discussed in the following. Both types reflect strong resarch branches of data-driven analysis with a manifold context. 

## 1. Point cloud data
!--two kernels are especially well suited if there is a-priori knowledge about distance -- problem specific. 
!-- kernels are essentially extracting non-linear patterns

The first type of data are unordered samples in high-dimensional point clouds. These datasets are often directly connected to the "manifold assumption", which states that the data is assumed to lie on an intrinsic lower dimensional manifold. A model then aims to find a low dimensional parametrization (embedding) of the manifold. In a machine learning context this is also referred to "non-linear unsupervised learning" or shorter "manifold learning". There exist a large variety of manifold learning models and many implementations are included in the scikit-learn package. Often the models are endowed with a kernel which encodes the proximity between data. Examples are the general "kernel principal comonent analysis" (Kernel PCA) |cite|, "Locally Linear Embedding" (LLE) |cite|, or ??? |cite|. datafold provides an efficient implementation of the Diffusion Maps model |cite|. The model includes a sparse kernel representation and besides dimension reduction can approximate the Laplace-Beltrami operator, Fokker-Plank operator or the graph Laplacian. 

An important further issue in many applications that apply manifold learning is to extent the image and/or pre-image mapping between the original and latent space to unseen data |cite|. This mapping is in general non-linear and therefore often requires a separate model that performs this mapping (see for an example analysis of different models |cite Cassimonous, Yannis|). These so-called "out-of-sample" models interpolate general function values defined on a manifold point cloud and, therefore, have to deal with large input data dimensions. In order to extent the function values to manifold to regions in the vicinity of the available data samples, models often follow a multi-scale approach. This allows the function to be extentded to a larger region when the manifold is "simple" (i.e. close to constant) or to further restricted regions where the manifold is "complicated" (i.e. highly non-linear) |cite|. Two examples of out-of-sample methods are the "geometric harmonics interpolation" with multiscale extension |cite| and the the Laplacian Pyramids |cite|, |cite|.

## 2. Time series data
!-- include kernel methods are often utilized for time series prediction 
!-- This study includes many classical algorithm for time series forecasting https://doi.org/10.1080/07474938.2010.481556
!-- i.i.d. assumption does not hold for time series data. 
!-- include also functional analysis as background theory
!-- Koopman: features of the statistics and the dynamics are more easily separated and studied independently. 
!-- Koopman encapsulates the dynamics in a matrix --> allows  linear algebra machinery. 
!-- find fields where Koopman has been applied?

datafold also adressed data with temporal context and sampled from a dynamical system. In this case a data-driven model aims to fit and generalize the underlying dynamics, also known as "system identification". The formulation of a dynamical system includes a phase space (i.e. set of possible states) and a rule of how to evolve a given state to a future state. The phase space is then usually assumed to have a manifold structure. The temporal context and inherent order of time series data require a different handling compared to point clouds. However, in order to describe the phase space manifold, models for point cloud data become relevant again and can improve the accuracy to identify a dynamical system from data.

datafold focuses on the Dynamic Mode Decomposition (DMD) |cite|. The DMD lineraly decomposes the available time series data into spatio-temporal components, which then define a linear dynamical system. Many DMD based variants address to the generally non-linear underlying dynamical system. This is usually done by changing the time series coordinates in a pre-step before the DMD is applied |Hankel|, |KernelDMD|, |Higher order DMD|, |...?|. The justification of this workflow is covered by operator theory and functional analysis, specifically the Koopman operator. The connection of DMD to the Koopman operator is explicitly made clear in the generalizing numerical model "Extended Dynamic Mode Decomposition" (E-DMD) |Williams|. In contrast to a non-linear flow operator in a typical dynamical system form, the Koopman operator acts linearly on a functional space (the so-called obserable space). The Koopman view on a dynamical system is exact if the basis of the observable space is infinite dimensional. In practice, the E-DMD approximates the Koopman operator with a matrix, based on a finite set of functionals, the so-called "dictionary". The functional representation of the dictionary defines a change of coordinates, which in an optimal case linearlizes the dynamics. Or in other words, the dictionary contains observable functions that linearlize the dynamical system's phase space and allows the Koopman matrix to describe a non-linear dynamical systems in this functional coordinate system. 

However, finding a good choice of dictionary is comparable to the machine learning task of "model selection". In addition to linearizing the phase space, there are often further reasons that make processing the time series data necessary. In the case of heterogenous data, there is often feature scaling required. Another important issue is that the given time samples are actually only partial observations of the phase space. In this case it is possible to exploit the time ordering with a time delay embedding. In this case a delay embedding can reconstruct a diffeomorphic manifold of the phase space (compare Takens theorem). In summary, obtaining and linearizing the phase space manifold can require a flexible number transformations and therefore requires this flexibility of a software implementation.

# datafold software 

datafold's software architecture consists of three package layers. The three layers encapsualte a workflow hierarchy, from low level data structures and associated objects such as kernels to high level meta models that are intended to solve complex machine learning tasks. The datafold software architecture aims to ensure a high degree of modularity to allow ongoing contribution in the active research of manifold aware data-driven models. 

Each of the layer contains clear distinguished classes, where the base classes of each class show it's purpose and scope. The model's  application interface was taken by example from the scikit-learn |cite|. All data-driven models integrate with base classes from scikit-learn's API or in the case of time series data generalize the API in a conformant (Python's duck typing) fashion. The models and data structures on each layer can therefore also be used on their own. Dependencies between the layers are unidirectional, where models can require functionality of the same or or lower levels.

datafold integrates with other projects of the Python's scientific computing stack. The major projects are Pandas |cite|, NumPy |cite| and scipy |cite|. This integration makes it easy to reuse well tested and widely used algorithms and data structures and provides a familiar handling to new datafold users that are already familiar with Python's scientific computing stack. 

The high degree of modularization in datafold's software structure ensures a high flexibility of setting up data processing pipeline. As highlighted for the E-DMD model, there is a need for high-level models to have a flexible pipeline of processing with different models. In a data-driven environment it is also eminent to explore quickly test model configurations and test their quality in a parameter space. 

# Example 

## Digit dataset 

cmp. to scikit-learn classification

## Time series 

## 1. Layer: pcfold

The first and lowest layer `datafold.pcfold` provides data structures, objects directly associated to the data and fundamental algorithms on data. There are two data structure provided by datafold which reflect the described data types of point cloud and time series data. The point cloud data on a manifold `PCManifold` is a subclass of NumPy's `ndarray`. It can therefore be used in-place but has guarantees of the format sample orientation and an associated kernel, which describes the proximity between points. 

 `PCManifold`s main purpose is to be used internally in kernel based methods. This allows to encapsualte the complexity and reocurring routines of kernel methods in a data structure. Likewise the kernels and distance matrix computations reside on this layer. With a distance cut off value it is possible to compute sparse kernel or distance matrices, and therefore scale with larger datasets. 
 
The second data structure in datafold manages collections of time series data `TSCDataFrame` and subclasses Pandas' `DataFrame`.  An indexed data structure is necessary because a matrix cannot capture the start and end of a time series and its time values. With `TSCDataFrame` it is possible to directly connect the temporal information to one or many sampled time series. The data structure is  required for system identfication models. The data structure provides many functions that allow to validate model assumptions, which includes evenly time samples, equal time values for all time series, and many more. Furthermore, there is more functionality required for machine learning around `TSCDataFrame` on this layer, such as time series splits into training and test sets and measuring error metrics betwen predicted and true time series.

## 2. Layer: dynfold

The second layer `datafold.dynfold` consists of data-driven models to deal with data manifold structure. The naming scheme indicates it's purpose of extracting dynamics. However, as stated before, the models can also be used on their own and can therefore also allow data without any temporal context. 

scikit-learn already provides a number of manifold learning algorithms. In scientfic works, however, the widely used algorithm "Diffusion Maps" is missing. datafold provides an implementation that through the internal use of `PCManifold` can scale to larger dataset with sparse kernel matrices. Furthermore, diffusion maps can for different purposes. Firstly, it can be used for manifold learning as a way to reduce the dimensionality on a dataset. Secondly, it approximates a functional basis of the Laplace-Beltrami operator, Fokker-Plank operator or graph Laplacian  (depending on it's configuration). This functional basis can build an usefule dictionary in an E-DMD dictionary. On this layer is a set of classes specifically for time series data (indicated with the prefix `TSC` in the classname). This includes time delay embedding `TSCTakensEmbedding` but also scikit-learn wrappers like `TSCPrincipalComponent` which allow for time series data in input and output. 

## 3. Layer: appfold

The highest and third layer `datafold.appfold` includes models that capture complex processing pipelines, which often includes multiple models. This type of model can be described as "meta model" as they allow to give a single point of access to usually multiple submodels. 

A contribution of datafold is an model implementation of E-DMD. Creating an instance of `EDMD` allows to set a flexible choice of dictionary and a DMD based model to compute the  


 where setting the dictionary allows a flexible number of transformation together with a DMD type model to be set. These models are at the end of the Machine Learning pipeline and are intended to solve complex application or analysis tasks. Furthermore, the meta model approach in this layer makes it easier to perform parameter optimization over the parameter space distributed in the single submodels. In the case of E-DMD, the class `EDMDCV` provides an exhaustive search over a user specifed parameter space, including cross-validation splitting of time series data.



# Acknowledgements

Daniel Lehmberg (DL) is supported by the German Research Foundation (DFG), grant no. KO 5257/3-1. DL thanks the research office (FORWIN) of Munich University of Applied Sciences and CeDoSIA of TUM Graduate School at the Technical University of Munich for their support.
