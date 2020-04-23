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

Increasing data availability has changed the analysis and interpretation of data in many scientific fields and practical applications. While complex underyling systems such as pyhsical systems remain the same, the number of measurements increases both in quantity and dimension. This creates challanges on the entire analysis workflow from processing, extracting, and interpreting the data. In mathematical terms the geometric structure encoded in the data is referred to as manifolds. All data-driven models assume that there is some pattern or manifold structure in a data. 

Machine learning is the process of finding structure and building models to solve application tasks like regression, classification or prediction. Many machine learning models only implicity assume the presence of structure in the data. An example are feedforward neural networks, which provide great predictive power, but at the cost of being a black box model. This type of models make it hard or impossible to analyze the structure. On the other hand, model types that explicitly parameterize the manifold structure and generalize it to regions outside the available data samples, provide a path to uncover information about the intrinsic behaviour of the underlying system. TODO: more here

There is ongoing research for the data-driven analysis of manifold structure, with many models that again have several extensions and adaptations. The model can range from higher level tasks (e.g., prediction |FIND| or building surrogate models |FD|) to lower level tasks, such as defining a kernel to encode locality information on a manifold |Giannakis_cone_kernel|. 

!-- kernels are especially well suited if there is a-priori knowledge about distance -- problem specific. 
!-- kernels are essentially extracting non-linear patterns

One major direction of deals with models that extract the non-linear manifold structure from high-dimensional point cloud data. In the machine learning context this is also referred to "non-linear unsupervised learning" or shorter "manifold learning". Manifold learning is directly connected to the "manifold assumption". It states that a high-dimensional point cloud is assumed to lie on an intrinsic lower dimensional manifold. Extracting the latent and parsinomenous representation of the data reduces the burden of the "curse of dimensionality" and can increase the accuracy of classification or regression models. Often methods of this class are endowed with a kernel, such as the  "kernel principal comonent analysis" (Kernel PCA) |cite|, "Diffusion Maps" |cite|, or "Locally Linear Embedding" (LLE) |cite|. 

An important issue in many applications if applying manifold learning models is to extent the image and/or pre-image mapping between the original and latent space to unseen data. Because this mapping is usually non-linear this required a new set of models. These models interpolate or regression general function values defined on a manifold space. To extent the manifold as far as possible these methods follow often a multi-scale approach, which allows to extent it far away from samples where the manifold is "simple" and restricts the extension in "complicated" regions of larger non-linearity. Two examples of out-of-sample methods are the "geometric harmonics interpolation" with multiscale extension |cite| and the the Laplacian Pyramids |cite|, |cite|.

!-- include kernel methods are often utilized for time series prediction 
!-- This study includes many classical algorithm for time series forecasting https://doi.org/10.1080/07474938.2010.481556
!-- i.i.d. assumption does not hold for time series data. 
!-- include also functional analysis as background theory
!-- Koopman: features of the statistics and the dynamics are more easily separated and studied independently. 
!-- Koopman encapsulates the dynamics in a matrix --> allows  linear algebra machinery. 
!-- find fields where Koopman has been applied?

A second strong research direction is to approximate underlying dynamical systems from collected time series data. This is also reffered to "system identification". The additional temporal context in time series and inherent order require a different handling compared to point clouds with no temporal information. However, because the dynamical system's phase space is assumed to lie on a manifold, the models from point clouds are useful to improve the accuracy to identify a dynamical system from data.

A numerical algorithm with many variants and extensions is the Dynamic Mode Decomposition (DMD) |Schmid|. The DMD lineraly decomposes the time series data into spatio-temporal components, which defines a linear dynamical system. The DMD variants address the general non-linearity of the system by defining a building the linear system on a different basis. This change in coordinates requires a transformation process of the time series data before it is applied to the DMD algorithm. The justification of this workflow is covered by the mathematical operator theory and specifically the Koopman operator. The connection to the Koopman operator is explicitly made clear in the generalizing Extended Dynamic Mode Decomposition (E-DMD) |cite|. In the theoretical exact representation of the dynamical system, the Koopman operator acts on an infinite dimensional function basis (the so-called obserables). In practice, the E-DMD approximates the Koopman operator with a matrix, by setting up a dictionary of a finite set of observables. In an optimal case the dictionary linearizes the dynamical system's phase space. This allows to describe the non-linear dynamical systems with a (linear) Koopman matrix in another basis. However, finding a good choice of dictionary is comparable to the machine learning task of "model selection" and therefore requires great flexibility on the software imlpementation.

# datafold software 

!-- many validation tests!
!-- state models that are implemented somewhere else are used. 

datafold is a Python package providing **data**-driven models with an explicit mani**fold** parametrization. The software targets experienced practitioners and researchers from different fields to analyze complex datasets. Researchers in the scope of datafold are welcome to contribute their work of new numerical models. The datafold software architecture is designed for ongoing contribution. To ensure a high degree of modularity, datafold's software architecture consists of three package layers refering encapsulating a workflow hierarchy, from low level data structures and associated objects as kernels to high level meta models that are intended to solve complex problems. Each of the layers contains clear distinguished classes, where the base classes of each class show it's purpose and scope. datafold also strongly integrates with the Python's scientific computing stack. A major role for the organization of the models was taken by example from scikit-learn |cite|. All datafold models integrate with base classes from scikit-learn's API or in the case of time series data generalize the API in a conformant (duck typing) fashion to Pandas' `DataFrame`. The strong integration used and well tested algorithms and data structures and provides a familiar handling for new datafold users that are already familiar with Python's scientific computing stack. 

The high degree of modularization in datafold's software structure fosters a high flexibility of setting up data processing pipeline. As highlighted for the E-DMD model, it is often different applications and datasets require a different set of transformaitons and be able to explore a model space. Furthermore, there is ongoing and active research in the field of manifold aware data-driven models. datafold aims to provide an architecture and modularization in which code can be reused. The three layers give orientation of different degrees at which models and algorithms reside on the general workflow of extracting hidden manifold information from data. The classes on each package layer can also be used on their own. Dependencies between the layers are only unidirectional, where models can only depend on functionality of the same or or lower levels.

!-- kernels -> sparse matrix

On the lowest layer `datafold.pcfold` provides data structures, objects directly associated to the data such as kernels and unfifying wrapper access to fundamental algorithms. The first data structure `PCManifold` includes a subclass of NumPy's `ndarray`. It represents point clouds on manifolds and associates a kernel with the data to describe locality. `PCManifold`s is mainly used in kernel based methods in which captures complexity such as choice of distance metric computation and sparse kernel matrices. A second data structure `TSCDataFrame`, subclassing Pandas' `DataFrame`, manages collections of time series data. It directly connects temporal information and differently sampled time series. The data structure is also required by models to learn a dynamical system for input. Building up on this data structure this layer also includes classes providing more functionality such as time series splitting into training and test sets, measuring error metrics betwen predicted and true data.

The second layer `datafold.dynfold` consists of data-driven models to extract manifold structure from data. The naming scheme indicates to the purpose of extracting dynamics. However, as stated before, the models shall also be useable on their own and can therefore also be used on data without any temporal context. An example is the `DiffusionMaps` algorithms. The diffusion maps algorithm can serve for different purposes. It can be used for manifold learning to extract a pasimoneous representation, but it also allows to approoximate the Laplacian-Beltrami operator eigenfunctions which can be used as functionals for dynamical data in an E-DMD dictionary. There is also a set of classes dedicated for time series functions (indicated with the prefix `TSC` in the classname). This includes time delay embedding `TSCTakensEmbedding` or represent the data in a different functional coordinates such as the `TSCRadialBasis`. 

On the third and last layer `datafold.appfold` are models that capture multiple models in one model. This type of model can be described as meta model as it allows to give a single point of access to an organization of usually multipled submodels.  This includes the datafold implementation of the E-DMD, where setting the dictionary allows a flexible number of transformation together with a DMD type model to be set. These models are at the end of the Machine Learning pipeline and are intended to solve complex application or analysis tasks. Furthermore, the meta model approach in this layer makes it easier to perform parameter optimization over the parameter space distributed in the single submodels. In the case of E-DMD, the class `EDMDCV` provides an exhaustive search over a user specifed parameter space, including cross-validation splitting of time series data.




# Acknowledgements

Daniel Lehmberg (DL) is supported by the German Research Foundation (DFG), grant no. KO 5257/3-1. DL thanks the research office (FORWIN) of Munich University of Applied Sciences and CeDoSIA of TUM Graduate School at the Technical University of Munich for their support.
