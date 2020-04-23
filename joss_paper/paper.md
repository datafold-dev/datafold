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
title: 'datafold: learn manifold from (non-) temporal data'
tags:
  - Python
  - extended dynamic mode decomposition
  - manifold assumption
  - data-driven
  - dynamical systems
  - time series
  
authors:
  - name: Daniel Lehmberg, (TODO: other?)
    orcid: 0000-0002-4012-5014
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
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


# Summary

!-- include: highly active field of manifold learning and Koopman many algorithms etc. Can be inserted in the architecture 
!-- the data may be on multiple scales both spatial and temporaly
!-- model reduction, surrogate models, predictions

Data availability has changed the analysis methods in many scientific fields and applications. While the complex underyling systems such as pyhsical systems remain the same, the number of measurements increases both in quantity and dimension. Data-driven models that learn the underlying geometrical structure, which we refer to manifolds, are a way to make use of this high-dimensional data. Explicitly parametrizing the data manifold and also generalize a model to new unseen data of the same system, allows us to understand the intrinsic structure in latent coordinates. 

In recent years there has been active research for data-driven models that explicitly address manifolds to describe the geometrical structure in the data. This includes a for a variety of tasks, such as model reduction, building surrogate models of simulation software, making predictive models from real world data.


!-- Address distance matrix, continuous etc.
One major direction in the research addresses the task to learn non-linear manifolds from high-dimensional point cloud data. In a Machine learning context this is also referred to "non-linear unsupervised learning" or simply "manifold learning". This is directly connected to the so-called "manifold assumption", which states that a high-dimensional point cloud is assumed to lie on an intrinsic lower dimensional manifold. The aim is to extract a latent and parsinomenous representation from data. Processing latent variable reduces the burden of the "curse of dimensionality" and can increase the accuracy of classification or regression models . Examples are the Diffusion Maps algorithm, [Variable bandwidth, connection to kernels, computing distance matrix]TODO:MORE. 

An important issue in manifold learning is to extent and generalize the mapping (forward and inverse) between the original and latent space to unseen data. This mapping is often non-linear is a started additional sets of methods, such as the Laplacian Pyriamid or the Geometric Harmonic interpolation. These methods try to address the complexity of the manifold, where the extension of "simple" is large and small ing "complicated" regions.

A second strong reasearch direction is to approximate the underlying dynamical systems from time series data, also referred to "system identification". In this case the system's phase space is assumed to lie on a manifold. The additional temporal context and inherent order in time series require a different handling. 

A numerical algorithm with many variants and extensions is the is the Dynamic Mode Decomposition (DMD). The DMD extracts the dynamics from time series data by decomposing the time series data into spatio-temporal components lineraly. This allows to define a linear dynamical system, while many of the extensions address non-linearity by pre-processing the time series data into another representation. The justifying background theory is covered by the mathematical operator theory with specifically the Koopman operator (also composition operator). The connection to the Koopman operator is explicitly made clear in the generalized Extended Dynamic Mode Decomposition (E-DMD). While the exact representation of the Koopman operator acts an infinite dimensional functional space, the E-DMD allows to set up a dictionary, which consists of fintetely many functionals, which in an optimium linearizes the dynamical system's phase space. The "right" choice of dictionary requires a flexible model and for without expert knowledge of the underlying system is comparable to "model selection". 

# Software

!-- Sindy
!-- PyDMD
!-- tensorflow
!-- statmodels

datafold is a Python package providing data-driven models with explicit manifold reference. The software architecture consists of three package layers with each containing clear separated classes. 

The high degree of modularization software structure serves two purposes. The first is to give the software user a high flexibility of setting up data processing pipeline. As highlighted before, models like the E-DMD require a great flexibility to be able to explore a model space. The second purpose addresses the ongoing and active research in this field. The three layers give orientation of different degrees at which models and algorithms reside on the general workflow of extracting hidden manifold information from data. This allows an easier integration of new algorithms. The classes on each package layer can be used on their own, there is a while there is an internal unidirectional layer dependency. The classes on a layer can only make use of the functionality of the same or the lower levels. 

datafold strongly integrates with Python's scientific computing stack. 


The two puposes of model flexibility and re-using clearly separated classes is fostered by a  


datafold integrates stongly with the Python scientific computing stack. The lowest level includes data structures to describe time series collections and point clouds on manifolds. on the lowest level subclass from the widely used data structures from NumPy and Pandas. All data-driven models align to the programming interface of scikit-learn. All models are also able to process the provided data structes  time series data may generalize or restrict the input type to a data frame containing time series collection.


The models contained in datafold align to the scikit-learn interface |cite|.  

 
 
 

  

A second data structure adressed by datafold are time series data. The temporal ordered samples come from a dynamical systems, such as an ordinary differential equation system or sensor measurements. datafold provides numerical models

There are further sources that make transformations of the time series data neccessary. In many applications, the individual time series' measurements do not lie on phase space manifold of underlying dynamical system. In such cases It can be neccesary to exploit the temporal information and perform a time delay embedding, which can reconstruct the phase space and increases the dimension. Further sources are feature scaling if the features are non-homogenous. 

To handle this complexity in a user friendly and robust way, datafold provides an implementation of the Extended Model Decomposition (EDMD). The EDMD model subclasses from the scikit-learn's Pipeline class, where all the transformation define the dictionary and the final estimator a DMD based model. This meta estimator captures all parameters of the dictionary and the DMD model in one model. It therefore gives an easy accessible framework to set up of dictionary functions, optimize the parameters with cross validation. 

datafold aligns with the scikit-learn interface. However, whenever necessary it restricts or generalizes the input data to time series data. 



In many cases the time series samples do notlie directly on the phase space manifold. In such cases we can actually increase the dimension of the time series data by exploiting the temporal ordering (e.g. Takens time delay embedding). The reconstructed phase space can then be  



In contrast to static point cloud data, the single time series' samples may not capture the phase space manifold. 



The aim is then to find a set of so-called observable functions that form a function space in which the manifold phase space linearlizes. The selection of suitable observables is similar to the problem of model selection in Machine Learning.



The aspects of transforming the data, describing the underlying manifold of the point cloud and finally learning the dynamical system is combined into the Extended Dynamic Mode Decomposition. datafold allows to the so-called dictionary, which are all transformation to linearize the manifold, in a flexible and user friendly way.



The models, however, have to ad data is inherently ordered by it's associated temporal context. 


The models imncluded in datafold use the Dynamic Mode Decomposition 

  


 we can find in many applications that the actual data lies on an intrinsic lower dimensional geometric structure, a mathematical manifold.



or have an intrinsic temporal order as in time-series data.  


If the  In the case of time series data we which to learn the manifold of the phase space, which allows us to 


we may only a low-dimensional time series, for example only a single feature. Exploiting the temporal order of the time series we use techniques to increase the dimension    


n many applications In both cases the data can be of high-dimension  the data lies on an unknown geometrical structure (a manifold). This manifold can be of lower dimension than the  actual high dimensional manifold data. 


* both static data and time series data for which there are data structures
* learn a dynamical system from time series data
* the API aligns with the scikit-learn and generalizes to time series data where required
* models that view non-temporal time clouds can be used to  
* Covers the entire machine learning pipeline. From low-level data structures (time
  -series), to manifold learning algorithms and high-level algorithms to extract dynamics
  from data. Furthermore: model selection with cross validation (incl. time series)
- time series models from operator-theory

The software architecture has three layers
 
 * pcfold: data structures TSCDataFrame and PCManifold (embedded with kernel)
 * dynfold: algorithms to extract dynamics from time series or parametrize underlying
  geometrical structures from data
 * appfold: algorithms characterized by combining multiple algorithms   
 
* [ ]  clear description of the high-level functionality and purpose of the software for a diverse, non-specialist audience been provided?

High-level functionality: 

- available time series data can be high-dimensional and single or collection.
- geometrical perspective
- high-dimensional data
- decompose time series data into spatial-temporal using algorithms from operator theory 
- data-driven representation in modes of dynamical systems
- operator theory: representing (non-linear) flows
- high vs. low spatial dimension -> Takens 

* [ ] A statement of need: Do the authors clearly state what problems the software is 
designed to solve and who the target audience is?

Problems to solve: 
- data analysis extract and create reduced representations of coherent dynamical patterns
 hidden in high-dimensional data 
- Construct data-driven models from non-linear, high-dimensional time-series data are
   not covered with manifold assumption assumption. Highlighting the EDMD which has a
- combine different levels of extracting dynamics into one code base (framework)
- EDMD and EDMDCV provide a high-level interface, where a variable number of
transformation (and back-transformations) are handled

Target audience: data scientists 

* [ ] State of the field: Do the authors describe how this software compares to other commonly-used packages?


Similar projects (and separation of scope):
  
* [ ] Quality of writing: Is the paper well written (i.e., it does not require editing for structure, language, or writing quality)?

* [ ] References: Is the list of references complete, and is everything cited appropriately that should be cited (e.g., papers, datasets, software)? Do references in the text use the proper citation syntax?

# Statement of Need
* [ ] Do the authors clearly state what problems the software is designed to solve and who
 the target audience is?

Software is designed to learn models from spatial-temporal data with manifold
 assumption and algorithms
 based on operator-theory. 
 
Target audience: data scientists dealing with temporal data. 

# Other related software


# Examples 

* [ ] move method_examples (Yannis group) to tutorials


# A list of key references, including to other software addressing related needs.

* [ ] pysindy compare to EDMD implementation 
* [ ] pydmd
* []


# Acknowledgements

Daniel Lehmberg 
is supported by the German Research Foundation (DFG), grant no. KO 5257/3-1. DL thanks
 the research office (FORWIN) of Munich University of Applied Sciences and CeDoSIA of
  TUM Graduate School at the Technical University of Munich for their support.

# References