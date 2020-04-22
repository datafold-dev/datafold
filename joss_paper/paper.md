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
title: 'datafold: extract manifold structure in time series and point cloud data'
tags:
  - Python
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

The package datafold consists of data-driven algorithms with manifold assumption
. The often implicit assumption of many machine learning algorithms states that
 available high-dimensional data lies on a manifold with intrinsic lower-dimension. 
 
The software architecture has three layers
 
 * pcfold: data structures TSCDataFrame and PCManifold (embedded with kernel)
 * dynfold: algorithms to extract dynamics from time series or parametrize underlying
  geometrical structures from data
 * appfold: algorithms characterized by combining multiple algorithms   
 
* [ ]  clear description of the high-level functionality and purpose of the software for a diverse, non-specialist audience been provided?

High-level functionality: 
- build data-driven non-parametric models from data (time series or static). This
  covers the entire machine learning pipeline. From low-level data structures (time
  -series), to manifold learning algorithms and high-level algorithms to extract dynamics
  from data. Furthermore: model selection with cross validation (incl. time series)
- available time series data can be high-dimensional and single or collection.
- geometrical perspective
- from operator-theory
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

pandas: TSCDataFrame subclasses from pandas DataFrame 
numpy: PCManifold subclasses from np.ndarray
scikit-learn:
  - Follow API of building data-driven models 
  - Uses many internals and BaseEstimator
  - Generalizes to time series data structure by wrapping 
 
Similar projects (and separation of scope):
 - pydmd
    - does implement variants of the DMD (the dictionary handling has to be done by user)
    - supports only single time series (not multiple)
    - follows also the scikit-learn API
    - a wrapper is used to have access to the DMD models (with the above stated
     restriction) 
 
  - pysindy
    - specializes on the SINDy (sparse regression) method
    - supports also 
  
  - forked code from this project https://github.com/jmbr/diffusion-maps/tree/master
  /diffusion_maps 
    - changed to scikit-learn API
    - improved performance speed
    - testing against forked code
  
* [ ] Quality of writing: Is the paper well written (i.e., it does not require editing for structure, language, or writing quality)?

* [ ] References: Is the list of references complete, and is everything cited appropriately that should be cited (e.g., papers, datasets, software)? Do references in the text use the proper citation syntax?

# Statement of Need
* [ ] Do the authors clearly state what problems the software is designed to solve and who
 the target audience is?

Software is designed to learn models from spatial-temporal data with manifold
 assumption and algorithms
 based on operator-theory. 
 
Target audience: data scientists dealing with temporal data. 


# Examples 

* [ ] Clear up DMDbook example
* [ ] move limit cycle to tutorial
* [ ] move method_examples (Yannis group) to tutorials
* [ ] Make "example", "tutorial", "shocase"? 
* [ ] How to manage data? Keep separate from Repo 


# A list of key references, including to other software addressing related needs.

* [ ] pysindy
compare to EDMD implementation 
* [ ] pydmd


# Acknowledgements

Daniel Lehmberg 
is supported by the German Research Foundation (DFG), grant no. KO 5257/3-1. DL thanks
 the research office (FORWIN) of Munich University of Applied Sciences and CeDoSIA of
  TUM Graduate School at the Technical University of Munich for their support.

# References