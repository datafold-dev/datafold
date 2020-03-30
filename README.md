<!---
see gitlab documentation for supported markdown features: 
https://docs.gitlab.com/ee/user/markdown.html
-->

[[_TOC_]]

**The project is under heavy development.**:
* Code quality varies ranging from "experimental" to "well-tested". If using
 experimental code warning are given. 
* There is no backwards compatibility, yet. 
* The API may change without warning or deprecation cycle.


## What is **datafold**?

**datafold** is a Python package consisting of **data**-driven algorithms with 
mani**fold** assumption. This means that **datafold** can aims to process high
-dimensional data that lies on an unknown manifold with intrinsic lower-dimension. One 
major objective of **datafold** is to build build non-parametric models of available time 
series data.

The source code is distributed under the [MIT](LICENSE) license. 

---

## Main features

#### Highlights
* Data structures to handle point clouds on manifolds and collections of
 series data. 
* Provides various distance algorithms and kernels (sparse/dense).  
* Efficient implementation of Diffusion Map algorithm, to parametrize a manifold
from data or to approximate the Laplace-Beltrami operator.
* Out-of-sample methods: (auto-tuned) Laplacian Pyramids or Geometric Harmonics
* (Extended-) Dynamic Mode Decomposition (e.g. `DMDFull` and `EDMD`) which uses the
 introduced data structures to transform time series data with a dictionary (e.g. scaling, 
 delay embedding, ...) and to extract the dynamics via the Koopman operator
 approximation. Furthermore, `EDMDCV`  allows to optimize the parameters of the EDMD
 model with time series cross validation methods.   

---

## Getting started and documentation

The software documentation is available at\
https://datafold-dev.gitlab.io/datafold 

Tutrials are available 
[here](https://gitlab.com/datafold-dev/datafold/-/tree/master/tutorials).

---
## How to get it?

#### From PyPI (NOTE: project not public yet)

datafold is also hosted on the official Python package index 
(https://pypi.org/project/datafold/) and can be installed with (requires: 
[pip](https://pip.pypa.io/en/stable/)) 

```bash
pip install datafold   
```

#### From source

(requires: [git](https://git-scm.com/) and 
[`setuptools`](https://pypi.org/project/setuptools/))

1. Clone the repository

```bash
git clone git@gitlab.com:datafold-dev/datafold.git
```

2. Install datafold by executing `setup.py` in the root folder   

```bash
python setup.py install
```


---

## Dependencies

**datafold** requires Python>=3.6 

The dependencies are managed in 
[setup.py](https://gitlab.com/datafold-dev/datafold/-/blob/master/setup.py) and install
(if not present) with the package manager (see next section).
 
**datafold** integrates tightly into widely used packaged from the [Python scientific
 computing stack](https://www.scipy.org/about.html). The packages involved are mainly

* [NumPy](https://numpy.org/)
  * `PCManifold` subclasses from NumPy's `ndarray` to represent point clouds lying on a
 manifold. For this to every array a kernel is attached. 
* [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html)
  * `TSCDataFrame` subclasses from pandas' `DataFrame` to structure a collection of time
 series with a pandas 
[DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html). 
  
* [scikit-learn](https://scikit-learn.org/stable/)
  * All **datafold** algorithms that part of the "machine learning 
pipeline" subclass from the  `BaseEstimator`. Generally, the philosopy of scikit
-learn is followed where ever possible. For required generalizations (such as dealing
with `TSCDataFrame` as input and output) own base classes are provided in **datafold**.


For developer dependencies, which also enable running run unit tests and building the
documentation, see [contributing section](#Contributing). 

## Contributing

#### Bug reports and user questions 

For all correspondence regarding the software please open a new issue in the
 datafold [issue tracker](https://gitlab.com/datafold-dev/datafold/-/issues) 

All code contributors are listed in the [contributor list](CONTRIBUTORS). 

#### Setting up development environment

##### Install dependencies

In the file `requirements-dev.txt` all developing dependencies are listed. Install the
 dependencies with Python's package manager

```bash
pip install -r requirements-dev.txt
``` 

The recommended way is to install all packages into a  
[virtual environment](https://virtualenv.pypa.io/en/stable/) such that the dependencies
are not conflicting with other packages.

##### Install git pre-commit hooks 

The **datafold** source code is automatically formatted with
* [black](https://black.readthedocs.io/en/stable/) for auto formatting
* [isort](https://timothycrosley.github.io/isort/) for sorting `import` statements
 alphabetrically and into sections.
* [nbstripout](https://github.com/kynan/nbstripout) for removing potentially large (in
 mega bytes) and
 binary formatted output cells of Jupyter notebooks before it is in the git history.  
 
It is highly recommended that the code is already formatted *before* it is commited to
the git history. For this the most convenient way is to setup git commit-hooks via the 
tool [pre-commit](https://pre-commit.com/) (it installs with the development dependencies). 
To install the hooks run from the root directory  
 
```bash
pre-commit install
``` 

The installed hooks run before each commit. To also execute the hooks without a commit or
for testing purposes) run from the root directory
 
```bash
pre-commit run --all-files
```

#### Run tests
The tests are executed with [nose](https://nose.readthedocs.io/en/latest/) (installs
with the development dependencies). 
 
To execute all **datafold** unit tests locally run

```bash
nosetests datafold/ -v
```

To execute the tutorials (only error checks) run:

```bash
nosetests turorials/ -v
```

All tests (unit and tutorial) are executed remotely in a "Continuous Integration" (CI
) setup for every push to the [remote repository](https://gitlab.com/datafold-dev/datafold).

#### Compile documentation

The documentation uses documentation generator [Sphinx](https://www.sphinx-doc.org/en
/stable/) (installs with the development dependencies), with the [read-the-docs theme](https://sphinx-rtd-theme.readthedocs.io/en/stable/)

To build the documentation run from root

```bash
sphinx-apidoc -f -o ./doc/source/_apidoc/ ./datafold/
sphinx-build -b html ./doc/source/ ./public/
```

The documentation in html is located in `public/index.html`.

---

## Affiliation

* Daniel Lehmberg
 is supported by the German Research Foundation (DFG), grant no. KO 5257/3-1. DL thanks
 the research office (FORWIN) of Munich University of Applied Sciences and CeDoSIA of
 TUM Graduate School at the Technical University of Munich for their support.