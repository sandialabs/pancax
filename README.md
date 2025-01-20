[![book](https://img.shields.io/badge/pancax-Book-blue?logo=mdbook&logoColor=000000)](https://sandialabs.github.io/pancax)
[![Build Status](https://github.com/sandialabs/pancax/workflows/CI/badge.svg)](https://github.com/sandialabs/pancax/actions?query=workflow%3ACI)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sandialabs.github.io/pancax/) 
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sandialabs.github.io/pancax/dev/) 
[![Coverage](https://codecov.io/gh/sandialabs/pancax/branch/main/graph/badge.svg)](https://codecov.io/gh/sandialabs/pancax)
[![PyPI version](https://badge.fury.io/py/pancax.svg)](https://pypi.org/project/pancax/)

# Pancax ("PANCAKES")
Physics augmented neural computations in jax

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Citation](#citation)

## Installation
### CPU installation instructions
To install pancax using pip (recommended) for CPU usage you can type the following command

``pip install pancax[cpu]``

### GPU installation instructions
Currently only CUDA has been tested, so only a CUDA option is supplied.
#### CUDA installation instructions
To install pancax using pip (recommended) for CPU usage you can type the following command

``pip install pancax[cuda]``

### Developer installation instructions
If you would like to do development in pancax, please first clone the repo and in the pancax 
folder, run the following command

``pip install -e .[cuda,docs,test]``

## Usage
Currently the main entry point to pancax is through a python script (although a yaml input file is also in the works).
To run a script you can run the following command

``python -m pancax -i my_script.py``

where ``my_script.py`` is the name of the scipt you've written. This will run the python script while also 
respecting several environment variables which can be supplied after the ``pancax`` keyword above. A list of
these can be displayed with the help message

``python -m pancax -h``

![Pancax](https://github.com/sandialabs/pancax/blob/main/assets/pancax.png?raw=true)

If you leverage these tools for your own research, please cite the following article

## Citation
```bibtex
@article{hamel2023calibrating,
  title={Calibrating constitutive models with full-field data via physics informed neural networks},
  author={Hamel, Craig M and Long, Kevin N and Kramer, Sharlotte LB},
  journal={Strain},
  volume={59},
  number={2},
  pages={e12431},
  year={2023},
  publisher={Wiley Online Library}
}
```
SCR #3050.0
