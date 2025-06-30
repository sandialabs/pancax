[![book](https://img.shields.io/badge/pancax-Book-blue?logo=mdbook&logoColor=000000)](https://sandialabs.github.io/pancax)
[![Build Status](https://github.com/sandialabs/pancax/workflows/CI/badge.svg)](https://github.com/sandialabs/pancax/actions?query=workflow%3ACI)
[![Coverage](https://codecov.io/gh/sandialabs/pancax/branch/main/graph/badge.svg)](https://codecov.io/gh/sandialabs/pancax)
[![PyPI version](https://badge.fury.io/py/pancax.svg)](https://pypi.org/project/pancax/)

# Pancax ("PANCAKES")
Physics augmented neural computations in jax

## Table of Contents
1. [Known Issues](#known_issues)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [Citation](#citation)


## Known Issues
[-] If running the dev branch (which is the most up to date) there is a known incompatability
between jax 0.6.1 and the cuda open blas version that is installed. 

See the following github [issue](https://github.com/jax-ml/jax/issues/29042).
```
pip install nvidia-cublas-cu12==12.9.0.13 --upgrade
```


## Installation
### CPU installation instructions
To install pancax using pip (recommended) for CPU usage you can type the following command

``pip install pancax[cpu]``

### GPU installation instructions
#### CUDA installation instructions
To install pancax using pip (recommended) for CPU usage you can type the following command

``pip install pancax[cuda]``

#### ROCm Installation instructions
To use pancax on amd gpus, matters are slightly more complicated due to the amd gpus being supported in an experimental status for jax. This requires using a docker container. The necessary commands to achieve this are the following
```
docker run -it -d --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 64G \
--group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $(pwd):/home/temp_user/pancax \
--name rocm_jax rocm/jax-community:rocm6.2.3-jax0.4.33-py3.12.6 /bin/bash

docker attach rocm_jax
```
then pancax can be installed with the following command within the container

``pip install pancax[rocm]``

This currently needs to be done each time you start the container listed above. We have plans to ship our own containers in the future to simplify matters.

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

## Contributing
If you would like to contribute to the project, please open a pull request with small changes. If you would like to see big changes in the source code, please open an issue or discussion so we can start a conversation.

## Citation
If you leverage these tools for your own research, please cite the following article
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
