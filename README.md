# Introduction
[![CI](https://github.com/steve1029/SHPF/actions/workflows/blank.yml/badge.svg)](https://github.com/steve1029/SHPF/actions/workflows/blank.yml)

The numerical solver of Maxwell's equations.
It provides three simulation methods: 

* the Finite-Difference Time-Domain (FDTD) method, 
* the Pseudo-Spectral Time-Domain (PSTD) method 
* the Staggered-grid Hybrid PSTD-FDTD (SHPF) method.

A user can choose to run the program with CPU or GPU, if one has a GPU manufactured by Nvidia.
It also provides two parallel computing methodology, the distributed memory system and shared memory system.
To run this package with full capability, one should use Linux-based COW (cluster of the workstation) where each node has an Nvidia GPU.

## Author
A Ph.D in Physics, received from Department of Physics, Korea University.
Currently working at LG innotek.

[**Google Scholar**](https://scholar.google.com/citations?user=iYm5ThEAAAAJ&hl=ko)

[**Paper**](https://doi.org/10.1016/j.cpc.2020.107631)

Lee, D., Kim, T., & Park, Q. H. (2021). Performance analysis of parallelized PSTD-FDTD method for large-scale electromagnetic simulation. Computer Physics Communications, 259, 107631.

[**Curriculum Vitae**](/CV.pdf)

## Features
#### Numerical solvers
- **FDTD**: Uses Finite-Difference method to approximate the derivatives in Maxwell's equations.
- **PSTD**: Uses pseudo-spectral methods to approximate the spatial derivatives in Maxwell's equations.
- **SHPF**: By hybridizing the PSTD and FDTD method, Staggered grid Hybrid PSTD-FDTD(SHPF) method is optimized for large-scale electromagnetic simulations using COW. 

#### Parallelism
* **OpenMP**: Shared memory parallelism is provided.
* **OpenMPI**: Distributed Parallel Computing with OpenMPI(wrapped with mpi4py) is implemented.

#### Computing Devices
* **CPU**: If a user wants to run with CPU, choose core engine as 'numpy'.
* **GPU**: If a user wants to run with GPU, choose core engine as 'cupy'.

#### Materials
* So far, only dielectric materials can be modeled.

#### Boundary Conditions
* PBC: Periodic Boundary Conditions.
* BBC: Bloch Boundary Conditions.
* PML: Convolutional PML (CPML) is implemented.

#### Sources
* Dipole source
* Plain wave source
* Gaussian source

## Requirements
* Debian/Ubuntu
* COW (not necessary but highly recommanded.)
* OpenMPI, OpenMP
* Nvidia toolkits
* SSH Login without password
* Python3
* Python3 libraries
    * Matplotlib
    * Scipy
    * Numpy
    * Cupy
    * h5py: $\varepsilon$ and $\mu$ are provided with h5 format.
    * mpi4py: Python wrapper for OpenMPI.
    * pharminv: A package for analyzing the dominant frequency component for a given signal. The band structure of the photonic crystal can be obtained.

## Installation
Please see `INSTALL.md` for installation.

## Basic Usage
#### Parallel computing
Run the code in the bash shell as the following command.
```Shell
$ mpirun -host <host1>,<host2>,...,<hostn> python3 examples/<example.py>
```
#### Reflectance / Transmittance calculation
#### Scattering Analysis
#### Band structure calculation
