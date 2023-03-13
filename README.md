# SHPF_GPU

The numerical solver of Maxwell's equations.
It provides three simulation methods: 

* the Finite-Difference Time-Domain (FDTD) method, 
* the Pseudo-Spectral Time-Domain (PSTD) method 
* the Staggered-grid Hybrid PSTD-FDTD (SHPF) method.

A user can choose to run the program with CPU or GPU, if one has a GPU manufactured by Nvidia.
It also provides two parallel computing methodology, the distributed memory system and shared memory system.
To run this package with its full capability, one should use Linux based COW (cluster of workstation).

Please see `INSTALL.md` for installation.

## Author
A Ph.D in Physics, received from Department of Physics, Korea University.
Currently working at LG innotek.

[**Google Scholar**](https://scholar.google.com/citations?user=iYm5ThEAAAAJ&hl=ko)

[**Paper**](https://doi.org/10.1016/j.cpc.2020.107631)

[**Curriculum Vitae**](/CV.pdf)

## Implements
### Numerical solvers
* **FDTD** Uses Finite-Difference method to approximate the derivatives in Maxwell's equations.
* **PSTD** Uses pseudo-spectral methods to approximate the spatial derivatives in Maxwell's equations.
* **SHPF** By hybridizing the PSTD and FDTD method, it is optimized for large-scale electromagnetic simulations using COW. 

### Parallelism
* **OpenMP**
* **OpenMPI** Distributed Parallel Computing with OpenMPI(wrapped with mpi4py) is implemented.
* **GPU** The simulation can be run with numpy or cupy.

### Materials
So far, only dielectric materials can be modeled.

### CPML
Convolutional PML is implemented.

## harminv
A package for analyzing the dominant frequency component for a given signal.
The band structure of the photonic crystal can be obtained.

## pycuda_prac
It contains the codes for testing the pycuda package, to determine the future usage for my research.
At first, I was intended to use pycuda.
I started to practice how to use pycuda to accelerate the simulation.
Then later I found 'cupy', a more readable and convenient package for using the GPGPU in python environment.
So I transferred from pycuda to cupy.
So this directory is a remnant of my earlier research and has been abandoned now.
