# 2nd_paper

A Ph.D student in Department of Physics, Korea University.
This directory contains the programs and codes used in my second paper.

## SHPF.cupy.diel.CPML.MPI
The simulation package for the electromagnetic wave simulation.

### Staggered Hybrid PSTD-FDTD method
The staggered-grid hybrid PSTD-FDTD method adopted for the EM wave simulation.

### cupy
The simulation can be run with numpy or cupy.

### dielectric
So far, only dielectric materials can be modeled.

### CPML
Convolutional PML is implemented.

### MPI
Distributed Parallel Computing with OpenMPI(wrapped with mpi4py) is implemented.

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
