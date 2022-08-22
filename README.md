# 2nd_paper

A Ph.D student in Department of Physics, Korea University.
This directory contains the programs and codes used in my second paper.

## SHPF.cupy.diel.CPML.MPI
The simulation package for the electromagnetic wave simulation.

### Staggered Hybrid PSTD-FDTD method
The staggered-grid hybrid PSTD-FDTD method adopted for the EM wave simulation.

### cupy
The simulation can be run with numpy and cupy.

### dielectric
Only a dielectric material can be modeled.

### CPML
Convolutional PML is developed.

### MPI
Distributed Parallel Computing with OpenMPI and mpi4py are developed.

## harminv
A package for analyzing the dominant frequency component for a given signal.
The band structure of the photonic crystal can be obtained.

## pycuda_prac
The codes to test the pycuda package for the research.

At first, I was intended to use pycuda.

I started to practice how to use pycuda to accelerate the simulation.

Then later I found 'cupy', a more readable and convenient package for using the GPGPU.

I transferred from pycuda to cupy.

So this directory is a remnant of my earlier research and has been abandoned now.
