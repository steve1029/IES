# Introduction
[![CI](https://github.com/steve1029/SHPF/actions/workflows/blank.yml/badge.svg)](https://github.com/steve1029/SHPF/actions/workflows/blank.yml)

The numerical solver of Maxwell's equations.
It provides three simulation methods: 

* the Finite-Difference Time-Domain (FDTD) method, 
* the Pseudo-Spectral Time-Domain (PSTD) method 
* the Staggered-grid Hybrid PSTD-FDTD (SHPF) method.

A user can choose to run the program with CPU or GPU, if one has a GPU manufactured by Nvidia.
It also provides two parallel computing methodology, the distributed memory parallelism and shared memory parallelism.
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
* **SMP**: Shared Memory Parallelism (SMP) using OpenMP is provided.
* **MPI**: Message Passing Interface (MPI) parallel programming with OpenMPI(wrapped with mpi4py) is provided.

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

# Basic Usage
#### Parallel computing
Run the code in the bash shell as the following command.
```
$ mpirun -host <host1>,<host2>,...,<hostn> python3 examples/<example.py>
```
#### Reflectance / Transmittance calculation
#### Scattering Analysis
#### Band structure calculation

# Installation Guide

## Installation on Debian/Ubuntu

## Installation on Windows
Unfortunately, installation on Windows is currently not available.

## Building a Ubuntu COW (Cluster of Workstation)
For users who want to build a cluster for high-performance computing from scratch, I introduce here how to build an Ubuntu Cluster using desktops.

### What is COW?
COW is the acronym of **_cluster of workstation_**. A workstation is a single computer equipped with high-performance hardwares.
Hardware for workstations usually involves additional functions that are not included in the hardware for ordinary uses.

### Ubuntu installation
1. Download the ISO image file from [Ubuntu download](https://ubuntu.com/download/server#download). We highly recommend a server image because, without GUI, a COW is more stable and nimble.
1. Install the Ubuntu using USB or other methods. During the setup, choose to install OpenSSH for convenience.
2. Acquire root authentication right after the installation is finished.
3. Repeat the aboves for each node.

### Installation options for Ubuntu

### Installing the required packages

#### Nvidia Driver
We will show you how to install Nvidia driver, CUDA and CuPy using Ubuntu 18.04 and GeForce GTX 1050 Ti.
This procedure can be applied to other GPUs and driver versions.

First, get root authentication for convenience. Then, check if a Nivdia GPU is identified.
```
$ su root
# lspci | grep NVIDIA
01:00.0 VGA compatible controller: NVIDIA Corporation GP107 [GeForce GTX 1050 Ti] (rev a1)
01:00.1 Audio device: NVIDIA Corporation GP107GL High Definition Audio Controller (rev a1)
```

Next, install Nvidia driver using the following command. 


```bash
# apt update
# apt install ubuntu-drivers-common
# ubuntu-drivers devices
== /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0 ==
modalias : pci:v000010DEd00001C82sv00001043sd000085CDbc03sc00i00
vendor   : NVIDIA Corporation
model    : GP107 [GeForce GTX 1050 Ti]
driver   : nvidia-driver-470 - distro non-free recommended
driver   : nvidia-driver-470-server - distro non-free
driver   : nvidia-driver-450-server - distro non-free
driver   : nvidia-driver-460 - distro non-free
driver   : nvidia-driver-390 - distro non-free
driver   : nvidia-driver-418-server - distro non-free
driver   : nvidia-driver-495 - distro non-free
driver   : nvidia-driver-460-server - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin
```
The version of the Nvidia driver depends on the GPU, the Ubuntu distribution and the version of CuPy. 
Here, we choose the driver version 450.
```
# apt install nvidia-driver-460-server
# reboot
```
Check if the installation is completed.
```
# nvidia-smi
```
If you installed the non-server version, then Gnome, the GUI of Ubuntu is automatically installed.
The sleep mode of Gnome is enabled by default. If you want to get rid of this, type the following command.

```
# systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```
If you want to remove Nvidia driver, use the following command.
```
# apt-get remove --purge 'nvidia-.<version>' 
```

#### CUDA
The version of CUDA should be chosen carefully. 
For GPU computing to operate properly, the version of CUDA must be compatible with the Nvidia driver, installed GPU and the current version of CuPy.
One might be aware that the CUDA toolkit installer includes Nvidia drivers.
However, it is recommended to install the Nvidia driver first, then install CUDA separately.
This is because the Nvidia driver version provided with the CUDA Toolkit installer may not be compatible with the GPU installed on your system.

According to [the official documentation of CuPy](https://docs.cupy.dev/en/stable/install.html), 
as of 03/22/2023, CuPy is compatible with CUDA GPU with compute capability 3.0 or larger and CUDA Toolkit version 10.2 ~ 12.0.
Also, according to [Your GPU Compute Capability](https://developer.nvidia.com/cuda-gpus), GeForce GTX 1050 Ti has to compute compatibility of 6.1.
Lastly, according to [GPUs supported](https://en.wikipedia.org/wiki/CUDA), a GPU of 6.1 compute compatibility is supported by CUDA version >9.0.
Thus, any CUDA with version 10.2 ~ 12.0 is fine with our device.
However, rather than installing the latest version, we recommend installing the slightly former version such as v11.7.
The latest version often provokes a stability issue.

Go to [CUDA download](https://developer.nvidia.com/cuda-toolkit-archive) and get CUDA toolkit __runfile (local)__.
```bash
# wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
# sh cuda_11.7.0_515.43.04_linux.run
```

Put the following command in `~/.bashrc`.
```bash
# vi ~/.bashrc
export PATH=$PATH:/usr/local/cuda-11.3/bin
export CUDADIR=/usr/local/cuda-11.3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib64
```
Check if CUDA is successfully installed.
```bash
# source ~/.bashrc
# nvcc –version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Mar_21_19:15:46_PDT_2021
Cuda compilation tools, release 11.3, V11.3.58
Build cuda_11.3.r11.3/compiler.29745058_0
```
#### Python packages
Install cupy and other Python packages by following command.
```
# apt install python3-matplotlib python3-numpy python3-mpi4py python3-scipy python3-h5py ipython3 python3-pandas python3-pip
# pip3 install cupy-cuda<version>
```

