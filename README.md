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
Doctorate in Physics, in Feb 2022, received from the Department of Physics, Korea University.
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
#### Run with single node
```
$ python3 examples/<example.py>
```
#### Parallel computing
Run the code in the bash shell as the following command.
```
$ mpirun -host <host1>,<host2>,...,<hostn> python3 examples/<example.py>
```
#### Reflectance / Transmittance calculation
The reflectance and transmittance of one-dimenstional slabs can be obtained by running `RT_simple_slabs.py`. 
The file takes 5 additional system arguments: <method> <Total time steps> <Nx> <Ny> <Nz>.
```
$ python3 RT_simple_slab.py FDTD 1000 100 100 100
```
#### Scattering Analysis
#### Band structure calculation

# Installation Guide

### Installation on Debian/Ubuntu
Installation on Debian/Ubuntu system is straighforward. Download all the files in your home directory `~/SHPF/` and follow the instructions in **Basic Usage**.

### Installation on Windows
Unfortunately, installation on Windows is currently not available.

### Building a COW
Please see `how_to_build_cow.md`

### Running on a docker container
For those who want to use this package on a single computer, i.e. without MPI, 
running with the nvidia container would be the most convenient way to run the SHPF package.
To use nvidia container, there are some pre-requisites before you get started.
For more details, see [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

#### Pre-requisistes
1. GNU/Linux x86_64 with kernel version > 3.10
1. Docker >= 19.03 (recommended, but some distributions may include older versions of Docker. The minimum supported version is 1.12)
1. NVIDIA GPU with Architecture >= Kepler (or compute capability 3.0)
1. NVIDIA Linux drivers >= 418.81.07 (Note that older driver releases or branches are unsupported.)

#### Summarized procedure to install nvidia-container-toolkit
1. Intall ubuntu on a PC and acquire root authentication.
   ```bash
   $ su passwd root
   $ su root
   ```
1. Check the GPU card and driver at _configuration_.
   ```bash
   # lshw -c display
   ```
1. If the Nvidia GPU card is identified, check the list of available nvidia driver.
   ```bash
   # apt update
   # apt install ubuntu-drivers-common
   # ubuntu-drivers devices
   == /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0 ==
   modalias : pci:v000010DEd00001C02sv000019DAsd00002438bc03sc00i00
   vendor   : NVIDIA Corporation
   model    : GP106 [GeForce GTX 1060 3GB]
   driver   : nvidia-driver-525-server - distro non-free
   driver   : nvidia-driver-450-server - distro non-free
   driver   : nvidia-driver-515-server - distro non-free
   driver   : nvidia-driver-470-server - distro non-free
   driver   : nvidia-driver-520 - third-party non-free
   driver   : nvidia-driver-470 - distro non-free
   driver   : nvidia-driver-510 - distro non-free recommended
   driver   : nvidia-driver-530 - third-party non-free
   driver   : nvidia-driver-515 - third-party non-free
   driver   : nvidia-driver-390 - distro non-free
   driver   : nvidia-driver-418-server - distro non-free
   driver   : nvidia-driver-525 - third-party non-free
   driver   : xserver-xorg-video-nouveau - distro free builtin
   ```
1. Install nvidia driver. We recommend to install server version.
   ```bash
   apt install nvidia-driver-525-server
   ```
1. Install docker and Git.
   ```bash
   # apt update
   # apt install docker git
   ```
1. Setting up Docker.
   ```bash
   # curl https://get.docker.com | sh && sudo systemctl --now enable docker
   ```
1. Setup a package repository, add a GPG key, and install nvidia-container-toolkit.
   ```bash
   # distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   # curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   # curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   # apt update && apt install -y nvidia-container-toolkit
   # systemctl restart docker
   ```
1. Configure the Docker daemon to recognize the NVIDIA container Runtime.
   ```bash
   nvidia-ctk runtime configure --runtime=docker
   ```
1. Restart the docker daemon.
   ```bash
   systemctl restart docker
   ```
1. Root authentication is not necessary from now. Test the setup by running a bash CUDA container.
   ```bash
   # exit()
   $ docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```
1. Download and run the cupy image. Note that cupy/cupy:latest image uses CUDA toolkit v11.7.
   ```bash
   $ docker run --gpus all -it --name cupy cupy/cupy:latest /bin/bash
   ```
   A docker image for cupy can be found [here](https://hub.docker.com/r/cupy/cupy/).
   
1. Install the rest of the python packages as a root of the container.
   ```bash
   # apt update
   # apt install python3-matplotlib python3-numpy python3-mpi4py python3-scipy python3-h5py ipython3 python3-pandas python3-pip
   ```
1. Commit the container as an image.
   ```bash
   # docker commit -m "The python packages installed." -a $USER cupy cupy:python
   ```

For more details, see the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to install Nvidia-container toolkit on a PC

#### Trouble shooting
* If you get an error when updating apt after adding a GPG key, like, do the following. [Ref](https://github.com/NVIDIA/nvidia-docker/issues/1238).
   ```bash
   E: Conflicting values set for option Signed-By regarding source https://nvidia.github.io/libnvidia-container/stable/ubuntu18.04/amd64/ /: /usr/share  /keyrings/nvidia-container-toolkit-keyring.gpg != 
   E: The list of sources could not be read.
   ```
   do the following.
   ```bash
   # rm /etc/apt/sources.list.d/nvidia-container-toolkit.list
   ```
   Then, follow the installation guide from _install_nvidia-container-toolkit_.
   
A full guide for users who want to run this package on a docker container will be update continuously.
