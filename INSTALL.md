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
First, get root authentication. Then, check if Nivdia GPU is identified.
```
# lspci | grep NVIDIA
01:00.0 VGA compatible controller: NVIDIA Corporation GP107 [GeForce GTX 1050 Ti] (rev a1)
01:00.1 Audio device: NVIDIA Corporation GP107GL High Definition Audio Controller (rev a1)
```

Next, install Nvidia driver using the following command. 
The version of the Nvidia driver depends on the GPU, the Ubuntu distribution and the version of CuPy. 
Here, we choose the driver version 450.

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
