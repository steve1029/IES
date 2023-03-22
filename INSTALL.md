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
First, check if Nivdia GPU is identified.
```
$ sudo lshw -c display
```

Follow the following example command. The version of the Nvidia driver depends on the GPU, the Ubuntu distribution and the version of CuPy. Here, we choose the driver version 450.

```Shell
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt update
$ sudo apt install ubuntu-drivers-common
$ ubuntu-drivers devices
== /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0 ==
modalias : pci:v000010DEd00001E02sv000010DEsd000012A3bc03sc00i00
vendor   : NVIDIA Corporation
model    : TU102 [TITAN RTX]
driver   : nvidia-driver-440 - distro non-free
driver   : nvidia-driver-418-server - distro non-free
driver   : nvidia-driver-440-server - distro non-free
driver   : nvidia-driver-435 - distro non-free
driver   : nvidia-driver-450 - third-party free recommended
driver   : xserver-xorg-video-nouveau - distro free builtin
$ sudo apt install nvidia-driver-450
$ sudo reboot
```

#### Ubuntu packages

#### Python packages
