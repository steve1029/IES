#!/usr/bin/env python
import os, sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import analyzer as az

um = 1e-6
nm = 1e-9

Lx, Ly, Lz = 574/8*nm, 574*nm, 574*nm
Nx, Ny, Nz = 32, 256, 256
dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz 

courant = 1./4
dt = courant * min(dx,dy,dz) /c

Q = 30
E = 1e-4
nf = 100
fmin = -5e14 
fmax = +5e14

loaddir = '/home/ldg/2nd_paper/SHPF.cupy.diel.CPML.MPI/graph/'
savedir = loaddir

names = ['fap1', 'fap2', 'fap3', 'fap4', 'fap5']

xlim = [-1,1]
ylim = [0,1]

test = az.CsvDataCollector(loaddir, 'nm', names, dt, Ly)
test.get_csv()
