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

#a = 779.42 * nm
a = 574 * nm

#Lx, Ly, Lz = 600*nm, 2*a, 2*a 
#Nx, Ny, Nz = 128, 128, 128
Lx, Ly, Lz = a, a, a 
Nx, Ny, Nz = 256, 256, 256
dx, dy, dz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)

courant = 1./4
dt = courant * min(dx,dy,dz) /c

Q = 30
E = 1e-4
nf = 100
fmin = -c/Ly
fmax = +c/Ly

method = sys.argv[1]
ptop = sys.argv[2] # point to point.
tsteps = sys.argv[3] # time steps.
fapn = sys.argv[4]

loaddir = '/home/ldg/2nd_paper/SHPF.cupy.diel.CPML.MPI/graph/{}/{}tsteps/{}/' .format(method, tsteps, ptop)
savedir = loaddir

fapns = ['fap1', 'fap2', 'fap3', 'fap4', 'fap5']
#fapns = ['fap1', 'fap2', 'fap3', 'fap4']
#fapns = ['fap1', 'fap2']

xlim = [-1,1]

get_tsteps_from = 'Ex'
test = az.CsvCreator(loaddir, fapns, dt, Ly, get_tsteps_from)
#test.get_fft_plot_csv(3, None, None, xlim, [])
#test.get_fft_plot_csv(3, None, None, xlim, [])
#test.get_fft_plot_csv(3, None, None, xlim, [])

test.get_pharminv_csv('Ex', fapn, tsteps, dt, fmin, fmax, nf)
test.get_pharminv_csv('Ey', fapn, tsteps, dt, fmin, fmax, nf)
test.get_pharminv_csv('Ez', fapn, tsteps, dt, fmin, fmax, nf)

#test.get_pharminv_csv('Ex', fapn, 150001, dt, fmin, fmax, nf)
#test.get_pharminv_csv('Ex', fapn, 150001, dt, fmin, fmax, nf)
#test.get_pharminv_csv('Ex', fapn, 150001, dt, fmin, fmax, nf)
#test.get_pharminv_csv('Ex', fapn, 150001, dt, fmin, fmax, nf)
#test.get_pharminv_csv('Hx', 'fap1', 200001, dt, fmin, fmax, nf)
#test.get_pharminv_csv('Hy', 'fap1', 200001, dt, fmin, fmax, nf)
