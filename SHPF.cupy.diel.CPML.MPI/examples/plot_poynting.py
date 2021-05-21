import sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from scipy.constants import c
import plotter

um = 1e-6
nm = 1e-9

wvc = 640*nm
w0 = (2*np.pi*c)/wvc
interval = 1
spread   = 0.08
ws = w0 * spread

w1 = w0 * (1-spread*2)
w2 = w0 * (1+spread*2)

l1 = 2*np.pi*c / w1 / nm
l2 = 2*np.pi*c / w2 / nm

wvlens = np.arange(l2,l1,interval) * nm

wvlen_unit = 'nm'
freq_unit = 'THz'

method = sys.argv[1] 
Tsteps = int(sys.argv[2])

Nx = int(sys.argv[3])
Ny = int(sys.argv[4])
Nz = int(sys.argv[5])

cells = (Nx,Ny,Nz)

a = 512*nm
Lx, Ly, Lz = 6*a, a, a 

painter = plotter.SpectrumPlotter(method, cells, wvlens, freq_unit, wvlen_unit)

dirs = f'../graph/sqhole_2slab_{method}/{int(Lx/nm):04d}nm{int(Ly/nm):04d}nm\
{int(Lz/nm):04d}nm_{Nx:04d}_{Ny:04d}_{Nz:04d}_{Tsteps:07d}/'

dirs = f'../graph/simple_2slab_{method}/{int(Lx/nm):04d}nm{int(Ly/nm):04d}nm\
{int(Lz/nm):04d}nm_{Nx:04d}_{Ny:04d}_{Nz:04d}_{Tsteps:07d}/'

Sy_L = ['../graph/Sy_SF_L_area.npy']
Sy_R = ['../graph/Sy_SF_R_area.npy']

Sz_L = ['../graph/Sz_SF_L_area.npy']
Sz_R = ['../graph/Sz_SF_R_area.npy']

S = ['../graph/Sx_SF_L_area.npy', '../graph/Sx_SF_R_area.npy', '../graph/Sy_SF_L_area.npy',\
     '../graph/Sy_SF_R_area.npy', '../graph/Sz_SF_L_area.npy', '../graph/Sz_SF_R_area.npy',]

#wvxlim = [0.4, .5]
wvxlim = [None, None]
#wvylim = [None, 1.1]
wvylim = [0, None]
#freqxlim = [600, 700]
freqxlim = [None, None]
#freqylim = [None, 1.1]
freqylim = [0, None]

for tt in range(10000, 110000, 10000):

    TF_Sx_R = [dirs+f'Sx/TF_R_{tt:07d}tstep_area.npy']
    SF_Sx_L = [dirs+f'Sx/SF_L_{tt:07d}tstep_area.npy']
    IF_Sx_L = [dirs+f'Sx/IF_R_{tt:07d}tstep_area.npy']

    painter.simple_plot(TF_Sx_R, dirs+f'TF_Sx_R_{tt:07d}tstep_spectrum.png')
    painter.simple_plot(SF_Sx_L, dirs+f'SF_Sx_L_{tt:07d}tstep_spectrum.png')
    painter.simple_plot(IF_Sx_L, dirs+f'IF_Sx_R_{tt:07d}tstep_spectrum.png')
    painter.plot_IRT(IF_Sx_L, SF_Sx_L, TF_Sx_R, tt, dirs+f'IRT_{tt:07d}tstep.png', wvxlim, wvylim, freqxlim, freqylim)


"""
painter2.simple_plot(Sy_L, './graph/Sy_L_SF_spectrum.png')
painter2.simple_plot(Sy_R, './graph/Sy_R_SF_spectrum.png')
painter2.simple_plot(Sz_L, './graph/Sz_L_SF_spectrum.png')
painter2.simple_plot(Sz_R, './graph/Sz_R_SF_spectrum.png')
painter2.simple_plot(S, './graph/S_SF_spectrum.png')
painter2.simple_plot(['./graph/Sx_IF_R_area.npy'], './graph/source.png')
"""
