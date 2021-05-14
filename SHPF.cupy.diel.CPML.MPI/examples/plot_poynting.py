import sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from scipy.constants import c
import plotter

um = 1e-6
nm = 1e-9

wvc = 660*nm
w0 = (2*np.pi*c)/wvc
interval = 1
spread   = 0.1
ws = w0 * spread

w1 = w0 * (1-spread*2)
w2 = w0 * (1+spread*2)

l1 = 2*np.pi*c / w1 / nm
l2 = 2*np.pi*c / w2 / nm

wvlen = np.arange(l2,l1,interval) * um

wvlen_unit = 'um'
freq_unit = 'THz'

painter = plotter.SpectrumPlotter(wvlen, freq_unit, wvlen_unit)

method = sys.argv[1] 
tsteps = sys.argv[2]

Nx = int(sys.argv[3])
Ny = int(sys.argv[4])
Nz = int(sys.argv[5])

dirs = '../graph/{}/{}tsteps/{}_{}_{}/'. format(method,tsteps,Nx,Ny,Nz)

TF_Sx_R = [dirs+'Sx/TF_R_area.npy']
SF_Sx_L = [dirs+'Sx/SF_L_area.npy']
IF_Sx_L = [dirs+'Sx/IF_L_area.npy']

Sy_L = ['../graph/Sy_SF_L_area.npy']
Sy_R = ['../graph/Sy_SF_R_area.npy']

Sz_L = ['../graph/Sz_SF_L_area.npy']
Sz_R = ['../graph/Sz_SF_R_area.npy']

S = ['../graph/Sx_SF_L_area.npy', '../graph/Sx_SF_R_area.npy', '../graph/Sy_SF_L_area.npy',\
     '../graph/Sy_SF_R_area.npy', '../graph/Sz_SF_L_area.npy', '../graph/Sz_SF_R_area.npy',]

wvxlim = [0.4, .5]
wvylim = [None, 1.1]
freqxlim = [600, 700]
freqylim = [None, 1.1]

painter.simple_plot(TF_Sx_R, dirs+'TF_Sx_R_spectrum.png')
painter.simple_plot(SF_Sx_L, dirs+'SF_Sx_L_spectrum.png')
painter.simple_plot(IF_Sx_L, dirs+'IF_Sx_L_spectrum.png')
painter.plot_IRT(IF_Sx_L, SF_Sx_L, TF_Sx_R, dirs+'IRT.png', wvxlim, wvylim, freqxlim, freqylim)

"""
painter2.simple_plot(Sy_L, './graph/Sy_L_SF_spectrum.png')
painter2.simple_plot(Sy_R, './graph/Sy_R_SF_spectrum.png')
painter2.simple_plot(Sz_L, './graph/Sz_L_SF_spectrum.png')
painter2.simple_plot(Sz_R, './graph/Sz_R_SF_spectrum.png')
painter2.simple_plot(S, './graph/S_SF_spectrum.png')
painter2.simple_plot(['./graph/Sx_IF_R_area.npy'], './graph/source.png')
"""
