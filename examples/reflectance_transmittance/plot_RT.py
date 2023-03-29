import sys, os, json
sys.path.append("/root/SHPF/")
import numpy as np
from scipy.constants import c
import plotter

sim_data = json.loads("/root/SHPF/graph/simple_2slab_FDTD/0720um0512um0512um_0050_0050_0050_0005000_100um_200um_100um/sim_data.json")

savedir = sim_data["save_dir"]

method = sim_data["method"]
Tsteps = sim_data["time_steps"]

lunit = sim_data["length_unit"]
lustr = sim_data["length_unit_str"]

funit = sim_data["freq_unit"]
fustr = sim_data["freq_unit_str"]

Lx = sim_data["Lx"]
Ly = sim_data["Ly"]
Lz = sim_data["Lz"]

if sim_data["source"] = "Gaussian"

    spread = sim_data["source_parameters"]["spread"]
    interval = sim_data["source_parameters"]["interval"]
    peak_pos = sim_data["source_parameters"]["peak_pos"]
    wvc = sim_data["source_parameters"]["wvc"]
    w0 = sim_data["source_parameters"]["w0"]
    ws = sim_data["source_parameters"]["ws"]
    l1 = sim_data["source_parameters"]["l1"]
    l2 = sim_data["source_parameters"]["l2"]
    w1 = sim_data["source_parameters"]["w1"]
    w2 = sim_data["source_parameters"]["w2"]

wvlens = np.arange(l2,l1,interval) * lunit

freq_unit = 'THz'

Nx = int(sys.argv[3])
Ny = int(sys.argv[4])
Nz = int(sys.argv[5])

cells = (Nx,Ny,Nz)

painter = plotter.SpectrumPlotter(method, cells, wvlens, fustr, lustr)

#savedir = f'../graph/sqhole_2slab_{method}/{int(Lx/lunit):04d}{lustr}{int(Ly/lunit):04d}{lustr}\
#{int(Lz/lunit):04d}{lustr}_{Nx:04d}_{Ny:04d}_{Nz:04d}_{Tsteps:07d}/'

savedir = f'~/SHPF/graph/sqhole_2slab_long_{method}/{int(Lx/lunit):04d}\
{lustr}{int(Ly/lunit):04d}{lustr}{int(Lz/lunit):04d}{lustr}_{Nx:04d}_{Ny:04d}_{Nz:04d}_{Tsteps:07d}\
_100{lustr}_200{lustr}_100{lustr}/'

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

for tt in range(100000, 2100000, 100000):
#for tt in [10000, 21000]:

    TF_Sx_R = [savedir+f'Sx/TF_R_{tt:07d}tstep_area.npy']
    SF_Sx_L = [savedir+f'Sx/SF_L_{tt:07d}tstep_area.npy']
    IF_Sx_L = [savedir+f'Sx/IF_R_{tt:07d}tstep_area.npy']

    painter.simple_plot(TF_Sx_R, savedir+f'TF_Sx_R_{tt:07d}tstep_spectrum.png')
    painter.simple_plot(SF_Sx_L, savedir+f'SF_Sx_L_{tt:07d}tstep_spectrum.png')
    painter.simple_plot(IF_Sx_L, savedir+f'IF_Sx_R_{tt:07d}tstep_spectrum.png')
    painter.plot_IRT(IF_Sx_L, SF_Sx_L, TF_Sx_R, tt, savedir+f'IRT_{tt:07d}tstep.png', wvxlim, wvylim, freqxlim, freqylim)

"""
painter2.simple_plot(Sy_L, './graph/Sy_L_SF_spectrum.png')
painter2.simple_plot(Sy_R, './graph/Sy_R_SF_spectrum.png')
painter2.simple_plot(Sz_L, './graph/Sz_L_SF_spectrum.png')
painter2.simple_plot(Sz_R, './graph/Sz_R_SF_spectrum.png')
painter2.simple_plot(S, './graph/S_SF_spectrum.png')
painter2.simple_plot(['./graph/Sx_IF_R_area.npy'], './graph/source.png')
"""
