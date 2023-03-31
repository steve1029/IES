import sys, os, json
sys.path.append("/root/SHPF/")
import numpy as np
from scipy.constants import c
import plotter

json_path = sys.argv[1]

with open(json_path) as f:
    sim_data = json.load(f)

savedir = sim_data["savedir"]

method = sim_data["method"]
tsteps = sim_data["time_steps"]

lunit = sim_data["length_unit"]
lustr = sim_data["length_unit_str"]

funit = sim_data["freq_unit"]
fustr = sim_data["freq_unit_str"]

Lx = sim_data["Lx"]
Ly = sim_data["Ly"]
Lz = sim_data["Lz"]

Nx = sim_data["Nx"]
Ny = sim_data["Ny"]
Nz = sim_data["Nz"]

if sim_data["source"] == "Gaussian":

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

cells = (Nx,Ny,Nz)

painter = plotter.SpectrumPlotter(method, cells, wvlens, fustr, lustr)

#wvxlim = [0.4, .5]
wvxlim = [None, None]
#wvylim = [None, 1.1]
wvylim = [0, None]
#freqxlim = [600, 700]
freqxlim = [None, None]
#freqylim = [None, 1.1]
freqylim = [0, None]

TF_Sx_R = [savedir+f'Sx/TF_R_{tsteps:07d}tstep_area.npy']
SF_Sx_L = [savedir+f'Sx/SF_L_{tsteps:07d}tstep_area.npy']
IF_Sx_L = [savedir+f'Sx/IF_R_{tsteps:07d}tstep_area.npy']

painter.simple_plot(TF_Sx_R, savedir+f'TF_Sx_R_{tsteps:07d}tstep_spectrum.png')
painter.simple_plot(SF_Sx_L, savedir+f'SF_Sx_L_{tsteps:07d}tstep_spectrum.png')
painter.simple_plot(IF_Sx_L, savedir+f'IF_Sx_R_{tsteps:07d}tstep_spectrum.png')
painter.plot_IRT(IF_Sx_L, SF_Sx_L, TF_Sx_R, tsteps, savedir+f'IRT_{tsteps:07d}tstep.png', wvxlim, wvylim, freqxlim, freqylim)
