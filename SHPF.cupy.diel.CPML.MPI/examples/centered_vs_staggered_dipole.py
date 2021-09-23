#!/usr/bin/env python
import os, time, datetime, sys, psutil
import matplotlib
matplotlib.use('Agg')
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.constants import c
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import source, space, plotter, structure, collector, recorder

#------------------------------------------------------------------#
#----------------------- Paramter settings ------------------------#
#------------------------------------------------------------------#

"""Description:

    sys.argv[1]: method
    sys.argv[2]: time steps

Command example:

    python3 ./centered_vs_staggered.py SHPF 5001
"""

nm = 1e-9
um = 1e-6

lunit = nm
lustr = 'nm'

funit = 1e12
fustr = 'THz'

Nx, Ny, Nz = 128, 128, 128
dx, dy, dz = 10*lunit, 10*lunit, 10*lunit
Lx, Ly, Lz = Nx*dx, Ny*dy, Nz*dz 

courant = 1./4
dt = courant * min(dx,dy,dz) / c
Tsteps = int(sys.argv[2])

method = sys.argv[1]
engine = 'cupy'
precision = 'DP'
if precision == 'SP': 
    floattype = np.float32
    complextype = np.complex64
if precision == 'DP':
    floattype = np.float64
    complextype = np.complex128

TF = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), dt, Tsteps, floattype, complextype, method=method, engine=engine) # Total field

TF.malloc()

########## Set PML and PBC
pmlthick = 10
pml = {'x':'+-','y':'','z':'+-'}
bbc = {'x':False, 'y':False, 'z':False}
pbc = {'x':False, 'y':True, 'z':False}

TF.apply_PML(pml, pmlthick)
TF.apply_BBC(bbc)
TF.apply_PBC(pbc)

#------------------------------------------------------------------#
#--------------------- Source object settings ---------------------#
#------------------------------------------------------------------#

# mmt for Gamma. point.

lamy = 0*lunit
lamz = 0*lunit
kx = 0
ky = 0
kz = 0
phi, theta = 0, 0
wvlen = lamy*np.cos(theta)

# mmt for plane wave normal to x axis.
# phi is the angle between k0 vector and xz-plane.
# theta is the angle between k0cos(phi) and x-axis.
#kx = k0 * np.cos(phi) * np.cos(theta)
#ky = k0 * np.sin(phi)
#kz = k0 * np.cos(phi) * np.sin(theta)

# mmt for plane wave normal to y axis.
# phi is the angle between k0 vector and xy-plane.
# theta is the angle between k0cos(phi) and y-axis.
#kx = k0 * np.cos(phi) * np.sin(theta)
#ky = k0 * np.cos(phi) * np.cos(theta)
#kz = k0 * np.sin(phi)

# mmt for plane wave normal to z axis.
# phi is the angle between k0 vector and yz-plane.
# theta is the angle between k0cos(phi) and z-axis.
#kx = k0 * np.sin(phi)
#ky = k0 * np.cos(phi) * np.sin(theta)
#kz = k0 * np.cos(phi) * np.cos(theta)

mmt = (kx, ky, kz)

########## Gaussian source
#wvc = float(sys.argv[2])*lunit
wvc = 140*lunit
w0 = (2*np.pi*c)/wvc
interval = 1
spread   = 0.3
pick_pos = 1500
ws = w0 * spread
src = source.Gaussian(dt, wvc, spread, pick_pos, dtype=np.float32)

savedir = f'../graph/cg_vs_sg/dipole/{method}/'

w1 = w0 * (1-spread*2)
w2 = w0 * (1+spread*2)

l1 = 2*np.pi*c / w1 / lunit
l2 = 2*np.pi*c / w2 / lunit

wvlens = np.arange(l2,l1, interval)*lunit
freqs = c / wvlens
src.plot_pulse(Tsteps, freqs, savedir)
#sys.exit()

########## Sine source
#smth = source.Smoothing(dt, 1000)
#src1 = source.Sine(dt, np.float64)
#wvlen = 250*lunit
#src1.set_wvlen(wvlen)

########## Harmonic source
#smth = source.Smoothing(dt, 1000)
#src1 = source.Harmonic(dt)
#wvlen = 250*lunit
#src1.set_wvlen(wvlen)

########## Delta source
#src1 = source.Delta(10)

########## Plane wave normal to x-axis.
#xsrt = Lx*0.2
#setterT1 = source.Setter(TF, (xsrt, 0, 0), (xsrt, Ly, Lz), mmt)
#sys.exit()

########## Plane wave normal to y-axis.
#setter1 = source.Setter(TF, (0, 200*lunit, 0), (Lx, 200*lunit+dy, Lz), mmt)
#setter2 = source.Setter(TF, (0, 150*lunit, 0), (Lx, 150*lunit+dy, Lz), mmt)
#setter3 = source.Setter(TF, (0, 250*lunit, 0), (Lx, 250*lunit+dy, Lz), mmt)
#setter4 = source.Setter(TF, (0, 350*lunit, 0), (Lx, 350*lunit+dy, Lz), mmt)

########## Plane wave normal to z-axis.
#setter1 = source.Setter(TF, (0, 0, 50*lunit), (Lx, Ly, 50*lunit+dz), mmt)
#setter2 = source.Setter(TF, (0, 0,150*lunit), (Lx, Ly,150*lunit+dz), mmt)
#setter3 = source.Setter(TF, (0, 0,250*lunit), (Lx, Ly,250*lunit+dz), mmt)
#setter4 = source.Setter(TF, (0, 0,350*lunit), (Lx, Ly,350*lunit+dz), mmt)

########## Line src along x-axis.
#setter1 = source.Setter(TF, (0, 200*lunit, 300*lunit), (Lx, 200*lunit+dy, 300*lunit+dz), mmt)
#setter2 = source.Setter(TF, (0, 150*lunit, 100*lunit), (Lx, 150*lunit+dy, 100*lunit+dz), mmt)
#setter3 = source.Setter(TF, (0, 300*lunit, 450*lunit), (Lx, 300*lunit+dy, 450*lunit+dz), mmt)
#setter4 = source.Setter(TF, (0, 450*lunit, 196*lunit), (Lx, 450*lunit+dy, 196*lunit+dz), mmt)

########## Line src along y-axis.
setterT1 = source.Setter(TF, (300*lunit, 0, 300*lunit), (300*lunit+dx, Ly, 300*lunit+dz), mmt)

########## Point src.
#setter1 = source.Setter(TF, ( 50*lunit, 3*Ly/8, 3*Lz/8), ( 50*lunit+dx, 3*Ly/8+dy, 3*Lz/8+dz), mmt)
#setter2 = source.Setter(TF, ( 70*lunit, 3*Ly/8, 6*Lz/8), ( 70*lunit+dx, 3*Ly/8+dy, 6*Lz/8+dz), mmt)
#setter3 = source.Setter(TF, ( 30*lunit, 6*Ly/8, 3*Lz/8), ( 30*lunit+dx, 6*Ly/8+dy, 3*Lz/8+dz), mmt)
#setter4 = source.Setter(TF, ( 90*lunit, 6*Ly/8, 6*Lz/8), ( 90*lunit+dx, 6*Ly/8+dy, 6*Lz/8+dz), mmt)
#setter1 = source.Setter(TF, ( 50*lunit, 1*Ly/16, 1*Lz/16), ( 50*lunit+dx, 1*Ly/16+dy, 1*Lz/16+dz), mmt)
#setter2 = source.Setter(TF, ( 70*lunit, 1*Ly/16, 8*Lz/16), ( 70*lunit+dx, 1*Ly/16+dy, 8*Lz/16+dz), mmt)
#setter3 = source.Setter(TF, ( 30*lunit, 1*Ly/16,15*Lz/16), ( 30*lunit+dx, 1*Ly/16+dy,15*Lz/16+dz), mmt)
#setter4 = source.Setter(TF, ( 90*lunit, 8*Ly/16, 1*Lz/16), ( 90*lunit+dx, 8*Ly/16+dy, 1*Lz/16+dz), mmt)
#setter5 = source.Setter(TF, ( 90*lunit, 8*Ly/16, 8*Lz/16), ( 90*lunit+dx, 8*Ly/16+dy, 8*Lz/16+dz), mmt)

#------------------------------------------------------------------#
#-------------------- Structure object settings -------------------#
#------------------------------------------------------------------#

# Put structures

srt1 = (3*Lx/8, 3*Ly/8, 3*Lz/8)
end1 = (5*Lx/8, 5*Ly/8, 5*Lz/8)

#Box = structure.Box('PEC_box', TF, srt1, end1, 1e4, 1.)
#Ball = structure.Sphere('diel_sphere', TF, (int(Nx/2)-1, int(Ny/2)-1, int(Nz/2)-1), radius_int*lunit, 4., 1.)
#Ball = structure.Sphere_percom('diel_sphere', TF, (Lx/2+3*dx, Ly/2+4*dy, Lz/2+5*dz), 100*lunit, 4., 1.)

#sys.exit()
# Save eps, mu and PML data.
#TF.save_pml_parameters('./')
#TF.save_eps_mu(savedir)

#------------------------------------------------------------------#
#------------------- Initialize update constants-------------------#
#------------------------------------------------------------------#

TF.init_update_constants()

#------------------------------------------------------------------#
#-------------------- Collector object settings -------------------#
#------------------------------------------------------------------#

# Determine the area of S calculator
#leftx, rightx = Lx/2-Ssca*lunit, Lx/2+Ssca*lunit
#lefty, righty = Ly/2-Ssca*lunit, Ly/2+Ssca*lunit
#leftz, rightz = Lz/2-Ssca*lunit, Lz/2+Ssca*lunit

#SF_Sx_L = collector.Sx("SF_L", savedir+"Sx/", SF, leftx,  (lefty, leftz), (righty, rightz), freqs, engine)
#SF_Sx_R = collector.Sx("SF_R", savedir+"Sx/", SF, rightx, (lefty, leftz), (righty, rightz), freqs, engine)

#SF_Sy_L = collector.Sy("SF_L", savedir+"Sy/", SF, lefty,  (leftx, leftz), (rightx, rightz), freqs, engine)
#SF_Sy_R = collector.Sy("SF_R", savedir+"Sy/", SF, righty, (leftx, leftz), (rightx, rightz), freqs, engine)

#SF_Sz_L = collector.Sz("SF_L", savedir+"Sz/", SF, leftz,  (leftx, lefty), (rightx, righty), freqs, engine)
#SF_Sz_R = collector.Sz("SF_R", savedir+"Sz/", SF, rightz, (leftx, lefty), (rightx, righty), freqs, engine)

#------------------------------------------------------------------#
#--------------------- Plotter object settings --------------------#
#------------------------------------------------------------------#

plot_per = 25
TFgraphtool = plotter.Graphtool(TF, 'TF', savedir)

cells = (Nx, Ny, Nz)

#------------------------------------------------------------------#
#--------------- Record simulation specifications------------------#
#------------------------------------------------------------------#

np.save(savedir+"freqs.npy", freqs)
np.save(savedir+"wvlens.npy", wvlens)

if TF.MPIrank == 0:

    spacespecs = f"""Space:

    VOLUME of the space: {TF.VOLUME:.2e} m^3
    Size of the space: {Lx/lunit}{lustr} x {Ly/lunit}{lustr} x {Lz/lunit}{lustr}
    The number of cells: {TF.Nx:4d} x {TF.Ny:4d} x {TF.Nz:4d}
    Grid spacing: {TF.dx/lunit:.3f} {lustr}, {TF.dy/lunit:.3f} {lustr}, {TF.dz/lunit:.3f} {lustr}
    Precision: {precision}
    """

    sourcespecs=f"""Source:

    SetterT1:
    \tglobal x location: {setterT1.src_xsrt}, {setterT1.src_xend}
    \t local x location: {setterT1.my_src_xsrt}, {setterT1.my_src_xend}

    SetterI1:
    Gaussian center wavelength: {wvc/um:.4f} um.
    Gaussian wavelength discretized per: {interval} {lustr}.
    Gaussian wave pick position at: {pick_pos} Tstep.
    Gaussian angular frequency spread: {spread:.3f} * w0
    Frequency points: {len(freqs)}
    Wavelength range: {round(l2,1)} to {round(l1,1)}
    """

    plotterspecs=f"""Plotters:

    Plot field profile per {plot_per} time steps.

    Save location: {savedir}

    Total time step: {TF.tsteps}
    Data size of a Ex field array: {TF.TOTAL_NUM_GRID_SIZE:05.2f} Mbytes
    """
    print(spacespecs)
    print(sourcespecs)
    print(plotterspecs)
    print(f'\nStructure and Collector objects in each Ranks:\n')
    #history = recorder.History(TF, "../history/")

TF.MPIcomm.Barrier()

structurespecs=\
f"""
    Structures:

"""

print(f'Rank {TF.MPIrank}:')
print(structurespecs)

#------------------------------------------------------------------#
#------------------------ Time loop begins ------------------------#
#------------------------------------------------------------------#

# Save what time the simulation begins.
start_time = datetime.datetime.now()

if TF.MPIrank == 0: f'\nSimulation start: {start_time}'
TF.MPIcomm.Barrier()

# time loop begins
for tstep in range(Tsteps):

    # pulse for gaussian wave
    #pulse1 = src.pulse_c(tstep)
    pulse1 = src.pulse_re(tstep)
    #print(type(pulse1))

    # pulse for Sine or Harmonic wave.
    #pulse1 = src1.apply(tstep) * smth.apply(tstep)

    # pulse for Delta function wave.
    #pulse1 = src.apply(tstep)

    setterT1.put_src('Ey', pulse1, 'soft')

    TF.updateH(tstep)
    TF.updateE(tstep)

    # Plot the field profile
    if tstep % plot_per == 0:

        if TF.MPIrank == 0:

            now = datetime.datetime.now()
            #uptime = (now - start_time).strftime('%H:%M:%S')
            uptime = (now - start_time)
            print(f"runtime: {uptime}, step: {tstep:7d}, {100.*tstep/TF.tsteps:05.2f}%, at {now}." )

        Ey = TFgraphtool.gather('Ey')
        #TFgraphtool.plot2D3D('Ex', tstep, yidx=TF.Nyc, colordeep=2, stride=1, zlim=2)
        TFgraphtool.plot2D3D(Ey, tstep, yidx=TF.Nyc, colordeep=.1, stride=1, zlim=.1, savenpy=True)
        #TFgraphtool.plot2D3D('Ez', tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D('Hx', tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D('Hy', tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D('Hz', tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
