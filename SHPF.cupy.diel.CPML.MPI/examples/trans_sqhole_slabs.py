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
#--------------------- Space object settings ----------------------#
#------------------------------------------------------------------#

nm = 1e-9
um = 1e-6

lunit = nm
lustr = 'nm'

funit = 1e12
fustr = 'THz'

#a = 779.42*lunit
a = 512*lunit

Lx, Ly, Lz = 6*a, a, a 
Nx, Ny, Nz = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
dx, dy, dz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)

courant = 1./4
dt = courant * min(dx,dy,dz) / c
Tsteps = int(sys.argv[2])

method = sys.argv[1]
engine = 'cupy'

TF = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), dt, Tsteps, np.complex64, np.complex64, method=method, engine=engine)
IF = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), dt, Tsteps, np.complex64, np.complex64, method=method, engine=engine)
SF = space.Empty3D((Nx, Ny, Nz), (dx, dy, dz), dt, Tsteps, np.complex64, np.complex64, method=method, engine=engine)

TF.malloc()
IF.malloc()

########## Set PML and PBC
thick = 10
pml = {'x':'+-','y':'','z':''}
bbc = {'x':False, 'y':False, 'z':False}
pbc = {'x':False, 'y':True, 'z':True}

TF.apply_PML(pml, thick)
TF.apply_BBC(bbc)
TF.apply_PBC(pbc)

IF.apply_PML(pml, thick)
IF.apply_BBC(bbc)
IF.apply_PBC(pbc)

########## Save PML data.
#TF.save_pml_parameters('./')

#------------------------------------------------------------------#
#--------------------- Source object settings ---------------------#
#------------------------------------------------------------------#

########## Momentum of the source.

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
wvc = 640*lunit
w0 = (2*np.pi*c)/wvc
interval = 1
spread   = 0.08
pick_pos = 3000
ws = w0 * spread
src = source.Gaussian(dt, wvc, spread, pick_pos, dtype=np.float32)

savedir = f'../graph/simple_2slab_{method}/{int(Lx/lunit):04d}\
{lustr}{int(Ly/lunit):04d}{lustr}{int(Lz/lunit):04d}{lustr}_{Nx:04d}_{Ny:04d}_{Nz:04d}_{Tsteps:07d}/'

if TF.MPIrank == 0:
    print(f'Gaussian center wavelength: {wvc/um:.4f} um.')
    print(f'Gaussian wavelength discretized per: {interval} {lustr}.')
    print(f'Gaussian wave pick position at: {pick_pos} Tstep.')
    print(f'Gaussian angular frequency spread: {spread:.3f} * w0\n')

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
xsrt = Lx*0.15
setterT1 = source.Setter(TF, (xsrt, 0, 0), (xsrt+dx, Ly, Lz), mmt)
setterI1 = source.Setter(IF, (xsrt, 0, 0), (xsrt+dx, Ly, Lz), mmt)

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
#setter1 = source.Setter(TF, (140*lunit, 0), (140*lunit+dx, Ly), mmt)
#setter2 = source.Setter(TF, (200*lunit, 0), (410*lunit+dx, Ly), mmt)
#setter3 = source.Setter(TF, (100*lunit, 0), (410*lunit+dx, Ly), mmt)
#setter4 = source.Setter(TF, (100*lunit, 0), (410*lunit+dx, Ly), mmt)

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

########## Box

t1 = Lx * 0.45
t2 = t1 + a*.2
t3 = t2 + a*.3
t4 = t3 + a*.2

srt1 = (t1,  0,  0)
end1 = (t2, Ly, Lz)

srt2 = (t3,  0,  0)
end2 = (t4, Ly, Lz)

structure.Box(TF, srt1, end1, 4, 1.)
structure.Box(TF, srt2, end2, 4, 1.)

########## Circle
radius = a / 4
height1 = (t1, t2)
height2 = (t3, t4)
center = np.array([Ly/2, Lz/2])
lcy = Ly/4
lcz = Lz/4

#structure.Cylinder3D(TF, 'x', radius, height1, center, 1, 1.)
#structure.Cylinder3D(TF, 'x', radius, height2, center, 1, 1.)

rot = 0
rot_cen = center
#structure.Cylinder3D_slab(TF, 'x', radius, height1, lcy, lcz, 0, rot_cen, 1, 1)
#structure.Cylinder3D_slab(TF, 'x', radius, height2, lcy, lcz, 0, rot_cen, 1, 1)

########## Save eps, mu data.
#TF.save_eps_mu(savedir)
#sys.exit()

#------------------------------------------------------------------#
#------------------- Initialize update constants-------------------#
#------------------------------------------------------------------#

TF.init_update_constants()
IF.init_update_constants()

#------------------------------------------------------------------#
#-------------------- Collector object settings -------------------#
#------------------------------------------------------------------#
leftx, rightx = 100*lunit, 500*lunit
lefty, righty = 0*lunit, Ly
leftz, rightz = 0*lunit, Lz

TF_Sx_R_calculator = collector.Sx("TF_R", savedir+"Sx/", TF, Lx*0.80, (lefty, leftz), (righty, rightz), freqs, engine)
IF_Sx_R_calculator = collector.Sx("IF_R", savedir+"Sx/", IF, Lx*0.80, (lefty, leftz), (righty, rightz), freqs, engine)
SF_Sx_L_calculator = collector.Sx("SF_L", savedir+"Sx/", SF, Lx*0.30, (lefty, leftz), (righty, rightz), freqs, engine)

cal_per = 10000

#------------------------------------------------------------------#
#--------------------- Plotter object settings --------------------#
#------------------------------------------------------------------#

plot_per = 1000
TFgraphtool = plotter.Graphtool(TF, 'TF', savedir)
IFgraphtool = plotter.Graphtool(IF, 'IF', savedir)
SFgraphtool = plotter.Graphtool(SF, 'SF', savedir)

cells = (Nx, Ny, Nz)
painter = plotter.SpectrumPlotter(method, cells, wvlens, fustr, lustr)

#wvxlim = [0.4, .5]
wvxlim = [None, None]
wvylim = [None, 1.1]
#freqxlim = [600, 700]
freqxlim = [None, None]
freqylim = [None, 1.1]

#sys.exit()
#------------------------------------------------------------------#
#------------------------ Time loop begins ------------------------#
#------------------------------------------------------------------#

# Save what time the simulation begins.
start_time = datetime.datetime.now()

# time loop begins
for tstep in range(Tsteps+1):

    # At the start point
    if tstep == 0:
        TF.MPIcomm.Barrier()
        if TF.MPIrank == 0:
            print(f"\nSave directory: {savedir}")
            print(f"\nTotal time step: {TF.tsteps}")
            print(f"Size of a total field array : {TF.TOTAL_NUM_GRID_SIZE:05.2f} Mbytes")
            print(f"\nSimulation start: {datetime.datetime.now()}")
        
    # pulse for gaussian wave.
    pulse1 = src.pulse_c(tstep)

    # pulse for Sine or Harmonic wave.
    #pulse1 = src1.apply(tstep) * smth.apply(tstep)

    # pulse for Delta function wave.
    #pulse1 = src.apply(tstep)

    setterT1.put_src('Ey', pulse1, 'soft')
    setterI1.put_src('Ey', pulse1, 'soft')

    TF.updateH(tstep)
    TF.updateE(tstep)

    IF.updateH(tstep)
    IF.updateE(tstep)

    SF.get_SF(TF, IF)

    TF_Sx_R_calculator.do_RFT(tstep)
    IF_Sx_R_calculator.do_RFT(tstep)
    SF_Sx_L_calculator.do_RFT(tstep)

    # Plot the field profile
    if tstep % plot_per == 0:
        
        #Ex = TFgraphtool.gather('Ex')
        #TFgraphtool.plot2D3D(Ex, tstep, xidx=TF.Nxc, colordeep=1e-3, stride=3, zlim=1e-3)
        #TFgraphtool.plot2D3D(Ey, tstep, yidx=TF.Nyc, colordeep=1., stride=2, zlim=1.)
        #TFgraphtool.plot2D3D(Ez, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hx, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hy, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hz, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)

        Ey = TFgraphtool.gather('Ey')
        #TFgraphtool.plot2D3D(Ex, tstep, yidx=TF.Nyc, colordeep=2., stride=1, zlim=2.)
        TFgraphtool.plot2D3D(Ey, tstep, yidx=TF.Nyc, colordeep=.2, stride=2, zlim=.2)
        #TFgraphtool.plot2D3D(Ez, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hx, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hy, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hz, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1) 

        Ey = IFgraphtool.gather('Ey')
        #IFgraphtool.plot2D3D(Ex, tstep, yidx=TF.Nyc, colordeep=2, stride=1, zlim=2)
        IFgraphtool.plot2D3D(Ey, tstep, yidx=IF.Nyc, colordeep=.2, stride=2, zlim=.2)
        #IFgraphtool.plot2D3D(Ez, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #IFgraphtool.plot2D3D(Hx, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #IFgraphtool.plot2D3D(Hy, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #IFgraphtool.plot2D3D(Hz, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)

        Ey = SFgraphtool.gather('Ey')
        #SFgraphtool.plot2D3D(Ex, tstep, yidx=TF.Nyc, colordeep=2., stride=1, zlim=2.)
        SFgraphtool.plot2D3D(Ey, tstep, yidx=SF.Nyc, colordeep=.2, stride=2, zlim=.2)
        #SFgraphtool.plot2D3D(Ez, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #SFgraphtool.plot2D3D(Hx, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #SFgraphtool.plot2D3D(Hy, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #SFgraphtool.plot2D3D(Hz, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)

        #Ez = TFgraphtool.gather('Ez')
        #TFgraphtool.plot2D3D(Ex, tstep, xidx=TF.Nxc, colordeep=1e-2, stride=3, zlim=1e-2)
        #TFgraphtool.plot2D3D(Ey, tstep, yidx=TF.Nyc, colordeep=1., stride=2, zlim=1.)
        #TFgraphtool.plot2D3D(Ez, tstep, zidx=TF.Nzc, colordeep=2e-4, stride=3, zlim=1e-5)
        #TFgraphtool.plot2D3D(Hx, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hy, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hz, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)

        if TF.MPIrank == 0:

            now = datetime.datetime.now()
            print(f"runtime: {now-start_time}, step: {tstep:7d}, {100.*tstep/TF.tsteps:05.2f}%, at {now}." )

    if tstep != 0 and tstep % cal_per == 0:

        TF_Sx_R_calculator.get_Sx(tstep, h5=False)
        IF_Sx_R_calculator.get_Sx(tstep, h5=False)
        SF_Sx_L_calculator.get_Sx(tstep, h5=False)

        #print(f'rank {TF.MPIrank:02d}: Sx has been saved at {tstep} time step.')

    """
    if TF.MPIrank == 0 and tstep != 0 and tstep % cal_per == 0:

        #------------------------------------------------------------------#
        #--------------------------- Data analysis ------------------------#
        #------------------------------------------------------------------#

        TF_Sx_R_calculator.get_Sx(tstep, h5=False)
        IF_Sx_R_calculator.get_Sx(tstep, h5=False)
        SF_Sx_L_calculator.get_Sx(tstep, h5=False)

        TF_Sx_R = [savedir+f'Sx/TF_R_{tstep}tstep_area.npy']
        SF_Sx_L = [savedir+f'Sx/SF_L_{tstep}tstep_area.npy']
        IF_Sx_R = [savedir+f'Sx/IF_L_{tstep}tstep_area.npy']

        painter.simple_plot(TF_Sx_R, savedir+f'TF_Sx_R_{tstep}tstep_spectrum.png')
        painter.simple_plot(SF_Sx_L, savedir+f'SF_Sx_L_{tstep}tstep_spectrum.png')
        painter.simple_plot(IF_Sx_R, savedir+f'IF_Sx_R_{tstep}tstep_spectrum.png')
        painter.plot_IRT(IF_Sx_R, SF_Sx_L, TF_Sx_R, tstep, savedir+f'IRT_{tstep}tstep.png', wvxlim, wvylim, freqxlim, freqylim)

        print(f'IRT.png are plotted at {tstep} time step.')
    """

#------------------------------------------------------------------#
#------------------- Record simulation history --------------------#
#------------------------------------------------------------------#

np.save(savedir+"freqs.npy", freqs)
np.save(savedir+"wvlens.npy", wvlens)
if TF.MPIrank == 0: recording = recorder.Recorder(TF, start_time, "../record/")
