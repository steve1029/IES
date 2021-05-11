#!/usr/bin/env python
import os, time, datetime, sys, psutil
import matplotlib
matplotlib.use('Agg')
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.constants import c
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import space_v2 as space
import source, plotter, structure, collector, recorder

#------------------------------------------------------------------#
#--------------------- Space object settings ----------------------#
#------------------------------------------------------------------#

nm = 1e-9
um = 1e-6

#a = 779.42*nm
a = 200*nm

Lx, Ly, Lz = 600*nm, a, a 
Nx, Ny, Nz = 200, 32, 32
dx, dy, dz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)

courant = 1./4
dt = courant * min(dx,dy,dz) / c
Tsteps = int(sys.argv[1])

method = 'FDTD'
engine = 'cupy'

TF = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), dt, Tsteps, np.complex64, np.complex64, method=method, engine=engine)
IF = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), dt, Tsteps, np.complex64, np.complex64, method=method, engine=engine)
SF = space.Empty3D((Nx, Ny, Nz), (dx, dy, dz), dt, Tsteps, np.complex64, np.complex64, method=method, engine=engine)

TF.malloc()
IF.malloc()

########## Set PML and PBC
thick = 10
pml = {'x':'+-','y':'','z':''}
bbc = {'x':False, 'y':True, 'z':True}
pbc = {'x':False, 'y':False, 'z':False}

TF.apply_PML(pml, thick)
TF.apply_BBC(bbc)
TF.apply_PBC(pbc)

IF.apply_PML(pml, thick)
IF.apply_BBC(bbc)
IF.apply_PBC(pbc)

########## Save PML data.
#TF.save_pml_parameters('./')

filename = "{}/{}tsteps/" .format(method, Tsteps)
savedir = '../graph/' + filename

#------------------------------------------------------------------#
#--------------------- Source object settings ---------------------#
#------------------------------------------------------------------#

########## Momentum of the source.

# mmt for Gamma. point.

lamy = 0*nm
lamz = 0*nm
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
#wvc = float(sys.argv[2])*nm
wvc = 150*nm
w0 = (2*np.pi*c)/wvc
interval = 2
spread   = 0.3
pick_pos = 1000
ws = w0 * spread
src = source.Gaussian(dt, wvc, spread, pick_pos, dtype=np.float32)

w1 = w0 * (1-spread*2)
w2 = w0 * (1+spread*2)

l1 = 2*np.pi*c / w1 / nm
l2 = 2*np.pi*c / w2 / nm

wvlens = np.arange(l2,l1, interval)*nm
freqs = c / wvlens
#np.save("../graph/freqs", freqs)
#src.plot_pulse(Tsteps, freqs, savedir)
#sys.exit()

########## Sine source
#smth = source.Smoothing(dt, 1000)
#src1 = source.Sine(dt, np.float64)
#wvlen = 250*nm
#src1.set_wvlen(wvlen)

########## Harmonic source
#smth = source.Smoothing(dt, 1000)
#src1 = source.Harmonic(dt)
#wvlen = 250*nm
#src1.set_wvlen(wvlen)

########## Delta source
#src1 = source.Delta(10)

########## Plane wave normal to x-axis.
setterT1 = source.Setter(TF, (70*nm, 0, 0), (70*nm+dx, Ly, Lz), mmt)
setterI1 = source.Setter(IF, (70*nm, 0, 0), (70*nm+dx, Ly, Lz), mmt)

########## Plane wave normal to y-axis.
#setter1 = source.Setter(TF, (0, 200*nm, 0), (Lx, 200*nm+dy, Lz), mmt)
#setter2 = source.Setter(TF, (0, 150*nm, 0), (Lx, 150*nm+dy, Lz), mmt)
#setter3 = source.Setter(TF, (0, 250*nm, 0), (Lx, 250*nm+dy, Lz), mmt)
#setter4 = source.Setter(TF, (0, 350*nm, 0), (Lx, 350*nm+dy, Lz), mmt)

########## Plane wave normal to z-axis.
#setter1 = source.Setter(TF, (0, 0, 50*nm), (Lx, Ly, 50*nm+dz), mmt)
#setter2 = source.Setter(TF, (0, 0,150*nm), (Lx, Ly,150*nm+dz), mmt)
#setter3 = source.Setter(TF, (0, 0,250*nm), (Lx, Ly,250*nm+dz), mmt)
#setter4 = source.Setter(TF, (0, 0,350*nm), (Lx, Ly,350*nm+dz), mmt)

########## Line src along x-axis.
#setter1 = source.Setter(TF, (0, 200*nm, 300*nm), (Lx, 200*nm+dy, 300*nm+dz), mmt)
#setter2 = source.Setter(TF, (0, 150*nm, 100*nm), (Lx, 150*nm+dy, 100*nm+dz), mmt)
#setter3 = source.Setter(TF, (0, 300*nm, 450*nm), (Lx, 300*nm+dy, 450*nm+dz), mmt)
#setter4 = source.Setter(TF, (0, 450*nm, 196*nm), (Lx, 450*nm+dy, 196*nm+dz), mmt)

########## Line src along y-axis.
#setter1 = source.Setter(TF, (140*nm, 0), (140*nm+dx, Ly), mmt)
#setter2 = source.Setter(TF, (200*nm, 0), (410*nm+dx, Ly), mmt)
#setter3 = source.Setter(TF, (100*nm, 0), (410*nm+dx, Ly), mmt)
#setter4 = source.Setter(TF, (100*nm, 0), (410*nm+dx, Ly), mmt)

########## Point src.
#setter1 = source.Setter(TF, ( 50*nm, 3*Ly/8, 3*Lz/8), ( 50*nm+dx, 3*Ly/8+dy, 3*Lz/8+dz), mmt)
#setter2 = source.Setter(TF, ( 70*nm, 3*Ly/8, 6*Lz/8), ( 70*nm+dx, 3*Ly/8+dy, 6*Lz/8+dz), mmt)
#setter3 = source.Setter(TF, ( 30*nm, 6*Ly/8, 3*Lz/8), ( 30*nm+dx, 6*Ly/8+dy, 3*Lz/8+dz), mmt)
#setter4 = source.Setter(TF, ( 90*nm, 6*Ly/8, 6*Lz/8), ( 90*nm+dx, 6*Ly/8+dy, 6*Lz/8+dz), mmt)
#setter1 = source.Setter(TF, ( 50*nm, 1*Ly/16, 1*Lz/16), ( 50*nm+dx, 1*Ly/16+dy, 1*Lz/16+dz), mmt)
#setter2 = source.Setter(TF, ( 70*nm, 1*Ly/16, 8*Lz/16), ( 70*nm+dx, 1*Ly/16+dy, 8*Lz/16+dz), mmt)
#setter3 = source.Setter(TF, ( 30*nm, 1*Ly/16,15*Lz/16), ( 30*nm+dx, 1*Ly/16+dy,15*Lz/16+dz), mmt)
#setter4 = source.Setter(TF, ( 90*nm, 8*Ly/16, 1*Lz/16), ( 90*nm+dx, 8*Ly/16+dy, 1*Lz/16+dz), mmt)
#setter5 = source.Setter(TF, ( 90*nm, 8*Ly/16, 8*Lz/16), ( 90*nm+dx, 8*Ly/16+dy, 8*Lz/16+dz), mmt)

#------------------------------------------------------------------#
#-------------------- Structure object settings -------------------#
#------------------------------------------------------------------#

########## Box

t1 = 200*nm
t2 = t1 + a*0.2

srt = (t1,  0,  0)
end = (t2, Ly, Lz)
box = structure.Box(TF, srt, end, 4, 1.)

########## Circle
radius = a / 4
height = (t1, t2)
center1 = np.array([Ly/2, Lz/2])
lcy = Ly/2
lcz = Lz/2

#structure.Cylinder3D(TF, 'x', radius, height, center1, 1, 1.)

rot = 0
rot_cen = center1
#structure.Cylinder3D_slab(TF, 'x', radius, height, lcy, lcz,  0, rot_cen, 1, 1)
#structure.Cylinder3D_slab(TF, 'x', radius, height, lcy, lcz, 45, rot_cen,  5, 1)

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
leftx, rightx = 100*nm, 500*nm
lefty, righty = 0*nm, Ly
leftz, rightz = 0*nm, Lz

TF_Sx_R_calculator = collector.Sx("TF_R", savedir+'Sx/', TF, 400*nm, (lefty, leftz), (righty, rightz), freqs, engine)
IF_Sx_L_calculator = collector.Sx("IF_L", savedir+'Sx/', IF, 100*nm, (lefty, leftz), (righty, rightz), freqs, engine)
SF_Sx_L_calculator = collector.Sx("SF_L", savedir+'Sx/', SF, 100*nm, (lefty, leftz), (righty, rightz), freqs, engine)

#------------------------------------------------------------------#
#-------------------- Graphtool object settings -------------------#
#------------------------------------------------------------------#

plot_per = 1000
TFgraphtool = plotter.Graphtool(TF, 'TF', savedir)
IFgraphtool = plotter.Graphtool(IF, 'IF', savedir)
SFgraphtool = plotter.Graphtool(SF, 'SF', savedir)

#------------------------------------------------------------------#
#------------------------ Time loop begins ------------------------#
#------------------------------------------------------------------#

# Save what time the simulation begins.
start_time = datetime.datetime.now()

# time loop begins
for tstep in range(Tsteps):

    # At the start point
    if tstep == 0:
        TF.MPIcomm.Barrier()
        if TF.MPIrank == 0:
            print("Total time step: %d" %(TF.tsteps))
            print(("Size of a total field array : %05.2f Mbytes" %(TF.TOTAL_NUM_GRID_SIZE)))
            print("Simulation start: {}".format(datetime.datetime.now()))
        
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
    IF_Sx_L_calculator.do_RFT(tstep)
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
        TFgraphtool.plot2D3D(Ey, tstep, yidx=TF.Nyc, colordeep=2, stride=2, zlim=2)
        #TFgraphtool.plot2D3D(Ez, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hx, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hy, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hz, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1) 

        Ey = IFgraphtool.gather('Ey')
        #IFgraphtool.plot2D3D(Ex, tstep, yidx=TF.Nyc, colordeep=2., stride=1, zlim=2.)
        IFgraphtool.plot2D3D(Ey, tstep, yidx=IF.Nyc, colordeep=2, stride=2, zlim=2)
        #IFgraphtool.plot2D3D(Ez, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #IFgraphtool.plot2D3D(Hx, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #IFgraphtool.plot2D3D(Hy, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #IFgraphtool.plot2D3D(Hz, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)

        Ey = SFgraphtool.gather('Ey')
        #SFgraphtool.plot2D3D(Ex, tstep, yidx=TF.Nyc, colordeep=2., stride=1, zlim=2.)
        SFgraphtool.plot2D3D(Ey, tstep, yidx=SF.Nyc, colordeep=2., stride=1, zlim=2.)
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

            interval_time = datetime.datetime.now()
            print(("time: %s, step: %05d, %5.2f%%" %(interval_time-start_time, tstep, 100.*tstep/TF.tsteps)))

#------------------------------------------------------------------#
#--------------------------- Data analysis ------------------------#
#------------------------------------------------------------------#

TF_Sx_R_calculator.get_Sx(h5=False)
IF_Sx_L_calculator.get_Sx(h5=False)
SF_Sx_L_calculator.get_Sx(h5=False)

#------------------------------------------------------------------#
#------------------- Record simulation history --------------------#
#------------------------------------------------------------------#

if TF.MPIrank == 0: recording = recorder.Recorder(TF, start_time, "../record/")
