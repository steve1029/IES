#!/usr/bin/env python
import os, time, datetime, sys, json
import matplotlib
matplotlib.use('Agg')
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.constants import c
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append("/root/SHPF/")
import source, space, plotter, structure, collector, recorder

#------------------------------------------------------------------#
#--------------------- Space object settings ----------------------#
#------------------------------------------------------------------#

"""Description.

sys.argv[1]: str. method.
sys.argv[2]: int. Time steps.
sys.argv[3]: int. Nx.
sys.argv[4]: int. Ny.
sys.argv[5]: int. Nz.

An execution example.

$ python3 RT_simple_slabs.py FDTD 5001 360 64 64

"""

nm = 1e-9
um = 1e-6

lunit = um
lustr = 'um'

funit = 1e12
fustr = 'THz'

#a = 779.42*lunit
a = 512*lunit

Lx, Ly, Lz = 720*lunit, a, a
Nx, Ny, Nz = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz

courant = 1./4
dt = courant * min(dx,dy,dz) / c
tsteps = int(sys.argv[2])

method = sys.argv[1]
engine = 'cupy'

TF = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), dt, tsteps, np.complex64, np.complex64, method=method, engine=engine)
IF = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), dt, tsteps, np.complex64, np.complex64, method=method, engine=engine)
SF = space.Empty3D((Nx, Ny, Nz), (dx, dy, dz), dt, tsteps, np.complex64, np.complex64, method=method, engine=engine)

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

# Apply mmt to the plane wave normal to x axis.
# phi is the angle between k0 vector and xz-plane.
# theta is the angle between k0cos(phi) and x-axis.
#kx = k0 * np.cos(phi) * np.cos(theta)
#ky = k0 * np.sin(phi)
#kz = k0 * np.cos(phi) * np.sin(theta)

# Apply mmt to plane wave normal to y axis.
# phi is the angle between k0 vector and xy-plane.
# theta is the angle between k0cos(phi) and y-axis.
#kx = k0 * np.cos(phi) * np.sin(theta)
#ky = k0 * np.cos(phi) * np.cos(theta)
#kz = k0 * np.sin(phi)

# Apply mmt to plane wave normal to z axis.
# phi is the angle between k0 vector and yz-plane.
# theta is the angle between k0cos(phi) and z-axis.
#kx = k0 * np.sin(phi)
#ky = k0 * np.cos(phi) * np.sin(theta)
#kz = k0 * np.cos(phi) * np.cos(theta)

mmt = (kx, ky, kz)

########## Gaussian source
#wvc = float(sys.argv[2])*lunit
wvc = 100*lunit
w0 = (2*np.pi*c)/wvc
interval = .2
spread   = 0.08
peak_pos = 2000
ws = w0 * spread
src = source.Gaussian(dt, wvc, spread, peak_pos, dtype=np.float32)

savedir = f'/root/SHPF/graph/simple_2slab_{method}/{int(Lx/lunit):04d}\
{lustr}{int(Ly/lunit):04d}{lustr}{int(Lz/lunit):04d}{lustr}_{Nx:04d}_{Ny:04d}_{Nz:04d}_{tsteps:07d}\
_100{lustr}_200{lustr}_100{lustr}/'

w1 = w0 * (1-spread*2) #  low omega where the amplitude becomes e**(-1).
w2 = w0 * (1+spread*2) # high omega where the amplitude becomes e**(-1).

l1 = 2*np.pi*c / w1 / lunit # the wavelength corresponds to w1.
l2 = 2*np.pi*c / w2 / lunit # the wavelength corresponds to w2.

wvlens = np.arange(l2,l1, interval)*lunit
freqs = c / wvlens
src.plot_pulse(tsteps, freqs, savedir)
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
xsrt = Lx*0.2
setterT1 = source.Setter(TF, (xsrt, 0, 0), (xsrt, Ly, Lz), mmt)
setterI1 = source.Setter(IF, (xsrt, 0, 0), (xsrt, Ly, Lz), mmt)
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

t1 = 160*lunit
#t2 = t1 + a*0.2
#t3 = t2 + a*0.3
#t4 = t3 + a*0.2
t2 = t1 + 100*lunit
t3 = t2 + 200*lunit
t4 = t3 + 100*lunit

srt1 = (t1,  0,  0)
end1 = (t2, Ly, Lz)

srt2 = (t3,  0,  0)
end2 = (t4, Ly, Lz)

eps_r = 4
mu_r = 1
Box1 = structure.Box('dielectric_slab1', TF, srt1, end1, eps_r, mu_r)
Box2 = structure.Box('dielectric_slab2', TF, srt2, end2, eps_r, mu_r)

########## Circle
radius = a / 4
height1 = (t1, t2)
height2 = (t3, t4)
center = np.array([Ly/2, Lz/2])
lcy = Ly/4
lcz = Lz/4

#ac1 = structure.Cylinder3D('air_cylinder1', TF, 'x', radius, height1, center, 1, 1.)
#ac2 = structure.Cylinder3D('air_cylinder2', TF, 'x', radius, height2, center, 1, 1.)

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

TF_Sx_R_calculator = collector.Sx("TF_R", savedir+"Sx/", TF, Lx*0.85, (lefty, leftz), (righty, rightz), freqs, engine)
IF_Sx_R_calculator = collector.Sx("IF_R", savedir+"Sx/", IF, Lx*0.85, (lefty, leftz), (righty, rightz), freqs, engine)
SF_Sx_L_calculator = collector.Sx("SF_L", savedir+"Sx/", SF, Lx*0.15, (lefty, leftz), (righty, rightz), freqs, engine)

cal_per = 1000

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
wvylim = [-0.1, 1.1]
#freqxlim = [600, 700]
freqxlim = [None, None]
freqylim = [-0.1, 1.1]

#------------------------------------------------------------------#
#--------------- Record simulation specifications------------------#
#------------------------------------------------------------------#

np.save(savedir+"freqs.npy", freqs)
np.save(savedir+"wvlens.npy", wvlens)

if TF.MPIrank == 0:

    spacespecs = f"""Space:

    VOLUME of the space: {TF.VOLUME:.2e} m^3
    Size of the space: {int(TF.Lx/lunit):04d}{lustr} x {int(TF.Ly/lunit):04d}{lustr} x {int(TF.Lz/lunit):04d}{lustr}
    The number of cells: {TF.Nx:4d} x {TF.Ny:4d} x {TF.Nz:4d}
    Grid spacing: {TF.dx/lunit:.3f} {lustr}, {TF.dy/lunit:.3f} {lustr}, {TF.dz/lunit:.3f} {lustr}
    """

    sourcespecs=f"""Source:

    SetterT1:
    \tglobal x location: {setterT1.src_xsrt}, {setterT1.src_xend}
    \t local x location: {setterT1.my_src_xsrt}, {setterT1.my_src_xend}

    SetterI1:
    \tglobal x location: {setterI1.src_xsrt}, {setterI1.src_xend}
    \t local x location: {setterI1.my_src_xsrt}, {setterI1.my_src_xend}

    Gaussian center wavelength: {wvc/um:.4f} um.
    Gaussian wavelength discretized per: {interval} {lustr}.
    Gaussian wave pick position at: {peak_pos} Tstep.
    Gaussian angular frequency spread: {spread:.3f} * w0
    Frequency points: {len(freqs)}
    Wavelength range: {round(l2,1)}{lustr} to {round(l1,1)}{lustr}
    """

    plotterspecs=f"""Plotters:

    Plot field profile per {plot_per} time steps.

    Save location: {savedir}
    \nTotal time step: {TF.tsteps}
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

        {Box1.name}:
            global x location: {Box1.gxloc}
             local x location: {Box1.lxloc}
            global y location: {Box1.ysrt}, {Box1.yend}
            global z location: {Box1.zsrt}, {Box1.zend}

        {Box2.name}:
            global x location: {Box2.gxloc}
             local x location: {Box2.lxloc}
            global y location: {Box2.ysrt}, {Box2.yend}
            global z location: {Box2.zsrt}, {Box2.zend}

"""
"""
        {ac1.name}
            global x location: {ac1.gxloc}
             local x location: {ac1.lxloc}
            axis: {ac1.axis}
            radius: {ac1.radius/lunit}{lustr}
            height: {(ac1.height[1] - ac1.height[0])/lunit}{lustr}
            center: {ac1.ry:6.1f}, {ac1.rz:6.1f}

        {ac2.name}
            global x location: {ac2.gxloc}
             local x location: {ac2.lxloc}
            axis: {ac2.axis}
            radius: {ac2.radius/lunit}{lustr}
            height: {(ac2.height[1] - ac2.height[0])/lunit}{lustr}
            center: {ac2.ry:6.1f}, {ac2.rz:6.1f}
"""

collectorspecs=\
f"""
    Collectors:

        {TF_Sx_R_calculator.name}:
            global x location: {TF_Sx_R_calculator.gxloc}
             local x location: {TF_Sx_R_calculator.lxloc}
            global y location: {TF_Sx_R_calculator.ysrt}, {TF_Sx_R_calculator.yend}
            global z location: {TF_Sx_R_calculator.zsrt}, {TF_Sx_R_calculator.zend}

        {IF_Sx_R_calculator.name}:
            global x location: {IF_Sx_R_calculator.gxloc}
             local x location: {IF_Sx_R_calculator.lxloc}
            global y location: {IF_Sx_R_calculator.ysrt}, {IF_Sx_R_calculator.yend}
            global z location: {IF_Sx_R_calculator.zsrt}, {IF_Sx_R_calculator.zend}

        {SF_Sx_L_calculator.name}:
            global x location: {SF_Sx_L_calculator.gxloc}
             local x location: {SF_Sx_L_calculator.lxloc}
            global y location: {SF_Sx_L_calculator.ysrt}, {SF_Sx_L_calculator.yend}
            global z location: {SF_Sx_L_calculator.zsrt}, {SF_Sx_L_calculator.zend}

Calculate Poynting vector per: {cal_per}
"""

print(f'Rank {TF.MPIrank}:')
print(structurespecs)
print(collectorspecs)

#sys.exit()
#------------------------------------------------------------------#
#------------------------ Time loop begins ------------------------#
#------------------------------------------------------------------#

# Save what time the simulation begins.
start_time = datetime.datetime.now()

if TF.MPIrank == 0: f'\nSimulation start: {start_time}'
TF.MPIcomm.Barrier()

# time loop begins
for tstep in range(tsteps+1):

    # At the start point
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
        
        if TF.MPIrank == 0:

            now = datetime.datetime.now()
            #uptime = (now - start_time).strftime('%H:%M:%S')
            uptime = now - start_time
            print(f"runtime: {uptime}, step: {tstep:7d}, {100.*tstep/TF.tsteps:05.2f}%, at {now}." )

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
        #IFgraphtool.plot2D3D(Ex, tstep, yidx=TF.Nyc, colordeep=2, stride=1, zlim=2)
        IFgraphtool.plot2D3D(Ey, tstep, yidx=IF.Nyc, colordeep=2, stride=2, zlim=2)
        #IFgraphtool.plot2D3D(Ez, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #IFgraphtool.plot2D3D(Hx, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #IFgraphtool.plot2D3D(Hy, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #IFgraphtool.plot2D3D(Hz, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)

        Ey = SFgraphtool.gather('Ey')
        #SFgraphtool.plot2D3D(Ex, tstep, yidx=TF.Nyc, colordeep=2., stride=1, zlim=2.)
        SFgraphtool.plot2D3D(Ey, tstep, yidx=SF.Nyc, colordeep=2, stride=2, zlim=2)
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

    if tstep != 0 and tstep % cal_per == 0:

        TF_Sx_R_calculator.get_Sx(tstep, h5=False)
        IF_Sx_R_calculator.get_Sx(tstep, h5=False)
        SF_Sx_L_calculator.get_Sx(tstep, h5=False)

        #print(f'rank {TF.MPIrank:02d}: Sx has been saved at {tstep} time step.')

if TF.MPIrank == 0:

    sim_data = {}
    sim_data["method"] = method
    sim_data["engin"] = engine
    sim_data["time_steps"] = tsteps
    sim_data["courant"] = courant
    sim_data["length_unit"] = lunit
    sim_data["length_unit_str"] = lustr
    sim_data["freq_unit"] = funit
    sim_data["freq_unit_str"] = fustr
    sim_data["dt"] = dt
    sim_data["Nx"] = Nx
    sim_data["Ny"] = Ny
    sim_data["Nz"] = Nz
    sim_data["Lx"] = Lx
    sim_data["Ly"] = Ly
    sim_data["Lz"] = Lz
    sim_data["dx"] = dx
    sim_data["dy"] = dy
    sim_data["dz"] = dz
    sim_data["pml"] = pml
    sim_data["pml_thick"] = thick
    sim_data["bbc"] = bbc
    sim_data["pbc"] = pbc
    sim_data["source"] = "Gaussian"
    sim_data["source_parameters"] = {"wvc":wvc, "w0":w0, "interval":interval, "spread":spread, "peak_pos":peak_pos, "ws":ws, "l1":l1, "l2":l2, "w1":w1, "w2":w2}
    sim_data["savedir"] = savedir

    json_data = json.dumps(sim_data, ensure_ascii=False, indent='\t')

    with open(f"{savedir}sim_data.json", 'w') as outfile: outfile.write(json_data)
