#!/usr/bin/env python
import os, time, datetime, sys, psutil
import matplotlib
matplotlib.use('Agg')
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.constants import c
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import source, space, plotfield, structure, collector

#------------------------------------------------------------------#
#--------------------- Space object settings ----------------------#
#------------------------------------------------------------------#

savedir = '/home/ldg/2nd_paper/SHPF.cupy.diel.CPML.MPI/'

nm = 1e-9
um = 1e-6

Lx, Ly, Lz = 574/32*nm, 574*nm, 574*nm
Nx, Ny, Nz = 8, 256, 256
dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz 

courant = 1./4
dt = courant * min(dx,dy,dz) / c
Tsteps = int(sys.argv[1])

TF = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), dt, Tsteps, np.complex64, np.complex64, method='SHPF', engine='cupy')

TF.malloc()

########## Set PML and PBC
TF.set_PML({'x':'','y':'','z':'+-'}, 10)

region = {'x':True, 'y':True, 'z':False}
TF.apply_BPBC(region, BBC=True, PBC=False)

########## Save PML data.
#TF.save_pml_parameters('./')

#------------------------------------------------------------------#
#--------------------- Source object settings ---------------------#
#------------------------------------------------------------------#

########## Gaussian source
#wvc = 200*nm
#interval = 2
#spread   = 0.3
#pick_pos = 1000
#wvlens = np.arange(100,500, interval)*nm
#freqs = c / wvlens
#np.save("../graph/freqs", freqs)
#src = source.Gaussian(dt, wvc, spread, pick_pos, dtype=np.float32)
#src.plot_pulse(Tsteps, freqs, savedir)
#wvlen = wvc 

########## Sine source
#smth = source.Smoothing(dt, 1000)
#src = source.Sine(dt, np.float64)
#wvlen = 300*um
#src.set_wvlen(wvlen)

########## Harmonic source
smth = source.Smoothing(dt, 1000)
src = source.Harmonic(dt)
wvlen = 200*nm
src.set_wvlen(wvlen)

########## Delta source
#src = source.Delta(1000)
#wvlen = 2*Lz

########## Momentum of the source.
k0 = 2*np.pi / wvlen
phi, theta = 0*np.pi/180, 30*np.pi/180

# mmt for plane wave normal to x axis
# phi is the angle between k0 vector and xz-plane.
# theta is the angle between k0cos(phi) and x-axis.
#kx = k0 * np.cos(phi) * np.cos(theta)
#ky = k0 * np.sin(phi)
#kz = k0 * np.cos(phi) * np.sin(theta)

# mmt for plane wave normal to y axis
# phi is the angle between k0 vector and xy-plane.
# theta is the angle between k0cos(phi) and y-axis.
#kx = k0 * np.cos(phi) * np.sin(theta)
#ky = k0 * np.cos(phi) * np.cos(theta)
#kz = k0 * np.sin(phi)

# mmt for plane wave normal to z axis
# phi is the angle between k0 vector and yz-plane.
# theta is the angle between k0cos(phi) and z-axis.
kx = k0 * np.sin(phi)
ky = k0 * np.cos(phi) * np.sin(theta)
kz = k0 * np.cos(phi) * np.cos(theta)

mmt = (kx, ky, kz)

########## Plane wave normal to x-axis.
#setter = source.Setter(TF, (200*um, 0, 0), (210*um, Ly, Lz), mmt)

########## Plane wave normal to y-axis.
#setter = source.Setter(TF, (0, 63*um, 0), (Lx, 64*um, Lz), mmt)

########## Plane wave normal to z-axis.
setter1 = source.Setter(TF, (0, 0, 50*nm), (Lx, Ly, 50*nm+dx), mmt)

########## Line src along x-axis.
#setter1 = source.Setter(TF, (0,  100*nm,  100*nm), (Lx,  105*nm, 105*nm), mmt)
#setter2 = source.Setter(TF, (0*um,  30*um, 200*um), (31*um,  31*um, 201*um), mmt)
#setter3 = source.Setter(TF, (0*um, 127*um, 127*um), (31*um, 128*um, 128*um), mmt)
#setter4 = source.Setter(TF, (0*um, 200*um, 200*um), (31*um, 201*um, 201*um), mmt)

########## Line src along y-axis.
#setter1 = source.Setter(TF, (640*um, 0, 640*um), (645*um, Ly, 645*um), mmt)
#setter2 = source.Setter(TF, (400*um, 0, 300*um), (410*um, Ly, 310*um), mmt)
#setter3 = source.Setter(TF, (400*um, 0, 400*um), (410*um, Ly, 410*um), mmt)
#setter4 = source.Setter(TF, (400*um, 0, 500*um), (410*um, Ly, 510*um), mmt)

########## Line src along z-axis.
#TF.set_src_pos((xpos, ypos, 0), (xpos+1, ypos+1, Nz))

########## Point src at the center.
#setter = source.Setter(TF, (xpos, ypos, zpos), (xpos+dx, ypos+dy, zpos+dz), mmt)

#------------------------------------------------------------------#
#-------------------- Structure object settings -------------------#
#------------------------------------------------------------------#

########## Box
#srt = ( 800*um,    0*um,    0*um)
#end = (1000*um, 1280*um, 1280*um)
#box = structure.Box(TF, srt, end, 4., 1.)

########## Cylinder
radius = 114.8*nm
height = (0, Lx)
center1 = (Ly/2, Lz/2)
#center2 = (  0, Lz)
#center3 = ( Lx, 0)
#center4 = ( Lx, Lz)
#cylinder1 = structure.Cylinder(TF, 'x', radius, height, center1, 8.9, 1.)
#cylinder2 = structure.Cylinder(TF, 'y', radius, height, center2, 8.9, 1.)
#cylinder3 = structure.Cylinder(TF, 'y', radius, height, center3, 8.9, 1.)
#cylinder4 = structure.Cylinder(TF, 'y', radius, height, center4, 8.9, 1.)

########## Save eps, mu data.
TF.save_eps_mu(savedir)

#------------------------------------------------------------------#
#-------------------- Collector object settings -------------------#
#------------------------------------------------------------------#

loc = (Lx/2, 200*nm, 200*nm)
field_at_point = collector.FieldAtPoint("fap1", savedir+"graph/", TF, loc, 'cupy')

#------------------------------------------------------------------#
#-------------------- Graphtool object settings -------------------#
#------------------------------------------------------------------#

# Set plotfield options
plot_per = 1000
TFgraphtool = plotfield.Graphtool(TF, 'TF', savedir)

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
    #pulse_re = src.pulse_re(tstep)

    # pulse for Sine or Harmonic wave.
    pulse_re = src.signal(tstep) * smth.apply(tstep)

    # pulse for Delta function wave.
    #pulse_re = src.apply(tstep)

    #setter.put_src('Ex', pulse_re, 'soft')

    setter1.put_src('Ex', pulse_re, 'soft')
    """
    setter2.put_src('Ex', pulse_re, 'soft')
    setter3.put_src('Ex', pulse_re, 'soft')
    setter4.put_src('Ex', pulse_re, 'soft')

    setter1.put_src('Ey', pulse_re, 'soft')
    setter2.put_src('Ey', pulse_re, 'soft')
    setter3.put_src('Ey', pulse_re, 'soft')
    setter4.put_src('Ey', pulse_re, 'soft')

    setter1.put_src('Ez', pulse_re, 'soft')
    setter2.put_src('Ez', pulse_re, 'soft')
    setter3.put_src('Ez', pulse_re, 'soft')
    setter4.put_src('Ez', pulse_re, 'soft')
    """

    TF.updateH(tstep)
    TF.updateE(tstep)

    field_at_point.get_field(tstep)

    # Plot the field profile
    if tstep % plot_per == 0:

        Ex = TFgraphtool.gather('Ex')
        TFgraphtool.plot2D3D(Ex, tstep, xidx=TF.Nxc, colordeep=2., stride=3, zlim=2.)
        #TFgraphtool.plot2D3D(Ey, tstep, yidx=TF.Nyc, colordeep=1., stride=2, zlim=1.)
        #TFgraphtool.plot2D3D(Ez, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hx, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hy, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hz, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)

        #Ey = TFgraphtool.gather('Ey')
        #TFgraphtool.plot2D3D(Ex, tstep, yidx=TF.Nyc, colordeep=2., stride=1, zlim=2.)
        #TFgraphtool.plot2D3D(Ey, tstep, yidx=TF.Nyc, colordeep=1., stride=2, zlim=1.)
        #TFgraphtool.plot2D3D(Ez, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hx, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hy, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hz, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)

        #Ey = IFgraphtool.gather('Ey')
        #IFgraphtool.plot2D3D(Ex, tstep, yidx=TF.Nyc, colordeep=2., stride=1, zlim=2.)
        #IFgraphtool.plot2D3D(Ey, tstep, yidx=IF.Nyc, colordeep=2., stride=1, zlim=2.)
        #IFgraphtool.plot2D3D(Ez, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #IFgraphtool.plot2D3D(Hx, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #IFgraphtool.plot2D3D(Hy, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #IFgraphtool.plot2D3D(Hz, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)

        #Ey = SFgraphtool.gather('Ey')
        #SFgraphtool.plot2D3D(Ex, tstep, yidx=TF.Nyc, colordeep=2., stride=1, zlim=2.)
        #SFgraphtool.plot2D3D(Ey, tstep, yidx=SF.Nyc, colordeep=2., stride=1, zlim=2.)
        #SFgraphtool.plot2D3D(Ez, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #SFgraphtool.plot2D3D(Hx, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #SFgraphtool.plot2D3D(Hy, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #SFgraphtool.plot2D3D(Hz, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)

        if TF.MPIrank == 0:

            interval_time = datetime.datetime.now()
            print(("time: %s, step: %05d, %5.2f%%" %(interval_time-start_time, tstep, 100.*tstep/TF.tsteps)))

#------------------------------------------------------------------#
#--------------------------- Data analysis ------------------------#
#------------------------------------------------------------------#

field_at_point.get_spectrum()

#------------------------------------------------------------------#
#------------------- Record simulation history --------------------#
#------------------------------------------------------------------#

if TF.MPIrank == 0:

    # Simulation finished time
    finished_time = datetime.datetime.now()

    # Record simulation size and operation time
    if not os.path.exists("../record") : os.mkdir("../record")
    record_path = "../record/record_%s.txt" %(datetime.date.today())

    if not os.path.exists(record_path):
        f = open( record_path,'a')
        f.write("{:4}\t{:4}\t{:4}\t{:4}\t{:4}\t\t{:4}\t\t{:4}\t\t{:8}\t{:4}\t\t\t\t{:12}\t{:12}\n\n" \
            .format("Node","Nx","Ny","Nz","dx","dy","dz","tsteps","Time","VM/Node(GB)","RM/Node(GB)"))
        f.close()

    me = psutil.Process(os.getpid())
    me_rssmem_GB = float(me.memory_info().rss)/1024/1024/1024
    me_vmsmem_GB = float(me.memory_info().vms)/1024/1024/1024

    cal_time = finished_time - start_time
    f = open( record_path,'a')
    f.write("{:2d}\t\t{:04d}\t{:04d}\t{:04d}\t{:5.2e}\t{:5.2e}\t{:5.2e}\t{:06d}\t\t{}\t\t{:06.3f}\t\t\t{:06.3f}\n" \
                .format(TF.MPIsize, TF.Nx, TF.Ny, TF.Nz,\
                    TF.dx, TF.dy, TF.dz, TF.tsteps, cal_time, me_vmsmem_GB, me_rssmem_GB))
    f.close()
    
    print("Simulation finished: {}".format(datetime.datetime.now()))
