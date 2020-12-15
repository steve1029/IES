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

Nx, Ny, Nz = 128, 128, 128
dx, dy, dz = 10*um, 10*um, 10*um
Lx, Ly, Lz = Nx*dx, Ny*dy, Nz*dz

courant = 1./4
dt = courant * min(dx,dy,dz) / c
Tsteps = int(sys.argv[1])

wvc = 300*um
interval = 2
spread   = 0.3
pick_pos = 1000
plot_per = 1000

wvlens = np.arange(200,600, interval)*um
freqs = c / wvlens
np.save("../graph/freqs", freqs)

TF = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), dt, Tsteps, np.complex64, np.complex64, method='FDTD', engine='cupy')

TF.malloc()

########## Set PML and PBC
TF.set_PML({'x':'+-','y':'','z':''}, 10)

region = {'x':False, 'y':False, 'z':True}
TF.apply_BPBC(region, BBC=True, PBC=False)

########## Save eps, mu and PML data.
#TF.save_pml_parameters('./')
#TF.save_eps_mu(savedir)

#------------------------------------------------------------------#
#--------------------- Source object settings ---------------------#
#------------------------------------------------------------------#

########## Gaussian source
#smth = source.Smoothing(dt, 1000)
#src = source.Gaussian(dt, wvc, spread, pick_pos, dtype=np.float32)
#src.plot_pulse(Tsteps, freqs, savedir)
#wvlen = 300*um

########## Sine source
#src = source.Sine(dt, np.float64)
#wvlen = 300*um
#src.set_wvlen(wvlen)

########## Harmonic source
#src = source.Harmonic(dt)
#wvlen = 300*um
#src.set_wvlen(wvlen)

########## Delta source
src = source.Delta(1000)
wvlen = 300*um

########## Momentum of the source.
# mmt for plane wave normal to x axis
# phi is the angle between k0 vector and xz-plane.
# theta is the angle between k0cos(phi) and x-axis.
k0 = 2*np.pi / wvlen
phi, theta = 0*np.pi/180, -30*np.pi/180
#phi, theta = 0, 0
kx = k0 * np.cos(phi) * np.cos(theta)
ky = k0 * np.sin(phi)
kz = k0 * np.cos(phi) * np.sin(theta)

mmt = (kx, ky, kz)

########## Plane wave normal to x-axis.
#TF.set_src((xpos, 0, 0), (xpos+1, Ny, Nz), mmt)

########## Line src along y-axis.
setter1 = source.Setter(TF, (500*um, 0, 700*um), (510*um, Ly, 710*um), mmt)
setter2 = source.Setter(TF, (600*um, 0, 300*um), (610*um, Ly, 310*um), mmt)
setter3 = source.Setter(TF, (300*um, 0, 400*um), (310*um, Ly, 410*um), mmt)
setter4 = source.Setter(TF, (700*um, 0, 800*um), (710*um, Ly, 810*um), mmt)
setter5 = source.Setter(TF, (650*um, 0,1100*um), (660*um, Ly,1110*um), mmt)

########## Line src along z-axis.
#TF.set_src_pos((xpos, ypos, 0), (xpos+1, ypos+1, Nz))

########## Point src at the center.
#setter = source.Setter(TF, (xpos, ypos, zpos), (xpos+dx, ypos+dy, zpos+dz), mmt)

#------------------------------------------------------------------#
#-------------------- Structure object settings -------------------#
#------------------------------------------------------------------#

srt = ( 800*um,    0*um,    0*um)
end = (1000*um, 1280*um, 1280*um)
box = structure.Box(TF, srt, end, 4., 1.)

radius = 100*um
height = (800*um, 1000*um)
llm = (0*um, 0*um)
dc = (640*um, 640*um)
cylinder = structure.Cylinder_slab(TF, radius, height, llm, dc, 1., 1.)

#------------------------------------------------------------------#
#-------------------- Collector object settings -------------------#
#------------------------------------------------------------------#

loc = (900*um, 640*um, 640*um)
field_at_point = collector.FieldAtPoint("fap1", "../graph/fap", TF, loc, freqs, 'cupy')

#------------------------------------------------------------------#
#-------------------- Graphtool object settings -------------------#
#------------------------------------------------------------------#

# Set plotfield options
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
        
    # pulse for gaussian wave
    #pulse_re = src.pulse_re(tstep) * smth.apply(tstep)
    pulse_re = src.apply(tstep)

    setter1.put_src('Ey', pulse_re, 'soft')
    setter2.put_src('Ey', pulse_re, 'soft')
    setter3.put_src('Ey', pulse_re, 'soft')
    setter4.put_src('Ey', pulse_re, 'soft')

    TF.updateH(tstep)
    TF.updateE(tstep)

    field_at_point.get_field(tstep)

    # Plot the field profile
    if tstep % plot_per == 0:

        Ey = TFgraphtool.gather('Ey')
        #TFgraphtool.plot2D3D(Ex, tstep, yidx=TF.Nyc, colordeep=2., stride=1, zlim=2.)
        TFgraphtool.plot2D3D(Ey, tstep, yidx=TF.Nyc, colordeep=1., stride=2, zlim=1.)
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
field_at_point.plot_spectrum()

#------------------------------------------------------------------#
#------------------- Record simulation history --------------------#
#------------------------------------------------------------------#

if TF.MPIrank == 0:

    # Simulation finished time
    finished_time = datetime.datetime.now()

    # Record simulation size and operation time
    if not os.path.exists("./record") : os.mkdir("./record")
    record_path = "./record/record_%s.txt" %(datetime.date.today())

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
