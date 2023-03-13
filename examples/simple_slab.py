import time, os, datetime, sys, ctypes, psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import c, mu_0, epsilon_0
import numpy as np
import cupy as cp
import source, space, plotfield, structure, rft

#------------------------------------------------------------------#
#----------------------- Paramter settings ------------------------#
#------------------------------------------------------------------#
savedir = '/home/ldg/2nd_paper/SHPF.cupy.diel.CPML.MPI/'

nm = 1e-9
um = 1e-6

Lx, Ly, Lz = 128*10*um, 128*10*um, 128*10*um
Nx, Ny, Nz = 128, 128, 128
dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz

courant = 1./4
dt = courant * min(dx,dy,dz) / c
Tsteps = 3001

wvc = 300*um
interval = 2
spread   = 0.3
pick_pos = 1000
plot_per = 100

wvlens = np.arange(200, 600, interval) * um
freqs = c / wvlens
np.save("./graph/freqs", freqs)

# Set the type of input source.
Src = source.Gaussian(dt, wvc, spread, pick_pos, np.float64)
Src.plot_pulse(Tsteps, freqs, savedir)
#Src = source.Sine(dt, np.float64)
#Src.set_wvlen( 50 * um)

#sys.exit()

#------------------------------------------------------------------#
#-------------------------- Call objects --------------------------#
#------------------------------------------------------------------#

TF = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), dt, Tsteps, np.float32, np.complex64, engine='cupy', method='SHPF') # Total field
IF = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), dt, Tsteps, np.float32, np.complex64, engine='cupy', method='SHPF') # Incident field
SF = space.Empty3D((Nx, Ny, Nz), (dx, dy, dz), dt, Tsteps, np.float32, np.complex64, engine='cupy', method='SHPF') # Scattered field

TF.malloc()
IF.malloc()
SF.malloc()

# Put structures
Box1_srt = (800*um,    0*um,    0*um)
Box1_end = (900*um, 1280*um, 1280*um)
box = structure.Box(TF, Box1_srt, Box1_end, 4., 1.)

#radius = 160*um
#height = (500*um, 700*um)
#center = (640*um, 640*um)
#cylinder = structure.Cylinder(TF, radius, height, center, 4., 1.)

# Set PML and PBC
TF.set_PML({'x':'+-','y':'','z':''}, 10)
IF.set_PML({'x':'+-','y':'','z':''}, 10)

# Save eps, mu and PML data.
#TF.save_PML_parameters('./')
#TF.save_eps_mu(savedir)

# Set source position.
#src_xpos = int(Nx/2)
src_xpos = 50

# plain wave normal to x.
TF.set_src_pos((src_xpos, 0, 0), (src_xpos+1, Ny, Nz))
IF.set_src_pos((src_xpos, 0, 0), (src_xpos+1, Ny, Nz))

# Line source along y axis.
#TF.set_src_pos((src_xpos, 0, TF.Nzc), (src_xpos+1, TF.Ny, TF.Nzc+1))

# Line source along z axis.
#TF.set_src_pos((src_xpos, TF.Nyc, 0), (src_xpos+1, TF.Nyc+1, TF.Nz))

# Set Poynting vector calculator.
leftx, rightx = 300*um, 900*um
lefty, righty = 300*um, 900*um
leftz, rightz = 300*um, 900*um

IF_Sx_R_getter = rft.Sx("IF_Sx_R", "./graph/Sx", IF, rightx, (lefty, leftz), (righty, rightz), freqs, 'cupy')
TF_Sx_R_getter = rft.Sx("TF_Sx_R", "./graph/Sx", TF, rightx, (lefty, leftz), (righty, rightz), freqs, 'cupy')
SF_Sx_L_getter = rft.Sx("SF_Sx_L", "./graph/Sx", SF,  leftx, (lefty, leftz), (righty, rightz), freqs, 'cupy')

# Set plotfield options
TFgraphtool = plotfield.Graphtool(TF, 'TF', savedir)
IFgraphtool = plotfield.Graphtool(IF, 'IF', savedir)
SFgraphtool = plotfield.Graphtool(SF, 'SF', savedir)

# Save what time the simulation begins.
start_time = datetime.datetime.now()

# time loop begins
for tstep in range(TF.tsteps):

    # At the start point
    if tstep == 0 and TF.MPIrank == 0:
        print("Total time step: %d" %(TF.tsteps))
        print(("Size of a total field array : %05.2f Mbytes" %(TF.TOTAL_NUM_GRID_SIZE)))
        print("Simulation start: {}".format(datetime.datetime.now()))
    
    pulse_re = Src.pulse_re(tstep, pick_pos)

    TF.put_src('Ey', pulse_re, 'soft')
    IF.put_src('Ey', pulse_re, 'soft')
    #TF.put_src('Ez', pulse_re, 'soft')

    TF.updateH(tstep)
    IF.updateH(tstep)

    TF.updateE(tstep)
    IF.updateE(tstep)

    SF.get_SF(TF, IF)

    IF_Sx_R_getter.do_RFT(tstep)
    TF_Sx_R_getter.do_RFT(tstep)
    SF_Sx_L_getter.do_RFT(tstep)

    # Plot the field profile
    if tstep % plot_per == 0:

        Ey = TFgraphtool.gather('Ey')
        #TFgraphtool.plot2D3D(Ex, tstep, yidx=TF.Nyc, colordeep=2, stride=1, zlim=2)
        TFgraphtool.plot2D3D(Ey, tstep, yidx=TF.Nyc, colordeep=2, stride=1, zlim=2)
        #TFgraphtool.plot2D3D(Ez, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hx, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hy, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #TFgraphtool.plot2D3D(Hz, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)

        Ey = IFgraphtool.gather('Ey')
        #IFgraphtool.plot2D3D(Ex, tstep, yidx=TF.Nyc, colordeep=2, stride=1, zlim=2)
        IFgraphtool.plot2D3D(Ey, tstep, yidx=IF.Nyc, colordeep=2, stride=1, zlim=2)
        #IFgraphtool.plot2D3D(Ez, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #IFgraphtool.plot2D3D(Hx, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #IFgraphtool.plot2D3D(Hy, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #IFgraphtool.plot2D3D(Hz, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)

        Ey = SFgraphtool.gather('Ey')
        #SFgraphtool.plot2D3D(Ex, tstep, yidx=TF.Nyc, colordeep=2, stride=1, zlim=2)
        SFgraphtool.plot2D3D(Ey, tstep, yidx=SF.Nyc, colordeep=2, stride=1, zlim=2)
        #SFgraphtool.plot2D3D(Ez, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #SFgraphtool.plot2D3D(Hx, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #SFgraphtool.plot2D3D(Hy, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
        #SFgraphtool.plot2D3D(Hz, tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)

        if TF.MPIrank == 0:

            interval_time = datetime.datetime.now()
            print(("time: %s, step: %05d, %5.2f%%" %(interval_time-start_time, tstep, 100.*tstep/TF.tsteps)))

IF_Sx_R_getter.get_Sx()
TF_Sx_R_getter.get_Sx()
SF_Sx_L_getter.get_Sx()

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
