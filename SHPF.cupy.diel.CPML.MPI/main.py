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
Tstep = 2001

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
Src.plot_pulse(Tstep, freqs, savedir)
#Src = source.Sine(dt, np.float64)
#Src.set_wvlen( 50 * um)

#------------------------------------------------------------------#
#-------------------------- Call objects --------------------------#
#------------------------------------------------------------------#

Space = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), dt, Tstep, np.float32, np.complex64, method='SHPF', engine='cupy')
Space.malloc()

# Put structures
Box1_srt = (round(222*um/dx), round( 0*um/dy), round(  0*um/dz))
Box1_end = (round(272*um/dx), round(96*um/dy), round( 96*um/dz))
#Box = structure.Box(Space, Box1_srt, Box1_end, 4., 1.)

# Set PML and PBC
Space.set_PML({'x':'+-','y':'+-','z':''}, 10)
#Space.set_PML({'x':'+-','y':'+-','z':'+-'}, 10)

# Save eps, mu and PML data.
#Space.save_PML_parameters('./')
#Space.save_eps_mu(savedir)

# Set source position.
#src_xpos = int(Nx/2)
src_xpos = 40
src_ypos = 40

# plain wave normal to x.
#Space.set_src_pos((src_xpos, 0, 0), (src_xpos+1, Ny, Nz)) # Plane wave for Ey, x-direction.

# plain wave normal to y.
#Space.set_src_pos((1, src_ypos, 0), (Nx, src_ypos+1, Nz)) # Plane wave for Ez, y-direction.

# Line source along y axis.
#Space.set_src_pos((src_xpos, 0, Space.Nzc), (src_xpos+1, Space.Ny, Space.Nzc+1))

# Line source along z axis.
Space.set_src_pos((src_xpos, Space.Nyc, 0), (src_xpos+1, Space.Nyc+1, Space.Nz))

# Set Poynting vector calculator.
leftx, rightx = int(Nx/4), int(Nx*3/4)
lefty, righty = int(Ny/4), int(Ny*3/4)
leftz, rightz = int(Nz/4), int(Nz*3/4)

#Sx_R_calculator = rft.Sx("SF_R", "./graph/Sx", Space, (rightx, lefty, leftz), (rightx+1, righty, rightz), freqs, 'cupy')

# Set plotfield options
graphtool = plotfield.Graphtool(Space, 'TF', savedir)

# Save what time the simulation begins.
start_time = datetime.datetime.now()

# time loop begins
for tstep in range(Space.tsteps):

    # At the start point
    if tstep == 0:
        Space.MPIcomm.Barrier()
        if Space.MPIrank == 0:
            print("Total time step: %d" %(Space.tsteps))
            print(("Size of a total field array : %05.2f Mbytes" %(Space.TOTAL_NUM_GRID_SIZE)))
            print("Simulation start: {}".format(datetime.datetime.now()))
        
    pulse_re = Src.pulse_re(tstep, pick_pos)
    #pulse_im = Src.pulse_im(tstep, pick_pos)

    #Space.put_src('Ey', pulse_re, 'soft')
    Space.put_src('Ez', pulse_re, 'soft')

    Space.updateH(tstep)
    Space.updateE(tstep)

    #Sx_R_calculator.do_RFT(tstep)

    # Plot the field profile
    if tstep % plot_per == 0:
        #graphtool.plot2D3D('Ex', tstep, xidx=Space.Nxc, colordeep=6., stride=2, zlim=6.)

        #Ey = graphtool.gather('Ey')
        #graphtool.plot2D3D(Ey, tstep, yidx=Space.Nyc, colordeep=1., stride=2, zlim=1.)
        
        Ez = graphtool.gather('Ez')
        graphtool.plot2D3D(Ez, tstep, zidx=Space.Nzc, colordeep=1., stride=2, zlim=1.)

        if Space.MPIrank == 0:

            interval_time = datetime.datetime.now()
            print(("time: %s, step: %05d, %5.2f%%" %(interval_time-start_time, tstep, 100.*tstep/Space.tsteps)))

#Sx_R_calculator.get_Sx()

if Space.MPIrank == 0:

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
                .format(Space.MPIsize, Space.Nx, Space.Ny, Space.Nz,\
                    Space.dx, Space.dy, Space.dz, Space.tsteps, cal_time, me_vmsmem_GB, me_rssmem_GB))
    f.close()
    
    print("Simulation finished: {}".format(datetime.datetime.now()))
