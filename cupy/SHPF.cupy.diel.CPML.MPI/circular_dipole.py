#!/usr/bin/env python
import os, time, datetime, sys, psutil
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import source, space, plotfield, structure
from scipy.constants import c

#------------------------------------------------------------------#
#----------------------- Paramter settings ------------------------#
#------------------------------------------------------------------#

savedir = '/home/ldg/script/pyctypes/FDTD.real.diel.CPML.MPI/'

nm = 1e-9
um = 1e-6

Lx, Ly, Lz = 128*5*um, 128*5*um, 128*5*um
Nx, Ny, Nz = 128, 128, 128
dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz

courant = 1./4
dt = courant * min(dx,dy,dz) / c
Tstep = 3001

wvc = 300*um
interval = 2
spread   = 0.3
pick_pos = 2000
plot_per = 100

wvlens = np.arange(200, 600, interval) * um
freqs = c / wvlens
np.save("./graph/freqs", freqs)

# Set the type of input source.
Sine = source.Sine(dt, np.float64)
Sine.set_wvlen( 150 * um)
Cosine = source.Cosine(dt, np.float64)
Cosine.set_wvlen( 100 * um)

#sys.exit()

#src_xpos = round( 100*um / dx)
#ref_xpos = round(  50*um / dx)
#trs_xpos = round( 900*um / dx)

src_xpos = int(Nx/2)
#ref_xpos = 
#trs_xpos = 

Box1_srt = (round(222*um/dx), round( 0*um/dy), round(  0*um/dz))
Box1_end = (round(272*um/dx), round(96*um/dy), round( 96*um/dz))
#------------------------------------------------------------------#
#-------------------------- Call objects --------------------------#
#------------------------------------------------------------------#

Space = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), courant, dt, Tstep, np.float64)

# Put structures
#Box = structure.Box(Space, Box1_srt, Box1_end, 4., 1.)

# Set PML and PBC
Space.set_PML({'x':'+-','y':'+-','z':'+-'}, 10)
Space.apply_PBC({'y':False,'z':False})

# Save eps, mu and PML data.
#Space.save_PML_parameters('./')
#Space.save_eps_mu(savedir)

# Point source
#Space.set_src_pos((Space.Nxc, Space.Nyc, Space.Nzc), (Space.Nxc+1, Space.Nyc+1, Space.Nzc+1))

# plain wave normal to x.
#Space.set_src_pos((src_xpos, 0, 0), (src_xpos+1, Space.Ny, Space.Nz)) # Plane wave for Ey, x-direction.

# Line source along y axis.
#Space.set_src_pos((src_xpos, 0, Space.Nzc), (src_xpos+1, Space.Ny, Space.Nzc+1))

# Line source along z axis.
Space.set_src_pos((src_xpos, Space.Nyc, 0), (src_xpos+1, Space.Nyc+1, Space.Nz))

# Set plotfield options
graphtool = plotfield.Graphtool(Space, '', savedir)

# initialize the core
Space.init_update_equations(omp_on=True)

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
        
    src_sine = Sine.signal(tstep)
    src_cosine = Cosine.signal(tstep)

    Space.put_src('Ex_re', src_cosine, 'soft')
    Space.put_src('Ey_re', src_sine, 'soft')
    #Space.put_src('Ez_re', 'Ez_im', 0, 0, 'soft')

    #Space.get_src('Ey', tstep)
    #Space.get_ref('Ey', tstep)
    #Space.get_trs('Ey', tstep)

    Space.updateH(tstep)
    Space.updateE(tstep)

    # Plot the field profile
    if tstep % plot_per == 0:

        Ex = graphtool.gather('Ex')
        #graphtool.plot2D3D('Ex', tstep, xidx=Space.Nxc, colordeep=1., stride=2, zlim=1.)
        #graphtool.plot2D3D('Ex', tstep, yidx=Space.Nyc, colordeep=1., stride=2, zlim=1.)
        graphtool.plot2D3D(Ex, tstep, zidx=Space.Nzc, colordeep=1., stride=2, zlim=1.)

        Ey = graphtool.gather('Ey')
        #graphtool.plot2D3D('Ey', tstep, xidx=Space.Nxc, colordeep=1., stride=2, zlim=1.)
        #graphtool.plot2D3D('Ey', tstep, yidx=Space.Nyc, colordeep=1., stride=2, zlim=1.)
        graphtool.plot2D3D(Ey, tstep, zidx=Space.Nzc, colordeep=1., stride=2, zlim=1.)

        graphtool.plot2D3D(Ey/Ex, tstep, what='phase', zidx=Space.Nzc, colordeep=1., stride=2, zlim=1.)
        #Ez = graphtool.gather('Ez')
        #graphtool.plot2D3D('Ez', tstep, zidx=Space.Nzc, colordeep=2., stride=2, zlim=2.)

        if Space.MPIrank == 0:

            interval_time = datetime.datetime.now()
            print(("time: %s, step: %05d, %5.2f%%" %(interval_time-start_time, tstep, 100.*tstep/Space.tsteps)))

#Space.save_RT()

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
