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

Lx, Ly, Lz = 540*30*um, 512*30*um, 512*30*um
Nx, Ny, Nz = 540, 512, 512
dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz

courant = 1./4
dt = courant * min(dx,dy,dz) / c
Tstep = 2001

wv_srt = 40*um
wv_end = 50*um
interval = 0.1*um
spread   = 0.3
pick_pos = 2000
plot_per = 100

# Set the type of input source.
#Src = source.Gaussian(dt, dtype=np.float64)
#Src.wvlen([wv_srt, wv_end, interval, spread])
#Src.plot_pulse(Tstep, pick_pos, savedir)
Src = source.Sine(dt, np.float64)
Src.set_wvlen( 600 * nm)

#------------------------------------------------------------------#
#-------------------------- Call objects --------------------------#
#------------------------------------------------------------------#

Space = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), courant, dt, Tstep, np.float64)

# Put structures
#Box = structure.Box(Space, Box1_srt, Box1_end, 4., 1.)

# Set PML and PBC
Space.set_PML({'x':'','y':'','z':''}, 10)
Space.apply_PBC({'y':False,'z':False})

# Save eps, mu and PML data.
#Space.save_PML_parameters('./')
#Space.save_eps_mu(savedir)

# Set position of Src, Ref and Trs.
Space.set_src_pos((Space.Nxc, Space.Nyc, Space.Nzc), (Space.Nxc+1, Space.Nyc+1, Space.Nzc+1))

# Set plotfield options
graphtool = plotfield.Graphtool(Space, 'TF', savedir)

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
        
    # Gaussian source
    #pulse_re = Src.pulse_re(tstep, pick_pos)
    #pulse_im = Src.pulse_im(tstep, pick_pos)

    # Sine source
    pulse_re = Src.pulse_re(tstep)
    pulse_im = Src.pulse_im(tstep)

    #Space.put_src('Ex_re', 'Ex_im', pulse_re, 0, 'soft')
    #Space.put_src('Ey_re', 'Ey_im', pulse_re, pulse_im, 'soft')
    Space.put_src('Ey_re', pulse_re, 'soft')
    #Space.put_src('Ey_re', 'Ey_im', pulse_re, 0, 'hard')
    #Space.put_src('Ez_re', 'Ez_im', pulse_re, 0, 'soft')
    #Space.put_src('Ez_re', 'Ez_im', 0, 0, 'soft')

    #Space.get_src('Ey', tstep)
    #Space.get_ref('Ey', tstep)
    #Space.get_trs('Ey', tstep)

    Space.updateH(tstep)
    Space.updateE(tstep)

    # Plot the field profile
    if tstep % plot_per == 0:
        #graphtool.plot2D3D('Ex', tstep, xidx=Space.Nxc, colordeep=6., stride=2, zlim=6.)
        #graphtool.plot2D3D('Ey', tstep, yidx=Space.Nyc, colordeep=2., stride=2, zlim=2.)
        #graphtool.plot2D3D('Ez', tstep, zidx=Space.Nzc, colordeep=2., stride=2, zlim=2.)

        if Space.MPIrank == 0:

            interval_time = datetime.datetime.now()
            print(("time: %s, step: %05d, %5.2f%%" %(interval_time-start_time, tstep, 100.*tstep/Space.tsteps)))

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
