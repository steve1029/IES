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

savedir = '/home/ldg/script/pyctypes/HPF.rfft.diel.CPML.MPI/'

nm = 1e-9
um = 1e-6

Lx, Ly, Lz = 4096*30*um, 4096*30*um, 2048*30*um
Nx, Ny, Nz = 4096, 512, 256
dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz
Tstep = 8001

courant = 1./4
dt = courant * min(dx,dy,dz) / c

wv_srt = 40*um
wv_end = 50*um
interval = 0.1*um
spread   = 0.3
pick_pos = 2000
plot_per = 1000

# Set the type of input source.
#Src = source.Gaussian(dt, dtype=np.float64)
#Src.wvlen([wv_srt, wv_end, interval, spread])
Src = source.Sine(dt, np.float64)
Src.set_wvlen( 600 * um)

#------------------------------------------------------------------#
#-------------------------- Call objects --------------------------#
#------------------------------------------------------------------#

# Space
Space = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), courant, dt, Tstep, np.float64)

# Slab
#Box1_srt = (0 , Space.Nyc-15, Space.Nzc-15)
#Box1_end = (Nx, Space.Nyc+15, Space.Nzc+15)
#Box = structure.Box(Space, Box1_srt, Box1_end, 4., 1.)

# Set PML
Space.set_pml({'x':'', 'y':'', 'z':''}, 10)

# Save eps, mu and PML data.
#Space.save_PML_parameters('./')
#Space.save_eps_mu(savedir)

# plane wave normal to x
#Space.set_src_pos((src_xpos, 0, 0), (src_xpos+1, Space.Ny, Space.Nz))

# plane wave normal to y
#Space.set_src_pos((0, Space.Nyc, 0), (Space.Nx, Space.Nyc+1, Space.Nz))

# plane wave normal to z
#Space.set_src_pos((0, 0, Space.Nzc), (Space.Nx, Space.Ny, Space.Nzc+1))

# Line source along x axis.
#Space.set_src_pos((0, int(Ny/4), int(Nz/4)), (Space.Nx, int(Ny/4)+1, int(Ny/4)+1))

# Line source along y axis.
#Space.set_src_pos((Space.Nxc, 0, Space.Nzc), (Space.Nxc+1, Ny, Space.Nzc+1))

# Line source along z axis.
#Space.set_src_pos((Space.Nxc, Space.Nyc, 0), (Space.Nxc+1, Space.Nyc+1, Nz))

# Point source at the center
Space.set_src_pos((Space.Nxc, Space.Nyc, Space.Nzc), (Space.Nxc+1, Space.Nyc+1, Space.Nzc+1))

# Set plotfield options
graphtool = plotfield.Graphtool(Space, 'TF', savedir)

# Initialize the core
Space.init_update_equations(omp_on=False)

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
	
	# Gaussian wave.	
	#pulse_re = Src.pulse_re(tstep, pick_pos=pick_pos)
	#pulse_im = Src.pulse_im(tstep, pick_pos=pick_pos)

	# Sine wave.	
	pulse_re = Src.pulse_re(tstep)
	pulse_im = Src.pulse_im(tstep)

	#Space.put_src('Ex_re', 'Ex_im', pulse_re, 0, 'soft')
	Space.put_src('Ey_re', pulse_re, 'soft')
	#Space.put_src('Ez_re', 'Ez_im', pulse_re, 0, 'soft')

	Space.updateH(tstep)
	Space.updateE(tstep)

	# Plot the field profile
	if tstep % plot_per == 0:
		#graphtool.plot2D3D('Ex', tstep, yidx=Space.Nyc, colordeep=.2, stride=2, zlim=.2)
		#graphtool.plot2D3D('Ey', tstep, yidx=Space.Nyc, colordeep=2, stride=2, zlim=2)
		#graphtool.plot2D3D('Ez', tstep, zidx=Space.Nzc, colordeep=.2, stride=2, zlim=.2)
		#graphtool.plot2D3D('Hx', tstep, yidx=Space.Nyc, colordeep=1e-3, stride=2, zlim=1e-3)
		#graphtool.plot2D3D('Hy', tstep, yidx=Space.Nyc, colordeep=1e-3, stride=2, zlim=1e-3)
		#graphtool.plot2D3D('Hz', tstep, yidx=Space.Nyc, colordeep=1e-3, stride=2, zlim=1e-3)

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
		f.write("{:4}\t{:4}\t{:4}\t{:4}\t{:4}\t\t{:4}\t\t{:4}\t\t{:8}\t{:4}\t\t\t\t{:12}\t{:12}\t{:6}\n\n" \
			.format("Node","Nx","Ny","Nz","dx","dy","dz","tsteps","Time","VM/Node(GB)","RM/Node(GB)","OMP"))
		f.close()

	me = psutil.Process(os.getpid())
	me_rssmem_GB = float(me.memory_info().rss)/1024/1024/1024
	me_vmsmem_GB = float(me.memory_info().vms)/1024/1024/1024

	cal_time = finished_time - start_time
	f = open( record_path,'a')
	f.write("{:2d}\t\t{:04d}\t{:04d}\t{:04d}\t{:5.2e}\t{:5.2e}\t{:5.2e}\t{:06d}\t\t{}\t\t{:06.3f}\t\t\t{:06.3f}\n" \
				.format(Space.MPIsize, Space.Nx, Space.Ny, Space.Nz,\
					Space.dx, Space.dy, Space.dz, Space.tsteps, cal_time, me_vmsmem_GB, me_rssmem_GB, Space.omp_on))
	f.close()
	
	print("Simulation finished: {}".format(datetime.datetime.now()))
