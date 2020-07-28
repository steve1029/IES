#!/usr/bin/env python
import os, time, datetime, sys, psutil
import numpy as np
from mpi4py import MPI
import matplotlib
import matplotlib.pyplot as plt
from scipy.constants import c
import source, space, plotfield, structure, rft

#------------------------------------------------------------------#
#----------------------- Paramter settings ------------------------#
#------------------------------------------------------------------#

savedir = '/home/ldg/script/pyctypes/HPF.rfft.diel.CPML.MPI/'

nm = 1e-9
um = 1e-6

Nx, Ny, Nz = 512, 128, 128
dx, dy, dz = 10*um, 40*um, 40*um
Lx, Ly, Lz = Nx*dx, Ny*dy, Nz*dz

courant = 1./4
dt = courant * min(dx,dy,dz) / c
Tsteps = 5001

wvc = 300*um
interval = 2
spread = 0.3
pick_pos = 1000
plot_per = 100

wvlens = np.arange(200,600,interval)*um
freqs = c / wvlens
np.save("./graph/freqs", freqs)
# Set the type of input source.
Src = source.Gaussian(dt, wvc, spread, pick_pos, dtype=np.float64)
#Src.plot_pulse(Tsteps, freqs, savedir)
#Src = source.Sine(dt, np.float64)
#Src.set_wvlen( 20 * um)

#sys.exit()

#------------------------------------------------------------------#
#-------------------------- Call objects --------------------------#
#------------------------------------------------------------------#

TF = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), courant, dt, Tsteps, np.float64) # Total field
IF = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), courant, dt, Tsteps, np.float64) # Incident field
SF = space.Empty3D((Nx, Ny, Nz), (dx, dy, dz), courant, dt, Tsteps, np.float64) # Scattered field

# Put structures
Ball = structure.Sphere(TF, (int(Nx/2), int(Ny/2), int(Nz/2)), 10*dx, 4., 1.)

# Set PML and PBC
TF.set_pml({'x':'+-','y':'+-','z':'+-'}, 10)
IF.set_pml({'x':'+-','y':'+-','z':'+-'}, 10)

# Save eps, mu and PML data.
TF.save_pml_parameters('./')
#TF.save_eps_mu(savedir)

# Set position of Src.
#src_xpos = int(Nx/2)
src_xpos = 20
src_ypos = int(Ny/2)
src_zpos = int(Nz/2)

# plane wave normal to x-axis.
TF.set_src_pos((src_xpos, 0, 0), (src_xpos+1, Ny, Nz))
IF.set_src_pos((src_xpos, 0, 0), (src_xpos+1, Ny, Nz))

# plane wave normal to y-axis.
#TF.set_src_pos((1, src_ypos, 0), (Nx, src_ypos+1, Nz))

# plane wave normal to z-axis.
#TF.set_src_pos((1, 0, src_zpos), (Nx, Ny, src_zpos+1))

# line src along x-axis.
#TF.set_src_pos((0, src_ypos, src_zpos), (Nx, src_ypos+1, src_zpos+1))
#IF.set_src_pos((1, src_ypos, src_zpos), (Nx, src_ypos+1, src_zpos+1))

# line src along y-axis.
#TF.set_src_pos((src_xpos, 0, src_zpos), (src_xpos+1, Ny, src_zpos+1))
#IF.set_src_pos((src_xpos, 0, src_zpos), (src_xpos+1, Ny, src_zpos+1))

# line src along z-axis.
#TF.set_src_pos((src_xpos, src_ypos, 0), (src_xpos+1, src_ypos+1, Nz))
#IF.set_src_pos((src_xpos, src_ypos, 0), (src_xpos+1, src_ypos+1, Nz))

# point src at the center.
#TF.set_src_pos((src_xpos, src_ypos, src_zpos), (src_xpos+1, src_ypos+1, src_zpos+1))
#IF.set_src_pos((src_xpos, src_ypos, src_zpos), (src_xpos+1, src_ypos+1, src_zpos+1))

# Set S calculator
leftx, rightx = int(Nx*1/4), int(Nx*3/4)
lefty, righty = int(Ny*1/4), int(Ny*3/4)
leftz, rightz = int(Nz*1/4), int(Nz*3/4)

IF_Sx_R_calculator = rft.Sx("Sx_IF_R", "./graph/Sx", IF, (rightx, lefty, leftz), (rightx+1, righty, rightz), freqs, True)

SF_Sx_L_calculator = rft.Sx("Sx_SF_L", "./graph/Sx", SF, (leftx , lefty, leftz), (leftx +1, righty, rightz), freqs, True)
SF_Sx_R_calculator = rft.Sx("Sx_SF_R", "./graph/Sx", SF, (rightx, lefty, leftz), (rightx+1, righty, rightz), freqs, True)

SF_Sy_L_calculator = rft.Sy("Sy_SF_L", "./graph/Sy", SF, (leftx, lefty , leftz), (rightx, lefty +1, rightz), freqs, True)
SF_Sy_R_calculator = rft.Sy("Sy_SF_R", "./graph/Sy", SF, (leftx, righty, leftz), (rightx, righty+1, rightz), freqs, True)

SF_Sz_L_calculator = rft.Sz("Sz_SF_L", "./graph/Sz", SF, (leftx, lefty, leftz ), (rightx, righty, leftz +1), freqs, True)
SF_Sz_R_calculator = rft.Sz("Sz_SF_R", "./graph/Sz", SF, (leftx, lefty, rightz), (rightx, righty, rightz+1), freqs, True)

"""
TF_Sx_L_calculator = rft.Sx("Sx_TF_L", "./graph/Sx", TF, (leftx , lefty, leftz), (leftx +1, righty, rightz), freqs, True)
TF_Sx_R_calculator = rft.Sx("Sx_TF_R", "./graph/Sx", TF, (rightx, lefty, leftz), (rightx+1, righty, rightz), freqs, True)

TF_Sy_L_calculator = rft.Sy("Sy_TF_L", "./graph/Sy", TF, (leftx, lefty , leftz), (rightx, lefty +1, rightz), freqs, True)
TF_Sy_R_calculator = rft.Sy("Sy_TF_R", "./graph/Sy", TF, (leftx, righty, leftz), (rightx, righty+1, rightz), freqs, True)

TF_Sz_L_calculator = rft.Sz("Sz_TF_L", "./graph/Sz", TF, (leftx, lefty, leftz ), (rightx, righty, leftz +1), freqs, True)
TF_Sz_R_calculator = rft.Sz("Sz_TF_R", "./graph/Sz", TF, (leftx, lefty, rightz), (rightx, righty, rightz+1), freqs, True)
"""
# Set plotfield options
TFgraphtool = plotfield.Graphtool(TF, 'TF', savedir)
IFgraphtool = plotfield.Graphtool(IF, 'IF', savedir)
SFgraphtool = plotfield.Graphtool(SF, 'SF', savedir)

# initialize the core
TF.init_update_equations(omp_on=True)
IF.init_update_equations(omp_on=True)

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
	pulse_re = Src.pulse_re(tstep, pick_pos=pick_pos)
	pulse_im = Src.pulse_im(tstep, pick_pos=pick_pos)

	# pulse for sine wave
	#pulse_re = Src.pulse_re(tstep)
	#pulse_im = Src.pulse_im(tstep)

	#TF.put_src('Ex_re', 'Ex_im', pulse_re, 0, 'soft')
	TF.put_src('Ey_re', pulse_re, 'soft')
	#TF.put_src('Ez_re', 'Ez_im', pulse_re, 0, 'soft')

	#IF.put_src('Ex_re', 'Ex_im', pulse_re, 0, 'soft')
	IF.put_src('Ey_re', pulse_re, 'soft')
	#IF.put_src('Ez_re', 'Ez_im', pulse_re, 0, 'soft')

	TF.updateH(tstep)
	IF.updateH(tstep)

	TF.updateE(tstep)
	IF.updateE(tstep)

	SF.get_SF(TF, IF)

	IF_Sx_R_calculator.do_RFT(tstep)

	SF_Sx_L_calculator.do_RFT(tstep)
	SF_Sx_R_calculator.do_RFT(tstep)

	SF_Sy_L_calculator.do_RFT(tstep)
	SF_Sy_R_calculator.do_RFT(tstep)

	SF_Sz_L_calculator.do_RFT(tstep)
	SF_Sz_R_calculator.do_RFT(tstep)

	"""
	TF_Sx_L_calculator.do_RFT(tstep)
	TF_Sx_R_calculator.do_RFT(tstep)

	TF_Sy_L_calculator.do_RFT(tstep)
	TF_Sy_R_calculator.do_RFT(tstep)

	TF_Sz_L_calculator.do_RFT(tstep)
	TF_Sz_R_calculator.do_RFT(tstep)
	"""
	# Plot the field profile
	if tstep % plot_per == 0:
		#TFgraphtool.plot2D3D('Ex', tstep, yidx=TF.Nyc, colordeep=2, stride=1, zlim=2)
		TFgraphtool.plot2D3D('Ey', tstep, yidx=TF.Nyc, colordeep=2, stride=1, zlim=2)
		#TFgraphtool.plot2D3D('Ez', tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
		#TFgraphtool.plot2D3D('Hx', tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
		#TFgraphtool.plot2D3D('Hy', tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
		#TFgraphtool.plot2D3D('Hz', tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)

		#IFgraphtool.plot2D3D('Ex', tstep, yidx=TF.Nyc, colordeep=2, stride=1, zlim=2)
		IFgraphtool.plot2D3D('Ey', tstep, yidx=IF.Nyc, colordeep=2, stride=1, zlim=2)
		#IFgraphtool.plot2D3D('Ez', tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
		#IFgraphtool.plot2D3D('Hx', tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
		#IFgraphtool.plot2D3D('Hy', tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
		#IFgraphtool.plot2D3D('Hz', tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)

		#SFgraphtool.plot2D3D('Ex', tstep, yidx=TF.Nyc, colordeep=2, stride=1, zlim=2)
		SFgraphtool.plot2D3D('Ey', tstep, yidx=SF.Nyc, colordeep=2, stride=1, zlim=2)
		#SFgraphtool.plot2D3D('Ez', tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
		#SFgraphtool.plot2D3D('Hx', tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
		#SFgraphtool.plot2D3D('Hy', tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)
		#SFgraphtool.plot2D3D('Hz', tstep, xidx=TF.Nxc, colordeep=.1, stride=1, zlim=.1)

		if TF.MPIrank == 0:

			interval_time = datetime.datetime.now()
			print(("time: %s, step: %05d, %5.2f%%" %(interval_time-start_time, tstep, 100.*tstep/TF.tsteps)))

IF_Sx_R_calculator.get_Sx()

SF_Sx_L_calculator.get_Sx()
SF_Sx_R_calculator.get_Sx()

SF_Sy_L_calculator.get_Sy()
SF_Sy_R_calculator.get_Sy()

SF_Sz_L_calculator.get_Sz()
SF_Sz_R_calculator.get_Sz()

"""
TF_Sx_L_calculator.get_Sx()
TF_Sx_R_calculator.get_Sx()

TF_Sy_L_calculator.get_Sy()
TF_Sy_R_calculator.get_Sy()

TF_Sz_L_calculator.get_Sz()
TF_Sz_R_calculator.get_Sz()
"""
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
