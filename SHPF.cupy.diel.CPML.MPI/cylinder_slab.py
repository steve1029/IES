#!/usr/bin/env python
import os, time, datetime, sys, psutil
import matplotlib
matplotlib.use('Agg')
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.constants import c
import source, space, plotfield, structure, rft

#------------------------------------------------------------------#
#----------------------- Paramter settings ------------------------#
#------------------------------------------------------------------#

savedir = '/home/ldg/2nd_paper/SHPF.cupy.diel.CPML.MPI/'

nm = 1e-9
um = 1e-6

Nx, Ny, Nz = 256, 256, 256
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
np.save("./graph/freqs", freqs)

# Set the type of input source.
Src = source.Gaussian(dt, wvc, spread, pick_pos, dtype=np.float32)
#Src.plot_pulse(Tsteps, freqs, savedir)
#Src = source.Sine(dt, np.float64)
#Src.set_wvlen( 20 * um)

#sys.exit()

#------------------------------------------------------------------#
#-------------------------- Call objects --------------------------#
#------------------------------------------------------------------#

TF = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), dt, Tsteps, np.float32, np.complex64, engine='cupy') # Total field
IF = space.Basic3D((Nx, Ny, Nz), (dx, dy, dz), dt, Tsteps, np.float32, np.complex64, engine='cupy') # Incident field
SF = space.Empty3D((Nx, Ny, Nz), (dx, dy, dz), dt, Tsteps, np.float32, np.complex64, engine='cupy') # Scattered field

TF.malloc()
IF.malloc()
SF.malloc()

# Put structures
srt = (1300*um,    0*um,    0*um)
end = (1500*um, 2560*um, 2560*um)
box = structure.Box(TF, srt, end, 4., 1.)

radius = 100*um
height = (1300*um, 1500*um)
llm = (0*um, 0*um)
dc = (640*um, 640*um)
cylinder = structure.Cylinder_slab(TF, radius, height, llm, dc, 1., 1.)

# Set PML and PBC
TF.set_PML({'x':'+-','y':'','z':''}, 10)
IF.set_PML({'x':'+-','y':'','z':''}, 10)

# Save eps, mu and PML data.
#TF.save_pml_parameters('./')
#TF.save_eps_mu(savedir)

# Set position of Src.
#src_xpos = int(Nx/2)
src_xpos = 20

# plane wave normal to x-axis.
TF.set_src_pos((src_xpos, 1, 1), (src_xpos+1, Ny, Nz))
IF.set_src_pos((src_xpos, 1, 1), (src_xpos+1, Ny, Nz))

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
leftx, rightx = 320*um, 960*um
lefty, righty = 320*um, 960*um
leftz, rightz = 320*um, 960*um

IF_Sx_R_calculator = rft.Sx("IF_Sx_R", "./graph/Sx", IF, rightx, (lefty, leftz), (righty, rightz), freqs, 'cupy')
TF_Sx_R_calculator = rft.Sx("TF_Sx_R", "./graph/Sx", TF, rightx, (lefty, leftz), (righty, rightz), freqs, 'cupy')
SF_Sx_L_calculator = rft.Sx("SF_Sx_L", "./graph/Sx", SF,  leftx, (lefty, leftz), (righty, rightz), freqs, 'cupy')

# Set plotfield options
TFgraphtool = plotfield.Graphtool(TF, 'TF', savedir)
IFgraphtool = plotfield.Graphtool(IF, 'IF', savedir)
SFgraphtool = plotfield.Graphtool(SF, 'SF', savedir)

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

    TF.put_src('Ey', pulse_re, 'soft')
    IF.put_src('Ey', pulse_re, 'soft')

    TF.updateH(tstep)
    IF.updateH(tstep)

    TF.updateE(tstep)
    IF.updateE(tstep)

    SF.get_SF(TF, IF)

    IF_Sx_R_calculator.do_RFT(tstep)
    TF_Sx_R_calculator.do_RFT(tstep)
    SF_Sx_L_calculator.do_RFT(tstep)

    # Plot the field profile
    if tstep % plot_per == 0:

        Ey = TFgraphtool.gather('Ey')
        #TFgraphtool.plot2D3D(Ex, tstep, yidx=TF.Nyc, colordeep=2., stride=1, zlim=2.)
        TFgraphtool.plot2D3D(Ey, tstep, yidx=TF.Nyc, colordeep=2., stride=1, zlim=2.)
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

IF_Sx_R_calculator.get_Sx()
TF_Sx_R_calculator.get_Sx()
SF_Sx_L_calculator.get_Sx()

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
