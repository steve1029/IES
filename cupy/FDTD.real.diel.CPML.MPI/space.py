import numpy as np
import matplotlib.pyplot as plt
import time, os, datetime, sys, ctypes
from mpi4py import MPI
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import c, mu_0, epsilon_0

class Basic3D(object):
	
	def __init__(self, grid, gridgap, courant, dt, tsteps, dtype, **kwargs):
		"""Create Simulation Space.

			ex) Space.grid((128,128,600), (50*nm,50*nm,5*nm), dtype=np.float64)

		PARAMETERS
		----------
		grid : tuple
			define the x,y,z grid.

		gridgap : tuple
			define the dx, dy, dz.

		dtype : class numpy dtype
			choose np.float32 or np.float64

		kwargs : string
			
			supported arguments
			-------------------

			courant : float
				Set the courant number. For FDTD, default is 1./2

		RETURNS
		-------
		None
		"""

		self.nm = 1e-9
		self.um = 1e-6	

		self.dtype	  = dtype
		self.MPIcomm  = MPI.COMM_WORLD
		self.MPIrank  = self.MPIcomm.Get_rank()
		self.MPIsize  = self.MPIcomm.Get_size()
		self.hostname = MPI.Get_processor_name()

		assert len(grid)	== 3, "Simulation grid should be a tuple with length 3."
		assert len(gridgap) == 3, "Argument 'gridgap' should be a tuple with length 3."

		self.tsteps = tsteps		

		self.grid = grid
		self.Nx   = self.grid[0]
		self.Ny   = self.grid[1]
		self.Nz   = self.grid[2]
		self.TOTAL_NUM_GRID	= self.Nx * self.Ny * self.Nz
		self.TOTAL_NUM_GRID_SIZE = (self.dtype(1).nbytes * self.TOTAL_NUM_GRID) / 1024 / 1024
		
		self.Nxc = int(self.Nx / 2)
		self.Nyc = int(self.Ny / 2)
		self.Nzc = int(self.Nz / 2)
		
		self.gridgap = gridgap
		self.dx = self.gridgap[0]
		self.dy = self.gridgap[1]
		self.dz = self.gridgap[2]

		self.Lx = self.Nx * self.dx
		self.Ly = self.Ny * self.dy
		self.Lz = self.Nz * self.dz

		self.VOLUME = self.Lx * self.Ly * self.Lz

		if self.MPIrank == 0:
			print("VOLUME of the space: {:.2e}" .format(self.VOLUME))
			print("Number of grid points: {:5d} x {:5d} x {:5d}" .format(self.Nx, self.Ny, self.Nz))
			print("Grid spacing: {:.3f} nm, {:.3f} nm, {:.3f} nm" .format(self.dx/self.nm, self.dy/self.nm, self.dz/self.nm))

		self.MPIcomm.Barrier()

		self.courant = courant

		for key, value in kwargs.items():
			if key == 'courant': self.courant = value

		self.dt = dt
		self.maxdt = 1. / c / np.sqrt( (1./self.dx)**2 + (1./self.dy)**2 + (1./self.dz)**2 )

		assert (c * self.dt * np.sqrt( (1./self.dx)**2 + (1./self.dy)**2 + (1./self.dz)**2 )) < 1.

		"""
		For more details about maximum dt in the Hybrid PSTD-FDTD method, see
		Combining the FDTD and PSTD methods, Y.F.Leung, C.H. Chan,
		Microwave and Optical technology letters, Vol.23, No.4, November 20 1999.
		"""

		self.myPMLregion_x = None
		self.myPMLregion_y = None
		self.myPMLregion_z = None
		self.myPBCregion_x = False
		self.myPBCregion_y = False
		self.myPBCregion_z = False

		assert self.dt < self.maxdt, "Time interval is too big so that causality is broken. Lower the courant number."
		assert float(self.Nx) % self.MPIsize == 0., "Nx must be a multiple of the number of nodes."
		
		############################################################################
		################# Set the loc_grid each node should possess ################
		############################################################################

		self.myNx	  = int(self.Nx/self.MPIsize)
		self.loc_grid = (self.myNx, self.Ny, self.Nz)

		self.Ex_re = np.zeros(self.loc_grid, dtype=self.dtype)
		self.Ey_re = np.zeros(self.loc_grid, dtype=self.dtype)
		self.Ez_re = np.zeros(self.loc_grid, dtype=self.dtype)

		self.Hx_re = np.zeros(self.loc_grid, dtype=self.dtype)
		self.Hy_re = np.zeros(self.loc_grid, dtype=self.dtype)
		self.Hz_re = np.zeros(self.loc_grid, dtype=self.dtype)
		###############################################################################

		self.diffxEy_re = np.zeros(self.loc_grid, dtype=self.dtype)
		self.diffxEz_re = np.zeros(self.loc_grid, dtype=self.dtype)
		self.diffyEx_re = np.zeros(self.loc_grid, dtype=self.dtype)
		self.diffyEz_re = np.zeros(self.loc_grid, dtype=self.dtype)
		self.diffzEx_re = np.zeros(self.loc_grid, dtype=self.dtype)
		self.diffzEy_re = np.zeros(self.loc_grid, dtype=self.dtype)

		self.diffxHy_re = np.zeros(self.loc_grid, dtype=self.dtype)
		self.diffxHz_re = np.zeros(self.loc_grid, dtype=self.dtype)
		self.diffyHx_re = np.zeros(self.loc_grid, dtype=self.dtype)
		self.diffyHz_re = np.zeros(self.loc_grid, dtype=self.dtype)
		self.diffzHx_re = np.zeros(self.loc_grid, dtype=self.dtype)
		self.diffzHy_re = np.zeros(self.loc_grid, dtype=self.dtype)
		############################################################################

		self.eps_Ex = np.ones(self.loc_grid, dtype=self.dtype) * epsilon_0
		self.eps_Ey = np.ones(self.loc_grid, dtype=self.dtype) * epsilon_0
		self.eps_Ez = np.ones(self.loc_grid, dtype=self.dtype) * epsilon_0

		self.mu_Hx	= np.ones(self.loc_grid, dtype=self.dtype) * mu_0
		self.mu_Hy	= np.ones(self.loc_grid, dtype=self.dtype) * mu_0
		self.mu_Hz	= np.ones(self.loc_grid, dtype=self.dtype) * mu_0

		self.econ_Ex = np.zeros(self.loc_grid, dtype=self.dtype)
		self.econ_Ey = np.zeros(self.loc_grid, dtype=self.dtype)
		self.econ_Ez = np.zeros(self.loc_grid, dtype=self.dtype)

		self.mcon_Hx = np.zeros(self.loc_grid, dtype=self.dtype)
		self.mcon_Hy = np.zeros(self.loc_grid, dtype=self.dtype)
		self.mcon_Hz = np.zeros(self.loc_grid, dtype=self.dtype)

		###############################################################################
		####################### Slices of zgrid that each node got ####################
		###############################################################################
		
		self.myNx_slices = []
		self.myNx_indice = []

		for rank in range(self.MPIsize):

			xsrt = (rank  ) * self.myNx
			xend = (rank+1) * self.myNx

			self.myNx_slices.append(slice(xsrt, xend))
			self.myNx_indice.append(	 (xsrt, xend))

		self.MPIcomm.Barrier()
		#print("rank {:>2}:\tmy xindex: {},\tmy xslice: {}" \
		#		.format(self.MPIrank, self.myNx_indice[self.MPIrank], self.myNx_slices[self.MPIrank]))

	def set_PML(self, region, npml):

		self.PMLregion	= region
		self.npml		= npml
		self.PMLgrading = 2 * self.npml

		self.rc0   = 1.e-16								# reflection coefficient
		self.imp   = np.sqrt(mu_0/epsilon_0)			# impedence
		self.gO    = 3.									# gradingOrder
		self.sO    = 3.									# scalingOrder
		self.bdw_x = (self.PMLgrading-1) * self.dx		# PML thickness along x (Boundarywidth)
		self.bdw_y = (self.PMLgrading-1) * self.dy		# PML thickness along y
		self.bdw_z = (self.PMLgrading-1) * self.dz		# PML thickness along z

		self.PMLsigmamaxx = -(self.gO+1) * np.log(self.rc0) / (2*self.imp*self.bdw_x)
		self.PMLsigmamaxy = -(self.gO+1) * np.log(self.rc0) / (2*self.imp*self.bdw_y)
		self.PMLsigmamaxz = -(self.gO+1) * np.log(self.rc0) / (2*self.imp*self.bdw_z)

		self.PMLkappamaxx = 1.
		self.PMLkappamaxy = 1.
		self.PMLkappamaxz = 1.

		self.PMLalphamaxx = 0.02
		self.PMLalphamaxy = 0.02
		self.PMLalphamaxz = 0.02

		self.PMLsigmax = np.zeros(self.PMLgrading)
		self.PMLalphax = np.zeros(self.PMLgrading)
		self.PMLkappax = np.ones (self.PMLgrading)

		self.PMLsigmay = np.zeros(self.PMLgrading)
		self.PMLalphay = np.zeros(self.PMLgrading)
		self.PMLkappay = np.ones (self.PMLgrading)

		self.PMLsigmaz = np.zeros(self.PMLgrading)
		self.PMLalphaz = np.zeros(self.PMLgrading)
		self.PMLkappaz = np.ones (self.PMLgrading)

		self.PMLbx = np.zeros(self.PMLgrading)
		self.PMLby = np.zeros(self.PMLgrading)
		self.PMLbz = np.zeros(self.PMLgrading)

		self.PMLax = np.zeros(self.PMLgrading)
		self.PMLay = np.zeros(self.PMLgrading)
		self.PMLaz = np.zeros(self.PMLgrading)

		#------------------------------------------------------------------------------------------------#
		#------------------------------- Grading kappa, sigma and alpha ---------------------------------#
		#------------------------------------------------------------------------------------------------#

		for key, value in self.PMLregion.items():

			if	 key == 'x' and value != '':

				self.psi_eyx_p_re = np.zeros((npml, self.Ny, self.Nz), dtype=self.dtype)
				self.psi_ezx_p_re = np.zeros((npml, self.Ny, self.Nz), dtype=self.dtype)
				self.psi_hyx_p_re = np.zeros((npml, self.Ny, self.Nz), dtype=self.dtype)
				self.psi_hzx_p_re = np.zeros((npml, self.Ny, self.Nz), dtype=self.dtype)

				self.psi_eyx_m_re = np.zeros((npml, self.Ny, self.Nz), dtype=self.dtype)
				self.psi_ezx_m_re = np.zeros((npml, self.Ny, self.Nz), dtype=self.dtype)
				self.psi_hyx_m_re = np.zeros((npml, self.Ny, self.Nz), dtype=self.dtype)
				self.psi_hzx_m_re = np.zeros((npml, self.Ny, self.Nz), dtype=self.dtype)

				for i in range(self.PMLgrading):

					loc  = np.float64(i) * self.dx / self.bdw_x

					self.PMLsigmax[i] = self.PMLsigmamaxx * (loc **self.gO)
					self.PMLkappax[i] = 1 + ((self.PMLkappamaxx-1) * (loc **self.gO))
					self.PMLalphax[i] = self.PMLalphamaxx * ((1-loc) **self.sO)

			elif key == 'y' and value != '':

				self.psi_exy_p_re = np.zeros((self.myNx, npml, self.Nz), dtype=self.dtype)
				self.psi_ezy_p_re = np.zeros((self.myNx, npml, self.Nz), dtype=self.dtype)
				self.psi_hxy_p_re = np.zeros((self.myNx, npml, self.Nz), dtype=self.dtype)
				self.psi_hzy_p_re = np.zeros((self.myNx, npml, self.Nz), dtype=self.dtype)

				self.psi_exy_m_re = np.zeros((self.myNx, npml, self.Nz), dtype=self.dtype)
				self.psi_ezy_m_re = np.zeros((self.myNx, npml, self.Nz), dtype=self.dtype)
				self.psi_hxy_m_re = np.zeros((self.myNx, npml, self.Nz), dtype=self.dtype)
				self.psi_hzy_m_re = np.zeros((self.myNx, npml, self.Nz), dtype=self.dtype)

				for i in range(self.PMLgrading):

					loc  = np.float64(i) * self.dy / self.bdw_y

					self.PMLsigmay[i] = self.PMLsigmamaxy * (loc **self.gO)
					self.PMLkappay[i] = 1 + ((self.PMLkappamaxy-1) * (loc **self.gO))
					self.PMLalphay[i] = self.PMLalphamaxy * ((1-loc) **self.sO)

			elif key == 'z' and value != '':

				self.psi_exz_p_re = np.zeros((self.myNx, self.Ny, npml), dtype=self.dtype)
				self.psi_eyz_p_re = np.zeros((self.myNx, self.Ny, npml), dtype=self.dtype)
				self.psi_hxz_p_re = np.zeros((self.myNx, self.Ny, npml), dtype=self.dtype)
				self.psi_hyz_p_re = np.zeros((self.myNx, self.Ny, npml), dtype=self.dtype)

				self.psi_exz_m_re = np.zeros((self.myNx, self.Ny, npml), dtype=self.dtype)
				self.psi_eyz_m_re = np.zeros((self.myNx, self.Ny, npml), dtype=self.dtype)
				self.psi_hxz_m_re = np.zeros((self.myNx, self.Ny, npml), dtype=self.dtype)
				self.psi_hyz_m_re = np.zeros((self.myNx, self.Ny, npml), dtype=self.dtype)

				for i in range(self.PMLgrading):

					loc  = np.float64(i) * self.dz / self.bdw_z

					self.PMLsigmaz[i] = self.PMLsigmamaxz * (loc **self.gO)
					self.PMLkappaz[i] = 1 + ((self.PMLkappamaxz-1) * (loc **self.gO))
					self.PMLalphaz[i] = self.PMLalphamaxz * ((1-loc) **self.sO)

		#------------------------------------------------------------------------------------------------#
		#--------------------------------- Get 'b' and 'a' for CPML theory ------------------------------#
		#------------------------------------------------------------------------------------------------#

		if 'x' in self.PMLregion.keys() and self.PMLregion.get('x') != '':
			self.PMLbx = np.exp(-(self.PMLsigmax/self.PMLkappax + self.PMLalphax) * self.dt / epsilon_0)
			self.PMLax = self.PMLsigmax / (self.PMLsigmax*self.PMLkappax + self.PMLalphax*self.PMLkappax**2) * (self.PMLbx - 1.)

		if 'y' in self.PMLregion.keys() and self.PMLregion.get('y') != '':
			self.PMLby = np.exp(-(self.PMLsigmay/self.PMLkappay + self.PMLalphay) * self.dt / epsilon_0)
			self.PMLay = self.PMLsigmay / (self.PMLsigmay*self.PMLkappay + self.PMLalphay*self.PMLkappay**2) * (self.PMLby - 1.)

		if 'z' in self.PMLregion.keys() and self.PMLregion.get('z') != '':
			self.PMLbz = np.exp(-(self.PMLsigmaz/self.PMLkappaz + self.PMLalphaz) * self.dt / epsilon_0)
			self.PMLaz = self.PMLsigmaz / (self.PMLsigmaz*self.PMLkappaz + self.PMLalphaz*self.PMLkappaz**2) * (self.PMLbz - 1.)

		return

	def save_pml_parameters(self, path):
		"""Save PML parameters to check"""

		if self.MPIrank == 0:
			try: import h5py
			except ImportError as e:
				print("Please install h5py and hdfviewer")
				return
			
			f = h5py.File(path+'pml_parameters.h5', 'w')

			for key,value in self.PMLregion.items():
				if key == 'x':
					f.create_dataset('PMLsigmax' ,	data=self.PMLsigmax)
					f.create_dataset('PMLkappax' ,	data=self.PMLkappax)
					f.create_dataset('PMLalphax' ,	data=self.PMLalphax)
					f.create_dataset('PMLbx',		data=self.PMLbx)
					f.create_dataset('PMLax',		data=self.PMLax)
				elif key == 'y':
					f.create_dataset('PMLsigmay' ,	data=self.PMLsigmay)
					f.create_dataset('PMLkappay' ,	data=self.PMLkappay)
					f.create_dataset('PMLalphay' ,	data=self.PMLalphay)
					f.create_dataset('PMLby',		data=self.PMLby)
					f.create_dataset('PMLay',		data=self.PMLay)
				elif key == 'z':
					f.create_dataset('PMLsigmaz' ,	data=self.PMLsigmaz)
					f.create_dataset('PMLkappaz' ,	data=self.PMLkappaz)
					f.create_dataset('PMLalphaz' ,	data=self.PMLalphaz)
					f.create_dataset('PMLbz',		data=self.PMLbz)
					f.create_dataset('PMLaz',		data=self.PMLaz)

		else: pass
			
		self.MPIcomm.Barrier()
		
		return

	def save_eps_mu(self, path):
		"""Save eps_r and mu_r to check

		"""

		try: import h5py
		except ImportError as e:
			print("rank {:>2}\tPlease install h5py and hdfviewer to save data." .format(self.MPIrank))
			return
		save_dir = path+'eps_mu/'		

		if os.path.exists(save_dir) == False: os.mkdir(save_dir)
		else: pass

		f = h5py.File(save_dir+'eps_r_mu_r_rank{:>02d}.h5' .format(self.MPIrank), 'w')

		f.create_dataset('eps_Ex',	data=self.eps_Ex)
		f.create_dataset('eps_Ey',	data=self.eps_Ey)
		f.create_dataset('eps_Ez',	data=self.eps_Ez)
		f.create_dataset( 'mu_Hx',	data=self. mu_Hx)
		f.create_dataset( 'mu_Hy',	data=self. mu_Hy)
		f.create_dataset( 'mu_Hz',	data=self. mu_Hz)
			
		self.MPIcomm.Barrier()

		return

	def apply_PBC(self, region):
		"""Specify the boundary to apply Periodic Boundary Condition.
		Phase compensation is not developed yet.

		PARAMETERS
		----------
		region : dictionary
			ex) {'x':'','y':'+-','z':'+-'}

		RETURNS
		-------
		None
		"""

		for key, value in region.items():

			if	 key == 'x':

				if	 value == '+'				   : raise ValueError("input '+-' or '-+'.")
				elif value == '-'				   : raise ValueError("input '+-' or '-+'.")
				elif value == '+-' or value == '-+':

					if	 self.MPIrank == 0				 : self.myPBCregion_x = '-'
					elif self.MPIrank == (self.MPIsize-1): self.myPBCregion_x = '+'

			elif key == 'y':

				if value == True: self.myPBCregion_y = True
				elif value == False: self.myPBCregion_y = False
				else: raise ValueError("Choose True or False")

			elif key == 'z':
	
				if value == True: self.myPBCregion_z = True
				elif value == False: self.myPBCregion_z = False
				else: raise ValueError("Choose True or False")

		self.MPIcomm.Barrier()
		#print("PBC region of rank: {}, x: {}, y: {}, z: {}" \
		#		.format(self.MPIrank, self.myPBCregion_x, self.myPBCregion_y, self.myPBCregion_z))

	def set_ref_trs_pos(self, ref_pos, trs_pos):
		"""Set x position to collect srcref and trs

		PARAMETERS
		----------
		pos : tuple
				x index of ref position and trs position

		RETURNS
		-------
		None
		"""

		assert self.tsteps != None, "Set time tstep first!"

		if ref_pos >= 0: self.ref_pos = ref_pos
		else		   : self.ref_pos = ref_pos + self.Nx
		if trs_pos >= 0: self.trs_pos = trs_pos
		else		   : self.trs_pos = trs_pos + self.Nx

		#----------------------------------------------------#
		#-------- All rank should know who gets trs ---------#
		#----------------------------------------------------#

		for rank in range(self.MPIsize) : 

			srt = self.myNx_indice[rank][0]
			end = self.myNx_indice[rank][1]

			if self.trs_pos >= srt and self.trs_pos < end : 
				self.who_get_trs	= rank 
				self.local_trs_xpos = self.trs_pos - srt

		#----------------------------------------------------#
		#------- All rank should know who gets the ref ------#
		#----------------------------------------------------#

		for rank in range(self.MPIsize):
			srt = self.myNx_indice[rank][0]
			end = self.myNx_indice[rank][1]

			if self.ref_pos >= srt and self.ref_pos < end :
				self.who_get_ref	= rank
				self.local_ref_xpos = self.ref_pos - srt 

		#----------------------------------------------------#
		#-------- Ready to put ref and trs collector --------#
		#----------------------------------------------------#

		self.MPIcomm.Barrier()

		if	 self.MPIrank == self.who_get_trs:
			#print("rank %d: I collect trs from %d which is essentially %d in my own grid."\
			#		 %(self.MPIrank, self.trs_pos, self.local_trs_xpos))
			self.trs_re = np.zeros(self.tsteps, dtype=self.dtype) 

		if self.MPIrank == self.who_get_ref: 
			#print("rank %d: I collect ref from %d which is essentially %d in my own grid."\
			#		 %(self.MPIrank, self.ref_pos, self.local_ref_xpos))
			self.ref_re = np.zeros(self.tsteps, dtype=self.dtype)

	def set_src_pos(self, src_srt, src_end):
		"""Set the position, type of the source and field.

		PARAMETERS
		----------
		src_srt : tuple
		src_end   : tuple

			A tuple which has three ints as its elements.
			The elements defines the position of the source in the field.
			
			ex)
				1. point source
					src_srt: (30, 30, 30), src_end: (30, 30, 30)
				2. line source
					src_srt: (30, 30, 0), src_end: (30, 30, Space.Nz)
				3. plane wave
					src_srt: (30,0,0), src_end: (30, Space.Ny, Space.Nz)

		RETURNS
		-------
		None
		"""

		assert len(src_srt) == 3, "src_srt argument is a list or tuple with length 3."
		assert len(src_end) == 3, "src_end argument is a list or tuple with length 3."

		self.who_put_src = None

		self.src_srt  = src_srt
		self.src_xsrt = src_srt[0]
		self.src_ysrt = src_srt[1]
		self.src_zsrt = src_srt[2]

		self.src_end  = src_end
		self.src_xend = src_end[0]
		self.src_yend = src_end[1]
		self.src_zend = src_end[2]

		#----------------------------------------------------------------------#
		#--------- All rank should know who put src to plot src graph ---------#
		#----------------------------------------------------------------------#

		self.MPIcomm.Barrier()
		for rank in range(self.MPIsize):

			my_xsrt = self.myNx_indice[rank][0]
			my_xend = self.myNx_indice[rank][1]

			# case 1. x position of source is fixed.
			if self.src_xsrt == (self.src_xend-1):

				if self.src_xsrt >= my_xsrt and self.src_xend <= my_xend:
					self.who_put_src   = rank

					if self.MPIrank == self.who_put_src:
						self.my_src_xsrt = self.src_xsrt - my_xsrt
						self.my_src_xend = self.src_xend - my_xsrt

						self.src_re = np.zeros(self.tsteps, dtype=self.dtype)

						#print("rank{:>2}: src_xsrt : {}, my_src_xsrt: {}, my_src_xend: {}"\
						#		.format(self.MPIrank, self.src_xsrt, self.my_src_xsrt, self.my_src_xend))
					else:
						pass
						#print("rank {:>2}: I don't put source".format(self.MPIrank))

				else: continue

			# case 2. x position of source has range.
			elif self.src_xsrt < self.src_xend:
				assert self.MPIsize == 1

				self.who_put_src = 0
				self.my_src_xsrt = self.src_xsrt
				self.my_src_xend = self.src_xend

				self.src_re = np.zeros(self.tsteps, dtype=self.dtype)

			# case 3. x position of source is reversed.
			elif self.src_xsrt > self.src_xend:
				raise ValueError("src_end[0] is bigger than src_srt[0]")

			else:
				raise IndexError("x position of src is not defined!")

	def put_src(self, where_re, pulse_re, put_type):
		"""Put source at the designated postion set by set_src_pos method.
		
		PARAMETERS
		----------	
		where : string
			ex)
				'Ex_re' or 'ex_re'
				'Ey_re' or 'ey_re'
				'Ez_re' or 'ez_re'

		pulse : float
			float returned by source.pulse_re.

		put_type : string
			'soft' or 'hard'

		"""
		#------------------------------------------------------------#
		#--------- Put the source into the designated field ---------#
		#------------------------------------------------------------#

		self.put_type = put_type

		self.where_re = where_re
		
		self.pulse_re = self.dtype(pulse_re)

		if self.MPIrank == self.who_put_src:

			x = slice(self.my_src_xsrt, self.my_src_xend)
			y = slice(self.   src_ysrt, self.	src_yend)
			z = slice(self.   src_zsrt, self.	src_zend)
			
			if	 self.put_type == 'soft':

				if (self.where_re == 'Ex_re') or (self.where_re == 'ex_re'): self.Ex_re[x,y,z] += self.pulse_re
				if (self.where_re == 'Ey_re') or (self.where_re == 'ey_re'): self.Ey_re[x,y,z] += self.pulse_re
				if (self.where_re == 'Ez_re') or (self.where_re == 'ez_re'): self.Ez_re[x,y,z] += self.pulse_re
				if (self.where_re == 'Hx_re') or (self.where_re == 'hx_re'): self.Hx_re[x,y,z] += self.pulse_re
				if (self.where_re == 'Hy_re') or (self.where_re == 'hy_re'): self.Hy_re[x,y,z] += self.pulse_re
				if (self.where_re == 'Hz_re') or (self.where_re == 'hz_re'): self.Hz_re[x,y,z] += self.pulse_re

			elif self.put_type == 'hard':
	
				if (self.where_re == 'Ex_re') or (self.where_re == 'ex_re'): self.Ex_re[x,y,z] = self.pulse_re
				if (self.where_re == 'Ey_re') or (self.where_re == 'ey_re'): self.Ey_re[x,y,z] = self.pulse_re
				if (self.where_re == 'Ez_re') or (self.where_re == 'ez_re'): self.Ez_re[x,y,z] = self.pulse_re
				if (self.where_re == 'Hx_re') or (self.where_re == 'hx_re'): self.Hx_re[x,y,z] = self.pulse_re
				if (self.where_re == 'Hy_re') or (self.where_re == 'hy_re'): self.Hy_re[x,y,z] = self.pulse_re
				if (self.where_re == 'Hz_re') or (self.where_re == 'hz_re'): self.Hz_re[x,y,z] = self.pulse_re

			else:
				raise ValueError("Please insert 'soft' or 'hard'")

	def init_update_equations(self, omp_on):
		"""Setter for PML, structures

			After applying structures, PML are finished, call this method.
			It will prepare DLL for update equations.
		"""

		ptr1d = np.ctypeslib.ndpointer(dtype=self.dtype, ndim=1, flags='C_CONTIGUOUS')
		ptr2d = np.ctypeslib.ndpointer(dtype=self.dtype, ndim=2, flags='C_CONTIGUOUS')
		ptr3d = np.ctypeslib.ndpointer(dtype=self.dtype, ndim=3, flags='C_CONTIGUOUS')

		# Turn on/off OpenMP parallelization.
		self.omp_on = omp_on

		# Load core, PML, PBC update equations.
		if	self.omp_on == False: 
			self.clib_core = ctypes.cdll.LoadLibrary("./core.so")
			self.clib_PML  = ctypes.cdll.LoadLibrary("./pml.so")
			self.clib_PBC  = ctypes.cdll.LoadLibrary("./pbc.so")
		elif self.omp_on == True : 
			self.clib_core = ctypes.cdll.LoadLibrary("./core.omp.so")
			self.clib_PML  = ctypes.cdll.LoadLibrary("./pml.omp.so")
			self.clib_PBC  = ctypes.cdll.LoadLibrary("./pbc.omp.so")
		else: raise ValueError("Choose True or False")

		# Setting up core update equations.
		self.clib_core.get_diff_of_H_rank_F.restype = None
		self.clib_core.get_diff_of_H_rankML.restype = None
		self.clib_core.get_diff_of_E_rankFM.restype = None
		self.clib_core.get_diff_of_E_rank_L.restype = None
		self.clib_core.updateE_rank_F.restype = None
		self.clib_core.updateE_rankML.restype = None
		self.clib_core.updateH_rankFM.restype = None
		self.clib_core.updateH_rank_L.restype = None

		self.clib_core.get_diff_of_H_rank_F.argtypes =	[\
															ctypes.c_int, ctypes.c_int, ctypes.c_int,	\
															ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
															ptr3d, 
															ptr3d, 
															ptr3d, 
															ptr3d, 
															ptr3d, 
															ptr3d, 
															ptr3d, 
															ptr3d, 
															ptr3d 
														]

		self.clib_core.get_diff_of_H_rankML.argtypes =	[\
															ctypes.c_int, ctypes.c_int, ctypes.c_int,	\
															ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
															ptr2d, 
															ptr2d, 
															ptr3d, 
															ptr3d, 
															ptr3d, 
															ptr3d, 
															ptr3d, 
															ptr3d, 
															ptr3d, 
															ptr3d, 
															ptr3d 
														]

		self.clib_core.get_diff_of_E_rankFM.argtypes = [\
															ctypes.c_int, ctypes.c_int, ctypes.c_int,	\
															ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
															ptr2d, 
															ptr2d,
															ptr3d, 
															ptr3d,
															ptr3d,
															ptr3d,
															ptr3d,
															ptr3d,
															ptr3d,
															ptr3d,
															ptr3d
														]

		self.clib_core.get_diff_of_E_rank_L.argtypes  = [\
															ctypes.c_int, ctypes.c_int, ctypes.c_int,	\
															ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
															ptr3d, 
															ptr3d, 
															ptr3d,
															ptr3d,
															ptr3d,
															ptr3d,
															ptr3d,
															ptr3d,
															ptr3d
														]

		self.clib_core.updateE_rank_F.argtypes =	[\
														ctypes.c_int, ctypes.c_int, ctypes.c_int,	\
														ctypes.c_double, \
														ptr3d, ptr3d, ptr3d,\
														ptr3d, ptr3d, ptr3d,\
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d
													]

		self.clib_core.updateE_rankML.argtypes =	[\
														ctypes.c_int, ctypes.c_int, ctypes.c_int,	\
														ctypes.c_double, \
														ptr3d, ptr3d, ptr3d,\
														ptr3d, ptr3d, ptr3d,\
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d
													]

		self.clib_core.updateH_rankFM.argtypes =	[\
														ctypes.c_int, ctypes.c_int, ctypes.c_int,	\
														ctypes.c_double, \
														ptr3d, ptr3d, ptr3d,\
														ptr3d, ptr3d, ptr3d,\
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d
													]

		self.clib_core.updateH_rank_L.argtypes =	[\
														ctypes.c_int, ctypes.c_int, ctypes.c_int,	\
														ctypes.c_double, \
														ptr3d, ptr3d, ptr3d,\
														ptr3d, ptr3d, ptr3d,\
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d, 
														ptr3d
													]

	# Load update equations for PML.
		self.clib_PML.PML_updateH_px.restype = None
		self.clib_PML.PML_updateE_px.restype = None
		self.clib_PML.PML_updateH_mx.restype = None
		self.clib_PML.PML_updateE_mx.restype = None

		self.clib_PML.PML_updateH_py.restype = None
		self.clib_PML.PML_updateE_py.restype = None
		self.clib_PML.PML_updateH_my.restype = None
		self.clib_PML.PML_updateE_my.restype = None

		self.clib_PML.PML_updateH_pz.restype = None
		self.clib_PML.PML_updateE_pz.restype = None
		self.clib_PML.PML_updateH_mz.restype = None
		self.clib_PML.PML_updateE_mz.restype = None

		# PML for x axis.
		self.clib_PML.PML_updateH_px.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, \
													ctypes.c_double, \
													ptr1d, ptr1d, ptr1d, \
													ptr3d, ptr3d, \
													ptr3d, ptr3d, \
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d 
												]

		self.clib_PML.PML_updateE_px.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, \
													ctypes.c_double, \
													ptr1d, ptr1d, ptr1d, \
													ptr3d, ptr3d, \
													ptr3d, ptr3d, \
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d
												]

		self.clib_PML.PML_updateH_mx.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, \
													ctypes.c_double, \
													ptr1d, ptr1d, ptr1d, \
													ptr3d, ptr3d, \
													ptr3d, ptr3d, \
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d
												]

		self.clib_PML.PML_updateE_mx.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, \
													ctypes.c_double, \
													ptr1d, ptr1d, ptr1d, \
													ptr3d, ptr3d, \
													ptr3d, ptr3d, \
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d
												]

		# PML for y axis.
		self.clib_PML.PML_updateH_py.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, \
													ctypes.c_double, \
													ptr1d, ptr1d, ptr1d, \
													ptr3d, ptr3d, \
													ptr3d, ptr3d, \
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d
												]

		self.clib_PML.PML_updateE_py.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, \
													ctypes.c_double, \
													ptr1d, ptr1d, ptr1d, \
													ptr3d, ptr3d, \
													ptr3d, ptr3d, \
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d
												]

		self.clib_PML.PML_updateH_my.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, \
													ctypes.c_double, \
													ptr1d, ptr1d, ptr1d, \
													ptr3d, ptr3d, \
													ptr3d, ptr3d, \
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d
												]

		self.clib_PML.PML_updateE_my.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, \
													ctypes.c_double, \
													ptr1d, ptr1d, ptr1d, \
													ptr3d, ptr3d, \
													ptr3d, ptr3d, \
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d
												]

		# PML for z axis.
		self.clib_PML.PML_updateH_pz.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, \
													ctypes.c_double, \
													ptr1d, ptr1d, ptr1d, \
													ptr3d, ptr3d, \
													ptr3d, ptr3d, \
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d, 
													ptr3d
												]

		self.clib_PML.PML_updateE_pz.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, \
													ctypes.c_double, \
													ptr1d, ptr1d, ptr1d, \
													ptr3d, ptr3d, \
													ptr3d, ptr3d, \
													ptr3d, 
													ptr3d, 
													ptr3d,
													ptr3d, 
													ptr3d, 
													ptr3d
												]

		self.clib_PML.PML_updateH_mz.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, \
													ctypes.c_double, \
													ptr1d, ptr1d, ptr1d, \
													ptr3d, ptr3d, \
													ptr3d, ptr3d, \
													ptr3d, ptr3d, \
													ptr3d, ptr3d, \
													ptr3d, ptr3d
												]

		self.clib_PML.PML_updateE_mz.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, \
													ctypes.c_double, \
													ptr1d, ptr1d, ptr1d, \
													ptr3d, ptr3d, \
													ptr3d, ptr3d, \
													ptr3d, ptr3d, \
													ptr3d, ptr3d, \
													ptr3d, ptr3d
												]

		# Load update equations for PBC.
		self.clib_PBC.my_rank_F.restype = None
		self.clib_PBC.my_rankML.restype = None

		self.clib_PBC.py_rankFM.restype = None
		self.clib_PBC.py_rank_L.restype = None

		self.clib_PBC.mz_rank_F.restype = None
		self.clib_PBC.mz_rankML.restype = None

		self.clib_PBC.pz_rankFM.restype = None
		self.clib_PBC.pz_rank_L.restype = None

		self.clib_PBC.my_rank_F.argtypes =	[\
												ctypes.c_int, ctypes.c_int, ctypes.c_int, \
												ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d\
											]

		self.clib_PBC.my_rankML.argtypes =	[\
												ctypes.c_int, ctypes.c_int, ctypes.c_int, \
												ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr2d, ptr2d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d\
											]

		self.clib_PBC.py_rankFM.argtypes =	[\
												ctypes.c_int, ctypes.c_int, ctypes.c_int, \
												ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr2d, ptr2d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d\
											]

		self.clib_PBC.py_rank_L.argtypes =	[\
												ctypes.c_int, ctypes.c_int, ctypes.c_int, \
												ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d\
											]

		self.clib_PBC.mz_rank_F.argtypes =	[\
												ctypes.c_int, ctypes.c_int, ctypes.c_int, \
												ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d\
											]

		self.clib_PBC.mz_rankML.argtypes =	[\
												ctypes.c_int, ctypes.c_int, ctypes.c_int, \
												ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr2d, ptr2d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d\
											]

		self.clib_PBC.pz_rankFM.argtypes =	[\
												ctypes.c_int, ctypes.c_int, ctypes.c_int, \
												ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr2d, ptr2d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d\
											]

		self.clib_PBC.pz_rank_L.argtypes =	[\
												ctypes.c_int, ctypes.c_int, ctypes.c_int, \
												ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d,\
												ptr3d, ptr3d\
											]

		# Load update equations for PBC on PML region.
		self.clib_PBC.mxPML_myPBC.restype = None
		self.clib_PBC.mxPML_pyPBC.restype = None
		self.clib_PBC.mxPML_mzPBC.restype = None
		self.clib_PBC.mxPML_pzPBC.restype = None

		self.clib_PBC.pxPML_myPBC.restype = None
		self.clib_PBC.pxPML_pyPBC.restype = None
		self.clib_PBC.pxPML_mzPBC.restype = None
		self.clib_PBC.pxPML_pzPBC.restype = None

		self.clib_PBC.mxPML_myPBC.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,\
													ctypes.c_double,\
													ptr1d, ptr1d, ptr1d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d\
												]

		self.clib_PBC.mxPML_pyPBC.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,\
													ctypes.c_double,\
													ptr1d, ptr1d, ptr1d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d\
												]

		self.clib_PBC.mxPML_mzPBC.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,\
													ctypes.c_double,\
													ptr1d, ptr1d, ptr1d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d\
												]

		self.clib_PBC.mxPML_pzPBC.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,\
													ctypes.c_double,\
													ptr1d, ptr1d, ptr1d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d\
												]

		self.clib_PBC.pxPML_myPBC.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,\
													ctypes.c_double,\
													ptr1d, ptr1d, ptr1d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d\
												]

		self.clib_PBC.pxPML_pyPBC.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,\
													ctypes.c_double,\
													ptr1d, ptr1d, ptr1d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d\
												]

		self.clib_PBC.pxPML_mzPBC.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,\
													ctypes.c_double,\
													ptr1d, ptr1d, ptr1d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d\
												]

		self.clib_PBC.pxPML_pzPBC.argtypes =	[\
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,\
													ctypes.c_double,\
													ptr1d, ptr1d, ptr1d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d,\
													ptr3d, ptr3d\
												]

	def updateH(self,tstep) :
		
		#--------------------------------------------------------------#
		#------------ MPI send Ex and Ey to previous rank -------------#
		#--------------------------------------------------------------#

		if (self.MPIrank > 0) and (self.MPIrank < self.MPIsize):

			sendEyfirst_re = self.Ey_re[0,:,:].copy()
			sendEzfirst_re = self.Ez_re[0,:,:].copy()

			self.MPIcomm.send( sendEyfirst_re, dest=(self.MPIrank-1), tag=(tstep*100+9 ))
			self.MPIcomm.send( sendEzfirst_re, dest=(self.MPIrank-1), tag=(tstep*100+11))

		else: pass

		#-----------------------------------------------------------#
		#------------ MPI recv Ex and Ey from next rank ------------#
		#-----------------------------------------------------------#

		if (self.MPIrank > (-1)) and (self.MPIrank < (self.MPIsize-1)):

			recvEylast_re = self.MPIcomm.recv( source=(self.MPIrank+1), tag=(tstep*100+9 ))
			recvEzlast_re = self.MPIcomm.recv( source=(self.MPIrank+1), tag=(tstep*100+11))

		else: pass

		#-----------------------------------------------------------#
		#---------------------- Get derivatives --------------------#
		#-----------------------------------------------------------#

		if self.MPIrank >= 0  and self.MPIrank < (self.MPIsize-1):

			self.clib_core.get_diff_of_E_rankFM(\
												self.myNx, self.Ny, self.Nz,\
												self.dt, self.dx, self.dy, self.dz, \
												recvEylast_re, 
												recvEzlast_re, 
												self.Ex_re, 
												self.Ey_re, 
												self.Ez_re, 
												self.diffxEy_re, 
												self.diffxEz_re, 
												self.diffyEx_re, 
												self.diffyEz_re, 
												self.diffzEx_re, 
												self.diffzEy_re
												)

		elif self.MPIrank == (self.MPIsize-1):

			self.clib_core.get_diff_of_E_rank_L(\
												self.myNx, self.Ny, self.Nz,\
												self.dt, self.dx, self.dy, self.dz, \
												self.Ex_re, 
												self.Ey_re, 
												self.Ez_re, 
												self.diffxEy_re, 
												self.diffxEz_re, 
												self.diffyEx_re, 
												self.diffyEz_re, 
												self.diffzEx_re, 
												self.diffzEy_re
												)

		#-----------------------------------------------------------#
		#--------------- Cast basic update equations ---------------#
		#-----------------------------------------------------------#
		if self.MPIrank > (-1) and self.MPIrank < (self.MPIsize-1):

			self.clib_core.updateH_rankFM	(\
												self.myNx, self.Ny, self.Nz,\
												self.dt, \
												self.mu_Hx, self.mu_Hy, self.mu_Hz, \
												self.mcon_Hx, self.mcon_Hy, self.mcon_Hz, \
												self.Hx_re, 
												self.Hy_re, 
												self.Hz_re, 
												self.diffxEy_re, 
												self.diffxEz_re, 
												self.diffyEx_re, 
												self.diffyEz_re, 
												self.diffzEx_re, 
												self.diffzEy_re
											)

		elif self.MPIrank == (self.MPIsize-1):

			self.clib_core.updateH_rank_L	(\
												self.myNx, self.Ny, self.Nz,\
												self.dt, \
												self.mu_Hx, self.mu_Hy, self.mu_Hz, \
												self.mcon_Hx, self.mcon_Hy, self.mcon_Hz, \
												self.Hx_re, 
												self.Hy_re, 
												self.Hz_re, 
												self.diffxEy_re, 
												self.diffxEz_re, 
												self.diffyEx_re, 
												self.diffyEz_re, 
												self.diffzEx_re, 
												self.diffzEy_re
											)

		#-----------------------------------------------------------#
		#---------------- Apply PML when it is given ---------------#
		#-----------------------------------------------------------#

		# First rank
		if self.MPIrank == 0:
			if 'x' in self.PMLregion.keys():
				if '+' in self.PMLregion.get('x') and self.MPIsize == 1:

					self.clib_PML.PML_updateH_px( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappax, self.PMLbx, self.PMLax, \
													self.mu_Hy, self.mu_Hz, \
													self.mcon_Hy, self.mcon_Hz, \
													self.Hy_re, 
													self.Hz_re, 
													self.diffxEy_re, 
													self.diffxEz_re, 
													self.psi_hyx_p_re, 
													self.psi_hzx_p_re
												)

				if '-' in self.PMLregion.get('x'):

					self.clib_PML.PML_updateH_mx( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappax, self.PMLbx, self.PMLax, \
													self.mu_Hy, self.mu_Hz, \
													self.mcon_Hy, self.mcon_Hz, \
													self.Hy_re, 
													self.Hz_re, 
													self.diffxEy_re, 
													self.diffxEz_re, 
													self.psi_hyx_m_re, 
													self.psi_hzx_m_re
												)

			if 'y' in self.PMLregion.keys():
				if '+' in self.PMLregion.get('y'):

					self.clib_PML.PML_updateH_py( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappay, self.PMLby, self.PMLay, \
													self.mu_Hx, self.mu_Hz, \
													self.mcon_Hx, self.mcon_Hz, \
													self.Hx_re, 
													self.Hz_re, 
													self.diffyEx_re, 
													self.diffyEz_re, 
													self.psi_hxy_p_re, 
													self.psi_hzy_p_re
												)

				if '-' in self.PMLregion.get('y'):

					self.clib_PML.PML_updateH_my( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappay, self.PMLby, self.PMLay, \
													self.mu_Hx, self.mu_Hz, \
													self.mcon_Hx, self.mcon_Hz, \
													self.Hx_re, 
													self.Hz_re, 
													self.diffyEx_re, 
													self.diffyEz_re, 
													self.psi_hxy_m_re, 
													self.psi_hzy_m_re
												)

			if 'z' in self.PMLregion.keys():
				if '+' in self.PMLregion.get('z'):

					self.clib_PML.PML_updateH_pz( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappaz, self.PMLbz, self.PMLaz, \
													self.mu_Hx, self.mu_Hy, \
													self.mcon_Hx, self.mcon_Hy, \
													self.Hx_re, 
													self.Hy_re, 
													self.diffzEx_re, 
													self.diffzEy_re, 
													self.psi_hxz_p_re, 
													self.psi_hyz_p_re
												)

				if '-' in self.PMLregion.get('z'):

					self.clib_PML.PML_updateH_mz( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappaz, self.PMLbz, self.PMLaz, \
													self.mu_Hx, self.mu_Hy, \
													self.mcon_Hx, self.mcon_Hy, \
													self.Hx_re, 
													self.Hy_re, 
													self.diffzEx_re, 
													self.diffzEy_re, 
													self.psi_hxz_m_re, 
													self.psi_hyz_m_re
												)

		# Middle rank
		elif self.MPIrank > 0 and self.MPIrank < (self.MPIsize-1):
			if 'x' in self.PMLregion.keys():
				if '+' in self.PMLregion.get('x'): pass
				if '-' in self.PMLregion.get('x'): pass

			if 'y' in self.PMLregion.keys():

				if '+' in self.PMLregion.get('y'):

					self.clib_PML.PML_updateH_py( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappay, self.PMLby, self.PMLay, \
													self.mu_Hx, self.mu_Hz, \
													self.mcon_Hx, self.mcon_Hz, \
													self.Hx_re, 
													self.Hz_re, 
													self.diffyEx_re, 
													self.diffyEz_re, 
													self.psi_hxy_p_re, 
													self.psi_hzy_p_re
												)

				if '-' in self.PMLregion.get('y'):

					self.clib_PML.PML_updateH_my( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappay, self.PMLby, self.PMLay, \
													self.mu_Hx, self.mu_Hz, \
													self.mcon_Hx, self.mcon_Hz, \
													self.Hx_re, 
													self.Hz_re, 
													self.diffyEx_re, 
													self.diffyEz_re, 
													self.psi_hxy_m_re, 
													self.psi_hzy_m_re
												)

			if 'z' in self.PMLregion.keys():

				if '+' in self.PMLregion.get('z'):

					self.clib_PML.PML_updateH_pz( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappaz, self.PMLbz, self.PMLaz, \
													self.mu_Hx, self.mu_Hy, \
													self.mcon_Hx, self.mcon_Hy, \
													self.Hx_re, 
													self.Hy_re, 
													self.diffzEx_re, 
													self.diffzEy_re, 
													self.psi_hxz_p_re, 
													self.psi_hyz_p_re
												)

				if '-' in self.PMLregion.get('z'):

					self.clib_PML.PML_updateH_mz( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappaz, self.PMLbz, self.PMLaz, \
													self.mu_Hx, self.mu_Hy, \
													self.mcon_Hx, self.mcon_Hy, \
													self.Hx_re, 
													self.Hy_re, 
													self.diffzEx_re, 
													self.diffzEy_re, 
													self.psi_hxz_m_re, 
													self.psi_hyz_m_re
												)

		# Last rank
		elif self.MPIrank == (self.MPIsize-1) and self.MPIsize != 1:

			if 'x' in self.PMLregion.keys():

				if '+' in self.PMLregion.get('x'):

					self.clib_PML.PML_updateH_px( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappax, self.PMLbx, self.PMLax, \
													self.mu_Hy, self.mu_Hz, \
													self.mcon_Hy, self.mcon_Hz, \
													self.Hy_re, 
													self.Hz_re, 
													self.diffxEy_re, 
													self.diffxEz_re, 
													self.psi_hyx_p_re, 
													self.psi_hzx_p_re
												)

				if '-' in self.PMLregion.get('x'): pass

			if 'y' in self.PMLregion.keys():

				if '+' in self.PMLregion.get('y'):

					self.clib_PML.PML_updateH_py( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappay, self.PMLby, self.PMLay, \
													self.mu_Hx, self.mu_Hz, \
													self.mcon_Hx, self.mcon_Hz, \
													self.Hx_re, 
													self.Hz_re, 
													self.diffyEx_re, 
													self.diffyEz_re, 
													self.psi_hxy_p_re, 
													self.psi_hzy_p_re
												)

				if '-' in self.PMLregion.get('y'):

					self.clib_PML.PML_updateH_my( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappay, self.PMLby, self.PMLay, \
													self.mu_Hx, self.mu_Hz, \
													self.mcon_Hx, self.mcon_Hz, \
													self.Hx_re, 
													self.Hz_re, 
													self.diffyEx_re, 
													self.diffyEz_re, 
													self.psi_hxy_m_re, 
													self.psi_hzy_m_re
												)

			if 'z' in self.PMLregion.keys():

				if '+' in self.PMLregion.get('z'):

					self.clib_PML.PML_updateH_pz( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappaz, self.PMLbz, self.PMLaz, \
													self.mu_Hx, self.mu_Hy, \
													self.mcon_Hx, self.mcon_Hy, \
													self.Hx_re, 
													self.Hy_re, 
													self.diffzEx_re, 
													self.diffzEy_re, 
													self.psi_hxz_p_re, 
													self.psi_hyz_p_re
												)

				if '-' in self.PMLregion.get('z'):

					self.clib_PML.PML_updateH_mz( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappaz, self.PMLbz, self.PMLaz, \
													self.mu_Hx, self.mu_Hy, \
													self.mcon_Hx, self.mcon_Hy, \
													self.Hx_re, 
													self.Hy_re, 
													self.diffzEx_re, 
													self.diffzEy_re, 
													self.psi_hxz_m_re, 
													self.psi_hyz_m_re
												)

		#-----------------------------------------------------------#
		#------------ Apply PBC along y when it is given -----------#
		#-----------------------------------------------------------#

		if self.myPBCregion_y == True:

			# Ranks except the last rank.
			if self.MPIsize == 0 or self.MPIrank < (self.MPIsize-1):

				self.clib_PBC.py_rankFM	(\
											self.myNx, self.Ny, self.Nz, \
											self.dt, self.dx, self.dy, self.dz, \
											self.mu_Hx, self.mu_Hz, \
											self.mcon_Hx, self.mcon_Hz, \
											recvEylast_re, 
											self.Hx_re, 
											self.Hz_re, 
											self.Ex_re, 
											self.Ey_re, 
											self.Ez_re, 
											self.diffxEy_re, 
											self.diffyEx_re, 
											self.diffyEz_re, 
											self.diffzEy_re
										)

				# The first rank apply PBC on PML region.
				if self.MPIrank == 0 and '-' in self.PMLregion.get('x'):

					self.clib_PBC.mxPML_pyPBC	(\
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt,\
													self.PMLkappax, self.PMLbx, self.PMLax,\
													self.mu_Hz, self.mcon_Hz,\
													self.Hz_re, 
													self.diffxEy_re, 
													self.psi_hzx_m_re
												)

			# The last rank.
			elif self.MPIrank == (self.MPIsize-1):
				self.clib_PBC.py_rank_L	(\
											self.myNx, self.Ny, self.Nz, \
											self.dt, self.dx, self.dy, self.dz, \
											self.mu_Hx, self.mu_Hz, \
											self.mcon_Hx, self.mcon_Hz, \
											self.Hx_re, 
											self.Hz_re, 
											self.Ex_re, 
											self.Ey_re, 
											self.Ez_re, 
											self.diffxEy_re, 
											self.diffyEx_re, 
											self.diffyEz_re, 
											self.diffzEy_re
										)

				# The last rank apply PBC on PML region.
				if '-' in self.PMLregion.get('x'):

					self.clib_PBC.pxPML_pyPBC	(\
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt,\
													self.PMLkappax, self.PMLbx, self.PMLax,\
													self.mu_Hz, self.mcon_Hz,\
													self.Hz_re, 
													self.diffxEy_re, 
													self.psi_hzx_p_re
												)


		else: pass

		#-----------------------------------------------------------#
		#------------ Apply PBC along z when it is given -----------#
		#-----------------------------------------------------------#

		if self.myPBCregion_z == True:

			# Ranks except the last rank.
			if self.MPIsize == 0 or self.MPIrank < (self.MPIsize-1):
				self.clib_PBC.pz_rankFM	(\
											self.myNx, self.Ny, self.Nz, \
											self.dt, self.dx, self.dy, self.dz, \
											self.mu_Hx, self.mu_Hy, \
											self.mcon_Hx, self.mcon_Hy, \
											recvEzlast_re, 
											self.Hx_re, 
											self.Hy_re, 
											self.Ex_re, 
											self.Ey_re, 
											self.Ez_re, 
											self.diffxEz_re, 
											self.diffyEz_re, 
											self.diffzEx_re, 
											self.diffzEy_re
										)

				# The first rank apply PBC on PML region.
				if self.MPIrank == 0 and '-' in self.PMLregion.get('x'):

					self.clib_PBC.mxPML_pzPBC	(\
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt,\
													self.PMLkappax, self.PMLbx, self.PMLax,\
													self.mu_Hy, self.mcon_Hy,\
													self.Hy_re, 
													self.diffxEz_re, 
													self.psi_hyx_m_re
												)

			# The last rank.
			else:
				self.clib_PBC.pz_rank_L	(\
											self.myNx, self.Ny, self.Nz, \
											self.dt, self.dx, self.dy, self.dz, \
											self.mu_Hx, self.mu_Hy, \
											self.mcon_Hx, self.mcon_Hy, \
											self.Hx_re, 
											self.Hy_re, 
											self.Ex_re, 
											self.Ey_re, 
											self.Ez_re, 
											self.diffxEz_re, 
											self.diffyEz_re, 
											self.diffzEx_re, 
											self.diffzEy_re
										)

				# The last rank apply PBC on PML region.
				if '-' in self.PMLregion.get('x'):

					self.clib_PBC.pxPML_pzPBC	(\
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt,\
													self.PMLkappax, self.PMLbx, self.PMLax,\
													self.mu_Hy, self.mcon_Hy,\
													self.Hy_re, 
													self.diffxEz_re, 
													self.psi_hyx_p_re
												)

		else: pass


	def updateE(self, tstep):
		"""Update E field.

		Update E field for a given time step using various update equations.
		Basic update equations, PBC update equations and PML update equations are included here.

		Args:
			tstep : int
			Given time step to update E field

		Returns:
			None

		Raises:
			Error
		"""

		#---------------------------------------------------------#
		#------------ MPI send Hy and Hz to next rank ------------#
		#---------------------------------------------------------#

		if self.MPIrank > (-1) and self.MPIrank < (self.MPIsize-1):

			sendHylast_re = self.Hy_re[-1,:,:].copy()
			sendHzlast_re = self.Hz_re[-1,:,:].copy()

			self.MPIcomm.send(sendHylast_re, dest=(self.MPIrank+1), tag=(tstep*100+3))
			self.MPIcomm.send(sendHzlast_re, dest=(self.MPIrank+1), tag=(tstep*100+5))
		
		else: pass

		#---------------------------------------------------------#
		#--------- MPI recv Hy and Hz from previous rank ---------#
		#---------------------------------------------------------#

		if self.MPIrank > 0 and self.MPIrank < self.MPIsize:

			recvHyfirst_re = self.MPIcomm.recv( source=(self.MPIrank-1), tag=(tstep*100+3))
			recvHzfirst_re = self.MPIcomm.recv( source=(self.MPIrank-1), tag=(tstep*100+5))
		
		else: pass

		#-----------------------------------------------------------#
		#---------------------- Get derivatives --------------------#
		#-----------------------------------------------------------#

		if self.MPIrank == 0:

			self.clib_core.get_diff_of_H_rank_F(\
												self.myNx, self.Ny, self.Nz,\
												self.dt, self.dx, self.dy, self.dz, \
												self.Hx_re, 
												self.Hy_re, 
												self.Hz_re, 
												self.diffxHy_re, 
												self.diffxHz_re, 
												self.diffyHx_re, 
												self.diffyHz_re, 
												self.diffzHx_re, 
												self.diffzHy_re
												)
		else:

			self.clib_core.get_diff_of_H_rankML(\
												self.myNx, self.Ny, self.Nz,\
												self.dt, self.dx, self.dy, self.dz, \
												recvHyfirst_re, 
												recvHzfirst_re, 
												self.Hx_re, 
												self.Hy_re, 
												self.Hz_re, 
												self.diffxHy_re, 
												self.diffxHz_re, 
												self.diffyHx_re, 
												self.diffyHz_re, 
												self.diffzHx_re, 
												self.diffzHy_re
												)

		#-----------------------------------------------------------#
		#--------------- Cast basic update equations ---------------#
		#-----------------------------------------------------------#

		if self.MPIrank == 0:

			self.clib_core.updateE_rank_F	(\
												self.myNx, self.Ny, self.Nz,\
												self.dt, \
												self.eps_Ex, self.eps_Ey, self.eps_Ez, \
												self.econ_Ex, self.econ_Ey, self.econ_Ez, \
												self.Ex_re, 
												self.Ey_re, 
												self.Ez_re, 
												self.diffxHy_re, 
												self.diffxHz_re, 
												self.diffyHx_re, 
												self.diffyHz_re, 
												self.diffzHx_re, 
												self.diffzHy_re
											)

		else:

			self.clib_core.updateE_rankML	(\
												self.myNx, self.Ny, self.Nz,\
												self.dt, \
												self.eps_Ex, self.eps_Ey, self.eps_Ez, \
												self.econ_Ex, self.econ_Ey, self.econ_Ez, \
												self.Ex_re, 
												self.Ey_re, 
												self.Ez_re, 
												self.diffxHy_re, 
												self.diffxHz_re, 
												self.diffyHx_re, 
												self.diffyHz_re, 
												self.diffzHx_re, 
												self.diffzHy_re
											)

		#-----------------------------------------------------------#
		#---------------- Apply PML when it is given ---------------#
		#-----------------------------------------------------------#

		# First rank
		if self.MPIrank == 0:
			if 'x' in self.PMLregion.keys():
				if '+' in self.PMLregion.get('x') and self.MPIsize == 1:

					self.clib_PML.PML_updateE_px( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappax, self.PMLbx, self.PMLax, \
													self.eps_Ey, self.eps_Ez, \
													self.econ_Ey, self.econ_Ez, \
													self.Ey_re, 
													self.Ez_re, 
													self.diffxHy_re, 
													self.diffxHz_re, 
													self.psi_eyx_p_re, 
													self.psi_ezx_p_re
												)

				if '-' in self.PMLregion.get('x'):

					self.clib_PML.PML_updateE_mx( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappax, self.PMLbx, self.PMLax, \
													self.eps_Ey, self.eps_Ez, \
													self.econ_Ey, self.econ_Ez, \
													self.Ey_re, 
													self.Ez_re, 
													self.diffxHy_re, 
													self.diffxHz_re, 
													self.psi_eyx_m_re,
													self.psi_ezx_m_re
												)

			if 'y' in self.PMLregion.keys():

				if '+' in self.PMLregion.get('y'):

					self.clib_PML.PML_updateE_py( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappay, self.PMLby, self.PMLay, \
													self.eps_Ex, self.eps_Ez, \
													self.econ_Ex, self.econ_Ez, \
													self.Ex_re, 
													self.Ez_re, 
													self.diffyHx_re, 
													self.diffyHz_re, 
													self.psi_exy_p_re, 
													self.psi_ezy_p_re
												)

				if '-' in self.PMLregion.get('y'):

					self.clib_PML.PML_updateE_my( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappay, self.PMLby, self.PMLay, \
													self.eps_Ex, self.eps_Ez, \
													self.econ_Ex, self.econ_Ez, \
													self.Ex_re, 
													self.Ez_re, 
													self.diffyHx_re, 
													self.diffyHz_re, 
													self.psi_exy_m_re, 
													self.psi_ezy_m_re
												)

			if 'z' in self.PMLregion.keys():
				if '+' in self.PMLregion.get('z'):
					self.clib_PML.PML_updateE_pz( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappaz, self.PMLbz, self.PMLaz, \
													self.eps_Ex, self.eps_Ey, \
													self.econ_Ex, self.econ_Ey, \
													self.Ex_re, 
													self.Ey_re, 
													self.diffzHx_re, 
													self.diffzHy_re, 
													self.psi_exz_p_re, 
													self.psi_eyz_p_re
												)

				if '-' in self.PMLregion.get('z'):
					self.clib_PML.PML_updateE_mz( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappaz, self.PMLbz, self.PMLaz, \
													self.eps_Ex, self.eps_Ey, \
													self.econ_Ex, self.econ_Ey, \
													self.Ex_re, 
													self.Ey_re, 
													self.diffzHx_re, 
													self.diffzHy_re, 
													self.psi_exz_m_re, 
													self.psi_eyz_m_re
												)

		# Middle rank
		elif self.MPIrank > 0 and self.MPIrank < (self.MPIsize-1):

			if 'x' in self.PMLregion.keys():
				if '+' in self.PMLregion.get('x'): pass
				if '-' in self.PMLregion.get('x'): pass

			if 'y' in self.PMLregion.keys():

				if '+' in self.PMLregion.get('y'):

					self.clib_PML.PML_updateE_py( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappay, self.PMLby, self.PMLay, \
													self.eps_Ex, self.eps_Ez, \
													self.econ_Ex, self.econ_Ez, \
													self.Ex_re, 
													self.Ez_re, 
													self.diffyHx_re, 
													self.diffyHz_re, 
													self.psi_exy_p_re, 
													self.psi_ezy_p_re
												)

				if '-' in self.PMLregion.get('y'):

					self.clib_PML.PML_updateE_my( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappay, self.PMLby, self.PMLay, \
													self.eps_Ex, self.eps_Ez, \
													self.econ_Ex, self.econ_Ez, \
													self.Ex_re, 
													self.Ez_re, 
													self.diffyHx_re, 
													self.diffyHz_re, 
													self.psi_exy_m_re, 
													self.psi_ezy_m_re
												)

			if 'z' in self.PMLregion.keys():
				if '+' in self.PMLregion.get('z'):
					self.clib_PML.PML_updateE_pz( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappaz, self.PMLbz, self.PMLaz, \
													self.eps_Ex, self.eps_Ey, \
													self.econ_Ex, self.econ_Ey, \
													self.Ex_re, 
													self.Ey_re, 
													self.diffzHx_re, 
													self.diffzHy_re, 
													self.psi_exz_p_re, 
													self.psi_eyz_p_re
												)

				if '-' in self.PMLregion.get('z'):
					self.clib_PML.PML_updateE_mz( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappaz, self.PMLbz, self.PMLaz, \
													self.eps_Ex, self.eps_Ey, \
													self.econ_Ex, self.econ_Ey, \
													self.Ex_re, 
													self.Ey_re, 
													self.diffzHx_re, 
													self.diffzHy_re, 
													self.psi_exz_m_re, 
													self.psi_eyz_m_re
												)

		# Last rank
		elif self.MPIrank == (self.MPIsize-1) and self.MPIsize != 1:
			if 'x' in self.PMLregion.keys():
				if '+' in self.PMLregion.get('x'):

					self.clib_PML.PML_updateE_px( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappax, self.PMLbx, self.PMLax, \
													self.eps_Ey, self.eps_Ez, \
													self.econ_Ey, self.econ_Ez, \
													self.Ey_re, 
													self.Ez_re, 
													self.diffxHy_re, 
													self.diffxHz_re, 
													self.psi_eyx_p_re, 
													self.psi_ezx_p_re
												)

				if '-' in self.PMLregion.get('x'): pass

			if 'y' in self.PMLregion.keys():

				if '+' in self.PMLregion.get('y'):

					self.clib_PML.PML_updateE_py( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappay, self.PMLby, self.PMLay, \
													self.eps_Ex, self.eps_Ez, \
													self.econ_Ex, self.econ_Ez, \
													self.Ex_re, 
													self.Ez_re, 
													self.diffyHx_re, 
													self.diffyHz_re, 
													self.psi_exy_p_re,
													self.psi_ezy_p_re
												)

				if '-' in self.PMLregion.get('y'):

					self.clib_PML.PML_updateE_my( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappay, self.PMLby, self.PMLay, \
													self.eps_Ex, self.eps_Ez, \
													self.econ_Ex, self.econ_Ez, \
													self.Ex_re, 
													self.Ez_re, 
													self.diffyHx_re, 
													self.diffyHz_re,
													self.psi_exy_m_re,
													self.psi_ezy_m_re
												)

			if 'z' in self.PMLregion.keys():
				if '+' in self.PMLregion.get('z'): 
					self.clib_PML.PML_updateE_pz( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappaz, self.PMLbz, self.PMLaz, \
													self.eps_Ex, self.eps_Ey, \
													self.econ_Ex, self.econ_Ey, \
													self.Ex_re,
													self.Ey_re,
													self.diffzHx_re,
													self.diffzHy_re,
													self.psi_exz_p_re,
													self.psi_eyz_p_re
												)

				if '-' in self.PMLregion.get('z'):
					self.clib_PML.PML_updateE_mz( \
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt, \
													self.PMLkappaz, self.PMLbz, self.PMLaz, \
													self.eps_Ex, self.eps_Ey, \
													self.econ_Ex, self.econ_Ey, \
													self.Ex_re,
													self.Ey_re,
													self.diffzHx_re,
													self.diffzHy_re,
													self.psi_exz_m_re,
													self.psi_eyz_m_re
												)

		#-----------------------------------------------------------#
		#------------ Apply PBC along y when it is given -----------#
		#-----------------------------------------------------------#

		if self.myPBCregion_y == True:

			# The first rank.
			if self.MPIrank == 0 :

				self.clib_PBC.my_rank_F( \
											self.myNx, self.Ny, self.Nz, \
											self.dt, self.dx, self.dy, self.dz,\
											self.eps_Ex, self.eps_Ez, \
											self.econ_Ex, self.econ_Ez, \
											self.Ex_re, 
											self.Ez_re, 
											self.Hx_re, 
											self.Hy_re, 
											self.Hz_re, 
											self.diffxHy_re, 
											self.diffyHx_re, 
											self.diffyHz_re, 
											self.diffzHy_re
										)

				# The first rank apply PBC on PML region.
				if '-' in self.PMLregion.get('x'):

					self.clib_PBC.mxPML_myPBC	(\
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt,\
													self.PMLkappax, self.PMLbx, self.PMLax,\
													self.eps_Ez, self.econ_Ez,\
													self.Ez_re, 
													self.diffxHy_re, 
													self.psi_ezx_m_re
												)

			# Ranks except the first rank.
			else:	
				self.clib_PBC.my_rankML( \
											self.myNx, self.Ny, self.Nz, \
											self.dt, self.dx, self.dy, self.dz,\
											self.eps_Ex, self.eps_Ez, \
											self.econ_Ex, self.econ_Ez, \
											recvHyfirst_re,
											self.Ex_re,
											self.Ez_re, 
											self.Hx_re, 
											self.Hy_re, 
											self.Hz_re,
											self.diffxHy_re, 
											self.diffyHx_re, 
											self.diffyHz_re, 
											self.diffzHy_re
										)

				# The last rank apply PBC on PML region.
				if self.MPIrank == (self.MPIsize-1) and '-' in self.PMLregion.get('x'):

					self.clib_PBC.pxPML_myPBC	(\
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt,\
													self.PMLkappax, self.PMLbx, self.PMLax,\
													self.eps_Ez, self.econ_Ez,\
													self.Ez_re, 
													self.diffxHy_re,
													self.psi_ezx_p_re
												)

		else: pass

		#-----------------------------------------------------------#
		#------------ Apply PBC along z when it is given -----------#
		#-----------------------------------------------------------#

		if self.myPBCregion_z == True:

			# The first rank.
			if self.MPIrank == 0:

				self.clib_PBC.mz_rank_F	(\
											self.myNx, self.Ny, self.Nz, \
											self.dt, self.dx, self.dy, self.dz,\
											self.eps_Ex, self.eps_Ez, \
											self.econ_Ex, self.econ_Ez, \
											self.Ex_re, 
											self.Ey_re, 
											self.Hx_re, 
											self.Hy_re, 
											self.Hz_re, 
											self.diffxHz_re, 
											self.diffyHz_re, 
											self.diffzHx_re, 
											self.diffzHy_re
										)

				# The first rank applies PBC on PML region.
				if '-' in self.PMLregion.get('x'):

					self.clib_PBC.mxPML_mzPBC	(\
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt,\
													self.PMLkappax, self.PMLbx, self.PMLax,\
													self.eps_Ey, self.econ_Ey,\
													self.Ey_re, 
													self.diffxHz_re, 
													self.psi_eyx_m_re
												)

			# Ranks except the first rank.
			else:

				self.clib_PBC.mz_rankML	(\
											self.myNx, self.Ny, self.Nz, \
											self.dt, self.dx, self.dy, self.dz,\
											self.eps_Ex, self.eps_Ez, \
											self.econ_Ex, self.econ_Ez, \
											recvHzfirst_re, 
											self.Ex_re, 
											self.Ey_re, 
											self.Hx_re, 
											self.Hy_re, 
											self.Hz_re, 
											self.diffxHz_re, 
											self.diffyHz_re, 
											self.diffzHx_re, 
											self.diffzHy_re
										)

				# The last rank applies PBC on PML region.
				if self.MPIrank == (self.MPIsize-1) and '-' in self.PMLregion.get('x'):

					self.clib_PBC.pxPML_mzPBC	(\
													self.myNx, self.Ny, self.Nz, self.npml,\
													self.dt,\
													self.PMLkappax, self.PMLbx, self.PMLax,\
													self.eps_Ey, self.econ_Ey,\
													self.Ey_re, 
													self.diffxHz_re, 
													self.psi_eyx_p_re
												)


		else: pass

	def get_src(self, what, tstep):

		if self.MPIrank == self.who_put_src:
			
			if	 what == 'Ex': 
				from_the_re = self.Ex_re
			elif what == 'Ey':
				from_the_re = self.Ey_re
			elif what == 'Ez': 
				from_the_re = self.Ez_re
			elif what == 'Hx': 
				from_the_re = self.Hx_re
			elif what == 'Hy':
				from_the_re = self.Hy_re
			elif what == 'Hz':
				from_the_re = self.Hz_re

			#if self.pulse_re != None: self.src_re[tstep] = self.pulse_re / 2. / self.courant
			if self.pulse_re != None: self.src_re[tstep] = self.pulse_re

	def get_ref(self, what, tstep):

		######################################################################################
		########################## All rank already knows who put src ########################
		######################################################################################

		if self.MPIrank == self.who_get_ref:

			if	 what == 'Ex': 
				from_the_re = self.Ex_re
			elif what == 'Ey':
				from_the_re = self.Ey_re
			elif what == 'Ez': 
				from_the_re = self.Ez_re
			elif what == 'Hx': 
				from_the_re = self.Hx_re
			elif what == 'Hy':
				from_the_re = self.Hy_re
			elif what == 'Hz':
				from_the_re = self.Hz_re
			
			#self.ref_re[tstep] = from_the_re[self.local_ref_xpos,:,:].mean() - (self.pulse_re / 2. / self.courant)
			self.ref_re[tstep] = from_the_re[self.local_ref_xpos,:,:].mean()

			#print(from_the_re[self.local_ref_xpos,:,:].mean(), self.pulse_re)

		else : pass
		
	def get_trs(self, what, tstep) : 

		if self.MPIrank == self.who_get_trs:
			
			if	 what == 'Ex': 
				from_the_re = self.Ex_re
			elif what == 'Ey':
				from_the_re = self.Ey_re
			elif what == 'Ez': 
				from_the_re = self.Ez_re
			elif what == 'Hx': 
				from_the_re = self.Hx_re
			elif what == 'Hy':
				from_the_re = self.Hy_re
			elif what == 'Hz':
				from_the_re = self.Hz_re

			self.trs_re[tstep] = from_the_re[self.local_trs_xpos,:,:].mean()

		else : pass

	def save_RT(self):

		self.MPIcomm.Barrier()

		if self.MPIrank == self.who_get_trs:
			np.save('./graph/trs_re.npy', self.trs_re)

		if self.MPIrank == self.who_get_ref:
			np.save('./graph/ref_re.npy', self.ref_re)


class Empty3D(object):
	
	def __init__(self, grid, gridgap, courant, dt, tsteps, dtype, **kwargs):
		"""Create Simulation Space.

			ex) Space.grid((128,128,600), (50*nm,50*nm,5*nm), dtype=np.float64)

		PARAMETERS
		----------
		grid : tuple
			define the x,y,z grid.

		gridgap : tuple
			define the dx, dy, dz.

		dtype : class numpy dtype
			choose np.float32 or np.float64

		kwargs : string
			
			supported arguments
			-------------------

			courant : float
				Set the courant number. For FDTD, default is 1./2

		RETURNS
		-------
		None
		"""

		self.nm = 1e-9
		self.um = 1e-6	

		self.dtype	  = dtype
		self.MPIcomm  = MPI.COMM_WORLD
		self.MPIrank  = self.MPIcomm.Get_rank()
		self.MPIsize  = self.MPIcomm.Get_size()
		self.hostname = MPI.Get_processor_name()

		assert len(grid)	== 3, "Simulation grid should be a tuple with length 3."
		assert len(gridgap) == 3, "Argument 'gridgap' should be a tuple with length 3."

		self.tsteps = tsteps		

		self.grid = grid
		self.Nx   = self.grid[0]
		self.Ny   = self.grid[1]
		self.Nz   = self.grid[2]
		self.TOTAL_NUM_GRID	= self.Nx * self.Ny * self.Nz
		self.TOTAL_NUM_GRID_SIZE = (self.dtype(1).nbytes * self.TOTAL_NUM_GRID) / 1024 / 1024
		
		self.Nxc = int(self.Nx / 2)
		self.Nyc = int(self.Ny / 2)
		self.Nzc = int(self.Nz / 2)
		
		self.gridgap = gridgap
		self.dx = self.gridgap[0]
		self.dy = self.gridgap[1]
		self.dz = self.gridgap[2]

		self.Lx = self.Nx * self.dx
		self.Ly = self.Ny * self.dy
		self.Lz = self.Nz * self.dz

		self.VOLUME = self.Lx * self.Ly * self.Lz

		if self.MPIrank == 0:
			print("VOLUME of the space: {:.2e}" .format(self.VOLUME))
			print("Number of grid points: {:5d} x {:5d} x {:5d}" .format(self.Nx, self.Ny, self.Nz))
			print("Grid spacing: {:.3f} nm, {:.3f} nm, {:.3f} nm" .format(self.dx/self.nm, self.dy/self.nm, self.dz/self.nm))

		self.MPIcomm.Barrier()

		self.courant = courant

		for key, value in kwargs.items():
			if key == 'courant': self.courant = value

		self.dt = dt
		self.maxdt = 1. / c / np.sqrt( (1./self.dx)**2 + (1./self.dy)**2 + (1./self.dz)**2 )

		assert (c * self.dt * np.sqrt( (1./self.dx)**2 + (1./self.dy)**2 + (1./self.dz)**2 )) < 1.

		"""
		For more details about maximum dt in the Hybrid PSTD-FDTD method, see
		Combining the FDTD and PSTD methods, Y.F.Leung, C.H. Chan,
		Microwave and Optical technology letters, Vol.23, No.4, November 20 1999.
		"""

		self.myPMLregion_x = None
		self.myPMLregion_y = None
		self.myPMLregion_z = None
		self.myPBCregion_x = False
		self.myPBCregion_y = False
		self.myPBCregion_z = False

		assert self.dt < self.maxdt, "Time interval is too big so that causality is broken. Lower the courant number."
		assert float(self.Nx) % self.MPIsize == 0., "Nx must be a multiple of the number of nodes."
		
		############################################################################
		################# Set the loc_grid each node should possess ################
		############################################################################

		self.myNx	  = int(self.Nx/self.MPIsize)
		self.loc_grid = (self.myNx, self.Ny, self.Nz)

		self.Ex_re = np.zeros(self.loc_grid, dtype=self.dtype)
		self.Ey_re = np.zeros(self.loc_grid, dtype=self.dtype)
		self.Ez_re = np.zeros(self.loc_grid, dtype=self.dtype)

		self.Hx_re = np.zeros(self.loc_grid, dtype=self.dtype)
		self.Hy_re = np.zeros(self.loc_grid, dtype=self.dtype)
		self.Hz_re = np.zeros(self.loc_grid, dtype=self.dtype)

		###############################################################################

		###############################################################################
		####################### Slices of zgrid that each node got ####################
		###############################################################################
		
		self.myNx_slices = []
		self.myNx_indice = []

		for rank in range(self.MPIsize):

			xsrt = (rank  ) * self.myNx
			xend = (rank+1) * self.myNx

			self.myNx_slices.append(slice(xsrt, xend))
			self.myNx_indice.append(	 (xsrt, xend))

		self.MPIcomm.Barrier()
		#print("rank {:>2}:\tmy xindex: {},\tmy xslice: {}" \
		#		.format(self.MPIrank, self.myNx_indice[self.MPIrank], self.myNx_slices[self.MPIrank]))

	def get_SF(self, TF, IF):

		self.Ex_re = TF.Ex_re - IF.Ex_re
		self.Ey_re = TF.Ey_re - IF.Ey_re
		self.Ez_re = TF.Ez_re - IF.Ez_re

		self.Hx_re = TF.Hx_re - IF.Hx_re
		self.Hy_re = TF.Hy_re - IF.Hy_re
		self.Hz_re = TF.Hz_re - IF.Hz_re
