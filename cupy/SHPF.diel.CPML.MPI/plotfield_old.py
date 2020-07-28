import numpy as np
import os, datetime, sys
from scipy.constants import c

class Graphtool(object):

	def __init__(self, Space, name, path):

		self.Space = Space
		self.name = name
		savedir = path + 'graph/'
		self.savedir = savedir 

		if self.Space.MPIrank == 0 : 

			while (os.path.exists(path) == False):

				print("Directory you put does not exists")
				path = input()
				
				if os.path.exists(path) == True: break
				else: continue

			if os.path.exists(savedir) == False: os.mkdir(savedir)
			else: pass

	def plot2D3D(self, what, tstep, xidx=None, yidx=None, zidx=None, **kwargs):
		"""Plot 2D and 3D intensity graph for a given field and position.

		Parameters
		------------
		what : string
			field to plot.
		figsize : tuple
			size of the figure.

		Return
		------
		None
		"""

		###################################################################################
		###################### Gather field data from all slave nodes #####################
		###################################################################################
		
		if	 what == 'Ex':
			self.gathered_fields_re = self.Space.MPIcomm.gather(self.Space.Ex_re, root=0)
		elif what == 'Ey':
			self.gathered_fields_re = self.Space.MPIcomm.gather(self.Space.Ey_re, root=0)
		elif what == 'Ez':
			self.gathered_fields_re = self.Space.MPIcomm.gather(self.Space.Ez_re, root=0)
		elif what == 'Hx':
			self.gathered_fields_re = self.Space.MPIcomm.gather(self.Space.Hx_re, root=0)
		elif what == 'Hy':
			self.gathered_fields_re = self.Space.MPIcomm.gather(self.Space.Hy_re, root=0)
		elif what == 'Hz':
			self.gathered_fields_re = self.Space.MPIcomm.gather(self.Space.Hz_re, root=0)

		if self.Space.MPIrank == 0: 

			try:
				import matplotlib.pyplot as plt
				from mpl_toolkits.mplot3d import axes3d
				from mpl_toolkits.axes_grid1 import make_axes_locatable

			except ImportError as err:
				print("Please install matplotlib at rank 0")
				sys.exit()

			colordeep = .1
			stride	  = 1
			zlim	  = 1
			figsize   = (18, 8)
			cmap	  = plt.cm.bwr
			lc = 'b'
			aspect = 'auto'

			for key, value in kwargs.items():

				if	 key == 'colordeep': colordeep = value
				elif key == 'stride'   : stride    = value
				elif key == 'zlim'	   : zlim	   = value
				elif key == 'figsize'  : figsize   = value
				elif key == 'cmap'	   : cmap	   = value
				elif key == 'lc': lc = value
				elif key == 'aspect': aspect = value

			if xidx != None : 
				assert type(xidx) == int
				yidx  = slice(None,None) # indices from beginning to end
				zidx  = slice(None,None)
				plane = 'yz'
				col = np.arange(self.Space.Ny)
				row = np.arange(self.Space.Nz)
				plane_to_plot = np.zeros((len(col),len(row)), dtype=self.Space.dtype)

			elif yidx != None :
				assert type(yidx) == int
				xidx  = slice(None,None)
				zidx  = slice(None,None)
				plane = 'xz'
				col = np.arange(self.Space.Nx)
				row = np.arange(self.Space.Nz)
				plane_to_plot = np.zeros((len(col), len(row)), dtype=self.Space.dtype)

			elif zidx != None :
				assert type(zidx) == int
				xidx  = slice(None,None)
				yidx  = slice(None,None)
				plane = 'xy'
				col = np.arange(self.Space.Nx)
				row = np.arange(self.Space.Ny)
				plane_to_plot = np.zeros((len(col),len(row)), dtype=self.Space.dtype)
		
			elif (xidx,yidx,zidx) == (None,None,None):
				raise ValueError("Plane is not defined. Please insert one of x,y or z index of the plane.")

			#####################################################################################
			######### Build up total field with the parts of the grid from slave nodes ##########
			#####################################################################################

			integrated_field_re = np.zeros((self.Space.grid), dtype=self.Space.dtype)

			for MPIrank in range(self.Space.MPIsize):
				integrated_field_re[self.Space.myNx_slices[MPIrank],:,:] = self.gathered_fields_re[MPIrank]

				#if MPIrank == 1: print(MPIrank, self.gathered_fields_re[MPIrank][xidx,yidx,zidx])

			plane_to_plot_re = integrated_field_re[xidx, yidx, zidx].copy()

			Row, Col = np.meshgrid(row, col)
			today	 = datetime.date.today()

			fig  = plt.figure(figsize=figsize)
			ax11 = fig.add_subplot(1,2,1)
			ax12 = fig.add_subplot(1,2,2, projection='3d')

			if plane == 'yz':

				image11 = ax11.imshow(plane_to_plot_re.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)
				ax12.plot_wireframe(Col, Row, plane_to_plot_re[Col, Row], color=lc, rstride=stride, cstride=stride)

				divider11 = make_axes_locatable(ax11)

				cax11  = divider11.append_axes('right', size='5%', pad=0.1)
				cbar11 = fig.colorbar(image11, cax=cax11)

				ax11.invert_yaxis()
				#ax12.invert_yaxis()

				ax11.set_xlabel('y')
				ax11.set_ylabel('z')
				ax12.set_xlabel('y')
				ax12.set_ylabel('z')

			elif plane == 'xy':

				image11 = ax11.imshow(plane_to_plot_re.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)
				ax12.plot_wireframe(Col, Row, plane_to_plot_re[Col, Row], color=lc, rstride=stride, cstride=stride)

				divider11 = make_axes_locatable(ax11)

				cax11  = divider11.append_axes('right', size='5%', pad=0.1)
				cbar11 = fig.colorbar(image11, cax=cax11)

				ax11.invert_yaxis()
				#ax12.invert_yaxis()

				ax11.set_xlabel('x')
				ax11.set_ylabel('y')
				ax12.set_xlabel('x')
				ax12.set_ylabel('y')

			elif plane == 'xz':

				image11 = ax11.imshow(plane_to_plot_re.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)
				ax12.plot_wireframe(Col, Row, plane_to_plot_re[Col, Row], color=lc, rstride=stride, cstride=stride)

				divider11 = make_axes_locatable(ax11)

				cax11  = divider11.append_axes('right', size='5%', pad=0.1)
				cbar11 = fig.colorbar(image11, cax=cax11)

				#ax11.invert_yaxis()
				ax12.invert_yaxis()

				ax11.set_xlabel('x')
				ax11.set_ylabel('z')
				ax12.set_xlabel('x')
				ax12.set_ylabel('z')

			ax11.set_title(r'$%s.real, 2D$' %what)
			ax12.set_title(r'$%s.real, 3D$' %what)

			ax12.set_zlim(-zlim,zlim)
			ax12.set_zlabel('field')

			foldername = 'plot2D3D/'
			save_dir   = self.savedir + foldername

			if os.path.exists(save_dir) == False: os.mkdir(save_dir)
			plt.tight_layout()
			fig.savefig('%s%s_%s_%s_%s_%s.png' %(save_dir, str(today), self.name, what, plane, tstep), format='png', bbox_inches='tight')
			plt.close('all')
