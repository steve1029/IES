import numpy as np
from scipy.constants import c, mu_0, epsilon_0

class Structure(object):

	def __init__(self,Space):
		"""Define structure object.

		This script is not perfect because it cannot put dispersive materials.
		Only simple isotropic dielectric materials are possible.
		"""
	
		self.Space = Space

class Box(Structure):
	def __init__(self, Space, srt, end, eps_r, mu_r):
		"""Place a rectangle inside of a simulation space.
		
		Args:

			eps_r : float
					Relative electric constant or permitivity.

			mu_ r : float
					Relative magnetic constant or permeability.
				
			size  : a list or tuple (iterable object) of ints
					x: height, y: width, z: thickness of a box.

			loc   : a list or typle (iterable objext) of ints
					x : x coordinate of bottom left upper coner
					y : y coordinate of bottom left upper coner
					z : z coordinate of bottom left upper coner

		Returns:
			None

		"""

		self.eps_r = eps_r
		self.mu_r = mu_r

		Structure.__init__(self, Space)

		assert len(srt)  == 3, "Only 3D material is possible."
		assert len(end)  == 3, "Only 3D material is possible."

		assert type(eps_r) == float, "Only isotropic media is possible. eps_r must be a single float."	
		assert type( mu_r) == float, "Only isotropic media is possible.  mu_r must be a single float."	

		# Start index of the structure.
		xsrt = srt[0]
		ysrt = srt[1]
		zsrt = srt[2]

		# End index of the structure.
		xend = end[0]
		yend = end[1]
		zend = end[2]

		assert xsrt < xend
		assert ysrt < yend
		assert zsrt < zend

		um = 1e-6
		nm = 1e-9

		Space.MPIcomm.barrier()

		if Space.MPIrank == 0:
			print("Box size: x={} um, y={} um, z={:.3f} um" .format((xend-xsrt)*Space.dx/um, (yend-ysrt)*Space.dy/um, (zend-zsrt)*Space.dz/um))

		MPIrank = self.Space.MPIrank
		MPIsize = self.Space.MPIsize

		# Global x index of each node.
		node_xsrt = self.Space.myNx_indice[MPIrank][0]
		node_xend = self.Space.myNx_indice[MPIrank][1]

		if xend <  node_xsrt:
			self.global_loc = None
			self. local_loc = None
		if xsrt <  node_xsrt and xend >= node_xsrt and xend <= node_xend:
			self.global_loc = ((node_xsrt          , ysrt, zsrt), (     xend          , yend, zend))
			self. local_loc = ((node_xsrt-node_xsrt, ysrt, zsrt), (     xend-node_xsrt, yend, zend))
		if xsrt <  node_xsrt and xend > node_xend:
			self.global_loc = ((node_xsrt          , ysrt, zsrt), (node_xend          , yend, zend))
			self. local_loc = ((node_xsrt-node_xsrt, ysrt, zsrt), (node_xend-node_xsrt, yend, zend))
		if xsrt >= node_xsrt and xsrt < node_xend and xend <  node_xend:
			self.global_loc = ((     xsrt          , ysrt, zsrt), (     xend          , yend, zend))
			self. local_loc = ((     xsrt-node_xsrt, ysrt, zsrt), (     xend-node_xsrt, yend, zend))
		if xsrt >= node_xsrt and xsrt < node_xend and xend >= node_xend:
			self.global_loc = ((     xsrt          , ysrt, zsrt), (node_xend          , yend, zend))
			self. local_loc = ((     xsrt-node_xsrt, ysrt, zsrt), (node_xend-node_xsrt, yend, zend))
		if xsrt >= node_xend:
			self.global_loc = None
			self. local_loc = None
		
		self.Space.MPIcomm.Barrier()

		if self.global_loc != None:
			self.local_size = (self.local_loc[1][0] - self.local_loc[0][0], yend-ysrt, zend-zsrt)
			#print("rank {:>2}: x idx of a Box >>> global \"{:4d},{:4d}\" and local \"{:4d},{:4d}\"" \
			#	.format(MPIrank, self.global_loc[0][0], self.global_loc[1][0], self.local_loc[0][0], self.local_loc[1][0]))

			loc_xsrt = self.local_loc[0][0]
			loc_ysrt = self.local_loc[0][1]
			loc_zsrt = self.local_loc[0][2]

			loc_xend = self.local_loc[1][0]
			loc_yend = self.local_loc[1][1]
			loc_zend = self.local_loc[1][2]

			self.Space.eps_HEE[loc_xsrt:loc_xend, loc_ysrt:loc_yend, loc_zsrt:loc_zend] = self.eps_r * epsilon_0
			self.Space.eps_EHH[loc_xsrt:loc_xend, loc_ysrt:loc_yend, loc_zsrt:loc_zend] = self.eps_r * epsilon_0

			self.Space. mu_HEE[loc_xsrt:loc_xend, loc_ysrt:loc_yend, loc_zsrt:loc_zend] = self. mu_r * mu_0
			self.Space. mu_EHH[loc_xsrt:loc_xend, loc_ysrt:loc_yend, loc_zsrt:loc_zend] = self. mu_r * mu_0

		return

class Cone(Structure):
	def __init__(self, Space, axis, height, radius, center, eps_r, mu_r):
		"""Place a rectangle inside of a simulation space.
		
		Args:
			Space : Space object

			axis : string
				A coordinate axis parallel to the center axis of the cone. Choose 'x','y' or 'z'.

			height : int
				A height of the cone in terms of index.

			radius : int
				A radius of the bottom of a cone.

			center : tuple
				A coordinate of the center of the bottom.

			eps_r : float
					Relative electric constant or permitivity.

			mu_ r : float
					Relative magnetic constant or permeability.

		Returns:
			None

		"""

		self.eps_r = eps_r
		self. mu_r =  mu_r

		Structure.__init__(self, Space)

		assert self.Space.dy == self.Space.dz, "dy and dz must be the same. For the other case, it is not developed yet."
		assert axis == 'x', "Sorry, a cone parallel to the y and z axis are not developed yet."

		assert len(center)  == 3, "Please insert x,y,z coordinate of the center."

		assert type(eps_r) == float, "Only isotropic media is possible. eps_r must be a single float."	
		assert type( mu_r) == float, "Only isotropic media is possible.  mu_r must be a single float."	

		# Global start index of the structure.
		gxsrt = center[0] - height

		# Global end index of the structure.
		gxend = center[0]

		assert gxsrt >= 0

		MPIrank = self.Space.MPIrank
		MPIsize = self.Space.MPIsize

		# Global x index of each node.
		node_xsrt = self.Space.myNx_indice[MPIrank][0]
		node_xend = self.Space.myNx_indice[MPIrank][1]

		if gxend <  node_xsrt:
			self.gxloc = None
			self.lxloc = None

			portion_srt  = None
			portion_end  = None
			self.portion = None

		# Last part
		if gxsrt <  node_xsrt and gxend >= node_xsrt and gxend <= node_xend:
			self.gxloc = (node_xsrt          , gxend          )
			self.lxloc = (node_xsrt-node_xsrt, gxend-node_xsrt)

			portion_srt  = height - (self.gxloc[1] - self.gxloc[0])
			portion_end  = height
			self.portion = (portion_srt, portion_end) 

			my_lxloc  = np.arange  (self.lxloc[0], self.lxloc[1] )
			my_height = np.linspace(portion_srt  , portion_end, len(my_lxloc))
			my_radius = (radius * my_height ) / height

			for i in range(len(my_radius)):
				for j in range(self.Space.Ny):
					for k in range(self.Space.Nz):

						if ((j-center[1])**2 + (k-center[2])**2) <= (my_radius[i]**2):

							self.Space.eps_HEE[my_lxloc[i], j, k] = self.eps_r * epsilon_0
							self.Space.eps_EHH[my_lxloc[i], j, k] = self.eps_r * epsilon_0

							self.Space. mu_HEE[my_lxloc[i], j, k] = self. mu_r * mu_0
							self.Space. mu_EHH[my_lxloc[i], j, k] = self. mu_r * mu_0

		# Middle part
		if gxsrt <= node_xsrt and gxend >= node_xend:
			self.gxloc = (node_xsrt          , node_xend          )
			self.lxloc = (node_xsrt-node_xsrt, node_xend-node_xsrt)

			portion_srt  = self.gxloc[0] - gxsrt
			portion_end  = portion_srt + (node_xend - node_xsrt)
			self.portion = (portion_srt, portion_end) 

			my_lxloc  = np.arange  (self.lxloc[0], self.lxloc[1] )
			my_height = np.linspace(portion_srt  , portion_end, len(my_lxloc))
			my_radius = (radius * my_height ) / height

			for i in range(len(my_radius)):
				for j in range(self.Space.Ny):
					for k in range(self.Space.Nz):

						if ((j-center[1])**2 + (k-center[2])**2) <= (my_radius[i]**2):

							self.Space.eps_HEE[my_lxloc[i], j, k] = self.eps_r * epsilon_0
							self.Space.eps_EHH[my_lxloc[i], j, k] = self.eps_r * epsilon_0

							self.Space. mu_HEE[my_lxloc[i], j, k] = self. mu_r * mu_0
							self.Space. mu_EHH[my_lxloc[i], j, k] = self. mu_r * mu_0

		# First part but small
		if gxsrt >= node_xsrt and gxsrt <= node_xend and gxend <=  node_xend:
			self.gxloc = (gxsrt          , gxend          )
			self.lxloc = (gxsrt-node_xsrt, gxend-node_xsrt)

			portion_srt  = self.lxloc[0]
			portion_end  = self.lxloc[1]
			self.portion = (portion_srt, portion_end) 

			my_lxloc  = np.arange  (self.lxloc[0], self.lxloc[1] )
			my_height = np.linspace(portion_srt  , portion_end, len(my_lxloc))
			my_radius = (radius * my_height ) / height

			for i in range(len(my_radius)):
				for j in range(self.Space.Ny):
					for k in range(self.Space.Nz):

						if ((j-center[1])**2 + (k-center[2])**2) <= (my_radius[i]**2):

							self.Space.eps_HEE[my_lxloc[i], j, k] = self.eps_r * epsilon_0
							self.Space.eps_EHH[my_lxloc[i], j, k] = self.eps_r * epsilon_0

							self.Space. mu_HEE[my_lxloc[i], j, k] = self. mu_r * mu_0
							self.Space. mu_EHH[my_lxloc[i], j, k] = self. mu_r * mu_0

		# First part but big
		if gxsrt >= node_xsrt and gxsrt <= node_xend and gxend >= node_xend:
			self.gxloc = (gxsrt          , node_xend          )
			self.lxloc = (gxsrt-node_xsrt, node_xend-node_xsrt)

			portion_srt  = 0
			portion_end  = self.gxloc[1] - self.gxloc[0]
			self.portion = (portion_srt, portion_end) 

			my_lxloc  = np.arange  (self.lxloc[0], self.lxloc[1] )
			my_height = np.linspace(portion_srt  , portion_end, len(my_lxloc))
			my_radius = (radius * my_height ) / height

			for i in range(len(my_radius)):
				for j in range(self.Space.Ny):
					for k in range(self.Space.Nz):

						if ((j-center[1])**2 + (k-center[2])**2) <= (my_radius[i]**2):

							self.Space.eps_HEE[my_lxloc[i], j, k] = self.eps_r * epsilon_0
							self.Space.eps_EHH[my_lxloc[i], j, k] = self.eps_r * epsilon_0

							self.Space. mu_HEE[my_lxloc[i], j, k] = self. mu_r * mu_0
							self.Space. mu_EHH[my_lxloc[i], j, k] = self. mu_r * mu_0

		if gxsrt >= node_xend:
				self.gxloc = None
				self.lxloc = None
			
				portion_srt  = None
				portion_end  = None
				self.portion = None
		"""
		if self.gxloc != None:
			print('rank: ', MPIrank)
			print('Global loc: ', self.gxloc)
			print('Local loc: ', self.lxloc)
			print('height portion: ', self.portion)
			print('Local loc array: ', my_lxloc, len(my_lxloc))
			print('my height array: ', my_height, len(my_height))
			print('my radius array: ', my_radius, len(my_radius))

		#print(MPIrank, self.portion, self.gxloc, self.lxloc)
		"""
		self.Space.MPIcomm.Barrier()

		return


class Sphere(Structure):

	def __init__(self, Space, center, radius, eps_r, mu_r):

		Structure.__init__(self, Space)

		assert len(center)	== 3, "Please insert x,y,z coordinate of the center."

		assert type(eps_r) == float, "Only isotropic media is possible. eps_r must be a single float."	
		assert type( mu_r) == float, "Only isotropic media is possible.  mu_r must be a single float."	

		self.eps_r = eps_r
		self. mu_r =  mu_r

		dx = self.Space.dx
		dy = self.Space.dy
		dz = self.Space.dz

		xsrt = center[0] - int(radius/dx) # Global srt index of the structure.
		xend = center[0] + int(radius/dx) # Global end index of the structure.

		assert xsrt >= 0
		assert xend < self.Space.Nx

		MPIrank = self.Space.MPIrank
		MPIsize = self.Space.MPIsize

		# Global x index of each node.
		node_xsrt = self.Space.myNx_indice[MPIrank][0]
		node_xend = self.Space.myNx_indice[MPIrank][1]

		self.gxloc = None
		self.lxloc = None

		# Front nodes that contains no structures.
		if xsrt >  node_xend:
			self.gxloc = None
			self.lxloc = None

		# Rear nodes that contains no structures.
		if xend <  node_xsrt:
			self.gxloc = None
			self.lxloc = None

		# First part when the structure is  small.
		if xsrt >= node_xsrt and xsrt < node_xend and xend <= node_xend:
			self.gxloc = (xsrt			, xend			)
			self.lxloc = (xsrt-node_xsrt, xend-node_xsrt)

		# First part when the structure is big.
		if xsrt >= node_xsrt and xsrt < node_xend and xend >  node_xend:
			self.gxloc = (xsrt			, node_xend			 )
			self.lxloc = (xsrt-node_xsrt, node_xend-node_xsrt)

		# Middle node but big.
		if xsrt <  node_xsrt and xend > node_xend:
			self.gxloc = (node_xsrt			 , node_xend		  )
			self.lxloc = (node_xsrt-node_xsrt, node_xend-node_xsrt)

		# Last part.
		if xsrt <  node_xsrt and xend > node_xsrt and xend <= node_xend:
			self.gxloc = (node_xsrt			 , xend			 )
			self.lxloc = (node_xsrt-node_xsrt, xend-node_xsrt)

		if self.gxloc != None:		

			lxloc = np.arange(self.lxloc[0], self.lxloc[1])

			portion_srt  = self.gxloc[0] - center[0] + int(radius/dx)
			portion_end  = self.gxloc[1] - center[0] + int(radius/dx)
			self.portion = np.arange(portion_srt, portion_end)

			rx = abs(self.portion - int(radius/dx))
			rr = np.zeros_like(rx, dtype=np.float64)
			theta = np.zeros_like(rx, dtype=np.float64)

			for i in range(len(rx)):
				theta[i] = np.arccos(rx[i]*dx/radius)
				rr[i] = radius * np.sin(theta[i])

				for j in range(self.Space.Ny):
					for k in range(self.Space.Nz):

						if (((j-center[1])*dy)**2 + ((k-center[2])*dz)**2) <= (rr[i]**2):

							self.Space.eps_HEE[lxloc[i], j, k] = self.eps_r * epsilon_0
							self.Space.eps_EHH[lxloc[i], j, k] = self.eps_r * epsilon_0

							self.Space. mu_HEE[lxloc[i], j, k] = self. mu_r * mu_0
							self.Space. mu_EHH[lxloc[i], j, k] = self. mu_r * mu_0

			#print(MPIrank, self.gxloc, self.lxloc, rx, rr)

		return
