import numpy as np
from scipy.constants import c, mu_0, epsilon_0

class Structure:

    def __init__(self,Space):
        
        """Define structure object.

        This script is not perfect because it cannot put dispersive materials.
        Only simple isotropic dielectric materials are possible.
        """
    
        self.Space = Space

    def _get_local_x_loc(self, gxsrts, gxends):
        """Each node get the local x location of the structure.

        Parameters
        ----------
        gxsrts: float
            global x start point of the structure.

        gxends: float
            global x end point of the structure.

        Returns
        -------
        gxloc: tuple.
            global x location of the structure.
        lxloc: tuple.
            local x location of the structure in each node.
        """

        assert gxsrts >= 0
        assert gxends < self.Space.Nx

        # Global x index of the border of each node.
        bxsrt = self.Space.myNx_indice[self.Space.MPIrank][0]
        bxend = self.Space.myNx_indice[self.Space.MPIrank][1]

        # Initialize global and local x locations of the structure.
        gxloc = None # global x location.
        lxloc = None # local x location.

        # Front nodes that contains no structures.
        if gxsrts > bxend:
            gxloc = None
            lxloc = None

        # Rear nodes that contains no structures.
        if gxends <  bxsrt:
            gxloc = None
            lxloc = None

        # First part when the structure is small.
        if gxsrts >= bxsrt and gxsrts < bxend and gxends <= bxend:
            gxloc = (gxsrts      , gxends      )
            lxloc = (gxsrts-bxsrt, gxends-bxsrt)

        # First part when the structure is big.
        if gxsrts >= bxsrt and gxsrts < bxend and gxends > bxend:
            gxloc = (gxsrts      , bxend      )
            lxloc = (gxsrts-bxsrt, bxend-bxsrt)

        # Middle node but big.
        if gxsrts < bxsrt and gxends > bxend:
            gxloc = (bxsrt      , bxend      )
            lxloc = (bxsrt-bxsrt, bxend-bxsrt)

        # Last part.
        if gxsrts < bxsrt and gxends > bxsrt and gxends <= bxend:
            gxloc = (bxsrt      , gxends      )
            lxloc = (bxsrt-bxsrt, gxends-bxsrt)

        """
        # Global x index of each node.
        node_xsrt = self.Space.myNx_indice[MPIrank][0]
        node_xend = self.Space.myNx_indice[MPIrank][1]

        self.gloc = None 
        self.lloc = None 
    
        if xend <  node_xsrt:
            self.gloc = None 
            self.lloc = None 
        if xsrt <  node_xsrt and xend > node_xsrt and xend <= node_xend:
            self.gloc = ((node_xsrt          , ysrt, zsrt), (xend          , yend, zend))
            self.lloc = ((node_xsrt-node_xsrt, ysrt, zsrt), (xend-node_xsrt, yend, zend))
        if xsrt <  node_xsrt and xend > node_xend:
            self.gloc = ((node_xsrt          , ysrt, zsrt), (node_xend          , yend, zend))
            self.lloc = ((node_xsrt-node_xsrt, ysrt, zsrt), (node_xend-node_xsrt, yend, zend))
        if xsrt >= node_xsrt and xsrt < node_xend and xend <= node_xend:
            self.gloc = ((xsrt          , ysrt, zsrt), (xend          , yend, zend))
            self.lloc = ((xsrt-node_xsrt, ysrt, zsrt), (xend-node_xsrt, yend, zend))
        if xsrt >= node_xsrt and xsrt < node_xend and xend >  node_xend:
            self.gloc = ((xsrt          , ysrt, zsrt), (node_xend          , yend, zend))
            self.lloc = ((xsrt-node_xsrt, ysrt, zsrt), (node_xend-node_xsrt, yend, zend))
        if xsrt >  node_xend:
            self.gloc = None 
            self.lloc = None 
        """

        return gxloc, lxloc


class Box(Structure):

    def __init__(self, Space, srt, end, eps_r, mu_r):
        """Set a rectangular box on a simulation space.
        
        PARAMETERS
        ----------
        Space: space object.

        srt: tuple.

        end: tuple.

        eps_r : float
            Relative electric constant or permitivity.

        mu_ r : float
            Relative magnetic constant or permeability.
            
        Returns
        -------
        None
        """

        self.eps_r = eps_r
        self.mu_r = mu_r    

        Structure.__init__(self, Space)

        assert len(srt) == 3, "Only 3D material is possible."
        assert len(end) == 3, "Only 3D material is possible."

        assert type(eps_r) == float, "Only isotropic media is possible. eps_r must be a single float."  
        assert type( mu_r) == float, "Only isotropic media is possible.  mu_r must be a single float."  

        # Start index of the structure.
        xsrt = round(srt[0]/self.Space.dx)
        ysrt = round(srt[1]/self.Space.dy)
        zsrt = round(srt[2]/self.Space.dz)

        # End index of the structure.
        xend = round(end[0]/self.Space.dx)
        yend = round(end[1]/self.Space.dy)
        zend = round(end[2]/self.Space.dz)

        assert xsrt < xend
        assert ysrt < yend
        assert zsrt < zend

        um = 1e-6
        nm = 1e-9

        if Space.MPIrank == 0:
            print("Box size: x={:5.1f} um, y={:5.1f} um, z={:5.1f} um" \
                .format((end[0]-srt[0])/um, (end[1]-srt[1])/um, (end[2]-srt[2])/um))

        self.gxloc, self.lxloc = Structure._get_local_x_loc(self, xsrt, xend)

        if self.gxloc != None:
            #self.local_size = (self.lxloc[1] - self.lxloc[0], yend-ysrt, zend-zsrt)
            print("rank {:>2}: x idx of a Box >>> global \"{:4d},{:4d}\" and local \"{:4d},{:4d}\"" \
                    .format(self.Space.MPIrank, self.gxloc[0], self.gxloc[1], self.lxloc[0], self.lxloc[1]))

            loc_xsrt = self.lxloc[0]
            loc_xend = self.lxloc[1]

            self.Space.eps_Ex[loc_xsrt:loc_xend, ysrt:yend, zsrt:zend] = self.eps_r * epsilon_0
            self.Space.eps_Ey[loc_xsrt:loc_xend, ysrt:yend, zsrt:zend] = self.eps_r * epsilon_0
            self.Space.eps_Ez[loc_xsrt:loc_xend, ysrt:yend, zsrt:zend] = self.eps_r * epsilon_0

            self.Space. mu_Hx[loc_xsrt:loc_xend, ysrt:yend, zsrt:zend] = self. mu_r * mu_0
            self.Space. mu_Hy[loc_xsrt:loc_xend, ysrt:yend, zsrt:zend] = self. mu_r * mu_0
            self.Space. mu_Hz[loc_xsrt:loc_xend, ysrt:yend, zsrt:zend] = self. mu_r * mu_0

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

        # Global srt index of the structure.
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

                            self.Space.eps_Ex[my_lxloc[i], j, k] = self.eps_r * epsilon_0
                            self.Space.eps_Ey[my_lxloc[i], j, k] = self.eps_r * epsilon_0
                            self.Space.eps_Ez[my_lxloc[i], j, k] = self.eps_r * epsilon_0

                            self.Space. mu_Hx[my_lxloc[i], j, k] = self. mu_r * mu_0
                            self.Space. mu_Hy[my_lxloc[i], j, k] = self. mu_r * mu_0
                            self.Space. mu_Hz[my_lxloc[i], j, k] = self. mu_r * mu_0

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

                            self.Space.eps_Ex[my_lxloc[i], j, k] = self.eps_r * epsilon_0
                            self.Space.eps_Ey[my_lxloc[i], j, k] = self.eps_r * epsilon_0
                            self.Space.eps_Ez[my_lxloc[i], j, k] = self.eps_r * epsilon_0

                            self.Space. mu_Hx[my_lxloc[i], j, k] = self. mu_r * mu_0
                            self.Space. mu_Hy[my_lxloc[i], j, k] = self. mu_r * mu_0
                            self.Space. mu_Hz[my_lxloc[i], j, k] = self. mu_r * mu_0


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

                            self.Space.eps_Ex[my_lxloc[i], j, k] = self.eps_r * epsilon_0
                            self.Space.eps_Ey[my_lxloc[i], j, k] = self.eps_r * epsilon_0
                            self.Space.eps_Ez[my_lxloc[i], j, k] = self.eps_r * epsilon_0

                            self.Space. mu_Hx[my_lxloc[i], j, k] = self. mu_r * mu_0
                            self.Space. mu_Hy[my_lxloc[i], j, k] = self. mu_r * mu_0
                            self.Space. mu_Hz[my_lxloc[i], j, k] = self. mu_r * mu_0

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

                            self.Space.eps_Ex[my_lxloc[i], j, k] = self.eps_r * epsilon_0
                            self.Space.eps_Ey[my_lxloc[i], j, k] = self.eps_r * epsilon_0
                            self.Space.eps_Ez[my_lxloc[i], j, k] = self.eps_r * epsilon_0

                            self.Space. mu_Hx[my_lxloc[i], j, k] = self. mu_r * mu_0
                            self.Space. mu_Hy[my_lxloc[i], j, k] = self. mu_r * mu_0
                            self.Space. mu_Hz[my_lxloc[i], j, k] = self. mu_r * mu_0

        if gxsrt >= node_xend:
                self.gxloc = None
                self.lxloc = None
            
                portion_srt  = None
                portion_end  = None
                self.portion = None

        #if self.gxloc != None:
        #   print('rank: ', MPIrank)
        #   print('Global loc: ', self.gxloc)
        #   print('Local loc: ', self.lxloc)
        #   print('height portion: ', self.portion)
        #   print('Local loc array: ', my_lxloc, len(my_lxloc))
        #   print('my height array: ', my_height, len(my_height))
        #   print('my radius array: ', my_radius, len(my_radius))

        #print(MPIrank, self.portion, self.gxloc, self.lxloc)
        self.Space.MPIcomm.Barrier()

        return


class Sphere(Structure):

    def __init__(self, Space, center, radius, eps_r, mu_r):

        Structure.__init__(self, Space)

        assert len(center)  == 3, "Please insert x,y,z coordinate of the center."

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
            self.gxloc = (xsrt          , xend          )
            self.lxloc = (xsrt-node_xsrt, xend-node_xsrt)

        # First part when the structure is big.
        if xsrt >= node_xsrt and xsrt < node_xend and xend >  node_xend:
            self.gxloc = (xsrt          , node_xend          )
            self.lxloc = (xsrt-node_xsrt, node_xend-node_xsrt)

        # Middle node but big.
        if xsrt <  node_xsrt and xend > node_xend:
            self.gxloc = (node_xsrt          , node_xend          )
            self.lxloc = (node_xsrt-node_xsrt, node_xend-node_xsrt)

        # Last part.
        if xsrt <  node_xsrt and xend > node_xsrt and xend <= node_xend:
            self.gxloc = (node_xsrt          , xend          )
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

                            self.Space.eps_Ex[lxloc[i], j, k] = self.eps_r * epsilon_0
                            self.Space.eps_Ey[lxloc[i], j, k] = self.eps_r * epsilon_0
                            self.Space.eps_Ez[lxloc[i], j, k] = self.eps_r * epsilon_0

                            self.Space. mu_Hx[lxloc[i], j, k] = self. mu_r * mu_0
                            self.Space. mu_Hy[lxloc[i], j, k] = self. mu_r * mu_0
                            self.Space. mu_Hz[lxloc[i], j, k] = self. mu_r * mu_0

            #print(MPIrank, self.gxloc, self.lxloc, rx, rr)

        return

class Cylinder(Structure):

    def __init__(self, space, radius, height, center, eps_r, mu_r):
        """Cylinder object in Basic3D structure.

        Parameters
        ----------
        radius: float.

        height: float.
            a tuple with shape (xsrt,xend) showing the height of the cylinder such that (xsrt, xend).

        center: tuple.
            a (y,z) coordinate of the center of the cylinder.

        eps_r: float.

        mu_r: float.

        Returns
        -------
        None.
   
        """

        Structure.__init__(self, space)

        self.eps_r = eps_r
        self. mu_r =  mu_r

        dx = self.Space.dx
        dy = self.Space.dy
        dz = self.Space.dz

        MPIrank = self.Space.MPIrank
        MPIsize = self.Space.MPIsize

        gxsrts = round(height[0]/dx) # Global srt index of the structure.
        gxends = round(height[1]/dx) # Global end index of the structure.

        self.gxloc, self.lxloc = Structure._get_local_x_loc(self, gxsrts, gxends)

        if self.gxloc != None:      

            for j in range(self.Space.Ny):
                for k in range(self.Space.Nz):

                    if (((j-center[0])*dy)**2 + ((k-center[1])*dz)**2) <= (radius**2):

                        self.Space.eps_Ex[self.lxloc[0]:self.lxloc[1], j, k] = self.eps_r * epsilon_0
                        self.Space.eps_Ey[self.lxloc[0]:self.lxloc[1], j, k] = self.eps_r * epsilon_0
                        self.Space.eps_Ez[self.lxloc[0]:self.lxloc[1], j, k] = self.eps_r * epsilon_0

                        self.Space. mu_Hx[self.lxloc[0]:self.lxloc[1], j, k] = self. mu_r * mu_0
                        self.Space. mu_Hy[self.lxloc[0]:self.lxloc[1], j, k] = self. mu_r * mu_0
                        self.Space. mu_Hz[self.lxloc[0]:self.lxloc[1], j, k] = self. mu_r * mu_0

        return


class Cylinder_slab(Structure):

    def __init__(self, space, srt, end, radius, llm, dc, eps_r, mu_r):
        """Cylinder object in Basic3D structure.

        Parameters
        ----------
        radius: float.

        srt: tuple.
            (x,y,z) coordinate of the left, lower most point of the slab.

        end: tuple.
            (x,y,z) coorfinate of the right, upper most point of the slab.

        llm: tuple.
            (y,z) coordinate of the center of the left, lower most cylinder.

        dc: tuple.
            y,z distance between the center of each hole.

        eps_r: float.

        mu_r: float.

        Returns
        -------
        None.
   
        """

        Structure.__init__(self, space)

        self.eps_r = eps_r
        self. mu_r =  mu_r

        dx = self.Space.dx
        dy = self.Space.dy
        dz = self.Space.dz

        MPIrank = self.Space.MPIrank
        MPIsize = self.Space.MPIsize

        self.gxloc, self.lxloc = Structure._get_local_x_loc(self, srt[0], end[0])

        if self.gxloc != None:      

            Box(self.Space, srt, end, eps_r, mu_r)
            hollows = []

            Lx = self.Space.Lx
            Ly = self.Space.Ly
            Lz = self.Space.Lz
            dy = self.Space.dy
            dz = self.Space.dz

            j = 0
            k = 0
            center = (0,0)

            while (llm[0]*dy + dc[0]*j*dy + radius) < Ly:
                while (llm[1]*dz + dc[1]*k*dz + radius) < Lz:

                    center[0] = llm[0]*dy + dc[0]*j*dy
                    center[1] = llm[1]*dz + dc[1]*k*dz

                    Cylinder(self.Space, radius, height, center, eps_r, mu_r)

                    k += 1

                j += 1

        return
