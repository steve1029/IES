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

        self.dtype    = dtype
        self.MPIcomm  = MPI.COMM_WORLD
        self.MPIrank  = self.MPIcomm.Get_rank()
        self.MPIsize  = self.MPIcomm.Get_size()
        self.hostname = MPI.Get_processor_name()

        assert len(grid)    == 3, "Simulation grid should be a tuple with length 3."
        assert len(gridgap) == 3, "Argument 'gridgap' should be a tuple with length 3."

        self.tsteps = tsteps        

        self.grid = grid
        self.Nx   = self.grid[0]
        self.Ny   = self.grid[1]
        self.Nz   = self.grid[2]
        self.TOTAL_NUM_GRID = self.Nx * self.Ny * self.Nz
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
        self.myBBCregion_x = False
        self.myBBCregion_y = False
        self.myBBCregion_z = False

        assert self.dt < self.maxdt, "Time interval is too big so that causality is broken. Lower the courant number."
        assert float(self.Nx) % self.MPIsize == 0., "Nx must be a multiple of the number of nodes."
        
        ############################################################################
        ################# Set the loc_grid each node should possess ################
        ############################################################################

        self.myNx     = int(self.Nx/self.MPIsize)
        self.loc_grid = (self.myNx, self.Ny, self.Nz)

        self.Ex = np.zeros(self.loc_grid, dtype=self.dtype)
        self.Ey = np.zeros(self.loc_grid, dtype=self.dtype)
        self.Ez = np.zeros(self.loc_grid, dtype=self.dtype)

        self.Hx = np.zeros(self.loc_grid, dtype=self.dtype)
        self.Hy = np.zeros(self.loc_grid, dtype=self.dtype)
        self.Hz = np.zeros(self.loc_grid, dtype=self.dtype)
        ###############################################################################

        self.diffxEy = np.zeros(self.loc_grid, dtype=self.dtype)
        self.diffxEz = np.zeros(self.loc_grid, dtype=self.dtype)
        self.diffyEx = np.zeros(self.loc_grid, dtype=self.dtype)
        self.diffyEz = np.zeros(self.loc_grid, dtype=self.dtype)
        self.diffzEx = np.zeros(self.loc_grid, dtype=self.dtype)
        self.diffzEy = np.zeros(self.loc_grid, dtype=self.dtype)

        self.diffxHy = np.zeros(self.loc_grid, dtype=self.dtype)
        self.diffxHz = np.zeros(self.loc_grid, dtype=self.dtype)
        self.diffyHx = np.zeros(self.loc_grid, dtype=self.dtype)
        self.diffyHz = np.zeros(self.loc_grid, dtype=self.dtype)
        self.diffzHx = np.zeros(self.loc_grid, dtype=self.dtype)
        self.diffzHy = np.zeros(self.loc_grid, dtype=self.dtype)
        ############################################################################

        self.eps_Ex = np.ones(self.loc_grid, dtype=self.dtype) * epsilon_0
        self.eps_Ey = np.ones(self.loc_grid, dtype=self.dtype) * epsilon_0
        self.eps_Ez = np.ones(self.loc_grid, dtype=self.dtype) * epsilon_0

        self.mu_Hx  = np.ones(self.loc_grid, dtype=self.dtype) * mu_0
        self.mu_Hy  = np.ones(self.loc_grid, dtype=self.dtype) * mu_0
        self.mu_Hz  = np.ones(self.loc_grid, dtype=self.dtype) * mu_0

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
            self.myNx_indice.append(     (xsrt, xend))

        self.MPIcomm.Barrier()
        #print("rank {:>2}:\tmy xindex: {},\tmy xslice: {}" \
        #       .format(self.MPIrank, self.myNx_indice[self.MPIrank], self.myNx_slices[self.MPIrank]))

    def set_PML(self, region, npml):

        self.PMLregion  = region
        self.npml       = npml
        self.PMLgrading = 2 * self.npml

        self.rc0   = 1.e-16                             # reflection coefficient
        self.imp   = np.sqrt(mu_0/epsilon_0)            # impedence
        self.gO    = 3.                                 # gradingOrder
        self.sO    = 3.                                 # scalingOrder
        self.bdw_x = (self.PMLgrading-1) * self.dx      # PML thickness along x (Boundarywidth)
        self.bdw_y = (self.PMLgrading-1) * self.dy      # PML thickness along y
        self.bdw_z = (self.PMLgrading-1) * self.dz      # PML thickness along z

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

            if   key == 'x' and value != '':

                self.psi_eyx_p = np.zeros((npml, self.Ny, self.Nz), dtype=self.dtype)
                self.psi_ezx_p = np.zeros((npml, self.Ny, self.Nz), dtype=self.dtype)
                self.psi_hyx_p = np.zeros((npml, self.Ny, self.Nz), dtype=self.dtype)
                self.psi_hzx_p = np.zeros((npml, self.Ny, self.Nz), dtype=self.dtype)

                self.psi_eyx_m = np.zeros((npml, self.Ny, self.Nz), dtype=self.dtype)
                self.psi_ezx_m = np.zeros((npml, self.Ny, self.Nz), dtype=self.dtype)
                self.psi_hyx_m = np.zeros((npml, self.Ny, self.Nz), dtype=self.dtype)
                self.psi_hzx_m = np.zeros((npml, self.Ny, self.Nz), dtype=self.dtype)

                """
                for i in range(self.PMLgrading):

                    loc  = np.float64(i) * self.dx / self.bdw_x

                    self.PMLsigmax[i] = self.PMLsigmamaxx * (loc **self.gO)
                    self.PMLkappax[i] = 1 + ((self.PMLkappamaxx-1) * (loc **self.gO))
                    self.PMLalphax[i] = self.PMLalphamaxx * ((1-loc) **self.sO)
                """
                loc = np.arange(self.PMLgrading) * self.dx / self.bdw_x
                self.PMLsigmax = self.PMLsigmamaxx * (loc **self.gO)
                self.PMLkappax = 1 + ((self.PMLkappamaxx-1) * (loc **self.gO))
                self.PMLalphax = self.PMLalphamaxx * ((1-loc) **self.sO)

            elif key == 'y' and value != '':

                self.psi_exy_p = np.zeros((self.myNx, npml, self.Nz), dtype=self.dtype)
                self.psi_ezy_p = np.zeros((self.myNx, npml, self.Nz), dtype=self.dtype)
                self.psi_hxy_p = np.zeros((self.myNx, npml, self.Nz), dtype=self.dtype)
                self.psi_hzy_p = np.zeros((self.myNx, npml, self.Nz), dtype=self.dtype)

                self.psi_exy_m = np.zeros((self.myNx, npml, self.Nz), dtype=self.dtype)
                self.psi_ezy_m = np.zeros((self.myNx, npml, self.Nz), dtype=self.dtype)
                self.psi_hxy_m = np.zeros((self.myNx, npml, self.Nz), dtype=self.dtype)
                self.psi_hzy_m = np.zeros((self.myNx, npml, self.Nz), dtype=self.dtype)
                """
                for i in range(self.PMLgrading):

                    loc  = np.float64(i) * self.dy / self.bdw_y

                    self.PMLsigmay[i] = self.PMLsigmamaxy * (loc **self.gO)
                    self.PMLkappay[i] = 1 + ((self.PMLkappamaxy-1) * (loc **self.gO))
                    self.PMLalphay[i] = self.PMLalphamaxy * ((1-loc) **self.sO)
                """

                loc  = np.arange(self.PMLgrading) * self.dy / self.bdw_y
                self.PMLsigmay = self.PMLsigmamaxy * (loc **self.gO)
                self.PMLkappay = 1 + ((self.PMLkappamaxy-1) * (loc **self.gO))
                self.PMLalphay = self.PMLalphamaxy * ((1-loc) **self.sO)

            elif key == 'z' and value != '':

                self.psi_exz_p = np.zeros((self.myNx, self.Ny, npml), dtype=self.dtype)
                self.psi_eyz_p = np.zeros((self.myNx, self.Ny, npml), dtype=self.dtype)
                self.psi_hxz_p = np.zeros((self.myNx, self.Ny, npml), dtype=self.dtype)
                self.psi_hyz_p = np.zeros((self.myNx, self.Ny, npml), dtype=self.dtype)

                self.psi_exz_m = np.zeros((self.myNx, self.Ny, npml), dtype=self.dtype)
                self.psi_eyz_m = np.zeros((self.myNx, self.Ny, npml), dtype=self.dtype)
                self.psi_hxz_m = np.zeros((self.myNx, self.Ny, npml), dtype=self.dtype)
                self.psi_hyz_m = np.zeros((self.myNx, self.Ny, npml), dtype=self.dtype)

                """
                for i in range(self.PMLgrading):

                    loc  = np.float64(i) * self.dz / self.bdw_z

                    self.PMLsigmaz[i] = self.PMLsigmamaxz * (loc **self.gO)
                    self.PMLkappaz[i] = 1 + ((self.PMLkappamaxz-1) * (loc **self.gO))
                    self.PMLalphaz[i] = self.PMLalphamaxz * ((1-loc) **self.sO)
                """

                loc  = np.arange(selfe.PMLgrading) * self.dz / self.bdw_z
                self.PMLsigmaz = self.PMLsigmamaxz * (loc **self.gO)
                self.PMLkappaz = 1 + ((self.PMLkappamaxz-1) * (loc **self.gO))
                self.PMLalphaz = self.PMLalphamaxz * ((1-loc) **self.sO)

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
                    f.create_dataset('PMLsigmax' ,  data=self.PMLsigmax)
                    f.create_dataset('PMLkappax' ,  data=self.PMLkappax)
                    f.create_dataset('PMLalphax' ,  data=self.PMLalphax)
                    f.create_dataset('PMLbx',       data=self.PMLbx)
                    f.create_dataset('PMLax',       data=self.PMLax)
                elif key == 'y':
                    f.create_dataset('PMLsigmay' ,  data=self.PMLsigmay)
                    f.create_dataset('PMLkappay' ,  data=self.PMLkappay)
                    f.create_dataset('PMLalphay' ,  data=self.PMLalphay)
                    f.create_dataset('PMLby',       data=self.PMLby)
                    f.create_dataset('PMLay',       data=self.PMLay)
                elif key == 'z':
                    f.create_dataset('PMLsigmaz' ,  data=self.PMLsigmaz)
                    f.create_dataset('PMLkappaz' ,  data=self.PMLkappaz)
                    f.create_dataset('PMLalphaz' ,  data=self.PMLalphaz)
                    f.create_dataset('PMLbz',       data=self.PMLbz)
                    f.create_dataset('PMLaz',       data=self.PMLaz)

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

        f.create_dataset('eps_Ex',  data=self.eps_Ex)
        f.create_dataset('eps_Ey',  data=self.eps_Ey)
        f.create_dataset('eps_Ez',  data=self.eps_Ez)
        f.create_dataset( 'mu_Hx',  data=self. mu_Hx)
        f.create_dataset( 'mu_Hy',  data=self. mu_Hy)
        f.create_dataset( 'mu_Hz',  data=self. mu_Hz)
            
        self.MPIcomm.Barrier()

        return

    def apply_PBC(self, region):
        """Specify the boundary to apply Periodic Boundary Condition.

        PARAMETERS
        ----------
        region : dictionary
            ex) {'x':'','y':'+-','z':'+-'}

        RETURNS
        -------
        None
        """

        value = region.get('x')
        if value == '+-' or value == '-+':
            if self.MPIsize > 1:
                if   self.MPIrank == 0               : self.myPBCregion_x = '-'
                elif self.MPIrank == (self.MPIsize-1): self.myPBCregion_x = '+'
        elif value == None: pass
        else: raise ValueError("The value of key 'x' should be None or '+-' or '-+'.")

        value = region.get('y')
        if   value == True:  self.myPBCregion_y = True
        elif value == False: self.myPBCregion_y = False
        else: raise ValueError("Choose True or False")

        value = region.get('z')
        if   value == True:  self.myPBCregion_z = True
        elif value == False: self.myPBCregion_z = False
        else: raise ValueError("Choose True or False")

        """
        for key, value in region.items():

            if   key == 'x':

                if   value == '+': raise ValueError("input '+-' or '-+'.")
                elif value == '-': raise ValueError("input '+-' or '-+'.")
                elif value == '+-' or value == '-+':

                    if   self.MPIrank == 0               : self.myPBCregion_x = '-'
                    elif self.MPIrank == (self.MPIsize-1): self.myPBCregion_x = '+'

            elif key == 'y':

                if value == True: self.myPBCregion_y = True
                elif value == False: self.myPBCregion_y = False
                else: raise ValueError("Choose True or False")

            elif key == 'z':
    
                if value == True: self.myPBCregion_z = True
                elif value == False: self.myPBCregion_z = False
                else: raise ValueError("Choose True or False")
        """

        self.MPIcomm.Barrier()
        #print("PBC region of rank: {}, x: {}, y: {}, z: {}" \
        #       .format(self.MPIrank, self.myPBCregion_x, self.myPBCregion_y, self.myPBCregion_z))

    def apply_BBC(self, region):
        """Specify the boundary to apply Bloch Boundary Condition.

        PARAMETERS
        ----------
        region : dictionary
            ex) {'x':'','y':'+-','z':'+-'}

        RETURNS
        -------
        None
        """

        value = region.get('x')
        if value == '+-' or value == '-+':
            if self.MPIsize > 1:
                if   self.MPIrank == 0               : self.myBBCregion_x = '-'
                elif self.MPIrank == (self.MPIsize-1): self.myBBCregion_x = '+'
        elif value == None: pass
        else: raise ValueError("The value of key 'x' should be None or '+-' or '-+'.")

        value = region.get('y')
        if   value == True:  self.myBBCregion_y = True
        elif value == False: self.myBBCregion_y = False
        else: raise ValueError("Choose True or False")

        value = region.get('z')
        if   value == True:  self.myBBCregion_z = True
        elif value == False: self.myBBCregion_z = False
        else: raise ValueError("Choose True or False")

        """
        for key, value in region.items():

            if   key == 'x':

                if   value == '+': raise ValueError("input '+-' or '-+'.")
                elif value == '-': raise ValueError("input '+-' or '-+'.")
                elif value == '+-' or value == '-+':

                    if   self.MPIrank == 0               : self.myBBCregion_x = '-'
                    elif self.MPIrank == (self.MPIsize-1): self.myBBCregion_x = '+'

            elif key == 'y':

                if value == True: self.myBBCregion_y = True
                elif value == False: self.myBBCregion_y = False
                else: raise ValueError("Choose True or False")

            elif key == 'z':
    
                if value == True: self.myBBCregion_z = True
                elif value == False: self.myBBCregion_z = False
                else: raise ValueError("Choose True or False")
        """
        self.MPIcomm.Barrier()

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

                        self.src = np.zeros(self.tsteps, dtype=self.dtype)

                        #print("rank{:>2}: src_xsrt : {}, my_src_xsrt: {}, my_src_xend: {}"\
                        #       .format(self.MPIrank, self.src_xsrt, self.my_src_xsrt, self.my_src_xend))
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

                self.src = np.zeros(self.tsteps, dtype=self.dtype)

            # case 3. x position of source is reversed.
            elif self.src_xsrt > self.src_xend:
                raise ValueError("src_end[0] is bigger than src_srt[0]")

            else:
                raise IndexError("x position of src is not defined!")

    def put_src(self, where, pulse, put_type):
        """Put source at the designated postion set by set_src_pos method.
        
        PARAMETERS
        ----------  
        where : string
            ex)
                'Ex' or 'ex'
                'Ey' or 'ey'
                'Ez' or 'ez'

        pulse : float
            float returned by source.pulse.

        put_type : string
            'soft' or 'hard'

        """
        #------------------------------------------------------------#
        #--------- Put the source into the designated field ---------#
        #------------------------------------------------------------#

        self.put_type = put_type

        self.where = where
        
        self.pulse = self.dtype(pulse)

        if self.MPIrank == self.who_put_src:

            x = slice(self.my_src_xsrt, self.my_src_xend)
            y = slice(self.   src_ysrt, self.   src_yend)
            z = slice(self.   src_zsrt, self.   src_zend)
            
            if   self.put_type == 'soft':

                if (self.where == 'Ex') or (self.where == 'ex'): self.Ex[x,y,z] += self.pulse
                if (self.where == 'Ey') or (self.where == 'ey'): self.Ey[x,y,z] += self.pulse
                if (self.where == 'Ez') or (self.where == 'ez'): self.Ez[x,y,z] += self.pulse
                if (self.where == 'Hx') or (self.where == 'hx'): self.Hx[x,y,z] += self.pulse
                if (self.where == 'Hy') or (self.where == 'hy'): self.Hy[x,y,z] += self.pulse
                if (self.where == 'Hz') or (self.where == 'hz'): self.Hz[x,y,z] += self.pulse

            elif self.put_type == 'hard':
    
                if (self.where == 'Ex') or (self.where == 'ex'): self.Ex[x,y,z] = self.pulse
                if (self.where == 'Ey') or (self.where == 'ey'): self.Ey[x,y,z] = self.pulse
                if (self.where == 'Ez') or (self.where == 'ez'): self.Ez[x,y,z] = self.pulse
                if (self.where == 'Hx') or (self.where == 'hx'): self.Hx[x,y,z] = self.pulse
                if (self.where == 'Hy') or (self.where == 'hy'): self.Hy[x,y,z] = self.pulse
                if (self.where == 'Hz') or (self.where == 'hz'): self.Hz[x,y,z] = self.pulse

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
        if  self.omp_on == False: 
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

        self.clib_core.get_diff_of_H_rank_F.argtypes =  [\
                                                            ctypes.c_int, ctypes.c_int, ctypes.c_int,   \
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

        self.clib_core.get_diff_of_H_rankML.argtypes =  [\
                                                            ctypes.c_int, ctypes.c_int, ctypes.c_int,   \
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
                                                            ctypes.c_int, ctypes.c_int, ctypes.c_int,   \
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
                                                            ctypes.c_int, ctypes.c_int, ctypes.c_int,   \
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

        self.clib_core.updateE_rank_F.argtypes =    [\
                                                        ctypes.c_int, ctypes.c_int, ctypes.c_int,   \
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

        self.clib_core.updateE_rankML.argtypes =    [\
                                                        ctypes.c_int, ctypes.c_int, ctypes.c_int,   \
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

        self.clib_core.updateH_rankFM.argtypes =    [\
                                                        ctypes.c_int, ctypes.c_int, ctypes.c_int,   \
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

        self.clib_core.updateH_rank_L.argtypes =    [\
                                                        ctypes.c_int, ctypes.c_int, ctypes.c_int,   \
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
        self.clib_PML.PML_updateH_px.argtypes = [\
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

        self.clib_PML.PML_updateE_px.argtypes = [\
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

        self.clib_PML.PML_updateH_mx.argtypes = [\
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

        self.clib_PML.PML_updateE_mx.argtypes = [\
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
        self.clib_PML.PML_updateH_py.argtypes = [\
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

        self.clib_PML.PML_updateE_py.argtypes = [\
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

        self.clib_PML.PML_updateH_my.argtypes = [\
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

        self.clib_PML.PML_updateE_my.argtypes = [\
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
        self.clib_PML.PML_updateH_pz.argtypes = [\
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

        self.clib_PML.PML_updateE_pz.argtypes = [\
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

        self.clib_PML.PML_updateH_mz.argtypes = [\
                                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, \
                                                    ctypes.c_double, \
                                                    ptr1d, ptr1d, ptr1d, \
                                                    ptr3d, ptr3d, \
                                                    ptr3d, ptr3d, \
                                                    ptr3d, ptr3d, \
                                                    ptr3d, ptr3d, \
                                                    ptr3d, ptr3d
                                                ]

        self.clib_PML.PML_updateE_mz.argtypes = [\
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

        self.clib_PBC.my_rank_F.argtypes =  [\
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

        self.clib_PBC.my_rankML.argtypes =  [\
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

        self.clib_PBC.py_rankFM.argtypes =  [\
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

        self.clib_PBC.py_rank_L.argtypes =  [\
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

        self.clib_PBC.mz_rank_F.argtypes =  [\
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

        self.clib_PBC.mz_rankML.argtypes =  [\
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

        self.clib_PBC.pz_rankFM.argtypes =  [\
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

        self.clib_PBC.pz_rank_L.argtypes =  [\
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

        self.clib_PBC.mxPML_myPBC.argtypes =    [\
                                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,\
                                                    ctypes.c_double,\
                                                    ptr1d, ptr1d, ptr1d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d\
                                                ]

        self.clib_PBC.mxPML_pyPBC.argtypes =    [\
                                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,\
                                                    ctypes.c_double,\
                                                    ptr1d, ptr1d, ptr1d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d\
                                                ]

        self.clib_PBC.mxPML_mzPBC.argtypes =    [\
                                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,\
                                                    ctypes.c_double,\
                                                    ptr1d, ptr1d, ptr1d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d\
                                                ]

        self.clib_PBC.mxPML_pzPBC.argtypes =    [\
                                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,\
                                                    ctypes.c_double,\
                                                    ptr1d, ptr1d, ptr1d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d\
                                                ]

        self.clib_PBC.pxPML_myPBC.argtypes =    [\
                                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,\
                                                    ctypes.c_double,\
                                                    ptr1d, ptr1d, ptr1d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d\
                                                ]

        self.clib_PBC.pxPML_pyPBC.argtypes =    [\
                                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,\
                                                    ctypes.c_double,\
                                                    ptr1d, ptr1d, ptr1d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d\
                                                ]

        self.clib_PBC.pxPML_mzPBC.argtypes =    [\
                                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,\
                                                    ctypes.c_double,\
                                                    ptr1d, ptr1d, ptr1d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d,\
                                                    ptr3d, ptr3d\
                                                ]

        self.clib_PBC.pxPML_pzPBC.argtypes =    [\
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

            sendEyfirst = self.Ey[0,:,:].copy()
            sendEzfirst = self.Ez[0,:,:].copy()

            self.MPIcomm.send( sendEyfirst, dest=(self.MPIrank-1), tag=(tstep*100+9 ))
            self.MPIcomm.send( sendEzfirst, dest=(self.MPIrank-1), tag=(tstep*100+11))

        else: pass

        #-----------------------------------------------------------#
        #------------ MPI recv Ex and Ey from next rank ------------#
        #-----------------------------------------------------------#

        if (self.MPIrank > (-1)) and (self.MPIrank < (self.MPIsize-1)):

            recvEylast = self.MPIcomm.recv( source=(self.MPIrank+1), tag=(tstep*100+9 ))
            recvEzlast = self.MPIcomm.recv( source=(self.MPIrank+1), tag=(tstep*100+11))

        else: pass

        #-----------------------------------------------------------#
        #---------------------- Get derivatives --------------------#
        #-----------------------------------------------------------#

        if self.MPIrank >= 0  and self.MPIrank < (self.MPIsize-1):

            self.clib_core.get_diff_of_E_rankFM(\
                                                self.myNx, self.Ny, self.Nz,\
                                                self.dt, self.dx, self.dy, self.dz, \
                                                recvEylast, 
                                                recvEzlast, 
                                                self.Ex, 
                                                self.Ey, 
                                                self.Ez, 
                                                self.diffxEy, 
                                                self.diffxEz, 
                                                self.diffyEx, 
                                                self.diffyEz, 
                                                self.diffzEx, 
                                                self.diffzEy
                                                )

        elif self.MPIrank == (self.MPIsize-1):

            self.clib_core.get_diff_of_E_rank_L(\
                                                self.myNx, self.Ny, self.Nz,\
                                                self.dt, self.dx, self.dy, self.dz, \
                                                self.Ex, 
                                                self.Ey, 
                                                self.Ez, 
                                                self.diffxEy, 
                                                self.diffxEz, 
                                                self.diffyEx, 
                                                self.diffyEz, 
                                                self.diffzEx, 
                                                self.diffzEy
                                                )

        #-----------------------------------------------------------#
        #--------------- Cast basic update equations ---------------#
        #-----------------------------------------------------------#
        if self.MPIrank > (-1) and self.MPIrank < (self.MPIsize-1):

            self.clib_core.updateH_rankFM   (\
                                                self.myNx, self.Ny, self.Nz,\
                                                self.dt, \
                                                self.mu_Hx, self.mu_Hy, self.mu_Hz, \
                                                self.mcon_Hx, self.mcon_Hy, self.mcon_Hz, \
                                                self.Hx, 
                                                self.Hy, 
                                                self.Hz, 
                                                self.diffxEy, 
                                                self.diffxEz, 
                                                self.diffyEx, 
                                                self.diffyEz, 
                                                self.diffzEx, 
                                                self.diffzEy
                                            )

        elif self.MPIrank == (self.MPIsize-1):

            self.clib_core.updateH_rank_L   (\
                                                self.myNx, self.Ny, self.Nz,\
                                                self.dt, \
                                                self.mu_Hx, self.mu_Hy, self.mu_Hz, \
                                                self.mcon_Hx, self.mcon_Hy, self.mcon_Hz, \
                                                self.Hx, 
                                                self.Hy, 
                                                self.Hz, 
                                                self.diffxEy, 
                                                self.diffxEz, 
                                                self.diffyEx, 
                                                self.diffyEz, 
                                                self.diffzEx, 
                                                self.diffzEy
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
                                                    self.Hy, 
                                                    self.Hz, 
                                                    self.diffxEy, 
                                                    self.diffxEz, 
                                                    self.psi_hyx_p, 
                                                    self.psi_hzx_p
                                                )

                if '-' in self.PMLregion.get('x'):

                    self.clib_PML.PML_updateH_mx( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappax, self.PMLbx, self.PMLax, \
                                                    self.mu_Hy, self.mu_Hz, \
                                                    self.mcon_Hy, self.mcon_Hz, \
                                                    self.Hy, 
                                                    self.Hz, 
                                                    self.diffxEy, 
                                                    self.diffxEz, 
                                                    self.psi_hyx_m, 
                                                    self.psi_hzx_m
                                                )

            if 'y' in self.PMLregion.keys():
                if '+' in self.PMLregion.get('y'):

                    self.clib_PML.PML_updateH_py( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappay, self.PMLby, self.PMLay, \
                                                    self.mu_Hx, self.mu_Hz, \
                                                    self.mcon_Hx, self.mcon_Hz, \
                                                    self.Hx, 
                                                    self.Hz, 
                                                    self.diffyEx, 
                                                    self.diffyEz, 
                                                    self.psi_hxy_p, 
                                                    self.psi_hzy_p
                                                )

                if '-' in self.PMLregion.get('y'):

                    self.clib_PML.PML_updateH_my( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappay, self.PMLby, self.PMLay, \
                                                    self.mu_Hx, self.mu_Hz, \
                                                    self.mcon_Hx, self.mcon_Hz, \
                                                    self.Hx, 
                                                    self.Hz, 
                                                    self.diffyEx, 
                                                    self.diffyEz, 
                                                    self.psi_hxy_m, 
                                                    self.psi_hzy_m
                                                )

            if 'z' in self.PMLregion.keys():
                if '+' in self.PMLregion.get('z'):

                    self.clib_PML.PML_updateH_pz( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappaz, self.PMLbz, self.PMLaz, \
                                                    self.mu_Hx, self.mu_Hy, \
                                                    self.mcon_Hx, self.mcon_Hy, \
                                                    self.Hx, 
                                                    self.Hy, 
                                                    self.diffzEx, 
                                                    self.diffzEy, 
                                                    self.psi_hxz_p, 
                                                    self.psi_hyz_p
                                                )

                if '-' in self.PMLregion.get('z'):

                    self.clib_PML.PML_updateH_mz( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappaz, self.PMLbz, self.PMLaz, \
                                                    self.mu_Hx, self.mu_Hy, \
                                                    self.mcon_Hx, self.mcon_Hy, \
                                                    self.Hx, 
                                                    self.Hy, 
                                                    self.diffzEx, 
                                                    self.diffzEy, 
                                                    self.psi_hxz_m, 
                                                    self.psi_hyz_m
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
                                                    self.Hx, 
                                                    self.Hz, 
                                                    self.diffyEx, 
                                                    self.diffyEz, 
                                                    self.psi_hxy_p, 
                                                    self.psi_hzy_p
                                                )

                if '-' in self.PMLregion.get('y'):

                    self.clib_PML.PML_updateH_my( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappay, self.PMLby, self.PMLay, \
                                                    self.mu_Hx, self.mu_Hz, \
                                                    self.mcon_Hx, self.mcon_Hz, \
                                                    self.Hx, 
                                                    self.Hz, 
                                                    self.diffyEx, 
                                                    self.diffyEz, 
                                                    self.psi_hxy_m, 
                                                    self.psi_hzy_m
                                                )

            if 'z' in self.PMLregion.keys():

                if '+' in self.PMLregion.get('z'):

                    self.clib_PML.PML_updateH_pz( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappaz, self.PMLbz, self.PMLaz, \
                                                    self.mu_Hx, self.mu_Hy, \
                                                    self.mcon_Hx, self.mcon_Hy, \
                                                    self.Hx, 
                                                    self.Hy, 
                                                    self.diffzEx, 
                                                    self.diffzEy, 
                                                    self.psi_hxz_p, 
                                                    self.psi_hyz_p
                                                )

                if '-' in self.PMLregion.get('z'):

                    self.clib_PML.PML_updateH_mz( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappaz, self.PMLbz, self.PMLaz, \
                                                    self.mu_Hx, self.mu_Hy, \
                                                    self.mcon_Hx, self.mcon_Hy, \
                                                    self.Hx, 
                                                    self.Hy, 
                                                    self.diffzEx, 
                                                    self.diffzEy, 
                                                    self.psi_hxz_m, 
                                                    self.psi_hyz_m
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
                                                    self.Hy, 
                                                    self.Hz, 
                                                    self.diffxEy, 
                                                    self.diffxEz, 
                                                    self.psi_hyx_p, 
                                                    self.psi_hzx_p
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
                                                    self.Hx, 
                                                    self.Hz, 
                                                    self.diffyEx, 
                                                    self.diffyEz, 
                                                    self.psi_hxy_p, 
                                                    self.psi_hzy_p
                                                )

                if '-' in self.PMLregion.get('y'):

                    self.clib_PML.PML_updateH_my( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappay, self.PMLby, self.PMLay, \
                                                    self.mu_Hx, self.mu_Hz, \
                                                    self.mcon_Hx, self.mcon_Hz, \
                                                    self.Hx, 
                                                    self.Hz, 
                                                    self.diffyEx, 
                                                    self.diffyEz, 
                                                    self.psi_hxy_m, 
                                                    self.psi_hzy_m
                                                )

            if 'z' in self.PMLregion.keys():

                if '+' in self.PMLregion.get('z'):

                    self.clib_PML.PML_updateH_pz( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappaz, self.PMLbz, self.PMLaz, \
                                                    self.mu_Hx, self.mu_Hy, \
                                                    self.mcon_Hx, self.mcon_Hy, \
                                                    self.Hx, 
                                                    self.Hy, 
                                                    self.diffzEx, 
                                                    self.diffzEy, 
                                                    self.psi_hxz_p, 
                                                    self.psi_hyz_p
                                                )

                if '-' in self.PMLregion.get('z'):

                    self.clib_PML.PML_updateH_mz( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappaz, self.PMLbz, self.PMLaz, \
                                                    self.mu_Hx, self.mu_Hy, \
                                                    self.mcon_Hx, self.mcon_Hy, \
                                                    self.Hx, 
                                                    self.Hy, 
                                                    self.diffzEx, 
                                                    self.diffzEy, 
                                                    self.psi_hxz_m, 
                                                    self.psi_hyz_m
                                                )

        #-----------------------------------------------------------#
        #------------ Apply PBC along y when it is given -----------#
        #-----------------------------------------------------------#

        if self.myPBCregion_y == True:

            # Ranks except the last rank.
            if self.MPIsize == 0 or self.MPIrank < (self.MPIsize-1):

                self.clib_PBC.py_rankFM (\
                                            self.myNx, self.Ny, self.Nz, \
                                            self.dt, self.dx, self.dy, self.dz, \
                                            self.mu_Hx, self.mu_Hz, \
                                            self.mcon_Hx, self.mcon_Hz, \
                                            recvEylast, 
                                            self.Hx, 
                                            self.Hz, 
                                            self.Ex, 
                                            self.Ey, 
                                            self.Ez, 
                                            self.diffxEy, 
                                            self.diffyEx, 
                                            self.diffyEz, 
                                            self.diffzEy
                                        )

                # The first rank apply PBC on PML region.
                if self.MPIrank == 0 and '-' in self.PMLregion.get('x'):

                    self.clib_PBC.mxPML_pyPBC   (\
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt,\
                                                    self.PMLkappax, self.PMLbx, self.PMLax,\
                                                    self.mu_Hz, self.mcon_Hz,\
                                                    self.Hz, 
                                                    self.diffxEy, 
                                                    self.psi_hzx_m
                                                )

            # The last rank.
            elif self.MPIrank == (self.MPIsize-1):
                self.clib_PBC.py_rank_L (\
                                            self.myNx, self.Ny, self.Nz, \
                                            self.dt, self.dx, self.dy, self.dz, \
                                            self.mu_Hx, self.mu_Hz, \
                                            self.mcon_Hx, self.mcon_Hz, \
                                            self.Hx, 
                                            self.Hz, 
                                            self.Ex, 
                                            self.Ey, 
                                            self.Ez, 
                                            self.diffxEy, 
                                            self.diffyEx, 
                                            self.diffyEz, 
                                            self.diffzEy
                                        )

                # The last rank apply PBC on PML region.
                if '-' in self.PMLregion.get('x'):

                    self.clib_PBC.pxPML_pyPBC   (\
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt,\
                                                    self.PMLkappax, self.PMLbx, self.PMLax,\
                                                    self.mu_Hz, self.mcon_Hz,\
                                                    self.Hz, 
                                                    self.diffxEy, 
                                                    self.psi_hzx_p
                                                )


        else: pass

        #-----------------------------------------------------------#
        #------------ Apply PBC along z when it is given -----------#
        #-----------------------------------------------------------#

        if self.myPBCregion_z == True:

            # Ranks except the last rank.
            if self.MPIsize == 0 or self.MPIrank < (self.MPIsize-1):
                self.clib_PBC.pz_rankFM (\
                                            self.myNx, self.Ny, self.Nz, \
                                            self.dt, self.dx, self.dy, self.dz, \
                                            self.mu_Hx, self.mu_Hy, \
                                            self.mcon_Hx, self.mcon_Hy, \
                                            recvEzlast, 
                                            self.Hx, 
                                            self.Hy, 
                                            self.Ex, 
                                            self.Ey, 
                                            self.Ez, 
                                            self.diffxEz, 
                                            self.diffyEz, 
                                            self.diffzEx, 
                                            self.diffzEy
                                        )

                # The first rank apply PBC on PML region.
                if self.MPIrank == 0 and '-' in self.PMLregion.get('x'):

                    self.clib_PBC.mxPML_pzPBC   (\
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt,\
                                                    self.PMLkappax, self.PMLbx, self.PMLax,\
                                                    self.mu_Hy, self.mcon_Hy,\
                                                    self.Hy, 
                                                    self.diffxEz, 
                                                    self.psi_hyx_m
                                                )

            # The last rank.
            else:
                self.clib_PBC.pz_rank_L (\
                                            self.myNx, self.Ny, self.Nz, \
                                            self.dt, self.dx, self.dy, self.dz, \
                                            self.mu_Hx, self.mu_Hy, \
                                            self.mcon_Hx, self.mcon_Hy, \
                                            self.Hx, 
                                            self.Hy, 
                                            self.Ex, 
                                            self.Ey, 
                                            self.Ez, 
                                            self.diffxEz, 
                                            self.diffyEz, 
                                            self.diffzEx, 
                                            self.diffzEy
                                        )

                # The last rank apply PBC on PML region.
                if '-' in self.PMLregion.get('x'):

                    self.clib_PBC.pxPML_pzPBC   (\
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt,\
                                                    self.PMLkappax, self.PMLbx, self.PMLax,\
                                                    self.mu_Hy, self.mcon_Hy,\
                                                    self.Hy, 
                                                    self.diffxEz, 
                                                    self.psi_hyx_p
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

            sendHylast = self.Hy[-1,:,:].copy()
            sendHzlast = self.Hz[-1,:,:].copy()

            self.MPIcomm.send(sendHylast, dest=(self.MPIrank+1), tag=(tstep*100+3))
            self.MPIcomm.send(sendHzlast, dest=(self.MPIrank+1), tag=(tstep*100+5))
        
        else: pass

        #---------------------------------------------------------#
        #--------- MPI recv Hy and Hz from previous rank ---------#
        #---------------------------------------------------------#

        if self.MPIrank > 0 and self.MPIrank < self.MPIsize:

            recvHyfirst = self.MPIcomm.recv( source=(self.MPIrank-1), tag=(tstep*100+3))
            recvHzfirst = self.MPIcomm.recv( source=(self.MPIrank-1), tag=(tstep*100+5))
        
        else: pass

        #-----------------------------------------------------------#
        #---------------------- Get derivatives --------------------#
        #-----------------------------------------------------------#

        if self.MPIrank == 0:

            self.clib_core.get_diff_of_H_rank_F(\
                                                self.myNx, self.Ny, self.Nz,\
                                                self.dt, self.dx, self.dy, self.dz, \
                                                self.Hx, 
                                                self.Hy, 
                                                self.Hz, 
                                                self.diffxHy, 
                                                self.diffxHz, 
                                                self.diffyHx, 
                                                self.diffyHz, 
                                                self.diffzHx, 
                                                self.diffzHy
                                                )
        else:

            self.clib_core.get_diff_of_H_rankML(\
                                                self.myNx, self.Ny, self.Nz,\
                                                self.dt, self.dx, self.dy, self.dz, \
                                                recvHyfirst, 
                                                recvHzfirst, 
                                                self.Hx, 
                                                self.Hy, 
                                                self.Hz, 
                                                self.diffxHy, 
                                                self.diffxHz, 
                                                self.diffyHx, 
                                                self.diffyHz, 
                                                self.diffzHx, 
                                                self.diffzHy
                                                )

        #-----------------------------------------------------------#
        #--------------- Cast basic update equations ---------------#
        #-----------------------------------------------------------#

        if self.MPIrank == 0:

            self.clib_core.updateE_rank_F   (\
                                                self.myNx, self.Ny, self.Nz,\
                                                self.dt, \
                                                self.eps_Ex, self.eps_Ey, self.eps_Ez, \
                                                self.econ_Ex, self.econ_Ey, self.econ_Ez, \
                                                self.Ex, 
                                                self.Ey, 
                                                self.Ez, 
                                                self.diffxHy, 
                                                self.diffxHz, 
                                                self.diffyHx, 
                                                self.diffyHz, 
                                                self.diffzHx, 
                                                self.diffzHy
                                            )

        else:

            self.clib_core.updateE_rankML   (\
                                                self.myNx, self.Ny, self.Nz,\
                                                self.dt, \
                                                self.eps_Ex, self.eps_Ey, self.eps_Ez, \
                                                self.econ_Ex, self.econ_Ey, self.econ_Ez, \
                                                self.Ex, 
                                                self.Ey, 
                                                self.Ez, 
                                                self.diffxHy, 
                                                self.diffxHz, 
                                                self.diffyHx, 
                                                self.diffyHz, 
                                                self.diffzHx, 
                                                self.diffzHy
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
                                                    self.Ey, 
                                                    self.Ez, 
                                                    self.diffxHy, 
                                                    self.diffxHz, 
                                                    self.psi_eyx_p, 
                                                    self.psi_ezx_p
                                                )

                if '-' in self.PMLregion.get('x'):

                    self.clib_PML.PML_updateE_mx( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappax, self.PMLbx, self.PMLax, \
                                                    self.eps_Ey, self.eps_Ez, \
                                                    self.econ_Ey, self.econ_Ez, \
                                                    self.Ey, 
                                                    self.Ez, 
                                                    self.diffxHy, 
                                                    self.diffxHz, 
                                                    self.psi_eyx_m,
                                                    self.psi_ezx_m
                                                )

            if 'y' in self.PMLregion.keys():

                if '+' in self.PMLregion.get('y'):

                    self.clib_PML.PML_updateE_py( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappay, self.PMLby, self.PMLay, \
                                                    self.eps_Ex, self.eps_Ez, \
                                                    self.econ_Ex, self.econ_Ez, \
                                                    self.Ex, 
                                                    self.Ez, 
                                                    self.diffyHx, 
                                                    self.diffyHz, 
                                                    self.psi_exy_p, 
                                                    self.psi_ezy_p
                                                )

                if '-' in self.PMLregion.get('y'):

                    self.clib_PML.PML_updateE_my( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappay, self.PMLby, self.PMLay, \
                                                    self.eps_Ex, self.eps_Ez, \
                                                    self.econ_Ex, self.econ_Ez, \
                                                    self.Ex, 
                                                    self.Ez, 
                                                    self.diffyHx, 
                                                    self.diffyHz, 
                                                    self.psi_exy_m, 
                                                    self.psi_ezy_m
                                                )

            if 'z' in self.PMLregion.keys():
                if '+' in self.PMLregion.get('z'):
                    self.clib_PML.PML_updateE_pz( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappaz, self.PMLbz, self.PMLaz, \
                                                    self.eps_Ex, self.eps_Ey, \
                                                    self.econ_Ex, self.econ_Ey, \
                                                    self.Ex, 
                                                    self.Ey, 
                                                    self.diffzHx, 
                                                    self.diffzHy, 
                                                    self.psi_exz_p, 
                                                    self.psi_eyz_p
                                                )

                if '-' in self.PMLregion.get('z'):
                    self.clib_PML.PML_updateE_mz( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappaz, self.PMLbz, self.PMLaz, \
                                                    self.eps_Ex, self.eps_Ey, \
                                                    self.econ_Ex, self.econ_Ey, \
                                                    self.Ex, 
                                                    self.Ey, 
                                                    self.diffzHx, 
                                                    self.diffzHy, 
                                                    self.psi_exz_m, 
                                                    self.psi_eyz_m
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
                                                    self.Ex, 
                                                    self.Ez, 
                                                    self.diffyHx, 
                                                    self.diffyHz, 
                                                    self.psi_exy_p, 
                                                    self.psi_ezy_p
                                                )

                if '-' in self.PMLregion.get('y'):

                    self.clib_PML.PML_updateE_my( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappay, self.PMLby, self.PMLay, \
                                                    self.eps_Ex, self.eps_Ez, \
                                                    self.econ_Ex, self.econ_Ez, \
                                                    self.Ex, 
                                                    self.Ez, 
                                                    self.diffyHx, 
                                                    self.diffyHz, 
                                                    self.psi_exy_m, 
                                                    self.psi_ezy_m
                                                )

            if 'z' in self.PMLregion.keys():
                if '+' in self.PMLregion.get('z'):
                    self.clib_PML.PML_updateE_pz( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappaz, self.PMLbz, self.PMLaz, \
                                                    self.eps_Ex, self.eps_Ey, \
                                                    self.econ_Ex, self.econ_Ey, \
                                                    self.Ex, 
                                                    self.Ey, 
                                                    self.diffzHx, 
                                                    self.diffzHy, 
                                                    self.psi_exz_p, 
                                                    self.psi_eyz_p
                                                )

                if '-' in self.PMLregion.get('z'):
                    self.clib_PML.PML_updateE_mz( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappaz, self.PMLbz, self.PMLaz, \
                                                    self.eps_Ex, self.eps_Ey, \
                                                    self.econ_Ex, self.econ_Ey, \
                                                    self.Ex, 
                                                    self.Ey, 
                                                    self.diffzHx, 
                                                    self.diffzHy, 
                                                    self.psi_exz_m, 
                                                    self.psi_eyz_m
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
                                                    self.Ey, 
                                                    self.Ez, 
                                                    self.diffxHy, 
                                                    self.diffxHz, 
                                                    self.psi_eyx_p, 
                                                    self.psi_ezx_p
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
                                                    self.Ex, 
                                                    self.Ez, 
                                                    self.diffyHx, 
                                                    self.diffyHz, 
                                                    self.psi_exy_p,
                                                    self.psi_ezy_p
                                                )

                if '-' in self.PMLregion.get('y'):

                    self.clib_PML.PML_updateE_my( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappay, self.PMLby, self.PMLay, \
                                                    self.eps_Ex, self.eps_Ez, \
                                                    self.econ_Ex, self.econ_Ez, \
                                                    self.Ex, 
                                                    self.Ez, 
                                                    self.diffyHx, 
                                                    self.diffyHz,
                                                    self.psi_exy_m,
                                                    self.psi_ezy_m
                                                )

            if 'z' in self.PMLregion.keys():
                if '+' in self.PMLregion.get('z'): 
                    self.clib_PML.PML_updateE_pz( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappaz, self.PMLbz, self.PMLaz, \
                                                    self.eps_Ex, self.eps_Ey, \
                                                    self.econ_Ex, self.econ_Ey, \
                                                    self.Ex,
                                                    self.Ey,
                                                    self.diffzHx,
                                                    self.diffzHy,
                                                    self.psi_exz_p,
                                                    self.psi_eyz_p
                                                )

                if '-' in self.PMLregion.get('z'):
                    self.clib_PML.PML_updateE_mz( \
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt, \
                                                    self.PMLkappaz, self.PMLbz, self.PMLaz, \
                                                    self.eps_Ex, self.eps_Ey, \
                                                    self.econ_Ex, self.econ_Ey, \
                                                    self.Ex,
                                                    self.Ey,
                                                    self.diffzHx,
                                                    self.diffzHy,
                                                    self.psi_exz_m,
                                                    self.psi_eyz_m
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
                                            self.Ex, 
                                            self.Ez, 
                                            self.Hx, 
                                            self.Hy, 
                                            self.Hz, 
                                            self.diffxHy, 
                                            self.diffyHx, 
                                            self.diffyHz, 
                                            self.diffzHy
                                        )

                # The first rank apply PBC on PML region.
                if '-' in self.PMLregion.get('x'):

                    self.clib_PBC.mxPML_myPBC   (\
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt,\
                                                    self.PMLkappax, self.PMLbx, self.PMLax,\
                                                    self.eps_Ez, self.econ_Ez,\
                                                    self.Ez, 
                                                    self.diffxHy, 
                                                    self.psi_ezx_m
                                                )

            # Ranks except the first rank.
            else:   
                self.clib_PBC.my_rankML( \
                                            self.myNx, self.Ny, self.Nz, \
                                            self.dt, self.dx, self.dy, self.dz,\
                                            self.eps_Ex, self.eps_Ez, \
                                            self.econ_Ex, self.econ_Ez, \
                                            recvHyfirst,
                                            self.Ex,
                                            self.Ez, 
                                            self.Hx, 
                                            self.Hy, 
                                            self.Hz,
                                            self.diffxHy, 
                                            self.diffyHx, 
                                            self.diffyHz, 
                                            self.diffzHy
                                        )

                # The last rank apply PBC on PML region.
                if self.MPIrank == (self.MPIsize-1) and '-' in self.PMLregion.get('x'):

                    self.clib_PBC.pxPML_myPBC   (\
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt,\
                                                    self.PMLkappax, self.PMLbx, self.PMLax,\
                                                    self.eps_Ez, self.econ_Ez,\
                                                    self.Ez, 
                                                    self.diffxHy,
                                                    self.psi_ezx_p
                                                )

        else: pass

        #-----------------------------------------------------------#
        #------------ Apply PBC along z when it is given -----------#
        #-----------------------------------------------------------#

        if self.myPBCregion_z == True:

            # The first rank.
            if self.MPIrank == 0:

                self.clib_PBC.mz_rank_F (\
                                            self.myNx, self.Ny, self.Nz, \
                                            self.dt, self.dx, self.dy, self.dz,\
                                            self.eps_Ex, self.eps_Ez, \
                                            self.econ_Ex, self.econ_Ez, \
                                            self.Ex, 
                                            self.Ey, 
                                            self.Hx, 
                                            self.Hy, 
                                            self.Hz, 
                                            self.diffxHz, 
                                            self.diffyHz, 
                                            self.diffzHx, 
                                            self.diffzHy
                                        )

                # The first rank applies PBC on PML region.
                if '-' in self.PMLregion.get('x'):

                    self.clib_PBC.mxPML_mzPBC   (\
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt,\
                                                    self.PMLkappax, self.PMLbx, self.PMLax,\
                                                    self.eps_Ey, self.econ_Ey,\
                                                    self.Ey, 
                                                    self.diffxHz, 
                                                    self.psi_eyx_m
                                                )

            # Ranks except the first rank.
            else:

                self.clib_PBC.mz_rankML (\
                                            self.myNx, self.Ny, self.Nz, \
                                            self.dt, self.dx, self.dy, self.dz,\
                                            self.eps_Ex, self.eps_Ez, \
                                            self.econ_Ex, self.econ_Ez, \
                                            recvHzfirst, 
                                            self.Ex, 
                                            self.Ey, 
                                            self.Hx, 
                                            self.Hy, 
                                            self.Hz, 
                                            self.diffxHz, 
                                            self.diffyHz, 
                                            self.diffzHx, 
                                            self.diffzHy
                                        )

                # The last rank applies PBC on PML region.
                if self.MPIrank == (self.MPIsize-1) and '-' in self.PMLregion.get('x'):

                    self.clib_PBC.pxPML_mzPBC   (\
                                                    self.myNx, self.Ny, self.Nz, self.npml,\
                                                    self.dt,\
                                                    self.PMLkappax, self.PMLbx, self.PMLax,\
                                                    self.eps_Ey, self.econ_Ey,\
                                                    self.Ey, 
                                                    self.diffxHz, 
                                                    self.psi_eyx_p
                                                )


        else: pass


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

        self.dtype    = dtype
        self.MPIcomm  = MPI.COMM_WORLD
        self.MPIrank  = self.MPIcomm.Get_rank()
        self.MPIsize  = self.MPIcomm.Get_size()
        self.hostname = MPI.Get_processor_name()

        assert len(grid)    == 3, "Simulation grid should be a tuple with length 3."
        assert len(gridgap) == 3, "Argument 'gridgap' should be a tuple with length 3."

        self.tsteps = tsteps        

        self.grid = grid
        self.Nx   = self.grid[0]
        self.Ny   = self.grid[1]
        self.Nz   = self.grid[2]
        self.TOTAL_NUM_GRID = self.Nx * self.Ny * self.Nz
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
        self.myBBCregion_x = False
        self.myBBCregion_y = False
        self.myBBCregion_z = False

        assert self.dt < self.maxdt, "Time interval is too big so that causality is broken. Lower the courant number."
        assert float(self.Nx) % self.MPIsize == 0., "Nx must be a multiple of the number of nodes."
        
        ############################################################################
        ################# Set the loc_grid each node should possess ################
        ############################################################################

        self.myNx     = int(self.Nx/self.MPIsize)
        self.loc_grid = (self.myNx, self.Ny, self.Nz)

        self.Ex = np.zeros(self.loc_grid, dtype=self.dtype)
        self.Ey = np.zeros(self.loc_grid, dtype=self.dtype)
        self.Ez = np.zeros(self.loc_grid, dtype=self.dtype)

        self.Hx = np.zeros(self.loc_grid, dtype=self.dtype)
        self.Hy = np.zeros(self.loc_grid, dtype=self.dtype)
        self.Hz = np.zeros(self.loc_grid, dtype=self.dtype)

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
            self.myNx_indice.append(     (xsrt, xend))

        self.MPIcomm.Barrier()
        #print("rank {:>2}:\tmy xindex: {},\tmy xslice: {}" \
        #       .format(self.MPIrank, self.myNx_indice[self.MPIrank], self.myNx_slices[self.MPIrank]))

    def get_SF(self, TF, IF):
        """Get scattered field

        Paramters
        ---------
        TF: Basic3D class object.
            Total field.

        IF: Basic3D class object.
            Input field.

        Returns
        -------
        None
        """

        self.Ex = TF.Ex - IF.Ex
        self.Ey = TF.Ey - IF.Ey
        self.Ez = TF.Ez - IF.Ez

        self.Hx = TF.Hx - IF.Hx
        self.Hy = TF.Hy - IF.Hy
        self.Hz = TF.Hz - IF.Hz
