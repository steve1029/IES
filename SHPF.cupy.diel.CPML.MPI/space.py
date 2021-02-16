import os
from mpi4py import MPI
from scipy.constants import c, mu_0, epsilon_0
import numpy as np
import cupy as cp

class Basic3D:
    
    def __init__(self, grid, gridgap, dt, tsteps, field_dtype, mmtdtype, **kwargs):
        """Create Simulation Space.

            ex) Space.grid((128,128,600), (50*nm,50*nm,5*nm), dtype=self.xp.complex64)

        PARAMETERS
        ----------
        grid : tuple
            define the x,y,z grid.

        gridgap : tuple
            define the dx, dy, dz.

        field_dtype : class numpy dtype
            dtype for field array. Choose self.xp.float32 or self.xp.float64

        mmtdtype : class numpy dtype
            dtype for FFT momentum vector array. Choose self.xp.complex64 or self.xp.complex128

        kwargs : string
            
            supported arguments
            -------------------

            courant : float
                Set the courant number. For HPF, default is 1/4.

        RETURNS
        -------
        None
        """

        self.nm = 1e-9
        self.um = 1e-6  

        self.field_dtype   = field_dtype
        self.mmtdtype   = mmtdtype
        self.MPIcomm  = MPI.COMM_WORLD
        self.MPIrank  = self.MPIcomm.Get_rank()
        self.MPIsize  = self.MPIcomm.Get_size()
        self.hostname = MPI.Get_processor_name()

        assert len(grid)    == 3, "Simulation grid should be a tuple with length 3."
        assert len(gridgap) == 3, "Argument 'gridgap' should be a tuple with length 3."

        self.tsteps = tsteps        

        self.grid = grid
        self.Nx = self.grid[0]
        self.Ny = self.grid[1]
        self.Nz = self.grid[2]
        self.TOTAL_NUM_GRID = self.Nx * self.Ny * self.Nz
        self.TOTAL_NUM_GRID_SIZE = (self.field_dtype(1).nbytes * self.TOTAL_NUM_GRID) / 1024 / 1024
        self.dimension = 3
        
        self.Nxc = round(self.Nx / 2)
        self.Nyc = round(self.Ny / 2)
        self.Nzc = round(self.Nz / 2)
        
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

        self.method = 'SHPF'
        self.engine = 'cupy'
        self.courant = 1./4

        if kwargs.get('engine') != None: self.engine = kwargs.get('engine')
        if kwargs.get('method') != None: self.method = kwargs.get('method')
        if kwargs.get('courant') != None: self.courant = kwargs.get('courant')

        if self.method == 'PSTD': assert self.MPIsize == 1, "MPI size must be 1 if you want to use the PSTD method."

        assert self.engine == 'numpy' or self.engine == 'cupy'

        if self.engine == 'cupy' : self.xp = cp
        else: self.xp = np

        self.dt = dt
        self.maxdt = 1. / c / self.xp.sqrt( (1./self.dx)**2 + (1./self.dy)**2 + (1./self.dz)**2 )

        assert (c * self.dt * self.xp.sqrt( (1./self.dx)**2 + (1./self.dy)**2 + (1./self.dz)**2 )) < 1.

        """
        For more details about maximum dt in the Hybrid PSTD-FDTD method, see
        Combining the FDTD and PSTD methods, Y.F.Leung, C.H. Chan,
        Microwave and Optical technology letters, Vol.23, No.4, November 20 1999.
        """

        assert self.dt < self.maxdt, "Time interval is too big so that causality is broken. Lower the courant number."
        assert float(self.Nx) % self.MPIsize == 0., "Nx must be a multiple of the number of nodes."
        
        ############################################################################
        ################# Set the loc_grid each node should possess ################
        ############################################################################

        self.myNx     = round(self.Nx/self.MPIsize)
        self.loc_grid = (self.myNx, self.Ny, self.Nz)

        self.Ex = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.Ey = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.Ez = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.Hx = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.Hy = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.Hz = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)

        ###############################################################################
        ####################### Slices of xgrid that each node got ####################
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

    def malloc(self):

        if self.field_dtype == np.complex64 or self.field_dtype == np.complex128:

            self.fft = self.xp.fft.fftn
            self.ifft = self.xp.fft.ifftn
            self.fftfreq = self.xp.fft.fftfreq

        elif self.field_dtype == np.float32 or self.field_dtype == np.float64:

            self.fft = self.xp.fft.rfftn
            self.ifft = self.xp.fft.irfftn
            self.fftfreq = self.xp.fft.rfftfreq

        else:
            raise ValueError("Please use field_dtype for numpy dtype!")

        self.kx = self.fftfreq(self.Nx, self.dx) * 2 * self.xp.pi
        self.ky = self.fftfreq(self.Ny, self.dy) * 2 * self.xp.pi
        self.kz = self.fftfreq(self.Nz, self.dz) * 2 * self.xp.pi

        if self.engine == 'cupy':

            self.ikx = (1j*self.kx[:,None,None]).astype(self.mmtdtype)
            self.iky = (1j*self.ky[None,:,None]).astype(self.mmtdtype)
            self.ikz = (1j*self.kz[None,None,:]).astype(self.mmtdtype)

            self.xpshift = self.xp.exp(self.ikx*+self.dx/2).astype(self.mmtdtype)
            self.xmshift = self.xp.exp(self.ikx*-self.dx/2).astype(self.mmtdtype)

            self.ypshift = self.xp.exp(self.iky*+self.dy/2).astype(self.mmtdtype)
            self.ymshift = self.xp.exp(self.iky*-self.dy/2).astype(self.mmtdtype)

            self.zpshift = self.xp.exp(self.ikz*+self.dz/2).astype(self.mmtdtype)
            self.zmshift = self.xp.exp(self.ikz*-self.dz/2).astype(self.mmtdtype)

        else:

            nax = np.newaxis
            self.iky = 1j*self.ky[nax,:,nax]
            self.ikz = 1j*self.kz[nax,nax,:]
            self.ypshift = self.xp.exp(self.iky*-self.dy/2)[None,:,None]
            self.zpshift = self.xp.exp(self.ikz*-self.dz/2)[None,None,:]
            self.ymshift = self.xp.exp(self.iky*+self.dy/2)[None,:,None]
            self.zmshift = self.xp.exp(self.ikz*+self.dz/2)[None,None,:]

        self.diffxEy = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.diffxEz = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.diffyEx = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.diffyEz = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.diffzEx = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.diffzEy = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)

        self.diffxHy = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.diffxHz = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.diffyHx = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.diffyHz = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.diffzHx = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.diffzHy = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)

        self.eps_Ex = self.xp.ones(self.loc_grid, dtype=self.field_dtype) * epsilon_0
        self.eps_Ey = self.xp.ones(self.loc_grid, dtype=self.field_dtype) * epsilon_0
        self.eps_Ez = self.xp.ones(self.loc_grid, dtype=self.field_dtype) * epsilon_0

        self.mu_Hx  = self.xp.ones(self.loc_grid, dtype=self.field_dtype) * mu_0
        self.mu_Hy  = self.xp.ones(self.loc_grid, dtype=self.field_dtype) * mu_0
        self.mu_Hz  = self.xp.ones(self.loc_grid, dtype=self.field_dtype) * mu_0

        self.econ_Ex = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.econ_Ey = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.econ_Ez = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)

        self.mcon_Hx = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.mcon_Hy = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
        self.mcon_Hz = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
 
    def apply_PML(self, region, npml):

        self.PMLregion  = region
        self.npml       = npml
        self.PMLgrading = 2 * self.npml

        self.rc0   = 1.e-16                             # reflection coefficient
        self.imp   = self.xp.sqrt(mu_0/epsilon_0)            # impedence
        self.gO    = 3.                                 # gradingOrder
        self.sO    = 3.                                 # scalingOrder
        self.bdw_x = (self.PMLgrading-1) * self.dx      # PML thickness along x (Boundarywidth)
        self.bdw_y = (self.PMLgrading-1) * self.dy      # PML thickness along y
        self.bdw_z = (self.PMLgrading-1) * self.dz      # PML thickness along z

        self.PMLsigmamaxx = -(self.gO+1) * self.xp.log(self.rc0) / (2*self.imp*self.bdw_x)
        self.PMLsigmamaxy = -(self.gO+1) * self.xp.log(self.rc0) / (2*self.imp*self.bdw_y)
        self.PMLsigmamaxz = -(self.gO+1) * self.xp.log(self.rc0) / (2*self.imp*self.bdw_z)

        self.PMLkappamaxx = 1.
        self.PMLkappamaxy = 1.
        self.PMLkappamaxz = 1.

        self.PMLalphamaxx = 0.02
        self.PMLalphamaxy = 0.02
        self.PMLalphamaxz = 0.02

        self.PMLsigmax = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)
        self.PMLalphax = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)
        self.PMLkappax = self.xp.ones (self.PMLgrading, dtype=self.field_dtype)

        self.PMLsigmay = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)
        self.PMLalphay = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)
        self.PMLkappay = self.xp.ones (self.PMLgrading, dtype=self.field_dtype)

        self.PMLsigmaz = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)
        self.PMLalphaz = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)
        self.PMLkappaz = self.xp.ones (self.PMLgrading, dtype=self.field_dtype)

        self.PMLbx = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)
        self.PMLby = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)
        self.PMLbz = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)

        self.PMLax = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)
        self.PMLay = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)
        self.PMLaz = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)

        #------------------------------------------------------------------------------------------------#
        #------------------------------- Grading kappa, sigma and alpha ---------------------------------#
        #------------------------------------------------------------------------------------------------#

        for key, value in self.PMLregion.items():

            if   key == 'x' and value != '':

                self.psi_eyx_p = self.xp.zeros((npml, self.Ny, self.Nz), dtype=self.field_dtype)
                self.psi_ezx_p = self.xp.zeros((npml, self.Ny, self.Nz), dtype=self.field_dtype)
                self.psi_hyx_p = self.xp.zeros((npml, self.Ny, self.Nz), dtype=self.field_dtype)
                self.psi_hzx_p = self.xp.zeros((npml, self.Ny, self.Nz), dtype=self.field_dtype)

                self.psi_eyx_m = self.xp.zeros((npml, self.Ny, self.Nz), dtype=self.field_dtype)
                self.psi_ezx_m = self.xp.zeros((npml, self.Ny, self.Nz), dtype=self.field_dtype)
                self.psi_hyx_m = self.xp.zeros((npml, self.Ny, self.Nz), dtype=self.field_dtype)
                self.psi_hzx_m = self.xp.zeros((npml, self.Ny, self.Nz), dtype=self.field_dtype)

                loc = self.xp.arange(self.PMLgrading) * self.dx / self.bdw_x
                self.PMLsigmax = self.PMLsigmamaxx * (loc **self.gO)
                self.PMLkappax = 1 + ((self.PMLkappamaxx-1) * (loc **self.gO))
                self.PMLalphax = self.PMLalphamaxx * ((1-loc) **self.sO)

            elif key == 'y' and value != '':

                self.psi_exy_p = self.xp.zeros((self.myNx, npml, self.Nz), dtype=self.field_dtype)
                self.psi_ezy_p = self.xp.zeros((self.myNx, npml, self.Nz), dtype=self.field_dtype)
                self.psi_hxy_p = self.xp.zeros((self.myNx, npml, self.Nz), dtype=self.field_dtype)
                self.psi_hzy_p = self.xp.zeros((self.myNx, npml, self.Nz), dtype=self.field_dtype)

                self.psi_exy_m = self.xp.zeros((self.myNx, npml, self.Nz), dtype=self.field_dtype)
                self.psi_ezy_m = self.xp.zeros((self.myNx, npml, self.Nz), dtype=self.field_dtype)
                self.psi_hxy_m = self.xp.zeros((self.myNx, npml, self.Nz), dtype=self.field_dtype)
                self.psi_hzy_m = self.xp.zeros((self.myNx, npml, self.Nz), dtype=self.field_dtype)

                loc  = self.xp.arange(self.PMLgrading) * self.dy / self.bdw_y
                self.PMLsigmay = self.PMLsigmamaxy * (loc **self.gO)
                self.PMLkappay = 1 + ((self.PMLkappamaxy-1) * (loc **self.gO))
                self.PMLalphay = self.PMLalphamaxy * ((1-loc) **self.sO)

            elif key == 'z' and value != '':

                self.psi_exz_p = self.xp.zeros((self.myNx, self.Ny, npml), dtype=self.field_dtype)
                self.psi_eyz_p = self.xp.zeros((self.myNx, self.Ny, npml), dtype=self.field_dtype)
                self.psi_hxz_p = self.xp.zeros((self.myNx, self.Ny, npml), dtype=self.field_dtype)
                self.psi_hyz_p = self.xp.zeros((self.myNx, self.Ny, npml), dtype=self.field_dtype)

                self.psi_exz_m = self.xp.zeros((self.myNx, self.Ny, npml), dtype=self.field_dtype)
                self.psi_eyz_m = self.xp.zeros((self.myNx, self.Ny, npml), dtype=self.field_dtype)
                self.psi_hxz_m = self.xp.zeros((self.myNx, self.Ny, npml), dtype=self.field_dtype)
                self.psi_hyz_m = self.xp.zeros((self.myNx, self.Ny, npml), dtype=self.field_dtype)

                loc  = self.xp.arange(self.PMLgrading) * self.dz / self.bdw_z
                self.PMLsigmaz = self.PMLsigmamaxz * (loc **self.gO)
                self.PMLkappaz = 1 + ((self.PMLkappamaxz-1) * (loc **self.gO))
                self.PMLalphaz = self.PMLalphamaxz * ((1-loc) **self.sO)

        #------------------------------------------------------------------------------------------------#
        #--------------------------------- Get 'b' and 'a' for CPML theory ------------------------------#
        #------------------------------------------------------------------------------------------------#

        if 'x' in self.PMLregion.keys() and self.PMLregion.get('x') != '':
            self.PMLbx = self.xp.exp(-(self.PMLsigmax/self.PMLkappax + self.PMLalphax) * self.dt / epsilon_0)
            self.PMLax = self.PMLsigmax \
                    / (self.PMLsigmax*self.PMLkappax + self.PMLalphax*self.PMLkappax**2) * (self.PMLbx - 1.)

        if 'y' in self.PMLregion.keys() and self.PMLregion.get('y') != '':
            self.PMLby = self.xp.exp(-(self.PMLsigmay/self.PMLkappay + self.PMLalphay) * self.dt / epsilon_0)
            self.PMLay = self.PMLsigmay \
                    / (self.PMLsigmay*self.PMLkappay + self.PMLalphay*self.PMLkappay**2) * (self.PMLby - 1.)

        if 'z' in self.PMLregion.keys() and self.PMLregion.get('z') != '':
            self.PMLbz = self.xp.exp(-(self.PMLsigmaz/self.PMLkappaz + self.PMLalphaz) * self.dt / epsilon_0)
            self.PMLaz = self.PMLsigmaz \
                    / (self.PMLsigmaz*self.PMLkappaz + self.PMLalphaz*self.PMLkappaz**2) * (self.PMLbz - 1.)

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

        if self.xp == cp:

            eps_Ex = cp.asnumpy(self.eps_Ex)
            eps_Ey = cp.asnumpy(self.eps_Ey)
            eps_Ez = cp.asnumpy(self.eps_Ez)
            mu_Hx  = cp.asnumpy(self. mu_Hx)
            mu_Hy  = cp.asnumpy(self. mu_Hy)
            mu_Hz  = cp.asnumpy(self. mu_Hz)

        else:

            eps_Ex = self.eps_Ex
            eps_Ey = self.eps_Ey
            eps_Ez = self.eps_Ez
            mu_Hx  = self. mu_Hx
            mu_Hy  = self. mu_Hy
            mu_Hz  = self. mu_Hz

        f.create_dataset('eps_Ex',  data=eps_Ex)
        f.create_dataset('eps_Ey',  data=eps_Ey)
        f.create_dataset('eps_Ez',  data=eps_Ez)
        f.create_dataset( 'mu_Hx',  data=mu_Hx)
        f.create_dataset( 'mu_Hy',  data=mu_Hy)
        f.create_dataset( 'mu_Hz',  data=mu_Hz)
            
        self.MPIcomm.Barrier()

        return

    def apply_BBC(self, region):
        """Apply Bloch Boundary Condition.

        Parameters
        ----------
        region: dictionary
            Choose to apply BBC or not, along all axes.

        Returns
        -------
        None
        """

        self.BBC_called = True
        self.apply_BBCx = region.get('x')
        self.apply_BBCy = region.get('y')
        self.apply_BBCz = region.get('z')

        if self.method == 'FDTD':
            
            if self.apply_BBCx == True: assert self.MPIsize == 1

        elif self.method == 'SHPF':

            if self.apply_BBCx == True: 

                #raise ValueError("BBC along x-axis is not developed yet!")
                assert self.MPIsize == 1

                self.ez_at_Hy = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
                self.ey_at_Hz = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)

                self.hz_at_Ey = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
                self.hy_at_Ez = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)

            if self.apply_BBCy == True: 

                self.hz_at_Ex = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
                self.hx_at_Ez = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)

                self.ez_at_Hx = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
                self.ex_at_Hz = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)

            if self.apply_BBCz == True: 

                self.hy_at_Ex = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
                self.hx_at_Ey = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)

                self.ey_at_Hx = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
                self.ex_at_Hy = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)

            if self.BBC_called == False:

                self.apply_BBCx = False
                self.apply_BBCy = False
                self.apply_BBCz = False

        return

    def apply_PBC(self, region):
        """Apply Periodic Boundary Condition.

        Parameters
        ----------
        region: dictionary
            Choose to apply BBC or not, along all axes.

        Returns
        -------
        None
        """

        self.PBC_called = True
        self.apply_PBCx = region.get('x')
        self.apply_PBCy = region.get('y')
        self.apply_PBCz = region.get('z')

        if self.apply_PBCx == True: assert self.MPIsize == 1

        return

    def updateH(self,tstep) :
        
        #--------------------------------------------------------------#
        #------------ MPI send Ex and Ey to previous rank -------------#
        #--------------------------------------------------------------#

        if self.MPIrank != 0:

            if self.engine == 'cupy':
                sendEyfirst = cp.asnumpy(self.Ey[0,:,:])
                sendEzfirst = cp.asnumpy(self.Ez[0,:,:])

            else: # engine is numpy.
                sendEyfirst = self.Ey[0,:,:].copy()
                sendEzfirst = self.Ez[0,:,:].copy()

            self.MPIcomm.send( sendEyfirst, dest=(self.MPIrank-1), tag=(tstep*100+9 ))
            self.MPIcomm.send( sendEzfirst, dest=(self.MPIrank-1), tag=(tstep*100+11))

        #-----------------------------------------------------------#
        #------------ MPI recv Ex and Ey from next rank ------------#
        #-----------------------------------------------------------#

        if self.MPIrank != (self.MPIsize-1):

            self.recvEylast = self.MPIcomm.recv( source=(self.MPIrank+1), tag=(tstep*100+9 ))
            self.recvEzlast = self.MPIcomm.recv( source=(self.MPIrank+1), tag=(tstep*100+11))

            if self.engine == 'cupy':

                self.recvEylast = cp.asarray(self.recvEylast)
                self.recvEzlast = cp.asarray(self.recvEzlast)

        #----------------------------------------------------------------------#
        #---------------- Apply BBC when the method is the FDTD ---------------#
        #----------------------------------------------------------------------#

        if self.method == 'FDTD': self._updateH_BBC_FDTD()
        #if self.method == 'PSTD' and self.BBC == True: self._updateH_BBC_FDTD()

        #-----------------------------------------------------------#
        #---------------------- Get derivatives --------------------#
        #-----------------------------------------------------------#

        if self.method == 'SHPF':

            # To update Hx
            self.diffyEz = self.ifft(self.iky*self.ypshift*self.fft(self.Ez, axes=(1,)), axes=(1,))
            self.diffzEy = self.ifft(self.ikz*self.zpshift*self.fft(self.Ey, axes=(2,)), axes=(2,))

            # To update Hy
            self.diffzEx = self.ifft(self.ikz*self.zpshift*self.fft(self.Ex, axes=(2,)), axes=(2,))
            self.diffxEz[:-1,:,:] = (self.Ez[1:,:,:] - self.Ez[:-1,:,:]) / self.dx

            # To update Hz
            self.diffyEx = self.ifft(self.iky*self.ypshift*self.fft(self.Ex, axes=(1,)), axes=(1,))
            self.diffxEy[:-1,:,:] = (self.Ey[1:,:,:] - self.Ey[:-1,:,:]) / self.dx

            if self.MPIrank != (self.MPIsize-1):

                # No need to update diffzEx and diffyEx because they are already done.
                # To update Hy at x=myNx-1.
                self.diffxEz[-1,:,:] = (self.recvEzlast[:,:] - self.Ez[-1,:,:]) / self.dx

                # To update Hz at x=myNx-1
                self.diffxEy[-1,:,:] = (self.recvEylast[:,:] - self.Ey[-1,:,:]) / self.dx

        elif self.method == 'PSTD':

            # To update Hx
            self.diffyEz = self.ifft(self.iky*self.fft(self.Ez, axes=(1,)), axes=(1,))
            self.diffzEy = self.ifft(self.ikz*self.fft(self.Ey, axes=(2,)), axes=(2,))

            # To update Hy
            self.diffzEx = self.ifft(self.ikz*self.fft(self.Ex, axes=(2,)), axes=(2,))
            self.diffxEz = self.ifft(self.ikx*self.fft(self.Ez, axes=(0,)), axes=(0,))

            # To update Hz
            self.diffyEx = self.ifft(self.iky*self.fft(self.Ex, axes=(1,)), axes=(1,))
            self.diffxEy = self.ifft(self.ikx*self.fft(self.Ey, axes=(0,)), axes=(0,))

        elif self.method == 'FDTD':

            # To update Hx
            self.diffyEz[:,:-1,:-1] = (self.Ez[:,1:,:-1] - self.Ez[:,:-1,:-1]) / self.dy
            self.diffzEy[:,:-1,:-1] = (self.Ey[:,:-1,1:] - self.Ey[:,:-1,:-1]) / self.dz

            # To update Hy
            self.diffzEx[:-1,:,:-1] = (self.Ex[:-1,:,1:] - self.Ex[:-1,:,:-1]) / self.dz
            self.diffxEz[:-1,:,:-1] = (self.Ez[1:,:,:-1] - self.Ez[:-1,:,:-1]) / self.dx

            # To update Hz
            self.diffyEx[:-1,:-1,:] = (self.Ex[:-1,1:,:] - self.Ex[:-1,:-1,:]) / self.dy
            self.diffxEy[:-1,:-1,:] = (self.Ey[1:,:-1,:] - self.Ey[:-1,:-1,:]) / self.dx

            if self.MPIrank != (self.MPIsize-1):

                # To update Hy at x=myNx-1.
                self.diffzEx[-1,:,:-1] = (self.      Ex[-1,:,1:] - self.Ex[-1,:,:-1]) / self.dz
                self.diffxEz[-1,:,:-1] = (self.recvEzlast[:,:-1] - self.Ez[-1,:,:-1]) / self.dx

                # To update Hz at x=myNx-1
                self.diffxEy[-1,:-1,:] = (self.recvEylast[:-1,:] - self.Ey[-1,:-1,:]) / self.dx
                self.diffyEx[-1,:-1,:] = (self.      Ex[-1,1:,:] - self.Ex[-1,:-1,:]) / self.dy

        #-----------------------------------------------------------#
        #--------------- Cast basic update equations ---------------#
        #-----------------------------------------------------------#

        if self.method == 'PSTD':

            CHx1 = (2.*self.mu_Hx - self.mcon_Hx*self.dt) / \
                   (2.*self.mu_Hx + self.mcon_Hx*self.dt)
            CHx2 = (-2*self.dt) / (2.*self.mu_Hx + self.mcon_Hx*self.dt)

            CHy1 = (2.*self.mu_Hy - self.mcon_Hy*self.dt) / \
                   (2.*self.mu_Hy + self.mcon_Hy*self.dt)
            CHy2 = (-2*self.dt) / (2.*self.mu_Hy + self.mcon_Hy*self.dt)

            CHz1 = (2.*self.mu_Hz - self.mcon_Hz*self.dt) / \
                   (2.*self.mu_Hz + self.mcon_Hz*self.dt)
            CHz2 = (-2*self.dt) / (2.*self.mu_Hz + self.mcon_Hz*self.dt)

            self.Hx = CHx1*self.Hx + CHx2*(self.diffyEz - self.diffzEy)
            self.Hy = CHy1*self.Hy + CHy2*(self.diffzEx - self.diffxEz)
            self.Hz = CHz1*self.Hz + CHz2*(self.diffxEy - self.diffyEx)

        if self.method == 'SHPF':

            CHx1 = (2.*self.mu_Hx[:,:,:] - self.mcon_Hx[:,:,:]*self.dt) / \
                   (2.*self.mu_Hx[:,:,:] + self.mcon_Hx[:,:,:]*self.dt)
            CHx2 = (-2*self.dt) / (2.*self.mu_Hx[:,:,:] + self.mcon_Hx[:,:,:]*self.dt)

            CHy1 = (2.*self.mu_Hy[:-1,:,:] - self.mcon_Hy[:-1,:,:]*self.dt) / \
                   (2.*self.mu_Hy[:-1,:,:] + self.mcon_Hy[:-1,:,:]*self.dt)
            CHy2 = (-2*self.dt) / (2.*self.mu_Hy[:-1,:,:] + self.mcon_Hy[:-1,:,:]*self.dt)

            CHz1 = (2.*self.mu_Hz[:-1,:,:] - self.mcon_Hz[:-1,:,:]*self.dt) / \
                   (2.*self.mu_Hz[:-1,:,:] + self.mcon_Hz[:-1,:,:]*self.dt)
            CHz2 = (-2*self.dt) / (2.*self.mu_Hz[:-1,:,:] + self.mcon_Hz[:-1,:,:]*self.dt)

            self.Hx[:  ,:,:] = CHx1*self.Hx[:  ,:,:] + CHx2*(self.diffyEz[:  ,:,:]-self.diffzEy[:  ,:,:])
            self.Hy[:-1,:,:] = CHy1*self.Hy[:-1,:,:] + CHy2*(self.diffzEx[:-1,:,:]-self.diffxEz[:-1,:,:])
            self.Hz[:-1,:,:] = CHz1*self.Hz[:-1,:,:] + CHz2*(self.diffxEy[:-1,:,:]-self.diffyEx[:-1,:,:])

            if self.MPIrank != (self.MPIsize-1):

                # Update Hy and Hz at x=myNx-1
                sli1 = [-1,slice(0,None),slice(0,None)]
                sli2 = [-1,slice(0,None),slice(0,None)]
                self.Hy[sli1] = CHy1[-1,:,:]*self.Hy[sli1] + CHy2[-1,:,:]*(self.diffzEx[sli1]-self.diffxEz[sli1])
                self.Hz[sli2] = CHz1[-1,:,:]*self.Hz[sli2] + CHz2[-1,:,:]*(self.diffxEy[sli2]-self.diffyEx[sli2])

        if self.method == 'FDTD':

            CHx1 = (2.*self.mu_Hx[:,:-1,:-1] - self.mcon_Hx[:,:-1,:-1]*self.dt) / \
                   (2.*self.mu_Hx[:,:-1,:-1] + self.mcon_Hx[:,:-1,:-1]*self.dt)
            CHx2 = (-2*self.dt) / (2.*self.mu_Hx[:,:-1,:-1] + self.mcon_Hx[:,:-1,:-1]*self.dt)

            CHy1 = (2.*self.mu_Hy[:-1,:,:-1] - self.mcon_Hy[:-1,:,:-1]*self.dt) / \
                   (2.*self.mu_Hy[:-1,:,:-1] + self.mcon_Hy[:-1,:,:-1]*self.dt)
            CHy2 = (-2*self.dt) / (2.*self.mu_Hy[:-1,:,:-1] + self.mcon_Hy[:-1,:,:-1]*self.dt)

            CHz1 = (2.*self.mu_Hz[:-1,:-1,:] - self.mcon_Hz[:-1,:-1,:]*self.dt) / \
                   (2.*self.mu_Hz[:-1,:-1,:] + self.mcon_Hz[:-1,:-1,:]*self.dt)
            CHz2 = (-2*self.dt) / (2.*self.mu_Hz[:-1,:-1,:] + self.mcon_Hz[:-1,:-1,:]*self.dt)

            self.Hx[:,:-1,:-1] = CHx1*self.Hx[:,:-1,:-1] + CHx2*(self.diffyEz[:,:-1,:-1]-self.diffzEy[:,:-1,:-1])
            self.Hy[:-1,:,:-1] = CHy1*self.Hy[:-1,:,:-1] + CHy2*(self.diffzEx[:-1,:,:-1]-self.diffxEz[:-1,:,:-1])
            self.Hz[:-1,:-1,:] = CHz1*self.Hz[:-1,:-1,:] + CHz2*(self.diffxEy[:-1,:-1,:]-self.diffyEx[:-1,:-1,:])

            if self.MPIrank != (self.MPIsize-1):

                # Update Hy and Hz at x=myNx-1
                sli1 = [-1,slice(0,None),slice(0,-1)]
                sli2 = [-1,slice(0,-1),slice(0,None)]
                self.Hy[sli1] = CHy1[-1,:,:]*self.Hy[sli1] + CHy2[-1,:,:]*(self.diffzEx[sli1]-self.diffxEz[sli1])
                self.Hz[sli2] = CHz1[-1,:,:]*self.Hz[sli2] + CHz2[-1,:,:]*(self.diffxEy[sli2]-self.diffyEx[sli2])

        #----------------------------------------------------------------------#
        #---------------- Apply BBC when the method is the SHPF ---------------#
        #----------------------------------------------------------------------#

        if self.method == 'SHPF' and self.BBC_called == True: self._updateH_BBC_SHPF()
        if self.method == 'PSTD' and self.BBC_called == True: self._updateH_BBC_PSTD()

        #-----------------------------------------------------------#
        #---------------- Apply PML when it is given ---------------#
        #-----------------------------------------------------------#

        self._updateH_PML()

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

        if self.MPIrank != (self.MPIsize-1):

            if self.engine == 'cupy':
                sendHylast = cp.asnumpy(self.Hy[-1,:,:])
                sendHzlast = cp.asnumpy(self.Hz[-1,:,:])

            else: # engine is numpy
                sendHylast = self.Hy[-1,:,:].copy()
                sendHzlast = self.Hz[-1,:,:].copy()

            self.MPIcomm.send(sendHylast, dest=(self.MPIrank+1), tag=(tstep*100+3))
            self.MPIcomm.send(sendHzlast, dest=(self.MPIrank+1), tag=(tstep*100+5))
        
        #---------------------------------------------------------#
        #--------- MPI recv Hy and Hz from previous rank ---------#
        #---------------------------------------------------------#

        if self.MPIrank != 0:

            self.recvHyfirst = self.MPIcomm.recv( source=(self.MPIrank-1), tag=(tstep*100+3))
            self.recvHzfirst = self.MPIcomm.recv( source=(self.MPIrank-1), tag=(tstep*100+5))
        
            if self.engine == 'cupy':
                self.recvHyfirst = cp.asarray(self.recvHyfirst)
                self.recvHzfirst = cp.asarray(self.recvHzfirst)

        #----------------------------------------------------------------------#
        #---------------- Apply BBC when the method is the FDTD ---------------#
        #----------------------------------------------------------------------#

        if self.method == 'FDTD': self._updateE_BBC_FDTD()
        #if self.method == 'PSTD' and self.BBC == True: self._updateE_BBC_FDTD()

        #-----------------------------------------------------------#
        #---------------------- Get derivatives --------------------#
        #-----------------------------------------------------------#

        if self.method == 'PSTD':

            # Get derivatives of Hy and Hz to update Ex
            self.diffyHz = self.ifft(self.iky*self.fft(self.Hz, axes=(1,)), axes=(1,))
            self.diffzHy = self.ifft(self.ikz*self.fft(self.Hy, axes=(2,)), axes=(2,))

            # Get derivatives of Hx and Hz to update Ey
            self.diffzHx = self.ifft(self.ikz*self.fft(self.Hx, axes=(2,)), axes=(2,))
            self.diffxHz = self.ifft(self.ikx*self.fft(self.Hz, axes=(0,)), axes=(0,))

            # Get derivatives of Hx and Hy to update Ez
            self.diffyHx = self.ifft(self.iky*self.fft(self.Hx, axes=(1,)), axes=(1,))
            self.diffxHy = self.ifft(self.ikx*self.fft(self.Hy, axes=(0,)), axes=(0,))

        if self.method == 'SHPF':

            # Get derivatives of Hy and Hz to update Ex
            self.diffyHz = self.ifft(self.iky*self.ymshift*self.fft(self.Hz, axes=(1,)), axes=(1,))
            self.diffzHy = self.ifft(self.ikz*self.zmshift*self.fft(self.Hy, axes=(2,)), axes=(2,))

            # Get derivatives of Hx and Hz to update Ey
            self.diffzHx = self.ifft(self.ikz*self.zmshift*self.fft(self.Hx, axes=(2,)), axes=(2,))
            self.diffxHz[1:,:,:] = (self.Hz[1:,:,:] - self.Hz[:-1,:,:]) / self.dx

            # Get derivatives of Hx and Hy to update Ez
            self.diffyHx = self.ifft(self.iky*self.ymshift*self.fft(self.Hx, axes=(1,)), axes=(1,))
            self.diffxHy[1:,:,:] = (self.Hy[1:,:,:] - self.Hy[:-1,:,:]) / self.dx

            if self.MPIrank != 0:

                # Get derivatives of Hx and Hz to update Ey at x=0.
                self.diffxHz[0,:,:] = (self.Hz[0,:,:]-self.recvHzfirst[:,:]) / self.dx

                # Get derivatives of Hx and Hy to update Ez at x=0.
                self.diffxHy[0,:,:] = (self.Hy[0,:,:]-self.recvHyfirst[:,:]) / self.dx

        if self.method == 'FDTD':

            # Get derivatives of Hy and Hz to update Ex
            self.diffyHz[:,1:,1:] = (self.Hz[:,1:,1:] - self.Hz[:,:-1,1:]) / self.dy
            self.diffzHy[:,1:,1:] = (self.Hy[:,1:,1:] - self.Hy[:,1:,:-1]) / self.dz

            # Get derivatives of Hx and Hz to update Ey
            self.diffzHx[1:,:,1:] = (self.Hx[1:,:,1:] - self.Hx[1:,:,:-1]) / self.dz
            self.diffxHz[1:,:,1:] = (self.Hz[1:,:,1:] - self.Hz[:-1,:,1:]) / self.dx

            # Get derivatives of Hx and Hy to update Ez
            self.diffyHx[1:,1:,:] = (self.Hx[1:,1:,:] - self.Hx[1:,:-1,:]) / self.dy
            self.diffxHy[1:,1:,:] = (self.Hy[1:,1:,:] - self.Hy[:-1,1:,:]) / self.dx

            if self.MPIrank != 0:

                # Get derivatives of Hx and Hz to update Ey at x=0.
                self.diffxHz[0,:,1:] = (self.Hz[0,:,1:] - self.recvHzfirst[:,1:]) / self.dx
                self.diffzHx[0,:,1:] = (self.Hx[0,:,1:] - self.      Hx[0,:,:-1]) / self.dz

                # Get derivatives of Hx and Hy to update Ez at x=0.
                self.diffxHy[0,1:,:] = (self.Hy[0,1:,:] - self.recvHyfirst[1:,:]) / self.dx
                self.diffyHx[0,1:,:] = (self.Hx[0,1:,:] - self.      Hx[0,:-1,:]) / self.dy

        #-----------------------------------------------------------#
        #--------------- Cast basic update equations ---------------#
        #-----------------------------------------------------------#

        # Update Ex, Ey, Ez
        if self.method == 'PSTD':

            CEx1 = (2.*self.eps_Ex-self.econ_Ex*self.dt) / \
                   (2.*self.eps_Ex+self.econ_Ex*self.dt)
            CEx2 = (2.*self.dt) / (2.*self.eps_Ex+self.econ_Ex*self.dt)

            CEy1 = (2.*self.eps_Ey-self.econ_Ey*self.dt) / \
                   (2.*self.eps_Ey+self.econ_Ey*self.dt)
            CEy2 = (2.*self.dt) / (2.*self.eps_Ey+self.econ_Ey*self.dt)

            CEz1 = (2.*self.eps_Ez-self.econ_Ez*self.dt) / \
                   (2.*self.eps_Ez+self.econ_Ez*self.dt)
            CEz2 = (2.*self.dt) / (2.*self.eps_Ez+self.econ_Ez*self.dt)

            # PEC condition.
            CEx1[self.eps_Ex > 1e3] = 0.
            CEx2[self.eps_Ex > 1e3] = 0.
            CEy1[self.eps_Ey > 1e3] = 0.
            CEy2[self.eps_Ey > 1e3] = 0.
            CEz1[self.eps_Ez > 1e3] = 0.
            CEz2[self.eps_Ez > 1e3] = 0.

            self.Ex = CEx1 * self.Ex + CEx2 * (self.diffyHz - self.diffzHy)
            self.Ey = CEy1 * self.Ey + CEy2 * (self.diffzHx - self.diffxHz)
            self.Ez = CEz1 * self.Ez + CEz2 * (self.diffxHy - self.diffyHx)

        if self.method == 'SHPF':

            CEx1 = (2.*self.eps_Ex[:,:,:]-self.econ_Ex[:,:,:]*self.dt) / \
                   (2.*self.eps_Ex[:,:,:]+self.econ_Ex[:,:,:]*self.dt)
            CEx2 = (2.*self.dt) / (2.*self.eps_Ex[:,:,:]+self.econ_Ex[:,:,:]*self.dt)

            CEy1 = (2.*self.eps_Ey[1:,:,:]-self.econ_Ey[1:,:,:]*self.dt) / \
                   (2.*self.eps_Ey[1:,:,:]+self.econ_Ey[1:,:,:]*self.dt)
            CEy2 = (2.*self.dt) / (2.*self.eps_Ey[1:,:,:]+self.econ_Ey[1:,:,:]*self.dt)

            CEz1 = (2.*self.eps_Ez[1:,:,:]-self.econ_Ez[1:,:,:]*self.dt) / \
                   (2.*self.eps_Ez[1:,:,:]+self.econ_Ez[1:,:,:]*self.dt)
            CEz2 = (2.*self.dt) / (2.*self.eps_Ez[1:,:,:]+self.econ_Ez[1:,:,:]*self.dt)

            # PEC condition.
            CEx1[self.eps_Ex[ :,:,:] > 1e3] = 0.
            CEx2[self.eps_Ex[ :,:,:] > 1e3] = 0.
            CEy1[self.eps_Ey[1:,:,:] > 1e3] = 0.
            CEy2[self.eps_Ey[1:,:,:] > 1e3] = 0.
            CEz1[self.eps_Ez[1:,:,:] > 1e3] = 0.
            CEz2[self.eps_Ez[1:,:,:] > 1e3] = 0.

            self.Ex[: ,:,:] = CEx1 * self.Ex[ :,:,:] + CEx2 * (self.diffyHz[ :,:,:] - self.diffzHy[ :,:,:])
            self.Ey[1:,:,:] = CEy1 * self.Ey[1:,:,:] + CEy2 * (self.diffzHx[1:,:,:] - self.diffxHz[1:,:,:])
            self.Ez[1:,:,:] = CEz1 * self.Ez[1:,:,:] + CEz2 * (self.diffxHy[1:,:,:] - self.diffyHx[1:,:,:])

            if self.MPIrank != 0:
        
                # Update Ey and Ez at x=0.
                self.Ey[0,:,:] = CEy1[0,:,:] * self.Ey[0,:,:] + CEy2[0,:,:] * (self.diffzHx[0,:,:]-self.diffxHz[0,:,:])
                self.Ez[0,:,:] = CEz1[0,:,:] * self.Ez[0,:,:] + CEz2[0,:,:] * (self.diffxHy[0,:,:]-self.diffyHx[0,:,:])

        elif self.method == 'FDTD':

            CEx1 = (2.*self.eps_Ex[:,1:,1:]-self.econ_Ex[:,1:,1:]*self.dt) / \
                   (2.*self.eps_Ex[:,1:,1:]+self.econ_Ex[:,1:,1:]*self.dt)
            CEx2 = (2.*self.dt) / (2.*self.eps_Ex[:,1:,1:]+self.econ_Ex[:,1:,1:]*self.dt)

            CEy1 = (2.*self.eps_Ey[1:,:,1:]-self.econ_Ey[1:,:,1:]*self.dt) / \
                   (2.*self.eps_Ey[1:,:,1:]+self.econ_Ey[1:,:,1:]*self.dt)
            CEy2 = (2.*self.dt) / (2.*self.eps_Ey[1:,:,1:]+self.econ_Ey[1:,:,1:]*self.dt)

            CEz1 = (2.*self.eps_Ez[1:,1:,:]-self.econ_Ez[1:,1:,:]*self.dt) / \
                   (2.*self.eps_Ez[1:,1:,:]+self.econ_Ez[1:,1:,:]*self.dt)
            CEz2 = (2.*self.dt) / (2.*self.eps_Ez[1:,1:,:]+self.econ_Ez[1:,1:,:]*self.dt)

            # PEC condition.
            CEx1[self.eps_Ex[:,1:,1:] > 1e3] = 0.
            CEx2[self.eps_Ex[:,1:,1:] > 1e3] = 0.
            CEy1[self.eps_Ey[1:,:,1:] > 1e3] = 0.
            CEy2[self.eps_Ey[1:,:,1:] > 1e3] = 0.
            CEz1[self.eps_Ez[1:,1:,:] > 1e3] = 0.
            CEz2[self.eps_Ez[1:,1:,:] > 1e3] = 0.

            self.Ex[:,1:,1:] = CEx1 * self.Ex[:,1:,1:] + CEx2 * (self.diffyHz[:,1:,1:] - self.diffzHy[:,1:,1:])
            self.Ey[1:,:,1:] = CEy1 * self.Ey[1:,:,1:] + CEy2 * (self.diffzHx[1:,:,1:] - self.diffxHz[1:,:,1:])
            self.Ez[1:,1:,:] = CEz1 * self.Ez[1:,1:,:] + CEz2 * (self.diffxHy[1:,1:,:] - self.diffyHx[1:,1:,:])

            if self.MPIrank != 0:
        
                # Update Ey and Ez at x=0.
                self.Ey[0,:,1:] = CEy1[0,:,:] * self.Ey[0,:,1:] + CEy2[0,:,:] * (self.diffzHx[0,:,1:]-self.diffxHz[0,:,1:])
                self.Ez[0,1:,:] = CEz1[0,:,:] * self.Ez[0,1:,:] + CEz2[0,:,:] * (self.diffxHy[0,1:,:]-self.diffyHx[0,1:,:])

        #----------------------------------------------------------------------#
        #---------------- Apply BBC when the method is the SHPF ---------------#
        #----------------------------------------------------------------------#

        if self.method == 'SHPF' and self.BBC_called == True: self._updateE_BBC_SHPF()
        if self.method == 'PSTD' and self.BBC_called == True: self._updateE_BBC_PSTD()

        #-----------------------------------------------------------#
        #---------------- Apply PML when it is given ---------------#
        #-----------------------------------------------------------#

        self._updateE_PML()

    def _updateH_PML(self):

        # For all ranks.
        if 'y' in self.PMLregion.keys():
            if '+' in self.PMLregion.get('y'): self._PML_updateH_py()
            if '-' in self.PMLregion.get('y'): self._PML_updateH_my()

        if 'z' in self.PMLregion.keys():
            if '+' in self.PMLregion.get('z'): self._PML_updateH_pz()
            if '-' in self.PMLregion.get('z'): self._PML_updateH_mz()

        # First rank
        if self.MPIrank == 0:
            if 'x' in self.PMLregion.keys():
                if '+' in self.PMLregion.get('x') and self.MPIsize == 1: self._PML_updateH_px()
                if '-' in self.PMLregion.get('x'): self._PML_updateH_mx()

        # Middle rank
        elif self.MPIrank > 0 and self.MPIrank < (self.MPIsize-1):
            if 'x' in self.PMLregion.keys():
                if '+' in self.PMLregion.get('x'): pass
                if '-' in self.PMLregion.get('x'): pass
        # Last rank
        elif self.MPIrank == (self.MPIsize-1) and self.MPIsize != 1:
            if 'x' in self.PMLregion.keys():
                if '+' in self.PMLregion.get('x'): self._PML_updateH_px()
                if '-' in self.PMLregion.get('x'): pass

    def _updateE_PML(self):

        # For all ranks.
        if 'y' in self.PMLregion.keys():
            if '+' in self.PMLregion.get('y'): self._PML_updateE_py()
            if '-' in self.PMLregion.get('y'): self._PML_updateE_my()
        if 'z' in self.PMLregion.keys():
            if '+' in self.PMLregion.get('z'): self._PML_updateE_pz()
            if '-' in self.PMLregion.get('z'): self._PML_updateE_mz()

        # First rank
        if self.MPIrank == 0:
            if 'x' in self.PMLregion.keys():
                if '+' in self.PMLregion.get('x') and self.MPIsize == 1: self._PML_updateE_px()
                if '-' in self.PMLregion.get('x'): self._PML_updateE_mx()

        # Middle rank
        elif self.MPIrank > 0 and self.MPIrank < (self.MPIsize-1):
            if 'x' in self.PMLregion.keys():
                if '+' in self.PMLregion.get('x'): pass
                if '-' in self.PMLregion.get('x'): pass

        # Last rank
        elif self.MPIrank == (self.MPIsize-1) and self.MPIsize != 1:
            if 'x' in self.PMLregion.keys():
                if '+' in self.PMLregion.get('x'): self._PML_updateE_px()
                if '-' in self.PMLregion.get('x'): pass

    def _PML_updateH_px(self):

        if self.method == 'PSTD':

            odd = [slice(0,None,2), None, None]

            psiidx_Hyxp = [slice(0,None), slice(0,None), slice(0,None)]
            myidx_Hyxp = [slice(-self.npml,None), slice(0,None), slice(0,None)]

            psiidx_Hzxp = [slice(0,None), slice(0,None), slice(0,None)]
            myidx_Hzxp = [slice(-self.npml,None), slice(0,None), slice(0,None)]

        if self.method == 'SHPF':

            odd = [slice(1,-1,2), None, None]

            psiidx_Hyxp = [slice(0,-1), slice(0,None), slice(0,None)]
            myidx_Hyxp = [slice(-self.npml,-1), slice(0,None), slice(0,None)]

            psiidx_Hzxp = [slice(0,-1), slice(0,None), slice(0,None)]
            myidx_Hzxp = [slice(-self.npml,-1), slice(0,None), slice(0,None)]

        if self.method == 'FDTD':

            odd = [slice(1,-1,2), None, None]

            psiidx_Hyxp = [slice(0,-1), slice(0,None), slice(0,-1)]
            myidx_Hyxp = [slice(-self.npml,-1), slice(0,None), slice(0,-1)]

            psiidx_Hzxp = [slice(0,-1), slice(0,-1), slice(0,None)]
            myidx_Hzxp = [slice(-self.npml,-1), slice(0,-1), slice(0,None)]

        # Update Hy at x+.
        CHy2 = (-2*self.dt) / (2.*self.mu_Hy[myidx_Hyxp] + self.mcon_Hy[myidx_Hyxp]*self.dt)
        self.psi_hyx_p[psiidx_Hyxp] = (self.PMLbx[odd]*self.psi_hyx_p[psiidx_Hyxp]) \
                                    + (self.PMLax[odd]*self.diffxEz[myidx_Hyxp])
        self.Hy[myidx_Hyxp] += CHy2*(-((1./self.PMLkappax[odd] - 1.)*self.diffxEz[myidx_Hyxp]) - self.psi_hyx_p[psiidx_Hyxp])

        # Update Hz at x+.
        CHz2 = (-2*self.dt) / (2.*self.mu_Hz[myidx_Hzxp] + self.mcon_Hz[myidx_Hzxp]*self.dt)
        self.psi_hzx_p[psiidx_Hzxp] = (self.PMLbx[odd]*self.psi_hzx_p[psiidx_Hzxp]) \
                                    + (self.PMLax[odd]*self.diffxEy[myidx_Hzxp])
        self.Hz[myidx_Hzxp] += CHz2*(+((1./self.PMLkappax[odd]-1.)*self.diffxEy[myidx_Hzxp]) + self.psi_hzx_p[psiidx_Hzxp])

    def _PML_updateE_px(self):

        if self.method == 'SHPF' or self.method == 'PSTD':

            even = [slice(0,None,2), None, None]

            psiidx_Eyxp = [slice(0,None), slice(0,None), slice(0,None)]
            myidx_Eyxp = [slice(-self.npml,None), slice(0,None), slice(0,None)]

            psiidx_Ezxp = [slice(0,None), slice(0,None), slice(0,None)]
            myidx_Ezxp = [slice(-self.npml,None), slice(0,None), slice(0,None)]

        if self.method == 'FDTD':

            even = [slice(0,None,2), None, None]

            psiidx_Eyxp = [slice(0,None), slice(0,None), slice(1,None)]
            myidx_Eyxp = [slice(-self.npml,None), slice(0,None), slice(1,None)]

            psiidx_Ezxp = [slice(0,None), slice(1,None), slice(0,None)]
            myidx_Ezxp = [slice(-self.npml,None), slice(1,None), slice(0,None)]

        # Update Ey at x+.
        CEy2 = (2.*self.dt) / (2.*self.eps_Ey[myidx_Eyxp] + self.econ_Ey[myidx_Eyxp]*self.dt)
        self.psi_eyx_p[psiidx_Eyxp] = (self.PMLbx[even]*self.psi_eyx_p[psiidx_Eyxp])\
                                    + (self.PMLax[even]*self.diffxHz[myidx_Eyxp])
        self.Ey[myidx_Eyxp] += CEy2*(-(1./self.PMLkappax[even]-1.)*self.diffxHz[myidx_Eyxp] - self.psi_eyx_p[psiidx_Eyxp])

        # Update Ez at x+.
        CEz2 = (2.*self.dt) / (2.*self.eps_Ez[myidx_Ezxp] + self.econ_Ez[myidx_Ezxp]*self.dt)
        self.psi_ezx_p[psiidx_Ezxp] = (self.PMLbx[even]*self.psi_ezx_p[psiidx_Ezxp])\
                                    + (self.PMLax[even]*self.diffxHy[myidx_Ezxp])
        self.Ez[myidx_Ezxp] += CEz2*(+(1./self.PMLkappax[even]-1.)*self.diffxHy[myidx_Ezxp] + self.psi_ezx_p[psiidx_Ezxp])

    def _PML_updateH_mx(self):

        if self.method == 'PSTD':

            even = [slice(-1,None,-2), None, None]

            psiidx_Hyxm = [slice(0,self.npml), slice(0,None), slice(0,None)]
            myidx_Hyxm  = [slice(0,self.npml), slice(0,None), slice(0,None)]

            psiidx_Hzxm = [slice(0,self.npml), slice(0,None), slice(0,None)]
            myidx_Hzxm  = [slice(0,self.npml), slice(0,None), slice(0,None)]

        if self.method == 'SHPF':

            even = [slice(-2,None,-2), None, None]

            psiidx_Hyxm = [slice(0,self.npml), slice(0,None), slice(0,None)]
            myidx_Hyxm  = [slice(0,self.npml), slice(0,None), slice(0,None)]

            psiidx_Hzxm = [slice(0,self.npml), slice(0,None), slice(0,None)]
            myidx_Hzxm  = [slice(0,self.npml), slice(0,None), slice(0,None)]

        if self.method == 'FDTD':

            even = [slice(-2,None,-2), None, None]

            psiidx_Hyxm = [slice(0,self.npml), slice(0,None), slice(0,-1)]
            myidx_Hyxm  = [slice(0,self.npml), slice(0,None), slice(0,-1)]

            psiidx_Hzxm = [slice(0, self.npml), slice(0,-1), slice(0,None)]
            myidx_Hzxm  = [slice(0, self.npml), slice(0,-1), slice(0,None)]

        # Update Hy at x-.
        CHy2 = (-2*self.dt) / (2.*self.mu_Hy[myidx_Hyxm] + self.mcon_Hy[myidx_Hyxm]*self.dt)
        self.psi_hyx_m[psiidx_Hyxm] = (self.PMLbx[even]*self.psi_hyx_m[psiidx_Hyxm])\
                                    + (self.PMLax[even]*self.diffxEz[myidx_Hyxm])
        self.Hy[myidx_Hyxm] += CHy2*(-((1./self.PMLkappax[even]-1.)*self.diffxEz[myidx_Hyxm]) - self.psi_hyx_m[psiidx_Hyxm])

        # Update Hz at x-.
        CHz2 = (-2*self.dt) / (2.*self.mu_Hz[myidx_Hzxm] + self.mcon_Hz[myidx_Hzxm]*self.dt)
        self.psi_hzx_m[psiidx_Hzxm] = (self.PMLbx[even]*self.psi_hzx_m[psiidx_Hzxm])\
                                    + (self.PMLax[even]*self.diffxEy[myidx_Hzxm])
        self.Hz[myidx_Hzxm] += CHz2*(+((1./self.PMLkappax[even]-1.)*self.diffxEy[myidx_Hzxm]) + self.psi_hzx_m[psiidx_Hzxm])

    def _PML_updateE_mx(self):

        if self.method == 'PSTD':

            odd = [slice(-1,None,-2),None,None]

            psiidx_Eymx = [slice(0,self.npml), slice(0,None), slice(0,None)]
            myidx_Eymx  = [slice(0,self.npml), slice(0,None), slice(0,None)]

            psiidx_Ezmx = [slice(0,self.npml), slice(0,None), slice(0,None)]
            myidx_Ezmx  = [slice(0,self.npml), slice(0,None), slice(0,None)]

        if self.method == 'SHPF':

            odd = [slice(-3,None,-2),None,None]

            psiidx_Eymx = [slice(1,self.npml), slice(0,None), slice(0,None)]
            myidx_Eymx  = [slice(1,self.npml), slice(0,None), slice(0,None)]

            psiidx_Ezmx = [slice(1,self.npml), slice(0,None), slice(0,None)]
            myidx_Ezmx  = [slice(1,self.npml), slice(0,None), slice(0,None)]

        if self.method == 'FDTD':

            odd = [slice(-3,None,-2),None,None]

            psiidx_Eymx = [slice(1,self.npml), slice(0,None), slice(1,None)]
            myidx_Eymx  = [slice(1,self.npml), slice(0,None), slice(1,None)]

            psiidx_Ezmx = [slice(1,self.npml), slice(1,None), slice(0,None)]
            myidx_Ezmx  = [slice(1,self.npml), slice(1,None), slice(0,None)]

        # Update Ey at x+.
        CEy2 = (2.*self.dt) / (2.*self.eps_Ey[myidx_Eymx] + self.econ_Ey[myidx_Eymx]*self.dt)
        self.psi_eyx_m[psiidx_Eymx] = (self.PMLbx[odd]*self.psi_eyx_m[psiidx_Eymx])\
                                    + (self.PMLax[odd]*self.diffxHz[myidx_Eymx])
        self.Ey[myidx_Eymx] += CEy2*(-(1./self.PMLkappax[odd]-1.)*self.diffxHz[myidx_Eymx] - self.psi_eyx_m[psiidx_Eymx])

        # Update Ez at x+.
        CEz2 = (2.*self.dt) / (2.*self.eps_Ez[myidx_Ezmx] + self.econ_Ez[myidx_Ezmx]*self.dt)
        self.psi_ezx_m[psiidx_Ezmx] = (self.PMLbx[odd]*self.psi_ezx_m[psiidx_Ezmx])\
                                    + (self.PMLax[odd] * self.diffxHy[myidx_Ezmx])
        self.Ez[myidx_Ezmx] += CEz2*(+(1./self.PMLkappax[odd]-1.)*self.diffxHy[myidx_Ezmx] + self.psi_ezx_m[psiidx_Ezmx])

    def _PML_updateH_py(self):

        if self.method == 'PSTD':

            odd = [None, slice(0,None,2), None]

            psiidx_Hxyp = [slice(0,None), slice(0,None), slice(0,None)]
            myidx_Hxyp  = [slice(0,None), slice(-self.npml,None), slice(0,None)]

            psiidx_Hzyp = [slice(0,None), slice(0,None), slice(0,None)]
            myidx_Hzyp  = [slice(0,None), slice(-self.npml,None), slice(0,None)]

        if self.method == 'SHPF':

            odd = [None, slice(1,None,2), None]

            psiidx_Hxyp = [slice(0,None), slice(0,None), slice(0,None)]
            myidx_Hxyp  = [slice(0,None), slice(-self.npml,None), slice(0,None)]

            if self.MPIrank < (self.MPIsize-1):
                psiidx_Hzyp = [slice(0,None), slice(0,None), slice(0,None)]
                myidx_Hzyp  = [slice(0,None), slice(-self.npml,None), slice(0,None)]
            else:
                psiidx_Hzyp = [slice(0,-1), slice(0,None), slice(0,None)]
                myidx_Hzyp  = [slice(0,-1), slice(-self.npml,None), slice(0,None)]

        if self.method == 'FDTD':

            odd = [None, slice(1,-1,2), None]

            psiidx_Hxyp = [slice(0,None), slice(0,-1), slice(0,-1)]
            myidx_Hxyp  = [slice(0,None), slice(-self.npml,-1), slice(0,-1)]

            if self.MPIrank < (self.MPIsize-1):
                psiidx_Hzyp = [slice(0,None), slice(0,-1), slice(0,None)]
                myidx_Hzyp  = [slice(0,None), slice(-self.npml,-1), slice(0,None)]
            else:
                psiidx_Hzyp = [slice(0,-1), slice(0,-1), slice(0,None)]
                myidx_Hzyp  = [slice(0,-1), slice(-self.npml,-1), slice(0,None)]

        # Update Hx at y+.
        CHx2 = (-2.*self.dt) / (2.*self.mu_Hx[myidx_Hxyp] + self.mcon_Hx[myidx_Hxyp]*self.dt)
        self.psi_hxy_p[psiidx_Hxyp] = (self.PMLby[odd]*self.psi_hxy_p[psiidx_Hxyp])\
                                    + (self.PMLay[odd]*self.diffyEz[myidx_Hxyp])
        self.Hx[myidx_Hxyp] += CHx2*(+((1./self.PMLkappay[odd] - 1.)*self.diffyEz[myidx_Hxyp])+self.psi_hxy_p[psiidx_Hxyp])

        # Update Hz at y+.
        CHz2 = (-2.*self.dt) / (2.*self.mu_Hz[myidx_Hzyp] + self.mcon_Hz[myidx_Hzyp]*self.dt)
        self.psi_hzy_p[psiidx_Hzyp] = (self.PMLby[odd] * self.psi_hzy_p[psiidx_Hzyp])\
                                    + (self.PMLay[odd] * self.diffyEx[myidx_Hzyp])
        self.Hz[myidx_Hzyp] += CHz2*(-((1./self.PMLkappay[odd]-1.)*self.diffyEx[myidx_Hzyp]) - self.psi_hzy_p[psiidx_Hzyp])
            
    def _PML_updateE_py(self):

        if self.method == 'PSTD':

            even = [None,slice(0,None,2),None]

            psiidx_Exyp = [slice(0,None), slice(0,self.npml), slice(0,None)]
            myidx_Exyp  = [slice(0,None), slice(-self.npml,None), slice(0,None)]

            psiidx_Ezyp = [slice(0,None), slice(0,self.npml), slice(0,None)]
            myidx_Ezyp  = [slice(0,None), slice(-self.npml,None), slice(0,None)]

        if self.method == 'SHPF':

            even = [None,slice(0,None,2),None]

            psiidx_Exyp = [slice(0,None), slice(0,self.npml), slice(0,None)]
            myidx_Exyp  = [slice(0,None), slice(-self.npml,None), slice(0,None)]

            if self.MPIrank > 0:
                psiidx_Ezyp = [slice(0,None), slice(0,self.npml), slice(0,None)]
                myidx_Ezyp  = [slice(0,None), slice(-self.npml,None), slice(0,None)]
            else:
                psiidx_Ezyp = [slice(1,None), slice(0,self.npml), slice(0,None)]
                myidx_Ezyp  = [slice(1,None), slice(-self.npml,None), slice(0,None)]

        if self.method == 'FDTD':
         
            even = [None,slice(0,None,2),None]

            psiidx_Exyp = [slice(0,None), slice(0,self.npml), slice(1,None)]
            myidx_Exyp  = [slice(0,None), slice(-self.npml,None), slice(1,None)]

            if self.MPIrank > 0:
                psiidx_Ezyp = [slice(0,None), slice(0,self.npml), slice(0,None)]
                myidx_Ezyp  = [slice(0,None), slice(-self.npml,None), slice(0,None)]
            else:
                psiidx_Ezyp = [slice(1,None), slice(0,self.npml), slice(0,None)]
                myidx_Ezyp  = [slice(1,None), slice(-self.npml,None), slice(0,None)]

        # Update Ex at y+.
        CEx2 = (2*self.dt) / (2.*self.eps_Ex[myidx_Exyp] + self.econ_Ex[myidx_Exyp]*self.dt)
        self.psi_exy_p[psiidx_Exyp] = (self.PMLby[even]*self.psi_exy_p[psiidx_Exyp])\
                                    + (self.PMLay[even]*self.diffyHz[myidx_Exyp])
        self.Ex[myidx_Exyp] += CEx2*(+((1./self.PMLkappay[even]-1.)*self.diffyHz[myidx_Exyp]) + self.psi_exy_p[psiidx_Exyp])

        # Update Ez at y+.
        CEz2 = (2.*self.dt) / (2.*self.eps_Ez[myidx_Ezyp] + self.econ_Ez[myidx_Ezyp]*self.dt)
        self.psi_ezy_p[psiidx_Ezyp] = (self.PMLby[even]*self.psi_ezy_p[psiidx_Ezyp])\
                                    + (self.PMLay[even]*self.diffyHx[myidx_Ezyp])
        self.Ez[myidx_Ezyp] += CEz2*(-((1./self.PMLkappay[even]-1.)*self.diffyHx[myidx_Ezyp]) - self.psi_ezy_p[psiidx_Ezyp])

    def _PML_updateH_my(self):

        if self.method == 'PSTD':

            even = [None, slice(-1,None,-2), None]

            psiidx_Hxym = [slice(0,None), slice(0,self.npml), slice(0,None)]
            myidx_Hxym  = [slice(0,None), slice(0,self.npml), slice(0,None)]

            psiidx_Hzym = [slice(0,None), slice(0,self.npml), slice(0,None)]
            myidx_Hzym  = [slice(0,None), slice(0,self.npml), slice(0,None)]

        if self.method == 'SHPF':

            even = [None, slice(-2,None,-2), None]

            psiidx_Hxym = [slice(0,None), slice(0,self.npml), slice(0,None)]
            myidx_Hxym  = [slice(0,None), slice(0,self.npml), slice(0,None)]

            if self.MPIrank < (self.MPIsize-1):
                psiidx_Hzym = [slice(0,None), slice(0,self.npml), slice(0,None)]
                myidx_Hzym  = [slice(0,None), slice(0,self.npml), slice(0,None)]
            else:
                psiidx_Hzym = [slice(0,-1), slice(0,self.npml), slice(0,None)]
                myidx_Hzym  = [slice(0,-1), slice(0,self.npml), slice(0,None)]

        if self.method == 'FDTD':

            even = [None, slice(-2,None,-2), None]

            psiidx_Hxym = [slice(0,None), slice(0,self.npml), slice(0,-1)]
            myidx_Hxym  = [slice(0,None), slice(0,self.npml), slice(0,-1)]

            if self.MPIrank < (self.MPIsize-1):
                psiidx_Hzym = [slice(0,None), slice(0,self.npml), slice(0,None)]
                myidx_Hzym  = [slice(0,None), slice(0,self.npml), slice(0,None)]
            else:
                psiidx_Hzym = [slice(0,-1), slice(0,self.npml), slice(0,None)]
                myidx_Hzym  = [slice(0,-1), slice(0,self.npml), slice(0,None)]

            even = [None, slice(-2,None,-2), None]

        # Update Hx at y-.
        CHx2 =  (-2*self.dt) / (2.*self.mu_Hx[myidx_Hxym] + self.mcon_Hx[myidx_Hxym]*self.dt)
        self.psi_hxy_m[psiidx_Hxym] = (self.PMLby[even]*self.psi_hxy_m[psiidx_Hxym])\
                                    + (self.PMLay[even]*self.diffyEz[myidx_Hxym])
        self.Hx[myidx_Hxym] += CHx2*(+((1./self.PMLkappay[even]-1.)*self.diffyEz[myidx_Hxym]) + self.psi_hxy_m[psiidx_Hxym])

        # Update Hz at y-.
        CHz2 =  (-2*self.dt) / (2.*self.mu_Hz[myidx_Hzym] + self.mcon_Hz[myidx_Hzym]*self.dt)
        self.psi_hzy_m[psiidx_Hzym] = (self.PMLby[even]*self.psi_hzy_m[psiidx_Hzym])\
                                    + (self.PMLay[even]*self.diffyEx[myidx_Hzym])
        self.Hz[myidx_Hzym] += CHz2*(-((1./self.PMLkappay[even]-1.)*self.diffyEx[myidx_Hzym]) - self.psi_hzy_m[psiidx_Hzym])

    def _PML_updateE_my(self):

        if self.method == 'PSTD':

            odd = [None, slice(-1,None,-2), None]

            psiidx_Exym = [slice(0,None), slice(0,self.npml), slice(0,None)]
            myidx_Exym = [slice(0,None), slice(0,self.npml), slice(0,None)]

            psiidx_Ezym = [slice(0,None), slice(0,self.npml), slice(0,None)]
            myidx_Ezym = [slice(0,None), slice(0,self.npml), slice(0,None)]

        if self.method == 'SHPF':

            odd = [None, slice(-1,None,-2), None]

            psiidx_Exym = [slice(0,None), slice(0,self.npml), slice(0,None)]
            myidx_Exym = [slice(0,None), slice(0,self.npml), slice(0,None)]

            if self.MPIrank > 0:
                psiidx_Ezym = [slice(0,None), slice(0,self.npml), slice(0,None)]
                myidx_Ezym = [slice(0,None), slice(0,self.npml), slice(0,None)]
            else:
                psiidx_Ezym = [slice(1,None), slice(0,self.npml), slice(0,None)]
                myidx_Ezym = [slice(1,None), slice(0,self.npml), slice(0,None)]

        elif self.method == 'FDTD':

            odd = [None, slice(-3,None,-2), None]

            psiidx_Exym = [slice(0,None), slice(1,self.npml), slice(1,None)]
            myidx_Exym = [slice(0,None), slice(1,self.npml), slice(1,None)]

            if self.MPIrank > 0:
                psiidx_Ezym = [slice(0,None), slice(1,self.npml), slice(0,None)]
                myidx_Ezym = [slice(0,None), slice(1,self.npml), slice(0,None)]
            else:
                psiidx_Ezym = [slice(1,None), slice(1,self.npml), slice(0,None)]
                myidx_Ezym = [slice(1,None), slice(1,self.npml), slice(0,None)]

        # Update Ex at y-.
        CEx2 = (2.*self.dt) / (2.*self.eps_Ex[myidx_Exym] + self.econ_Ex[myidx_Exym]*self.dt)
        self.psi_exy_m[psiidx_Exym] = (self.PMLby[odd]*self.psi_exy_m[psiidx_Exym])\
                                    + (self.PMLay[odd]*self.diffyHz[myidx_Exym])
        self.Ex[myidx_Exym] += CEx2*(+((1./self.PMLkappay[odd]-1.)*self.diffyHz[myidx_Exym]) + self.psi_exy_m[psiidx_Exym])

        # Update Ez at y-.
        CEz2 = (2*self.dt) / (2.*self.eps_Ez[myidx_Ezym] + self.econ_Ez[myidx_Ezym]*self.dt)
        self.psi_ezy_m[psiidx_Ezym] = (self.PMLby[odd]*self.psi_ezy_m[psiidx_Ezym])\
                                    + (self.PMLay[odd]*self.diffyHx[myidx_Ezym])
        self.Ez[myidx_Ezym] += CEz2*(-((1./self.PMLkappay[odd]-1.)*self.diffyHx[myidx_Ezym]) - self.psi_ezy_m[psiidx_Ezym])

    def _PML_updateH_pz(self):

        if self.method == 'PSTD':

            odd = [None, None, slice(0,None,2)]

            psiidx_Hxzp = [slice(0,None), slice(0,None), slice(0,self.npml)]
            myidx_Hxzp  = [slice(0,None), slice(0,None), slice(-self.npml,None)]

            psiidx_Hyzp = [slice(0,None), slice(0,None), slice(0,self.npml)]
            myidx_Hyzp  = [slice(0,None), slice(0,None), slice(-self.npml,None)]

        if self.method == 'SHPF':

            odd = [None, None, slice(1,None,2)]

            psiidx_Hxzp = [slice(0,None), slice(0,None), slice(0,self.npml)]
            myidx_Hxzp  = [slice(0,None), slice(0,None), slice(-self.npml,None)]

            if self.MPIrank < (self.MPIsize-1):
                psiidx_Hyzp = [slice(0,None), slice(0,None), slice(0,self.npml)]
                myidx_Hyzp  = [slice(0,None), slice(0,None), slice(-self.npml,None)]
            else:
                psiidx_Hyzp = [slice(0,-1), slice(0,None), slice(0,self.npml)]
                myidx_Hyzp  = [slice(0,-1), slice(0,None), slice(-self.npml,None)]

        if self.method == 'FDTD':

            odd = [None, None, slice(1,-1,2)]

            psiidx_Hxzp = [slice(0,None), slice(0,-1), slice(0,self.npml-1)]
            myidx_Hxzp  = [slice(0,None), slice(0,-1), slice(-self.npml,-1)]

            if self.MPIrank < (self.MPIsize-1):
                psiidx_Hyzp = [slice(0,None), slice(0,None), slice(0,self.npml-1)]
                myidx_Hyzp  = [slice(0,None), slice(0,None), slice(-self.npml,-1)]
            else:
                psiidx_Hyzp = [slice(0,-1), slice(0,None), slice(0,self.npml-1)]
                myidx_Hyzp  = [slice(0,-1), slice(0,None), slice(-self.npml,-1)]

        # Update Hx at z+.
        CHx2 = (-2*self.dt) / (2.*self.mu_Hx[myidx_Hxzp] + self.mcon_Hx[myidx_Hxzp]*self.dt)
        self.psi_hxz_p[psiidx_Hxzp] = (self.PMLbz[odd]*self.psi_hxz_p[psiidx_Hxzp])\
                                    + (self.PMLaz[odd]*self.diffzEy[myidx_Hxzp])
        self.Hx[myidx_Hxzp] += CHx2*(-((1./self.PMLkappaz[odd]-1.)*self.diffzEy[myidx_Hxzp]) - self.psi_hxz_p[psiidx_Hxzp])

        # Update Hy at z+.
        CHy2 = (-2*self.dt) / (2.*self.mu_Hy[myidx_Hyzp] + self.mcon_Hy[myidx_Hyzp]*self.dt)
        self.psi_hyz_p[psiidx_Hyzp] = (self.PMLbz[odd]*self.psi_hyz_p[psiidx_Hyzp])\
                                    + (self.PMLaz[odd]*self.diffzEx[myidx_Hyzp])
        self.Hy[myidx_Hyzp] += CHy2*(+((1./self.PMLkappaz[odd]-1.)*self.diffzEx[myidx_Hyzp]) + self.psi_hyz_p[psiidx_Hyzp])

    def _PML_updateE_pz(self):

        if self.method == 'PSTD':

            even = [None, None, slice(0,None,2)]

            psiidx_Exzp = [slice(0,None), slice(0,None), slice(0,self.npml)]
            myidx_Exzp  = [slice(0,None), slice(0,None), slice(-self.npml,None)]

            psiidx_Eyzp = [slice(0,None), slice(0,None), slice(0,self.npml)]
            myidx_Eyzp  = [slice(0,None), slice(0,None), slice(-self.npml,None)]

        if self.method == 'SHPF':

            even = [None, None, slice(0,None,2)]

            psiidx_Exzp = [slice(0,None), slice(0,None), slice(0,self.npml)]
            myidx_Exzp  = [slice(0,None), slice(0,None), slice(-self.npml,None)]

            if self.MPIrank > 0:
                psiidx_Eyzp = [slice(0,None), slice(0,None), slice(0,self.npml)]
                myidx_Eyzp  = [slice(0,None), slice(0,None), slice(-self.npml,None)]
            else:
                psiidx_Eyzp = [slice(1,None), slice(0,None), slice(0,self.npml)]
                myidx_Eyzp  = [slice(1,None), slice(0,None), slice(-self.npml,None)]

        elif self.method == 'FDTD':

            even = [None, None, slice(0,None,2)]

            psiidx_Exzp = [slice(0,None), slice(1,None), slice(0,self.npml)]
            myidx_Exzp  = [slice(0,None), slice(1,None), slice(-self.npml,None)]

            if self.MPIrank > 0:
                psiidx_Eyzp = [slice(0,None), slice(0,None), slice(0,self.npml)]
                myidx_Eyzp  = [slice(0,None), slice(0,None), slice(-self.npml,None)]
            else:
                psiidx_Eyzp = [slice(1,None), slice(0,None), slice(0,self.npml)]
                myidx_Eyzp  = [slice(1,None), slice(0,None), slice(-self.npml,None)]

        # Update Ex at z+.
        CEx2 = (2*self.dt) / (2.*self.eps_Ex[myidx_Exzp] + self.econ_Ex[myidx_Exzp]*self.dt)
        self.psi_exz_p[psiidx_Exzp] = (self.PMLbz[even]*self.psi_exz_p[psiidx_Exzp])\
                                    + (self.PMLaz[even]*self.diffzHy[myidx_Exzp])
        self.Ex[myidx_Exzp] += CEx2*(-((1./self.PMLkappaz[even]-1.)*self.diffzHy[myidx_Exzp]) - self.psi_exz_p[psiidx_Exzp])

        # Update Ey at z+.
        CEy2 = (2*self.dt) / (2.*self.eps_Ey[myidx_Eyzp] + self.econ_Ey[myidx_Eyzp]*self.dt)
        self.psi_eyz_p[psiidx_Eyzp] = (self.PMLbz[even]*self.psi_eyz_p[psiidx_Eyzp])\
                                    + (self.PMLaz[even]*self.diffzHx[myidx_Eyzp])
        self.Ey[myidx_Eyzp] += CEy2*(+((1./self.PMLkappaz[even]-1.)*self.diffzHx[myidx_Eyzp]) + self.psi_eyz_p[psiidx_Eyzp])

    def _PML_updateH_mz(self):

        if self.method == 'PSTD':

            even = [None, None, slice(-2,None,-2)]

            psiidx_Hxzm = [slice(0,None), slice(0,None), slice(0,self.npml)]
            myidx_Hxzm = [slice(0,None), slice(0,None), slice(0,self.npml)]

            psiidx_Hyzm = [slice(0,None), slice(0,None), slice(0,self.npml)]
            myidx_Hyzm = [slice(0,None), slice(0,None), slice(0,self.npml)]

        if self.method == 'SHPF':

            even = [None, None, slice(-2,None,-2)]

            psiidx_Hxzm = [slice(0,None), slice(0,None), slice(0,self.npml)]
            myidx_Hxzm = [slice(0,None), slice(0,None), slice(0,self.npml)]

            if self.MPIrank < (self.MPIsize-1):
                psiidx_Hyzm = [slice(0,None), slice(0,None), slice(0,self.npml)]
                myidx_Hyzm = [slice(0,None), slice(0,None), slice(0,self.npml)]
            else:
                psiidx_Hyzm = [slice(0,-1), slice(0,None), slice(0,self.npml)]
                myidx_Hyzm = [slice(0,-1), slice(0,None), slice(0,self.npml)]

        elif self.method == 'FDTD':

            even = [None, None, slice(-2,None,-2)]

            psiidx_Hxzm = [slice(0,None), slice(0,-1), slice(0,self.npml)]
            myidx_Hxzm = [slice(0,None), slice(0,-1), slice(0,self.npml)]

            if self.MPIrank < (self.MPIsize-1):
                psiidx_Hyzm = [slice(0,None), slice(0,None), slice(0,self.npml)]
                myidx_Hyzm = [slice(0,None), slice(0,None), slice(0,self.npml)]
            else:
                psiidx_Hyzm = [slice(0,-1), slice(0,-1), slice(0,self.npml)]
                myidx_Hyzm = [slice(0,-1), slice(0,-1), slice(0,self.npml)]

        # Update Hx at z-.
        CHx2 = (-2*self.dt) / (2.*self.mu_Hx[myidx_Hxzm] + self.mcon_Hx[myidx_Hxzm]*self.dt)
        self.psi_hxz_m[psiidx_Hxzm] = (self.PMLbz[even]*self.psi_hxz_m[psiidx_Hxzm])\
                                    + (self.PMLaz[even]*self.diffzEy[myidx_Hxzm])
        self.Hx[myidx_Hxzm] += CHx2*(-((1./self.PMLkappaz[even]-1.)*self.diffzEy[myidx_Hxzm]) - self.psi_hxz_m[psiidx_Hxzm])

        # Update Hy at z-.
        CHy2 = (-2*self.dt) / (2.*self.mu_Hy[myidx_Hyzm] + self.mcon_Hy[myidx_Hyzm]*self.dt)
        self.psi_hyz_m[psiidx_Hyzm] = (self.PMLbz[even]*self.psi_hyz_m[psiidx_Hyzm])\
                                    + (self.PMLaz[even]*self.diffzEx[myidx_Hyzm])
        self.Hy[myidx_Hyzm] += CHy2*(+((1./self.PMLkappaz[even]-1.)*self.diffzEx[myidx_Hyzm]) + self.psi_hyz_m[psiidx_Hyzm])

    def _PML_updateE_mz(self):

        if self.method == 'PSTD':

            odd = [None, None, slice(-2,None,-2)]

            psiidx_Exzm = [slice(0,None), slice(0,None), slice(0,self.npml)]
            myidx_Exzm  = [slice(0,None), slice(0,None), slice(0,self.npml)]

            psiidx_Eyzm = [slice(0,None), slice(0,None), slice(0,self.npml)]
            myidx_Eyzm  = [slice(0,None), slice(0,None), slice(0,self.npml)]

        if self.method == 'SHPF':

            odd = [None, None, slice(-1,None,-2)]

            psiidx_Exzm = [slice(0,None), slice(0,None), slice(0,self.npml)]
            myidx_Exzm  = [slice(0,None), slice(0,None), slice(0,self.npml)]

            if (self.MPIrank > 0):
                psiidx_Eyzm = [slice(0,None), slice(0,None), slice(0,self.npml)]
                myidx_Eyzm  = [slice(0,None), slice(0,None), slice(0,self.npml)]
            else:
                psiidx_Eyzm = [slice(1,None), slice(0,None), slice(0,self.npml)]
                myidx_Eyzm  = [slice(1,None), slice(0,None), slice(0,self.npml)]

        elif self.method == 'FDTD':

            odd = [None, None, slice(-3,None,-2)]

            psiidx_Exzm = [slice(0,None), slice(1,None), slice(1,self.npml)]
            myidx_Exzm  = [slice(0,None), slice(1,None), slice(1,self.npml)]

            if (self.MPIrank > 0):
                psiidx_Eyzm = [slice(0,None), slice(0,None), slice(1,self.npml)]
                myidx_Eyzm  = [slice(0,None), slice(0,None), slice(1,self.npml)]
            else:
                psiidx_Eyzm = [slice(1,None), slice(0,None), slice(1,self.npml)]
                myidx_Eyzm  = [slice(1,None), slice(0,None), slice(1,self.npml)]

        # Update Ex at z-.
        CEx2 = (2*self.dt) / (2.*self.eps_Ex[myidx_Exzm] + self.econ_Ex[myidx_Exzm]*self.dt)
        self.psi_exz_m[psiidx_Exzm] = (self.PMLbz[odd] * self.psi_exz_m[psiidx_Exzm])\
                                    + (self.PMLaz[odd] * self.diffzHy[myidx_Exzm])
        self.Ex[myidx_Exzm] += CEx2*(-((1./self.PMLkappaz[odd]-1.)*self.diffzHy[myidx_Exzm])-self.psi_exz_m[psiidx_Exzm])

        # Update Ey at z-.
        CEy2 = (2*self.dt) / (2.*self.eps_Ey[myidx_Eyzm] + self.econ_Ey[myidx_Eyzm]*self.dt)
        self.psi_eyz_m[psiidx_Eyzm] = (self.PMLbz[odd] * self.psi_eyz_m[psiidx_Eyzm])\
                                    + (self.PMLaz[odd] * self.diffzHx[myidx_Eyzm])
        self.Ey[myidx_Eyzm] += CEy2*(+((1./self.PMLkappaz[odd]-1.)*self.diffzHx[myidx_Eyzm])+self.psi_eyz_m[psiidx_Eyzm])

    def _exchange_BBCx(self, k, newL, fx, fy, fz, recv1, recv2, mpi):

        p0 = [ 0, slice(0,None), slice(0,None)]
        p1 = [ 1, slice(0,None), slice(0,None)]
        m1 = [-1, slice(0,None), slice(0,None)]
        m2 = [-2, slice(0,None), slice(0,None)]

        fx[m1] = fx[p1] * self.xp.exp(+1j*k*newL) 
        fy[m1] = fy[p1] * self.xp.exp(+1j*k*newL) 
        fz[m1] = fz[p1] * self.xp.exp(+1j*k*newL) 

        fy[p0] = fy[m2] * self.xp.exp(-1j*k*newL)
        fx[p0] = fx[m2] * self.xp.exp(-1j*k*newL)
        fz[p0] = fz[m2] * self.xp.exp(-1j*k*newL)

        if mpi == True:

            p0 = [ 0, slice(0,None)]
            p1 = [ 1, slice(0,None)]
            m1 = [-1, slice(0,None)]
            m2 = [-2, slice(0,None)]

            recv1[m1] = recv1[p1] * self.xp.exp(+1j*k*newL)
            recv2[m1] = recv2[p1] * self.xp.exp(+1j*k*newL)

            recv1[p0] = recv1[m2] * self.xp.exp(-1j*k*newL)
            recv2[p0] = recv2[m2] * self.xp.exp(-1j*k*newL)

    def _exchange_BBCy(self, k, newL, fx, fy, fz, recv1, recv2, mpi):

        p0 = [slice(0,None), 0, slice(0,None)]
        p1 = [slice(0,None), 1, slice(0,None)]
        m1 = [slice(0,None),-1, slice(0,None)]
        m2 = [slice(0,None),-2, slice(0,None)]

        fx[m1] = fx[p1] * self.xp.exp(+1j*k*newL) 
        fy[m1] = fy[p1] * self.xp.exp(+1j*k*newL) 
        fz[m1] = fz[p1] * self.xp.exp(+1j*k*newL) 

        fy[p0] = fy[m2] * self.xp.exp(-1j*k*newL)
        fx[p0] = fx[m2] * self.xp.exp(-1j*k*newL)
        fz[p0] = fz[m2] * self.xp.exp(-1j*k*newL)

        if mpi == True:

            p0 = [ 0, slice(0,None)]
            p1 = [ 1, slice(0,None)]
            m1 = [-1, slice(0,None)]
            m2 = [-2, slice(0,None)]

            recv1[m1] = recv1[p1] * self.xp.exp(+1j*k*newL)
            recv2[m1] = recv2[p1] * self.xp.exp(+1j*k*newL)

            recv1[p0] = recv1[m2] * self.xp.exp(-1j*k*newL)
            recv2[p0] = recv2[m2] * self.xp.exp(-1j*k*newL)

    def _exchange_BBCz(self, k, newL, fx, fy, fz, recv1, recv2, mpi):

        p0 = [slice(0,None), slice(0,None),  0]
        p1 = [slice(0,None), slice(0,None),  1]
        m1 = [slice(0,None), slice(0,None), -1]
        m2 = [slice(0,None), slice(0,None), -2]

        fx[m1] = fx[p1] * self.xp.exp(+1j*k*newL) 
        fy[m1] = fy[p1] * self.xp.exp(+1j*k*newL) 
        fz[m1] = fz[p1] * self.xp.exp(+1j*k*newL) 

        fx[p0] = fx[m2] * self.xp.exp(-1j*k*newL)
        fy[p0] = fy[m2] * self.xp.exp(-1j*k*newL)
        fz[p0] = fz[m2] * self.xp.exp(-1j*k*newL)

        if mpi == True:

            p0 = [slice(0,None),  0]
            p1 = [slice(0,None),  1]
            m1 = [slice(0,None), -1]
            m2 = [slice(0,None), -2]

            recv1[m1] = recv1[p1] * self.xp.exp(+1j*k*newL)
            recv2[m1] = recv2[p1] * self.xp.exp(+1j*k*newL)

            recv1[p0] = recv1[m2] * self.xp.exp(-1j*k*newL)
            recv2[p0] = recv2[m2] * self.xp.exp(-1j*k*newL)

    def _updateH_BBC_FDTD(self):

        if self.apply_BBCx == True: 

            assert self.apply_PBCx == False
            newL = self.Lx - 2*self.dx

            # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
            self._exchange_BBCx(self.mmt[0], newL, self.Ex, self.Ey, self.Ez, None, None, mpi=False)

        if self.apply_PBCx == True: 

            assert self.apply_BBCx == False
            newL = 0

            # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
            self._exchange_BBCx(self.mmt[0], newL, self.Ex, self.Ey, self.Ez, None, None, mpi=False)

        if self.apply_BBCy == True: 

            assert self.apply_PBCy == False
            newL = self.Ly - 2*self.dy

            # Exchange Ex,Ey,Ez at j=0,1 with j=Ny-2, Ny-1.
            if self.MPIrank == (self.MPIsize-1): 
                self._exchange_BBCy(self.mmt[1], newL, self.Ex, self.Ey, self.Ez, None, None, mpi=False)
            else: 
                self._exchange_BBCy(self.mmt[1], newL, self.Ex, self.Ey, self.Ez, self.recvEylast, self.recvEzlast, mpi=True)

        if self.apply_PBCy == True: 
        
            assert self.apply_BBCy == False
            newL = 0

            # Exchange Ex,Ey,Ez at j=0,1 with j=Ny-2, Ny-1.
            if self.MPIrank == (self.MPIsize-1): 
                self._exchange_BBCy(self.mmt[1], newL, self.Ex, self.Ey, self.Ez, None, None, mpi=False)
            else: 
                self._exchange_BBCy(self.mmt[1], newL, self.Ex, self.Ey, self.Ez, self.recvEylast, self.recvEzlast, mpi=True)

            """
            # First rank
            if self.MPIrank == 0:
                if 'x' in self.PMLregion.keys():
                    if '+' in self.PMLregion.get('x') and self.MPIsize == 1: self._pxPML_pyPBC()
                    if '-' in self.PMLregion.get('x'): self._mxPML_pyPBC()
            # Middle rank
            elif self.MPIrank > 0 and self.MPIrank < (self.MPIsize-1):
                if 'x' in self.PMLregion.keys():
                    if '+' in self.PMLregion.get('x'): pass
                    if '-' in self.PMLregion.get('x'): pass
            # Last rank
            elif self.MPIrank == (self.MPIsize-1) and self.MPIsize != 1:
                if 'x' in self.PMLregion.keys():
                    if '+' in self.PMLregion.get('x'): self._pxPML_pyPBC()
                    if '-' in self.PMLregion.get('x'): pass
            """

        if self.apply_BBCz == True: 

            assert self.apply_PBCz == False
            newL = self.Lz - 2*self.dz

            # Exchange Ex,Ey,Ez at k=0,1 with k=Nz-2, Nz-1.
            if self.MPIrank == (self.MPIsize-1): 
                self._exchange_BBCz(self.mmt[2], newL, self.Ex, self.Ey, self.Ez, None, None, mpi=False)
            else: 
                self._exchange_BBCz(self.mmt[2], newL, self.Ex, self.Ey, self.Ez, self.recvEylast, self.recvEzlast, mpi=True)

        if self.apply_PBCz == True: 
        
            assert self.apply_BBCz == False
            newL = 0

            # Exchange Ex,Ey,Ez at k=0,1 with k=Nz-2, Nz-1.
            if self.MPIrank == (self.MPIsize-1): 
                self._exchange_BBCz(self.mmt[2], newL, self.Ex, self.Ey, self.Ez, None, None, mpi=False)
            else: 
                self._exchange_BBCz(self.mmt[2], newL, self.Ex, self.Ey, self.Ez, self.recvEylast, self.recvEzlast, mpi=True)

            """
            # First rank
            if self.MPIrank == 0:
                if 'x' in self.PMLregion.keys():
                    if '+' in self.PMLregion.get('x') and self.MPIsize == 1: self._pxPML_pzPBC()
                    if '-' in self.PMLregion.get('x'): self._mxPML_pzPBC()

            # Middle rank
            elif self.MPIrank > 0 and self.MPIrank < (self.MPIsize-1):
                if 'x' in self.PMLregion.keys():
                    if '+' in self.PMLregion.get('x'): pass
                    if '-' in self.PMLregion.get('x'): pass
            # Last rank
            elif self.MPIrank == (self.MPIsize-1) and self.MPIsize != 1:
                if 'x' in self.PMLregion.keys():
                    if '+' in self.PMLregion.get('x'): self._pxPML_pzPBC()
                    if '-' in self.PMLregion.get('x'): pass
            """

    def _updateH_BBC_SHPF(self):

        sli1 = [slice(None,None), slice(None,None), slice(None,None)]
        sli2 = [slice(None,  -1), slice(None,None), slice(None,None)]

        if self.apply_BBCx == True: 

            assert self.apply_PBCx == False
            newL = self.Lx - 2*self.dx

            # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
            self._exchange_BBCx(self.mmt[0], newL, self.Ex, self.Ey, self.Ez, None, None, mpi=False)

        if self.apply_PBCx == True: 

            assert self.apply_BBCx == False
            newL = 0

            # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
            self._exchange_BBCx(self.mmt[0], newL, self.Ex, self.Ey, self.Ez, None, None, mpi=False)

            #self.ez_at_Hy[:-1,:,:] = (self.Ez[:-1,:,:] + self.Ez[1:,:,:]) / 2
            #self.ey_at_Hz[:-1,:,:] = (self.Ey[1:,:,:] + self.Ey[:-1,:,:]) / 2

            #self.Hx[sli1] += -self.dt/self.mu_Hx[sli1]*1j*(self.mmt[1]*self.ez_at_Hx[sli1] - self.mmt[2]*self.ey_at_Hx[sli1])
            #self.Hy[sli2] += -self.dt/self.mu_Hy[sli2]*1j*(self.mmt[2]*self.ex_at_Hy[sli2] - self.mmt[0]*self.ez_at_Hy[sli2])
            #self.Hz[sli2] += -self.dt/self.mu_Hz[sli2]*1j*(self.mmt[0]*self.ey_at_Hz[sli2] - self.mmt[1]*self.ex_at_Hz[sli2])

            #raise ValueError("BBC along x axis is not developed yet!")

        if self.apply_BBCy == True:

            assert self.apply_PBCy == False
            self.ez_at_Hx = self.ifft(self.ypshift*self.fft(self.Ez, axes=(1,)), axes=(1,))
            self.ex_at_Hz = self.ifft(self.ypshift*self.fft(self.Ex, axes=(1,)), axes=(1,))

            self.Hx[sli1] += -self.dt/self.mu_Hx[sli1]*1j*(-self.mmt[1]*self.ez_at_Hx[sli1])
            self.Hz[sli2] += -self.dt/self.mu_Hz[sli2]*1j*(+self.mmt[1]*self.ex_at_Hz[sli2])

        if self.apply_BBCz == True:

            assert self.apply_PBCz == False
            self.ey_at_Hx = self.ifft(self.zpshift*self.fft(self.Ey, axes=(2,)), axes=(2,))
            self.ex_at_Hy = self.ifft(self.zpshift*self.fft(self.Ex, axes=(2,)), axes=(2,))

            self.Hx[sli1] += -self.dt/self.mu_Hx[sli1]*1j*(+self.mmt[2]*self.ey_at_Hx[sli1])
            self.Hy[sli2] += -self.dt/self.mu_Hy[sli2]*1j*(-self.mmt[2]*self.ex_at_Hy[sli2])

    def _updateH_BBC_PSTD(self):

        if self.apply_BBCx == True:

            #self.Hx[sli1] += -self.dt/self.mu_Hx[sli1]*1j*(self.mmt[1]*self.ez_at_Hx[sli1] - self.mmt[2]*self.ey_at_Hx[sli1])
            #self.Hy[sli2] += -self.dt/self.mu_Hy[sli2]*1j*(self.mmt[2]*self.ex_at_Hy[sli2] - self.mmt[0]*self.ez_at_Hy[sli2])
            #self.Hz[sli2] += -self.dt/self.mu_Hz[sli2]*1j*(self.mmt[0]*self.ey_at_Hz[sli2] - self.mmt[1]*self.ex_at_Hz[sli2])

            assert self.apply_PBCx == False
            self.Hy += -self.dt/self.mu_Hy*1j*(+self.mmt[0]*self.Ez)
            self.Hz += -self.dt/self.mu_Hz*1j*(-self.mmt[0]*self.Ey)

        if self.apply_BBCy == True:

            assert self.apply_PBCy == False
            self.Hx += -self.dt/self.mu_Hx*1j*(-self.mmt[1]*self.Ez)
            self.Hz += -self.dt/self.mu_Hz*1j*(+self.mmt[1]*self.Ex)

        if self.apply_BBCz == True:

            assert self.apply_PBCz == False
            self.Hx += -self.dt/self.mu_Hx*1j*(+self.mmt[2]*self.Ey)
            self.Hy += -self.dt/self.mu_Hy*1j*(-self.mmt[2]*self.Ex)

    def _updateE_BBC_FDTD(self):

        if   self.apply_BBCx == True: 
        
            newL = self.Lx - 2*self.dx

            # Exchange Hx,Hy,Hz at i=0,1 with i=Nx-2, Nx-1.
            self._exchange_BBCx(self.mmt[0], newL, self.Hx, self.Hy, self.Hz, None, None, mpi=False)

        elif self.apply_PBCx == True:
        
            newL = 0

            # Exchange Hx,Hy,Hz at i=0,1 with i=Nx-2, Nx-1.
            self._exchange_BBCx(self.mmt[0], newL, self.Hx, self.Hy, self.Hz, None, None, mpi=False)

            #self.hy_at_Ez[1:,:,:] = self.ifft(self.xmshift*self.fft(self.Hy, axes=(0,)), axes=(0,))
            #self.hz_at_Ey[1:,:,:] = (self.Hz[1:,:,:] + self.Hz[:-1,:,:]) / 2

            #self.Ex[sli1] += self.dt/self.eps_Ex[sli1]*1j*(self.mmt[1]*self.hz_at_Ex[sli1] - self.mmt[2]*self.hy_at_Ex[sli1])
            #self.Ey[sli2] += self.dt/self.eps_Ey[sli2]*1j*(self.mmt[2]*self.hx_at_Ey[sli2] - self.mmt[0]*self.hz_at_Ey[sli2])
            #self.Ez[sli2] += self.dt/self.eps_Ez[sli2]*1j*(self.mmt[0]*self.hy_at_Ez[sli2] - self.mmt[1]*self.hx_at_Ez[sli2])

        if   self.apply_BBCy == True: 
        
            newL = self.Ly - 2*self.dy

            # Exchange Hx,Hy,Hz at j=0,1 with j=Ny-2, Ny-1.
            if self.MPIrank == 0: 
                self._exchange_BBCy(self.mmt[1], newL, self.Hx, self.Hy, self.Hz, None, None, mpi=False)
            else: 
                self._exchange_BBCy(self.mmt[1], newL, self.Hx, self.Hy, self.Hz, self.recvHyfirst, self.recvHzfirst, mpi=True)

        elif self.apply_PBCy == True: 
        
            newL = 0

            # Exchange Hx,Hy,Hz at j=0,1 with j=Ny-2, Ny-1.
            if self.MPIrank == 0: 
                self._exchange_BBCy(self.mmt[1], newL, self.Hx, self.Hy, self.Hz, None, None, mpi=False)
            else: 
                self._exchange_BBCy(self.mmt[1], newL, self.Hx, self.Hy, self.Hz, self.recvHyfirst, self.recvHzfirst, mpi=True)

            """
            # First rank
            if self.MPIrank == 0:
                if 'x' in self.PMLregion.keys():
                    if '+' in self.PMLregion.get('x') and self.MPIsize == 1: self._pxPML_myPBC()
                    if '-' in self.PMLregion.get('x'): self._mxPML_myPBC()
            # Middle rank
            elif self.MPIrank > 0 and self.MPIrank < (self.MPIsize-1):
                if 'x' in self.PMLregion.keys():
                    if '+' in self.PMLregion.get('x'): pass
                    if '-' in self.PMLregion.get('x'): pass
            # Last rank
            elif self.MPIrank == (self.MPIsize-1) and self.MPIsize != 1:
                if 'x' in self.PMLregion.keys():
                    if '+' in self.PMLregion.get('x'): self._pxPML_myPBC()
                    if '-' in self.PMLregion.get('x'): pass
            """

        if   self.apply_BBCz == True: 
        
            newL = self.Lz - 2*self.dz

            if self.MPIrank == 0: 
                self._exchange_BBCz(self.mmt[2], newL, self.Hx, self.Hy, self.Hz, None, None, mpi=False)
            else: 
                self._exchange_BBCz(self.mmt[2], newL, self.Hx, self.Hy, self.Hz, self.recvHyfirst, self.recvHzfirst, mpi=True)

        elif self.apply_PBCz == True: 
        
            newL = 0

            if self.MPIrank == 0: 
                self._exchange_BBCz(self.mmt[2], newL, self.Hx, self.Hy, self.Hz, None, None, mpi=False)
            else: 
                self._exchange_BBCz(self.mmt[2], newL, self.Hx, self.Hy, self.Hz, self.recvHyfirst, self.recvHzfirst, mpi=True)

            """
            # First rank
            if self.MPIrank == 0:
                if 'x' in self.PMLregion.keys():
                    if '+' in self.PMLregion.get('x') and self.MPIsize == 1: self._pxPML_mzPBC()
                    if '-' in self.PMLregion.get('x'): self._mxPML_mzPBC()

            # Middle rank
            elif self.MPIrank > 0 and self.MPIrank < (self.MPIsize-1):
                if 'x' in self.PMLregion.keys():
                    if '+' in self.PMLregion.get('x'): pass
                    if '-' in self.PMLregion.get('x'): pass
            # Last rank
            elif self.MPIrank == (self.MPIsize-1) and self.MPIsize != 1:
                if 'x' in self.PMLregion.keys():
                    if '+' in self.PMLregion.get('x'): self._pxPML_mzPBC()
                    if '-' in self.PMLregion.get('x'): pass
            """

    def _updateE_BBC_SHPF(self):

        sli1 = [slice(None,None), slice(None,None), slice(None,None)]
        sli2 = [slice(   1,None), slice(None,None), slice(None,None)]

        if self.apply_BBCx == True: 

            newL = self.Lx - 2*self.dx

            # Exchange Hx,Hy,Hz at i=0,1 with i=Nx-2, Nx-1.
            self._exchange_BBCx(self.mmt[0], newL, self.Hx, self.Hy, self.Hz, None, None, mpi=False)

        elif self.apply_PBCx == True: 

            newL = 0

            # Exchange Hx,Hy,Hz at i=0,1 with i=Nx-2, Nx-1.
            self._exchange_BBCx(self.mmt[0], newL, self.Hx, self.Hy, self.Hz, None, None, mpi=False)

            #self.hy_at_Ez[1:,:,:] = self.ifft(self.xmshift*self.fft(self.Hy, axes=(0,)), axes=(0,))
            #self.hz_at_Ey[1:,:,:] = (self.Hz[1:,:,:] + self.Hz[:-1,:,:]) / 2

            #self.Ex[sli1] += self.dt/self.eps_Ex[sli1]*1j*(self.mmt[1]*self.hz_at_Ex[sli1] - self.mmt[2]*self.hy_at_Ex[sli1])
            #self.Ey[sli2] += self.dt/self.eps_Ey[sli2]*1j*(self.mmt[2]*self.hx_at_Ey[sli2] - self.mmt[0]*self.hz_at_Ey[sli2])
            #self.Ez[sli2] += self.dt/self.eps_Ez[sli2]*1j*(self.mmt[0]*self.hy_at_Ez[sli2] - self.mmt[1]*self.hx_at_Ez[sli2])

        if self.apply_BBCy == True: 
        
            newL = self.Ly - 2*self.dy

            self.hz_at_Ex = self.ifft(self.ymshift*self.fft(self.Hz, axes=(1,)), axes=(1,))
            self.hx_at_Ez = self.ifft(self.ymshift*self.fft(self.Hx, axes=(1,)), axes=(1,))

            self.Ex[sli1] += self.dt/self.eps_Ex[sli1]*1j*(-self.mmt[1]*self.hz_at_Ex[sli1])
            self.Ez[sli2] += self.dt/self.eps_Ez[sli2]*1j*(+self.mmt[1]*self.hx_at_Ez[sli2])

        if self.apply_BBCz == True:

            newL = self.Lz - 2*self.dz

            self.hy_at_Ex = self.ifft(self.zmshift*self.fft(self.Hy, axes=(2,)), axes=(2,))
            self.hx_at_Ey = self.ifft(self.zmshift*self.fft(self.Hx, axes=(2,)), axes=(2,))

            self.Ex[sli1] += self.dt/self.eps_Ex[sli1]*1j*(+self.mmt[2]*self.hy_at_Ex[sli1])
            self.Ey[sli2] += self.dt/self.eps_Ey[sli2]*1j*(-self.mmt[2]*self.hx_at_Ey[sli2])

    def _updateE_BBC_PSTD(self):

        if self.apply_BBCx == True:

            #self.Ex[sli1] += self.dt/self.eps_Ex[sli1]*1j*(self.mmt[1]*self.hz_at_Ex[sli1] - self.mmt[2]*self.hy_at_Ex[sli1])
            #self.Ey[sli2] += self.dt/self.eps_Ey[sli2]*1j*(self.mmt[2]*self.hx_at_Ey[sli2] - self.mmt[0]*self.hz_at_Ey[sli2])
            #self.Ez[sli2] += self.dt/self.eps_Ez[sli2]*1j*(self.mmt[0]*self.hy_at_Ez[sli2] - self.mmt[1]*self.hx_at_Ez[sli2])

            self.Ey += self.dt/self.eps_Ey*1j*(+self.mmt[0]*self.Hz)
            self.Ez += self.dt/self.eps_Ez*1j*(-self.mmt[0]*self.Hy)

        if self.apply_BBCy == True:
            
            self.Ex += self.dt/self.eps_Ex*1j*(-self.mmt[1]*self.Hz)
            self.Ez += self.dt/self.eps_Ez*1j*(+self.mmt[1]*self.Hx)

        if self.apply_BBCz == True:

            self.Ex += self.dt/self.eps_Ex*1j*(+self.mmt[2]*self.Hy)
            self.Ey += self.dt/self.eps_Ey*1j*(-self.mmt[2]*self.Hx)

    """
    def _updateH_PBC_py(self, mpi):

        assert self.method == 'FDTD'
        shift = self.xp.exp(1j*self.mmt[1]*self.Ly)

	# Update Hx at j=(Ny-1) by using Ez at j=0
        fidx   = [slice(None,None),  0, slice(0,  -1)]
        lidx   = [slice(None,None), -1, slice(0,  -1)]
        k_lidx = [slice(None,None), -1, slice(1,None)]

        CHx1 = (2.*self.mu_Hx[lidx] - self.mcon_Hx[lidx]*self.dt) / (2.*self.mu_Hx[lidx] + self.mcon_Hx[lidx]*self.dt)
        CHx2 = (-2*self.dt) / (2.*self.mu_Hx[lidx] + self.mcon_Hx[lidx]*self.dt)

        self.diffyEz[lidx] = (self.Ez[fidx]   - self.Ez[lidx]) / self.dy
        self.diffzEy[lidx] = (self.Ey[k_lidx] - self.Ey[lidx]) / self.dz
        self.Hx[lidx] = CHx1 * self.Hx[lidx] + CHx2 * (self.diffyEz[lidx] - self.diffzEy[lidx])

	# Update Hz at j=(Ny-1) by using Ex at j=0
        fidx   = [slice(0,-1  ),  0, slice(0,None)]
        lidx   = [slice(0,-1  ), -1, slice(0,None)]
        i_lidx = [slice(1,None), -1, slice(0,None)]

        CHz1 = (2.*self.mu_Hz[lidx] - self.mcon_Hz[lidx]*self.dt) / (2.*self.mu_Hz[lidx] + self.mcon_Hz[lidx]*self.dt)
        CHz2 = (-2*self.dt) / (2.*self.mu_Hz[lidx] + self.mcon_Hz[lidx]*self.dt)

        self.diffxEy[lidx] = (self.Ey[i_lidx] - self.Ey[lidx]) / self.dx
        self.diffyEx[lidx] = (self.Ex[fidx]   - self.Ex[lidx]) / self.dy
        self.Hz[lidx] = CHz1 * self.Hz[lidx] + CHz2 * (self.diffxEy[lidx] - self.diffyEx[lidx])

	# Update Hz at j=(Ny-1) and i=(myNx-1) by using Ex at j=0 and Ey from next rank.
        if mpi == True:

            lidx  = [-1, -1, slice(0,None)]
            fiidx = [-1, slice(0,None)]
            fjidx = [-1,  0, slice(0,None)]

            CHz1 = (2.*self.mu_Hz[lidx] - self.mcon_Hz[lidx]*self.dt) / (2.*self.mu_Hz[lidx] + self.mcon_Hz[lidx]*self.dt)
            CHz2 = (-2*self.dt) / (2.*self.mu_Hz[lidx] + self.mcon_Hz[lidx]*self.dt)

            self.diffxEy[lidx] = (self.recvEylast[fiidx] - self.Ey[lidx]) / self.dx
            self.diffyEx[lidx] = (self.Ex[fjidx]         - self.Ex[lidx]) / self.dy

            self.Hz[lidx] = CHz1 * self.Hz[lidx] + CHz2 * (self.diffxEy[lidx] - self.diffyEx[lidx])

    def _updateE_PBC_my(self, mpi):

        assert self.method == 'FDTD'

        # Update Ex at j=0 by using Hy at j=Ny-1
        fidx   = [slice(None,None),  0, slice(1,None)] # first index.
        fidx_k = [slice(None,None),  0, slice(0,-1  )]
        lidx   = [slice(None,None), -1, slice(1,None)] # last index.

        CEx1 = (2.*self.eps_Ex[fidx] - self.econ_Ex[fidx]*self.dt) / (2.*self.eps_Ex[fidx] + self.econ_Ex[fidx]*self.dt)
        CEx2 = (2.*self.dt) / (2.*self.eps_Ex[fidx] + self.econ_Ex[fidx]*self.dt)

        # PEC condition.
        pec = (self.eps_Ex[fidx] > 1e3)
        CEx1[pec] = 0.
        CEx2[pec] = 0.

        self.diffyHz[fidx] = (self.Hz[fidx] - self.Hz[lidx]  ) / self.dy
        self.diffzHy[fidx] = (self.Hy[fidx] - self.Hy[fidx_k]) / self.dz
        self.Ex[fidx] = CEx1 * self.Ex[fidx] + CEx2 * (self.diffyHz[fidx] - self.diffzHy[fidx])

        # Update Ez at j=0 by using Hx at j=Ny-1
        fidx   = [slice(1,None),  0, slice(None,None)] # first index.
        fidx_i = [slice(0,  -1),  0, slice(None,None)]
        lidx   = [slice(1,None), -1, slice(None,None)] # last index.

        CEz1 = (2.*self.eps_Ez[fidx] - self.econ_Ez[fidx]*self.dt) / (2.*self.eps_Ez[fidx] + self.econ_Ez[fidx]*self.dt)
        CEz2 = (2.*self.dt) / (2.*self.eps_Ez[fidx] + self.econ_Ez[fidx]*self.dt)

        # PEC condition.
        pec = self.eps_Ez[fidx] > 1e3
        CEz1[pec] = 0.
        CEz2[pec] = 0.

        self.diffxHy[fidx] = (self.Hy[fidx] - self.Hy[fidx_i]) / self.dx
        self.diffyHx[fidx] = (self.Hx[fidx] - self.Hx[lidx]  ) / self.dy
        self.Ez[fidx] = CEz1 * self.Ez[fidx] + CEz2 * (self.diffxHy[fidx] - self.diffyHx[fidx])

	# Update Ez at i=0 and j=0 by using Hx at j=(Ny-1) and Hy from previous rank.
        if mpi == True:

            fidx = [0,  0, slice(None,None)] # first index.
            fiidx = [0, slice(None,None)] # i=0, first index.
            lidx = [0, -1, slice(None,None)] # last index.

            CEz1 = (2.*self.eps_Ez[fidx] - self.econ_Ez[fidx]*self.dt) / (2.*self.eps_Ez[fidx] + self.econ_Ez[fidx]*self.dt)
            CEz2 = (2.*self.dt) / (2.*self.eps_Ez[fidx] + self.econ_Ez[fidx]*self.dt)

            # PEC condition.
            pec = self.eps_Ez[fidx] > 1e3
            CEz1[pec] = 0.
            CEz2[pec] = 0.

            self.diffxHy[fidx] = (self.Hy[fidx] - self.recvHyfirst[fiidx])/ self.dx
            self.diffyHx[fidx] = (self.Hx[fidx] - self.Hx[lidx]         ) / self.dy
            self.Ez[fidx] = CEz1 * self.Ez[fidx] + CEz2 * (self.diffxHy[fidx] - self.diffyHx[fidx])

    def _updateE_PBC_mz(self, mpi):

        # Update Ex at k=0
        fidx   = [slice(0,None), slice(1,None), 0]
        fidx_j = [slice(0,None), slice(0,  -1), 0]
        lidx   = [slice(0,None), slice(1,None),-1]

        CEx1 = (2.*self.eps_Ex[fidx] - self.econ_Ex[fidx]*self.dt) / (2.*self.eps_Ex[fidx] + self.econ_Ex[fidx]*self.dt)
        CEx2 = (2.*self.dt) / (2.*self.eps_Ex[fidx] + self.econ_Ex[fidx]*self.dt)

        # PEC condition.
        pec = self.eps_Ex[fidx] > 1e3
        CEx1[pec] = 0.
        CEx2[pec] = 0.

        #self.Ex[fidx] = self.Ex[:,1:,self.Nz-2] * shift
        #self.Ex[lidx] = self.Ex[fidx] * 

        self.diffyHz[fidx] = (self.Hz[fidx] - self.Hz[fidx_j]) / self.dy	
        self.diffzHy[fidx] = (self.Hy[fidx] - self.Hy[lidx]  ) / self.dz
        self.Ex[fidx] = CEx1 * self.Ex[fidx] + CEx2 * (self.diffyHz[fidx] - self.diffzHy[fidx])

        # Update Ey at k=0
        fidx   = [slice(1,None), slice(0,None), 0]
        fidx_i = [slice(0,  -1), slice(0,None), 0]
        lidx   = [slice(1,None), slice(0,None),-1]

        CEy1 = (2.*self.eps_Ey[fidx] - self.econ_Ey[fidx]*self.dt) / (2.*self.eps_Ey[fidx] + self.econ_Ey[fidx]*self.dt)
        CEy2 = (2.*self.dt) / (2.*self.eps_Ey[fidx] + self.econ_Ey[fidx]*self.dt)

        # PEC condition.
        pec = self.eps_Ey[fidx] > 1e3
        CEy1[pec] = 0.
        CEy2[pec] = 0.

        self.diffxHz[fidx] = (self.Hz[fidx] - self.Hz[fidx_i]) / self.dx	
        self.diffzHx[fidx] = (self.Hx[fidx] - self.Hx[lidx]  ) / self.dz
        self.Ey[fidx] = CEy1 * self.Ey[fidx] + CEy2 * (self.diffzHx[fidx] - self.diffxHz[fidx])

        if mpi == True:

            fidx = [0, slice(0,None), 0]
            fiidx = [slice(0,None), 0] # i=0, first index.
            lidx = [0, slice(0,None),-1]

            CEy1 = (2.*self.eps_Ey[fidx] - self.econ_Ey[fidx]*self.dt) / (2.*self.eps_Ey[fidx] + self.econ_Ey[fidx]*self.dt)
            CEy2 = (2.*self.dt) / (2.*self.eps_Ey[fidx] + self.econ_Ey[fidx]*self.dt)

            pec = self.eps_Ey[fidx] > 1e3

            CEy1[pec] = 0.
            CEy2[pec] = 0.

            self.diffxHz[fidx] = (self.Hz[fidx] - self.recvHzfirst[fiidx]) / self.dx
            self.diffzHx[fidx] = (self.Hx[fidx] - self.Hx[lidx]          ) / self.dz
            self.Ey[fidx] = CEy1 * self.Ey[fidx] + CEy2 * (self.diffzHx[fidx] - self.diffxHz[fidx])

    def _updateH_PBC_pz(self, mpi):

        assert self.method == 'FDTD'

        # Update Hx at k=(Nz-1)
        lidx   = [slice(0,None), slice(0,  -1), -1]
        j_lidx = [slice(0,None), slice(1,None), -1]
        fidx   = [slice(0,None), slice(0,  -1),  0]

        CHx1 = (2.*self.mu_Hx[lidx] - self.mcon_Hx[lidx]*self.dt) / (2.*self.mu_Hx[lidx] + self.mcon_Hx[lidx]*self.dt)
        CHx2 = (-2*self.dt) / (2.*self.mu_Hx[lidx] + self.mcon_Hx[lidx]*self.dt)

        self.diffyEz[lidx] = (self.Ez[j_lidx] - self.Ez[lidx]) / self.dy	
        self.diffzEy[lidx] = (self.Ey[fidx]   - self.Ey[lidx]) / self.dz
        self.Hx[lidx] = CHx1 * self.Hx[lidx] + CHx2 * (self.diffyEz[lidx] - self.diffzEy[lidx])

        # Update Hy at k=(Nz-1)
        lidx   = [slice(0,  -1), slice(0,None), -1]
        i_lidx = [slice(1,None), slice(0,None), -1]
        fidx   = [slice(0,  -1), slice(0,None),  0]

        CHy1 = (2.*self.mu_Hy[lidx] - self.mcon_Hy[lidx]*self.dt) / (2.*self.mu_Hy[lidx] + self.mcon_Hy[lidx]*self.dt)
        CHy2 = (-2*self.dt) / (2.*self.mu_Hy[lidx] + self.mcon_Hy[lidx]*self.dt)

        self.diffxEz[lidx] = (self.Ez[i_lidx] - self.Ez[lidx]) / self.dx	
        self.diffzEx[lidx] = (self.Ex[fidx]   - self.Ex[lidx]) / self.dz
        self.Hy[lidx] = CHy1 * self.Hy[lidx] + CHy2 * (self.diffzEx[lidx] - self.diffxEz[lidx])

        if mpi == True:

            lidx  = [-1, slice(0,None), -1]
            fiidx = [slice(0,None), -1]
            fkidx = [-1, slice(0,None),  0]

            CHy1 = (2.*self.mu_Hy[lidx] - self.mcon_Hy[lidx]*self.dt) / (2.*self.mu_Hy[lidx] + self.mcon_Hy[lidx]*self.dt)
            CHy2 = (-2*self.dt) / (2.*self.mu_Hy[lidx] + self.mcon_Hy[lidx]*self.dt)

            self.diffxEz[lidx] = (self.recvEzlast[fiidx] - self.Ez[lidx]) / self.dx	
            self.diffzEx[lidx] = (self.Ex[fkidx]         - self.Ex[lidx]) / self.dz
            self.Hy[lidx] = CHy1 * self.Hy[lidx] + CHy2 * (self.diffzEx[lidx] - self.diffxEz[lidx])

    def _mxPML_myPBC(self):

        assert self.method == 'FDTD'

        odd = slice(-3,None,-2)

        psiidx = [slice(1,self.npml), 0, slice(0,None)]
        fidx   = [slice(1,self.npml), 0, slice(0,None)]

        CEz2 = (2.*self.dt) / (2.*self.eps_Ez[fidx] + self.econ_Ez[fidx]*self.dt)

        # PEC condition.
        pec = self.eps_Ez[fidx] > 1e3
        CEz2[pec] = 0.

        self.psi_ezx_m[psiidx] = (self.PMLbx[odd,None] * self.psi_ezx_m[psiidx]) + (self.PMLax[odd,None] * self.diffxHy[fidx])
        self.Ez[fidx] += CEz2 * (+(1./self.PMLkappax[odd,None] - 1.) * self.diffxHy[fidx] + self.psi_ezx_m[psiidx])

    def _mxPML_pyPBC(self):

        assert self.method == 'FDTD'

        even = slice(-2,None,-2)

        psiidx = [slice(0,self.npml), -1, slice(0,None)]
        lidx   = [slice(0,self.npml), -1, slice(0,None)]

        CHz2 = (-2.*self.dt) / (2.*self.mu_Hz[lidx] + self.mcon_Hz[lidx]*self.dt)

        self.psi_hzx_m[psiidx] = (self.PMLbx[even,None] * self.psi_hzx_m[psiidx]) + (self.PMLax[even,None] * self.diffxEy[lidx])
        self.Hz[lidx] += CHz2 * (+((1./self.PMLkappax[even,None] - 1.) * self.diffxEy[lidx]) + self.psi_hzx_m[psiidx])

    def _mxPML_mzPBC(self):

        assert self.method == 'FDTD'

        odd = slice(-3,None,-2)

        psiidx = [slice(1,self.npml), slice(0,None), 0]
        fidx   = [slice(1,self.npml), slice(0,None), 0]

        CEy2 = (2.*self.dt) / (2.*self.eps_Ey[fidx] + self.econ_Ey[fidx]*self.dt)

        # PEC condition.
        pec = self.eps_Ey[fidx] > 1e3
        CEy2[pec] = 0.

        self.psi_eyx_m[psiidx] = (self.PMLbx[odd,None] * self.psi_eyx_m[psiidx]) + (self.PMLax[odd,None] * self.diffxHz[fidx])
        self.Ey[fidx] += CEy2 * (-(1./self.PMLkappax[odd,None] - 1.) * self.diffxHz[fidx] - self.psi_eyx_m[psiidx])

    def _mxPML_pzPBC(self):

        assert self.method == 'FDTD'

        even = slice(-2,None,-2)

        psiidx = [slice(0,self.npml), slice(0,None), -1]
        lidx   = [slice(0,self.npml), slice(0,None), -1]

        CHy2 =	(-2*self.dt) / (2.*self.mu_Hy[lidx] + self.mcon_Hy[lidx]*self.dt)

        self.psi_hyx_m[psiidx] = (self.PMLbx[even,None] * self.psi_hyx_m[psiidx]) + (self.PMLax[even,None] * self.diffxEz[lidx])
        self.Hy[lidx] += CHy2 * (-((1./self.PMLkappax[even,None] - 1.) * self.diffxEz[lidx]) - self.psi_hyx_m[psiidx])

    def _pxPML_myPBC(self):

        assert self.method == 'FDTD'

        even = slice(0,None,2)

        psiidx = [slice(0,self.npml), 0, slice(0,None)]
        myidx = [slice(-self.npml,None), 0, slice(0,None)]

        CEz2 = (2.*self.dt) / (2.*self.eps_Ez[myidx] + self.econ_Ez[myidx]*self.dt)

        # PEC condition.
        pec = self.eps_Ez[myidx] > 1e3
        CEz2[pec] = 0.

        self.psi_ezx_p[psiidx] = (self.PMLbx[even,None] * self.psi_ezx_p[psiidx]) + (self.PMLax[even,None] * self.diffxHy[myidx])
        self.Ez[myidx] += CEz2 * (+(1./self.PMLkappax[even,None] - 1.) * self.diffxHy[myidx] + self.psi_ezx_p[psiidx])

    def _pxPML_pyPBC(self):

        assert self.method == 'FDTD'

        odd = slice(1,-1,2)
        psiidx = [slice(0,self.npml-1), -1, slice(0,None)]
        myidx  = [slice(-self.npml,-1), -1, slice(0,None)]

        CHz2 =	(-2*self.dt) / (2.*self.mu_Hz[myidx] + self.mcon_Hz[myidx]*self.dt)

        self.psi_hzx_p[psiidx] = (self.PMLbx[odd,None] * self.psi_hzx_p[psiidx]) + (self.PMLax[odd,None] * self.diffxEy[myidx])
        self.Hz[myidx] += CHz2 * (+((1./self.PMLkappax[odd,None] - 1.) * self.diffxEy[myidx]) + self.psi_hzx_p[psiidx])

    def _pxPML_mzPBC(self):

        assert self.method == 'FDTD'

        even = slice(0,None,2)
        psiidx = [slice(0,self.npml), slice(0,None), 0]
        myidx = [slice(-self.npml,None), slice(0,None), 0]

        CEy2 = (2.*self.dt) / (2.*self.eps_Ey[myidx] + self.econ_Ey[myidx]*self.dt)

        # PEC condition.
        pec = self.eps_Ey[myidx] > 1e3
        CEy2[pec] = 0.

        self.psi_eyx_p[psiidx] = (self.PMLbx[even,None] * self.psi_eyx_p[psiidx]) + (self.PMLax[even,None] * self.diffxHz[myidx])
        self.Ey[myidx] += CEy2 * (-(1./self.PMLkappax[even,None] - 1.) * self.diffxHz[myidx] - self.psi_eyx_p[psiidx])

    def _pxPML_pzPBC(self):

        assert self.method == 'FDTD'

        odd = slice(1,-1,2)
        psiidx = [slice(0,self.npml-1), slice(0,None), -1]
        myidx  = [slice(-self.npml,-1), slice(0,None), -1]

        CHy2 =	(-2*self.dt) / (2.*self.mu_Hy[myidx] + self.mcon_Hy[myidx]*self.dt)

        self.psi_hyx_p[psiidx] = (self.PMLbx[odd,None] * self.psi_hyx_p[psiidx]) + (self.PMLax[odd,None] * self.diffxEz[myidx])
        self.Hy[myidx] += CHy2 * (-((1./self.PMLkappax[odd,None] - 1.) * self.diffxEz[myidx]) - self.psi_hyx_p[psiidx])

    def memloc_field_at_point(self, loc):

        self.point_loc = (round(loc[0]/self.dx), round(loc[1]/self.dy), round(loc[2]/self.dz))

        x = self.point_loc[0]
        y = self.point_loc[1]
        z = self.point_loc[2]

        xsrt = self.myNx_indice[self.MPIrank][0]
        xend = self.myNx_indice[self.MPIrank][1]

        if x >= xsrt and x < xend:

            self.field_at_point = self.xp.zeros(self.tsteps)

            return self.field_at_point

        else: return None

    def get_field_at_point(self, field_at_point, field, tstep):

        if type(field_at_point) != None: 

            x = self.point_loc[0]
            y = self.point_loc[1]
            z = self.point_loc[2]

            xsrt = self.myNx_indice[self.MPIrank][0]

            myx = x - xsrt
            field_at_point[tstep] = field[myx,y,z]

        else: pass
    """


class Empty3D(Basic3D):
    
    def __init__(self, grid, gridgap, dt, tsteps, field_dtype, mmtdtype, **kwargs):

        Basic3D.__init__(self, grid, gridgap, dt, tsteps, field_dtype, mmtdtype, **kwargs)

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
