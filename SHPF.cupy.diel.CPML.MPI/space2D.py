import os
from mpi4py import MPI
from scipy.constants import c, mu_0, epsilon_0
import numpy as np
import cupy as cp

class Basic2D:
    
    def __init__(self, mode, grid, gridgap, dt, tsteps, field_dtype, mmtdtype, **kwargs):
        """Create Simulation Space.

        PARAMETERS
        ----------
        mode: string.
            Choose between 'TM' or 'TE'.

        grid: tuple
            define Nx, Ny and Nz.

        gridgap: tuple
            define the dx, dy and dz.

        field_dtype: class numpy dtype
            dtype for field array.

        mmtdtype: class numpy dtype
            dtype for FFT momentum vector array.

        kwargs: string
            
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

        assert len(grid)    == 2, "Simulation grid should be a tuple with length 2."
        assert len(gridgap) == 2, "Argument 'gridgap' should be a tuple with length 2."

        self.mode = mode
        self.tsteps = tsteps        
        self.dimension = 2
        self.grid = grid
        self.gridgap = gridgap

        self.Nx = grid[0] 
        self.Ny = grid[1]

        self.TOTAL_NUM_GRID = self.Nx * self.Ny
        self.TOTAL_NUM_GRID_SIZE = (self.field_dtype(1).nbytes * self.TOTAL_NUM_GRID) / 1024 / 1024
        
        self.Nxc = int(round(self.Nx / 2))
        self.Nyc = int(round(self.Ny / 2))
        
        self.dx = gridgap[0]
        self.dy = gridgap[1]

        self.Lx = self.Nx * self.dx
        self.Ly = self.Ny * self.dy

        self.VOLUME = self.Lx * self.Ly

        if self.MPIrank == 0:

            print("VOLUME of the space: {:.2e}" .format(self.VOLUME))
            print("Number of grid points: {:5d} x {:5d}" .format(self.Nx, self.Ny))
            print("Grid spacing: {:.3f} nm, {:.3f} nm" .format(self.dx/self.nm, self.dy/self.nm))

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
        self.maxdt = 1. / c / self.xp.sqrt( (1./self.dx)**2 + (1./self.dy)**2 )

        assert (c * self.dt * self.xp.sqrt( (1./self.dx)**2 + (1./self.dy)**2 )) < 1.

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
        self.loc_grid = (self.myNx, self.Ny)

        if self.mode == 'TM':
            self.Ez = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
            self.Hx = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
            self.Hy = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)

        if self.mode == 'TE':
            self.Ex = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
            self.Ey = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
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

        if self.engine == 'cupy':

            self.ikx = (1j*self.kx[:,None]).astype(self.mmtdtype)
            self.iky = (1j*self.ky[None,:]).astype(self.mmtdtype)

            self.xpshift = self.xp.exp(self.ikx*+self.dx/2).astype(self.mmtdtype)
            self.xmshift = self.xp.exp(self.ikx*-self.dx/2).astype(self.mmtdtype)

            self.ypshift = self.xp.exp(self.iky*+self.dy/2).astype(self.mmtdtype)
            self.ymshift = self.xp.exp(self.iky*-self.dy/2).astype(self.mmtdtype)

        else:

            nax = np.newaxis
            self.ikx = 1j*self.ky[:,nax]
            self.iky = 1j*self.ky[nax,:]
            self.xpshift = self.xp.exp(self.ikx*+self.dx/2)[:,None]
            self.xmshift = self.xp.exp(self.ikx*-self.dx/2)[:,None]
            self.ypshift = self.xp.exp(self.iky*+self.dy/2)[None,:]
            self.ymshift = self.xp.exp(self.iky*-self.dy/2)[None,:]

        if self.mode == 'TM':

            self.diffxEz = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
            self.diffyEz = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)

            self.diffxHy = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
            self.diffyHx = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)

            self.eps_Ez = self.xp.ones(self.loc_grid, dtype=self.field_dtype) * epsilon_0
            self.mu_Hx  = self.xp.ones(self.loc_grid, dtype=self.field_dtype) * mu_0
            self.mu_Hy  = self.xp.ones(self.loc_grid, dtype=self.field_dtype) * mu_0

            self.econ_Ez = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
            self.mcon_Hx = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
            self.mcon_Hy = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)

        if self.mode == 'TE':

            self.diffxEy = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
            self.diffyEx = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)

            self.diffxHz = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
            self.diffyHz = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)

            self.eps_Ex = self.xp.ones(self.loc_grid, dtype=self.field_dtype) * epsilon_0
            self.eps_Ey = self.xp.ones(self.loc_grid, dtype=self.field_dtype) * epsilon_0
            self.mu_Hz  = self.xp.ones(self.loc_grid, dtype=self.field_dtype) * mu_0

            self.econ_Ex = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
            self.econ_Ey = self.xp.zeros(self.loc_grid, dtype=self.field_dtype)
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

        self.PMLsigmamaxx = -(self.gO+1) * self.xp.log(self.rc0) / (2*self.imp*self.bdw_x)
        self.PMLsigmamaxy = -(self.gO+1) * self.xp.log(self.rc0) / (2*self.imp*self.bdw_y)

        self.PMLkappamaxx = 1.
        self.PMLkappamaxy = 1.

        self.PMLalphamaxx = 0.02
        self.PMLalphamaxy = 0.02

        self.PMLsigmax = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)
        self.PMLalphax = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)
        self.PMLkappax = self.xp.ones (self.PMLgrading, dtype=self.field_dtype)

        self.PMLsigmay = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)
        self.PMLalphay = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)
        self.PMLkappay = self.xp.ones (self.PMLgrading, dtype=self.field_dtype)

        self.PMLbx = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)
        self.PMLby = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)

        self.PMLax = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)
        self.PMLay = self.xp.zeros(self.PMLgrading, dtype=self.field_dtype)

        #------------------------------------------------------------------------------------------------#
        #------------------------------- Grading kappa, sigma and alpha ---------------------------------#
        #------------------------------------------------------------------------------------------------#

        for key, value in self.PMLregion.items():

            if key == 'x' and value != '':

                # PML for TM mode.
                if self.mode == 'TM':
                    self.psi_ezx_m = self.xp.zeros((npml, self.Ny), dtype=self.field_dtype)
                    self.psi_ezx_p = self.xp.zeros((npml, self.Ny), dtype=self.field_dtype)
                    self.psi_hyx_m = self.xp.zeros((npml, self.Ny), dtype=self.field_dtype)
                    self.psi_hyx_p = self.xp.zeros((npml, self.Ny), dtype=self.field_dtype)

                # PML for TE mode.
                if self.mode == 'TE':
                    self.psi_hzx_m = self.xp.zeros((npml, self.Ny), dtype=self.field_dtype)
                    self.psi_hzx_p = self.xp.zeros((npml, self.Ny), dtype=self.field_dtype)
                    self.psi_eyx_m = self.xp.zeros((npml, self.Ny), dtype=self.field_dtype)
                    self.psi_eyx_p = self.xp.zeros((npml, self.Ny), dtype=self.field_dtype)

                loc = self.xp.arange(self.PMLgrading) * self.dx / self.bdw_x
                self.PMLsigmax = self.PMLsigmamaxx * (loc **self.gO)
                self.PMLkappax = 1 + ((self.PMLkappamaxx-1) * (loc **self.gO))
                self.PMLalphax = self.PMLalphamaxx * ((1-loc) **self.sO)

            elif key == 'y' and value != '':

                # PML for TM mode.
                if self.mode == 'TM':
                    self.psi_hxy_p = self.xp.zeros((self.myNx, npml), dtype=self.field_dtype)
                    self.psi_hxy_m = self.xp.zeros((self.myNx, npml), dtype=self.field_dtype)
                    self.psi_ezy_p = self.xp.zeros((self.myNx, npml), dtype=self.field_dtype)
                    self.psi_ezy_m = self.xp.zeros((self.myNx, npml), dtype=self.field_dtype)

                # PML for TE mode.
                if self.mode == 'TE':
                    self.psi_exy_m = self.xp.zeros((self.myNx, npml), dtype=self.field_dtype)
                    self.psi_exy_p = self.xp.zeros((self.myNx, npml), dtype=self.field_dtype)
                    self.psi_hzy_m = self.xp.zeros((self.myNx, npml), dtype=self.field_dtype)
                    self.psi_hzy_p = self.xp.zeros((self.myNx, npml), dtype=self.field_dtype)

                loc  = self.xp.arange(self.PMLgrading) * self.dy / self.bdw_y
                self.PMLsigmay = self.PMLsigmamaxy * (loc **self.gO)
                self.PMLkappay = 1 + ((self.PMLkappamaxy-1) * (loc **self.gO))
                self.PMLalphay = self.PMLalphamaxy * ((1-loc) **self.sO)

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
            mu_Hx  = cp.asnumpy(self. mu_Hx)
            mu_Hy  = cp.asnumpy(self. mu_Hy)

        else:

            eps_Ex = self.eps_Ex
            eps_Ey = self.eps_Ey
            mu_Hx  = self. mu_Hx
            mu_Hy  = self. mu_Hy

        f.create_dataset('eps_Ex',  data=eps_Ex)
        f.create_dataset('eps_Ey',  data=eps_Ey)
        f.create_dataset( 'mu_Hx',  data=mu_Hx)
        f.create_dataset( 'mu_Hy',  data=mu_Hy)
            
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

        if self.method == 'FDTD':
            
            if self.apply_BBCx == True: assert self.MPIsize == 1

        elif self.method == 'SHPF':

            raise ValueError("BBC for 2D is not developed yet!")

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

            if self.apply_BBCx == False and self.apply_BBCy == False: self.BBC_called == False

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

        if self.apply_PBCx == True: assert self.MPIsize == 1
        if self.apply_PBCx == False and self.apply_PBCy == False: self.PBC_called == False

        return

    def updateH(self,tstep) :

        #-----------------------------------------------------------#
        #---------------------- Get derivatives --------------------#
        #-----------------------------------------------------------#

        if self.method == 'SHPF':

            # To update Hx
            self.diffyEz = self.ifft(self.iky*self.ypshift*self.fft(self.Ez, axes=(1,)), axes=(1,))
            self.diffzEy = self.ifft(self.ikz*self.zpshift*self.fft(self.Ey, axes=(2,)), axes=(2,))

            # To update Hy
            self.diffzEx = self.ifft(self.ikz*self.zpshift*self.fft(self.Ex, axes=(2,)), axes=(2,))
            self.diffxEz[:-1,:] = (self.Ez[1:,:] - self.Ez[:-1,:]) / self.dx

            # To update Hz
            self.diffyEx = self.ifft(self.iky*self.ypshift*self.fft(self.Ex, axes=(1,)), axes=(1,))
            self.diffxEy[:-1,:] = (self.Ey[1:,:] - self.Ey[:-1,:]) / self.dx

        if self.method == 'SPSTD':

            if self.mode == 'TM':

                # To update Hx
                self.diffyEz = self.ifft(self.iky*self.ypshift*self.fft(self.Ez, axes=(1,)), axes=(1,))

                # To update Hy
                self.diffxEz = self.ifft(self.ikx*self.xpshift*self.fft(self.Ez, axes=(0,)), axes=(0,))

            if self.mode == 'TE':

                # To update Hz
                self.diffyEx = self.ifft(self.iky*self.ypshift*self.fft(self.Ex, axes=(1,)), axes=(1,))
                self.diffxEy = self.ifft(self.ikx*self.yxshift*self.fft(self.Ey, axes=(0,)), axes=(0,))

        if self.method == 'PSTD':

            if self.mode == 'TM':

                # To update Hx
                self.diffyEz = self.ifft(self.iky*self.fft(self.Ez, axes=(1,)), axes=(1,))

                # To update Hy
                self.diffxEz = self.ifft(self.ikx*self.fft(self.Ez, axes=(0,)), axes=(0,))

            if self.mode == 'TE':

                # To update Hz
                self.diffyEx = self.ifft(self.iky*self.fft(self.Ex, axes=(1,)), axes=(1,))
                self.diffxEy = self.ifft(self.ikx*self.fft(self.Ey, axes=(0,)), axes=(0,))

        if self.method == 'FDTD':

            # For TM mode.
            if self.mode == 'TM':

                # To update Hx
                self.diffyEz[:,:-1] = (self.Ez[:,1:] - self.Ez[:,:-1]) / self.dy

                # To update Hy
                self.diffxEz[:-1,:] = (self.Ez[1:,:] - self.Ez[:-1,:]) / self.dx

            # For TE mode.
            if self.mode == 'TE':

                # To update Hz
                self.diffyEx[:-1,:-1] = (self.Ex[:-1,1:] - self.Ex[:-1,:-1]) / self.dy
                self.diffxEy[:-1,:-1] = (self.Ey[1:,:-1] - self.Ey[:-1,:-1]) / self.dx

        #-----------------------------------------------------------#
        #--------------- Cast basic update equations ---------------#
        #-----------------------------------------------------------#

        if self.method == 'PSTD' or self.method == 'SPSTD':

            if self.mode == 'TM':

                CHx1 = (2.*self.mu_Hx - self.mcon_Hx*self.dt) / \
                       (2.*self.mu_Hx + self.mcon_Hx*self.dt)
                CHx2 = (-2*self.dt) / (2.*self.mu_Hx + self.mcon_Hx*self.dt)

                CHy1 = (2.*self.mu_Hy - self.mcon_Hy*self.dt) / \
                       (2.*self.mu_Hy + self.mcon_Hy*self.dt)
                CHy2 = (-2*self.dt) / (2.*self.mu_Hy + self.mcon_Hy*self.dt)

                self.Hx = CHx1*self.Hx + CHx2*(+self.diffyEz)
                self.Hy = CHy1*self.Hy + CHy2*(-self.diffxEz)

            if self.mode == 'TE':

                CHz1 = (2.*self.mu_Hz - self.mcon_Hz*self.dt) / \
                       (2.*self.mu_Hz + self.mcon_Hz*self.dt)
                CHz2 = (-2*self.dt) / (2.*self.mu_Hz + self.mcon_Hz*self.dt)

                self.Hz = CHz1*self.Hz + CHz2*(self.diffxEy - self.diffyEx)

        if self.method == 'SHPF':

            if self.mode == 'TM':

                CHx1 = (2.*self.mu_Hx[:,:] - self.mcon_Hx[:,:]*self.dt) / \
                       (2.*self.mu_Hx[:,:] + self.mcon_Hx[:,:]*self.dt)
                CHx2 = (-2*self.dt) / (2.*self.mu_Hx[:,:] + self.mcon_Hx[:,:]*self.dt)

                CHy1 = (2.*self.mu_Hy[:-1,:] - self.mcon_Hy[:-1,:]*self.dt) / \
                       (2.*self.mu_Hy[:-1,:] + self.mcon_Hy[:-1,:]*self.dt)
                CHy2 = (-2*self.dt) / (2.*self.mu_Hy[:-1,:] + self.mcon_Hy[:-1,:]*self.dt)

                self.Hx[:  ,:] = CHx1*self.Hx[:  ,:] + CHx2*(+self.diffyEz[:  ,:])
                self.Hy[:-1,:] = CHy1*self.Hy[:-1,:] + CHy2*(-self.diffxEz[:-1,:])

            if self.mode == 'TE':

                CHz1 = (2.*self.mu_Hz[:-1,:] - self.mcon_Hz[:-1,:]*self.dt) / \
                       (2.*self.mu_Hz[:-1,:] + self.mcon_Hz[:-1,:]*self.dt)
                CHz2 = (-2*self.dt) / (2.*self.mu_Hz[:-1,:] + self.mcon_Hz[:-1,:]*self.dt)

                self.Hz[:-1,:] = CHz1*self.Hz[:-1,:] + CHz2*(self.diffxEy[:-1,:]-self.diffyEx[:-1,:])

        if self.method == 'FDTD':

            if self.mode == 'TM':

                CHx1 = (2.*self.mu_Hx - self.mcon_Hx*self.dt) / \
                       (2.*self.mu_Hx + self.mcon_Hx*self.dt)
                CHx2 = (-2*self.dt) / (2.*self.mu_Hx + self.mcon_Hx*self.dt)

                CHy1 = (2.*self.mu_Hy - self.mcon_Hy*self.dt) / \
                       (2.*self.mu_Hy + self.mcon_Hy*self.dt)
                CHy2 = (-2*self.dt) / (2.*self.mu_Hy + self.mcon_Hy*self.dt)

                self.Hx = CHx1*self.Hx + CHx2*(+self.diffyEz)
                self.Hy = CHy1*self.Hy + CHy2*(-self.diffxEz)

                """
                CHx1 = (2.*self.mu_Hx[:,:-1] - self.mcon_Hx[:,:-1]*self.dt) / \
                       (2.*self.mu_Hx[:,:-1] + self.mcon_Hx[:,:-1]*self.dt)
                CHx2 = (-2*self.dt) / (2.*self.mu_Hx[:,:-1] + self.mcon_Hx[:,:-1]*self.dt)

                CHy1 = (2.*self.mu_Hy[:-1,:] - self.mcon_Hy[:-1,:]*self.dt) / \
                       (2.*self.mu_Hy[:-1,:] + self.mcon_Hy[:-1,:]*self.dt)
                CHy2 = (-2*self.dt) / (2.*self.mu_Hy[:-1,:] + self.mcon_Hy[:-1,:]*self.dt)

                self.Hx[:,:-1] = CHx1*self.Hx[:,:-1] + CHx2*(+self.diffyEz[:,:-1])
                self.Hy[:-1,:] = CHy1*self.Hy[:-1,:] + CHy2*(-self.diffxEz[:-1,:])
                """

            if self.mode == 'TE':

                CHz1 = (2.*self.mu_Hz[:-1,:-1] - self.mcon_Hz[:-1,:-1]*self.dt) / \
                       (2.*self.mu_Hz[:-1,:-1] + self.mcon_Hz[:-1,:-1]*self.dt)
                CHz2 = (-2*self.dt) / (2.*self.mu_Hz[:-1,:-1] + self.mcon_Hz[:-1,:-1]*self.dt)

                self.Hz[:-1,:-1] = CHz1*self.Hz[:-1,:-1] + CHz2*(self.diffxEy[:-1,:-1]-self.diffyEx[:-1,:-1])

        #----------------------------------------------------------------------#
        #--------------------------- Apply PBC or BBC -------------------------#
        #----------------------------------------------------------------------#

        if self.method == 'FDTD' and self.BBC_called == True: self._updateH_BBC_FDTD()
        if self.method == 'FDTD' and self.PBC_called == True: self._updateH_PBC_FDTD()
        if self.method == 'PSTD' and self.BBC_called == True: self._updateH_BBC_PSTD()
        if self.method == 'SPSTD' and self.BBC_called == True: self._updateH_BBC_SPSTD()

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

        #-----------------------------------------------------------#
        #---------------------- Get derivatives --------------------#
        #-----------------------------------------------------------#

        if self.method == 'PSTD':

            if self.mode == 'TM':

                # Get derivatives of Hx and Hy to update Ez
                self.diffyHx = self.ifft(self.iky*self.fft(self.Hx, axes=(1,)), axes=(1,))
                self.diffxHy = self.ifft(self.ikx*self.fft(self.Hy, axes=(0,)), axes=(0,))

            if self.mode == 'TE':

                # Get derivatives of Hy and Hz to update Ex
                self.diffyHz = self.ifft(self.iky*self.fft(self.Hz, axes=(1,)), axes=(1,))

                # Get derivatives of Hx and Hz to update Ey
                self.diffxHz = self.ifft(self.ikx*self.fft(self.Hz, axes=(0,)), axes=(0,))

        if self.method == 'SPSTD':

            if self.mode == 'TM':

                # Get derivatives of Hx and Hy to update Ez
                self.diffyHx = self.ifft(self.iky*self.ymshift*self.fft(self.Hx, axes=(1,)), axes=(1,))
                self.diffxHy = self.ifft(self.ikx*self.xmshift*self.fft(self.Hy, axes=(0,)), axes=(0,))

            if self.mode == 'TE':

                # Get derivatives of Hy and Hz to update Ex
                self.diffyHz = self.ifft(self.iky*self.ymshift*self.fft(self.Hz, axes=(1,)), axes=(1,))

                # Get derivatives of Hx and Hz to update Ey
                self.diffxHz = self.ifft(self.ikx*self.xmshift*self.fft(self.Hz, axes=(0,)), axes=(0,))

        if self.method == 'SHPF':

            if self.mode == 'TM':

                # Get derivatives of Hx and Hy to update Ez
                self.diffyHx = self.ifft(self.iky*self.ymshift*self.fft(self.Hx, axes=(1,)), axes=(1,))
                self.diffxHy[1:,:] = (self.Hy[1:,:] - self.Hy[:-1,:]) / self.dx

            if self.mode == 'TE':

                # Get derivatives of Hy and Hz to update Ex
                self.diffyHz = self.ifft(self.iky*self.ymshift*self.fft(self.Hz, axes=(1,)), axes=(1,))

                # Get derivatives of Hx and Hz to update Ey
                self.diffxHz[1:,:] = (self.Hz[1:,:] - self.Hz[:-1,:]) / self.dx

        if self.method == 'FDTD':

            if self.mode == 'TM':

                # Get derivatives of Hx and Hy to update Ez
                self.diffyHx[1:,1:] = (self.Hx[1:,1:] - self.Hx[1:,:-1]) / self.dy
                self.diffxHy[1:,1:] = (self.Hy[1:,1:] - self.Hy[:-1,1:]) / self.dx

            if self.mode == 'TE':

                # Get derivatives of Hy and Hz to update Ex
                self.diffyHz[:,1:] = (self.Hz[:,1:] - self.Hz[:,:-1]) / self.dy

                # Get derivatives of Hx and Hz to update Ey
                self.diffxHz[1:,:] = (self.Hz[1:,:] - self.Hz[:-1,:]) / self.dx

        #-----------------------------------------------------------#
        #--------------- Cast basic update equations ---------------#
        #-----------------------------------------------------------#

        # Update Ex, Ey, Ez
        if self.method == 'PSTD' or self.method == 'SPSTD':

            if self.mode == 'TM':

                CEz1 = (2.*self.eps_Ez-self.econ_Ez*self.dt) / \
                       (2.*self.eps_Ez+self.econ_Ez*self.dt)
                CEz2 = (2.*self.dt) / (2.*self.eps_Ez+self.econ_Ez*self.dt)

                # PEC condition.
                CEz1[self.eps_Ez > 1e3] = 0.
                CEz2[self.eps_Ez > 1e3] = 0.

                self.Ez = CEz1 * self.Ez + CEz2 * (self.diffxHy - self.diffyHx)

            if self.mode == 'TE':

                CEx1 = (2.*self.eps_Ex-self.econ_Ex*self.dt) / \
                       (2.*self.eps_Ex+self.econ_Ex*self.dt)
                CEx2 = (2.*self.dt) / (2.*self.eps_Ex+self.econ_Ex*self.dt)

                CEy1 = (2.*self.eps_Ey-self.econ_Ey*self.dt) / \
                       (2.*self.eps_Ey+self.econ_Ey*self.dt)
                CEy2 = (2.*self.dt) / (2.*self.eps_Ey+self.econ_Ey*self.dt)

                # PEC condition.
                CEx1[self.eps_Ex > 1e3] = 0.
                CEx2[self.eps_Ex > 1e3] = 0.
                CEy1[self.eps_Ey > 1e3] = 0.
                CEy2[self.eps_Ey > 1e3] = 0.

                self.Ex = CEx1 * self.Ex + CEx2 * (self.diffyHz - self.diffzHy)
                self.Ey = CEy1 * self.Ey + CEy2 * (self.diffzHx - self.diffxHz)

        if self.method == 'SHPF':

            if self.mode == 'TM':

                CEz1 = (2.*self.eps_Ez[1:,:]-self.econ_Ez[1:,:]*self.dt) / \
                       (2.*self.eps_Ez[1:,:]+self.econ_Ez[1:,:]*self.dt)
                CEz2 = (2.*self.dt) / (2.*self.eps_Ez[1:,:]+self.econ_Ez[1:,:]*self.dt)

                # PEC condition.
                CEz1[self.eps_Ez[1:,:] > 1e3] = 0.
                CEz2[self.eps_Ez[1:,:] > 1e3] = 0.

                self.Ez[1:,:] = CEz1 * self.Ez[1:,:] + CEz2 * (self.diffxHy[1:,:] - self.diffyHx[1:,:])

            if self.mode == 'TE':

                CEx1 = (2.*self.eps_Ex[:,:]-self.econ_Ex[:,:]*self.dt) / \
                       (2.*self.eps_Ex[:,:]+self.econ_Ex[:,:]*self.dt)
                CEx2 = (2.*self.dt) / (2.*self.eps_Ex[:,:]+self.econ_Ex[:,:]*self.dt)

                CEy1 = (2.*self.eps_Ey[1:,:]-self.econ_Ey[1:,:]*self.dt) / \
                       (2.*self.eps_Ey[1:,:]+self.econ_Ey[1:,:]*self.dt)
                CEy2 = (2.*self.dt) / (2.*self.eps_Ey[1:,:]+self.econ_Ey[1:,:]*self.dt)

                # PEC condition.
                CEx1[self.eps_Ex[ :,:] > 1e3] = 0.
                CEx2[self.eps_Ex[ :,:] > 1e3] = 0.
                CEy1[self.eps_Ey[1:,:] > 1e3] = 0.
                CEy2[self.eps_Ey[1:,:] > 1e3] = 0.

                self.Ex[: ,:] = CEx1 * self.Ex[ :,:] + CEx2 * (self.diffyHz[ :,:] - self.diffzHy[ :,:])
                self.Ey[1:,:] = CEy1 * self.Ey[1:,:] + CEy2 * (self.diffzHx[1:,:] - self.diffxHz[1:,:])

        if self.method == 'FDTD':

            if self.mode == 'TM':

                CEz1 = (2.*self.eps_Ez-self.econ_Ez*self.dt) / \
                       (2.*self.eps_Ez+self.econ_Ez*self.dt)
                CEz2 = (2.*self.dt) / (2.*self.eps_Ez+self.econ_Ez*self.dt)

                # PEC condition.
                CEz1[self.eps_Ez > 1e3] = 0.
                CEz2[self.eps_Ez > 1e3] = 0.

                self.Ez = CEz1 * self.Ez + CEz2 * (self.diffxHy - self.diffyHx)

                """
                CEz1 = (2.*self.eps_Ez[1:,1:]-self.econ_Ez[1:,1:]*self.dt) / \
                       (2.*self.eps_Ez[1:,1:]+self.econ_Ez[1:,1:]*self.dt)
                CEz2 = (2.*self.dt) / (2.*self.eps_Ez[1:,1:]+self.econ_Ez[1:,1:]*self.dt)

                # PEC condition.
                CEz1[self.eps_Ez[1:,1:] > 1e3] = 0.
                CEz2[self.eps_Ez[1:,1:] > 1e3] = 0.

                self.Ez[1:,1:] = CEz1 * self.Ez[1:,1:] + CEz2 * (self.diffxHy[1:,1:] - self.diffyHx[1:,1:])
                """

            if self.mode == 'TE':

                CEx1 = (2.*self.eps_Ex[:,1:]-self.econ_Ex[:,1:]*self.dt) / \
                       (2.*self.eps_Ex[:,1:]+self.econ_Ex[:,1:]*self.dt)
                CEx2 = (2.*self.dt) / (2.*self.eps_Ex[:,1:]+self.econ_Ex[:,1:]*self.dt)

                CEy1 = (2.*self.eps_Ey[1:,:]-self.econ_Ey[1:,:]*self.dt) / \
                       (2.*self.eps_Ey[1:,:]+self.econ_Ey[1:,:]*self.dt)
                CEy2 = (2.*self.dt) / (2.*self.eps_Ey[1:,:]+self.econ_Ey[1:,:]*self.dt)

                # PEC condition.
                CEx1[self.eps_Ex[:,1:] > 1e3] = 0.
                CEx2[self.eps_Ex[:,1:] > 1e3] = 0.
                CEy1[self.eps_Ey[1:,:] > 1e3] = 0.
                CEy2[self.eps_Ey[1:,:] > 1e3] = 0.

                self.Ex[:,1:] = CEx1 * self.Ex[:,1:] + CEx2 * (self.diffyHz[:,1:] - self.diffzHy[:,1:])
                self.Ey[1:,:] = CEy1 * self.Ey[1:,:] + CEy2 * (self.diffzHx[1:,:] - self.diffxHz[1:,:])

        #----------------------------------------------------------------------#
        #--------------------------- Apply PBC or BBC -------------------------#
        #----------------------------------------------------------------------#

        if self.method == 'FDTD' and self.BBC_called == True: self._updateE_BBC_FDTD()
        if self.method == 'FDTD' and self.PBC_called == True: self._updateE_PBC_FDTD()
        if self.method == 'SHPF' and self.PBC_called == True: self._updateE_PBC_SHPF()
        if self.method == 'SHPF' and self.BBC_called == True: self._updateE_BBC_SHPF()
        if self.method == 'PSTD' and self.BBC_called == True: self._updateE_BBC_PSTD()
        if self.method == 'SPSTD' and self.BBC_called == True: self._updateE_BBC_SPSTD()

        #-----------------------------------------------------------#
        #---------------- Apply PML when it is given ---------------#
        #-----------------------------------------------------------#

        self._updateE_PML()

    def _updateH_PML(self):

        if 'x' in self.PMLregion.keys():
            if '+' in self.PMLregion.get('x'): self._PML_updateH_px()
            if '-' in self.PMLregion.get('x'): self._PML_updateH_mx()

        if 'y' in self.PMLregion.keys():
            if '+' in self.PMLregion.get('y'): self._PML_updateH_py()
            if '-' in self.PMLregion.get('y'): self._PML_updateH_my()

    def _updateE_PML(self):

        if 'x' in self.PMLregion.keys():
            if '+' in self.PMLregion.get('x'): self._PML_updateE_px()
            if '-' in self.PMLregion.get('x'): self._PML_updateE_mx()

        if 'y' in self.PMLregion.keys():
            if '+' in self.PMLregion.get('y'): self._PML_updateE_py()
            if '-' in self.PMLregion.get('y'): self._PML_updateE_my()

    def _PML_updateH_px(self):

        if self.method == 'PSTD' or self.method == 'SPSTD':

            odd = [slice(0,None,2), None]

            psiidx_Hyxp = [slice(0,None), slice(0,None)]
            myidx_Hyxp = [slice(-self.npml,None), slice(0,None)]

            psiidx_Hzxp = [slice(0,None), slice(0,None)]
            myidx_Hzxp = [slice(-self.npml,None), slice(0,None)]

        if self.method == 'SHPF':

            odd = [slice(1,-1,2), None]

            psiidx_Hyxp = [slice(0,-1), slice(0,None)]
            myidx_Hyxp = [slice(-self.npml,-1), slice(0,None)]

            psiidx_Hzxp = [slice(0,-1), slice(0,None)]
            myidx_Hzxp = [slice(-self.npml,-1), slice(0,None)]

        if self.method == 'FDTD':

            odd = [slice(1,-1,2), None]

            psiidx_Hyxp = [slice(0,-1), slice(0,None)]
            myidx_Hyxp = [slice(-self.npml,-1), slice(0,None)]

            psiidx_Hzxp = [slice(0,-1), slice(0,-1)]
            myidx_Hzxp = [slice(-self.npml,-1), slice(0,-1)]

        if self.mode == 'TM':

            # Update Hy at x+.
            CHy2 = (-2*self.dt) / (2.*self.mu_Hy[myidx_Hyxp] + self.mcon_Hy[myidx_Hyxp]*self.dt)
            self.psi_hyx_p[psiidx_Hyxp] = (self.PMLbx[odd]*self.psi_hyx_p[psiidx_Hyxp]) \
                                        + (self.PMLax[odd]*self.diffxEz[myidx_Hyxp])
            self.Hy[myidx_Hyxp] += CHy2*(-((1./self.PMLkappax[odd] - 1.)*self.diffxEz[myidx_Hyxp]) - self.psi_hyx_p[psiidx_Hyxp])

        if self.mode == 'TE':

            # Update Hz at x+.
            CHz2 = (-2*self.dt) / (2.*self.mu_Hz[myidx_Hzxp] + self.mcon_Hz[myidx_Hzxp]*self.dt)
            self.psi_hzx_p[psiidx_Hzxp] = (self.PMLbx[odd]*self.psi_hzx_p[psiidx_Hzxp]) \
                                        + (self.PMLax[odd]*self.diffxEy[myidx_Hzxp])
            self.Hz[myidx_Hzxp] += CHz2*(+((1./self.PMLkappax[odd]-1.)*self.diffxEy[myidx_Hzxp]) + self.psi_hzx_p[psiidx_Hzxp])

    def _PML_updateE_px(self):

        if self.method == 'SHPF' or self.method == 'PSTD' or self.method == 'SPSTD':

            even = [slice(0,None,2), None]

            psiidx_Eyxp = [slice(0,None), slice(0,None)]
            myidx_Eyxp = [slice(-self.npml,None), slice(0,None)]

            psiidx_Ezxp = [slice(0,None), slice(0,None)]
            myidx_Ezxp = [slice(-self.npml,None), slice(0,None)]

        if self.method == 'FDTD':

            even = [slice(0,None,2), None]

            psiidx_Eyxp = [slice(0,None), slice(0,None)]
            myidx_Eyxp = [slice(-self.npml,None), slice(0,None)]

            psiidx_Ezxp = [slice(0,None), slice(1,None)]
            myidx_Ezxp = [slice(-self.npml,None), slice(1,None)]

        if self.mode == 'TM':

            # Update Ez at x+.
            CEz2 = (2.*self.dt) / (2.*self.eps_Ez[myidx_Ezxp] + self.econ_Ez[myidx_Ezxp]*self.dt)
            self.psi_ezx_p[psiidx_Ezxp] = (self.PMLbx[even]*self.psi_ezx_p[psiidx_Ezxp])\
                                        + (self.PMLax[even]*self.diffxHy[myidx_Ezxp])
            self.Ez[myidx_Ezxp] += CEz2*(+(1./self.PMLkappax[even]-1.)*self.diffxHy[myidx_Ezxp] + self.psi_ezx_p[psiidx_Ezxp])

        if self.mode == 'TE':

            # Update Ey at x+.
            CEy2 = (2.*self.dt) / (2.*self.eps_Ey[myidx_Eyxp] + self.econ_Ey[myidx_Eyxp]*self.dt)
            self.psi_eyx_p[psiidx_Eyxp] = (self.PMLbx[even]*self.psi_eyx_p[psiidx_Eyxp])\
                                        + (self.PMLax[even]*self.diffxHz[myidx_Eyxp])
            self.Ey[myidx_Eyxp] += CEy2*(-(1./self.PMLkappax[even]-1.)*self.diffxHz[myidx_Eyxp] - self.psi_eyx_p[psiidx_Eyxp])

    def _PML_updateH_mx(self):

        if self.method == 'PSTD' or self.method == 'SPSTD':

            even = [slice(-1,None,-2), None]

            psiidx_Hyxm = [slice(0,self.npml), slice(0,None)]
            myidx_Hyxm  = [slice(0,self.npml), slice(0,None)]

            psiidx_Hzxm = [slice(0,self.npml), slice(0,None)]
            myidx_Hzxm  = [slice(0,self.npml), slice(0,None)]

        if self.method == 'SHPF':

            even = [slice(-2,None,-2), None]

            psiidx_Hyxm = [slice(0,self.npml), slice(0,None)]
            myidx_Hyxm  = [slice(0,self.npml), slice(0,None)]

            psiidx_Hzxm = [slice(0,self.npml), slice(0,None)]
            myidx_Hzxm  = [slice(0,self.npml), slice(0,None)]

        if self.method == 'FDTD':

            even = [slice(-2,None,-2), None]

            psiidx_Hyxm = [slice(0,self.npml), slice(0,None)]
            myidx_Hyxm  = [slice(0,self.npml), slice(0,None)]

            psiidx_Hzxm = [slice(0, self.npml), slice(0,-1)]
            myidx_Hzxm  = [slice(0, self.npml), slice(0,-1)]

        if self.mode == 'TM':

            # Update Hy at x-.
            CHy2 = (-2*self.dt) / (2.*self.mu_Hy[myidx_Hyxm] + self.mcon_Hy[myidx_Hyxm]*self.dt)
            self.psi_hyx_m[psiidx_Hyxm] = (self.PMLbx[even]*self.psi_hyx_m[psiidx_Hyxm])\
                                        + (self.PMLax[even]*self.diffxEz[myidx_Hyxm])
            self.Hy[myidx_Hyxm] += CHy2*(-((1./self.PMLkappax[even]-1.)*self.diffxEz[myidx_Hyxm]) - self.psi_hyx_m[psiidx_Hyxm])

        if self.mode == 'TE':

            # Update Hz at x-.
            CHz2 = (-2*self.dt) / (2.*self.mu_Hz[myidx_Hzxm] + self.mcon_Hz[myidx_Hzxm]*self.dt)
            self.psi_hzx_m[psiidx_Hzxm] = (self.PMLbx[even]*self.psi_hzx_m[psiidx_Hzxm])\
                                        + (self.PMLax[even]*self.diffxEy[myidx_Hzxm])
            self.Hz[myidx_Hzxm] += CHz2*(+((1./self.PMLkappax[even]-1.)*self.diffxEy[myidx_Hzxm]) + self.psi_hzx_m[psiidx_Hzxm])

    def _PML_updateE_mx(self):

        if self.method == 'PSTD' or self.method == 'SPSTD':

            odd = [slice(-1,None,-2),None]

            psiidx_Eymx = [slice(0,self.npml), slice(0,None)]
            myidx_Eymx  = [slice(0,self.npml), slice(0,None)]

            psiidx_Ezmx = [slice(0,self.npml), slice(0,None)]
            myidx_Ezmx  = [slice(0,self.npml), slice(0,None)]

        if self.method == 'SHPF':

            odd = [slice(-3,None,-2),None]

            psiidx_Eymx = [slice(1,self.npml), slice(0,None)]
            myidx_Eymx  = [slice(1,self.npml), slice(0,None)]

            psiidx_Ezmx = [slice(1,self.npml), slice(0,None)]
            myidx_Ezmx  = [slice(1,self.npml), slice(0,None)]

        if self.method == 'FDTD':

            odd = [slice(-3,None,-2),None]

            psiidx_Eymx = [slice(1,self.npml), slice(0,None)]
            myidx_Eymx  = [slice(1,self.npml), slice(0,None)]

            psiidx_Ezmx = [slice(1,self.npml), slice(1,None)]
            myidx_Ezmx  = [slice(1,self.npml), slice(1,None)]

        if self.mode == 'TE':

            # Update Ey at x+.
            CEy2 = (2.*self.dt) / (2.*self.eps_Ey[myidx_Eymx] + self.econ_Ey[myidx_Eymx]*self.dt)
            self.psi_eyx_m[psiidx_Eymx] = (self.PMLbx[odd]*self.psi_eyx_m[psiidx_Eymx])\
                                        + (self.PMLax[odd]*self.diffxHz[myidx_Eymx])
            self.Ey[myidx_Eymx] += CEy2*(-(1./self.PMLkappax[odd]-1.)*self.diffxHz[myidx_Eymx] - self.psi_eyx_m[psiidx_Eymx])

        if self.mode == 'TM':

            # Update Ez at x+.
            CEz2 = (2.*self.dt) / (2.*self.eps_Ez[myidx_Ezmx] + self.econ_Ez[myidx_Ezmx]*self.dt)
            self.psi_ezx_m[psiidx_Ezmx] = (self.PMLbx[odd]*self.psi_ezx_m[psiidx_Ezmx])\
                                        + (self.PMLax[odd] * self.diffxHy[myidx_Ezmx])
            self.Ez[myidx_Ezmx] += CEz2*(+(1./self.PMLkappax[odd]-1.)*self.diffxHy[myidx_Ezmx] + self.psi_ezx_m[psiidx_Ezmx])

    def _PML_updateH_py(self):

        if self.method == 'PSTD' or self.method == 'SPSTD':

            odd = [None, slice(0,None,2)]

            psiidx_Hxyp = [slice(0,None), slice(0,None)]
            myidx_Hxyp  = [slice(0,None), slice(-self.npml,None)]

            psiidx_Hzyp = [slice(0,None), slice(0,None)]
            myidx_Hzyp  = [slice(0,None), slice(-self.npml,None)]

        if self.method == 'SHPF':

            odd = [None, slice(1,None,2)]

            psiidx_Hxyp = [slice(0,None), slice(0,None)]
            myidx_Hxyp  = [slice(0,None), slice(-self.npml,None)]

            psiidx_Hzyp = [slice(0,-1), slice(0,None)]
            myidx_Hzyp  = [slice(0,-1), slice(-self.npml)]

        if self.method == 'FDTD':

            odd = [None, slice(1,-1,2)]

            psiidx_Hxyp = [slice(0,None), slice(0,-1)]
            myidx_Hxyp  = [slice(0,None), slice(-self.npml,-1)]

            psiidx_Hzyp = [slice(0,-1), slice(0,-1)]
            myidx_Hzyp  = [slice(0,-1), slice(-self.npml,-1)]

        if self.mode == 'TM':

            # Update Hx at y+.
            CHx2 = (-2.*self.dt) / (2.*self.mu_Hx[myidx_Hxyp] + self.mcon_Hx[myidx_Hxyp]*self.dt)
            self.psi_hxy_p[psiidx_Hxyp] = (self.PMLby[odd]*self.psi_hxy_p[psiidx_Hxyp])\
                                        + (self.PMLay[odd]*self.diffyEz[myidx_Hxyp])
            self.Hx[myidx_Hxyp] += CHx2*(+((1./self.PMLkappay[odd] - 1.)*self.diffyEz[myidx_Hxyp])+self.psi_hxy_p[psiidx_Hxyp])

        if self.mode == 'TE':

            # Update Hz at y+.
            CHz2 = (-2.*self.dt) / (2.*self.mu_Hz[myidx_Hzyp] + self.mcon_Hz[myidx_Hzyp]*self.dt)
            self.psi_hzy_p[psiidx_Hzyp] = (self.PMLby[odd] * self.psi_hzy_p[psiidx_Hzyp])\
                                        + (self.PMLay[odd] * self.diffyEx[myidx_Hzyp])
            self.Hz[myidx_Hzyp] += CHz2*(-((1./self.PMLkappay[odd]-1.)*self.diffyEx[myidx_Hzyp]) - self.psi_hzy_p[psiidx_Hzyp])
                
    def _PML_updateE_py(self):

        if self.method == 'PSTD' or self.method == 'SPSTD':

            even = [None,slice(0,None,2)]

            psiidx_Exyp = [slice(0,None), slice(0,self.npml)]
            myidx_Exyp  = [slice(0,None), slice(-self.npml,None)]

            psiidx_Ezyp = [slice(0,None), slice(0,self.npml)]
            myidx_Ezyp  = [slice(0,None), slice(-self.npml,None)]

        if self.method == 'SHPF':

            even = [None,slice(0,None,2)]

            psiidx_Exyp = [slice(0,None), slice(0,self.npml)]
            myidx_Exyp  = [slice(0,None), slice(-self.npml,None)]

            psiidx_Ezyp = [slice(1,None), slice(0,self.npml)]
            myidx_Ezyp  = [slice(1,None), slice(-self.npml,None)]

        if self.method == 'FDTD':
         
            even = [None,slice(0,None,2)]

            psiidx_Exyp = [slice(0,None), slice(0,self.npml)]
            myidx_Exyp  = [slice(0,None), slice(-self.npml,None)]

            psiidx_Ezyp = [slice(1,None), slice(0,self.npml)]
            myidx_Ezyp  = [slice(1,None), slice(-self.npml,None)]

        if self.mode == 'TE':

            # Update Ex at y+.
            CEx2 = (2*self.dt) / (2.*self.eps_Ex[myidx_Exyp] + self.econ_Ex[myidx_Exyp]*self.dt)
            self.psi_exy_p[psiidx_Exyp] = (self.PMLby[even]*self.psi_exy_p[psiidx_Exyp])\
                                        + (self.PMLay[even]*self.diffyHz[myidx_Exyp])
            self.Ex[myidx_Exyp] += CEx2*(+((1./self.PMLkappay[even]-1.)*self.diffyHz[myidx_Exyp]) + self.psi_exy_p[psiidx_Exyp])

        if self.mode == 'TM':

            # Update Ez at y+.
            CEz2 = (2.*self.dt) / (2.*self.eps_Ez[myidx_Ezyp] + self.econ_Ez[myidx_Ezyp]*self.dt)
            self.psi_ezy_p[psiidx_Ezyp] = (self.PMLby[even]*self.psi_ezy_p[psiidx_Ezyp])\
                                        + (self.PMLay[even]*self.diffyHx[myidx_Ezyp])
            self.Ez[myidx_Ezyp] += CEz2*(-((1./self.PMLkappay[even]-1.)*self.diffyHx[myidx_Ezyp]) - self.psi_ezy_p[psiidx_Ezyp])

    def _PML_updateH_my(self):

        if self.method == 'PSTD' or self.method == 'SPSTD':

            even = [None, slice(-1,None,-2)]

            psiidx_Hxym = [slice(0,None), slice(0,self.npml)]
            myidx_Hxym  = [slice(0,None), slice(0,self.npml)]

            psiidx_Hzym = [slice(0,None), slice(0,self.npml)]
            myidx_Hzym  = [slice(0,None), slice(0,self.npml)]

        if self.method == 'SHPF':

            even = [None, slice(-2,None,-2)]

            psiidx_Hxym = [slice(0,None), slice(0,self.npml)]
            myidx_Hxym  = [slice(0,None), slice(0,self.npml)]

            psiidx_Hzym = [slice(0,-1), slice(0,self.npml)]
            myidx_Hzym  = [slice(0,-1), slice(0,self.npml)]

        if self.method == 'FDTD':

            even = [None, slice(-2,None,-2)]

            psiidx_Hxym = [slice(0,None), slice(0,self.npml)]
            myidx_Hxym  = [slice(0,None), slice(0,self.npml)]

            psiidx_Hzym = [slice(0,-1), slice(0,self.npml)]
            myidx_Hzym  = [slice(0,-1), slice(0,self.npml)]

        if self.mode == 'TM':

            # Update Hx at y-.
            CHx2 =  (-2*self.dt) / (2.*self.mu_Hx[myidx_Hxym] + self.mcon_Hx[myidx_Hxym]*self.dt)
            self.psi_hxy_m[psiidx_Hxym] = (self.PMLby[even]*self.psi_hxy_m[psiidx_Hxym])\
                                        + (self.PMLay[even]*self.diffyEz[myidx_Hxym])
            self.Hx[myidx_Hxym] += CHx2*(+((1./self.PMLkappay[even]-1.)*self.diffyEz[myidx_Hxym]) + self.psi_hxy_m[psiidx_Hxym])

        if self.mode == 'TE':

            # Update Hz at y-.
            CHz2 =  (-2*self.dt) / (2.*self.mu_Hz[myidx_Hzym] + self.mcon_Hz[myidx_Hzym]*self.dt)
            self.psi_hzy_m[psiidx_Hzym] = (self.PMLby[even]*self.psi_hzy_m[psiidx_Hzym])\
                                        + (self.PMLay[even]*self.diffyEx[myidx_Hzym])
            self.Hz[myidx_Hzym] += CHz2*(-((1./self.PMLkappay[even]-1.)*self.diffyEx[myidx_Hzym]) - self.psi_hzy_m[psiidx_Hzym])

    def _PML_updateE_my(self):

        if self.method == 'PSTD' or self.method == 'SPSTD':

            odd = [None, slice(-1,None,-2)]

            psiidx_Exym = [slice(0,None), slice(0,self.npml)]
            myidx_Exym = [slice(0,None), slice(0,self.npml)]

            psiidx_Ezym = [slice(0,None), slice(0,self.npml)]
            myidx_Ezym = [slice(0,None), slice(0,self.npml)]

        if self.method == 'SHPF':

            odd = [None, slice(-1,None,-2)]

            psiidx_Exym = [slice(0,None), slice(0,self.npml)]
            myidx_Exym = [slice(0,None), slice(0,self.npml)]

            psiidx_Ezym = [slice(1,None), slice(0,self.npml)]
            myidx_Ezym = [slice(1,None), slice(0,self.npml)]

        if self.method == 'FDTD':

            odd = [None, slice(-3,None,-2)]

            psiidx_Exym = [slice(0,None), slice(1,self.npml)]
            myidx_Exym = [slice(0,None), slice(1,self.npml)]

            psiidx_Ezym = [slice(1,None), slice(1,self.npml)]
            myidx_Ezym = [slice(1,None), slice(1,self.npml)]

        if self.mode == 'TE':

            # Update Ex at y-.
            CEx2 = (2.*self.dt) / (2.*self.eps_Ex[myidx_Exym] + self.econ_Ex[myidx_Exym]*self.dt)
            self.psi_exy_m[psiidx_Exym] = (self.PMLby[odd]*self.psi_exy_m[psiidx_Exym])\
                                        + (self.PMLay[odd]*self.diffyHz[myidx_Exym])
            self.Ex[myidx_Exym] += CEx2*(+((1./self.PMLkappay[odd]-1.)*self.diffyHz[myidx_Exym]) + self.psi_exy_m[psiidx_Exym])

        if self.mode == 'TM':

            # Update Ez at y-.
            CEz2 = (2*self.dt) / (2.*self.eps_Ez[myidx_Ezym] + self.econ_Ez[myidx_Ezym]*self.dt)
            self.psi_ezy_m[psiidx_Ezym] = (self.PMLby[odd]*self.psi_ezy_m[psiidx_Ezym])\
                                        + (self.PMLay[odd]*self.diffyHx[myidx_Ezym])
            self.Ez[myidx_Ezym] += CEz2*(-((1./self.PMLkappay[odd]-1.)*self.diffyHx[myidx_Ezym]) - self.psi_ezy_m[psiidx_Ezym])

    def _exchange_BBCx(self, k, newL, f):

        p0 = [ 0, slice(0,None)]
        p1 = [ 1, slice(0,None)]
        m1 = [-1, slice(0,None)]
        m2 = [-2, slice(0,None)]

        f[m1] = f[p1] * self.xp.exp(+1j*k*newL) 
        f[p0] = f[m2] * self.xp.exp(-1j*k*newL)

    def _exchange_BBCy(self, k, newL, f):

        p0 = [slice(0,None), 0]
        p1 = [slice(0,None), 1]
        m1 = [slice(0,None),-1]
        m2 = [slice(0,None),-2]

        f[m1] = f[p1] * self.xp.exp(+1j*k*newL) 
        f[p0] = f[m2] * self.xp.exp(-1j*k*newL)

    def _updateE_BBC_FDTD(self):

        if self.apply_BBCx == True: 

            assert self.apply_PBCx == False
            newL = self.Lx - 2*self.dx

            if self.mode == 'TM':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCx(self.mmt[0], newL, self.Ez)

            elif self.mode == 'TE':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCx(self.mmt[0], newL, self.Ex)
                self._exchange_BBCx(self.mmt[0], newL, self.Ey)

        if self.apply_BBCy == True: 

            assert self.apply_PBCy == False
            newL = self.Ly - 2*self.dy

            if self.mode == 'TM':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCy(self.mmt[1], newL, self.Ez)

            elif self.mode == 'TE':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCy(self.mmt[1], newL, self.Ex)
                self._exchange_BBCy(self.mmt[1], newL, self.Ey)

    def _updateE_PBC_FDTD(self):

        newL = 0

        if self.apply_PBCx == True: 

            assert self.apply_BBCx == False
            if self.mode == 'TM':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCx(self.mmt[0], newL, self.Ez)

            elif self.mode == 'TE':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCx(self.mmt[0], newL, self.Ex)
                self._exchange_BBCx(self.mmt[0], newL, self.Ey)

        if self.apply_PBCy == True: 
        
            assert self.apply_BBCy == False
            if self.mode == 'TM':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCy(self.mmt[1], newL, self.Ez)

            elif self.mode == 'TE':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCy(self.mmt[1], newL, self.Ex)
                self._exchange_BBCy(self.mmt[1], newL, self.Ey)

    def _updateH_BBC_PSTD(self):

        #self.Hx[sli1] += -self.dt/self.mu_Hx[sli1]*1j*(self.mmt[1]*self.ez_at_Hx[sli1] - self.mmt[2]*self.ey_at_Hx[sli1])
        #self.Hy[sli2] += -self.dt/self.mu_Hy[sli2]*1j*(self.mmt[2]*self.ex_at_Hy[sli2] - self.mmt[0]*self.ez_at_Hy[sli2])
        #self.Hz[sli2] += -self.dt/self.mu_Hz[sli2]*1j*(self.mmt[0]*self.ey_at_Hz[sli2] - self.mmt[1]*self.ex_at_Hz[sli2])

        if self.apply_BBCx == True:

            assert self.apply_PBCx == False
            if self.mode == 'TM': self.Hy += -self.dt/self.mu_Hy*1j*(+self.mmt[0]*self.Ez)
            if self.mode == 'TE': self.Hz += -self.dt/self.mu_Hz*1j*(-self.mmt[0]*self.Ey)

        if self.apply_BBCy == True:

            assert self.apply_PBCy == False
            if self.mode == 'TM': self.Hx += -self.dt/self.mu_Hx*1j*(-self.mmt[1]*self.Ez)
            if self.mode == 'TE': self.Hz += -self.dt/self.mu_Hz*1j*(+self.mmt[1]*self.Ex)

    def _updateH_BBC_SPSTD(self):

        #self.Hx[sli1] += -self.dt/self.mu_Hx[sli1]*1j*(self.mmt[1]*self.ez_at_Hx[sli1] - self.mmt[2]*self.ey_at_Hx[sli1])
        #self.Hy[sli2] += -self.dt/self.mu_Hy[sli2]*1j*(self.mmt[2]*self.ex_at_Hy[sli2] - self.mmt[0]*self.ez_at_Hy[sli2])
        #self.Hz[sli2] += -self.dt/self.mu_Hz[sli2]*1j*(self.mmt[0]*self.ey_at_Hz[sli2] - self.mmt[1]*self.ex_at_Hz[sli2])

        if self.apply_BBCx == True:

            assert self.apply_PBCx == False
            if self.mode == 'TM': 

                self.ez_at_Hy = self.ifft(self.xpshift*self.fft(self.Ez, axes=(0,)), axes=(0,))
                self.Hy += -self.dt/self.mu_Hy*1j*(+self.mmt[0]*self.ez_at_Hy)

            if self.mode == 'TE': 

                self.ey_at_Hz = self.ifft(self.xpshift*self.fft(self.Ez, axes=(0,)), axes=(0,))
                self.Hz += -self.dt/self.mu_Hz*1j*(-self.mmt[0]*self.ey_at_Hz)

        if self.apply_BBCy == True:

            assert self.apply_PBCy == False
            if self.mode == 'TM': 

                self.ez_at_Hx = self.ifft(self.ypshift*self.fft(self.Ez, axes=(1,)), axes=(1,))
                self.Hx += -self.dt/self.mu_Hx*1j*(-self.mmt[1]*self.ez_at_Hx)

            if self.mode == 'TE': 

                self.ex_at_Hz = self.ifft(self.ypshift*self.fft(self.Ex, axes=(1,)), axes=(1,))
                self.Hz += -self.dt/self.mu_Hz*1j*(+self.mmt[1]*self.ex_at_Hz)

    def _updateH_BBC_SHPF(self):

        sli1 = [slice(None,None), slice(None,None)]
        sli2 = [slice(None,  -1), slice(None,None)]

        if self.apply_BBCx == True: 

            assert self.apply_PBCx == False
            newL = self.Lx - 2*self.dx

            if self.mode == 'TM':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCx(self.mmt[0], newL, self.Ez)

            elif self.mode == 'TE':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCx(self.mmt[0], newL, self.Ex)
                self._exchange_BBCx(self.mmt[0], newL, self.Ey)

        if self.apply_BBCy == True:

            assert self.apply_PBCy == False
            if self.mode == 'TM': self.Hx += -self.dt/self.mu_Hx*1j*(-self.mmt[1]*self.Ez)
            if self.mode == 'TE': self.Hz += -self.dt/self.mu_Hz*1j*(+self.mmt[1]*self.Ex)

    def _updateH_PBC_SHPF(self):

        sli1 = [slice(None,None), slice(None,None)]
        sli2 = [slice(None,  -1), slice(None,None)]

        newL = 0

        if self.apply_BBCx == True: 

            assert self.apply_PBCx == False

            if self.mode == 'TM':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCx(self.mmt[0], newL, self.Ez)

            elif self.mode == 'TE':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCx(self.mmt[0], newL, self.Ex)
                self._exchange_BBCx(self.mmt[0], newL, self.Ey)

    def _updateH_BBC_FDTD(self):

        if self.apply_BBCx == True: 

            assert self.apply_PBCx == False

            if self.mode == 'TE':

                newL = self.Lx - 2*self.dx

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCx(self.mmt[0], newL, self.Hz)

            elif self.mode == 'TM':

                newL = self.Lx - 2*self.dx

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCx(self.mmt[0], newL, self.Hx)
                self._exchange_BBCx(self.mmt[0], newL, self.Hy)

        if self.apply_BBCy == True: 

            assert self.apply_PBCy == False

            if self.mode == 'TE':

                newL = self.Ly - 2*self.dy

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCy(self.mmt[1], newL, self.Hz)

            elif self.mode == 'TM':

                newL = self.Ly - 2*self.dy

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCy(self.mmt[1], newL, self.Hx)
                self._exchange_BBCy(self.mmt[1], newL, self.Hy)

    def _updateH_PBC_FDTD(self):

        newL = 0

        if self.apply_BBCx == True: 

            assert self.apply_PBCx == False

            if self.mode == 'TE':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCx(self.mmt[0], newL, self.Hz)

            elif self.mode == 'TM':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCx(self.mmt[0], newL, self.Hx)
                self._exchange_BBCx(self.mmt[0], newL, self.Hy)

        if self.apply_BBCy == True: 

            assert self.apply_PBCy == False

            if self.mode == 'TE':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCy(self.mmt[1], newL, self.Hz)

            elif self.mode == 'TM':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCy(self.mmt[1], newL, self.Hx)
                self._exchange_BBCy(self.mmt[1], newL, self.Hy)

    def _updateE_BBC_PSTD(self):

        #self.Ex[sli1] += self.dt/self.eps_Ex[sli1]*1j*(self.mmt[1]*self.hz_at_Ex[sli1] - self.mmt[2]*self.hy_at_Ex[sli1])
        #self.Ey[sli2] += self.dt/self.eps_Ey[sli2]*1j*(self.mmt[2]*self.hx_at_Ey[sli2] - self.mmt[0]*self.hz_at_Ey[sli2])
        #self.Ez[sli2] += self.dt/self.eps_Ez[sli2]*1j*(self.mmt[0]*self.hy_at_Ez[sli2] - self.mmt[1]*self.hx_at_Ez[sli2])

        if self.apply_BBCx == True:

            if self.mode == 'TM': self.Ez += self.dt/self.eps_Ez*1j*(-self.mmt[0]*self.Hy)
            if self.mode == 'TE': self.Ey += self.dt/self.eps_Ey*1j*(+self.mmt[0]*self.Hz)

        if self.apply_BBCy == True:
            
            if self.mode == 'TM': self.Ez += self.dt/self.eps_Ez*1j*(+self.mmt[1]*self.Hx)
            if self.mode == 'TE': self.Ex += self.dt/self.eps_Ex*1j*(-self.mmt[1]*self.Hz)

    def _updateE_BBC_SPSTD(self):

        #self.Ex[sli1] += self.dt/self.eps_Ex[sli1]*1j*(self.mmt[1]*self.hz_at_Ex[sli1] - self.mmt[2]*self.hy_at_Ex[sli1])
        #self.Ey[sli2] += self.dt/self.eps_Ey[sli2]*1j*(self.mmt[2]*self.hx_at_Ey[sli2] - self.mmt[0]*self.hz_at_Ey[sli2])
        #self.Ez[sli2] += self.dt/self.eps_Ez[sli2]*1j*(self.mmt[0]*self.hy_at_Ez[sli2] - self.mmt[1]*self.hx_at_Ez[sli2])

        if self.apply_BBCx == True:

            if self.mode == 'TM':

                self.hy_at_Ez = self.ifft(self.xmshift*self.fft(self.Hy, axes=(0,)), axes=(0,))
                self.Ez += self.dt/self.eps_Ez*1j*(-self.mmt[0]*self.hy_at_Ez)

            if self.mode == 'TE':

                self.hz_at_Ey = self.ifft(self.xmshift*self.fft(self.Hz, axes=(0,)), axes=(0,))
                self.Ey += self.dt/self.eps_Ey*1j*(+self.mmt[0]*self.hz_at_Ey)

        if self.apply_BBCy == True:
            
            if self.mode == 'TM':

                self.hx_at_Ez = self.ifft(self.ymshift*self.fft(self.Hx, axes=(1,)), axes=(1,))
                self.Ez += self.dt/self.eps_Ez*1j*(+self.mmt[1]*self.hx_at_Ez)

            if self.mode == 'TE':

                self.hz_at_Ex = self.ifft(self.ymshift*self.fft(self.Hz, axes=(1,)), axes=(1,))
                self.Ex += self.dt/self.eps_Ex*1j*(-self.mmt[1]*self.hz_at_Ez)

    def _updateE_BBC_SHPF(self):

        sli1 = [slice(None,None), slice(None,None)]
        sli2 = [slice(   1,None), slice(None,None)]

        if self.apply_BBCx == True: 

            assert self.apply_PBCx == False
            newL = self.Lx - 2*self.dx

            if self.mode == 'TM':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCx(self.mmt[0], newL, self.Hz)

            elif self.mode == 'TE':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCx(self.mmt[0], newL, self.Hx)
                self._exchange_BBCx(self.mmt[0], newL, self.Hy)

        if self.apply_BBCy == True:
            
            if self.mode == 'TM': self.Ez += self.dt/self.eps_Ez*1j*(+self.mmt[1]*self.Hx)
            if self.mode == 'TE': self.Ex += self.dt/self.eps_Ex*1j*(-self.mmt[1]*self.Hz)

    def _updateE_PBC_SHPF(self):

        sli1 = [slice(None,None), slice(None,None)]
        sli2 = [slice(   1,None), slice(None,None)]

        newL = 0

        if self.apply_BBCx == True: 

            assert self.apply_PBCx == False

            if self.mode == 'TM':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCx(self.mmt[0], newL, self.Hz)

            elif self.mode == 'TE':

                # Exchange Ex,Ey,Ez at i=0,1 with i=Nx-2, Nx-1.
                self._exchange_BBCx(self.mmt[0], newL, self.Hx)
                self._exchange_BBCx(self.mmt[0], newL, self.Hy)
