from mpi4py import MPI
from scipy.constants import c, mu_0, epsilon_0
import numpy as np
import cupy as cp

class Basic3D:
    
    def __init__(self, grid, gridgap, dt, tsteps, rdtype, cdtype, **kwargs):
        """Create Simulation Space.

            ex) Space.grid((128,128,600), (50*nm,50*nm,5*nm), dtype=self.xp.complex64)

        PARAMETERS
        ----------
        grid : tuple
            define the x,y,z grid.

        gridgap : tuple
            define the dx, dy, dz.

        rdtype : class numpy dtype
            choose self.xp.float32 or self.xp.float64

        cdtype : class numpy dtype
            choose self.xp.complex64 or self.xp.complex128

        kwargs : string
            
            supported arguments
            -------------------

            courant : float
                Set the courant number. For HPF, default is 1./4 and for FDTD, default is 1./2

        RETURNS
        -------
        None
        """

        self.nm = 1e-9
        self.um = 1e-6  

        self.rdtype   = rdtype
        self.cdtype   = cdtype
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
        self.TOTAL_NUM_GRID_SIZE = (self.rdtype(1).nbytes * self.TOTAL_NUM_GRID) / 1024 / 1024
        
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

        self.method = 'SHPF'
        self.engine = 'cupy'
        self.courant = 1./4

        if kwargs.get('engine') != None: self.engine = kwargs.get('engine')
        if kwargs.get('courant') != None: self.courant = kwargs.get('courant')
        if kwargs.get('method') != None: self.method = kwargs.get('method')

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

        self.myNx     = int(self.Nx/self.MPIsize)
        self.loc_grid = (self.myNx, self.Ny, self.Nz)

        self.Ex = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
        self.Ey = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
        self.Ez = self.xp.zeros(self.loc_grid, dtype=self.rdtype)

        self.Hx = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
        self.Hy = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
        self.Hz = self.xp.zeros(self.loc_grid, dtype=self.rdtype)

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

        self.ky = self.xp.fft.rfftfreq(self.Ny, self.dy) * 2 * self.xp.pi
        self.kz = self.xp.fft.rfftfreq(self.Nz, self.dz) * 2 * self.xp.pi

        if self.engine == 'cupy':
            self.iky = (1j*self.ky[None,:,None]).astype(self.cdtype)
            self.ikz = (1J*self.kz[None,None,:]).astype(self.cdtype)
            self.ypshift = self.xp.exp(self.iky* self.dy/2).astype(self.cdtype)
            self.zpshift = self.xp.exp(self.ikz* self.dz/2).astype(self.cdtype)
            self.ymshift = self.xp.exp(self.iky*-self.dy/2).astype(self.cdtype)
            self.zmshift = self.xp.exp(self.ikz*-self.dz/2).astype(self.cdtype)
        else:
            nax = np.newaxis
            self.iky = 1j*self.ky[:,nax,:]
            self.ikz = 1j*self.kz[:,:,nax]
            self.ypshift = self.xp.exp(1j*self.ky*self.dy/2)[:,nax,:]
            self.zpshift = self.xp.exp(1j*self.kz*self.dz/2)[:,:,nax]

        self.diffxEy = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
        self.diffxEz = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
        self.diffyEx = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
        self.diffyEz = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
        self.diffzEx = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
        self.diffzEy = self.xp.zeros(self.loc_grid, dtype=self.rdtype)

        self.diffxHy = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
        self.diffxHz = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
        self.diffyHx = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
        self.diffyHz = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
        self.diffzHx = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
        self.diffzHy = self.xp.zeros(self.loc_grid, dtype=self.rdtype)

        self.eps_Ex = self.xp.ones(self.loc_grid, dtype=self.rdtype) * epsilon_0
        self.eps_Ey = self.xp.ones(self.loc_grid, dtype=self.rdtype) * epsilon_0
        self.eps_Ez = self.xp.ones(self.loc_grid, dtype=self.rdtype) * epsilon_0

        self.mu_Hx  = self.xp.ones(self.loc_grid, dtype=self.rdtype) * mu_0
        self.mu_Hy  = self.xp.ones(self.loc_grid, dtype=self.rdtype) * mu_0
        self.mu_Hz  = self.xp.ones(self.loc_grid, dtype=self.rdtype) * mu_0

        self.econ_Ex = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
        self.econ_Ey = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
        self.econ_Ez = self.xp.zeros(self.loc_grid, dtype=self.rdtype)

        self.mcon_Hx = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
        self.mcon_Hy = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
        self.mcon_Hz = self.xp.zeros(self.loc_grid, dtype=self.rdtype)
 
    def set_PML(self, region, npml):

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

        self.PMLsigmax = self.xp.zeros(self.PMLgrading, dtype=self.rdtype)
        self.PMLalphax = self.xp.zeros(self.PMLgrading, dtype=self.rdtype)
        self.PMLkappax = self.xp.ones (self.PMLgrading, dtype=self.rdtype)

        self.PMLsigmay = self.xp.zeros(self.PMLgrading, dtype=self.rdtype)
        self.PMLalphay = self.xp.zeros(self.PMLgrading, dtype=self.rdtype)
        self.PMLkappay = self.xp.ones (self.PMLgrading, dtype=self.rdtype)

        self.PMLsigmaz = self.xp.zeros(self.PMLgrading, dtype=self.rdtype)
        self.PMLalphaz = self.xp.zeros(self.PMLgrading, dtype=self.rdtype)
        self.PMLkappaz = self.xp.ones (self.PMLgrading, dtype=self.rdtype)

        self.PMLbx = self.xp.zeros(self.PMLgrading, dtype=self.rdtype)
        self.PMLby = self.xp.zeros(self.PMLgrading, dtype=self.rdtype)
        self.PMLbz = self.xp.zeros(self.PMLgrading, dtype=self.rdtype)

        self.PMLax = self.xp.zeros(self.PMLgrading, dtype=self.rdtype)
        self.PMLay = self.xp.zeros(self.PMLgrading, dtype=self.rdtype)
        self.PMLaz = self.xp.zeros(self.PMLgrading, dtype=self.rdtype)

        #------------------------------------------------------------------------------------------------#
        #------------------------------- Grading kappa, sigma and alpha ---------------------------------#
        #------------------------------------------------------------------------------------------------#

        for key, value in self.PMLregion.items():

            if   key == 'x' and value != '':

                self.psi_eyx_p = self.xp.zeros((npml, self.Ny, self.Nz), dtype=self.rdtype)
                self.psi_ezx_p = self.xp.zeros((npml, self.Ny, self.Nz), dtype=self.rdtype)
                self.psi_hyx_p = self.xp.zeros((npml, self.Ny, self.Nz), dtype=self.rdtype)
                self.psi_hzx_p = self.xp.zeros((npml, self.Ny, self.Nz), dtype=self.rdtype)

                self.psi_eyx_m = self.xp.zeros((npml, self.Ny, self.Nz), dtype=self.rdtype)
                self.psi_ezx_m = self.xp.zeros((npml, self.Ny, self.Nz), dtype=self.rdtype)
                self.psi_hyx_m = self.xp.zeros((npml, self.Ny, self.Nz), dtype=self.rdtype)
                self.psi_hzx_m = self.xp.zeros((npml, self.Ny, self.Nz), dtype=self.rdtype)

                loc = self.xp.arange(self.PMLgrading) * self.dx / self.bdw_x
                self.PMLsigmax = self.PMLsigmamaxx * (loc **self.gO)
                self.PMLkappax = 1 + ((self.PMLkappamaxx-1) * (loc **self.gO))
                self.PMLalphax = self.PMLalphamaxx * ((1-loc) **self.sO)

            elif key == 'y' and value != '':

                self.psi_exy_p = self.xp.zeros((self.myNx, npml, self.Nz), dtype=self.rdtype)
                self.psi_ezy_p = self.xp.zeros((self.myNx, npml, self.Nz), dtype=self.rdtype)
                self.psi_hxy_p = self.xp.zeros((self.myNx, npml, self.Nz), dtype=self.rdtype)
                self.psi_hzy_p = self.xp.zeros((self.myNx, npml, self.Nz), dtype=self.rdtype)

                self.psi_exy_m = self.xp.zeros((self.myNx, npml, self.Nz), dtype=self.rdtype)
                self.psi_ezy_m = self.xp.zeros((self.myNx, npml, self.Nz), dtype=self.rdtype)
                self.psi_hxy_m = self.xp.zeros((self.myNx, npml, self.Nz), dtype=self.rdtype)
                self.psi_hzy_m = self.xp.zeros((self.myNx, npml, self.Nz), dtype=self.rdtype)

                loc  = self.xp.arange(self.PMLgrading) * self.dy / self.bdw_y
                self.PMLsigmay = self.PMLsigmamaxy * (loc **self.gO)
                self.PMLkappay = 1 + ((self.PMLkappamaxy-1) * (loc **self.gO))
                self.PMLalphay = self.PMLalphamaxy * ((1-loc) **self.sO)

            elif key == 'z' and value != '':

                self.psi_exz_p = self.xp.zeros((self.myNx, self.Ny, npml), dtype=self.rdtype)
                self.psi_eyz_p = self.xp.zeros((self.myNx, self.Ny, npml), dtype=self.rdtype)
                self.psi_hxz_p = self.xp.zeros((self.myNx, self.Ny, npml), dtype=self.rdtype)
                self.psi_hyz_p = self.xp.zeros((self.myNx, self.Ny, npml), dtype=self.rdtype)

                self.psi_exz_m = self.xp.zeros((self.myNx, self.Ny, npml), dtype=self.rdtype)
                self.psi_eyz_m = self.xp.zeros((self.myNx, self.Ny, npml), dtype=self.rdtype)
                self.psi_hxz_m = self.xp.zeros((self.myNx, self.Ny, npml), dtype=self.rdtype)
                self.psi_hyz_m = self.xp.zeros((self.myNx, self.Ny, npml), dtype=self.rdtype)

                loc  = self.xp.arange(self.PMLgrading) * self.dz / self.bdw_z
                self.PMLsigmaz = self.PMLsigmamaxz * (loc **self.gO)
                self.PMLkappaz = 1 + ((self.PMLkappamaxz-1) * (loc **self.gO))
                self.PMLalphaz = self.PMLalphamaxz * ((1-loc) **self.sO)

        #------------------------------------------------------------------------------------------------#
        #--------------------------------- Get 'b' and 'a' for CPML theory ------------------------------#
        #------------------------------------------------------------------------------------------------#

        if 'x' in self.PMLregion.keys() and self.PMLregion.get('x') != '':
            self.PMLbx = self.xp.exp(-(self.PMLsigmax/self.PMLkappax + self.PMLalphax) * self.dt / epsilon_0)
            self.PMLax = self.PMLsigmax / (self.PMLsigmax*self.PMLkappax + self.PMLalphax*self.PMLkappax**2) * (self.PMLbx - 1.)

        if 'y' in self.PMLregion.keys() and self.PMLregion.get('y') != '':
            self.PMLby = self.xp.exp(-(self.PMLsigmay/self.PMLkappay + self.PMLalphay) * self.dt / epsilon_0)
            self.PMLay = self.PMLsigmay / (self.PMLsigmay*self.PMLkappay + self.PMLalphay*self.PMLkappay**2) * (self.PMLby - 1.)

        if 'z' in self.PMLregion.keys() and self.PMLregion.get('z') != '':
            self.PMLbz = self.xp.exp(-(self.PMLsigmaz/self.PMLkappaz + self.PMLalphaz) * self.dt / epsilon_0)
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

                        self.src = self.xp.zeros(self.tsteps, dtype=self.rdtype)

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

                self.src = self.xp.zeros(self.tsteps, dtype=self.rdtype)

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
        
        self.pulse = self.rdtype(pulse)

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

            recvEylast = self.MPIcomm.recv( source=(self.MPIrank+1), tag=(tstep*100+9 ))
            recvEzlast = self.MPIcomm.recv( source=(self.MPIrank+1), tag=(tstep*100+11))

            if self.engine == 'cupy':
                recvEylast = cp.asarray(recvEylast)
                recvEzlast = cp.asarray(recvEzlast)

        #-----------------------------------------------------------#
        #---------------------- Get derivatives --------------------#
        #-----------------------------------------------------------#

        if self.method == 'SHPF':

            # To update Hx
            self.diffyEz = self.xp.fft.irfftn(self.iky*self.ypshift*self.xp.fft.rfftn(self.Ez, axes=(1,)), axes=(1,))
            self.diffzEy = self.xp.fft.irfftn(self.ikz*self.zpshift*self.xp.fft.rfftn(self.Ey, axes=(2,)), axes=(2,))

            # To update Hy
            self.diffzEx = self.xp.fft.irfftn(self.ikz*self.zpshift*self.xp.fft.rfftn(self.Ex, axes=(2,)), axes=(2,))
            self.diffxEz[:-1,:,:] = (self.Ez[1:,:,:] - self.Ez[:-1,:,:]) / self.dx

            # To update Hz
            self.diffyEx = self.xp.fft.irfftn(self.iky*self.ypshift*self.xp.fft.rfftn(self.Ex, axes=(1,)), axes=(1,))
            self.diffxEy[:-1,:,:] = (self.Ey[1:,:,:] - self.Ey[:-1,:,:]) / self.dx

            if self.MPIrank != (self.MPIsize-1):

                # No need to update diffzEx and diffyEx because they are already done.
                # To update Hy at x=myNx-1.
                self.diffxEz[-1,:,:] = (recvEzlast[:,:] - self.Ez[-1,:,:]) / self.dx

                # To update Hz at x=myNx-1
                self.diffxEy[-1,:,:] = (recvEylast[:,:] - self.Ey[-1,:,:]) / self.dx

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
                self.diffzEx[-1,:,:-1] = ( self.Ex[-1,:,1:] - self.Ex[-1,:,:-1]) / self.dz
                self.diffxEz[-1,:,:-1] = (recvEzlast[:,:-1] - self.Ez[-1,:,:-1]) / self.dx

                # To update Hz at x=myNx-1
                self.diffxEy[-1,:-1,:] = (recvEylast[:-1,:] - self.Ey[-1,:-1,:]) / self.dx
                self.diffyEx[-1,:-1,:] = ( self.Ex[-1,1:,:] - self.Ex[-1,:-1,:]) / self.dy

        #-----------------------------------------------------------#
        #--------------- Cast basic update equations ---------------#
        #-----------------------------------------------------------#

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

        elif self.method == 'FDTD':

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

        #-----------------------------------------------------------#
        #---------------- Apply PML when it is given ---------------#
        #-----------------------------------------------------------#

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

        #-----------------------------------------------------------#
        #---------------- Apply BBC when it is given ---------------#
        #-----------------------------------------------------------#

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

            recvHyfirst = self.MPIcomm.recv( source=(self.MPIrank-1), tag=(tstep*100+3))
            recvHzfirst = self.MPIcomm.recv( source=(self.MPIrank-1), tag=(tstep*100+5))
        
            if self.engine == 'cupy':
                recvHyfirst = cp.asarray(recvHyfirst)
                recvHzfirst = cp.asarray(recvHzfirst)

        #-----------------------------------------------------------#
        #---------------------- Get derivatives --------------------#
        #-----------------------------------------------------------#

        if self.method == 'SHPF':

            # Get derivatives of Hy and Hz to update Ex
            self.diffyHz = self.xp.fft.irfftn(self.iky*self.ymshift*self.xp.fft.rfftn(self.Hz, axes=(1,)), axes=(1,))
            self.diffzHy = self.xp.fft.irfftn(self.ikz*self.zmshift*self.xp.fft.rfftn(self.Hy, axes=(2,)), axes=(2,))

            # Get derivatives of Hx and Hz to update Ey
            self.diffzHx = self.xp.fft.irfftn(self.ikz*self.zmshift*self.xp.fft.rfftn(self.Hx, axes=(2,)), axes=(2,))
            self.diffxHz[1:,:,:] = (self.Hz[1:,:,:] - self.Hz[:-1,:,:]) / self.dx

            # Get derivatives of Hx and Hy to update Ez
            self.diffyHx = self.xp.fft.irfftn(self.iky*self.ymshift*self.xp.fft.rfftn(self.Hx, axes=(1,)), axes=(1,))
            self.diffxHy[1:,:,:] = (self.Hy[1:,:,:] - self.Hy[:-1,:,:]) / self.dx

            if self.MPIrank != 0:

                # Get derivatives of Hx and Hz to update Ey at x=0.
                self.diffxHz[0,:,:] = (self.Hz[0,:,:]-recvHzfirst[:,:]) / self.dx

                # Get derivatives of Hx and Hy to update Ez at x=0.
                self.diffxHy[0,:,:] = (self.Hy[0,:,:]-recvHyfirst[:,:]) / self.dx

        elif self.method == 'FDTD':

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
                self.diffxHz[0,:,1:] = (self.Hz[0,:,1:]-recvHzfirst[:,1:]) / self.dx
                self.diffzHx[0,:,1:] = (self.Hx[0,:,1:]- self.Hx[0,:,:-1]) / self.dz

                # Get derivatives of Hx and Hy to update Ez at x=0.
                self.diffxHy[0,1:,:] = (self.Hy[0,1:,:]-recvHyfirst[1:,:]) / self.dx
                self.diffyHx[0,1:,:] = (self.Hx[0,1:,:]- self.Hx[0,:-1,:]) / self.dy

        #-----------------------------------------------------------#
        #--------------- Cast basic update equations ---------------#
        #-----------------------------------------------------------#

        # Update Ex, Ey, Ez
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

        #-----------------------------------------------------------#
        #---------------- Apply PML when it is given ---------------#
        #-----------------------------------------------------------#

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

        #-----------------------------------------------------------#
        #---------------- Apply BBC when it is given ---------------#
        #-----------------------------------------------------------#

    def _PML_updateH_px(self):

        if self.method == 'SHPF':

            psiidx_Hyxp = [slice(0,-1), slice(0,None), slice(0,None)]
            myidx_Hyxp = [slice(-self.npml,-1), slice(0,None), slice(0,None)]

            psiidx_Hzxp = [slice(0,-1), slice(0,None), slice(0,None)]
            myidx_Hzxp = [slice(-self.npml,-1), slice(0,None), slice(0,None)]

        elif self.method == 'FDTD':

            psiidx_Hyxp = [slice(0,-1), slice(0,None), slice(0,-1)]
            myidx_Hyxp = [slice(-self.npml,-1), slice(0,None), slice(0,-1)]

            psiidx_Hzxp = [slice(0,-1), slice(0,-1), slice(0,None)]
            myidx_Hzxp = [slice(-self.npml,-1), slice(0,-1), slice(0,None)]

        odd = [slice(1,-1,2), None, None]

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

        if self.method == 'SHPF':

            psiidx_Eyxp = [slice(0,None), slice(0,None), slice(0,None)]
            myidx_Eyxp = [slice(-self.npml,None), slice(0,None), slice(0,None)]

            psiidx_Ezxp = [slice(0,None), slice(0,None), slice(0,None)]
            myidx_Ezxp = [slice(-self.npml,None), slice(0,None), slice(0,None)]

        elif self.method == 'FDTD':

            psiidx_Eyxp = [slice(0,None), slice(0,None), slice(1,None)]
            myidx_Eyxp = [slice(-self.npml,None), slice(0,None), slice(1,None)]

            psiidx_Ezxp = [slice(0,None), slice(1,None), slice(0,None)]
            myidx_Ezxp = [slice(-self.npml,None), slice(1,None), slice(0,None)]

        even = [slice(0,None,2), None, None]

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

        if self.method == 'SHPF':

            psiidx_Hyxm = [slice(0,self.npml), slice(0,None), slice(0,None)]
            myidx_Hyxm  = [slice(0,self.npml), slice(0,None), slice(0,None)]

            psiidx_Hzxm = [slice(0,self.npml), slice(0,None), slice(0,None)]
            myidx_Hzxm  = [slice(0,self.npml), slice(0,None), slice(0,None)]

        elif self.method == 'FDTD':

            psiidx_Hyxm = [slice(0,self.npml), slice(0,None), slice(0,-1)]
            myidx_Hyxm  = [slice(0,self.npml), slice(0,None), slice(0,-1)]

            psiidx_Hzxm = [slice(0, self.npml), slice(0,-1), slice(0,None)]
            myidx_Hzxm  = [slice(0, self.npml), slice(0,-1), slice(0,None)]

        even = [slice(-2,None,-2), None, None]

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

        if self.method == 'SHPF':

            psiidx_Eymx = [slice(1,self.npml), slice(0,None), slice(0,None)]
            myidx_Eymx  = [slice(1,self.npml), slice(0,None), slice(0,None)]

            psiidx_Ezmx = [slice(1,self.npml), slice(0,None), slice(0,None)]
            myidx_Ezmx  = [slice(1,self.npml), slice(0,None), slice(0,None)]

        elif self.method == 'FDTD':

            psiidx_Eymx = [slice(1,self.npml), slice(0,None), slice(1,None)]
            myidx_Eymx  = [slice(1,self.npml), slice(0,None), slice(1,None)]

            psiidx_Ezmx = [slice(1,self.npml), slice(1,None), slice(0,None)]
            myidx_Ezmx  = [slice(1,self.npml), slice(1,None), slice(0,None)]

        odd = [slice(-3,None,-2),None,None]

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

        elif self.method == 'FDTD':

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

        if self.method == 'SHPF':

            psiidx_Exyp = [slice(0,None), slice(0,self.npml), slice(0,None)]
            myidx_Exyp  = [slice(0,None), slice(-self.npml,None), slice(0,None)]

            if self.MPIrank > 0:
                psiidx_Ezyp = [slice(0,None), slice(0,self.npml), slice(0,None)]
                myidx_Ezyp  = [slice(0,None), slice(-self.npml,None), slice(0,None)]
            else:
                psiidx_Ezyp = [slice(1,None), slice(0,self.npml), slice(0,None)]
                myidx_Ezyp  = [slice(1,None), slice(-self.npml,None), slice(0,None)]

        elif self.method == 'FDTD':
         
            psiidx_Exyp = [slice(0,None), slice(0,self.npml), slice(1,None)]
            myidx_Exyp  = [slice(0,None), slice(-self.npml,None), slice(1,None)]

            if self.MPIrank > 0:
                psiidx_Ezyp = [slice(0,None), slice(0,self.npml), slice(0,None)]
                myidx_Ezyp  = [slice(0,None), slice(-self.npml,None), slice(0,None)]
            else:
                psiidx_Ezyp = [slice(1,None), slice(0,self.npml), slice(0,None)]
                myidx_Ezyp  = [slice(1,None), slice(-self.npml,None), slice(0,None)]

        even = [None,slice(0,None,2),None]

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

        if self.method == 'SHPF':

            psiidx_Hxym = [slice(0,None), slice(0,self.npml), slice(0,None)]
            myidx_Hxym  = [slice(0,None), slice(0,self.npml), slice(0,None)]

            if self.MPIrank < (self.MPIsize-1):
                psiidx_Hzym = [slice(0,None), slice(0,self.npml), slice(0,None)]
                myidx_Hzym  = [slice(0,None), slice(0,self.npml), slice(0,None)]
            else:
                psiidx_Hzym = [slice(0,-1), slice(0,self.npml), slice(0,None)]
                myidx_Hzym  = [slice(0,-1), slice(0,self.npml), slice(0,None)]

        elif self.method == 'FDTD':

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

        elif self.method == 'FDTD':

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
        self.psi_eyz_p[psiidx_Eyzp] = (self.PMLbz[even]*self.psi_eyz_p[psiidx_Eyzp])
                                    + (self.PMLaz[even]*self.diffzHx[myidx_Eyzp])
        self.Ey[myidx_Eyzp] += CEy2*(+((1./self.PMLkappaz[even]-1.)*self.diffzHx[myidx_Eyzp]) + self.psi_eyz_p[psiidx_Eyzp])

    def _PML_updateH_mz(self):

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
        CHx2 = (-2*self.dt) / (2.*self.mu_Hx[myidx] + self.mcon_Hx[myidx]*self.dt)
        self.psi_hxz_m[psiidx] = (self.PMLbz[even]*self.psi_hxz_m[psiidx]) + (self.PMLaz[even]*self.diffzEy[myidx])
        self.Hx[myidx] += CHx2*(-((1./self.PMLkappaz[even]-1.)*self.diffzEy[myidx]) - self.psi_hxz_m[psiidx])

        # Update Hy at z-.
        CHy2 = (-2*self.dt) / (2.*self.mu_Hy[myidx_Hyzm] + self.mcon_Hy[myidx_Hyzm]*self.dt)
        self.psi_hyz_m[psiidx_Hyzm] = (self.PMLbz[even]*self.psi_hyz_m[psiidx_Hyzm])\
                                    + (self.PMLaz[even]*self.diffzEx[myidx_Hyzm])
        self.Hy[myidx_Hyzm] += CHy2*(+((1./self.PMLkappaz[even]-1.)*self.diffzEx[myidx_Hyzm]) + self.psi_hyz_m[psiidx_Hyzm])

    def _PML_updateE_mz(self):

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


class Empty3D(Basic3D):
    
    def __init__(self, grid, gridgap, dt, tsteps, rdtype, cdtype, **kwargs):

        Basic3D.__init__(self, grid, gridgap, dt, tsteps, rdtype, cdtype, **kwargs)

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
