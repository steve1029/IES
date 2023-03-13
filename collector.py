import ctypes, os
from functools import reduce
import numpy as np
import cupy as cp
import h5py
import matplotlib.pyplot as plt

class collector:

    def __init__(self, name, path, space, engine):

        self.engine = engine
        if self.engine == 'cupy' : self.xp = cp
        else: self.xp = np

        self.name = name
        self.space = space
        self.path = path

        # Make a directory to save data.
        if self.space.MPIrank == 0:

            if os.path.exists(self.path) == True: pass
            else: os.makedirs(self.path)

        # Initial global/local location.
        self.gloc = None
        self.lloc = None

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
        assert gxends < self.space.Nx

        # Global x index of the border of each node.
        bxsrt = self.space.myNx_indice[self.space.MPIrank][0]
        bxend = self.space.myNx_indice[self.space.MPIrank][1]

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
        node_xsrt = self.space.myNx_indice[MPIrank][0]
        node_xend = self.space.myNx_indice[MPIrank][1]

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


class FieldAtPoint(collector):

    def __init__(self, name, path, space, loc, engine):
        """Collector object to collect the fields at a point.

        Args:
            name: string.

            space: space object.

            loc: float
                location of a collector.

            engine: string
                choose 'numpy' or 'cupy'.

        Returns:
            None
        """

        collector.__init__(self, name, path, space, engine)
        self.loc = loc

        # Location of the structure.

        if len(self.loc) == 3:

            self.xloc = round(loc[0]/space.dx)
            self.yloc = round(loc[1]/space.dy)
            self.zloc = round(loc[2]/space.dz)

        elif len(self.loc) == 2:

            self.xloc = round(loc[0]/space.dx)
            self.yloc = round(loc[1]/space.dy)

        self.gxloc, self.lxloc = collector._get_local_x_loc(self, self.xloc, self.xloc)

        if self.gxloc != None:

            self.Ex_t = self.xp.zeros(space.tsteps, dtype=space.field_dtype)
            self.Ey_t = self.xp.zeros(space.tsteps, dtype=space.field_dtype)
            self.Ez_t = self.xp.zeros(space.tsteps, dtype=space.field_dtype)

            self.Hx_t = self.xp.zeros(space.tsteps, dtype=space.field_dtype)
            self.Hy_t = self.xp.zeros(space.tsteps, dtype=space.field_dtype)
            self.Hz_t = self.xp.zeros(space.tsteps, dtype=space.field_dtype)

    def get_time_signal(self, tstep):

        if self.gxloc != None:

            dt = self.space.dt
            xsrt = self.lxloc[0]
            xend = self.lxloc[1]

            f = slice(0,None)

            if self.space.dimension == 3:

                self.Ex_t[tstep] = self.space.Ex[xsrt, self.yloc, self.zloc]
                self.Ey_t[tstep] = self.space.Ey[xsrt, self.yloc, self.zloc]
                self.Ez_t[tstep] = self.space.Ez[xsrt, self.yloc, self.zloc]

                self.Hx_t[tstep] = self.space.Hx[xsrt, self.yloc, self.zloc]
                self.Hy_t[tstep] = self.space.Hy[xsrt, self.yloc, self.zloc]
                self.Hz_t[tstep] = self.space.Hz[xsrt, self.yloc, self.zloc]

            elif self.space.dimension == 2:

                if self.space.mode == 'TM':

                    self.Ez_t[tstep] = self.space.Ez[xsrt, self.yloc]
                    self.Hx_t[tstep] = self.space.Hx[xsrt, self.yloc]
                    self.Hy_t[tstep] = self.space.Hy[xsrt, self.yloc]

                if self.space.mode == 'TE':

                    self.Ex_t[tstep] = self.space.Ex[xsrt, self.yloc]
                    self.Ey_t[tstep] = self.space.Ey[xsrt, self.yloc]
                    self.Hz_t[tstep] = self.space.Hz[xsrt, self.yloc]

    def save_time_signal(self, **kwargs):

        self.space.MPIcomm.barrier()
        self.binary = True
        self.txt = False

        if kwargs.get('binary') != None: 

            hey = kwargs.get('binary')

            assert hey == True or hey == False
            self.binary =  hey

        if kwargs.get('txt') != None: 

            hey = kwargs.get('txt')

            assert hey == True or hey == False
            self.txt =  hey

        if self.gxloc != None:

            if self.binary == True:

                self.xp.save("{}/{}_Ex_t.npy" .format(self.path, self.name), self.Ex_t)
                self.xp.save("{}/{}_Ey_t.npy" .format(self.path, self.name), self.Ey_t)
                self.xp.save("{}/{}_Ez_t.npy" .format(self.path, self.name), self.Ez_t)

                self.xp.save("{}/{}_Hx_t.npy" .format(self.path, self.name), self.Hx_t)
                self.xp.save("{}/{}_Hy_t.npy" .format(self.path, self.name), self.Hy_t)
                self.xp.save("{}/{}_Hz_t.npy" .format(self.path, self.name), self.Hz_t)

            if self.txt == True:

                if self.xp == cp: 

                    self.Ex_t = self.xp.asnumpy(self.Ex_t)
                    self.Ey_t = self.xp.asnumpy(self.Ey_t)
                    self.Ez_t = self.xp.asnumpy(self.Ez_t)

                    self.Hx_t = self.xp.asnumpy(self.Hx_t)
                    self.Hy_t = self.xp.asnumpy(self.Hy_t)
                    self.Hz_t = self.xp.asnumpy(self.Hz_t)

                Ext_name_rank = "{}/{}_Ex_t.txt" .format(self.path, self.name)
                Eyt_name_rank = "{}/{}_Ey_t.txt" .format(self.path, self.name)
                Ezt_name_rank = "{}/{}_Ez_t.txt" .format(self.path, self.name)

                Hxt_name_rank = "{}/{}_Hx_t.txt" .format(self.path, self.name)
                Hyt_name_rank = "{}/{}_Hy_t.txt" .format(self.path, self.name)
                Hzt_name_rank = "{}/{}_Hz_t.txt" .format(self.path, self.name)

                np.savetxt(Ext_name_rank, self.Ex_t, newline='\n', fmt='%1.15f+%1.15fi')
                np.savetxt(Eyt_name_rank, self.Ey_t, newline='\n', fmt='%1.15f+%1.15fi')
                np.savetxt(Ezt_name_rank, self.Ez_t, newline='\n', fmt='%1.15f+%1.15fi')

                np.savetxt(Hxt_name_rank, self.Hx_t, newline='\n', fmt='%1.15f+%1.15fi')
                np.savetxt(Hyt_name_rank, self.Hy_t, newline='\n', fmt='%1.15f+%1.15fi')
                np.savetxt(Hzt_name_rank, self.Hz_t, newline='\n', fmt='%1.15f+%1.15fi')


class Sx(collector):

    def __init__(self, name, path, space, xloc, srt, end, freqs, engine):
        """Sx collector object.

        Args:
            name: string.

            space: space object.

            xloc: float
                x location of a collector.

            srt: tuple
                (ysrt, zsrt)

            end: tuple
                (yend, zend)

            freqs: ndarray

            engine: string
                choose 'numpy' or 'cupy'.

        Returns:
            None
        """

        collector.__init__(self, name, path, space, engine)

        self.Nf = len(freqs)
        if self.engine == 'cupy': self.freqs = cp.asarray(freqs)
        else: self.freqs = freqs

        # Start loc of the structure.
        self.xsrt = round(xloc  /space.dx)
        self.ysrt = round(srt[0]/space.dy)
        self.zsrt = round(srt[1]/space.dz)

        # End loc of the structure.
        self.xend = self.xsrt + 1
        self.yend = round(end[0]/space.dy)
        self.zend = round(end[1]/space.dz)

        self.gxloc, self.lxloc = collector._get_local_x_loc(self, self.xsrt, self.xend)

        if self.gxloc != None:

           # print("rank {:>2}: xloc of Sx collector >>> global \"{},{}\" and local \"{},{}\"" \
           #       .format(self.space.MPIrank, self.gxloc[0], self.gxloc[1], self.lxloc[0], self.lxloc[1]))
            #print(self.ysrt, self.yend)
            #print(self.zsrt, self.zend)

            self.DFT_Ey = self.xp.zeros((self.Nf, self.yend-self.ysrt, self.zend-self.zsrt), dtype=np.complex128)
            self.DFT_Ez = self.xp.zeros((self.Nf, self.yend-self.ysrt, self.zend-self.zsrt), dtype=np.complex128)

            self.DFT_Hy = self.xp.zeros((self.Nf, self.yend-self.ysrt, self.zend-self.zsrt), dtype=np.complex128)
            self.DFT_Hz = self.xp.zeros((self.Nf, self.yend-self.ysrt, self.zend-self.zsrt), dtype=np.complex128)

    def do_RFT(self, tstep):

        if self.gxloc != None:

            dt = self.space.dt
            xsrt = self.lxloc[0]
            xend = self.lxloc[1]

            f = (slice(0,None), None, None)
            Fidx = (slice(xsrt,xend), slice(self.ysrt, self.yend), slice(self.zsrt, self.zend))

            self.DFT_Ey += self.space.Ey[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            self.DFT_Hz += self.space.Hz[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

            self.DFT_Ez += self.space.Ez[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            self.DFT_Hy += self.space.Hy[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

    def get_Sx(self, tstep, h5=False):

        self.space.MPIcomm.barrier()

        if self.gxloc != None:

            self.Sx = 0.5 * (  (self.DFT_Ey.real*self.DFT_Hz.real) + (self.DFT_Ey.imag*self.DFT_Hz.imag)
                              -(self.DFT_Ez.real*self.DFT_Hy.real) - (self.DFT_Ez.imag*self.DFT_Hy.imag)  )

            self.Sx_area = self.Sx.sum(axis=(1,2)) * self.space.dy * self.space.dz

            Eyname = f"{self.path}{self.name}_DFT_Ey_{tstep:07d}tstep_rank{self.space.MPIrank:02d}"
            Ezname = f"{self.path}{self.name}_DFT_Ez_{tstep:07d}tstep_rank{self.space.MPIrank:02d}"
            Hyname = f"{self.path}{self.name}_DFT_Hy_{tstep:07d}tstep_rank{self.space.MPIrank:02d}"
            Hzname = f"{self.path}{self.name}_DFT_Hz_{tstep:07d}tstep_rank{self.space.MPIrank:02d}"

            self.xp.save(Eyname, self.DFT_Ey)
            self.xp.save(Ezname, self.DFT_Ez)
            self.xp.save(Hyname, self.DFT_Hy)
            self.xp.save(Hzname, self.DFT_Hz)
            self.xp.save(f"{self.path}{self.name}_{tstep:07d}tstep_area" , self.Sx_area)

            if h5 == True:

                if self.space.engine == 'cupy':

                    with h5py.File(f'{self.path}{self.name}_DFTs_{tstep:07d}tstep_rank{self.space.MPIrank:02d}.h5', 'w') as hf:

                        hf.create_dataset('Sx_Ey', data=cp.asnumpy(self.DFT_Ey))
                        hf.create_dataset('Sx_Ez', data=cp.asnumpy(self.DFT_Ez))
                        hf.create_dataset('Sx_Hy', data=cp.asnumpy(self.DFT_Hy))
                        hf.create_dataset('Sx_Hz', data=cp.asnumpy(self.DFT_Hz))
                        hf.create_dataset('Sx_area', data=cp.asnumpy(self.Sx_area))

                else:

                    with h5py.File(f'{self.path}{self.name}_DFTs_{tstep:07d}tstep_rank{self.space.MPIrank:02d}.h5', 'w') as hf:

                        hf.create_dataset('Sx_Ey', data=self.DFT_Ey)
                        hf.create_dataset('Sx_Ez', data=self.DFT_Ez)
                        hf.create_dataset('Sx_Hy', data=self.DFT_Hy)
                        hf.create_dataset('Sx_Hz', data=self.DFT_Hz)
                        hf.create_dataset('Sx_area', data=self.Sx_area)


class Sy(collector):

    def __init__(self, name, path, space, yloc, srt, end, freqs, engine):
        """Sy collector object.

        Args:
            name: string.

            space: space object.

            yloc: float.

            srt: tuple.
                (xsrt, zsrt).

            end: tuple.
                (xend, zend).

            freqs: ndarray.

            engine: string.

        Returns:
            None
        """

        collector.__init__(self, name, path, space, engine)

        self.Nf = len(freqs)
        if self.engine == 'cupy': self.freqs = cp.asarray(freqs)
        else: self.freqs = freqs

        # Start loc of the structure.
        self.xsrt = round(srt[0]/space.dx)
        self.ysrt = round(  yloc/space.dy)
        self.zsrt = round(srt[1]/space.dz)

        # End loc of the structure.
        self.xend = round(end[0]/space.dx)
        self.yend = self.ysrt+1
        self.zend = round(end[1]/space.dz)

        # Local variables for readable code.
        xsrt = self.xsrt
        ysrt = self.ysrt
        zsrt = self.zsrt
        xend = self.xend
        yend = self.yend
        zend = self.zend

        self.who_get_Sy_gxloc = {} # global locations
        self.who_get_Sy_lxloc = {} # local locations

        # Every node has to know who collects Sy.
        for MPIrank in range(self.space.MPIsize):

            # Global x index of each node.
            node_xsrt = self.space.myNx_indice[MPIrank][0]
            node_xend = self.space.myNx_indice[MPIrank][1]

            if xsrt >  node_xend: pass
            if xend <  node_xsrt: pass
            if xsrt <  node_xsrt and xend > node_xsrt and xend <= node_xend:

                gloc = ((node_xsrt          , ysrt, zsrt), (xend          , yend, zend))
                lloc = ((node_xsrt-node_xsrt, ysrt, zsrt), (xend-node_xsrt, yend, zend))

                self.who_get_Sy_gxloc[MPIrank] = gloc
                self.who_get_Sy_lxloc[MPIrank] = lloc

            if xsrt <  node_xsrt and xend > node_xend:
                gloc = ((node_xsrt          , ysrt, zsrt), (node_xend          , yend, zend))
                lloc = ((node_xsrt-node_xsrt, ysrt, zsrt), (node_xend-node_xsrt, yend, zend))

                self.who_get_Sy_gxloc[MPIrank] = gloc
                self.who_get_Sy_lxloc[MPIrank] = lloc

            if xsrt >= node_xsrt and xsrt < node_xend and xend <= node_xend:
                gloc = ((xsrt          , ysrt, zsrt), (xend          , yend, zend))
                lloc = ((xsrt-node_xsrt, ysrt, zsrt), (xend-node_xsrt, yend, zend))

                self.who_get_Sy_gxloc[MPIrank] = gloc
                self.who_get_Sy_lxloc[MPIrank] = lloc

            if xsrt >= node_xsrt and xsrt < node_xend and xend >  node_xend:
                gloc = ((xsrt          , ysrt, zsrt), (node_xend          , yend, zend))
                lloc = ((xsrt-node_xsrt, ysrt, zsrt), (node_xend-node_xsrt, yend, zend))

                self.who_get_Sy_gxloc[MPIrank] = gloc
                self.who_get_Sy_lxloc[MPIrank] = lloc

        #if self.space.MPIrank == 0: print("{} collectors: rank{}" .format(self.name, list(self.who_get_Sy_gxloc)))

        self.space.MPIcomm.barrier()

        if self.space.MPIrank in self.who_get_Sy_lxloc:

            self.gloc = self.who_get_Sy_gxloc[self.space.MPIrank]
            self.lloc = self.who_get_Sy_lxloc[self.space.MPIrank]

            """
            print("rank {:>2}: x loc of {} collector >>> global range({:4d},{:4d}) // local range({:4d},{:4d})\"" \
                   .format(self.space.MPIrank, self.name, self.gloc[0][0], self.gloc[1][0], self.lloc[0][0], self.lloc[1][0]))

            print("rank {:>2}: y loc of {} collector >>> global range({:4d},{:4d}) // local range({:4d},{:4d})\"" \
                   .format(self.space.MPIrank, self.name, self.gloc[0][1], self.gloc[1][1], self.lloc[0][1], self.lloc[1][1]))

            print("rank {:>2}: z loc of {} collector >>> global range({:4d},{:4d}) // local range({:4d},{:4d})\"" \
                   .format(self.space.MPIrank, self.name, self.gloc[0][2], self.gloc[1][2], self.lloc[0][2], self.lloc[1][2]))
            """

            xsrt = self.lloc[0][0]
            xend = self.lloc[1][0]

            self.DFT_Ex = self.xp.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=np.complex128)
            self.DFT_Ez = self.xp.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=np.complex128)

            self.DFT_Hx = self.xp.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=np.complex128)
            self.DFT_Hz = self.xp.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=np.complex128)
        
        #print(self.who_get_Sy_gxloc)
        #print(self.who_get_Sy_lxloc)

    def do_RFT(self, tstep):

        if self.space.MPIrank in self.who_get_Sy_lxloc:

            dt = self.space.dt
            xsrt = self.lloc[0][0]
            xend = self.lloc[1][0]
            ysrt = self.lloc[0][1]
            yend = self.lloc[1][1]
            zsrt = self.lloc[0][2]
            zend = self.lloc[1][2]

            f = (slice(0,None), None, None)
            Fidx = (slice(xsrt,xend), ysrt, slice(zsrt, zend))

            self.DFT_Ex += self.space.Ex[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            self.DFT_Hz += self.space.Hz[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

            self.DFT_Ez += self.space.Ez[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            self.DFT_Hx += self.space.Hx[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

    def get_Sy(self, tstep, h5=False):

        self.space.MPIcomm.barrier()

        Exname = f"{self.path}{self.name}_DFT_Ex_{tstep:07d}tstep_rank{self.space.MPIrank:02d}"
        Ezname = f"{self.path}{self.name}_DFT_Ez_{tstep:07d}tstep_rank{self.space.MPIrank:02d}"
        Hxname = f"{self.path}{self.name}_DFT_Hx_{tstep:07d}tstep_rank{self.space.MPIrank:02d}"
        Hzname = f"{self.path}{self.name}_DFT_Hz_{tstep:07d}tstep_rank{self.space.MPIrank:02d}"

        if self.space.MPIrank in self.who_get_Sy_lxloc:

            self.xp.save(Exname, self.DFT_Ex)
            self.xp.save(Ezname, self.DFT_Ez)
            self.xp.save(Hxname, self.DFT_Hx)
            self.xp.save(Hzname, self.DFT_Hz)

        self.space.MPIcomm.barrier()

        if self.space.MPIrank == 0:

            DFT_Sy_Exs = []
            DFT_Sy_Ezs = []

            DFT_Sy_Hxs = []
            DFT_Sy_Hzs = []

            for rank in self.who_get_Sy_lxloc:

                Exname = f"{self.path}{self.name}_DFT_Ex_{tstep:07d}tstep_rank{rank:02d}.npy"
                Ezname = f"{self.path}{self.name}_DFT_Ez_{tstep:07d}tstep_rank{rank:02d}.npy"
                Hxname = f"{self.path}{self.name}_DFT_Hx_{tstep:07d}tstep_rank{rank:02d}.npy"
                Hzname = f"{self.path}{self.name}_DFT_Hz_{tstep:07d}tstep_rank{rank:02d}.npy"

                DFT_Sy_Exs.append(np.load(Exname))
                DFT_Sy_Ezs.append(np.load(Ezname))
                DFT_Sy_Hxs.append(np.load(Hxname))
                DFT_Sy_Hzs.append(np.load(Hzname))

            DFT_Ex = np.concatenate(DFT_Sy_Exs, axis=1)
            DFT_Ez = np.concatenate(DFT_Sy_Ezs, axis=1)
            DFT_Hx = np.concatenate(DFT_Sy_Hxs, axis=1)
            DFT_Hz = np.concatenate(DFT_Sy_Hzs, axis=1)

            self.Sy = 0.5 * ( -(DFT_Ex.real*DFT_Hz.real) - (DFT_Ex.imag*DFT_Hz.imag)
                              +(DFT_Ez.real*DFT_Hx.real) + (DFT_Ez.imag*DFT_Hx.imag)  )

            self.Sy_area = self.Sy.sum(axis=(1,2)) * self.space.dx * self.space.dz
            np.save(f"{self.path}{self.name}_{tstep:07d}tstep_area", self.Sy_area)

            if h5 == True:

                with h5py.File(f'{self.path}{self.name}_DFTs_rank{self.space.MPIrank:02d}.h5', 'w') as hf:

                    if self.space.engine == 'cupy':
                        hf.create_dataset('Sy_Ex', data=cp.asnumpy(self.DFT_Ex))
                        hf.create_dataset('Sy_Ez', data=cp.asnumpy(self.DFT_Ez))
                        hf.create_dataset('Sy_Hx', data=cp.asnumpy(self.DFT_Hx))
                        hf.create_dataset('Sy_Hz', data=cp.asnumpy(self.DFT_Hz))
                        hf.create_dataset('Sy_area', data=cp.asnumpy(self.Sx_area))

                    else:
                        hf.create_dataset('Sy_Ex', data=self.DFT_Ex)
                        hf.create_dataset('Sy_Ez', data=self.DFT_Ez)
                        hf.create_dataset('Sy_Hx', data=self.DFT_Hx)
                        hf.create_dataset('Sy_Hz', data=self.DFT_Hz)
                        hf.create_dataset('Sy_area', data=self.Sx_area)


class Sz(collector):

    def __init__(self, name, path, space, zloc, srt, end, freqs, engine):
        """Sy collector object.

        Args:
            name: string.

            path: string.

            space: space object.

            zloc: float.

            srt: tuple
                (xsrt, ysrt)

            end: tuple
                (xend, yend)

            freqs: ndarray

            engine: string

        Returns:
            None
        """

        collector.__init__(self, name, path, space, engine)

        self.Nf = len(freqs)
        if self.engine == 'cupy': self.freqs = cp.asarray(freqs)
        else: self.freqs = freqs

        # Start loc of the structure.
        self.xsrt = round(srt[0]/space.dx)
        self.ysrt = round(srt[1]/space.dy)
        self.zsrt = round(  zloc/space.dz)

        # End loc of the structure.
        self.xend = round(end[0]/space.dx)
        self.yend = round(end[1]/space.dz)
        self.zend = self.zsrt + 1

        # Local variables for readable code.
        xsrt = self.xsrt
        ysrt = self.ysrt
        zsrt = self.zsrt
        xend = self.xend
        yend = self.yend
        zend = self.zend

        self.who_get_Sz_gxloc = {} # global locations
        self.who_get_Sz_lxloc = {} # local locations

        # Every node has to know who collects Sz.
        for MPIrank in range(self.space.MPIsize):

            # Global x index of each node.
            node_xsrt = self.space.myNx_indice[MPIrank][0]
            node_xend = self.space.myNx_indice[MPIrank][1]

            if xsrt >  node_xend: pass
            if xend <  node_xsrt: pass
            if xsrt <  node_xsrt and xend > node_xsrt and xend <= node_xend:

                gloc = ((node_xsrt          , ysrt, zsrt), (xend          , yend, zend))
                lloc = ((node_xsrt-node_xsrt, ysrt, zsrt), (xend-node_xsrt, yend, zend))

                self.who_get_Sz_gxloc[MPIrank] = gloc
                self.who_get_Sz_lxloc[MPIrank] = lloc

            if xsrt <  node_xsrt and xend > node_xend:
                gloc = ((node_xsrt          , ysrt, zsrt), (node_xend          , yend, zend))
                lloc = ((node_xsrt-node_xsrt, ysrt, zsrt), (node_xend-node_xsrt, yend, zend))

                self.who_get_Sz_gxloc[MPIrank] = gloc
                self.who_get_Sz_lxloc[MPIrank] = lloc

            if xsrt >= node_xsrt and xsrt < node_xend and xend <= node_xend:
                gloc = ((xsrt          , ysrt, zsrt), (xend          , yend, zend))
                lloc = ((xsrt-node_xsrt, ysrt, zsrt), (xend-node_xsrt, yend, zend))

                self.who_get_Sz_gxloc[MPIrank] = gloc
                self.who_get_Sz_lxloc[MPIrank] = lloc

            if xsrt >= node_xsrt and xsrt < node_xend and xend >  node_xend:
                gloc = ((xsrt          , ysrt, zsrt), (node_xend          , yend, zend))
                lloc = ((xsrt-node_xsrt, ysrt, zsrt), (node_xend-node_xsrt, yend, zend))

                self.who_get_Sz_gxloc[MPIrank] = gloc
                self.who_get_Sz_lxloc[MPIrank] = lloc

        #if self.space.MPIrank == 0: print("{} collectors: rank{}" .format(self.name, list(self.who_get_Sz_gxloc)))

        self.space.MPIcomm.barrier()

        if self.space.MPIrank in self.who_get_Sz_lxloc:

            self.gloc = self.who_get_Sz_gxloc[self.space.MPIrank]
            self.lloc = self.who_get_Sz_lxloc[self.space.MPIrank]

            #print("rank {:>2}: x loc of {} collector >>> global range({:4d},{:4d}) // local range({:4d},{:4d})\"" \
            #      .format(self.space.MPIrank, self.name, self.gloc[0][0], self.gloc[1][0], self.lloc[0][0], self.lloc[1][0]))

            #print("rank {:>2}: y loc of {} collector >>> global range({:4d},{:4d}) // local range({:4d},{:4d})\"" \
            #      .format(self.space.MPIrank, self.name, self.gloc[0][1], self.gloc[1][1], self.lloc[0][1], self.lloc[1][1]))

            #print("rank {:>2}: z loc of {} collector >>> global range({:4d},{:4d}) // local range({:4d},{:4d})\"" \
            #      .format(self.space.MPIrank, self.name, self.gloc[0][2], self.gloc[1][2], self.lloc[0][2], self.lloc[1][2]))

            xsrt = self.lloc[0][0]
            xend = self.lloc[1][0]

            self.DFT_Ex = self.xp.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=np.complex128)
            self.DFT_Ey = self.xp.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=np.complex128)
            self.DFT_Hx = self.xp.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=np.complex128)
            self.DFT_Hy = self.xp.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=np.complex128)
        
    def do_RFT(self, tstep):

        if self.space.MPIrank in self.who_get_Sz_lxloc:

            dt = self.space.dt
            xsrt = self.lloc[0][0]
            xend = self.lloc[1][0]
            ysrt = self.lloc[0][1]
            yend = self.lloc[1][1]
            zsrt = self.lloc[0][2]
            zend = self.lloc[1][2]

            f = (slice(0,None), None, None)
            Fidx = (slice(xsrt,xend), slice(ysrt, yend), zsrt)

            self.DFT_Ex += self.space.Ex[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            self.DFT_Hy += self.space.Hy[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

            self.DFT_Ey += self.space.Ey[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            self.DFT_Hx += self.space.Hx[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

    def get_Sz(self, tstep, h5=False):

        self.space.MPIcomm.barrier()

        if self.space.MPIrank in self.who_get_Sz_lxloc:

            Exname = f"{self.path}{self.name}_DFT_Ex_{tstep:07d}tstep_rank{self.space.MPIrank:02d}"
            Eyname = f"{self.path}{self.name}_DFT_Ey_{tstep:07d}tstep_rank{self.space.MPIrank:02d}"
            Hxname = f"{self.path}{self.name}_DFT_Hx_{tstep:07d}tstep_rank{self.space.MPIrank:02d}"
            Hyname = f"{self.path}{self.name}_DFT_Hy_{tstep:07d}tstep_rank{self.space.MPIrank:02d}"

            self.xp.save(Exname, self.DFT_Ex)
            self.xp.save(Eyname, self.DFT_Ey)
            self.xp.save(Hxname, self.DFT_Hx)
            self.xp.save(Hyname, self.DFT_Hy)

        self.space.MPIcomm.barrier()

        if self.space.MPIrank == 0:

            DFT_Sz_Exs = []
            DFT_Sz_Eys = []
            DFT_Sz_Hxs = []
            DFT_Sz_Hys = []

            for rank in self.who_get_Sz_lxloc:

                Exname = f"{self.path}{self.name}_DFT_Ex_{tstep:07d}tstep_rank{rank:02d}.npy"
                Eyname = f"{self.path}{self.name}_DFT_Ey_{tstep:07d}tstep_rank{rank:02d}.npy"
                Hxname = f"{self.path}{self.name}_DFT_Hx_{tstep:07d}tstep_rank{rank:02d}.npy"
                Hyname = f"{self.path}{self.name}_DFT_Hy_{tstep:07d}tstep_rank{rank:02d}.npy"

                DFT_Sz_Exs.append(np.load(Exname))
                DFT_Sz_Eys.append(np.load(Eyname))
                DFT_Sz_Hxs.append(np.load(Hxname))
                DFT_Sz_Hys.append(np.load(Hyname))

            DFT_Ex = np.concatenate(DFT_Sz_Exs, axis=1)
            DFT_Ey = np.concatenate(DFT_Sz_Eys, axis=1)
            DFT_Hx = np.concatenate(DFT_Sz_Hxs, axis=1)
            DFT_Hy = np.concatenate(DFT_Sz_Hys, axis=1)

            self.Sz = 0.5 * ( -(DFT_Ey.real*DFT_Hx.real) - (DFT_Ey.imag*DFT_Hx.imag)
                              +(DFT_Ex.real*DFT_Hy.real) + (DFT_Ex.imag*DFT_Hy.imag)  )

            self.Sz_area = self.Sz.sum(axis=(1,2)) * self.space.dx * self.space.dy
            np.save(f"{self.path}{self.name}_{tstep:07d}tstep_area" , self.Sz_area)

            if h5 == True:

                with h5py.File(f'{self.path}{self.name}_DFTs_rank{self.space.MPIrank:02d}.h5', 'w') as hf:

                    if self.space.engine == 'cupy':
                        hf.create_dataset('Sz_Ex', data=cp.asnumpy(self.DFT_Ex))
                        hf.create_dataset('Sz_Ey', data=cp.asnumpy(self.DFT_Ey))
                        hf.create_dataset('Sz_Hx', data=cp.asnumpy(self.DFT_Hx))
                        hf.create_dataset('Sz_Hy', data=cp.asnumpy(self.DFT_Hy))
                        hf.create_dataset('Sz_area', data=cp.asnumpy(self.Sx_area))

                    else:
                        hf.create_dataset('Sz_Ex', data=self.DFT_Ex)
                        hf.create_dataset('Sz_Ey', data=self.DFT_Ey)
                        hf.create_dataset('Sz_Hx', data=self.DFT_Hx)
                        hf.create_dataset('Sz_Hy', data=self.DFT_Hy)
                        hf.create_dataset('Sz_area', data=self.Sx_area)
