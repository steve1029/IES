import ctypes, os
from functools import reduce
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

class S_calculator:

    def __init__(self, name, path, Space, freqs, engine):

        self.engine = engine
        if self.engine == 'cupy' : self.xp = cp
        else: self.xp = np

        self.name = name
        self.Nf = len(freqs)
        self.Space = Space
        self.path = path

        if self.engine == 'cupy': self.freqs = cp.asarray(freqs)
        else: self.freqs = freqs

        # Make a directory to save data.
        if self.Space.MPIrank == 0:

            if os.path.exists(self.path) == True: pass
            else: os.mkdir(self.path)

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

 
class Sx(S_calculator):

    def __init__(self, name, path, Space, xloc, srt, end, freqs, engine):
        """Sx collector object.

        Args:
            name: string.

            Space: Space object.

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

        S_calculator.__init__(self, name, path, Space, freqs, engine)

        # Start loc of the structure.
        self.xsrt = int(xloc  /Space.dx)
        self.ysrt = int(srt[0]/Space.dy)
        self.zsrt = int(srt[1]/Space.dz)

        # End loc of the structure.
        self.xend = self.xsrt + 1
        self.yend = int(end[0]/Space.dy)
        self.zend = int(end[1]/Space.dz)

        self.gxloc, self.lxloc = S_calculator._get_local_x_loc(self, self.xsrt, self.xend)

        if self.gxloc != None:

            #print("rank {:>2}: loc of Sx collector >>> global \"{},{}\" and local \"{},{}\"" \
            #      .format(self.Space.MPIrank, self.gloc[0], self.gloc[1], self.lloc[0], self.lloc[1]))

            self.DFT_Ey = self.xp.zeros((self.Nf, self.yend-self.ysrt, self.zend-self.zsrt), dtype=self.Space.cdtype)
            self.DFT_Ez = self.xp.zeros((self.Nf, self.yend-self.ysrt, self.zend-self.zsrt), dtype=self.Space.cdtype)

            self.DFT_Hy = self.xp.zeros((self.Nf, self.yend-self.ysrt, self.zend-self.zsrt), dtype=self.Space.cdtype)
            self.DFT_Hz = self.xp.zeros((self.Nf, self.yend-self.ysrt, self.zend-self.zsrt), dtype=self.Space.cdtype)

    def do_RFT(self, tstep):

        if self.gxloc != None:

            dt = self.Space.dt
            xsrt = self.lxloc[0]
            xend = self.lxloc[1]

            f = [slice(0,None), None, None]
            Fidx = [slice(xsrt,xend), slice(self.ysrt, self.yend), slice(self.zsrt, self.zend)]

            self.DFT_Ey += self.Space.Ey[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            self.DFT_Hz += self.Space.Hz[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

            self.DFT_Ez += self.Space.Ez[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            self.DFT_Hy += self.Space.Hy[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

    def get_Sx(self):

        self.Space.MPIcomm.barrier()

        if self.gxloc != None:

            self.Sx = 0.5 * (  (self.DFT_Ey.real*self.DFT_Hz.real) + (self.DFT_Ey.imag*self.DFT_Hz.imag)
                              -(self.DFT_Ez.real*self.DFT_Hy.real) - (self.DFT_Ez.imag*self.DFT_Hy.imag)  )

            self.Sx_area = self.Sx.sum(axis=(1,2)) * self.Space.dy * self.Space.dz

            self.xp.save("{}/{}_DFT_Ey_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Ey)
            self.xp.save("{}/{}_DFT_Ez_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Ez)
            self.xp.save("{}/{}_DFT_Hy_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Hy)
            self.xp.save("{}/{}_DFT_Hz_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Hz)
            self.xp.save("./graph/%s_area" %self.name, self.Sx_area)


class Sy(S_calculator):

    def __init__(self, name, path, Space, yloc, srt, end, freqs, engine):
        """Sy collector object.

        Args:
            name: string.

            Space: Space object.

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

        S_calculator.__init__(self, name, path, Space, freqs, engine)

        # Start loc of the structure.
        self.xsrt = int(srt[0]/Space.dx)
        self.ysrt = int(  yloc/Space.dy)
        self.zsrt = int(srt[1]/Space.dz)

        # End loc of the structure.
        self.xend = int(end[0]/Space.dx)
        self.yend = self.ysrt+1
        self.zend = int(end[1]/Space.dz)

        # Local variables for readable code.
        xsrt = self.xsrt
        ysrt = self.ysrt
        zsrt = self.zsrt
        xend = self.xend
        yend = self.yend
        zend = self.zend

        self.who_get_Sy_gloc = {} # global locations
        self.who_get_Sy_lloc = {} # local locations

        # Every node has to know who collects Sy.
        for MPIrank in range(self.Space.MPIsize):

            # Global x index of each node.
            node_xsrt = self.Space.myNx_indice[MPIrank][0]
            node_xend = self.Space.myNx_indice[MPIrank][1]

            if xsrt >  node_xend: pass
            if xend <  node_xsrt: pass
            if xsrt <  node_xsrt and xend > node_xsrt and xend <= node_xend:

                gloc = ((node_xsrt          , ysrt, zsrt), (xend          , yend, zend))
                lloc = ((node_xsrt-node_xsrt, ysrt, zsrt), (xend-node_xsrt, yend, zend))

                self.who_get_Sy_gloc[MPIrank] = gloc
                self.who_get_Sy_lloc[MPIrank] = lloc

            if xsrt <  node_xsrt and xend > node_xend:
                gloc = ((node_xsrt          , ysrt, zsrt), (node_xend          , yend, zend))
                lloc = ((node_xsrt-node_xsrt, ysrt, zsrt), (node_xend-node_xsrt, yend, zend))

                self.who_get_Sy_gloc[MPIrank] = gloc
                self.who_get_Sy_lloc[MPIrank] = lloc

            if xsrt >= node_xsrt and xsrt < node_xend and xend <= node_xend:
                gloc = ((xsrt          , ysrt, zsrt), (xend          , yend, zend))
                lloc = ((xsrt-node_xsrt, ysrt, zsrt), (xend-node_xsrt, yend, zend))

                self.who_get_Sy_gloc[MPIrank] = gloc
                self.who_get_Sy_lloc[MPIrank] = lloc

            if xsrt >= node_xsrt and xsrt < node_xend and xend >  node_xend:
                gloc = ((xsrt          , ysrt, zsrt), (node_xend          , yend, zend))
                lloc = ((xsrt-node_xsrt, ysrt, zsrt), (node_xend-node_xsrt, yend, zend))

                self.who_get_Sy_gloc[MPIrank] = gloc
                self.who_get_Sy_lloc[MPIrank] = lloc

        #if self.Space.MPIrank == 0: print("{} collectors: rank{}" .format(self.name, list(self.who_get_Sy_gloc)))

        self.Space.MPIcomm.barrier()

        if self.Space.MPIrank in self.who_get_Sy_lloc:

            self.gloc = self.who_get_Sy_gloc[self.Space.MPIrank]
            self.lloc = self.who_get_Sy_lloc[self.Space.MPIrank]

            """
            print("rank {:>2}: x loc of {} collector >>> global range({:4d},{:4d}) // local range({:4d},{:4d})\"" \
                   .format(self.Space.MPIrank, self.name, self.gloc[0][0], self.gloc[1][0], self.lloc[0][0], self.lloc[1][0]))

            print("rank {:>2}: y loc of {} collector >>> global range({:4d},{:4d}) // local range({:4d},{:4d})\"" \
                   .format(self.Space.MPIrank, self.name, self.gloc[0][1], self.gloc[1][1], self.lloc[0][1], self.lloc[1][1]))

            print("rank {:>2}: z loc of {} collector >>> global range({:4d},{:4d}) // local range({:4d},{:4d})\"" \
                   .format(self.Space.MPIrank, self.name, self.gloc[0][2], self.gloc[1][2], self.lloc[0][2], self.lloc[1][2]))
            """

            xsrt = self.lloc[0][0]
            xend = self.lloc[1][0]

            self.DFT_Ex = self.xp.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=self.Space.cdtype)
            self.DFT_Ez = self.xp.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=self.Space.cdtype)

            self.DFT_Hx = self.xp.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=self.Space.cdtype)
            self.DFT_Hz = self.xp.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=self.Space.cdtype)
        
        #print(self.who_get_Sy_gloc)
        #print(self.who_get_Sy_lloc)

    def do_RFT(self, tstep):

        if self.Space.MPIrank in self.who_get_Sy_lloc:

            dt = self.Space.dt
            xsrt = self.lloc[0][0]
            xend = self.lloc[1][0]
            ysrt = self.lloc[0][1]
            yend = self.lloc[1][1]
            zsrt = self.lloc[0][2]
            zend = self.lloc[1][2]

            f = [slice(0,None), None, None]
            Fidx = [slice(xsrt,xend), ysrt, slice(zsrt, zend)]

            self.DFT_Ex += self.Space.Ex[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            self.DFT_Hz += self.Space.Hz[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

            self.DFT_Ez += self.Space.Ez[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            self.DFT_Hx += self.Space.Hx[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

    def get_Sy(self):

        self.Space.MPIcomm.barrier()

        if self.Space.MPIrank in self.who_get_Sy_lloc:

            self.xp.save("{}/{}_DFT_Ex_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Ex)
            self.xp.save("{}/{}_DFT_Ez_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Ez)
            self.xp.save("{}/{}_DFT_Hx_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Hx)
            self.xp.save("{}/{}_DFT_Hz_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Hz)

        self.Space.MPIcomm.barrier()

        if self.Space.MPIrank == 0:

            DFT_Sy_Exs = []
            DFT_Sy_Ezs = []

            DFT_Sy_Hxs = []
            DFT_Sy_Hzs = []

            for rank in self.who_get_Sy_lloc:

                DFT_Sy_Exs.append(np.load("{}/{}_DFT_Ex_rank{:02d}.npy" .format(self.path, self.name, rank)))
                DFT_Sy_Ezs.append(np.load("{}/{}_DFT_Ez_rank{:02d}.npy" .format(self.path, self.name, rank)))
                DFT_Sy_Hxs.append(np.load("{}/{}_DFT_Hx_rank{:02d}.npy" .format(self.path, self.name, rank)))
                DFT_Sy_Hzs.append(np.load("{}/{}_DFT_Hz_rank{:02d}.npy" .format(self.path, self.name, rank)))

            DFT_Ex = np.concatenate(DFT_Sy_Exs, axis=1)
            DFT_Ez = np.concatenate(DFT_Sy_Ezs, axis=1)
            DFT_Hx = np.concatenate(DFT_Sy_Hxs, axis=1)
            DFT_Hz = np.concatenate(DFT_Sy_Hzs, axis=1)

            self.Sy = 0.5 * ( -(DFT_Ex.real*DFT_Hz.real) - (DFT_Ex.imag*DFT_Hz.imag)
                              +(DFT_Ez.real*DFT_Hx.real) + (DFT_Ez.imag*DFT_Hx.imag)  )

            self.Sy_area = self.Sy.sum(axis=(1,2)) * self.Space.dx * self.Space.dz
            np.save("./graph/%s_area" %self.name, self.Sy_area)


class Sz(S_calculator):

    def __init__(self, name, path, Space, zloc, srt, end, freqs, engine):
        """Sy collector object.

        Args:
            name: string.

            path: string.

            Space: Space object.

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

        S_calculator.__init__(self, name, path, Space, freqs, engine)

        # Start loc of the structure.
        self.xsrt = int(srt[0]/Space.dx)
        self.ysrt = int(srt[1]/Space.dy)
        self.zsrt = int(  zloc/Space.dz)

        # End loc of the structure.
        self.xend = int(end[0]/Space.dx)
        self.yend = int(end[1]/Space.dz)
        self.zend = self.zsrt + 1

        # Local variables for readable code.
        xsrt = self.xsrt
        ysrt = self.ysrt
        zsrt = self.zsrt
        xend = self.xend
        yend = self.yend
        zend = self.zend

        self.who_get_Sz_gloc = {} # global locations
        self.who_get_Sz_lloc = {} # local locations

        # Every node has to know who collects Sz.
        for MPIrank in range(self.Space.MPIsize):

            # Global x index of each node.
            node_xsrt = self.Space.myNx_indice[MPIrank][0]
            node_xend = self.Space.myNx_indice[MPIrank][1]

            if xsrt >  node_xend: pass
            if xend <  node_xsrt: pass
            if xsrt <  node_xsrt and xend > node_xsrt and xend <= node_xend:

                gloc = ((node_xsrt          , ysrt, zsrt), (xend          , yend, zend))
                lloc = ((node_xsrt-node_xsrt, ysrt, zsrt), (xend-node_xsrt, yend, zend))

                self.who_get_Sz_gloc[MPIrank] = gloc
                self.who_get_Sz_lloc[MPIrank] = lloc

            if xsrt <  node_xsrt and xend > node_xend:
                gloc = ((node_xsrt          , ysrt, zsrt), (node_xend          , yend, zend))
                lloc = ((node_xsrt-node_xsrt, ysrt, zsrt), (node_xend-node_xsrt, yend, zend))

                self.who_get_Sz_gloc[MPIrank] = gloc
                self.who_get_Sz_lloc[MPIrank] = lloc

            if xsrt >= node_xsrt and xsrt < node_xend and xend <= node_xend:
                gloc = ((xsrt          , ysrt, zsrt), (xend          , yend, zend))
                lloc = ((xsrt-node_xsrt, ysrt, zsrt), (xend-node_xsrt, yend, zend))

                self.who_get_Sz_gloc[MPIrank] = gloc
                self.who_get_Sz_lloc[MPIrank] = lloc

            if xsrt >= node_xsrt and xsrt < node_xend and xend >  node_xend:
                gloc = ((xsrt          , ysrt, zsrt), (node_xend          , yend, zend))
                lloc = ((xsrt-node_xsrt, ysrt, zsrt), (node_xend-node_xsrt, yend, zend))

                self.who_get_Sz_gloc[MPIrank] = gloc
                self.who_get_Sz_lloc[MPIrank] = lloc

        #if self.Space.MPIrank == 0: print("{} collectors: rank{}" .format(self.name, list(self.who_get_Sz_gloc)))

        self.Space.MPIcomm.barrier()

        if self.Space.MPIrank in self.who_get_Sz_lloc:

            self.gloc = self.who_get_Sz_gloc[self.Space.MPIrank]
            self.lloc = self.who_get_Sz_lloc[self.Space.MPIrank]

            #print("rank {:>2}: x loc of {} collector >>> global range({:4d},{:4d}) // local range({:4d},{:4d})\"" \
            #      .format(self.Space.MPIrank, self.name, self.gloc[0][0], self.gloc[1][0], self.lloc[0][0], self.lloc[1][0]))

            #print("rank {:>2}: y loc of {} collector >>> global range({:4d},{:4d}) // local range({:4d},{:4d})\"" \
            #      .format(self.Space.MPIrank, self.name, self.gloc[0][1], self.gloc[1][1], self.lloc[0][1], self.lloc[1][1]))

            #print("rank {:>2}: z loc of {} collector >>> global range({:4d},{:4d}) // local range({:4d},{:4d})\"" \
            #      .format(self.Space.MPIrank, self.name, self.gloc[0][2], self.gloc[1][2], self.lloc[0][2], self.lloc[1][2]))

            xsrt = self.lloc[0][0]
            xend = self.lloc[1][0]

            self.DFT_Ex = self.xp.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=self.Space.cdtype)
            self.DFT_Ey = self.xp.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=self.Space.cdtype)
            self.DFT_Hx = self.xp.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=self.Space.cdtype)
            self.DFT_Hy = self.xp.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=self.Space.cdtype)
        
    def do_RFT(self, tstep):

        if self.Space.MPIrank in self.who_get_Sz_lloc:

            dt = self.Space.dt
            xsrt = self.lloc[0][0]
            xend = self.lloc[1][0]
            ysrt = self.lloc[0][1]
            yend = self.lloc[1][1]
            zsrt = self.lloc[0][2]
            zend = self.lloc[1][2]

            f = [slice(0,None), None, None]
            Fidx = [slice(xsrt,xend), slice(ysrt, yend), zsrt]

            self.DFT_Ex += self.Space.Ex[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            self.DFT_Hy += self.Space.Hy[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

            self.DFT_Ey += self.Space.Ey[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            self.DFT_Hx += self.Space.Hx[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

    def get_Sz(self):

        self.Space.MPIcomm.barrier()

        if self.Space.MPIrank in self.who_get_Sz_lloc:

            self.xp.save("{}/{}_DFT_Ex_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Ex)
            self.xp.save("{}/{}_DFT_Ey_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Ey)
            self.xp.save("{}/{}_DFT_Hx_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Hx)
            self.xp.save("{}/{}_DFT_Hy_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Hy)

        self.Space.MPIcomm.barrier()

        if self.Space.MPIrank == 0:

            DFT_Sz_Exs = []
            DFT_Sz_Eys = []
            DFT_Sz_Hxs = []
            DFT_Sz_Hys = []

            for rank in self.who_get_Sz_lloc:

                DFT_Sz_Exs.append(self.xp.load("{}/{}_DFT_Ex_rank{:02d}.npy" .format(self.path, self.name, rank)))
                DFT_Sz_Eys.append(self.xp.load("{}/{}_DFT_Ey_rank{:02d}.npy" .format(self.path, self.name, rank)))
                DFT_Sz_Hxs.append(self.xp.load("{}/{}_DFT_Hx_rank{:02d}.npy" .format(self.path, self.name, rank)))
                DFT_Sz_Hys.append(self.xp.load("{}/{}_DFT_Hy_rank{:02d}.npy" .format(self.path, self.name, rank)))

            DFT_Ex = self.xp.concatenate(DFT_Sz_Exs, axis=1)
            DFT_Ey = self.xp.concatenate(DFT_Sz_Eys, axis=1)
            DFT_Hx = self.xp.concatenate(DFT_Sz_Hxs, axis=1)
            DFT_Hy = self.xp.concatenate(DFT_Sz_Hys, axis=1)

            self.Sz = 0.5 * ( -(DFT_Ey.real*DFT_Hx.real) - (DFT_Ey.imag*DFT_Hx.imag)
                              +(DFT_Ex.real*DFT_Hy.real) + (DFT_Ex.imag*DFT_Hy.imag)  )

            self.Sz_area = self.Sz.sum(axis=(1,2)) * self.Space.dx * self.Space.dy
            self.xp.save("./graph/%s_area" %self.name, self.Sz_area)
