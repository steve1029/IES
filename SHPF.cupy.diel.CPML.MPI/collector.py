import ctypes, os
from functools import reduce
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

class collector:

    def __init__(self, name, path, space, freqs, engine):

        self.engine = engine
        if self.engine == 'cupy' : self.xp = cp
        else: self.xp = np

        self.name = name
        self.Nf = len(freqs)
        self.space = space
        self.path = path

        if self.engine == 'cupy': self.freqs = cp.asarray(freqs)
        else: self.freqs = freqs

        # Make a directory to save data.
        if self.space.MPIrank == 0:

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

    def __init__(self, name, path, space, loc, freqs, engine):
        """Collector object to collect the fields at a point.

        Args:
            name: string.

            space: space object.

            loc: float
                location of a collector.

            freqs: ndarray

            engine: string
                choose 'numpy' or 'cupy'.

        Returns:
            None
        """

        collector.__init__(self, name, path, space, freqs, engine)

        # Location of the structure.
        self.xloc = int(loc[0]/space.dx)
        self.yloc = int(loc[1]/space.dy)
        self.zloc = int(loc[2]/space.dz)

        self.gxloc, self.lxloc = collector._get_local_x_loc(self, self.xloc, self.xloc+1)

        if self.gxloc != None:

            self.Ex_t = self.xp.zeros(space.tsteps, dtype=space.field_dtype)
            self.Ey_t = self.xp.zeros(space.tsteps, dtype=space.field_dtype)
            self.Ez_t = self.xp.zeros(space.tsteps, dtype=space.field_dtype)

            self.Hx_t = self.xp.zeros(space.tsteps, dtype=space.field_dtype)
            self.Hy_t = self.xp.zeros(space.tsteps, dtype=space.field_dtype)
            self.Hz_t = self.xp.zeros(space.tsteps, dtype=space.field_dtype)

            #self.DFT_Ex = self.xp.zeros(self.Nf, dtype=self.space.field_dtype)
            #self.DFT_Ey = self.xp.zeros(self.Nf, dtype=self.space.field_dtype)
            #self.DFT_Ez = self.xp.zeros(self.Nf, dtype=self.space.field_dtype)

            #self.DFT_Hx = self.xp.zeros(self.Nf, dtype=self.space.field_dtype)
            #self.DFT_Hy = self.xp.zeros(self.Nf, dtype=self.space.field_dtype)
            #self.DFT_Hz = self.xp.zeros(self.Nf, dtype=self.space.field_dtype)

    def get_field(self, tstep):

        if self.gxloc != None:

            dt = self.space.dt
            xsrt = self.lxloc[0]
            xend = self.lxloc[1]

            f = slice(0,None)
            Fidx = [self.xloc, self.yloc, self.zloc]

            self.Ex_t[tstep] = self.space.Ex[self.xloc, self.yloc, self.zloc]
            self.Ey_t[tstep] = self.space.Ey[self.xloc, self.yloc, self.zloc]
            self.Ez_t[tstep] = self.space.Ez[self.xloc, self.yloc, self.zloc]

            self.Hx_t[tstep] = self.space.Hx[self.xloc, self.yloc, self.zloc]
            self.Hy_t[tstep] = self.space.Hy[self.xloc, self.yloc, self.zloc]
            self.Hz_t[tstep] = self.space.Hz[self.xloc, self.yloc, self.zloc]

            #self.DFT_Ex += self.space.Ex[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            #self.DFT_Ey += self.space.Ey[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            #self.DFT_Ez += self.space.Ez[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

            #self.DFT_Hx += self.space.Hx[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            #self.DFT_Hy += self.space.Hy[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            #self.DFT_Hz += self.space.Hz[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

    def get_spectrum(self):

        self.space.MPIcomm.barrier()

        if self.gxloc != None:

            self.Ex_w = self.xp.fft.fft(self.Ex_t)
            self.Ey_w = self.xp.fft.fft(self.Ey_t)
            self.Ez_w = self.xp.fft.fft(self.Ez_t)

            self.Hx_w = self.xp.fft.fft(self.Hx_t)
            self.Hy_w = self.xp.fft.fft(self.Hy_t)
            self.Hz_w = self.xp.fft.fft(self.Hz_t)

            self.Ex_w_shift = self.xp.fft.fftshift(self.Ex_w)
            self.Ey_w_shift = self.xp.fft.fftshift(self.Ey_w)
            self.Ez_w_shift = self.xp.fft.fftshift(self.Ez_w)

            self.Hx_w_shift = self.xp.fft.fftshift(self.Hx_w)
            self.Hy_w_shift = self.xp.fft.fftshift(self.Hy_w)
            self.Hz_w_shift = self.xp.fft.fftshift(self.Hz_w)

            self.xp.save("{}/{}_Ex_t_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.Ex_t)
            self.xp.save("{}/{}_Ey_t_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.Ey_t)
            self.xp.save("{}/{}_Ez_t_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.Ez_t)

            self.xp.save("{}/{}_Hx_t_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.Hx_t)
            self.xp.save("{}/{}_Hy_t_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.Hy_t)
            self.xp.save("{}/{}_Hz_t_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.Hz_t)

            self.xp.save("{}/{}_Ex_w_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.Ex_w)
            self.xp.save("{}/{}_Ey_w_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.Ey_w)
            self.xp.save("{}/{}_Ez_w_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.Ez_w)

            self.xp.save("{}/{}_Hx_w_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.Hx_w)
            self.xp.save("{}/{}_Hy_w_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.Hy_w)
            self.xp.save("{}/{}_Hz_w_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.Hz_w)

    def plot_spectrum(self):

        self.Ex_w_shift = cp.asnumpy(self.Ex_w_shift)
        self.Ey_w_shift = cp.asnumpy(self.Ey_w_shift)
        self.Ez_w_shift = cp.asnumpy(self.Ez_w_shift)

        self.Hx_w_shift = cp.asnumpy(self.Hx_w_shift)
        self.Hy_w_shift = cp.asnumpy(self.Hy_w_shift)
        self.Hz_w_shift = cp.asnumpy(self.Hz_w_shift)

        fftfreq = np.fft.fftfreq(len(self.Ex_t), self.space.dt)

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24,16))

        axes[0,0].plot(fftfreq, abs(self.Ex_w_shift), label='Ex_w')
        axes[0,1].plot(fftfreq, abs(self.Ey_w_shift), label='Ey_w')
        axes[0,2].plot(fftfreq, abs(self.Ez_w_shift), label='Ez_w')

        axes[1,0].plot(fftfreq, abs(self.Hx_w_shift), label='Hx_w')
        axes[1,1].plot(fftfreq, abs(self.Hy_w_shift), label='Hy_w')
        axes[1,2].plot(fftfreq, abs(self.Hz_w_shift), label='Hz_w')

        axes[0,0].legend(loc='best')
        axes[0,1].legend(loc='best')
        axes[0,2].legend(loc='best')

        axes[1,0].legend(loc='best')
        axes[1,1].legend(loc='best')
        axes[1,2].legend(loc='best')

        fig.savefig("../graph/field_at_point.png")

 
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

        collector.__init__(self, name, path, space, freqs, engine)

        # Start loc of the structure.
        self.xsrt = int(xloc  /space.dx)
        self.ysrt = int(srt[0]/space.dy)
        self.zsrt = int(srt[1]/space.dz)

        # End loc of the structure.
        self.xend = self.xsrt + 1
        self.yend = int(end[0]/space.dy)
        self.zend = int(end[1]/space.dz)

        self.gxloc, self.lxloc = collector._get_local_x_loc(self, self.xsrt, self.xend)

        if self.gxloc != None:

            #print("rank {:>2}: loc of Sx collector >>> global \"{},{}\" and local \"{},{}\"" \
            #      .format(self.space.MPIrank, self.gloc[0], self.gloc[1], self.lloc[0], self.lloc[1]))

            self.DFT_Ey = self.xp.zeros((self.Nf, self.yend-self.ysrt, self.zend-self.zsrt), dtype=self.space.field_dtype)
            self.DFT_Ez = self.xp.zeros((self.Nf, self.yend-self.ysrt, self.zend-self.zsrt), dtype=self.space.field_dtype)

            self.DFT_Hy = self.xp.zeros((self.Nf, self.yend-self.ysrt, self.zend-self.zsrt), dtype=self.space.field_dtype)
            self.DFT_Hz = self.xp.zeros((self.Nf, self.yend-self.ysrt, self.zend-self.zsrt), dtype=self.space.field_dtype)

    def do_RFT(self, tstep):

        if self.gxloc != None:

            dt = self.space.dt
            xsrt = self.lxloc[0]
            xend = self.lxloc[1]

            f = [slice(0,None), None, None]
            Fidx = [slice(xsrt,xend), slice(self.ysrt, self.yend), slice(self.zsrt, self.zend)]

            self.DFT_Ey += self.space.Ey[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            self.DFT_Hz += self.space.Hz[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

            self.DFT_Ez += self.space.Ez[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            self.DFT_Hy += self.space.Hy[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

    def get_Sx(self):

        self.space.MPIcomm.barrier()

        if self.gxloc != None:

            self.Sx = 0.5 * (  (self.DFT_Ey.real*self.DFT_Hz.real) + (self.DFT_Ey.imag*self.DFT_Hz.imag)
                              -(self.DFT_Ez.real*self.DFT_Hy.real) - (self.DFT_Ez.imag*self.DFT_Hy.imag)  )

            self.Sx_area = self.Sx.sum(axis=(1,2)) * self.space.dy * self.space.dz

            self.xp.save("{}/{}_DFT_Ey_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.DFT_Ey)
            self.xp.save("{}/{}_DFT_Ez_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.DFT_Ez)
            self.xp.save("{}/{}_DFT_Hy_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.DFT_Hy)
            self.xp.save("{}/{}_DFT_Hz_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.DFT_Hz)
            self.xp.save("./graph/%s_area" %self.name, self.Sx_area)


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

        collector.__init__(self, name, path, space, freqs, engine)

        # Start loc of the structure.
        self.xsrt = int(srt[0]/space.dx)
        self.ysrt = int(  yloc/space.dy)
        self.zsrt = int(srt[1]/space.dz)

        # End loc of the structure.
        self.xend = int(end[0]/space.dx)
        self.yend = self.ysrt+1
        self.zend = int(end[1]/space.dz)

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
        for MPIrank in range(self.space.MPIsize):

            # Global x index of each node.
            node_xsrt = self.space.myNx_indice[MPIrank][0]
            node_xend = self.space.myNx_indice[MPIrank][1]

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

        #if self.space.MPIrank == 0: print("{} collectors: rank{}" .format(self.name, list(self.who_get_Sy_gloc)))

        self.space.MPIcomm.barrier()

        if self.space.MPIrank in self.who_get_Sy_lloc:

            self.gloc = self.who_get_Sy_gloc[self.space.MPIrank]
            self.lloc = self.who_get_Sy_lloc[self.space.MPIrank]

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

            self.DFT_Ex = self.xp.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=self.space.field_dtype)
            self.DFT_Ez = self.xp.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=self.space.field_dtype)

            self.DFT_Hx = self.xp.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=self.space.field_dtype)
            self.DFT_Hz = self.xp.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=self.space.field_dtype)
        
        #print(self.who_get_Sy_gloc)
        #print(self.who_get_Sy_lloc)

    def do_RFT(self, tstep):

        if self.space.MPIrank in self.who_get_Sy_lloc:

            dt = self.space.dt
            xsrt = self.lloc[0][0]
            xend = self.lloc[1][0]
            ysrt = self.lloc[0][1]
            yend = self.lloc[1][1]
            zsrt = self.lloc[0][2]
            zend = self.lloc[1][2]

            f = [slice(0,None), None, None]
            Fidx = [slice(xsrt,xend), ysrt, slice(zsrt, zend)]

            self.DFT_Ex += self.space.Ex[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            self.DFT_Hz += self.space.Hz[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

            self.DFT_Ez += self.space.Ez[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            self.DFT_Hx += self.space.Hx[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

    def get_Sy(self):

        self.space.MPIcomm.barrier()

        if self.space.MPIrank in self.who_get_Sy_lloc:

            self.xp.save("{}/{}_DFT_Ex_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.DFT_Ex)
            self.xp.save("{}/{}_DFT_Ez_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.DFT_Ez)
            self.xp.save("{}/{}_DFT_Hx_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.DFT_Hx)
            self.xp.save("{}/{}_DFT_Hz_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.DFT_Hz)

        self.space.MPIcomm.barrier()

        if self.space.MPIrank == 0:

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

            self.Sy_area = self.Sy.sum(axis=(1,2)) * self.space.dx * self.space.dz
            np.save("./graph/%s_area" %self.name, self.Sy_area)


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

        collector.__init__(self, name, path, space, freqs, engine)

        # Start loc of the structure.
        self.xsrt = int(srt[0]/space.dx)
        self.ysrt = int(srt[1]/space.dy)
        self.zsrt = int(  zloc/space.dz)

        # End loc of the structure.
        self.xend = int(end[0]/space.dx)
        self.yend = int(end[1]/space.dz)
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
        for MPIrank in range(self.space.MPIsize):

            # Global x index of each node.
            node_xsrt = self.space.myNx_indice[MPIrank][0]
            node_xend = self.space.myNx_indice[MPIrank][1]

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

        #if self.space.MPIrank == 0: print("{} collectors: rank{}" .format(self.name, list(self.who_get_Sz_gloc)))

        self.space.MPIcomm.barrier()

        if self.space.MPIrank in self.who_get_Sz_lloc:

            self.gloc = self.who_get_Sz_gloc[self.space.MPIrank]
            self.lloc = self.who_get_Sz_lloc[self.space.MPIrank]

            #print("rank {:>2}: x loc of {} collector >>> global range({:4d},{:4d}) // local range({:4d},{:4d})\"" \
            #      .format(self.space.MPIrank, self.name, self.gloc[0][0], self.gloc[1][0], self.lloc[0][0], self.lloc[1][0]))

            #print("rank {:>2}: y loc of {} collector >>> global range({:4d},{:4d}) // local range({:4d},{:4d})\"" \
            #      .format(self.space.MPIrank, self.name, self.gloc[0][1], self.gloc[1][1], self.lloc[0][1], self.lloc[1][1]))

            #print("rank {:>2}: z loc of {} collector >>> global range({:4d},{:4d}) // local range({:4d},{:4d})\"" \
            #      .format(self.space.MPIrank, self.name, self.gloc[0][2], self.gloc[1][2], self.lloc[0][2], self.lloc[1][2]))

            xsrt = self.lloc[0][0]
            xend = self.lloc[1][0]

            self.DFT_Ex = self.xp.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=self.space.field_dtype)
            self.DFT_Ey = self.xp.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=self.space.field_dtype)
            self.DFT_Hx = self.xp.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=self.space.field_dtype)
            self.DFT_Hy = self.xp.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=self.space.field_dtype)
        
    def do_RFT(self, tstep):

        if self.space.MPIrank in self.who_get_Sz_lloc:

            dt = self.space.dt
            xsrt = self.lloc[0][0]
            xend = self.lloc[1][0]
            ysrt = self.lloc[0][1]
            yend = self.lloc[1][1]
            zsrt = self.lloc[0][2]
            zend = self.lloc[1][2]

            f = [slice(0,None), None, None]
            Fidx = [slice(xsrt,xend), slice(ysrt, yend), zsrt]

            self.DFT_Ex += self.space.Ex[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            self.DFT_Hy += self.space.Hy[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

            self.DFT_Ey += self.space.Ey[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt
            self.DFT_Hx += self.space.Hx[Fidx] * self.xp.exp(2.j*self.xp.pi*self.freqs[f]*tstep*dt) * dt

    def get_Sz(self):

        self.space.MPIcomm.barrier()

        if self.space.MPIrank in self.who_get_Sz_lloc:

            self.xp.save("{}/{}_DFT_Ex_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.DFT_Ex)
            self.xp.save("{}/{}_DFT_Ey_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.DFT_Ey)
            self.xp.save("{}/{}_DFT_Hx_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.DFT_Hx)
            self.xp.save("{}/{}_DFT_Hy_rank{:02d}" .format(self.path, self.name, self.space.MPIrank), self.DFT_Hy)

        self.space.MPIcomm.barrier()

        if self.space.MPIrank == 0:

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

            self.Sz_area = self.Sz.sum(axis=(1,2)) * self.space.dx * self.space.dy
            self.xp.save("./graph/%s_area" %self.name, self.Sz_area)
