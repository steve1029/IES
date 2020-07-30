import ctypes, os
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

class Sx(object):

    def __init__(self, name, path, Space, srt, end, freqs, omp_on):
        """Sx collector object.

        Args:
            name: string.

            Space: Space object.

            srt: tuple

            end: tuple

            freqs: ndarray

            omp_on: boolean

        Returns:
            None
        """

        assert type(srt) == tuple
        assert type(end) == tuple
        
        assert len(srt) == 3
        assert len(end) == 3

        assert (end[0]-srt[0]) == 1, "Sx Collector must have 2D shape with x-thick = 1."

        self.name = name
        self.freqs = freqs

        if type(self.freqs) == np.ndarray: self.Nf = len(self.freqs)
        else: self.Nf = 1

        self.Space = Space
        self.path = path

        # Make save directory.
        if self.Space.MPIrank == 0:

            if os.path.exists(self.path) == True: pass
            else: os.mkdir(self.path)

        # Turn on/off OpenMP parallelization.
        self.omp_on = omp_on

        # Start index of the structure.
        self.xsrt = srt[0]
        self.ysrt = srt[1]
        self.zsrt = srt[2]

        # End index of the structure.
        self.xend = end[0]
        self.yend = end[1]
        self.zend = end[2]

        # Local variables for readable code.
        xsrt = self.xsrt
        ysrt = self.ysrt
        zsrt = self.zsrt
        xend = self.xend
        yend = self.yend
        zend = self.zend

        # Global x index of each node.
        node_xsrt = self.Space.myNx_indice[self.Space.MPIrank][0]
        node_xend = self.Space.myNx_indice[self.Space.MPIrank][1]

        self.gloc = None
        self.lloc = None

        if xend <  node_xsrt:
            self.gloc = None
            self.lloc = None
        if xsrt <  node_xsrt and xend > node_xsrt and xend <= node_xend:
            self.gloc = ((node_xsrt          , ysrt, zsrt), (       xend          , yend, zend))
            self.lloc = ((node_xsrt-node_xsrt, ysrt, zsrt), (       xend-node_xsrt, yend, zend))
        if xsrt <  node_xsrt and xend > node_xend:
            self.gloc = ((node_xsrt          , ysrt, zsrt), (node_xend          , yend, zend))
            self.lloc = ((node_xsrt-node_xsrt, ysrt, zsrt), (node_xend-node_xsrt, yend, zend))
        if xsrt >= node_xsrt and xsrt < node_xend and xend <= node_xend:
            self.gloc = ((xsrt          , ysrt, zsrt), (        xend          , yend, zend))
            self.lloc = ((xsrt-node_xsrt, ysrt, zsrt), (        xend-node_xsrt, yend, zend))
        if xsrt >= node_xsrt and xsrt < node_xend and xend >  node_xend:
            self.gloc = ((xsrt          , ysrt, zsrt), (node_xend          , yend, zend))
            self.lloc = ((xsrt-node_xsrt, ysrt, zsrt), (node_xend-node_xsrt, yend, zend))
        if xsrt >  node_xend:
            self.gloc = None
            self.lloc = None

        if self.gloc != None:

            #print("rank {:>2}: loc of Sx collector >>> global \"{},{}\" and local \"{},{}\"" \
            #      .format(self.Space.MPIrank, self.gloc[0], self.gloc[1], self.lloc[0], self.lloc[1]))

            self.DFT_Ey_re = np.zeros((self.Nf, yend-ysrt, zend-zsrt), dtype=self.Space.dtype)
            self.DFT_Ey_im = np.zeros((self.Nf, yend-ysrt, zend-zsrt), dtype=self.Space.dtype)

            self.DFT_Ez_re = np.zeros((self.Nf, yend-ysrt, zend-zsrt), dtype=self.Space.dtype)
            self.DFT_Ez_im = np.zeros((self.Nf, yend-ysrt, zend-zsrt), dtype=self.Space.dtype)

            self.DFT_Hy_re = np.zeros((self.Nf, yend-ysrt, zend-zsrt), dtype=self.Space.dtype)
            self.DFT_Hy_im = np.zeros((self.Nf, yend-ysrt, zend-zsrt), dtype=self.Space.dtype)

            self.DFT_Hz_re = np.zeros((self.Nf, yend-ysrt, zend-zsrt), dtype=self.Space.dtype)
            self.DFT_Hz_im = np.zeros((self.Nf, yend-ysrt, zend-zsrt), dtype=self.Space.dtype)

        # Load kernel.
        if   self.omp_on == False: self.clib_rftkernel = ctypes.cdll.LoadLibrary("./rftkernel.so")
        elif self.omp_on == True : self.clib_rftkernel = ctypes.cdll.LoadLibrary("./rftkernel.omp.so")
        else: raise ValueError("Choose True or False")

        ptr1d = np.ctypeslib.ndpointer(dtype=self.Space.dtype, ndim=1, flags='C_CONTIGUOUS')
        ptr2d = np.ctypeslib.ndpointer(dtype=self.Space.dtype, ndim=2, flags='C_CONTIGUOUS')
        ptr3d = np.ctypeslib.ndpointer(dtype=self.Space.dtype, ndim=3, flags='C_CONTIGUOUS')

        self.clib_rftkernel.do_RFT_to_get_Sx.restype    = None
        self.clib_rftkernel.do_RFT_to_get_Sx.argtypes = [
                                                            ctypes.c_int,
                                                            ctypes.c_int, ctypes.c_int,
                                                            ctypes.c_int, ctypes.c_int,
                                                            ctypes.c_int, ctypes.c_int,
                                                            ctypes.c_int, ctypes.c_int,
                                                            ctypes.c_int, ctypes.c_int,
                                                            ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                                            ptr1d,
                                                            ptr3d, ptr3d,
                                                            ptr3d, ptr3d,
                                                            ptr3d, ptr3d,
                                                            ptr3d, ptr3d,
                                                            ptr3d, ptr3d,
                                                            ptr3d, ptr3d
                                                         ]
    def do_RFT(self, tstep):

        if self.gloc != None:

            self.clib_rftkernel.do_RFT_to_get_Sx(
                                                    self.Space.MPIrank,
                                                    self.Nf, tstep,
                                                    self.Space.Ny, self.Space.Nz,
                                                    self.lloc[0][0], self.lloc[1][0],
                                                    self.ysrt, self.yend,
                                                    self.zsrt, self.zend,
                                                    self.Space.dt, self.Space.dy, self.Space.dz,
                                                    self.freqs,
                                                    self.DFT_Ey_re, self.DFT_Ez_re,
                                                    self.DFT_Ey_im, self.DFT_Ez_im,
                                                    self.DFT_Hy_re, self.DFT_Hz_re,
                                                    self.DFT_Hy_im, self.DFT_Hz_im,
                                                    self.Space.Ey_re, self.Space.Ez_re,
                                                    self.Space.Hy_re, self.Space.Hz_re
                                                )

    def get_Sx(self):

        self.Space.MPIcomm.barrier()

        if self.gloc != None:

            np.save("{}/{}_DFT_Ey_re_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Ey_re)
            np.save("{}/{}_DFT_Ez_re_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Ez_re)
            np.save("{}/{}_DFT_Hy_re_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Hy_re)
            np.save("{}/{}_DFT_Hz_re_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Hz_re)

            np.save("{}/{}_DFT_Ey_im_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Ey_im)
            np.save("{}/{}_DFT_Ez_im_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Ez_im)
            np.save("{}/{}_DFT_Hy_im_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Hy_im)
            np.save("{}/{}_DFT_Hz_im_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Hz_im)

            self.Sx = 0.5 * (  (self.DFT_Ey_re*self.DFT_Hz_re) + (self.DFT_Ey_im*self.DFT_Hz_im)
                              -(self.DFT_Ez_re*self.DFT_Hy_re) - (self.DFT_Ez_im*self.DFT_Hy_im)  )

            self.Sx_area = self.Sx.sum(axis=(1,2)) * self.Space.dy * self.Space.dz
            np.save("./graph/%s_area" %self.name, self.Sx_area)


class Sy(object):

    def __init__(self, name, path, Space, srt, end, freqs, omp_on):
        """Sy collector object.

        Args:
            name: string.

            Space: Space object.

            srt: tuple

            end: tuple

            freqs: ndarray

            omp_on: boolean

        Returns:
            None
        """

        assert type(srt) == tuple
        assert type(end) == tuple
        
        assert len(srt) == 3
        assert len(end) == 3

        assert (end[1]-srt[1]) == 1, "Sy Collector must have 2D shape with y-thick = 1."

        self.name = name
        self.freqs = freqs

        if type(self.freqs) == np.ndarray: self.Nf = len(self.freqs)
        else: self.Nf = 1

        self.Space = Space
        self.path = path

        # Make save directory.
        if self.Space.MPIrank == 0:

            if os.path.exists(self.path) == True: pass
            else: os.mkdir(self.path)

        # Turn on/off OpenMP parallelization.
        self.omp_on = omp_on

        # Start index of the structure.
        self.xsrt = srt[0]
        self.ysrt = srt[1]
        self.zsrt = srt[2]

        # End index of the structure.
        self.xend = end[0]
        self.yend = end[1]
        self.zend = end[2]

        # Local variables for readable code.
        xsrt = self.xsrt
        ysrt = self.ysrt
        zsrt = self.zsrt
        xend = self.xend
        yend = self.yend
        zend = self.zend

        self.who_get_Sy_gloc = {} # global locations
        self.who_get_Sy_lloc = {} # local locations

        self.gloc = None
        self.lloc = None

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

            self.DFT_Ex_re = np.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=self.Space.dtype)
            self.DFT_Ex_im = np.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=self.Space.dtype)

            self.DFT_Ez_re = np.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=self.Space.dtype)
            self.DFT_Ez_im = np.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=self.Space.dtype)

            self.DFT_Hx_re = np.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=self.Space.dtype)
            self.DFT_Hx_im = np.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=self.Space.dtype)

            self.DFT_Hz_re = np.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=self.Space.dtype)
            self.DFT_Hz_im = np.zeros((self.Nf, xend-xsrt, zend-zsrt), dtype=self.Space.dtype)
        
        # Load kernel.
        if   self.omp_on == False: self.clib_rftkernel = ctypes.cdll.LoadLibrary("./rftkernel.so")
        elif self.omp_on == True : self.clib_rftkernel = ctypes.cdll.LoadLibrary("./rftkernel.omp.so")
        else: raise ValueError("Choose True or False")

        ptr1d = np.ctypeslib.ndpointer(dtype=self.Space.dtype, ndim=1, flags='C_CONTIGUOUS')
        ptr2d = np.ctypeslib.ndpointer(dtype=self.Space.dtype, ndim=2, flags='C_CONTIGUOUS')
        ptr3d = np.ctypeslib.ndpointer(dtype=self.Space.dtype, ndim=3, flags='C_CONTIGUOUS')

        self.clib_rftkernel.do_RFT_to_get_Sy.restype  = None
        self.clib_rftkernel.do_RFT_to_get_Sy.argtypes = [
                                                            ctypes.c_int,
                                                            ctypes.c_int, ctypes.c_int,
                                                            ctypes.c_int, ctypes.c_int,
                                                            ctypes.c_int, ctypes.c_int,
                                                            ctypes.c_int, ctypes.c_int,
                                                            ctypes.c_int, ctypes.c_int,
                                                            ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                                            ptr1d,
                                                            ptr3d, ptr3d,
                                                            ptr3d, ptr3d,
                                                            ptr3d, ptr3d,
                                                            ptr3d, ptr3d,
                                                            ptr3d, ptr3d,
                                                            ptr3d, ptr3d
                                                         ]

        #print(self.who_get_Sy_gloc)
        #print(self.who_get_Sy_lloc)

    def do_RFT(self, tstep):

        if self.Space.MPIrank in self.who_get_Sy_lloc:

            self.clib_rftkernel.do_RFT_to_get_Sy(
                                                    self.Space.MPIrank,
                                                    self.Nf, tstep,
                                                    self.Space.Ny, self.Space.Nz,
                                                    self.lloc[0][0], self.lloc[1][0],
                                                    self.lloc[0][1], self.lloc[1][1],
                                                    self.lloc[0][2], self.lloc[1][2],
                                                    self.Space.dt, self.Space.dy, self.Space.dz,
                                                    self.freqs,
                                                    self.DFT_Ex_re, self.DFT_Ez_re,
                                                    self.DFT_Ex_im, self.DFT_Ez_im,
                                                    self.DFT_Hx_re, self.DFT_Hz_re,
                                                    self.DFT_Hx_im, self.DFT_Hz_im,
                                                    self.Space.Ex_re, self.Space.Ez_re,
                                                    self.Space.Hx_re, self.Space.Hz_re
                                                )

    def get_Sy(self):

        self.Space.MPIcomm.barrier()

        if self.Space.MPIrank in self.who_get_Sy_lloc:

            np.save("{}/{}_DFT_Ex_re_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Ex_re)
            np.save("{}/{}_DFT_Ez_re_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Ez_re)
            np.save("{}/{}_DFT_Hx_re_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Hx_re)
            np.save("{}/{}_DFT_Hz_re_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Hz_re)

            np.save("{}/{}_DFT_Ex_im_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Ex_im)
            np.save("{}/{}_DFT_Ez_im_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Ez_im)
            np.save("{}/{}_DFT_Hx_im_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Hx_im)
            np.save("{}/{}_DFT_Hz_im_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Hz_im)

        self.Space.MPIcomm.barrier()

        if self.Space.MPIrank == 0:

            DFT_Sy_Ex_res = []
            DFT_Sy_Ex_ims = []

            DFT_Sy_Ez_res = []
            DFT_Sy_Ez_ims = []

            DFT_Sy_Hx_res = []
            DFT_Sy_Hx_ims = []

            DFT_Sy_Hz_res = []
            DFT_Sy_Hz_ims = []

            for rank in self.who_get_Sy_lloc:

                DFT_Sy_Ex_res.append(np.load("{}/{}_DFT_Ex_re_rank{:02d}.npy" .format(self.path, self.name, rank)))
                DFT_Sy_Ex_ims.append(np.load("{}/{}_DFT_Ex_im_rank{:02d}.npy" .format(self.path, self.name, rank)))

                DFT_Sy_Ez_res.append(np.load("{}/{}_DFT_Ez_re_rank{:02d}.npy" .format(self.path, self.name, rank)))
                DFT_Sy_Ez_ims.append(np.load("{}/{}_DFT_Ez_im_rank{:02d}.npy" .format(self.path, self.name, rank)))

                DFT_Sy_Hx_res.append(np.load("{}/{}_DFT_Hx_re_rank{:02d}.npy" .format(self.path, self.name, rank)))
                DFT_Sy_Hx_ims.append(np.load("{}/{}_DFT_Hx_im_rank{:02d}.npy" .format(self.path, self.name, rank)))

                DFT_Sy_Hz_res.append(np.load("{}/{}_DFT_Hz_re_rank{:02d}.npy" .format(self.path, self.name, rank)))
                DFT_Sy_Hz_ims.append(np.load("{}/{}_DFT_Hz_im_rank{:02d}.npy" .format(self.path, self.name, rank)))

            DFT_Ex_re = np.concatenate(DFT_Sy_Ex_res, axis=1)
            DFT_Ex_im = np.concatenate(DFT_Sy_Ex_ims, axis=1)

            DFT_Ez_re = np.concatenate(DFT_Sy_Ez_res, axis=1)
            DFT_Ez_im = np.concatenate(DFT_Sy_Ez_ims, axis=1)

            DFT_Hx_re = np.concatenate(DFT_Sy_Hx_res, axis=1)
            DFT_Hx_im = np.concatenate(DFT_Sy_Hx_ims, axis=1)

            DFT_Hz_re = np.concatenate(DFT_Sy_Hz_res, axis=1)
            DFT_Hz_im = np.concatenate(DFT_Sy_Hz_ims, axis=1)

            self.Sy = 0.5 * ( -(DFT_Ex_re*DFT_Hz_re) - (DFT_Ex_im*DFT_Hz_im)
                              +(DFT_Ez_re*DFT_Hx_re) + (DFT_Ez_im*DFT_Hx_im)  )

            self.Sy_area = self.Sy.sum(axis=(1,2)) * self.Space.dx * self.Space.dz
            np.save("./graph/%s_area" %self.name, self.Sy_area)



class Sz(object):

    def __init__(self, name, path, Space, srt, end, freqs, omp_on):
        """Sy collector object.

        Args:
            name: string.

            path: string.

            Space: Space object.

            srt: tuple

            end: tuple

            freqs: ndarray

            omp_on: boolean

        Returns:
            None
        """

        assert type(srt) == tuple
        assert type(end) == tuple
        
        assert len(srt) == 3
        assert len(end) == 3

        assert (end[2]-srt[2]) == 1, "Sz Collector must have 2D shape with z-thick = 1."

        self.name = name
        self.freqs = freqs
        if type(self.freqs) == np.ndarray: self.Nf = len(self.freqs)
        else: self.Nf = 1

        self.Space = Space
        self.path = path

        # Make save directory.
        if self.Space.MPIrank == 0:

            if os.path.exists(self.path) == True: pass
            else: os.mkdir(self.path)

        # Turn on/off OpenMP parallelization.
        self.omp_on = omp_on

        # Start index of the structure.
        self.xsrt = srt[0]
        self.ysrt = srt[1]
        self.zsrt = srt[2]

        # End index of the structure.
        self.xend = end[0]
        self.yend = end[1]
        self.zend = end[2]

        # Local variables for readable code.
        xsrt = self.xsrt
        ysrt = self.ysrt
        zsrt = self.zsrt
        xend = self.xend
        yend = self.yend
        zend = self.zend

        self.who_get_Sz_gloc = {} # global locations
        self.who_get_Sz_lloc = {} # local locations

        self.gloc = None
        self.lloc = None

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

            self.DFT_Ex_re = np.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=self.Space.dtype)
            self.DFT_Ex_im = np.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=self.Space.dtype)

            self.DFT_Ey_re = np.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=self.Space.dtype)
            self.DFT_Ey_im = np.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=self.Space.dtype)

            self.DFT_Hx_re = np.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=self.Space.dtype)
            self.DFT_Hx_im = np.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=self.Space.dtype)

            self.DFT_Hy_re = np.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=self.Space.dtype)
            self.DFT_Hy_im = np.zeros((self.Nf, xend-xsrt, yend-ysrt), dtype=self.Space.dtype)
        
        # Load kernel.
        if   self.omp_on == False: self.clib_rftkernel = ctypes.cdll.LoadLibrary("./rftkernel.so")
        elif self.omp_on == True : self.clib_rftkernel = ctypes.cdll.LoadLibrary("./rftkernel.omp.so")
        else: raise ValueError("Choose True or False")

        ptr1d = np.ctypeslib.ndpointer(dtype=self.Space.dtype, ndim=1, flags='C_CONTIGUOUS')
        ptr2d = np.ctypeslib.ndpointer(dtype=self.Space.dtype, ndim=2, flags='C_CONTIGUOUS')
        ptr3d = np.ctypeslib.ndpointer(dtype=self.Space.dtype, ndim=3, flags='C_CONTIGUOUS')

        self.clib_rftkernel.do_RFT_to_get_Sz.restype  = None
        self.clib_rftkernel.do_RFT_to_get_Sz.argtypes = [
                                                            ctypes.c_int,
                                                            ctypes.c_int, ctypes.c_int,
                                                            ctypes.c_int, ctypes.c_int,
                                                            ctypes.c_int, ctypes.c_int,
                                                            ctypes.c_int, ctypes.c_int,
                                                            ctypes.c_int, ctypes.c_int,
                                                            ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                                            ptr1d,
                                                            ptr3d, ptr3d,
                                                            ptr3d, ptr3d,
                                                            ptr3d, ptr3d,
                                                            ptr3d, ptr3d,
                                                            ptr3d, ptr3d,
                                                            ptr3d, ptr3d
                                                         ]

    def do_RFT(self, tstep):

        if self.Space.MPIrank in self.who_get_Sz_lloc:

            self.clib_rftkernel.do_RFT_to_get_Sz(
                                                    self.Space.MPIrank,
                                                    len(self.freqs), tstep,
                                                    self.Space.Ny, self.Space.Nz,
                                                    self.lloc[0][0], self.lloc[1][0],
                                                    self.lloc[0][1], self.lloc[1][1],
                                                    self.lloc[0][2], self.lloc[1][2],
                                                    self.Space.dt, self.Space.dx, self.Space.dy,
                                                    self.freqs,
                                                    self.DFT_Ex_re, self.DFT_Ey_re,
                                                    self.DFT_Ex_im, self.DFT_Ey_im,
                                                    self.DFT_Hx_re, self.DFT_Hy_re,
                                                    self.DFT_Hx_im, self.DFT_Hy_im,
                                                    self.Space.Ex_re, self.Space.Ey_re,
                                                    self.Space.Hx_re, self.Space.Hy_re
                                                )

    def get_Sz(self):

        self.Space.MPIcomm.barrier()

        if self.Space.MPIrank in self.who_get_Sz_lloc:

            np.save("{}/{}_DFT_Ex_re_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Ex_re)
            np.save("{}/{}_DFT_Ey_re_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Ey_re)
            np.save("{}/{}_DFT_Hx_re_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Hx_re)
            np.save("{}/{}_DFT_Hy_re_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Hy_re)

            np.save("{}/{}_DFT_Ex_im_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Ex_im)
            np.save("{}/{}_DFT_Ey_im_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Ey_im)
            np.save("{}/{}_DFT_Hx_im_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Hx_im)
            np.save("{}/{}_DFT_Hy_im_rank{:02d}" .format(self.path, self.name, self.Space.MPIrank), self.DFT_Hy_im)

        self.Space.MPIcomm.barrier()

        if self.Space.MPIrank == 0:

            DFT_Sz_Ex_res = []
            DFT_Sz_Ex_ims = []

            DFT_Sz_Ey_res = []
            DFT_Sz_Ey_ims = []

            DFT_Sz_Hx_res = []
            DFT_Sz_Hx_ims = []

            DFT_Sz_Hy_res = []
            DFT_Sz_Hy_ims = []

            for rank in self.who_get_Sz_lloc:

                DFT_Sz_Ex_res.append(np.load("{}/{}_DFT_Ex_re_rank{:02d}.npy" .format(self.path, self.name, rank)))
                DFT_Sz_Ex_ims.append(np.load("{}/{}_DFT_Ex_im_rank{:02d}.npy" .format(self.path, self.name, rank)))

                DFT_Sz_Ey_res.append(np.load("{}/{}_DFT_Ey_re_rank{:02d}.npy" .format(self.path, self.name, rank)))
                DFT_Sz_Ey_ims.append(np.load("{}/{}_DFT_Ey_im_rank{:02d}.npy" .format(self.path, self.name, rank)))

                DFT_Sz_Hx_res.append(np.load("{}/{}_DFT_Hx_re_rank{:02d}.npy" .format(self.path, self.name, rank)))
                DFT_Sz_Hx_ims.append(np.load("{}/{}_DFT_Hx_im_rank{:02d}.npy" .format(self.path, self.name, rank)))

                DFT_Sz_Hy_res.append(np.load("{}/{}_DFT_Hy_re_rank{:02d}.npy" .format(self.path, self.name, rank)))
                DFT_Sz_Hy_ims.append(np.load("{}/{}_DFT_Hy_im_rank{:02d}.npy" .format(self.path, self.name, rank)))

            DFT_Ex_re = np.concatenate(DFT_Sz_Ex_res, axis=1)
            DFT_Ex_im = np.concatenate(DFT_Sz_Ex_ims, axis=1)

            DFT_Ey_re = np.concatenate(DFT_Sz_Ey_res, axis=1)
            DFT_Ey_im = np.concatenate(DFT_Sz_Ey_ims, axis=1)

            DFT_Hx_re = np.concatenate(DFT_Sz_Hx_res, axis=1)
            DFT_Hx_im = np.concatenate(DFT_Sz_Hx_ims, axis=1)

            DFT_Hy_re = np.concatenate(DFT_Sz_Hy_res, axis=1)
            DFT_Hy_im = np.concatenate(DFT_Sz_Hy_ims, axis=1)

            self.Sz = 0.5 * ( -(DFT_Ey_re*DFT_Hx_re) - (DFT_Ey_im*DFT_Hx_im)
                              +(DFT_Ex_re*DFT_Hy_re) + (DFT_Ex_im*DFT_Hy_im)  )

            self.Sz_area = self.Sz.sum(axis=(1,2)) * self.Space.dx * self.Space.dy
            np.save("./graph/%s_area" %self.name, self.Sz_area)
