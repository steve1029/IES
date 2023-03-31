import os, datetime, sys
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import c, epsilon_0, mu_0

class Graphtool(object):

    def __init__(self, Space, name, path):

        self.Space = Space
        self.name = name
        self.savedir = path

        if self.Space.MPIrank == 0 : 

            while (os.path.exists(path) == False):

                print("Directory you put does not exists")
                path = input()
                
                if os.path.exists(path) == True: break
                else: continue

            if os.path.exists(self.savedir) == False: os.makedirs(self.savedir)
            else: pass

    def gather(self, what):
        """
        Gather the data resident in rank >0 to rank 0.
        """
        ###################################################################################
        ###################### Gather field data from all slave nodes #####################
        ###################################################################################
        
        if self.Space.engine == 'cupy':

            if   what == 'Ex': 
                Ex = cp.asnumpy(self.Space.Ex)
                gathered = self.Space.MPIcomm.gather(Ex, root=0)
            elif what == 'Ey': 
                Ey = cp.asnumpy(self.Space.Ey)
                gathered = self.Space.MPIcomm.gather(Ey, root=0)
            elif what == 'Ez': 
                Ez = cp.asnumpy(self.Space.Ez)
                gathered = self.Space.MPIcomm.gather(Ez, root=0)
            elif what == 'Hx': 
                Hx = cp.asnumpy(self.Space.Hx)
                gathered = self.Space.MPIcomm.gather(Hx, root=0)
            elif what == 'Hy': 
                Hy = cp.asnumpy(self.Space.Hy)
                gathered = self.Space.MPIcomm.gather(Hy, root=0)
            elif what == 'Hz': 
                Hz = cp.asnumpy(self.Space.Hz)
                gathered = self.Space.MPIcomm.gather(Hz, root=0)

        else:
            if   what == 'Ex': gathered = self.Space.MPIcomm.gather(self.Space.Ex, root=0)
            elif what == 'Ey': gathered = self.Space.MPIcomm.gather(self.Space.Ey, root=0)
            elif what == 'Ez': gathered = self.Space.MPIcomm.gather(self.Space.Ez, root=0)
            elif what == 'Hx': gathered = self.Space.MPIcomm.gather(self.Space.Hx, root=0)
            elif what == 'Hy': gathered = self.Space.MPIcomm.gather(self.Space.Hy, root=0)
            elif what == 'Hz': gathered = self.Space.MPIcomm.gather(self.Space.Hz, root=0)

        self.what = what

        if self.Space.MPIrank == 0: 
        
            self.integrated = np.zeros((self.Space.grid), dtype=self.Space.field_dtype)

            for MPIrank in range(self.Space.MPIsize):
                if self.Space.dimension == 3: self.integrated[self.Space.myNx_slices[MPIrank],:,:] = gathered[MPIrank]
                if self.Space.dimension == 2: self.integrated[self.Space.myNx_slices[MPIrank],:] = gathered[MPIrank]

                #if MPIrank == 1: print(MPIrank, gathered[MPIrank][xidx,yidx,zidx])

            return self.integrated

        else: return None

    def plot2D3D(self, integrated, tstep, xidx=None, yidx=None, zidx=None, **kwargs):

        if self.Space.MPIrank == 0: 

            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import axes3d
                from mpl_toolkits.axes_grid1 import make_axes_locatable

            except ImportError as err:
                print("Please install matplotlib at rank 0")
                sys.exit()

            colordeep = .1
            stride = 1
            zlim = 1
            figsize = (18, 8)
            cmap = plt.cm.bwr
            lc = 'b'
            aspect = 'auto'
            savenpy = False

            for key, value in kwargs.items():

                if   key == 'colordeep': colordeep = value
                elif key == 'figsize': figsize = value
                elif key == 'aspect': aspect = value
                elif key == 'stride': stride = value
                elif key == 'what': self.what = value
                elif key == 'zlim': zlim = value
                elif key == 'cmap': cmap = value
                elif key == 'lc': lc = value
                elif key == 'savenpy': savenpy = value

            #if kwargs.get('colordeep') != None: colordeep = kwargs.get('colordeep')

            #####################################################################################
            ######### Build up total field with the parts of the grid from slave nodes ##########
            #####################################################################################

            if self.Space.dimension == 3:

                if xidx != None: 
                    assert type(xidx) == int
                    yidx  = slice(None,None) # indices from beginning to end
                    zidx  = slice(None,None)
                    plane = 'yz'
                    col = np.arange(self.Space.Ny)
                    row = np.arange(self.Space.Nz)
                    #plane_to_plot = np.zeros((len(row),len(col)), dtype=np.float32)

                elif yidx != None :
                    assert type(yidx) == int
                    xidx  = slice(None,None)
                    zidx  = slice(None,None)
                    plane = 'xz'
                    col = np.arange(self.Space.Nx)
                    row = np.arange(self.Space.Nz)
                    #plane_to_plot = np.zeros((len(row), len(col)), dtype=np.float32)

                elif zidx != None :
                    assert type(zidx) == int
                    xidx  = slice(None,None)
                    yidx  = slice(None,None)
                    plane = 'xy'
                    col = np.arange(self.Space.Nx)
                    row = np.arange(self.Space.Ny)
                    #plane_to_plot = np.zeros((len(row),len(col)), dtype=np.float32)
            
                elif (xidx,yidx,zidx) == (None,None,None):
                    raise ValueError("Plane is not defined. Please insert one of x,y or z index of the plane.")

                if integrated.dtype == np.complex64 or integrated.dtype == np.complex128:
                    self.plane_to_plot = integrated[xidx, yidx, zidx].real
                else: self.plane_to_plot = integrated[xidx, yidx, zidx]

            elif self.Space.dimension == 2:

                xidx  = slice(None,None)
                yidx  = slice(None,None)
                plane = 'xy'
                col = np.arange(self.Space.Nx)
                row = np.arange(self.Space.Ny)
                plane_to_plot = np.zeros((len(row),len(col)), dtype=np.float32)
        
                if integrated.dtype == np.complex64 or integrated.dtype == np.complex128:
                    self.plane_to_plot = integrated[xidx, yidx].real
                else: self.plane_to_plot = integrated[xidx, yidx]

            X, Y = np.meshgrid(col, row, indexing='ij', sparse=False)
            today = datetime.date.today()

            fig  = plt.figure(figsize=figsize)
            ax11 = fig.add_subplot(1,2,1)
            ax12 = fig.add_subplot(1,2,2, projection='3d')

            if plane == 'yz':

                image11 = ax11.imshow(self.plane_to_plot.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)
                ax12.plot_wireframe(Y, X, self.plane_to_plot[X, Y], color=lc, rstride=stride, cstride=stride)

                divider11 = make_axes_locatable(ax11)

                cax11  = divider11.append_axes('right', size='5%', pad=0.1)
                cbar11 = fig.colorbar(image11, cax=cax11)

                ax11.invert_yaxis()
                #ax12.invert_yaxis()

                ax11.set_xlabel('y')
                ax11.set_ylabel('z')
                ax12.set_xlabel('y')
                ax12.set_ylabel('z')

            elif plane == 'xy':

                image11 = ax11.imshow(self.plane_to_plot.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)
                ax12.plot_wireframe(X, Y, self.plane_to_plot[X, Y], color=lc, rstride=stride, cstride=stride)

                divider11 = make_axes_locatable(ax11)

                cax11  = divider11.append_axes('right', size='5%', pad=0.1)
                cbar11 = fig.colorbar(image11, cax=cax11)

                ax11.invert_yaxis()
                #ax12.invert_yaxis()

                ax11.set_xlabel('x')
                ax11.set_ylabel('y')
                ax12.set_xlabel('x')
                ax12.set_ylabel('y')

            elif plane == 'xz':

                #print(self.name, "here?")
                #print(self.plane_to_plot.shape)
                #print(X.shape)
                #print(Y.shape)
                #print(col.shape)
                #print(row.shape)
                #print(X)
                #print(Y)

                image11 = ax11.imshow(self.plane_to_plot.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)
                ax12.plot_wireframe(X, Y, self.plane_to_plot[X, Y], color=lc, rstride=stride, cstride=stride)
                divider11 = make_axes_locatable(ax11)

                cax11  = divider11.append_axes('right', size='5%', pad=0.1)
                cbar11 = fig.colorbar(image11, cax=cax11)

                #ax11.invert_yaxis()
                #ax12.invert_yaxis()
                #ax12.invert_xaxis()

                ax11.set_xlabel('x')
                ax11.set_ylabel('z')
                ax12.set_xlabel('x')
                ax12.set_ylabel('z')

            if savenpy == True: 
            
                saveloc = f'{self.savedir}{plane}_profile/'
                if os.path.exists(saveloc) == False: os.makedirs(saveloc)
                np.save(saveloc+f'{tstep:07d}', self.plane_to_plot.T)

            ax11.set_title(r'$%s.real, 2D$' %self.what)
            ax12.set_title(r'$%s.real, 3D$' %self.what)

            ax12.set_zlim(-zlim,zlim)
            ax12.set_zlabel('field')

            foldername = 'plot2D3D/'
            save_dir   = self.savedir + foldername

            if os.path.exists(save_dir) == False: os.makedirs(save_dir)
            plt.tight_layout()
            #fig.savefig('%s%s_%s_%s_%s_%s.png' %(save_dir, str(today), self.name, self.what, plane, tstep), format='png', bbox_inches='tight')
            fig.savefig(f'{save_dir}{str(today)}_{self.name}_{self.what}_{plane}_{tstep:07d}.png', format='png', bbox_inches='tight')
            plt.close('all')


class SpectrumPlotter(object):

    def __init__(self, method, cells, wavelength, freq_unit, wvlen_unit):

        self.method = method
        self.cells = cells
        self.Nx = self.cells[0]
        self.Ny = self.cells[1]
        self.Nz = self.cells[2]
        self.wvlen_unit = wvlen_unit
        self.freq_unit = freq_unit

        self.wvlens = wavelength
        self.freqs = c / wavelength

        if   wvlen_unit == 'mm' : self.wvlens = self.wvlens / 1e-3
        elif wvlen_unit == 'um' : self.wvlens = self.wvlens / 1e-6
        elif wvlen_unit == 'nm' : self.wvlens = self.wvlens / 1e-9
        elif wvlen_unit == 'pm' : self.wvlens = self.wvlens / 1e-12
        else: raise ValueError("Please specify the length unit")

        if   freq_unit == 'THz' : self.freqs = self.freqs / 1e12
        elif freq_unit == 'GHz' : self.freqs = self.freqs / 1e9
        elif freq_unit == 'MHz' : self.freqs = self.freqs / 1e6
        elif freq_unit == 'KHz' : self.freqs = self.freqs / 1e3
        else: raise ValueError("Please specify the frequency unit")

    def simple_plot(self, spectrum, name):
        """Plot spectrum of the Poynting vector.

        Args:

            spectrum: a list of string
                location of the numpy ndarray. ex) spectrum = ['./graph/S_rank02.npy', './graph/S_rank03.npy']

            name: a string
                image file name and location. ex) './graph/spectrum.png'

        Returns:
            None
        """

        self.spectrum = np.zeros(len(self.freqs), dtype=np.float64)

        for data in spectrum:
            loaded = np.load(data)
            self.spectrum += abs(loaded)

        #print(self.spectrum)

        #self.spectrum = abs(self.spectrum)

        fig = plt.figure(figsize=(21,9))

        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)

        ax1.plot(self.freqs, self.spectrum)
        ax1.grid(True)
        ax1.set_xlabel("freqs({})" .format(self.freq_unit))
        ax1.set_ylabel(r"Sx$\times$Area(W)")
        #ax1.set_ylim(2.6e-49, 3.3e-49)

        ax2.plot(self.wvlens, self.spectrum)
        ax2.grid(True)
        ax2.set_xlabel("wavelength({})" .format(self.wvlen_unit))
        ax2.set_ylabel(r"Sx$\times$Area(W)")
        #ax2.set_ylim(2.6e-49, 3.3e-49)

        fig.savefig(name)

    def plot_IRT(self, incs, refs, trss, tsteps, savedir, \
                wvxlim, wvylim, freqxlim, freqylim,\
                plot_trs=True, plot_ref=True, plot_sum=True):
        """Plot transmittance and reflectance.

        Parameters
        ----------
        incs: a list of str.
        trss: a list of str.
        refs: a list of str.

        Returns
        -------
        None
        """
        self.tsteps = tsteps

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,8))

        inc = np.zeros(len(self.freqs), dtype=np.float64)
        trs = np.zeros(len(self.freqs), dtype=np.float64)
        ref = np.zeros(len(self.freqs), dtype=np.float64)

        for data in incs: inc+=abs(np.load(data))
        #axes[0].plot(self.freqs, inc, label='incidence')
        #axes[1].plot(self.wvlens, inc, label='incidence')

        if plot_trs == True:
            for data in trss: trs+=abs(np.load(data))
            trs /= inc
            axes[0].plot(self.freqs, trs, label='Trs')
            axes[1].plot(self.wvlens, trs, label='Trs')

        if plot_ref == True:
            for data in refs: ref+=abs(np.load(data))
            ref /= inc
            axes[0].plot(self.freqs, ref, label='Ref')
            axes[1].plot(self.wvlens, ref, label='Ref')

        if plot_sum == True:
            axes[0].plot(self.freqs, ref+trs, label='Sum')
            axes[1].plot(self.wvlens, ref+trs, label='Sum')

        axes[0].grid(True)
        axes[0].set_xlabel("freqs({})" .format(self.freq_unit))
        axes[0].set_ylabel('Ratio')
        axes[0].legend(loc='best')
        axes[0].set_xlim(wvxlim[0], wvxlim[1])
        axes[0].set_ylim(wvylim[0], wvylim[1])
        axes[0].set_title('freq vs TRS')

        axes[1].grid(True)
        axes[1].set_xlabel("wavelength({})" .format(self.wvlen_unit))
        axes[1].set_ylabel('Ratio')
        axes[1].legend(loc='best')
        axes[1].set_xlim(freqxlim[0], freqxlim[1])
        axes[1].set_ylim(freqylim[0], freqylim[1])
        axes[1].set_title('wvlen vs TRS')

        fig.suptitle('{} {} {}'.format(self.method, self.tsteps, self.cells))
        #fig.tight_layout()
        fig.savefig(savedir, bbox_inches='tight')
        plt.close('all')
