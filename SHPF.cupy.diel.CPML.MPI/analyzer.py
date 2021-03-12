import os, sys
import numpy as np
import harminv as hv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import c

class SpectrumAnalyzer:
    """Spectrum analysis with Filter Diagonalization Method (FDM) or Fast Fourier Transform (FFT).
    We use Harminv package for FDM implemented by Steven G. Johnson, who is
    also widely known as a developer of the FFTW package.
    """

    def __init__(self, loaddir, savedir, name, **kwargs):

        self.cname = name # name of the collector. ex) fap1.
        self.savedir = savedir
        self.loaddir = loaddir
        binary = True
        txt = False

        if kwargs.get('binary') != None: 

            hey = kwargs.get('binary')

            assert hey == True or hey == False
            self.binary =  hey

        if kwargs.get('txt') != None: 

            hey = kwargs.get('txt')

            assert hey == True or hey == False
            self.txt =  hey

        self.Ext_npyname = self.loaddir+"{}_Ex_t.npy" .format(name)
        self.Eyt_npyname = self.loaddir+"{}_Ey_t.npy" .format(name)
        self.Ezt_npyname = self.loaddir+"{}_Ez_t.npy" .format(name)

        self.Hxt_npyname = self.loaddir+"{}_Hx_t.npy" .format(name)
        self.Hyt_npyname = self.loaddir+"{}_Hy_t.npy" .format(name)
        self.Hzt_npyname = self.loaddir+"{}_Hz_t.npy" .format(name)

        self.Ext_txtname = self.loaddir+"{}_Ex_t.txt" .format(name)
        self.Eyt_txtname = self.loaddir+"{}_Ey_t.txt" .format(name)
        self.Ezt_txtname = self.loaddir+"{}_Ez_t.txt" .format(name)

        self.Hxt_txtname = self.loaddir+"{}_Hx_t.txt" .format(name)
        self.Hyt_txtname = self.loaddir+"{}_Hy_t.txt" .format(name)
        self.Hzt_txtname = self.loaddir+"{}_Hz_t.txt" .format(name)

        # Load time domain signal.
        if binary == True:

            self.Ex_t = np.load(self.Ext_npyname)
            self.Ey_t = np.load(self.Eyt_npyname)
            self.Ez_t = np.load(self.Ezt_npyname)

            self.Hx_t = np.load(self.Hxt_npyname)
            self.Hy_t = np.load(self.Hyt_npyname)
            self.Hz_t = np.load(self.Hzt_npyname)

        elif txt == True:

            self.Ex_t = np.load(self.Ext_txtname)
            self.Ey_t = np.load(self.Eyt_txtname)
            self.Ez_t = np.load(self.Ezt_txtname)

            self.Hx_t = np.load(self.Hxt_txtname)
            self.Hy_t = np.load(self.Hyt_txtname)
            self.Hz_t = np.load(self.Hzt_txtname)

    def normalized_freq(self, freqs, lattice_constant):

        self.lc = lattice_constant

        return freqs * self.lc / c

    def use_fft(self, dt, lc, **kwargs):

        self.binary = False
        self.txt = False
        self.csv = False

        if kwargs.get('binary') != None: 

            hey = kwargs.get('binary')

            assert hey == True or hey == False
            self.binary =  hey

        if kwargs.get('txt') != None: 

            hey = kwargs.get('txt')

            assert hey == True or hey == False
            self.txt =  hey

        if kwargs.get('csv') != None: 

            hey = kwargs.get('csv')

            assert hey == True or hey == False
            self.csv =  hey

        self.dt = dt
        self.Ex_w = np.fft.fft(self.Ex_t)
        self.Ey_w = np.fft.fft(self.Ey_t)
        self.Ez_w = np.fft.fft(self.Ez_t)

        self.Hx_w = np.fft.fft(self.Hx_t)
        self.Hy_w = np.fft.fft(self.Hy_t)
        self.Hz_w = np.fft.fft(self.Hz_t)

        self.Exw_npyname = "{}/{}_Ex_w_fft.npy" .format(self.savedir, self.cname)
        self.Eyw_npyname = "{}/{}_Ey_w_fft.npy" .format(self.savedir, self.cname)
        self.Ezw_npyname = "{}/{}_Ez_w_fft.npy" .format(self.savedir, self.cname)

        self.Hxw_npyname = "{}/{}_Hx_w_fft.npy" .format(self.savedir, self.cname)
        self.Hyw_npyname = "{}/{}_Hy_w_fft.npy" .format(self.savedir, self.cname)
        self.Hzw_npyname = "{}/{}_Hz_w_fft.npy" .format(self.savedir, self.cname)

        self.Exw_txtname = "{}/{}_Ex_w_fft.txt" .format(self.savedir, self.cname)
        self.Eyw_txtname = "{}/{}_Ey_w_fft.txt" .format(self.savedir, self.cname)
        self.Ezw_txtname = "{}/{}_Ez_w_fft.txt" .format(self.savedir, self.cname)

        self.Hxw_txtname = "{}/{}_Hx_w_fft.txt" .format(self.savedir, self.cname)
        self.Hyw_txtname = "{}/{}_Hy_w_fft.txt" .format(self.savedir, self.cname)
        self.Hzw_txtname = "{}/{}_Hz_w_fft.txt" .format(self.savedir, self.cname)

        if self.binary == True:

            np.save(self.Exw_npyname, self.Ex_w)
            np.save(self.Eyw_npyname, self.Ey_w)
            np.save(self.Ezw_npyname, self.Ez_w)

            np.save(self.Hxw_npyname, self.Hx_w)
            np.save(self.Hyw_npyname, self.Hy_w)
            np.save(self.Hzw_npyname, self.Hz_w)

        if self.txt == True:

            np.savetxt(self.Exw_txtname, self.Ex_w, newline='\n', fmt='%1.15f+%1.15fi')
            np.savetxt(self.Eyw_txtname, self.Ey_w, newline='\n', fmt='%1.15f+%1.15fi')
            np.savetxt(self.Ezw_txtname, self.Ez_w, newline='\n', fmt='%1.15f+%1.15fi')

            np.savetxt(self.Hxw_txtname, self.Hx_w, newline='\n', fmt='%1.15f+%1.15fi')
            np.savetxt(self.Hyw_txtname, self.Hy_w, newline='\n', fmt='%1.15f+%1.15fi')
            np.savetxt(self.Hzw_txtname, self.Hz_w, newline='\n', fmt='%1.15f+%1.15fi')

        if self.csv == True:

            fftfreq = np.fft.fftfreq(len(self.Ex_t), self.dt)
            nfreqs = self.normalized_freq(fftfreq, lc)

            df = pd.DataFrame()

            df['Nfreqs'] = nfreqs
            df['freqs'] = fftfreq

            df['Ex_w'] = abs(self.Ex_w)
            df['Ey_w'] = abs(self.Ey_w)
            df['Ez_w'] = abs(self.Ez_w)
            df['Hx_w'] = abs(self.Hx_w)
            df['Hy_w'] = abs(self.Hy_w)
            df['Hz_w'] = abs(self.Hz_w)

            df.to_csv("{}/{}_fft_results.csv" .format(self.savedir, self.cname))

    def use_pharminv(self, name, dt, fmin, fmax, spacing, **kwargs):

        if name == "Ex": signal = self.Ex_t
        if name == "Ey": signal = self.Ey_t
        if name == "Ez": signal = self.Ez_t
        if name == "Hx": signal = self.Hx_t
        if name == "Hy": signal = self.Hy_t
        if name == "Hz": signal = self.Hz_t

        phase = False
        wvlen = True
        printing = False
        nf = 10

        if kwargs.get('nf') != None: nf = kwargs.get('nf')
        if kwargs.get('phase') != None: phase = kwargs.get('phase')
        if kwargs.get('wvlen') != None: wvlen = kwargs.get('wvlen')
        if kwargs.get('printing') != None: printing = kwargs.get('printing')

        harm = hv.Harminv(signal=signal, fmin=fmin, fmax=fmax, dt=dt, nf=nf)

        if harm.freq[-1] > 1e3 : 
            funit = 'KHz'
            wunit = 'km'
            hfreq = harm.freq / 1e3
            wv = c / harm.freq / 1e3

        if harm.freq[-1] > 1e6 :
            funit = 'MHz'
            wunit = 'm'
            hfreq = harm.freq / 1e6
            wv = c / harm.freq / 1e0

        if harm.freq[-1] > 1e9 :
            funit = 'GHz'
            wunit = 'mm'
            hfreq = harm.freq / 1e9
            wv = c / harm.freq / 1e-3

        if harm.freq[-1] > 1e12:
            funit = 'THz'
            wunit = 'um'
            hfreq = harm.freq / 1e12
            wv = c / harm.freq / 1e-6

        if harm.freq[-1] > 1e15:
            funit = 'PHz'
            wunit = 'nm'
            hfreq = harm.freq / 1e15
            wv = c / harm.freq / 1e-9

        nfreqs = self.normalized_freq(harm.freq, spacing)

        if printing == True:

            print(name, ':')
            for i in range(harm.freq.size):

                if phase == True and wvlen == False:

                    print("NFreq: {:+7.4f}, Freq: {:+5.3e}{:>4s}, Decay: {:+5.3e}, Q: {:+5.3e}, Amp: {:+5.3e}, Phase: {:+5.3e}, Err: {:+5.3e}"\
                        .format(nfreqs[i], hfreq[i], unit, harm.decay[i], harm.Q[i], \
                        harm.amplitude[i], harm.phase[i], harm.error[i]))

                elif wvlen == True and phase == False:

                    print("NFreq: {:+7.4f}, Freq: {:+5.3e}{:>4s}, WL: {:+5.3e}{:>3s}, Q: {:+5.3e}, Amp: {:+5.3e}, Decay: {:+5.3e}, Err: {:+5.3e}"\
                        .format(nfreqs[i], hfreq[i], funit, wv[i], wunit, harm.Q[i], \
                        harm.amplitude[i], harm.decay[i], harm.error[i]))

                elif wvlen == True and phase == True:

                    print("NFreq: {:+7.4f}, Freq: {:+5.3e}{:>4s}, WL: {:+5.3e}{:>3s}, Q: {:+5.3e}, Amp: {:+5.3e}, phase: {:+5.3e}, Err: {:+5.3e}"\
                        .format(nfreqs[i], hfreq[i], funit, wv[i], wunit, harm.Q[i], \
                        harm.amplitude[i], harm.phase[i], harm.error[i]))

        return harm

    def use_harminv(self, Q, E, dt, fmin, fmax, **kwargs):

        nf = 10
        if kwargs.get('nf') != None: nf = kwargs.get('nf')

        self.Exw_txtname = "{}/{}_Ex_w_hv.txt" .format(self.savedir, self.cname)
        self.Eyw_txtname = "{}/{}_Ey_w_hv.txt" .format(self.savedir, self.cname)
        self.Ezw_txtname = "{}/{}_Ez_w_hv.txt" .format(self.savedir, self.cname)

        self.Hxw_txtname = "{}/{}_Hx_w_hv.txt" .format(self.savedir, self.cname)
        self.Hyw_txtname = "{}/{}_Hy_w_hv.txt" .format(self.savedir, self.cname)
        self.Hzw_txtname = "{}/{}_Hz_w_hv.txt" .format(self.savedir, self.cname)

        os.system("harminv -Q {} -E {} -f {} -F -t {} {}-{} < {} > {}" .format(Q, E, nf, dt, fmin, fmax, self.Ext_txtname, self.Exw_txtname))
        os.system("harminv -Q {} -E {} -f {} -F -t {} {}-{} < {} > {}" .format(Q, E, nf, dt, fmin, fmax, self.Eyt_txtname, self.Eyw_txtname))
        os.system("harminv -Q {} -E {} -f {} -F -t {} {}-{} < {} > {}" .format(Q, E, nf, dt, fmin, fmax, self.Ezt_txtname, self.Ezw_txtname))

        os.system("harminv -Q {} -E {} -f {} -F -t {} {}-{} < {} > {}" .format(Q, E, nf, dt, fmin, fmax, self.Hxt_txtname, self.Hxw_txtname))
        os.system("harminv -Q {} -E {} -f {} -F -t {} {}-{} < {} > {}" .format(Q, E, nf, dt, fmin, fmax, self.Hyt_txtname, self.Hyw_txtname))
        os.system("harminv -Q {} -E {} -f {} -F -t {} {}-{} < {} > {}" .format(Q, E, nf, dt, fmin, fmax, self.Hzt_txtname, self.Hzw_txtname))

    def plot_fft_result(self, xlim, ylim, file_name, norm_freq=True):

        fftfreq = np.fft.fftfreq(len(self.Ex_t), self.dt)
        fftfreq_fs = np.fft.fftshift(fftfreq)

        if norm_freq == True: fftfreq = fftfreq * self.lc / c

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24,16))

        axes[0,0].plot(fftfreq, abs(self.Ex_w), '-o', ms=0.5, label='Ex_w')
        axes[0,1].plot(fftfreq, abs(self.Ey_w), '-o', ms=0.5, label='Ey_w')
        axes[0,2].plot(fftfreq, abs(self.Ez_w), '-o', ms=0.5, label='Ez_w')

        axes[1,0].plot(fftfreq, abs(self.Hx_w), '-o', ms=0.5, label='Hx_w')
        axes[1,1].plot(fftfreq, abs(self.Hy_w), '-o', ms=0.5, label='Hy_w')
        axes[1,2].plot(fftfreq, abs(self.Hz_w), '-o', ms=0.5, label='Hz_w')

        '''
        # FFT frequency shifted data.
        self.Ex_w_fs = np.fft.fftshift(self.Ex_w)
        self.Ey_w_fs = np.fft.fftshift(self.Ey_w)
        self.Ez_w_fs = np.fft.fftshift(self.Ez_w)
        self.Hx_w_fs = np.fft.fftshift(self.Hx_w)
        self.Hy_w_fs = np.fft.fftshift(self.Hy_w)
        self.Hz_w_fs = np.fft.fftshift(self.Hz_w)

        axes[0,0].plot(fftfreq_fs, abs(self.Ex_w_fs), '-o', ms=0.5, label='Ex_w')
        axes[0,1].plot(fftfreq_fs, abs(self.Ey_w_fs), '-o', ms=0.5, label='Ey_w')
        axes[0,2].plot(fftfreq_fs, abs(self.Ez_w_fs), '-o', ms=0.5, label='Ez_w')

        axes[1,0].plot(fftfreq_fs, abs(self.Hx_w_fs), '-o', ms=0.5, label='Hx_w')
        axes[1,1].plot(fftfreq_fs, abs(self.Hy_w_fs), '-o', ms=0.5, label='Hy_w')
        axes[1,2].plot(fftfreq_fs, abs(self.Hz_w_fs), '-o', ms=0.5, label='Hz_w')

        axes[0,0].scatter(fftfreq, abs(self.Ex_w_fs), label='Ex_w')
        axes[0,1].scatter(fftfreq, abs(self.Ey_w_fs), label='Ey_w')
        axes[0,2].scatter(fftfreq, abs(self.Ez_w_fs), label='Ez_w')

        axes[1,0].scatter(fftfreq, abs(self.Hx_w_fs), label='Hx_w')
        axes[1,1].scatter(fftfreq, abs(self.Hy_w_fs), label='Hy_w')
        axes[1,2].scatter(fftfreq, abs(self.Hz_w_fs), label='Hz_w')
        '''

        hylim = [ylim[0], ylim[1]]

        if ylim[0] != None: hylim[0] = ylim[0] * 3e-3
        if ylim[1] != None: hylim[1] = ylim[1] * 3e-3

        axes[0,0].legend(loc='best')
        axes[0,1].legend(loc='best')
        axes[0,2].legend(loc='best')

        axes[1,0].legend(loc='best')
        axes[1,1].legend(loc='best')
        axes[1,2].legend(loc='best')

        axes[0,0].set_xlim(xlim[0],xlim[1])
        axes[0,1].set_xlim(xlim[0],xlim[1])
        axes[0,2].set_xlim(xlim[0],xlim[1])

        axes[1,0].set_xlim(xlim[0],xlim[1])
        axes[1,1].set_xlim(xlim[0],xlim[1])
        axes[1,2].set_xlim(xlim[0],xlim[1])

        axes[0,0].set_ylim(ylim[0],ylim[1])
        axes[0,1].set_ylim(ylim[0],ylim[1])
        axes[0,2].set_ylim(ylim[0],ylim[1])

        axes[1,0].set_ylim(hylim[0],hylim[1])
        axes[1,1].set_ylim(hylim[0],hylim[1])
        axes[1,2].set_ylim(hylim[0],hylim[1])

        fig.savefig(self.savedir+file_name, bbox_inches='tight')

class CsvCreator(SpectrumAnalyzer):

    def __init__(self, loaddir, names, dt, lattice_constant, where):
        """Load all .npy files and make averages .csv file.

        Parameters
        ----------
        loaddir: str
            The location of the .npy data files.

        names: list
            Name of the FieldAtPoint object.

        lattice_constant: float
            lattice constant of the simulated space.

        Returns
        -------
        None
        """

        nm = 1e-9
        um = 1e-6

        self.dt = dt
        self.lc = lattice_constant
        self.loaddir = loaddir

        self.folders = os.listdir(self.loaddir)
        useless = []

        for folder in self.folders:

            try: 
                self.tsteps = len(np.load(self.loaddir+"{}/{}_{}_t.npy" .format(folder, names[0], where)))
            except Exception as err:
                useless.append(folder)
                #print(err)
                print("Time domain data is not found in {}. Get total time step from the next folder." .format(folder))
                continue

        for ul in useless: self.folders.remove(ul)

        self.names = names
        self.wvlens = []

        for fname in self.folders:

            try:

                if fname[5] == '0': wvlen = fname[6:10]
                else: wvlen = fname[5:10]

                self.wvlens.append(int(wvlen))

            except: continue

        self.wvlens = np.array(self.wvlens)

    def _plot_fft_space2D_fields(self, name, fig, axes, j, nfreq, fs, labels, xlim, ylim, df, whos_csv):

        for i, label in enumerate(labels): 

            if i == 0: axes[i,j].set_title(name)
        
            to_plot = abs(fs[i])
            axes[i,j].plot(nfreq, to_plot, '-o', ms=0.5, label=label)
            axes[i,j].set_xlim(xlim[0], xlim[1])

            if 'H' in label: axes[i,j].set_ylim(ylim[0], ylim[1]*1e-3)
            else: axes[i,j].set_ylim(ylim[0], ylim[1])

            axes[i,j].legend(loc='upper center')
            axes[i,j].set_xlabel("Nfreq")
           
            if name in whos_csv: df[name+'_{}_w' .format(label)] = to_plot 

    def get_fft_plot_csv(self, dim, mode, flag, xlim, ylim, whos_csv):

        self.fftfreq = np.fft.fftfreq(self.tsteps, self.dt)
        self.nfreq = self.normalized_freq(self.fftfreq, self.lc)

        for wv, folder in enumerate(self.folders):

            try:

                if dim == 2: nrows, ncols = 3, len(self.names)
                if dim == 3: nrows, ncols = 6, len(self.names)

                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*8,nrows*8))

                df = pd.DataFrame()

                df['Nfreq'] = self.nfreq
                df['fftfreq'] = self.fftfreq

                for j, name in enumerate(self.names):

                    self.Ext_npyname = self.loaddir+"{}/{}_Ex_t.npy" .format(folder, name)
                    self.Eyt_npyname = self.loaddir+"{}/{}_Ey_t.npy" .format(folder, name)
                    self.Ezt_npyname = self.loaddir+"{}/{}_Ez_t.npy" .format(folder, name)

                    self.Hxt_npyname = self.loaddir+"{}/{}_Hx_t.npy" .format(folder, name)
                    self.Hyt_npyname = self.loaddir+"{}/{}_Hy_t.npy" .format(folder, name)
                    self.Hzt_npyname = self.loaddir+"{}/{}_Hz_t.npy" .format(folder, name)

                    # Get average of the len(name) collector objects.
                    if dim==2 and mode=='TM' and flag == 'Ex':

                        f1 = np.fft.fft(np.load(self.Ext_npyname))
                        f2 = np.fft.fft(np.load(self.Hyt_npyname))
                        f3 = np.fft.fft(np.load(self.Hzt_npyname))

                        fs = [f1, f2, f3]
                        labels = ['Ex', 'Hy', 'Hz']
                        self._plot_fft_space2D_fields(name, fig, axes, j, self.nfreq, fs, labels, xlim, ylim, df, whos_csv)

                    elif dim==2 and mode=='TM' and flag == 'Ez':

                        f1 = np.fft.fft(np.load(self.Ezt_npyname))
                        f2 = np.fft.fft(np.load(self.Hxt_npyname))
                        f3 = np.fft.fft(np.load(self.Hyt_npyname))

                        fs = [f1, f2, f3]
                        labels = ['Ez', 'Hx', 'Hy']
                        self._plot_fft_space2D_fields(name, fig, axes, j, self.nfreq, fs, labels, xlim, ylim, df, whos_csv)

                    elif dim==2 and mode=='TE' and flag == 'Hx':

                        f1 = np.fft.fft(np.load(self.Eyt_npyname))
                        f2 = np.fft.fft(np.load(self.Ezt_npyname))
                        f3 = np.fft.fft(np.load(self.Hxt_npyname))

                        fs = [f1, f2, f3]
                        labels = ['Ey', 'Ez', 'Hx']
                        self._plot_fft_space2D_fields(name, fig, axes, j, self.nfreq, fs, labels, xlim, ylim, df, whos_csv)

                    elif dim==2 and mode=='TE' and flag == 'Hz':

                        f1 = np.fft.fft(np.load(self.Ext_npyname))
                        f2 = np.fft.fft(np.load(self.Eyt_npyname))
                        f3 = np.fft.fft(np.load(self.Hzt_npyname))

                        fs = [f1, f2, f3]
                        labels = ['Ex', 'Ey', 'Hz']
                        self._plot_fft_space2D_fields(name, fig, axes, j, self.nfreq, fs, labels, xlim, ylim, df, whos_csv)

                    elif dim==3:

                        self.Ex_w = np.fft.fft(np.load(self.Ext_npyname))
                        self.Ey_w = np.fft.fft(np.load(self.Eyt_npyname))
                        self.Ez_w = np.fft.fft(np.load(self.Ezt_npyname))

                        self.Hx_w = np.fft.fft(np.load(self.Hxt_npyname))
                        self.Hy_w = np.fft.fft(np.load(self.Hyt_npyname))
                        self.Hz_w = np.fft.fft(np.load(self.Hzt_npyname))

                        if name in whos_csv:
                            df[name+'_Ex_w'] = self.Ex_w
                            df[name+'_Ey_w'] = self.Ey_w
                            df[name+'_Ez_w'] = self.Ez_w
                            df[name+'_Hx_w'] = self.Hx_w
                            df[name+'_Hy_w'] = self.Hy_w
                            df[name+'_Hz_w'] = self.Hz_w

                        axes[0,i].plot(self.nfreq, abs(self.Ex_w), '-o', ms=0.5, label='Ez_w')
                        axes[1,i].plot(self.nfreq, abs(self.Ey_w), '-o', ms=0.5, label='Hx_w')
                        axes[2,i].plot(self.nfreq, abs(self.Ez_w), '-o', ms=0.5, label='Ez_w')
                        axes[3,i].plot(self.nfreq, abs(self.Hx_w), '-o', ms=0.5, label='Hx_w')
                        axes[4,i].plot(self.nfreq, abs(self.Hy_w), '-o', ms=0.5, label='Hy_w')
                        axes[5,i].plot(self.nfreq, abs(self.Hz_w), '-o', ms=0.5, label='Hy_w')

                        axes[0,i].set_xlim(xlim[0], xlim[1])
                        axes[1,i].set_xlim(xlim[0], xlim[1])
                        axes[2,i].set_xlim(xlim[0], xlim[1])
                        axes[3,i].set_xlim(xlim[0], xlim[1])
                        axes[4,i].set_xlim(xlim[0], xlim[1])
                        axes[5,i].set_xlim(xlim[0], xlim[1])

                        axes[0,i].set_ylim(ylim[0], ylim[1])
                        axes[1,i].set_ylim(ylim[0], ylim[1])
                        axes[2,i].set_ylim(ylim[0], ylim[1])
                        axes[3,i].set_ylim(ylim[0], ylim[1])
                        axes[4,i].set_ylim(ylim[0], ylim[1])
                        axes[5,i].set_ylim(ylim[0], ylim[1])

                    else: raise ValueError("dim must be 1,2 or 3,  mode should be defined if dim==2 and \
                        flag should be defined since it indicates the plane.")

                fig.savefig("{}/{}_fft_results.png" .format(self.loaddir, self.wvlens[wv], bbox_inches='tight'))
                print('{} fft results are plotted.' .format(self.wvlens[wv]))
                plt.close('all')

                df.to_csv("{}/{}_fft_results.csv" .format(self.loaddir, folder[5:10]))

            except Exception as err: 
                print(err)

    def _record_pharminv(self, name, field, wvlen, harm):

        cols = ['nfreq', 'freq', 'WL', 'Q', 'Amp', 'Decay', 'phase', 'Err']
        df = pd.DataFrame(columns=cols)

        if harm.freq[-1] > 1e3 : 
            funit = 'KHz'
            wunit = 'km'
            hfreq = harm.freq / 1e3
            wv = c / harm.freq / 1e3

        elif harm.freq[-1] > 1e6 :
            funit = 'MHz'
            wunit = 'm'
            hfreq = harm.freq / 1e6
            wv = c / harm.freq / 1e0

        elif harm.freq[-1] > 1e9 :
            funit = 'GHz'
            wunit = 'mm'
            hfreq = harm.freq / 1e9
            wv = c / harm.freq / 1e-3

        elif harm.freq[-1] > 1e12:
            funit = 'THz'
            wunit = 'um'
            hfreq = harm.freq / 1e12
            wv = c / harm.freq / 1e-6

        elif harm.freq[-1] > 1e15:
            funit = 'PHz'
            wunit = 'nm'
            hfreq = harm.freq / 1e15
            wv = c / harm.freq / 1e-9

        for i in range(harm.freq.size):

            nfreq = self.normalized_freq(harm.freq[i], self.lc)
            wv = c/harm.freq[i]

            row = [nfreq, harm.freq[i], wv, harm.Q[i], harm.amplitude[i], harm.decay[i], harm.phase[i], harm.error[i]]
            df.loc[len(df)] = row 
            
        df.to_csv("{}/{:05d}_{}_{}_pharminv_results.csv" .format(self.loaddir, wvlen, field, name))

    def get_pharminv_csv(self, field, name, tsteps, dt, fmin, fmax, nf, **kwargs):

        self.dt = dt
        self.tsteps = tsteps

        for i, folder in enumerate(self.folders):

            field_npyname = self.loaddir+"{}/{}_{}_t.npy" .format(folder, name, field)
            field_t = np.load(field_npyname)

            harm_field = hv.Harminv(signal=field_t, fmin=fmin, fmax=fmax, dt=dt, nf=nf)

            self._record_pharminv(name, field, self.wvlens[i], harm_field)

            """
            elif self.dim == 2 and self.mode == 'TE':

                self.Hxt_npyname = self.loaddir+"{}/{}_Hx_t.npy" .format(folder, name)
                self.Eyt_npyname = self.loaddir+"{}/{}_Ey_t.npy" .format(folder, name)
                self.Ezt_npyname = self.loaddir+"{}/{}_Ez_t.npy" .format(folder, name)

                self.Hx_t = np.load(self.Hxt_npyname)
                self.Ey_t = np.load(self.Eyt_npyname)
                self.Ez_t = np.load(self.Ezt_npyname)

                harm_Eyt = hv.Harminv(signal=self.Ey_t, fmin=fmin, fmax=fmax, dt=dt, nf=nf)
                harm_Ezt = hv.Harminv(signal=self.Ez_t, fmin=fmin, fmax=fmax, dt=dt, nf=nf)
                harm_Hxt = hv.Harminv(signal=self.Hx_t, fmin=fmin, fmax=fmax, dt=dt, nf=nf)

                self._record_pharminv('Ey', harm_Eyt)
                self._record_pharminv('Ez', harm_Ezt)
                self._record_pharminv('Hx', harm_Hxt)

            elif self.dim == 3:

                self.Hxt_npyname = self.loaddir+"{}/{}_Hx_t.npy" .format(folder, name)
                self.Hyt_npyname = self.loaddir+"{}/{}_Hy_t.npy" .format(folder, name)
                self.Hzt_npyname = self.loaddir+"{}/{}_Hz_t.npy" .format(folder, name)
                self.Ext_npyname = self.loaddir+"{}/{}_Ex_t.npy" .format(folder, name)
                self.Eyt_npyname = self.loaddir+"{}/{}_Ey_t.npy" .format(folder, name)
                self.Ezt_npyname = self.loaddir+"{}/{}_Ez_t.npy" .format(folder, name)

                self.Ex_t = np.load(self.Ext_npyname)
                self.Ey_t = np.load(self.Eyt_npyname)
                self.Ez_t = np.load(self.Ezt_npyname)
                self.Hx_t = np.load(self.Hxt_npyname)
                self.Hy_t = np.load(self.Hyt_npyname)
                self.Hz_t = np.load(self.Hzt_npyname)

                harm_Ext = hv.Harminv(signal=self.Ex_t, fmin=fmin, fmax=fmax, dt=dt, nf=nf)
                harm_Eyt = hv.Harminv(signal=self.Ey_t, fmin=fmin, fmax=fmax, dt=dt, nf=nf)
                harm_Ezt = hv.Harminv(signal=self.Ez_t, fmin=fmin, fmax=fmax, dt=dt, nf=nf)
                harm_Hxt = hv.Harminv(signal=self.Hx_t, fmin=fmin, fmax=fmax, dt=dt, nf=nf)
                harm_Hyt = hv.Harminv(signal=self.Hy_t, fmin=fmin, fmax=fmax, dt=dt, nf=nf)
                harm_Hzt = hv.Harminv(signal=self.Hz_t, fmin=fmin, fmax=fmax, dt=dt, nf=nf)
            """

            print("{:05d} {} pharminv calculation finished." .format(self.wvlens[i], field))


if __name__ == '__main__':

    um = 1e-6
    nm = 1e-9

    Lx, Ly, Lz = 574/8*nm, 574*nm, 574*nm
    Nx, Ny, Nz = 32, 256, 256
    dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz 

    courant = 1./4
    dt = courant * min(dx,dy,dz) /c

    Q = 30
    E = 1e-4
    nf = 100
    fmin = -5e14 
    fmax = +5e14

    #loaddir = '/home/ldg/2nd_paper/SHPF.cupy.diel.CPML.MPI/graph/wvlen18000_theta0/'
    loaddir = '/home/ldg/2nd_paper/SHPF.cupy.diel.CPML.MPI/graph/wvlen01200_theta0/'
    savedir = loaddir

    test = SpectrumAnalyzer(loaddir, savedir, 'fap1')
    #test.use_harminv(Q, E, dt, fmin, fmax, nf=nf)
    test.use_pharminv('Ex', dt, fmin, fmax, Ly, nf=nf, printing=True, phase=True, wvlen=True)
    test.use_fft(dt, Ly, binary=False, txt=False, csv=True)
    #test.plot_fft_result([None, None], [None, None],"fap1_fft_1.png")
    #test.plot_fft_result([fmin, fmax], [-0.1, 2], "fap1_fft_2.png", norm_freq=False)
    test.plot_fft_result([-1, 1], [-0.1, 2], "fap1_fft_3.png", norm_freq=True)

    test2 = SpectrumAnalyzer(loaddir, savedir, 'fap2')
    #test2.use_harminv(Q, E, dt, fmin, fmax, nf=nf)
    test2.use_pharminv('Ex', dt, fmin, fmax, Ly, nf=nf, phase=True, wvlen=True)
    test2.use_fft(dt, Ly)
    #test2.plot_fft_result([None, None], [None, None],"fap2_fft_1.png")
    #test2.plot_fft_result([fmin, fmax], [-.1, 2], "fap2_fft_2.png", norm_freq=False)
    test2.plot_fft_result([-1, 1], [-.1, 2], "fap2_fft_3.png", norm_freq=True)

    test3 = SpectrumAnalyzer(loaddir, savedir, 'fap3')
    #test3.use_harminv(Q, E, dt, fmin, fmax, nf=nf)
    test3.use_pharminv('Ex', dt, fmin, fmax, Ly, nf=nf, phase=True, wvlen=True)
    test3.use_fft(dt, Ly)
    #test3.plot_fft_result([fmin, fmax], [-.1, 2],"fap3_fft_1.png", norm_freq=False)
    test3.plot_fft_result([-1, 1], [-.1, 2], "fap3_fft_3.png", norm_freq=True)
    
    test4 = SpectrumAnalyzer(loaddir, savedir, 'fap4')
    #test4.use_harminv(Q, E, dt, fmin, fmax, nf=nf)
    test4.use_pharminv('Ex', dt, fmin, fmax, Ly, nf=nf, phase=True, wvlen=True)
    test4.use_fft(dt, Ly)
    #test4.plot_fft_result([fmin, fmax], [-.1, 2],"fap4_fft_1.png", norm_freq=False)
    test4.plot_fft_result([-1, 1], [-.1, 2], "fap4_fft_3.png", norm_freq=True)

    test5 = SpectrumAnalyzer(loaddir, savedir, 'fap5')
    #test4.use_harminv(Q, E, dt, fmin, fmax, nf=nf)
    test5.use_pharminv('Ex', dt, fmin, fmax, Ly, nf=nf, phase=True, wvlen=True)
    test5.use_fft(dt, Ly)
    #test4.plot_fft_result([fmin, fmax], [-.1, 2],"fap4_fft_1.png", norm_freq=False)
    test5.plot_fft_result([-1, 1], [-.1, 2], "fap5_fft_3.png", norm_freq=True)

    """
    wvlens = np.arange(574, 601, 100)
    names = ['fap1', 'fap2', 'fap3', 'fap4']

    xlim = [-1,1]
    ylim = [0,1]

    #print(wvlens)
    test = CsvDataCollector(loaddir, wvlens, 'nm', names, dt, Ly)
    test.get_csv()
    """
