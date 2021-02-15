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
        nf = 10

        if kwargs.get('nf') != None: nf = kwargs.get('nf')
        if kwargs.get('phase') != None: phase = kwargs.get('phase')
        if kwargs.get('wvlen') != None: wvlen = kwargs.get('wvlen')

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

        print(name, ':')
        for i in range(harm.freq.size):

            if phase == True and wvlen == False:

                print("NFreq: {:+6.3f}, Freq: {:+5.3e}{:>4s}, Decay: {:+5.3e}, Q: {:+5.3e}, Amp: {:+5.3e}, Phase: {:+5.3e}, Err: {:+5.3e}"\
                    .format(nfreqs[i], hfreq[i], unit, harm.decay[i], harm.Q[i], \
                    harm.amplitude[i], harm.phase[i], harm.error[i]))

            elif wvlen == True and phase == False:

                print("NFreq: {:+6.3f}, Freq: {:+5.3e}{:>4s}, WL: {:+5.3e}{:>3s}, Q: {:+5.3e}, Amp: {:+5.3e}, Decay: {:+5.3e}, Err: {:+5.3e}"\
                    .format(nfreqs[i], hfreq[i], funit, wv[i], wunit, harm.Q[i], \
                    harm.amplitude[i], harm.decay[i], harm.error[i]))

            elif wvlen == True and phase == True:

                print("NFreq: {:+6.3f}, Freq: {:+5.3e}{:>4s}, WL: {:+5.3e}{:>3s}, Q: {:+5.3e}, Amp: {:+5.3e}, phase: {:+5.3e}, Err: {:+5.3e}"\
                    .format(nfreqs[i], hfreq[i], funit, wv[i], wunit, harm.Q[i], \
                    harm.amplitude[i], harm.phase[i], harm.error[i]))

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

        # FFT frequency shifted data.
        self.Ex_w_fs = np.fft.fftshift(self.Ex_w)
        self.Ey_w_fs = np.fft.fftshift(self.Ey_w)
        self.Ez_w_fs = np.fft.fftshift(self.Ez_w)
        self.Hx_w_fs = np.fft.fftshift(self.Hx_w)
        self.Hy_w_fs = np.fft.fftshift(self.Hy_w)
        self.Hz_w_fs = np.fft.fftshift(self.Hz_w)

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

class CsvDataCollector(SpectrumAnalyzer):

    def __init__(self, loaddir, unit, names, dt, lattice_constant):
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

        if unit == 'nm': self.unit = nm
        if unit == 'um': self.unit = um

        self.dt = dt
        self.lc = lattice_constant
        self.loaddir = loaddir

        self.folders = os.listdir(self.loaddir)
        useless = []

        for folder in self.folders:

            try: 
                self.tsteps = len(np.load(self.loaddir+"{}/{}_Ex_t.npy" .format(folder, names[0])))
            except Exception as err:
                useless.append(folder)
                #print(err)
                print("{} is not found. Get total time step from the next folder." .format(folder))
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

        self.wvlens = np.sort(np.array(self.wvlens))

    def get_csv(self):

        for i, folder in enumerate(self.folders):

            try:

                self.Ex_w = np.zeros(self.tsteps, dtype=np.complex128)
                self.Ey_w = np.zeros(self.tsteps, dtype=np.complex128)
                self.Ez_w = np.zeros(self.tsteps, dtype=np.complex128)

                self.Hx_w = np.zeros(self.tsteps, dtype=np.complex128)
                self.Hy_w = np.zeros(self.tsteps, dtype=np.complex128)
                self.Hz_w = np.zeros(self.tsteps, dtype=np.complex128)

                for name in self.names:

                    self.Ext_npyname = self.loaddir+"{}/{}_Ex_t.npy" .format(folder, name)
                    self.Eyt_npyname = self.loaddir+"{}/{}_Ey_t.npy" .format(folder, name)
                    self.Ezt_npyname = self.loaddir+"{}/{}_Ez_t.npy" .format(folder, name)

                    self.Hxt_npyname = self.loaddir+"{}/{}_Hx_t.npy" .format(folder, name)
                    self.Hyt_npyname = self.loaddir+"{}/{}_Hy_t.npy" .format(folder, name)
                    self.Hzt_npyname = self.loaddir+"{}/{}_Hz_t.npy" .format(folder, name)

                    self.Ex_w += np.fft.fft(np.load(self.Ext_npyname))/len(name)
                    self.Ey_w += np.fft.fft(np.load(self.Eyt_npyname))/len(name)
                    self.Ez_w += np.fft.fft(np.load(self.Ezt_npyname))/len(name)

                    self.Hx_w += np.fft.fft(np.load(self.Hxt_npyname))/len(name)
                    self.Hy_w += np.fft.fft(np.load(self.Hyt_npyname))/len(name)
                    self.Hz_w += np.fft.fft(np.load(self.Hzt_npyname))/len(name)

                self.fftfreq = np.fft.fftfreq(self.tsteps, self.dt)
                self.nfreqs = self.normalized_freq(self.fftfreq, self.lc)

                df = pd.DataFrame()

                df['Nfreqs'] = self.nfreqs
                df['freqs'] = self.fftfreq

                df['Ex_w'] = abs(self.Ex_w)
                df['Ey_w'] = abs(self.Ey_w)
                df['Ez_w'] = abs(self.Ez_w)
                df['Hx_w'] = abs(self.Hx_w)
                df['Hy_w'] = abs(self.Hy_w)
                df['Hz_w'] = abs(self.Hz_w)

                df.to_csv("{}/{:05d}_avg_fft_results.csv" .format(self.loaddir, self.wvlens[i]))
                self.plot_avg_fft([-1, 1], [-0.1, 2], "{:05d}_avg_fft.png" .format(self.wvlens[i]))
                print("{:05d} csv and graph are created." .format(self.wvlens[i]))

            except Exception as err: 
                print(err)
                continue

    def plot_avg_fft(self, xlim, ylim, file_name):

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24,16))

        axes[0,0].plot(self.nfreqs, abs(self.Ex_w), '-o', ms=0.5, label='Ex_w')
        axes[0,1].plot(self.nfreqs, abs(self.Ey_w), '-o', ms=0.5, label='Ey_w')
        axes[0,2].plot(self.nfreqs, abs(self.Ez_w), '-o', ms=0.5, label='Ez_w')

        axes[1,0].plot(self.nfreqs, abs(self.Hx_w), '-o', ms=0.5, label='Hx_w')
        axes[1,1].plot(self.nfreqs, abs(self.Hy_w), '-o', ms=0.5, label='Hy_w')
        axes[1,2].plot(self.nfreqs, abs(self.Hz_w), '-o', ms=0.5, label='Hz_w')

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

        fig.savefig(self.loaddir+file_name, bbox_inches='tight')
        plt.close('all')

    def plot_peak(self, lower_limit):

        cols = ['Nmmt', 'Nfreq', 'Ex_w']
        df = pd.DataFrame(columns=cols)

        for wvlen in self.wvlens:

            try: 
                data = pd.read_csv(self.loaddir+"{}_avg_fft_results.csv" .format(wvlen))

            except FileNotFoundError:
                print("{:04d} are not found. Continue to next one.")
                continue

            band = data.loc[(data['Nfreqs'] <= 1) & (data['Nfreqs'] >= -1) & (data['Ex_w'] >= lower_limit)]
            #print(band)
            #print(band.shape)
            #print(type(band))

            #Nmmt = np.array([self.lc/wvlen/self.unit] * len(band))[np.newaxis].T
            #Nmmt = np.array([self.lc/wvlen/self.unit] * len(band))

            #for i, nmmt in enumerate(Nmmt):
            #    new = {'Nmmt': Nmmt[i], 'Nfreq':band['Nfreqs'][i], 'Ex_w':band['Ex_w'][i]}
            df.append(band, ignore_index=True) 

        #fig = df.plot.scatter(x='Nmmt', y='Nfreq')
        #fig.savefig("{}/band_structure.png". format(self.loaddir))
        df.to_csv("{}/peaks.csv" .format(self.loaddir))


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

    loaddir = '/home/ldg/2nd_paper/SHPF.cupy.diel.CPML.MPI/graph/wvlen1148_phi0_theta90/'
    savedir = loaddir

    test = SpectrumAnalyzer(loaddir, savedir, 'fap1')
    #test.use_harminv(Q, E, dt, fmin, fmax, nf=nf)
    test.use_pharminv('Ex', dt, fmin, fmax, Ly, nf=nf, phase=True, wvlen=True)
    test.use_fft(dt, Ly, binary=False, txt=False, csv=True)
    test.plot_fft_result([None, None], [None, None],"fap1_fft_1.png")
    test.plot_fft_result([fmin, fmax], [-0.1, 2], "fap1_fft_2.png", norm_freq=False)
    test.plot_fft_result([-1, 1], [-0.1, 2], "fap1_fft_3.png", norm_freq=True)

    test2 = SpectrumAnalyzer(loaddir, savedir, 'fap2')
    #test2.use_harminv(Q, E, dt, fmin, fmax, nf=nf)
    test2.use_pharminv('Ex', dt, fmin, fmax, Ly, nf=nf, phase=True, wvlen=True)
    test2.use_fft(dt, Ly)
    test2.plot_fft_result([None, None], [None, None],"fap2_fft_1.png")
    test2.plot_fft_result([fmin, fmax], [-.1, 2], "fap2_fft_2.png", norm_freq=False)
    test2.plot_fft_result([-1, 1], [-.1, 2], "fap2_fft_3.png", norm_freq=True)

    test3 = SpectrumAnalyzer(loaddir, savedir, 'fap3')
    #test3.use_harminv(Q, E, dt, fmin, fmax, nf=nf)
    test3.use_pharminv('Ex', dt, fmin, fmax, Ly, nf=nf, phase=True, wvlen=True)
    test3.use_fft(dt, Ly)
    test3.plot_fft_result([fmin, fmax], [-.1, 2],"fap3_fft_1.png", norm_freq=False)
    test3.plot_fft_result([-1, 1], [-.1, 2], "fap3_fft_2.png", norm_freq=True)
    
    test4 = SpectrumAnalyzer(loaddir, savedir, 'fap4')
    #test4.use_harminv(Q, E, dt, fmin, fmax, nf=nf)
    test4.use_pharminv('Ex', dt, fmin, fmax, Ly, nf=nf, phase=True, wvlen=True)
    test4.use_fft(dt, Ly)
    test4.plot_fft_result([fmin, fmax], [-.1, 2],"fap4_fft_1.png", norm_freq=False)
    test4.plot_fft_result([-1, 1], [-.1, 2], "fap4_fft_2.png", norm_freq=True)

    """
    wvlens = np.arange(574, 601, 100)
    names = ['fap1', 'fap2', 'fap3', 'fap4']

    xlim = [-1,1]
    ylim = [0,1]

    #print(wvlens)
    test = CsvDataCollector(loaddir, wvlens, 'nm', names, dt, Ly)
    test.get_csv()
    #test.plot_peak(0.5)
    """
