import os, sys
import numpy as np
import harminv as hv
import matplotlib.pyplot as plt
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

    def norm_freq(self, freqs, spacing):

        return freqs * spacing / c

    def use_fft(self, dt, **kwargs):

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

        if binary == True:

            np.save(self.Exw_npyname, self.Ex_w)
            np.save(self.Eyw_npyname, self.Ey_w)
            np.save(self.Ezw_npyname, self.Ez_w)

            np.save(self.Hxw_npyname, self.Hx_w)
            np.save(self.Hyw_npyname, self.Hy_w)
            np.save(self.Hzw_npyname, self.Hz_w)

        if txt == True:

            np.savetxt(self.Exw_txtname, self.Ex_w, newline='\n', fmt='%1.15f+%1.15fi')
            np.savetxt(self.Eyw_txtname, self.Ey_w, newline='\n', fmt='%1.15f+%1.15fi')
            np.savetxt(self.Ezw_txtname, self.Ez_w, newline='\n', fmt='%1.15f+%1.15fi')

            np.savetxt(self.Hxw_txtname, self.Hx_w, newline='\n', fmt='%1.15f+%1.15fi')
            np.savetxt(self.Hyw_txtname, self.Hy_w, newline='\n', fmt='%1.15f+%1.15fi')
            np.savetxt(self.Hzw_txtname, self.Hz_w, newline='\n', fmt='%1.15f+%1.15fi')

    def use_pharminv(self, name, dt, fmin, fmax, spacing, **kwargs):

        if name == "Ex": signal = self.Ex_t
        if name == "Ey": signal = self.Ey_t
        if name == "Ez": signal = self.Ez_t
        if name == "Hx": signal = self.Hx_t
        if name == "Hy": signal = self.Hy_t
        if name == "Hz": signal = self.Hz_t

        nf = 10
        if kwargs.get('nf') != None: self.nf = kwargs.get('nf')

        harm = hv.Harminv(signal=signal, fmin=fmin, fmax=fmax, dt=dt, nf=self.nf)

        if harm.freq[-1] > 1e3 : 
            unit = 'KHz'
            hfreq = harm.freq / 1e3
        if harm.freq[-1] > 1e6 :
            unit = 'MHz'
            hfreq = harm.freq / 1e6
        if harm.freq[-1] > 1e9 :
            unit = 'GHz'
            hfreq = harm.freq / 1e9
        if harm.freq[-1] > 1e12:
            unit = 'THz'
            hfreq = harm.freq / 1e12
        if harm.freq[-1] > 1e15:
            unit = 'PHz'
            hfreq = harm.freq / 1e15

        nfreqs = self.norm_freq(harm.freq, spacing)

        print(name, ':')
        for i in range(harm.freq.size):
            print("NFreq: {:7.3f}, Freq: {:+5.3e}{:>4s}, Decay: {:+5.3e}, Q: {:+5.3e}, Amp: {:+5.3e}, Phase: {:+5.3e}, Err: {:+5.3e}"\
                .format(nfreqs[i], hfreq[i], unit, harm.decay[i], harm.Q[i], \
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

    def plot_fft_result(self, xlim, ylim, file_name):

        # FFT frequency shifted data.
        self.Ex_w_fs = np.fft.fftshift(self.Ex_w)
        self.Ey_w_fs = np.fft.fftshift(self.Ey_w)
        self.Ez_w_fs = np.fft.fftshift(self.Ez_w)
        self.Hx_w_fs = np.fft.fftshift(self.Hx_w)
        self.Hy_w_fs = np.fft.fftshift(self.Hy_w)
        self.Hz_w_fs = np.fft.fftshift(self.Hz_w)

        fftfreq = np.fft.fftfreq(len(self.Ex_t), self.dt)
        fftfreq_fs = np.fft.fftshift(fftfreq)

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

        axes[1,0].set_ylim(ylim[0],ylim[1])
        axes[1,1].set_ylim(ylim[0],ylim[1])
        axes[1,2].set_ylim(ylim[0],ylim[1])

        fig.savefig(self.savedir+file_name, bbox_inches='tight')

if __name__ == '__main__':

    um = 1e-6
    nm = 1e-9

    Lx, Ly, Lz = 574/8*nm, 574*nm, 574*nm
    Nx, Ny, Nz = 32, 256, 256
    dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz 

    courant = 1./4
    dt = courant * min(dx,dy,dz) /c

    Q = 30
    E = 1e-3
    nf = 100
    fmin = 0 
    fmax = 5.3e14

    loaddir = '/home/ldg/2nd_paper/SHPF.cupy.diel.CPML.MPI/graph/'
    savedir = loaddir

    test = SpectrumAnalyzer(loaddir, savedir, 'fap1')
    test.use_harminv(Q, E, dt, fmin, fmax, nf=nf)
    test.use_pharminv('Ex', dt, fmin, fmax, Ly, nf=nf)
    test.use_fft(dt)
    test.plot_fft_result([None, None], [None, None],"fap1_fft_1.png")
    test.plot_fft_result([-523e12, 0], [0, 1], "fap1_fft_2.png")

    test2 = SpectrumAnalyzer(loaddir, savedir, 'fap2')
    test2.use_harminv(Q, E, dt, fmin, fmax, nf=nf)
    test2.use_pharminv('Ex', dt, fmin, fmax, Ly, nf=nf)
    test2.use_fft(dt)
    test2.plot_fft_result([2.5e17, 2.7e17], [None, None],"fap2_fft_1.png")
    test2.plot_fft_result([0.5e15, 3e15], [-1, None], "fap2_fft_2.png")

    test3 = SpectrumAnalyzer(loaddir, savedir, 'fap3')
    test3.use_harminv(Q, E, dt, fmin, fmax, nf=nf)
    test3.use_pharminv('Ex', dt, fmin, fmax, Ly, nf=nf)
    test3.use_fft(dt)
    test3.plot_fft_result([2.64e17, 2.7e17], [None, None],"fap3_fft_1.png")
    test3.plot_fft_result([0.5e15, 3e15], [-1, None], "fap3_fft_2.png")
    
    test4 = SpectrumAnalyzer(loaddir, savedir, 'fap4')
    test4.use_harminv(Q, E, dt, fmin, fmax, nf=nf)
    test4.use_pharminv('Ex', dt, fmin, fmax, Ly, nf=nf)
    test4.use_fft(dt)
    test4.plot_fft_result([2.64e17, 2.7e17], [None, None],"fap4_fft_1.png")
    test4.plot_fft_result([0.5e15, 3e15], [-1, None], "fap4_fft_2.png")
