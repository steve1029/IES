import sys
import numpy as np
import harminv as hv
import matplotlib.pyplot as plt
from scipy.constants import c

class PlotHarminv:

    def __init__(self, data_loc, dt, rank):

        assert type(rank) == str

        self.dt = dt
        self.savedir = data_loc

        self.Ex_t = np.load(self.savedir+"fap1_Ex_t_rank{}.npy" .format(rank))
        self.Ey_t = np.load(self.savedir+"fap1_Ey_t_rank{}.npy" .format(rank))
        self.Ez_t = np.load(self.savedir+"fap1_Ez_t_rank{}.npy" .format(rank))

        self.Hx_t = np.load(self.savedir+"fap1_Hx_t_rank{}.npy" .format(rank))
        self.Hy_t = np.load(self.savedir+"fap1_Hy_t_rank{}.npy" .format(rank))
        self.Hz_t = np.load(self.savedir+"fap1_Hz_t_rank{}.npy" .format(rank))

        self.Ex_w = np.load(self.savedir+"fap1_Ex_w_rank00.npy")
        self.Ey_w = np.load(self.savedir+"fap1_Ey_w_rank00.npy")
        self.Ez_w = np.load(self.savedir+"fap1_Ez_w_rank00.npy")

        self.Hx_w = np.load(self.savedir+"fap1_Hx_w_rank00.npy")
        self.Hy_w = np.load(self.savedir+"fap1_Hy_w_rank00.npy")
        self.Hz_w = np.load(self.savedir+"fap1_Hz_w_rank00.npy")

        # FFT frequency shifted data.
        self.Ex_w_fs = np.fft.fftshift(self.Ex_w)
        self.Ey_w_fs = np.fft.fftshift(self.Ey_w)
        self.Ez_w_fs = np.fft.fftshift(self.Ez_w)
        self.Hx_w_fs = np.fft.fftshift(self.Hx_w)
        self.Hy_w_fs = np.fft.fftshift(self.Hy_w)
        self.Hz_w_fs = np.fft.fftshift(self.Hz_w)

    def norm_freq(self, freqs, spacing):

        return freqs * spacing / c

    def try_harminv(self, name, fmin, fmax, spacing):

        if name == "Ex": signal = self.Ex_t
        if name == "Ey": signal = self.Ey_t
        if name == "Ez": signal = self.Ez_t
        if name == "Hx": signal = self.Hx_t
        if name == "Hy": signal = self.Hy_t
        if name == "Hz": signal = self.Hz_t

        harm = hv.Harminv(signal=signal, fmin=fmin, fmax=fmax, dt=self.dt)

        nfreqs = self.norm_freq(harm.freq, spacing)

        print(name, ':')
        for i in range(harm.freq.size):
            print("NFreq: %8.5f, Freq: %8.5e, Decay:%6.2e, Q:%6.2e, Amp:%6.2e, Phase:%6.2e, Err:%6.2e" % (nfreqs[i], harm.freq[i], \
                harm.decay[i], harm.Q[i], harm.amplitude[i], harm.phase[i], harm.error[i]))

    def plot_fft_all(self, loc, xlim):

        fftfreq = np.fft.fftfreq(len(self.Ex_t), self.dt)

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24,16))

        axes[0,0].plot(fftfreq, abs(self.Ex_w_fs), '-o', ms=0.5, label='Ex_w')
        axes[0,1].plot(fftfreq, abs(self.Ey_w_fs), '-o', ms=0.5, label='Ey_w')
        axes[0,2].plot(fftfreq, abs(self.Ez_w_fs), '-o', ms=0.5, label='Ez_w')

        axes[1,0].plot(fftfreq, abs(self.Hx_w_fs), '-o', ms=0.5, label='Hx_w')
        axes[1,1].plot(fftfreq, abs(self.Hy_w_fs), '-o', ms=0.5, label='Hy_w')
        axes[1,2].plot(fftfreq, abs(self.Hz_w_fs), '-o', ms=0.5, label='Hz_w')

        '''
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

        fig.savefig(self.savedir+"field_at_point.png", bbox_inches='tight')

if __name__ == '__main__':

    um = 1e-6
    nm = 1e-9

    Lx, Ly, Lz = 574/32*nm, 574*nm, 574*nm
    Nx, Ny, Nz = 8, 256, 256
    dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz 

    courant = 1./4
    dt = courant * min(dx,dy,dz) /c

    fmax = 1./2/dt

    loc = '/home/ldg/2nd_paper/SHPF.cupy.diel.CPML.MPI/graph/'
    test = PlotHarminv(loc, dt, '00')
    test.try_harminv('Ex', 0, fmax, Lx)
    test.plot_fft_all(loc, [None, None])
