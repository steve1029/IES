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
            print("NFreq: %8.5f, Freq: %8.5e, Decay:%5.2e, Q:%4.2f, Amp:%4.2f, Phase:%4.2f, Err:%5.3f" % (nfreqs[i], harm.freq[i], \
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

        fig.savefig("./graph/field_at_point.png", bbox_inches='tight')

if __name__ == '__main__':

    um = 1e-6
    Nx, Ny, Nz = 256, 8, 356
    dx, dy, dz = 5*um, 5*um, 5*um
    Lx, Ly, Lz = Nx*dx, Ny*dy, Nz*dz

    courant = 1./4
    dt = courant * min(dx,dy,dz) /c

    loc = './graph/fap/'
    test = PlotHarminv(loc, dt, '00')
    test.try_harminv('Ex', -12e11, 12e11, Lx)
    test.plot_fft_all(loc, [0.8e14, None])
