import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, mu_0, epsilon_0

class Gaussian(object):
    
    def __init__(self, dt, center_wv, spread, pick_pos, dtype):
        
        self.dt    = dt
        self.dtype = dtype
        self.wvlenc = center_wv
        self.spread = spread
        self.pick_pos = pick_pos
        self.freqc = c / self.wvlenc

        self.w0 = 2 * np.pi * self.freqc
        self.ws = self.spread * self.w0
        self.ts = 1./self.ws
        self.tc = self.pick_pos * self.dt   

    def pulse_re(self,step,pick_pos):
        
        pulse_re = np.exp((-.5) * (((step*self.dt-self.tc)*self.ws)**2)) * np.cos(self.w0*(step*self.dt-self.tc))

        return pulse_re

    def pulse_im(self,step,pick_pos):
        
        pulse_im = np.exp((-.5) * (((step*self.dt-self.tc)*self.ws)**2)) * np.sin(self.w0*(step*self.dt-self.tc))

        return pulse_im

    def plot_pulse(self, tsteps, freqs, savedir):
        
        nax = np.newaxis

        time_domain = np.arange(tsteps, dtype=self.dtype)
        t = time_domain * self.dt

        self.freqs = freqs
        self.wvlens = c / self.freqs[::-1]

        pulse_re = np.exp((-.5) * (((t-self.tc)*self.ws)**2)) * np.cos(self.w0*(t-self.tc))
        pulse_im = np.exp((-.5) * (((t-self.tc)*self.ws)**2)) * np.sin(self.w0*(t-self.tc))

        pulse_re_ft = (self.dt * pulse_re[nax,:]* np.exp(1j*2*np.pi*self.freqs[:,nax]*t[nax,:])).sum(1) / np.sqrt(2*np.pi)
        pulse_im_ft = (self.dt * pulse_im[nax,:]* np.exp(1j*2*np.pi*self.freqs[:,nax]*t[nax,:])).sum(1) / np.sqrt(2*np.pi)

        pulse_re_ft_amp = abs(pulse_re_ft)**2
        pulse_im_ft_amp = abs(pulse_im_ft)**2

        fig = plt.figure(figsize=(21,7))

        ax1 = fig.add_subplot(1,3,1)
        ax2 = fig.add_subplot(1,3,2)
        ax3 = fig.add_subplot(1,3,3)

        ax1.plot(time_domain, pulse_re, color='b', label='real')
        ax1.plot(time_domain, pulse_im, color='r', label='imag', linewidth='1.5', alpha=0.5)
        ax2.plot(self.freqs/10**12, pulse_re_ft_amp, color='b', label='real')
        ax2.plot(self.freqs/10**12, pulse_im_ft_amp, color='r', label='imag', linewidth='1.5', alpha=0.5)

        ax3.plot(self.wvlens/1e-6, pulse_re_ft_amp, color='b', label='real')
        ax3.plot(self.wvlens/1e-6, pulse_im_ft_amp, color='r', label='imag', linewidth='1.5', alpha=0.5)

        ax1.set_xlabel('time step')
        ax1.set_ylabel('Amp')
        ax1.legend(loc='best')
        ax1.grid(True)
        #ax1.set_xlim(4000,6000)

        ax2.set_xlabel('freq(THz)')
        ax2.set_ylabel('Amp')
        ax2.legend(loc='best')
        ax2.grid(True)
        ax2.set_ylim(0,None)

        ax3.set_xlabel('wavelength(um)')
        ax3.set_ylabel('Amp')
        ax3.legend(loc='best')
        ax3.grid(True)
        ax3.set_ylim(0,None)

        fig.savefig(savedir+"/graph/src_input.png")


class Sine(object):

    def __init__(self, dt, dtype):

        self.dt = dt
        self.dtype = dtype

    def set_freq(self, freq):
        
        self.freq = freq
        self.wvlen = c / self.freq
        self.omega = 2 * np.pi * self.freq
        self.wvector = 2 * np.pi / self.wvlen

    def set_wvlen(self, wvlen):

        self.wvlen = wvlen
        self.freq = c / self.wvlen
        self.omega = 2 * np.pi * self.freq
        self.wvector = 2 * np.pi / self.wvlen

    def signal(self, tstep):

        pulse_re = np.sin(self.omega * tstep * self.dt)

        return pulse_re


class Cosine(object):

    def __init__(self, dt, dtype):

        self.dt = dt
        self.dtype = dtype

    def set_freq(self, freq):
        
        self.freq = freq
        self.wvlen = c / self.freq
        self.omega = 2 * np.pi * self.freq
        self.wvector = 2 * np.pi / self.wvlen

    def set_wvlen(self, wvlen):

        self.wvlen = wvlen
        self.freq = c / self.wvlen
        self.omega = 2 * np.pi * self.freq
        self.wvector = 2 * np.pi / self.wvlen

    def signal(self, tstep):

        pulse_re = np.cos(self.omega * tstep * self.dt)

        return pulse_re
