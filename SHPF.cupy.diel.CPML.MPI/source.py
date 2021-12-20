import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, mu_0, epsilon_0

class Setter:

    def __init__(self, space, src_srt, src_end, mmt):
        """Set the position, type of the source and field.

        PARAMETERS
        ----------
        space: Space object.

        src_srt: tuple
            The starting location of the source. 

            ex) 
                1. A plane wave.
                    For a plane wave propagating along x-axis: (300um, 0, 0)

                2. A dipole source.
                    (300um, 300um, 300um)

                3. A line source long y-axis.
                    (300um, 0, 300um)

        src_end: tuple
            A tuple indicating the location of a point, like (x,y,z).
            The elements designate the end position of the source.
            
            ex)

                1. A plane wave.
                    For a plane wave propagating along x-axis: (300um+dx, 0, 0)

                2. A dipole source.
                    (300um+dx, 300um+dy, 300um+dz)

                3. A line source long y-axis.
                    (300um, 300um, 300um)

        mmt: tuple.
            momentum vector (kx,ky,kz). Only non-zero when the source is monochromatic.

        RETURNS
        -------
        None
        """

        self.space = space
        self.xp = self.space.xp

        self.who_put_src = None

        # For 2D simulation.
        self.src_xsrt = round(src_srt[0] / self.space.dx)
        self.src_xend = round(src_end[0] / self.space.dx)

        self.src_ysrt = round(src_srt[1] / self.space.dy)
        self.src_yend = round(src_end[1] / self.space.dy)

        #print(self.src_xsrt, self.src_xend)
        #print(self.src_ysrt, self.src_yend)
        #if self.src_ysrt == self.src_yend: self.src_ysrt = self.src_yend - 1

        # For 3D simluation.
        if space.dimension == 3:

            self.src_zsrt = round(src_srt[2] / self.space.dz)
            self.src_zend = round(src_end[2] / self.space.dz)

            #print(self.src_zsrt, self.src_zend)
            #if self.src_zsrt == self.src_zend: self.src_zsrt = self.src_zend - 1

        #----------------------------------------------------------------------#
        #--------- All ranks should know who put src to plot src graph --------#
        #----------------------------------------------------------------------#

        self.space.MPIcomm.Barrier()

        for rank in range(self.space.MPIsize):

            my_xsrt = self.space.myNx_indice[rank][0]
            my_xend = self.space.myNx_indice[rank][1]

            # case 4. x position of the source has zero length.
            if self.src_xsrt == self.src_xend: self.src_xsrt = self.src_xend-1

            # case 1. x position of source is fixed.
            if self.src_xsrt == (self.src_xend-1):

                if self.src_xsrt >= my_xsrt and self.src_xend <= my_xend:
                    self.who_put_src = rank

                    if self.space.MPIrank == self.who_put_src:

                        self.my_src_xsrt = self.src_xsrt - my_xsrt
                        self.my_src_xend = self.src_xend - my_xsrt

                        self.src = self.xp.zeros(self.space.tsteps, dtype=self.space.field_dtype)

                        #print("rank{:>2}: src_xsrt : {}, my_src_xsrt: {}, my_src_xend: {}"\
                        #       .format(self.space.MPIrank, self.src_xsrt, self.my_src_xsrt, self.my_src_xend))

                    else:
                        pass
                        #print("rank {:>2}: I don't put source".format(self.space.MPIrank))

                else: continue

            # case 2. x position of source has range.
            elif self.src_xsrt < self.src_xend:

                assert self.space.MPIsize == 1

                self.who_put_src = 0
                self.my_src_xsrt = self.src_xsrt
                self.my_src_xend = self.src_xend

                self.src = self.xp.zeros(self.space.tsteps, dtype=self.space.field_dtype)

            # case 3. x position of source is reversed.
            elif self.src_xsrt > self.src_xend:
                raise ValueError("src_end[0] should be bigger than src_srt[0]")

            else:
                raise ValueError('x location of the source is not defined!')

        #if self.space.MPIrank == self.who_put_src:
            #print(self.my_src_xsrt, self.my_src_xend)
            #print(self.src_ysrt, self.src_yend)
            #print(self.src_zsrt, self.src_zend)

        #--------------------------------------------------------------------------#
        #--------- Apply phase difference according to the incident angle ---------#
        #--------------------------------------------------------------------------#

        self.space.mmt = mmt

        if self.space.MPIrank == self.who_put_src:

            kx = mmt[0]
            ky = mmt[1]

            # For 2D simualtion.
            self.px = self.xp.exp(+1j*kx*self.xp.arange(self.my_src_xsrt, self.my_src_xend)*self.space.dx)
            self.py = self.xp.exp(+1j*ky*self.xp.arange(self.   src_ysrt, self.   src_yend)*self.space.dy)

            xdist = self.my_src_xend - self.my_src_xsrt
            ydist = self.   src_yend - self.   src_ysrt

            if xdist == 1: self.px = self.xp.exp(1j*kx*self.xp.arange(1)*self.space.dx)
            if ydist == 1: self.py = self.xp.exp(1j*ky*self.xp.arange(1)*self.space.dy)

            # For 3D simualtion.
            if space.dimension == 3:

                kz = mmt[2]

                self.pz = self.xp.exp(+1j*kz*self.xp.arange(self.src_zsrt, self.src_zend)*self.space.dz)

                zdist = self.src_zend - self.src_zsrt

                if zdist == 1: self.pz = self.xp.exp(1j*kz*self.xp.arange(1)*self.space.dz)

    def put_src(self, where, pulse, put_type):
        """Put source at the designated postion set by set_src method.
        
        PARAMETERS
        ----------  
        where : string
            ex)
                'Ex' or 'ex'
                'Ey' or 'ey'
                'Ez' or 'ez'

        pulse : float
            float returned by source.pulse.

        put_type : string
            'soft' or 'hard'

        """
        #------------------------------------------------------------#
        #--------- Put the source into the designated field ---------#
        #------------------------------------------------------------#

        self.put_type = put_type
        self.where = where
        self.pulse = pulse

        if self.space.MPIrank == self.who_put_src:

            # For 2D simulation.
            if self.space.dimension == 2:

                x = slice(self.my_src_xsrt, self.my_src_xend)
                y = slice(self.   src_ysrt, self.   src_yend)

                self.pulse *= self.px[:,None] * self.py[None,:]

                if   self.put_type == 'soft':

                    if (self.where == 'Ex') or (self.where == 'ex'): self.space.Ex[x,y] += self.pulse
                    if (self.where == 'Ey') or (self.where == 'ey'): self.space.Ey[x,y] += self.pulse
                    if (self.where == 'Ez') or (self.where == 'ez'): self.space.Ez[x,y] += self.pulse
                    if (self.where == 'Hx') or (self.where == 'hx'): self.space.Hx[x,y] += self.pulse
                    if (self.where == 'Hy') or (self.where == 'hy'): self.space.Hy[x,y] += self.pulse
                    if (self.where == 'Hz') or (self.where == 'hz'): self.space.Hz[x,y] += self.pulse

                elif self.put_type == 'hard':
        
                    if (self.where == 'Ex') or (self.where == 'ex'): self.space.Ex[x,y] = self.pulse
                    if (self.where == 'Ey') or (self.where == 'ey'): self.space.Ey[x,y] = self.pulse
                    if (self.where == 'Ez') or (self.where == 'ez'): self.space.Ez[x,y] = self.pulse
                    if (self.where == 'Hx') or (self.where == 'hx'): self.space.Hx[x,y] = self.pulse
                    if (self.where == 'Hy') or (self.where == 'hy'): self.space.Hy[x,y] = self.pulse
                    if (self.where == 'Hz') or (self.where == 'hz'): self.space.Hz[x,y] = self.pulse

                else:
                    raise ValueError("Please insert 'soft' or 'hard'")

            # For 3D simulation.
            if self.space.dimension == 3:

                x = slice(self.my_src_xsrt, self.my_src_xend)
                y = slice(self.   src_ysrt, self.   src_yend)
                z = slice(self.   src_zsrt, self.   src_zend)

                if self.space.BBC_called == True:
                    self.pulse *= self.px[:,None,None] * self.py[None,:,None] * self.pz[None,None,:]

                if   self.put_type == 'soft':

                    if (self.where == 'Ex') or (self.where == 'ex'): self.space.Ex[x,y,z] += self.pulse
                    if (self.where == 'Ey') or (self.where == 'ey'): self.space.Ey[x,y,z] += self.pulse
                    if (self.where == 'Ez') or (self.where == 'ez'): self.space.Ez[x,y,z] += self.pulse
                    if (self.where == 'Hx') or (self.where == 'hx'): self.space.Hx[x,y,z] += self.pulse
                    if (self.where == 'Hy') or (self.where == 'hy'): self.space.Hy[x,y,z] += self.pulse
                    if (self.where == 'Hz') or (self.where == 'hz'): self.space.Hz[x,y,z] += self.pulse

                elif self.put_type == 'hard':
        
                    if (self.where == 'Ex') or (self.where == 'ex'): self.space.Ex[x,y,z] = self.pulse
                    if (self.where == 'Ey') or (self.where == 'ey'): self.space.Ey[x,y,z] = self.pulse
                    if (self.where == 'Ez') or (self.where == 'ez'): self.space.Ez[x,y,z] = self.pulse
                    if (self.where == 'Hx') or (self.where == 'hx'): self.space.Hx[x,y,z] = self.pulse
                    if (self.where == 'Hy') or (self.where == 'hy'): self.space.Hy[x,y,z] = self.pulse
                    if (self.where == 'Hz') or (self.where == 'hz'): self.space.Hz[x,y,z] = self.pulse

                else:
                    raise ValueError("Please insert 'soft' or 'hard'")


class Gaussian:
    
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

        um = 1e-6
        nm = 1e-9

        #print(f'Gaussian center wavelength: {self.wvlenc/um:.4f} um.')
        #print(f'Gaussian angular frequency spread: {self.spread:.3f}*w0')

    def pulse_c(self, step):

        pulse = np.exp((-.5) * (((step*self.dt-self.tc)*self.ws)**2)) * \
                    np.exp(-1j*self.w0*(step*self.dt-self.tc))

        return pulse

    def pulse_re(self,step):
        
        pulse_re = np.exp((-.5) * (((step*self.dt-self.tc)*self.ws)**2)) * \
                    np.cos(self.w0*(step*self.dt-self.tc))

        return pulse_re

    def pulse_im(self,step):
        
        pulse_im = np.exp((-.5) * (((step*self.dt-self.tc)*self.ws)**2)) * \
                    -np.sin(self.w0*(step*self.dt-self.tc))

        return pulse_im

    def plot_pulse(self, tsteps, freqs, savedir):
        
        time_domain = np.arange(tsteps, dtype=self.dtype)
        t = time_domain * self.dt

        self.freqs = freqs
        self.wvlens = c / self.freqs

        pulse_re = np.exp((-.5) * (((t-self.tc)*self.ws)**2)) * np.cos(self.w0*(t-self.tc))
        pulse_im = np.exp((-.5) * (((t-self.tc)*self.ws)**2)) * np.sin(self.w0*(t-self.tc))

        pulse_re_ft = (self.dt * pulse_re[None,:] * np.exp(1j*2*np.pi*self.freqs[:,None]*t[None,:])).sum(1) / np.sqrt(2*np.pi)
        pulse_im_ft = (self.dt * pulse_im[None,:] * np.exp(1j*2*np.pi*self.freqs[:,None]*t[None,:])).sum(1) / np.sqrt(2*np.pi)

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

        um = 1e-6
        nm = 1e-9
        text = f'wvlenc: {self.wvlenc/um:.4f}um\nspread: {self.spread:.2f}'
        ax1.text(0.8, 0.2, text, ha='center', va='center', transform=ax1.transAxes, bbox=dict(facecolor='w', alpha=0.7))

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

        if os.path.exists(savedir) != True: os.makedirs(savedir)
        fig.savefig(savedir+"src_theoretical.png")


class Sine:

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

        #pulse = np.exp(1j*self.omega * tstep * self.dt)
        pulse = np.sin(self.omega * tstep * self.dt)

        return pulse


class Cosine:

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


class Harmonic:

    def __init__(self, dt):

        self.dt = dt

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

    def apply(self, tstep):

        pulse = np.exp(-1j*self.omega*tstep*self.dt)

        return pulse


class Smoothing:

    def __init__(self, dt, threshold):
        
        self.dt = dt
        self.threshold = threshold

    def apply(self, tstep):

        smoother = 0

        if tstep < self.threshold: smoother = tstep / self.threshold
        else: smoother = 1.

        return smoother


class SmoothInOut:
 
    def __init__(self, dt, inc, dec):
        
        self.dt = dt
        self.inc = inc # increase until tstep reaches inc.
        self.dec = dec # decrease after tstep exceeds dec.

    def apply(self, tstep):

        smoother = 0

        if tstep < self.inc: smoother = tstep / self.inc
        elif tstep >= self.inc and tstep <= self.dec: smoother = 1
        elif tstep > self.dec and tstep < (self.inc+self.dec):
            smoother = (self.inc - tstep) / (self.inc - self.dec)
        else: smoother = 0

        return smoother


class Delta:

    def __init__(self, pick):

        self.pick = pick

    def apply(self, tstep):

        if tstep == self.pick: return 1.
        else: return 0.
