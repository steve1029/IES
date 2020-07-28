import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import c, epsilon_0, mu_0

class SRTpainter():

	def __init__(self, where_re, where_im, dx, dy, dz, courant, wv_srt, wv_end, interval, spread, cutter1, wvlen_unit, freq_unit):

		self.src_re = np.load('./graph/src_re.npy')
		self.src_im = np.load('./graph/src_im.npy')

		self.trs_re = np.load("./graph/trs_re.npy")
		self.trs_im = np.load("./graph/trs_im.npy")

		self.ref_re = np.load("./graph/ref_re.npy")
		self.ref_im = np.load("./graph/ref_im.npy")

		self.ref_re[0:cutter1] = 0
		self.ref_im[0:cutter1] = 0

		self.src = self.src_re + 1j * self.src_im
		self.ref = self.ref_re + 1j * self.ref_im
		self.trs = self.trs_re + 1j * self.trs_im

		self.nm = 1e-9
		self.dx, self.dy, self.dz = dx,dy,dz
		self.courant = courant
		self.dt = self.courant * min(self.dx, self.dy, self.dz)/c

		self.wv_srt = wv_srt
		self.wv_end = wv_end
		self.interval = interval
		self.spread = spread
		self.wvlen = np.arange(self.wv_srt, self.wv_end, self.interval, dtype=np.float64)
		self.freq = c / self.wvlen

		if   wvlen_unit == 'um': self.wvlen_graph = self.wvlen.real / 1e-6
		elif wvlen_unit == 'nm': self.wvlen_graph = self.wvlen.real / 1e-9
		else: raise ValueError("Choose um or nm")
			
		if   freq_unit == 'THz': self.freq_graph = self.freq.real / 1e12 
		elif freq_unit == 'GHz': self.freq_graph = self.freq.real / 1e9
		else: raise ValueError("Choose GHz or THz")

		self.wvlen_unit = wvlen_unit
		self.freq_unit = freq_unit

		self.figsize = (21,10)
		self.loc     = 'best'

		self.where_re = 'Ey'
		self.where_im = 'Ey'

		self.TS = len(self.src_re)
		self.TR = len(self.trs_re)
		self.TT = len(self.ref_re)
		self.tstepsS = np.arange(self.TS, dtype=int)
		self.tstepsR = np.arange(self.TR, dtype=int)
		self.tstepsT = np.arange(self.TT, dtype=int)
		self.tS = self.tstepsS * self.dt
		self.tR = self.tstepsR * self.dt
		self.tT = self.tstepsT * self.dt

		nax = np.newaxis
		self.src_dft = (self.dt*self.src[nax,:] * np.exp(1.j*2.*np.pi*self.freq[:,nax]*self.tS[nax,:])).sum(1) / np.sqrt(2.*np.pi)
		self.trs_dft = (self.dt*self.trs[nax,:] * np.exp(1.j*2.*np.pi*self.freq[:,nax]*self.tR[nax,:])).sum(1) / np.sqrt(2.*np.pi)
		self.ref_dft = (self.dt*self.ref[nax,:] * np.exp(1.j*2.*np.pi*self.freq[:,nax]*self.tT[nax,:])).sum(1) / np.sqrt(2.*np.pi)

		self.Trs = (abs(self.trs_dft)**2) / (abs(self.src_dft)**2)
		self.Ref = (abs(self.ref_dft)**2) / (abs(self.src_dft)**2)
		#self.Ref = (abs(self.ref_dft)**2 - abs(self.src_dft)**2) / (abs(self.src_dft)**2)
		self.Tot = self.Trs + self.Ref

		np.save('./graph/Trs.npy', self.Trs)
		np.save('./graph/Ref.npy', self.Ref)
		np.save('./graph/wvlen.npy', self.wvlen.real)
		np.save('./graph/freq.npy', self.freq.real)

		np.savetxt('./graph/Trs.txt', self.Trs)
		np.savetxt('./graph/Ref.txt', self.Ref)
		np.savetxt('./graph/wvlen.txt', self.wvlen.real)
		np.savetxt('./graph/freq.txt', self.freq.real)

	def plot_src(self):

		src_fig = plt.figure(figsize=self.figsize)

		ax1 = src_fig.add_subplot(2,3,1)
		ax2 = src_fig.add_subplot(2,3,2)
		ax3 = src_fig.add_subplot(2,3,3)
		ax4 = src_fig.add_subplot(2,3,4)
		ax5 = src_fig.add_subplot(2,3,5)
		ax6 = src_fig.add_subplot(2,3,6)

		label11 = self.where_re + r'$(t)$, real'
		label12 = self.where_im + r'$(t)$, imag'

		label21 = self.where_re + r'$(f)$, real'
		label22 = self.where_re + r'$(f)$, imag'
		label23 = self.where_im + r'$(f)$, real'
		label24 = self.where_im + r'$(f)$, imag'

		label31 = self.where_re + r'$(\lambda)$, real'
		label32 = self.where_re + r'$(\lambda)$, imag'
		label33 = self.where_im + r'$(\lambda)$, real'
		label34 = self.where_im + r'$(\lambda)$, imag'

		label4  = r'$abs(%s(self.t))$' %(self.where_re)
		label51 = r'$abs(%s(f))$' %(self.where_re)
		#label51 = r'$abs(%s_{re}(f))$' %(self.where_re)
		#label52 = r'$abs(%s_{im}(f))$' %(self.where_im)
		label61 = r'$abs(%s(\lambda))$' %(self.where_re)
		#label61 = r'$abs(%s_{real}(\lambda))$' %(self.where_re)
		#label62 = r'$abs(%s_{imag}(\lambda))$' %(self.where_im)

		# Source data in time domain.
		src_abs = np.sqrt((self.src_re)**2 + (self.src_im)**2)

		# Source data in self.frequency domain.
		nax = np.newaxis
		self.src_re_dft = (self.dt*self.src_re[nax,:] * np.exp(1.j*2.*np.pi*self.freq[:,nax]*self.tS[nax,:])).sum(1) / np.sqrt(2.*np.pi)
		self.src_im_dft = (self.dt*self.src_im[nax,:] * np.exp(1.j*2.*np.pi*self.freq[:,nax]*self.tS[nax,:])).sum(1) / np.sqrt(2.*np.pi)

		#np.save('./graph/self.src_re_dft.npy', self.src_re_dft)
		#np.save('./graph/self.src_im_dft.npy', self.src_im_dft)

		ax1.plot(self.tstepsS, self.src_re, color='b', label=label11) 
		ax1.plot(self.tstepsS, self.src_im, color='r', label=label12, linewidth='3', alpha=0.3)

		ax2.plot(self.freq_graph, self.src_re_dft.real, label=label21)                                    
		ax2.plot(self.freq_graph, self.src_re_dft.imag, label=label22) 

		ax3.plot(self.wvlen_graph, self.src_dft.real, label=label31)
		ax3.plot(self.wvlen_graph, self.src_dft.imag, label=label31)
		#ax3.plot(self.wvlen_graph, self.src_re_dft.real, label=label31)
		#ax3.plot(self.wvlen_graph, self.src_re_dft.imag, label=label32) 
		#ax3.plot(self.wvlen_graph, self.src_im_dft.real, label=label33, linewidth='5', alpha=0.3)             
		#ax3.plot(self.wvlen_graph, self.src_im_dft.imag, label=label34, linewidth='5', alpha=0.3)           

		ax4.plot(self.tstepsS, src_abs, color='b', label=label4)                                             

		ax5.plot(self.freq_graph, abs(self.src_dft)**2, label=label51) 
		#ax5.plot(self.freq_graph, abs(self.src_re_dft)**2, label=label51) 
		#ax5.plot(self.freq_graph, abs(self.src_im_dft)**2, linewidth='4', alpha=0.3, label=label52)

		ax6.plot(self.wvlen_graph, abs(self.src_dft)**2, label=label61) 
		#ax6.plot(self.wvlen_graph, abs(self.src_re_dft)**2, label=label61) 
		#ax6.plot(self.wvlen_graph, abs(self.src_im_dft)**2, linewidth='4', alpha=0.3, label=label62) 

		ax1.set_xlabel("time step")                        
		ax1.set_ylabel("Amp")
		ax1.legend(loc=self.loc) 
		ax1.grid(True)

		ax2.set_xlabel("freq({})" .format(freq_unit))
		ax2.set_ylabel("Amp")
		ax2.legend(loc=self.loc)
		ax2.grid(True)

		ax3.set_xlabel("wvlen({})" .format(wvlen_unit))
		ax3.set_ylabel("Amp")
		ax3.legend(loc=self.loc)
		ax3.grid(True)

		ax4.set_xlabel("time step")
		ax4.set_ylabel("Intensity")
		ax4.legend(loc=self.loc)
		ax4.grid(True)

		ax5.set_xlabel("freq({})" .format(freq_unit))
		ax5.set_ylabel("Intensity")
		ax5.legend(loc=self.loc)
		ax5.grid(True)
		ax5.set_ylim(0,None)

		ax6.set_xlabel("wvlen({})" .format(wvlen_unit))
		ax6.set_ylabel("Intensity")
		ax6.legend(loc=self.loc)
		ax6.grid(True)
		ax6.set_ylim(0,None)

		src_fig.savefig("./graph/src.png", bbox_inches='tight')          

	def plot_trs(self):

		trs_fig = plt.figure(figsize=self.figsize)

		ax1 = trs_fig.add_subplot(1,1,1)
		ax1.plot(self.tstepsT, self.trs_re, label='T, real')
		ax1.plot(self.tstepsT, self.trs_im, label='T, imag')
		ax1.grid(True)
		ax1.legend(loc=self.loc)
		
		trs_fig.savefig("./graph/trs.png", bbox_inches='tight')          

	def plot_ref(self):

		ref_fig = plt.figure(figsize=self.figsize)

		ax1 = ref_fig.add_subplot(2,3,1)
		ax1.plot(self.tstepsR, self.ref_re, label=r'$R(t)$, real')
		ax1.plot(self.tstepsR, self.ref_im, label=r'$R(t)$, imag')
		ax1.grid(True)
		ax1.legend(loc=self.loc)
		ax1.set_xlabel("time step")                        
		ax1.set_ylabel("Amp")
		
		ax2 = ref_fig.add_subplot(2,3,2)
		ax2.plot(self.freq_graph, self.ref_dft.real, label='real')
		ax2.plot(self.freq_graph, self.ref_dft.imag, label='imag')
		ax2.grid(True)
		ax2.legend(loc=self.loc)
		ax2.set_xlabel("freq({})" .format(freq_unit))                        
		ax2.set_ylabel("ref DFT")

		ax3 = ref_fig.add_subplot(2,3,3)
		ax3.plot(self.wvlen_graph, self.ref_dft.real, label='real')
		ax3.plot(self.wvlen_graph, self.ref_dft.imag, label='imag')
		ax3.grid(True)
		ax3.legend(loc=self.loc)
		ax3.set_xlabel("wvlen({})" .format(wvlen_unit))                        
		ax3.set_ylabel("ref DFT")

		ax4 = ref_fig.add_subplot(2,3,4)
		ax4.plot(self.freq_graph, abs(self.src_dft)**2, label='abs(src_dft)')
		ax4.plot(self.freq_graph, abs(self.ref_dft)**2, label='abs(ref_dft)')
		ax4.plot(self.freq_graph, abs(self.trs_dft)**2, label='abs(trs_dft)')
		ax4.grid(True)
		ax4.legend(loc=self.loc)
		ax4.set_xlabel("freq({})" .format(freq_unit))                        
		ax4.set_ylabel("src DFT")

		ax5 = ref_fig.add_subplot(2,3,5)
		ax5.plot(self.wvlen_graph, self.Ref, label='R')
		ax5.grid(True)
		ax5.legend(loc=self.loc)
		ax5.set_xlabel("wvlen({})" .format(wvlen_unit))                        
		ax5.set_ylabel("Ref")

		ax6 = ref_fig.add_subplot(2,3,6)
		ax6.plot(self.freq_graph, self.Ref, label='R')
		ax6.grid(True)
		ax6.legend(loc=self.loc)
		ax6.set_xlabel("freq({})" .format(freq_unit))                        
		ax6.set_ylabel("Ref")

		ref_fig.savefig("./graph/ref.png", bbox_inches='tight')          
	
	def plot_RT(self, **kwargs):

		figsize = (10,7)
		ylim    = 1.1
		Sum     = True

		for key, value in list(kwargs.items()):

			if key == 'figsize': figsize = value
			if key == 'xlim'   : xlim    = value
			if key == 'ylim'   : ylim    = value
			if key == 'Sum'    : Sum	 = value

		#----------------------------------------------------------------------#
		#------------------------ Plot freq vs ref and trs --------------------#
		#----------------------------------------------------------------------#

		freq_vs_RT = plt.figure(figsize=figsize)
		ax1 = freq_vs_RT.add_subplot(1,1,1)
		ax1.plot(self.freq_graph, self.Ref.real, color='g', label='Ref')
		ax1.plot(self.freq_graph, self.Trs.real, color='r', label='Trs')
		ax1.plot(self.freq_graph, self.Tot.real, color='b', label='Sum')

		ax1.set_xlabel("freq({})" .format(freq_unit))
		ax1.set_ylabel("Ratio")
		ax1.set_title("Ref,Trs")
		ax1.set_ylim(0, ylim)
		ax1.legend(loc='best')
		ax1.grid(True)

		ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%5.1f'))
		freq_vs_RT.savefig("./graph/freq_vs_RT.png", format='png', bbox_inches='tight')

		#----------------------------------------------------------------------#
		#----------------------- Plot wvlen vs ref and trs --------------------#
		#----------------------------------------------------------------------#

		wvlen_vs_RT = plt.figure(figsize=figsize)
		ax1 = wvlen_vs_RT.add_subplot(1,1,1)
		ax1.plot(self.wvlen_graph, self.Ref.real, color='g', label='Ref')
		ax1.plot(self.wvlen_graph, self.Trs.real, color='r', label='Trs')
		ax1.plot(self.wvlen_graph, self.Tot.real, color='b', label='Sum')

		ax1.set_xlabel("wavelength({})" .format(wvlen_unit))
		ax1.set_ylabel("Ratio")
		ax1.set_title("Ref,Trs")
		ax1.set_ylim(0, ylim)
		ax1.legend(loc='best')
		ax1.grid(True)

		ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%5.1f'))
		wvlen_vs_RT.savefig("./graph/wvlen_vs_RT.png", format='png', bbox_inches='tight')


if __name__ == "__main__":

	um = 1e-6
	nm = 1e-9
	Lx, Ly, Lz = 384*um, 96*um, 96*um
	Nx, Ny, Nz = 128, 32, 32
	dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz
	courant = 1./4

	wv_srt = 60*um
	wv_end = 100*um
	interval = 0.2*um
	spread = 0.3
	cutter1 = 1300

	wvlen_unit = 'um'
	freq_unit = 'THz'

	painter = SRTpainter('Ey', 'Ey', dx, dy, dz, courant, wv_srt, wv_end, interval, spread, cutter1, wvlen_unit, freq_unit)
	painter.plot_src()
	painter.plot_trs()
	painter.plot_ref()
	painter.plot_RT(ylim=None)
