import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

#folder = "HPF rfft/512_256_256_10um_20um_200umto600um"
folder = "HPF rfft/512_128_128_10um_40um_200to600_n2_r100um/"

freqs = np.load(folder+"freqs.npy")
wvlens = c/freqs
#wvlens = wvlens / 1e-6
#freqs = freqs/1e12

Sx_L = np.load(folder+"/Sx_SF_L_area.npy")
Sx_R = np.load(folder+"/Sx_SF_R_area.npy")

Sy_L = np.load(folder+"/Sy_SF_L_area.npy")
Sy_R = np.load(folder+"/Sy_SF_R_area.npy")

Sz_L = np.load(folder+"/Sz_SF_L_area.npy")
Sz_R = np.load(folder+"/Sz_SF_R_area.npy")

source = np.load(folder+"/Sx_IF_R_area.npy")

radius = 100e-6 # radius of a dielectric sphere.
area = np.pi * radius**2 # area of a cross section.

source = source / (2560e-6)**2 # incident energy per unit area over the scattering cubic.
#source = source * np.pi * radius**2
scattered = abs(Sx_L) + abs(Sx_R) + abs(Sy_L) + abs(Sy_R) + abs(Sz_L) + abs(Sz_R)
scattered = scattered / area
#scattered = scattered / area
scattered_ratio = (scattered / source) # the unit of ratio is m^2.

fig = plt.figure(figsize=(14,7))

ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.set_title("Energy spectrum")
ax1.set_xlabel("frequency(THz)")
ax1.set_ylabel("Scattered E/m^2 / input E/m^2")

ax1.plot(freqs, scattered_ratio, linewidth=1, alpha=1, label="ratio")
ax1.grid(True)
ax1.legend(loc="best")
ax1.set_ylim(0,None)
#ax1.set_xlim(1.4e12,None)

ax2.set_title("Energy spectrum")
ax2.set_xlabel("wavelength(um)")
ax2.set_ylabel("Scattered E/m^2 / input E/m^2")
ax2.plot(wvlens/1e-6, scattered_ratio, linewidth=1, alpha=1, label="ratio")
ax2.grid(True)
ax2.legend(loc="best")
ax2.set_ylim(0,None)
#ax2.set_xlim(200e-6,300e-6)

fig.savefig(folder+"/S_spectrum_ratio.png")