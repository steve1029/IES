#!/usr/bin/env python3
import numpy as np

# setup
nx, ny = 1000, 800
tmax, tgap = 600, 100


# allocation
c = np.ones((nx,ny))*0.25
f = np.zeros_like(c)
g = np.zeros_like(c)

# main time loop
sl = slice(1,-1)
sls = (sl, sl)

# plot
import matplotlib.pyplot as plt
imag = plt.imshow(f, vmin=-0.1, vmax=0.1)
plt.colorbar()

for tstep in np.arange(1, tmax+1):

	f[sls] = c[sls] * (g[2:,sl] + g[:-2,sl] + g[sl,2:] + g[sl,:-2]-4*g[sls]) + 2*g[sls]-f[sls]
	g[sls] = c[sls] * (f[2:,sl] + f[:-2,sl] + f[sl,2:] + f[sl,:-2]-4*f[sls]) + 2*f[sls]-g[sls]
	g[int(nx/2),int(ny/2)] = np.sin(0.4 * tstep)

	if tstep % tgap == 0:
		print('tstep=%d' % tstep)
		imag.set_array(f)
		plt.savefig('./png/%.4d.png' %tstep)
