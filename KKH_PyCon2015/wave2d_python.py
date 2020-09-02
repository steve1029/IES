#!/usr/bin/env python3
import datetime
import numpy as np
#import matplotlib.pyplot as plt

def update_core(f,g,c):
	sl = slice(1,-1)
	sls = (sl, sl)
	f[sls] = c[sls] * (g[2:,sl] + g[:-2,sl] + g[sl,2:] + g[sl,:-2] - 4*g[sls]) + 2*g[sls] - f[sls]

# setup
nx, ny = 1000, 1000
tmax, tgap = 1000, 100


# allocation
c = np.ones((nx,ny))*0.25
f = np.zeros_like(c)
g = np.zeros_like(c)

# main time loop
sl = slice(1,-1)
sls = (sl, sl)

# plot
#imag = plt.imshow(f, vmin=-0.1, vmax=0.1)
#plt.colorbar()

t0 = datetime.datetime.now()

for tstep in np.arange(1, tmax+1):

	update_core(f,g,c)
	update_core(g,f,c)

	g[int(nx/2),int(ny/2)] = np.sin(0.4 * tstep)

	if tstep % tgap == 0:
		t1 = datetime.datetime.now()
		print('tstep=%d, time: %s' % (tstep, t1-t0))
		#imag.set_array(f)
		#plt.savefig('./png/%.4d.png' %tstep)
