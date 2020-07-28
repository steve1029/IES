#!/usr/bin/env python3
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit

# setup
nx, ny = 1000, 1000
tmax, tgap = 1000, 100

# allocation
c = np.ones((nx,ny))*0.25
f = np.zeros_like(c)

c_gpu = cuda.to_device(c)
f_gpu = cuda.to_device(f)
g_gpu = cuda.to_device(f)

# cuda kernels
from pycuda.compiler import SourceModule
kernels = open("./ext_core.cu").read()
mod = SourceModule(kernels)
update_core = mod.get_function('update_core')
update_src = mod.get_function('update_src')

bs, gs = (256,1,1), (int(nx*ny/256)+1,1)
nnx, nny = np.int32(nx), np.int32(ny)
src_val = lambda tstep: np.sin(0.4 * tstep)
src_idx = np.int32((nx/2)*ny + ny/2)

# plot
#imag = plt.imshow(f, vmin=-0.1, vmax=0.1)
#plt.colorbar()

t0 = datetime.datetime.now()

# main time loop
for tstep in np.arange(1, tmax+1):

	update_core(f_gpu,g_gpu,c_gpu, nnx, nny, block=bs, grid=gs)
	update_core(g_gpu,f_gpu,c_gpu, nnx, nny, block=bs, grid=gs)
	update_src (g_gpu,src_val(tstep), src_idx, block=bs, grid=(1,1))

	if tstep % tgap == 0:
		t1 = datetime.datetime.now()
		print('tstep=%d, time:%s' % (tstep, t1-t0))
		#f = cuda.from_device_like(f_gpu, f)
		#imag.set_array(f)
		#plt.savefig('./png/%.4d.png' %tstep)
