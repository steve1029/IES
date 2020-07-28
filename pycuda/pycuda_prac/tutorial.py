import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import atexit
from datetime import datetime as dtm

a = np.random.random_integers(1,10,size=(4,4))
print('dtype of a :',a.dtype)

a = a.astype(np.float32)	# change datatype into 32bit float (single precision number)
print('dtype of a :',a.dtype)

# create a space in gpu to locate 'data a'.


drv.init()
dev = drv.Device(0)
ctx = dev.make_context()
atexit.register(ctx.pop, *[])

print(dev.name())
print('a occupies',a.nbytes,'bytes')
a_gpu = drv.mem_alloc(a.nbytes)	
# argument should be the magnitude of the data to transfer.

drv.memcpy_htod(a_gpu,a)

mod = SourceModule("""
	__global__ void doublify(float *a)
	{
		int idx = threadIdx.x + threadIdx.y*blockDim.x;
		a[idx] *= 2;
		
	}
	""")

func = mod.get_function("doublify")

t0 = dtm.now()
func(a_gpu,block=(4,4,1))
t1 = dtm.now()
dt_gpu = t1 - t0

t0 = dtm.now()
b = 2*a
t1 = dtm.now()
dt_cpu = t1 - t0

a_doubled = np.empty_like(a)
drv.memcpy_dtoh(a_doubled,a_gpu)

print(a_doubled)
print(a)
print(a_doubled - b)
print(dt_cpu, dt_gpu)
