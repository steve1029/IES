import pycuda
import pycuda.driver as drv
import numpy as np
import atexit

print("pycuda module version : ",pycuda.VERSION)
# print(pycuda.VERSION_TEXT)
drv.init()
print("CUDA toolkit driver version : ",drv.get_version())
print("cuda gpu in this system : ",drv.Device.count())

dev = drv.Device(0)
print("GPU name : ",dev.name())
print("If you want to check device attributes,\
	check this dictionary : ",type(dev.get_attributes()))
ctx = dev.make_context()
print(ctx.get_device())
# ctx.pop()
atexit.register(ctx.pop,)

print("global memory (free,total) : ",\
	[drv.mem_get_info()[i]/1024/1024 for i in range(2)], 'MB')

a = np.arange(10)
# print(a)
a_gpu = drv.mem_alloc(a.nbytes)
drv.memcpy_htod(a_gpu,a)
a_rcv = np.empty_like(a)
print("a_rcv before: ",a_rcv)
drv.memcpy_dtoh(a_rcv,a_gpu)
print("a_rcv after: ",a_rcv)
# ctx.push()
# ctx.detach()
