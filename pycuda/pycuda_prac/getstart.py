import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
from datetime import  datetime as dtm
#from pyfftw.cuda import Plan

drv.init()
print(drv.Device(0))
print(drv.Device(1))
