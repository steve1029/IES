# %%
from scipy.linalg.blas import dgemm, sgemm
import pycuda.driver as drv
from pycuda import gpuarray, tools
import numpy as np
from skcuda import cublas
from time import time

m = 5000
n = 10000
k = 10000

def compute_gflops_cpu(precision='D'):

	if precision=='S':
		float_type = 'float32'
	elif precision=='D':
		float_type = 'float64'
	else:
		return -1

	A = np.random.randn(m, k).astype(float_type)
	B = np.random.randn(k, n).astype(float_type)
	C = np.random.randn(m, n).astype(float_type)

	alpha = np.random.randn()
	beta = np.random.randn()

	trans_A = 0
	trans_B = 0
	ow_C = 1

	t = time()

	exec('%sgemm(alpha, A, B, beta, C, trans_A, trans_B, ow_C)' %(precision.lower()))

	t = time() - t

	gflops = 2*m*n*(k+1)*(10**-9) / t 
	return gflops

def compute_gflops_gpu(precision='S'):

	if precision=='S':
		float_type = 'float32'
	elif precision=='D':
		float_type = 'float64'
	else:
		return -1
		
	A = np.random.randn(m, k).astype(float_type)
	B = np.random.randn(k, n).astype(float_type)
	C = np.random.randn(m, n).astype(float_type)

	A_cm = A.T.copy()
	B_cm = B.T.copy()
	C_cm = C.T.copy()

	A_gpu = gpuarray.to_gpu(A_cm)
	B_gpu = gpuarray.to_gpu(B_cm)
	C_gpu = gpuarray.to_gpu(C_cm)

	alpha = np.random.randn()
	beta = np.random.randn()

	transa = cublas._CUBLAS_OP['N']
	transb = cublas._CUBLAS_OP['N']

	lda = m
	ldb = k
	ldc = m

	t = time()
	handle = cublas.cublasCreate()
	
	exec('cublas.cublas%sgemm(handle, transa, transb, m, n, k, alpha, A_gpu.gpudata, lda, \
						B_gpu.gpudata, ldb, beta, C_gpu.gpudata, ldc)' % precision)
	
	cublas.cublasDestroy(handle)
	t = time() - t

	gflops = 2*m*n*(k+1)*(10**-9) / t 
	
	return gflops

# %%

if __name__ == '__main__':

	print('Single-precision performance: {} GFLOPS' .format(compute_gflops_cpu('S')))
	print('Double-precision performance: {} GFLOPS' .format(compute_gflops_cpu('D')))

# %%
if __name__ == '__main__':

	drv.init()
	dev = drv.Device(0)
	ctx = dev.make_context()

	print('Single-precision performance: {} GFLOPS' .format(compute_gflops_gpu('S')))
	print('Double-precision performance: {} GFLOPS' .format(compute_gflops_gpu('D')))

	ctx.pop()
	tools.clear_context_caches()
