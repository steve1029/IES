# %%
from scipy.linalg.blas import dgemm, sgemm
import numpy as np
import cupy as cp
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
            
    A = cp.random.randn(m, k).astype(float_type)
    B = cp.random.randn(k, n).astype(float_type)
    C = cp.random.randn(m, n).astype(float_type)

    alpha = np.random.randn()
    beta = np.random.randn()

    t = time()

    C = alpha * cp.dot(A,B) + beta * C
    
    t = time() - t

    gflops = 2*m*n*(k+1)*(10**-9) / t 
    
    return gflops

# %%

if __name__ == '__main__':

    #print('Single-precision performance: {} GFLOPS' .format(compute_gflops_cpu('S')))
    #print('Double-precision performance: {} GFLOPS' .format(compute_gflops_cpu('D')))

    print('Single-precision performance: {} GFLOPS' .format(compute_gflops_gpu('S')))
    print('Double-precision performance: {} GFLOPS' .format(compute_gflops_gpu('D')))
