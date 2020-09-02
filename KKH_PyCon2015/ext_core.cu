__global__ void update_core(double *f, double *g, double *c, int nx, int ny){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i = tid / ny;
	int j = tid % ny;

	if(i > 0 && j > 0 && i < nx-1 && j < ny-1){
		f[tid] = c[tid] * (g[tid-ny] + g[tid+ny] + g[tid-1] + g[tid+1] - 4*g[tid]) + 2*g[tid] - f[tid];
	}
}

__global__ void update_src(double *f, double val, int idx0){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid==0) f[idx0] += val;
}
