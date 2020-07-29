#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define PI 3.1415926535897

/**********************************************************/
/******************** Function Prototype ******************/
/**********************************************************/

void do_RFT_to_get_Sx(
	int MPIrank,
	int Nf,	  int tstep,
	int Ny,	  int Nz,
	int xsrt, int xend,
	int ysrt, int yend,
	int zsrt, int zend,
	double dt, double dy, double dz,
	double* freqs,
	double* DFT_Ey_re, double* DFT_Ez_re,
	double* DFT_Ey_im, double* DFT_Ez_im,
	double* DFT_Hy_re, double* DFT_Hz_re,
	double* DFT_Hy_im, double* DFT_Hz_im,
	double* Ey_re, double* Ez_re,
	double* Hy_re, double* Hz_re
);

void do_RFT_to_get_Sy(
	int MPIrank,
	int Nf,	  int tstep,
	int Ny,	  int Nz,
	int xsrt, int xend,
	int ysrt, int yend,
	int zsrt, int zend,
	double dt, double dx, double dz,
	double* freqs,
	double* DFT_Ex_re, double* DFT_Ez_re,
	double* DFT_Ex_im, double* DFT_Ez_im,
	double* DFT_Hx_re, double* DFT_Hz_re,
	double* DFT_Hx_im, double* DFT_Hz_im,
	double* Ex_re, double* Ez_re,
	double* Hx_re, double* Hz_re
);

void do_RFT_to_get_Sz(
	int MPIrank,
	int Nf,	  int tstep,
	int Ny,	  int Nz,
	int xsrt, int xend,
	int ysrt, int yend,
	int zsrt, int zend,
	double dt, double dx, double dy,
	double* freqs,
	double* DFT_Ex_re, double* DFT_Ey_re,
	double* DFT_Ex_im, double* DFT_Ey_im,
	double* DFT_Hx_re, double* DFT_Hy_re,
	double* DFT_Hx_im, double* DFT_Hy_im,
	double* Ex_re, double* Ey_re,
	double* Hx_re, double* Hy_re
);

/**********************************************************/
/******************** Function Definition *****************/
/**********************************************************/

void do_RFT_to_get_Sx(
	int MPIrank,
	int Nf,	  int tstep,
	int Ny,	  int Nz,
	int xsrt, int xend,
	int ysrt, int yend,
	int zsrt, int zend,
	double dt, double dy, double dz,
	double* freqs,
	double* DFT_Ey_re, double* DFT_Ez_re,
	double* DFT_Ey_im, double* DFT_Ez_im,
	double* DFT_Hy_re, double* DFT_Hz_re,
	double* DFT_Hy_im, double* DFT_Hz_im,
	double* Ey_re, double* Ez_re,
	double* Hy_re, double* Hz_re
){

	int i, j, k, f;
	int Fidx, Sidx; // Field yz-plane index, Sx yz-plane index
	int nx, ny, nz;

	int omp_max = omp_get_max_threads();
	omp_set_num_threads(omp_max);
	int omp_in_use = omp_get_num_threads();
	int omp_me = omp_get_thread_num();

	i  = xsrt;
	nx = xend - xsrt;
	ny = yend - ysrt;
	nz = zend - zsrt;

	/*
	#pragma omp parallel for \
		shared( ny, nz, dt,\
				freqs, \
				DFT_Ey_re, DFT_Ey_im, \
				DFT_Ez_re, DFT_Ez_im, \
				DFT_Hy_re, DFT_Hy_im, \
				DFT_Hz_re, DFT_Hz_im, \
			) \
		private(f, j, k, Nf, ysrt, yend, zsrt, zend, idx, tstep)
	*/
	#pragma omp parallel for schedule(dynamic)
	for(f=0; f<Nf; f++){
		for(j=0; j<ny; j++){
			for(k=0; k<nz; k++){
				
				Sidx   = (k     ) + (j     )*nz + (f  )*ny*nz;
				Fidx   = (k+zsrt) + (j+ysrt)*Nz + (i  )*Ny*Nz;

				// First term.
				DFT_Ey_re[Sidx] +=  Ey_re[Fidx] * cos(2.*PI*freqs[f]*tstep*dt) * dt;
				DFT_Ey_im[Sidx] += -Ey_re[Fidx] * sin(2.*PI*freqs[f]*tstep*dt) * dt;

				DFT_Hz_re[Sidx] +=  Hz_re[Fidx] * cos(2.*PI*freqs[f]*tstep*dt) * dt;
				DFT_Hz_im[Sidx] += -Hz_re[Fidx] * sin(2.*PI*freqs[f]*tstep*dt) * dt;

				// Second term.
				DFT_Ez_re[Sidx] +=  Ez_re[Fidx] * cos(2.*PI*freqs[f]*tstep*dt) * dt;
				DFT_Ez_im[Sidx] += -Ez_re[Fidx] * sin(2.*PI*freqs[f]*tstep*dt) * dt;

				DFT_Hy_re[Sidx] +=  Hy_re[Fidx] * cos(2.*PI*freqs[f]*tstep*dt) * dt;
				DFT_Hy_im[Sidx] += -Hy_re[Fidx] * sin(2.*PI*freqs[f]*tstep*dt) * dt;
			}
		}
	}

	return;

}

void do_RFT_to_get_Sy(
	int MPIrank,
	int Nf,	  int tstep,
	int Ny,	  int Nz,
	int xsrt, int xend,
	int ysrt, int yend,
	int zsrt, int zend,
	double dt, double dx, double dz,
	double* freqs,
	double* DFT_Ex_re, double* DFT_Ez_re,
	double* DFT_Ex_im, double* DFT_Ez_im,
	double* DFT_Hx_re, double* DFT_Hz_re,
	double* DFT_Hx_im, double* DFT_Hz_im,
	double* Ex_re, double* Ez_re,
	double* Hx_re, double* Hz_re
){

	int i, j, k, f;
	int Fidx, Sidx; // Field xz-plane index, Sy xz-plane index
	int nx, ny, nz;

	int omp_max = omp_get_max_threads();
	omp_set_num_threads(omp_max);

	int omp_in_use = omp_get_num_threads();
	int omp_me = omp_get_thread_num();

	j  = ysrt;
	nx = xend - xsrt;
	ny = yend - ysrt;
	nz = zend - zsrt;

	/*
	printf("%d, %d, %d\n", nx, ny, nz);

	if(MPIrank == 0){
		printf("%d, %d\n", xsrt, xend);
		printf("%d, %d\n", ysrt, yend);
		printf("%d, %d\n", zsrt, zend);
	}
	*/
	
	/*
	#pragma omp parallel for \
		shared(Ex_re, Ez_re, Hx_re, Hz_re, freqs, tstep, dt, \
			DFT_Ex_re, DFT_Ex_im, \
			DFT_Ez_re, DFT_Ez_im, \
			DFT_Hx_re, DFT_Hx_im, \
			DFT_Hz_re, DFT_Hz_im, \
			) \
		private(Nf, nx, nz, i, k, f, Sidx, Fidx)
	*/
	#pragma omp parallel for schedule(dynamic)
	for(f=0; f<Nf; f++){
		for(i=0; i<nx; i++){
			for(k=0; k<nz; k++){
				
				Sidx   = (k     ) + (i)*nz + (f     )*nx*nz;
				Fidx   = (k+zsrt) + (j)*Nz + (i+xsrt)*Ny*Nz;

				//if(MPIrank == 0) printf("%d, %d\n", Sidx, Fidx);
				//if(MPIrank == 0) printf("%d, %d\n", Sidx-f*nx*nz, Fidx);
				//if(MPIrank == 2) printf("%d, %d\n", i+xsrt, k+zsrt);
				//if(MPIrank == 4) printf("%d, %d\n", Sidx-f*nx*nz, Fidx);

				DFT_Ex_re[Sidx] +=  Ex_re[Fidx] * cos(2.*PI*freqs[f]*tstep*dt) * dt;
				DFT_Ex_im[Sidx] += -Ex_re[Fidx] * sin(2.*PI*freqs[f]*tstep*dt) * dt;

				DFT_Hz_re[Sidx] +=  Hz_re[Fidx] * cos(2.*PI*freqs[f]*tstep*dt) * dt;
				DFT_Hz_im[Sidx] += -Hz_re[Fidx] * sin(2.*PI*freqs[f]*tstep*dt) * dt;

				DFT_Ez_re[Sidx] +=  Ez_re[Fidx] * cos(2.*PI*freqs[f]*tstep*dt) * dt;
				DFT_Ez_im[Sidx] += -Ez_re[Fidx] * sin(2.*PI*freqs[f]*tstep*dt) * dt;

				DFT_Hx_re[Sidx] +=  Hx_re[Fidx] * cos(2.*PI*freqs[f]*tstep*dt) * dt;
				DFT_Hx_im[Sidx] += -Hx_re[Fidx] * sin(2.*PI*freqs[f]*tstep*dt) * dt;
			}
		}
	}

	return;

}

void do_RFT_to_get_Sz(
	int MPIrank,
	int Nf,	  int tstep,
	int Ny,	  int Nz,
	int xsrt, int xend,
	int ysrt, int yend,
	int zsrt, int zend,
	double dt, double dx, double dy,
	double* freqs,
	double* DFT_Ex_re, double* DFT_Ey_re,
	double* DFT_Ex_im, double* DFT_Ey_im,
	double* DFT_Hx_re, double* DFT_Hy_re,
	double* DFT_Hx_im, double* DFT_Hy_im,
	double* Ex_re, double* Ey_re,
	double* Hx_re, double* Hy_re
){

	int i, j, k, f;
	int Fidx, Sidx; // Field xy-plane index, Sz xy-plane index
	int nx, ny, nz;

	int omp_max = omp_get_max_threads();
	omp_set_num_threads(omp_max);

	int omp_in_use = omp_get_num_threads();
	int omp_me = omp_get_thread_num();

	k  = zsrt;
	nx = xend - xsrt;
	ny = yend - ysrt;
	nz = zend - zsrt;

	#pragma omp parallel for schedule(dynamic)
	for(f=0; f<Nf; f++){
		for(i=0; i<nx; i++){
			for(j=0; j<ny; j++){
				
				// idx for Sz
				Sidx   = (j   ) + (i     )*ny + (f     )*nx*ny;
				Fidx   = (zsrt) + (j+ysrt)*Nz + (i+xsrt)*Ny*Nz;

				DFT_Ex_re[Sidx] +=  Ex_re[Fidx] * cos(2*PI*freqs[f]*tstep*dt) * dt;
				DFT_Ex_im[Sidx] += -Ex_re[Fidx] * sin(2*PI*freqs[f]*tstep*dt) * dt;

				DFT_Ey_re[Sidx] +=  Ey_re[Fidx] * cos(2*PI*freqs[f]*tstep*dt) * dt;
				DFT_Ey_im[Sidx] += -Ey_re[Fidx] * sin(2*PI*freqs[f]*tstep*dt) * dt;

				DFT_Hx_re[Sidx] +=  Hx_re[Fidx] * cos(2*PI*freqs[f]*tstep*dt) * dt;
				DFT_Hx_im[Sidx] += -Hx_re[Fidx] * sin(2*PI*freqs[f]*tstep*dt) * dt;

				DFT_Hy_re[Sidx] +=  Hy_re[Fidx] * cos(2*PI*freqs[f]*tstep*dt) * dt;
				DFT_Hy_im[Sidx] += -Hy_re[Fidx] * sin(2*PI*freqs[f]*tstep*dt) * dt;
			}
		}
	}

	return;

}
