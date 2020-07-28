#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
//#include <omp.h>
//#include <complex.h>

/*	
	Author: Donggun Lee	
	Date  : 18.02.14

	This script only contains update equations of the hybrid PSTD-FDTD method.
	Update equations for UPML or CPML, ADE-FDTD is not developed here.

	Core update equations are useful when testing the speed of an algorithm
	or the performance of the hardware such as CPU, GPU or memory.

	Update Equations for E and H field
	Update Equations for MPI Boundary

	Discription of variables
	------------------------
	i: x index
	j: y index
	k: z index
	
	myidx  : 1 dimensional index of elements where its index in 3D is (i  , j  , k  ).
	i_myidx: 1 dimensional index of elements where its index in 3D is (i+1, j  , k  ).
	myidx_i: 1 dimensional index of elements where its index in 3D is (i-1, j  , k  ).
	myidx_0: 1 dimensional index of elements where its index in 3D is (0  , j  , k  ).

	ex)
		myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
		i_myidx = (k  ) + (j  ) * Nz + (i+1) * Nz * Ny;
		myidx_i = (k  ) + (j  ) * Nz + (i-1) * Nz * Ny;
		myidx_0 = (k  ) + (j  ) * Nz + (0  ) * Nz * Ny;

*/

/***********************************************************************************/
/******************************** FUNCTION DECLARATION *****************************/
/***********************************************************************************/

// Get derivatives along last axis.
void get_deriv_last_axis(
	int myNx, int Ny, int Nz,
	double* data,
	double* kz,
	double* diffz_data
);

//Get derivatives of E field.
void get_deriv_z_E_FML(
	int myNx, int Ny, int Nz,
	double* Ex_re,
	double* Ey_re,
	double* kz,
	double* diffzEx_re,
	double* diffzEy_re
);

void get_deriv_y_E_FML(
	int myNx, int Ny, int Nz,
	double* Ex_re,
	double* Ez_re,
	double* ky,
	double* diffyEx_re,
	double* diffyEz_re
);

void get_deriv_x_E_FM0(
	int	myNx, int Ny, int Nz,
	double  dx,
	double* Ey_re,
	double* Ez_re,
	double* diffxEy_re,
	double* diffxEz_re,
	double* recvEylast_re,
	double* recvEzlast_re
);

void get_deriv_x_E_00L(
	int	myNx, int Ny, int Nz,
	double  dx,
	double* Ey_re,
	double* Ez_re,
	double* diffxEy_re,
	double* diffxEz_re
);

// Update H field.
void updateH(										\
	int		MPIsize,	int MPIrank,
	int		myNx,		int		Ny,		int		Nz,	\
	double  dt,										\
	double *Hx_re,		
	double *Hy_re,	
	double *Hz_re,	
	double *mu_HEE,		double *mu_EHH,				\
	double *mcon_HEE,	double *mcon_EHH,			\
	double *diffxEy_re, 
	double *diffxEz_re, 
	double *diffyEx_re,
	double *diffyEz_re,
	double *diffzEx_re,
	double *diffzEy_re
);

// Get derivatives of H field.
void get_deriv_z_H_FML(
	int myNx, int Ny, int Nz,
	double* Hx_re,
	double* Hy_re,
	double* kz,
	double* diffzHx_re,
	double* diffzHy_re
);

void get_deriv_y_H_FML(
	int myNx, int Ny, int Nz,
	double* Hx_re,
	double* Hz_re,
	double* ky,
	double* diffyHx_re,
	double* diffyHz_re
);

void get_deriv_x_H_F00(
	int myNx, int Ny, int Nz,
	double dx,
	double* Hy_re,
	double* Hz_re,
	double* diffxHy_re,
	double* diffxHz_re
);

void get_deriv_x_H_0ML(
	int myNx, int Ny, int Nz,
	double dx,
	double* Hy_re,
	double* Hz_re,
	double* diffxHy_re,
	double* diffxHz_re,
	double* recvHyfirst_re,
	double* recvHzfirst_re
);

// Update E field
void updateE(
	int		MPIsize,	int MPIrank,
	int		myNx,		int		Ny,		int		Nz,	\
	double dt,										\
	double *Ex_re,		
	double *Ey_re,		
	double *Ez_re,		
	double *eps_HEE,	double *eps_EHH,			\
	double *econ_HEE,	double *econ_EHH,			\
	double *diffxHy_re, 
	double *diffxHz_re, 
	double *diffyHx_re, 
	double *diffyHz_re, 
	double *diffzHx_re, 
	double *diffzHy_re
);

/***********************************************************************************/
/******************************** FUNCTION DESCRIPTION *****************************/
/***********************************************************************************/

void get_deriv_last_axis(
	int Nx, int Ny, int Nz,
	double* data,
	double* kz,
	double* diffz_data
){

	int Nzh = Nz/2+1;

	int i,j,k,idx,idx_T;
	double real, imag;

	double* diffz = fftw_alloc_real(Nx*Ny*Nz);
	fftw_complex *FFTz = fftw_alloc_complex(Nx*Ny*Nzh);

	// Set forward FFTz parameters
	int rankz = 1;
	int nz[1] = {Nz};
	int howmanyz = (Nx*Ny);
	const int *inembedz = NULL, *onembedz = NULL;
	int istridez = 1, ostridez = 1;
	int idistz = Nz, odistz= Nzh;

	// Setup Forward plans.
	fftw_plan FFTz_for_plan = fftw_plan_many_dft_r2c(rankz, nz, howmanyz, diffz, inembedz, istridez, idistz, \
														FFTz, onembedz, ostridez, odistz, FFTW_ESTIMATE);

	// Set backward FFTz parameters
	int rankbz = 1;
	int nbz[1] = {Nz};
	int howmanybz = (Nx*Ny);
	const int *inembedbz = NULL, *onembedbz = NULL;
	int istridebz = 1, ostridebz = 1;
	int idistbz = Nzh, odistbz = Nz;

	// Setup Backward plans.
	fftw_plan FFTz_bak_plan = fftw_plan_many_dft_c2r(rankbz, nbz, howmanybz, FFTz, inembedbz, istridebz, idistbz, \
														diffz, onembedbz, ostridebz, odistbz, FFTW_ESTIMATE);

	// Initialize diffz.
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				idx   = k + j*Nz + i*Nz*Ny;

				diffz[idx] = data[idx];
			}
		}
	}

	// Perform 1D FFT along z axis.
	fftw_execute(FFTz_for_plan);

	// Multiply ikz.
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nzh; k++){
				
				idx   = k + j*Nzh + i*Nzh*Ny;

				real = FFTz[idx][0];
				imag = FFTz[idx][1];

				FFTz[idx][0] = -kz[k] * imag;
				FFTz[idx][1] =  kz[k] * real;

			}
		}
	}

	// Perform 1D IFFT along y and z axis.
	fftw_execute(FFTz_bak_plan);

	// Normalize reconstructed signal.
	// Transpose to restore original field array.
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				idx   = k + j*Nz + i*Nz*Ny;

				diffz_data[idx] = diffz[idx] / Nz;

			}
		}
	}

	// Destroy the plan
	fftw_destroy_plan(FFTz_for_plan);
	fftw_destroy_plan(FFTz_bak_plan);

	// Free the memory.
	fftw_free(FFTz);
	fftw_free(diffz);

	return;
}

// Get derivatives of E field in the first and middle rank.
void get_deriv_z_E_FML(
	int myNx, int Ny, int Nz,
	double* Ex_re,
	double* Ey_re,
	double* kz,
	double* diffzEx_re,
	double* diffzEy_re
){
	// int for index
	int i, j, k, myidx;
	double real, imag;

	// Memory allocation for transpose of the field data.
	fftw_complex *FFTzEx = fftw_alloc_complex(myNx*Ny*(Nz/2+1));
	fftw_complex *FFTzEy = fftw_alloc_complex(myNx*Ny*(Nz/2+1));

	// Set Plans for real FFT along z axis.
	int rankz = 1;
	int nz[1]  = {Nz};
	int howmanyz = myNx*Ny;
	int istridez = 1, ostridez = 1;
	int idistz = Nz, odistz = Nz/2+1;
	const int *inembedz = NULL, *onembedz = NULL;

	fftw_plan FFTz_Ex_FOR_plan = fftw_plan_many_dft_r2c(rankz, nz, howmanyz, Ex_re, inembedz, istridez, idistz, \
														FFTzEx, onembedz, ostridez, odistz, FFTW_ESTIMATE);

	fftw_plan FFTz_Ey_FOR_plan = fftw_plan_many_dft_r2c(rankz, nz, howmanyz, Ey_re, inembedz, istridez, idistz, \
														FFTzEy, onembedz, ostridez, odistz, FFTW_ESTIMATE);

	// Set Plans for inverse real FFT along z axis.
	int rankbz = 1;
	int nbz[1]  = {Nz};
	int howmanybz = myNx*Ny;
	int istridebz = 1, ostridebz = 1;
	int idistbz = Nz/2+1, odistbz = Nz;
	const int *inembedbz = NULL, *onembedbz = NULL;

	// Setup BACKWARD plans.
	fftw_plan FFTz_Ex_BAK_plan = fftw_plan_many_dft_c2r(rankbz, nbz, howmanybz, FFTzEx, inembedbz, istridebz, idistbz, \
														diffzEx_re, onembedbz, ostridebz, odistbz, FFTW_ESTIMATE);

	fftw_plan FFTz_Ey_BAK_plan = fftw_plan_many_dft_c2r(rankbz, nbz, howmanybz, FFTzEy, inembedbz, istridebz, idistbz, \
														diffzEy_re, onembedbz, ostridebz, odistbz, FFTW_ESTIMATE);

	fftw_execute(FFTz_Ex_FOR_plan);
	fftw_execute(FFTz_Ey_FOR_plan);

	// Multiply ikz.
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < (Nz/2+1); k++){

				myidx = k + j*(Nz/2+1) + i*(Nz/2+1)*Ny;

				real = FFTzEx[myidx][0];
				imag = FFTzEx[myidx][1];

				FFTzEx[myidx][0] = -kz[k] * imag;
				FFTzEx[myidx][1] =  kz[k] * real;

				real = FFTzEy[myidx][0];
				imag = FFTzEy[myidx][1];

				FFTzEy[myidx][0] = -kz[k] * imag;
				FFTzEy[myidx][1] =  kz[k] * real;

			}
		}
	}

	// Backward FFT.
	fftw_execute(FFTz_Ex_BAK_plan);
	fftw_execute(FFTz_Ey_BAK_plan);

	// Normalize the results of pseudo-spectral method.
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  )*Nz + (i  )*Nz*Ny;

				diffzEx_re[myidx] = diffzEx_re[myidx] / Nz;
				diffzEy_re[myidx] = diffzEy_re[myidx] / Nz;

			}
		}
	}

	// Destroy the plan and free the memory.
	fftw_destroy_plan(FFTz_Ex_FOR_plan);
	fftw_destroy_plan(FFTz_Ey_FOR_plan);
	fftw_destroy_plan(FFTz_Ex_BAK_plan);
	fftw_destroy_plan(FFTz_Ey_BAK_plan);
	fftw_free(FFTzEx);
	fftw_free(FFTzEy);

	return;
}

void get_deriv_y_E_FML(
	int myNx, int Ny, int Nz,
	double* Ex_re,
	double* Ez_re,
	double* ky,
	double* diffyEx_re,
	double* diffyEz_re
){

	// int for index
	int i,j,k;
	int myidx, i_myidx, myidx_T, myidx_0;
	double real, imag;

	// Memory allocation for transpose of the field data.
	double* diffyEx_T = fftw_alloc_real(myNx*Ny*Nz);
	double* diffyEz_T = fftw_alloc_real(myNx*Ny*Nz);
	fftw_complex* FFTyEx_T = fftw_alloc_complex(myNx*(Ny/2+1)*Nz);
	fftw_complex* FFTyEz_T = fftw_alloc_complex(myNx*(Ny/2+1)*Nz);

	// Set Plans for real FFT along y axis.
	int ranky = 1;
	int ny[1]  = {Ny};
	int howmanyy = myNx*Nz;
	int istridey = 1, ostridey = 1;
	int idisty = Ny, odisty = Ny/2+1;
	const int *inembedy = NULL, *onembedy = NULL;

	fftw_plan FFTy_Ex_FOR_plan = fftw_plan_many_dft_r2c(ranky, ny, howmanyy, diffyEx_T, inembedy, istridey, idisty, \
														FFTyEx_T, onembedy, ostridey, odisty, FFTW_ESTIMATE);

	fftw_plan FFTy_Ez_FOR_plan = fftw_plan_many_dft_r2c(ranky, ny, howmanyy, diffyEz_T, inembedy, istridey, idisty, \
														FFTyEz_T, onembedy, ostridey, odisty, FFTW_ESTIMATE);
	
	// Set Plans for inverse real FFT along y axis.
	int rankby = 1;
	int nby[1]  = {Ny};
	int howmanyby = myNx*Nz;
	int istrideby = 1, ostrideby = 1;
	int idistby = Ny/2+1, odistby = Ny;
	const int *inembedby = NULL, *onembedby = NULL;

	fftw_plan FFTy_Ex_BAK_plan = fftw_plan_many_dft_c2r(rankby, nby, howmanyby, FFTyEx_T, inembedby, istrideby, idistby, \
														diffyEx_T, onembedby, ostrideby, odistby, FFTW_ESTIMATE);

	fftw_plan FFTy_Ez_BAK_plan = fftw_plan_many_dft_c2r(rankby, nby, howmanyby, FFTyEz_T, inembedby, istrideby, idistby, \
														diffyEz_T, onembedby, ostrideby, odistby, FFTW_ESTIMATE);

	// Transpose y and z axis of the Ex and Ez to get y-derivatives of them.
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = k + j*Nz + i*Nz*Ny;
				myidx_T = j + k*Ny + i*Nz*Ny;

				diffyEx_T[myidx_T] = Ex_re[myidx];
				diffyEz_T[myidx_T] = Ez_re[myidx];

			}
		}
	}

	// Perform 1D rFFT along y-axis.
	fftw_execute(FFTy_Ex_FOR_plan);
	fftw_execute(FFTy_Ez_FOR_plan);

	// Multiply iky.
	for(i=0; i < myNx; i++){
		for(k=0; k < Nz; k++){
			for(j=0; j < (Ny/2+1); j++){

				myidx_T = j + k*(Ny/2+1) + i*(Ny/2+1)*Nz;

				real = FFTyEx_T[myidx_T][0];
				imag = FFTyEx_T[myidx_T][1];

				FFTyEx_T[myidx_T][0] = -ky[j] * imag;
				FFTyEx_T[myidx_T][1] =  ky[j] * real;

				real = FFTyEz_T[myidx_T][0];
				imag = FFTyEz_T[myidx_T][1];

				FFTyEz_T[myidx_T][0] = -ky[j] * imag;
				FFTyEz_T[myidx_T][1] =  ky[j] * real;

			}
		}
	}

	// Perform Inverse FFT.
	fftw_execute(FFTy_Ex_BAK_plan);
	fftw_execute(FFTy_Ez_BAK_plan);

	// Normalize the results of pseudo-spectral method. Get diffx, diffy and diffz of H fields.
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  )*Nz + (i  )*Nz*Ny;
				myidx_T = (j  ) + (k  )*Ny + (i  )*Nz*Ny;

				diffyEx_re[myidx] = diffyEx_T[myidx_T] / Ny;
				diffyEz_re[myidx] = diffyEz_T[myidx_T] / Ny;

			}
		}
	}

	fftw_destroy_plan(FFTy_Ex_FOR_plan);
	fftw_destroy_plan(FFTy_Ez_FOR_plan);
	fftw_destroy_plan(FFTy_Ex_BAK_plan);
	fftw_destroy_plan(FFTy_Ez_BAK_plan);

	fftw_free(FFTyEx_T);
	fftw_free(FFTyEz_T);
	fftw_free(diffyEx_T);
	fftw_free(diffyEz_T);

	return;
}


void get_deriv_x_E_FM0(
	int	myNx, int Ny, int Nz,
	double  dx,
	double* Ey_re,
	double* Ez_re,
	double* diffxEy_re,
	double* diffxEz_re,
	double* recvEylast_re,
	double* recvEzlast_re
){

	// Integers for index.
	int i,j,k, myidx, i_myidx, myidx_0;

	// Get x derivatives of E fields.
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  )*Nz + (i  )*Nz*Ny;
				i_myidx = (k  ) + (j  )*Nz + (i+1)*Nz*Ny;
				myidx_0 = (k  ) + (j  )*Nz + (0  )*Nz*Ny;

				if(i < (myNx-1)){

					diffxEy_re[myidx] = (Ey_re[i_myidx] - Ey_re[myidx]) / dx;
					diffxEz_re[myidx] = (Ez_re[i_myidx] - Ez_re[myidx]) / dx;

				} 

				else if (i == (myNx-1)){		

					diffxEy_re[myidx] = (recvEylast_re[myidx_0] - Ey_re[myidx]) / dx;
					diffxEz_re[myidx] = (recvEzlast_re[myidx_0] - Ez_re[myidx]) / dx;

				}

				else{printf("Something is wrong!\n");}
			}
		}
	}

	return;
}

void get_deriv_x_E_00L(
	int	myNx, int Ny, int Nz,
	double  dx,
	double* Ey_re,
	double* Ez_re,
	double* diffxEy_re,
	double* diffxEz_re
){

	// Integers for index.
	int i,j,k, myidx, i_myidx;

	// Get x derivatives of E fields.
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  )*Nz + (i  )*Nz*Ny;
				i_myidx = (k  ) + (j  )*Nz + (i+1)*Nz*Ny;

				if(i < (myNx-1)){

					diffxEy_re[myidx] = (Ey_re[i_myidx] - Ey_re[myidx]) / dx;
					diffxEz_re[myidx] = (Ez_re[i_myidx] - Ez_re[myidx]) / dx;

				} 
			}
		}
	}

	return;
}

// Update H field.
void updateH(
	int		MPIsize,	int MPIrank,
	int		myNx,		int		Ny,		int		Nz,
	double  dt,
	double *Hx_re,		
	double *Hy_re,	
	double *Hz_re,	
	double *mu_HEE,		double *mu_EHH,
	double *mcon_HEE,	double *mcon_EHH,
	double *diffxEy_re, 
	double *diffxEz_re, 
	double *diffyEx_re, 
	double *diffyEz_re, 
	double *diffzEx_re, 
	double *diffzEy_re
){
	/* MAIN UPDATE EQUATIONS */
	int i,j,k;
	int myidx;

	double CHx1, CHx2;
	double CHy1, CHy2;
	double CHz1, CHz2;

	// Update H field.
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CHx1 =	(2.*mu_HEE[myidx] - mcon_HEE[myidx]*dt) / (2.*mu_HEE[myidx] + mcon_HEE[myidx]*dt);
				CHy1 =	(2.*mu_EHH[myidx] - mcon_EHH[myidx]*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);
				CHz1 =	(2.*mu_EHH[myidx] - mcon_EHH[myidx]*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);

				CHx2 =	(-2*dt) / (2.*mu_HEE[myidx] + mcon_HEE[myidx]*dt);
				CHy2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);
				CHz2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);

				// Update Hx
				Hx_re[myidx] = CHx1 * Hx_re[myidx] + CHx2 * (diffyEz_re[myidx] - diffzEy_re[myidx]);

				if ( i < (myNx-1)){
					// Update Hy
					Hy_re[myidx] = CHy1 * Hy_re[myidx] + CHy2 * (diffzEx_re[myidx] - diffxEz_re[myidx]);

					// Update Hz
					Hz_re[myidx] = CHz1 * Hz_re[myidx] + CHz2 * (diffxEy_re[myidx] - diffyEx_re[myidx]);
				}
			}
		}
	}

	// First and Middle nodes update the last Hy and Hz which need MPI communication.
	if (MPIrank < (MPIsize-1)){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				myidx   = (k  ) + (j  ) * Nz + (myNx-1) * Nz * Ny;

				CHy1 =	(2.*mu_EHH[myidx] - mcon_EHH[myidx]*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);
				CHz1 =	(2.*mu_EHH[myidx] - mcon_EHH[myidx]*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);

				CHy2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);
				CHz2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);

				// Update Hy
				Hy_re[myidx] = CHy1 * Hy_re[myidx] + CHy2 * (diffzEx_re[myidx] - diffxEz_re[myidx]);

				// Update Hz
				Hz_re[myidx] = CHz1 * Hz_re[myidx] + CHz2 * (diffxEy_re[myidx] - diffyEx_re[myidx]);
			}
		}
	}

	return;
}

void get_deriv_z_H_FML(
	int myNx, int Ny, int Nz,
	double* Hx_re,
	double* Hy_re,
	double* kz,
	double* diffzHx_re,
	double* diffzHy_re
){

	// int for index
	int i,j,k;
	int myidx, myidx_i, myidx_T;
	double real, imag;

	fftw_complex *FFTzHx = fftw_alloc_complex(myNx*Ny*(Nz/2+1));
	fftw_complex *FFTzHy = fftw_alloc_complex(myNx*Ny*(Nz/2+1));

	// Set Plans for real FFT along z axis.
	int rankz = 1;
	int nz[1]  = {Nz};
	int howmanyz = myNx*Ny;
	int istridez = 1, ostridez = 1;
	int idistz = Nz, odistz = Nz/2+1;
	const int *inembedz = NULL, *onembedz = NULL;

	fftw_plan FFTz_Hx_FOR_plan = fftw_plan_many_dft_r2c(rankz, nz, howmanyz, Hx_re, inembedz, istridez, idistz, \
														FFTzHx, onembedz, ostridez, odistz, FFTW_ESTIMATE);

	fftw_plan FFTz_Hy_FOR_plan = fftw_plan_many_dft_r2c(rankz, nz, howmanyz, Hy_re, inembedz, istridez, idistz, \
														FFTzHy, onembedz, ostridez, odistz, FFTW_ESTIMATE);

	// Set Plans for inverse real FFT along z axis.
	int rankbz = 1;
	int nbz[1]  = {Nz};
	int howmanybz = myNx*Ny;
	int istridebz = 1, ostridebz = 1;
	int idistbz = Nz/2+1, odistbz = Nz;
	const int *inembedbz = NULL, *onembedbz = NULL;

	fftw_plan FFTz_Hx_BAK_plan = fftw_plan_many_dft_c2r(rankbz, nbz, howmanybz, FFTzHx, inembedbz, istridebz, idistbz, \
														diffzHx_re, onembedbz, ostridebz, odistbz, FFTW_ESTIMATE);

	fftw_plan FFTz_Hy_BAK_plan = fftw_plan_many_dft_c2r(rankbz, nbz, howmanybz, FFTzHy, inembedbz, istridebz, idistbz, \
														diffzHy_re, onembedbz, ostridebz, odistbz, FFTW_ESTIMATE);

	// Perform 1D FFT along z-axis.
	fftw_execute(FFTz_Hx_FOR_plan);
	fftw_execute(FFTz_Hy_FOR_plan);

	// Multiply ikz.
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < (Nz/2+1); k++){

				myidx = k + j*(Nz/2+1) + i*(Nz/2+1)*Ny;

				real = FFTzHx[myidx][0];
				imag = FFTzHx[myidx][1];

				FFTzHx[myidx][0] = -kz[k] * imag;
				FFTzHx[myidx][1] =  kz[k] * real;

				real = FFTzHy[myidx][0];
				imag = FFTzHy[myidx][1];

				FFTzHy[myidx][0] = -kz[k] * imag;
				FFTzHy[myidx][1] =  kz[k] * real;
				
			}
		}
	}

	fftw_execute(FFTz_Hx_BAK_plan);
	fftw_execute(FFTz_Hy_BAK_plan);

	// Normalize the results of pseudo-spectral method
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  )*Nz + (i  )*Nz*Ny;

				diffzHx_re[myidx] = diffzHx_re[myidx] / Nz;
				diffzHy_re[myidx] = diffzHy_re[myidx] / Nz;

			}
		}
	}

	// Destroy the plan and free the memory.
	fftw_destroy_plan(FFTz_Hx_FOR_plan);
	fftw_destroy_plan(FFTz_Hy_FOR_plan);
	fftw_destroy_plan(FFTz_Hx_BAK_plan);
	fftw_destroy_plan(FFTz_Hy_BAK_plan);
	fftw_free(FFTzHx);
	fftw_free(FFTzHy);

	return;
}

void get_deriv_y_H_FML(
	int myNx, int Ny, int Nz,
	double* Hx_re,
	double* Hz_re,
	double* ky,
	double* diffyHx_re,
	double* diffyHz_re
){

	// int for index
	int i,j,k;
	int myidx, myidx_i, myidx_T;
	double real, imag;

	// Memory allocation for transpose of the field data.
	double* diffyHx_T = fftw_alloc_real(myNx*Ny*Nz);
	double* diffyHz_T = fftw_alloc_real(myNx*Ny*Nz);
	fftw_complex *FFTyHx_T = fftw_alloc_complex(myNx*(Ny/2+1)*Nz);
	fftw_complex *FFTyHz_T = fftw_alloc_complex(myNx*(Ny/2+1)*Nz);

	// Set Plans for real FFT along y axis.
	int ranky = 1;
	int ny[1]  = {Ny};
	int howmanyy = myNx*Nz;
	int istridey = 1, ostridey = 1;
	int idisty = Ny, odisty = Ny/2+1;
	const int *inembedy = NULL, *onembedy = NULL;

	fftw_plan FFTy_Hx_FOR_plan = fftw_plan_many_dft_r2c(ranky, ny, howmanyy, diffyHx_T, inembedy, istridey, idisty, \
														FFTyHx_T, onembedy, ostridey, odisty, FFTW_ESTIMATE);

	fftw_plan FFTy_Hz_FOR_plan = fftw_plan_many_dft_r2c(ranky, ny, howmanyy, diffyHz_T, inembedy, istridey, idisty, \
														FFTyHz_T, onembedy, ostridey, odisty, FFTW_ESTIMATE);

	// Set Plans for inverse real FFT along y axis.
	int rankby = 1;
	int nby[1]  = {Ny};
	int howmanyby = myNx*Nz;
	int istrideby = 1, ostrideby = 1;
	int idistby = Ny/2+1, odistby = Ny;
	const int *inembedby = NULL, *onembedby = NULL;

	fftw_plan FFTy_Hx_BAK_plan = fftw_plan_many_dft_c2r(rankby, nby, howmanyby, FFTyHx_T, inembedby, istrideby, idistby, \
																	diffyHx_T, onembedby, ostrideby, odistby, FFTW_ESTIMATE);

	fftw_plan FFTy_Hz_BAK_plan = fftw_plan_many_dft_c2r(rankby, nby, howmanyby, FFTyHz_T, inembedby, istrideby, idistby, \
																	diffyHz_T, onembedby, ostrideby, odistby, FFTW_ESTIMATE);

	// Transpose y and z axis of the Hx and Hz to get y-derivatives of them.
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = k + j*Nz + i*Nz*Ny; // (x,y,z)
				myidx_T = j + k*Ny + i*Nz*Ny; // (x,z,y)

				diffyHx_T[myidx_T] = Hx_re[myidx];
				diffyHz_T[myidx_T] = Hz_re[myidx];

			}
		}
	}

	// Perform 1D FFT along z-axis.
	fftw_execute(FFTy_Hx_FOR_plan);
	fftw_execute(FFTy_Hz_FOR_plan);

	// Multiply iky.
	for(i=0; i < myNx; i++){
		for(k=0; k < Nz; k++){
			for(j=0; j < (Ny/2+1); j++){

				myidx_T = j + k*(Ny/2+1) + i*Nz*(Ny/2+1);

				real = FFTyHx_T[myidx_T][0];
				imag = FFTyHx_T[myidx_T][1];

				FFTyHx_T[myidx_T][0] = -ky[j] * imag;
				FFTyHx_T[myidx_T][1] =  ky[j] * real;

				real = FFTyHz_T[myidx_T][0];
				imag = FFTyHz_T[myidx_T][1];

				FFTyHz_T[myidx_T][0] = -ky[j] * imag;
				FFTyHz_T[myidx_T][1] =  ky[j] * real;

			}
		}
	}

	// Perform Inverse FFT.
	fftw_execute(FFTy_Hx_BAK_plan);
	fftw_execute(FFTy_Hz_BAK_plan);

	// Normalize the results of pseudo-spectral method. Get diffx, diffy, diffz of H field
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  )*Nz + (i  )*Nz*Ny;
				myidx_T = (j  ) + (k  )*Ny + (i  )*Nz*Ny;

				diffyHx_re[myidx] = diffyHx_T[myidx_T] / Ny;
				diffyHz_re[myidx] = diffyHz_T[myidx_T] / Ny;

			}
		}
	}

	fftw_destroy_plan(FFTy_Hx_FOR_plan);
	fftw_destroy_plan(FFTy_Hz_FOR_plan);
	fftw_destroy_plan(FFTy_Hx_BAK_plan);
	fftw_destroy_plan(FFTy_Hz_BAK_plan);
	fftw_free(FFTyHx_T);
	fftw_free(FFTyHz_T);
	free(diffyHx_T);
	free(diffyHz_T);

	return;
}

void get_deriv_x_H_F00(
	int myNx, int Ny, int Nz,
	double dx,
	double* Hy_re,
	double* Hz_re,
	double* diffxHy_re,
	double* diffxHz_re
){

	int i, j, k, myidx, myidx_i;

	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  )*Nz + (i  )*Nz*Ny;
				myidx_i = (k  ) + (j  )*Nz + (i-1)*Nz*Ny;

				if (i > 0){
					diffxHy_re[myidx] = (Hy_re[myidx] - Hy_re[myidx_i]) / dx;
					diffxHz_re[myidx] = (Hz_re[myidx] - Hz_re[myidx_i]) / dx;
				}
			}
		}
	}

	return;
}

void get_deriv_x_H_0ML(
	int myNx, int Ny, int Nz,
	double dx,
	double* Hy_re,
	double* Hz_re,
	double* diffxHy_re,
	double* diffxHz_re,
	double* recvHyfirst_re,
	double* recvHzfirst_re
){

	int i, j, k, myidx, myidx_i;

	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  )*Nz + (i  )*Nz*Ny;
				myidx_i = (k  ) + (j  )*Nz + (i-1)*Nz*Ny;

				if (i > 0){
					diffxHy_re[myidx] = (Hy_re[myidx] - Hy_re[myidx_i]) / dx;
					diffxHz_re[myidx] = (Hz_re[myidx] - Hz_re[myidx_i]) / dx;
				}
				else {
					diffxHy_re[myidx] = (Hy_re[myidx] - recvHyfirst_re[myidx]) / dx;
					diffxHz_re[myidx] = (Hz_re[myidx] - recvHzfirst_re[myidx]) / dx;
				}
			}
		}
	}

	return;
}

void updateE(
	int		MPIsize,	int MPIrank,
	int		myNx,		int		Ny,		int		Nz,	\
	double dt,										\
	double *Ex_re,	
	double *Ey_re,	
	double *Ez_re,	
	double *eps_HEE,	double *eps_EHH,			\
	double *econ_HEE,	double *econ_EHH,			\
	double *diffxHy_re, 
	double *diffxHz_re, 
	double *diffyHx_re, 
	double *diffyHz_re, 
	double *diffzHx_re, 
	double *diffzHy_re
){
	/* MAIN UPDATE EQUATIONS */
	int i,j,k;
	int myidx;

	double CEx1, CEx2;
	double CEy1, CEy2;
	double CEz1, CEz2;

	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CEx1 = (2.*eps_EHH[myidx] - econ_EHH[myidx]*dt) / (2.*eps_EHH[myidx] + econ_EHH[myidx]*dt);
				CEy1 = (2.*eps_HEE[myidx] - econ_HEE[myidx]*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);
				CEz1 = (2.*eps_HEE[myidx] - econ_HEE[myidx]*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);

				CEx2 =	(2.*dt) / (2.*eps_EHH[myidx] + econ_EHH[myidx]*dt);
				CEy2 =	(2.*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);
				CEz2 =	(2.*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);

				// PEC condition.
				if(eps_EHH[myidx] > 1e3){
					CEx1 = 0.;
					CEx2 = 0.;
				}

				if(eps_HEE[myidx] > 1e3){
					CEy1 = 0.;
					CEy2 = 0.;
					CEz1 = 0.;
					CEz2 = 0.;
				}

				// Update Ex.
				Ex_re[myidx] = CEx1 * Ex_re[myidx] + CEx2 * (diffyHz_re[myidx] - diffzHy_re[myidx]);

				if (i > 0) {
					// Update Ey.
					Ey_re[myidx] = CEy1 * Ey_re[myidx] + CEy2 * (diffzHx_re[myidx] - diffxHz_re[myidx]);

					// Update Ez.
					Ez_re[myidx] = CEz1 * Ez_re[myidx] + CEz2 * (diffxHy_re[myidx] - diffyHx_re[myidx]);
				}
			}		
		}
	}

	// Middle and last nodes update first Ey and Ez which need MPI communication.
	if (MPIrank > 0){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (0  ) * Nz * Ny;

				CEy1 = (2.*eps_HEE[myidx] - econ_HEE[myidx]*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);
				CEz1 = (2.*eps_HEE[myidx] - econ_HEE[myidx]*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);

				CEy2 =	(2.*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);
				CEz2 =	(2.*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);

				// PEC condition.
				if(eps_HEE[myidx] > 1e3){
					CEy1 = 0.;
					CEy2 = 0.;
					CEz1 = 0.;
					CEz2 = 0.;
				}

				// Update Ey.
				Ey_re[myidx] = CEy1 * Ey_re[myidx] + CEy2 * (diffzHx_re[myidx] - diffxHz_re[myidx]);

				// Update Ez.
				Ez_re[myidx] = CEz1 * Ez_re[myidx] + CEz2 * (diffxHy_re[myidx] - diffyHx_re[myidx]);
			}
		}
	}

	return;
}
