#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*
	Author: Donggun Lee
	Date : 18.01.22

	This script contains update equations to apply PBC.

	Total grid is disassembled along x-axis, which is the most significant index.
	Decomposed grid is distributed to each node and each nodes are connected through
	MPI communication. Thus, PBC along x-axis needs MPI communication between
	the first node and the last node. PBC along y and z-axis does not need MPI communication.

	LOG
	---
	18.01.19: yplus has been finished.
	18.01.22: update equations for PBC in core.c is detached to PBC.c

*/

/***********************************************************************************/
/******************************** FUNCTION DECLARATION *****************************/
/***********************************************************************************/

void my_rank_F( \
	int myNx, int Ny, int Nz, \
	double dt, double dx, double dy, double dz, \
	double *eps_Ex, double *eps_Ez, \
	double *econ_Ex, double *econ_Ez, \
	double *Ex_re, 
	double *Ez_re, 
	double *Hx_re, 
	double *Hy_re, 
	double *Hz_re, 
	double *diffxHy_re, 
	double *diffyHx_re, 
	double *diffyHz_re, 
	double *diffzHy_re
);


void my_rankML( \
	int myNx, int Ny, int Nz, \
	double dt, double dx, double dy, double dz, \
	double *eps_Ex, double *eps_Ez, \
	double *econ_Ex, double *econ_Ez, \
	double *recvHyfirst_re, 
	double *Ex_re, 
	double *Ez_re, 
	double *Hx_re, 
	double *Hy_re, 
	double *Hz_re, 
	double *diffxHy_re, 
	double *diffyHx_re, 
	double *diffyHz_re, 
	double *diffzHy_re
);	


void py_rankFM( \
	int myNx, int Ny, int Nz, \
	double dt, double dx, double dy, double dz, \
	double *mu_Hx, double *mu_Hz, \
	double *mcon_Hx, double *mcon_Hz, \
	double *recvEylast_re, 
	double *Hx_re, 
	double *Hz_re, 
	double *Ex_re, 
	double *Ey_re, 
	double *Ez_re, 
	double *diffxEy_re, 
	double *diffyEx_re, 
	double *diffyEz_re, 
	double *diffzEy_re
); 


void py_rank_L( \
	int myNx, int Ny, int Nz, \
	double dt, double dx, double dy, double dz, \
	double *mu_Hx, double *mu_Hz, \
	double *mcon_Hx, double *mcon_Hz, \
	double *Hx_re, 
	double *Hz_re, 
	double *Ex_re, 
	double *Ey_re, 
	double *Ez_re, 
	double *diffxEy_re, 
	double *diffyEx_re, 
	double *diffyEz_re, 
	double *diffzEy_re
);


void mz_rank_F( \
	int myNx, int Ny, int Nz, \
	double dt, double dx, double dy, double dz, \
	double *eps_Ex, double *eps_Ey, \
	double *econ_Ex, double *econ_Ey, \
	double *Ex_re, 
	double *Ey_re, 
	double *Hx_re, 
	double *Hy_re, 
	double *Hz_re, 
	double *diffxHz_re, 
	double *diffyHz_re, 
	double *diffzHx_re, 
	double *diffzHy_re
);


void mz_rankML( \
	int myNx, int Ny, int Nz,\
	double dt, double dx, double dy, double dz, \
	double *eps_Ex, double *eps_Ey, \
	double *econ_Ex, double *econ_Ey, \
	double *recvHzfirst_re, 
	double *Ex_re, 
	double *Ey_re, 
	double *Hx_re, 
	double *Hy_re, 
	double *Hz_re, 
	double *diffxHz_re, 
	double *diffyHz_re, 
	double *diffzHx_re, 
	double *diffzHy_re
);


void pz_rankFM(\
	int myNx, int Ny, int Nz,\
	double dt, double dx, double dy, double dz, \
	double *mu_Hx, double *mu_Hy, \
	double *mcon_Hx, double *mcon_Hy, \
	double *recvEzlast_re, 
	double *Hx_re, 
	double *Hy_re, 
	double *Ex_re, 
	double *Ey_re, 
	double *Ez_re, 
	double *diffxEz_re, 
	double *diffyEz_re, 
	double *diffzEx_re, 
	double *diffzEy_re
);


void pz_rank_L(\
	int myNx, int Ny, int Nz, \
	double dt, double dx, double dy, double dz, \
	double *mu_Hx, double *mu_Hy, \
	double *mcon_Hx, double *mcon_Hy, \
	double *Hx_re, 
	double *Hy_re, 
	double *Ex_re, 
	double *Ey_re, 
	double *Ez_re, 
	double *diffxEz_re, 
	double *diffyEz_re, 
	double *diffzEx_re, 
	double *diffzEy_re
);


/***********************************************************************************/
/******************* DECLARE FUNCTIONS TO APPLY PBC ON PML REGION ******************/
/***********************************************************************************/

void mxPML_myPBC( \
	int myNx, int Ny, int Nz, int npml, \
	double dt, \
	double *PMLkappax, double *PMLbx, double *PMLax, \
	double *eps_Ez, double *econ_Ez, \
	double *Ez_re, 
	double *diffxHy_re, 
	double *psi_ezx_m_re
);


void mxPML_pyPBC( \
	int myNx, int Ny, int Nz, int npml,\
	double dt, \
	double *PMLkappax, double *PMLbx, double *PMLax, \
	double *mu_Hz, double *mcon_Hz, \
	double *Hz_re, 
	double *diffxEy_re, 
	double *psi_hzx_m_re
);


void mxPML_mzPBC( \
	int myNx, int Ny, int Nz, int npml,\
	double dt, \
    double *PMLkappax, double *PMLbx,  double *PMLax, \
	double *eps_Ey, double *econ_Ey, \
	double *Ey_re, 
	double *diffxHz_re, 
	double *psi_eyx_m_re
);


void mxPML_pzPBC(\
	int myNx, int Ny, int Nz, int npml,\
	double dt, \
    double *PMLkappax, double *PMLbx,  double *PMLax, \
	double *mu_Hy, double *mcon_Hy, \
	double *Hy_re, 
	double *diffxEz_re, 
	double *psi_hyx_m_re
);


void pxPML_myPBC( \
	int myNx, int Ny, int Nz, int npml, \
	double dt, \
	double *PMLkappax, double *PMLbx, double *PMLax, \
	double *eps_Ez, double *econ_Ez, \
	double *Ez_re, 
	double *diffxHy_re, 
	double *psi_ezx_m_re
);


void pxPML_pyPBC( \
	int myNx, int Ny, int Nz, int npml,\
	double dt, \
	double *PMLkappax, double *PMLbx, double *PMLax, \
	double *mu_Hz, double *mcon_Hz, \
	double *Hz_re, 
	double *diffxEy_re, 
	double *psi_hzx_m_re
);


void pxPML_mzPBC( \
	int myNx, int Ny, int Nz, int npml,\
	double dt, \
    double *PMLkappax, double *PMLbx,  double *PMLax, \
	double *eps_Ey, double *econ_Ey, \
	double *Ey_re, 
	double *diffxHz_re, 
	double *psi_eyx_m_re
);


void pxPML_pzPBC(\
	int myNx, int Ny, int Nz, int npml,\
	double dt, \
    double *PMLkappax, double *PMLbx,  double *PMLax, \
	double *mu_Hy, double *mcon_Hy, \
	double *Hy_re, 
	double *diffxEz_re, 
	double *psi_hyx_m_re
);


/***********************************************************************************/
/******************************** FUNCTION DESCRIPTION *****************************/
/***********************************************************************************/

void my_rank_F(	\
	int myNx, int Ny, int Nz,	\
	double dt, double dx, double dy, double dz, \
	double *eps_Ex, double *eps_Ez, \
	double *econ_Ex, double *econ_Ez, \
	double *Ex_re, 
	double *Ez_re, 
	double *Hx_re, 
	double *Hy_re, 
	double *Hz_re, 
	double *diffxHy_re, 
	double *diffyHx_re, 
	double *diffyHz_re, 
	double *diffzHy_re
){

	// By the staggered nature of the Yee grid, a plane of Ex or Ez at j=0 cannot be updated.
	// We will update this plane with the plane of Hz or Hx at j=(Ny-1), respectively.
	// This procedure effectively makes the simulation space periodic along y-axis.

	int i,k;
	int first_idx, last_idx, first_idx_k, first_idx_i;

	double CEx1, CEx2;
	double CEz1, CEz2;

	// Update Ex at j=0 by using Hy at j=Ny-1
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dy, dz, Hy_re, Hz_re, Ex_re, eps_Ex, econ_Ex, diffyHz_re, diffzHy_re)	\
		private(i, k, first_idx, first_idx_k, last_idx, CEx1, CEx2)
	for(i=0; i < myNx; i++){
		for(k=1; k < Nz; k++){
	
			first_idx   = (k  ) + (0   ) * Nz + i * Nz * Ny;
			first_idx_k = (k-1) + (0   ) * Nz + i * Nz * Ny;
			last_idx    = (k  ) + (Ny-1) * Nz + i * Nz * Ny;

			CEx1 = (2.*eps_Ex[first_idx] - econ_Ex[first_idx]*dt) / (2.*eps_Ex[first_idx] + econ_Ex[first_idx]*dt);
			CEx2 = (2.*dt) / (2.*eps_Ex[first_idx] + econ_Ex[first_idx]*dt);

			// PEC condtion.
			if(eps_Ex[first_idx] > 1e3){
				CEx1 = 0.;
				CEx2 = 0.;
			}

			diffyHz_re[first_idx] = (Hz_re[first_idx] - Hz_re[last_idx]) / dy;

			diffzHy_re[first_idx] = (Hy_re[first_idx] - Hy_re[first_idx_k]) / dz;

			Ex_re[first_idx] = CEx1 * Ex_re[first_idx] + CEx2 * (diffyHz_re[first_idx] - diffzHy_re[first_idx]);

		}
	}

	// Update Ez at j=0 by using Hx at j=Ny-1
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dy, Hx_re, Hy_re, Ez_re, eps_Ez, econ_Ez, diffxHy_re, diffyHx_re)	\
		private(i, k, first_idx, first_idx_i, last_idx, CEz1, CEz2)
	for(i=1; i < myNx; i++){
		for(k=0; k < Nz; k++){
	
			first_idx   = (k  ) + (0   ) * Nz + (i  ) * Nz * Ny;
			first_idx_i = (k  ) + (0   ) * Nz + (i-1) * Nz * Ny;
			last_idx    = (k  ) + (Ny-1) * Nz + (i  ) * Nz * Ny;

			CEz1 = (2.*eps_Ez[first_idx] - econ_Ez[first_idx]*dt) / (2.*eps_Ez[first_idx] + econ_Ez[first_idx]*dt);
			CEz2 = (2.*dt) / (2.*eps_Ez[first_idx] + econ_Ez[first_idx]*dt);

			if(eps_Ez[first_idx] > 1e3){
				CEz1 = 0.;
				CEz2 = 0.;
			}

			diffxHy_re[first_idx] = (Hy_re[first_idx] - Hy_re[first_idx_i]) / dx;

			diffyHx_re[first_idx] = (Hx_re[first_idx] - Hx_re[last_idx]) / dy;
		
			Ez_re[first_idx] = CEz1 * Ez_re[first_idx] + CEz2 * (diffxHy_re[first_idx] - diffyHx_re[first_idx]);

		}
	}
			
	return;
}


void my_rankML( \
	int myNx, int Ny, int Nz, \
	double dt, double dx, double dy, double dz, \
	double *eps_Ex, double *eps_Ez, \
	double *econ_Ex, double *econ_Ez, \
	double *recvHyfirst_re, 
	double *Ex_re, 
	double *Ez_re, 
	double *Hx_re, 
	double *Hy_re, 
	double *Hz_re, 
	double *diffxHy_re, 
	double *diffyHx_re, 
	double *diffyHz_re, 
	double *diffzHy_re
){

	// By the staggered nature of the Yee grid, a plane of Ex or Ez at j=0 cannot be updated.
	// We will update this plane with the plane of Hz or Hx at j=(Ny-1), respectively.
	// This procedure effectively makes the simulation space periodic along y-axis.

	int i,k;
	int first_idx, last_idx, first_idx_k, first_idx_i;
	int first_ij_idx, last_j_idx;

	double CEx1, CEx2;
	double CEy1, CEy2;
	double CEz1, CEz2;

	// Update Ex at j=0 by using Hy at j=Ny-1
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dy, dz, Hy_re, Hz_re, Ex_re, eps_Ex, econ_Ex, diffyHz_re, diffzHy_re)	\
		private(i, k, first_idx, first_idx_k, last_idx, CEx1, CEx2)
	for(i=0; i < myNx; i++){
		for(k=1; k < Nz; k++){
	
			first_idx   = (k  ) + (0   ) * Nz + i * Nz * Ny;
			first_idx_k = (k-1) + (0   ) * Nz + i * Nz * Ny;
			last_idx    = (k  ) + (Ny-1) * Nz + i * Nz * Ny;

			CEx1 = (2.*eps_Ex[first_idx] - econ_Ex[first_idx]*dt) / (2.*eps_Ex[first_idx] + econ_Ex[first_idx]*dt);
			CEx2 = (2.*dt) / (2.*eps_Ex[first_idx] + econ_Ex[first_idx]*dt);

			// PEC condtion.
			if(eps_Ex[first_idx] > 1e3){
				CEx1 = 0.;
				CEx2 = 0.;
			}

			diffyHz_re[first_idx] = (Hz_re[first_idx] - Hz_re[last_idx]) / dy;

			diffzHy_re[first_idx] = (Hy_re[first_idx] - Hy_re[first_idx_k]) / dz;

			Ex_re[first_idx] = CEx1 * Ex_re[first_idx] + CEx2 * (diffyHz_re[first_idx] - diffzHy_re[first_idx]);

		}
	}

	// Update Ez at j=0 by using Hx at j=Ny-1
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dy, Hx_re, Hy_re, Ez_re, eps_Ez, econ_Ez, diffxHy_re, diffyHx_re)	\
		private(i, k, first_idx, first_idx_i, last_idx, CEz1, CEz2)
	for(i=1; i < myNx; i++){
		for(k=0; k < Nz; k++){
	
			first_idx   = (k  ) + (0   ) * Nz + (i  ) * Nz * Ny;
			first_idx_i = (k  ) + (0   ) * Nz + (i-1) * Nz * Ny;
			last_idx    = (k  ) + (Ny-1) * Nz + (i  ) * Nz * Ny;

			CEz1 = (2.*eps_Ez[first_idx] - econ_Ez[first_idx]*dt) / (2.*eps_Ez[first_idx] + econ_Ez[first_idx]*dt);
			CEz2 = (2.*dt) / (2.*eps_Ez[first_idx] + econ_Ez[first_idx]*dt);

			if(eps_Ez[first_idx] > 1e3){
				CEz1 = 0.;
				CEz2 = 0.;
			}

			diffxHy_re[first_idx] = (Hy_re[first_idx] - Hy_re[first_idx_i]) / dx;

			diffyHx_re[first_idx] = (Hx_re[first_idx] - Hx_re[last_idx]) / dy;
		
			Ez_re[first_idx] = CEz1 * Ez_re[first_idx] + CEz2 * (diffxHy_re[first_idx] - diffyHx_re[first_idx]);

		}
	}

	// Update Ez at i=0 and j=0 by using Hx at j=(Ny-1) and Hy from previous rank.
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dy, Hx_re, Hy_re, recvHyfirst_re, Ez_re, eps_Ez, econ_Ez, diffxHy_re, diffyHx_re)	\
		private(k, first_ij_idx, last_j_idx, CEz1, CEz2)
	for(k=0; k < Nz; k++){
		
		first_ij_idx = k + (0   ) * Nz + (0) * Nz * Ny;
		last_j_idx   = k + (Ny-1) * Nz + (0) * Nz * Ny;

		CEz1 = (2.*eps_Ez[first_ij_idx] - econ_Ez[first_ij_idx]*dt) / (2.*eps_Ez[first_ij_idx] + econ_Ez[first_ij_idx]*dt);
		CEz2 = (2.*dt) / (2.*eps_Ez[first_ij_idx] + econ_Ez[first_ij_idx]*dt);

		if(eps_Ez[first_ij_idx] > 1e3){
			CEz1 = 0.;
			CEz2 = 0.;
		}

		diffxHy_re[first_ij_idx] = (Hy_re[first_ij_idx] - recvHyfirst_re[first_ij_idx]) / dx;

		diffyHx_re[first_ij_idx] = (Hx_re[first_ij_idx] - Hx_re[last_j_idx]) / dy;

		Ez_re[first_ij_idx] = CEz1 * Ez_re[first_ij_idx] + CEz2 * (diffxHy_re[first_ij_idx] - diffyHx_re[first_ij_idx]);
	}

	return;
}

void py_rankFM( \
	int myNx, int Ny, int Nz, \
	double dt, double dx, double dy, double dz, \
	double *mu_Hx, double *mu_Hz, \
	double *mcon_Hx, double *mcon_Hz, \
	double *recvEylast_re, 
	double *Hx_re, 
	double *Hz_re, 
	double *Ex_re, 
	double *Ey_re, 
	double *Ez_re, 
	double *diffxEy_re, 
	double *diffyEx_re, 
	double *diffyEz_re, 
	double *diffzEy_re
){

	int i,k;
	int first_idx, k_last_idx, i_last_idx, last_idx;
	int last_ij_idx, first_i_idx, first_j_idx;

	double CHx1, CHx2;
	double CHz1, CHz2;

	// Update Hx at j=(Ny-1) by using Ez at j=0
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dy, dz, Ey_re, Ez_re, Hx_re, mu_Hx, mcon_Hx, diffyEz_re, diffzEy_re)	\
		private(i, k, first_idx, last_idx, k_last_idx, CHx1, CHx2)
	for(i=0; i < myNx; i++){
		for(k=0; k < (Nz-1); k++){
	
			first_idx  = (k  ) + (0   ) * Nz + (i  ) * Nz * Ny;
			last_idx   = (k  ) + (Ny-1) * Nz + (i  ) * Nz * Ny;
			k_last_idx = (k+1) + (Ny-1) * Nz + (i  ) * Nz * Ny;

			CHx1 = (2.*mu_Hx[last_idx] - mcon_Hx[last_idx]*dt) / (2.*mu_Hx[last_idx] + mcon_Hx[last_idx]*dt);
			CHx2 = (-2*dt) / (2.*mu_Hx[last_idx] + mcon_Hx[last_idx]*dt);

			diffyEz_re[last_idx] = (Ez_re[first_idx] - Ez_re[last_idx]) / dy;

			diffzEy_re[last_idx] = (Ey_re[k_last_idx] - Ey_re[last_idx]) / dz;

			Hx_re[last_idx] = CHx1 * Hx_re[last_idx] + CHx2 * (diffyEz_re[last_idx] - diffzEy_re[last_idx]);

		}
	}

	// Update Hz at j=(Ny-1) by using Ex at j=0
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dy, Ex_re, Ey_re, Hz_re, mu_Hz, mcon_Hz, diffxEy_re, diffyEx_re)	\
		private(i, k, first_idx, last_idx, i_last_idx, CHz1, CHz2)
	for(i=0; i < (myNx-1); i++){
		for(k=0; k < Nz; k++){
	
			first_idx  = (k  ) + (0   ) * Nz + (i  ) * Nz * Ny;
			last_idx   = (k  ) + (Ny-1) * Nz + (i  ) * Nz * Ny;
			i_last_idx = (k  ) + (Ny-1) * Nz + (i+1) * Nz * Ny;

			CHz1 = (2.*mu_Hz[last_idx] - mcon_Hz[last_idx]*dt) / (2.*mu_Hz[last_idx] + mcon_Hz[last_idx]*dt);
			CHz2 = (-2*dt) / (2.*mu_Hz[last_idx] + mcon_Hz[last_idx]*dt);

			diffxEy_re[last_idx] = (Ey_re[i_last_idx] - Ey_re[last_idx]) / dx;

			diffyEx_re[last_idx] = (Ex_re[first_idx] - Ex_re[last_idx]) / dy;
		
			Hz_re[last_idx] = CHz1 * Hz_re[last_idx] + CHz2 * (diffxEy_re[last_idx] - diffyEx_re[last_idx]);

		}
	}

	// Update Hz at j=(Ny-1) and i=(myNx-1) by using Ex at j=0 and Ey from next rank.
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dy, Ex_re, Ey_re, recvEylast_re, Hz_re, mu_Hz, mcon_Hz, diffxEy_re, diffyEx_re)	\
		private(k, last_ij_idx, first_i_idx, first_j_idx, CHz1, CHz2)
	for(k=0; k < Nz; k++){

		last_ij_idx = k + (Ny-1) * Nz + (myNx-1) * Nz * Ny;
		first_i_idx = k + (Ny-1) * Nz + (0     ) * Nz * Ny;
		first_j_idx = k + (0   ) * Nz + (myNx-1) * Nz * Ny;

		CHz1 =	(2.*mu_Hz[last_ij_idx] - mcon_Hz[last_ij_idx]*dt) / (2.*mu_Hz[last_ij_idx] + mcon_Hz[last_ij_idx]*dt);
		CHz2 =	(-2*dt) / (2.*mu_Hz[last_ij_idx] + mcon_Hz[last_ij_idx]*dt);

		diffxEy_re[last_ij_idx] = (recvEylast_re[first_i_idx] - Ey_re[last_ij_idx]) / dx;

		diffyEx_re[last_ij_idx] = (Ex_re[first_j_idx] - Ex_re[last_ij_idx]) / dy;

		Hz_re[last_ij_idx] = CHz1 * Hz_re[last_ij_idx] + CHz2 * (diffxEy_re[last_ij_idx] - diffyEx_re[last_ij_idx]);

	}

	return;
}


void py_rank_L( \
	int myNx, int Ny, int Nz, \
	double dt, double dx, double dy, double dz, \
	double *mu_Hx, double *mu_Hz, \
	double *mcon_Hx, double *mcon_Hz, \
	double *Hx_re, 
	double *Hz_re, 
	double *Ex_re, 
	double *Ey_re, 
	double *Ez_re, 
	double *diffxEy_re, 
	double *diffyEx_re, 
	double *diffyEz_re, 
	double *diffzEy_re
){

   //By the Yee grid, a plane of Hx and Hz at j=(Ny-1) cannot be updated.
   //We will update this plane with the Ez- and Ex-plane at j=0, respectively.

	int i,k;
	int first_idx, k_last_idx, i_last_idx, last_idx;

	double CHx1, CHx2;
	double CHz1, CHz2;

	// Update Hx at j=(Ny-1) by using Ez at j=0
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dy, dz, Ey_re, Ez_re, Hx_re, mu_Hx, mcon_Hx, diffyEz_re, diffzEy_re)	\
		private(i, k, first_idx, last_idx, k_last_idx, CHx1, CHx2)
	for(i=0; i < myNx; i++){
		for(k=0; k < (Nz-1); k++){
	
			first_idx  = (k  ) + (0   ) * Nz + (i  ) * Nz * Ny;
			last_idx   = (k  ) + (Ny-1) * Nz + (i  ) * Nz * Ny;
			k_last_idx = (k+1) + (Ny-1) * Nz + (i  ) * Nz * Ny;

			CHx1 = (2.*mu_Hx[last_idx] - mcon_Hx[last_idx]*dt) / (2.*mu_Hx[last_idx] + mcon_Hx[last_idx]*dt);
			CHx2 = (-2*dt) / (2.*mu_Hx[last_idx] + mcon_Hx[last_idx]*dt);

			diffyEz_re[last_idx] = (Ez_re[first_idx] - Ez_re[last_idx]) / dy;

			diffzEy_re[last_idx] = (Ey_re[k_last_idx] - Ey_re[last_idx]) / dz;

			Hx_re[last_idx] = CHx1 * Hx_re[last_idx] + CHx2 * (diffyEz_re[last_idx] - diffzEy_re[last_idx]);

		}
	}

	// Update Hz at j=(Ny-1) by using Ex at j=0
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dy, Ex_re, Ey_re, Hz_re, mu_Hz, mcon_Hz, diffxEy_re, diffyEx_re)	\
		private(i, k, first_idx, last_idx, i_last_idx, CHz1, CHz2)
	for(i=0; i < (myNx-1); i++){
		for(k=0; k < Nz; k++){
	
			first_idx  = (k  ) + (0   ) * Nz + (i  ) * Nz * Ny;
			last_idx   = (k  ) + (Ny-1) * Nz + (i  ) * Nz * Ny;
			i_last_idx = (k  ) + (Ny-1) * Nz + (i+1) * Nz * Ny;

			CHz1 = (2.*mu_Hz[last_idx] - mcon_Hz[last_idx]*dt) / (2.*mu_Hz[last_idx] + mcon_Hz[last_idx]*dt);
			CHz2 = (-2*dt) / (2.*mu_Hz[last_idx] + mcon_Hz[last_idx]*dt);

			diffxEy_re[last_idx] = (Ey_re[i_last_idx] - Ey_re[last_idx]) / dx;

			diffyEx_re[last_idx] = (Ex_re[first_idx] - Ex_re[last_idx]) / dy;
		
			Hz_re[last_idx] = CHz1 * Hz_re[last_idx] + CHz2 * (diffxEy_re[last_idx] - diffyEx_re[last_idx]);

		}
	}
		
	return;
}


void mz_rank_F( \
	int myNx, int Ny, int Nz,
	double dt, double dx, double dy, double dz, \
	double *eps_Ex, double *eps_Ey, \
	double *econ_Ex, double *econ_Ey, \
	double *Ex_re, 
	double *Ey_re, 
	double *Hx_re, 
	double *Hy_re, 
	double *Hz_re, 
	double *diffxHz_re, 
	double *diffyHz_re, 
	double *diffzHx_re, 
	double *diffzHy_re
){

	// By the staggered nature of the Yee grid, a plane of Ex or Ey at j=0 cannot be updated.
	// We will update this plane with the plane of Hz or Hx at j=(Ny-1), respectively.
	// This procedure effectively makes the simulation space periodic along y-axis.

	int i,j,k;
	int first_idx, first_idx_i, first_idx_j, last_idx;

	double CEx1, CEx2;
	double CEy1, CEy2;

	// Update Ex at k=0
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dy, dz, Hy_re, Hz_re, Ex_re, eps_Ex, econ_Ex, diffyHz_re, diffzHy_re)	\
		private(i, j, first_idx, first_idx_j, last_idx, CEx1, CEx2)
	for(i=0; i < myNx; i++){
		for(j=1; j < Ny; j++){

			first_idx   = (0   ) + (j  ) * Nz + (i  ) * Nz * Ny;
			first_idx_j = (0   ) + (j-1) * Nz + (i  ) * Nz * Ny;
			last_idx    = (Nz-1) + (j  ) * Nz + (i  ) * Nz * Ny;
			
			CEx1 = (2.*eps_Ex[first_idx] - econ_Ex[first_idx]*dt) / (2.*eps_Ex[first_idx] + econ_Ex[first_idx]*dt);
			CEx2 = (2.*dt) / (2.*eps_Ex[first_idx] + econ_Ex[first_idx]*dt);

			// PEC condtion.
			if(eps_Ex[first_idx] > 1e3){
				CEx1 = 0.;
				CEx2 = 0.;
			}

			diffyHz_re[first_idx] = (Hz_re[first_idx] - Hz_re[first_idx_j]) / dy;	

			diffzHy_re[first_idx] = (Hy_re[first_idx] - Hy_re[last_idx]) / dz;

			Ex_re[first_idx] = CEx1 * Ex_re[first_idx] + CEx2 * (diffyHz_re[first_idx] - diffzHy_re[first_idx]);

		}
	}

	// Update Ey at k=0
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dz, Hx_re, Hz_re, Ey_re, eps_Ey, econ_Ey, diffxHz_re, diffzHx_re)	\
		private(i, j, first_idx, first_idx_i, last_idx, CEy1, CEy2)
	for(i=1; i < myNx; i++){
		for(j=0; j < Ny; j++){

			first_idx   = (0   ) + (j  ) * Nz + (i  ) * Nz * Ny;
			first_idx_i = (0   ) + (j  ) * Nz + (i-1) * Nz * Ny;
			last_idx    = (Nz-1) + (j  ) * Nz + (i  ) * Nz * Ny;
			
			CEy1 = (2.*eps_Ey[first_idx] - econ_Ey[first_idx]*dt) / (2.*eps_Ey[first_idx] + econ_Ey[first_idx]*dt);
			CEy2 = (2.*dt) / (2.*eps_Ey[first_idx] + econ_Ey[first_idx]*dt);

			if(eps_Ey[first_idx] > 1e3){
				CEy1 = 0.;
				CEy2 = 0.;
			}

			diffxHz_re[first_idx] = (Hz_re[first_idx] - Hz_re[first_idx_i]) / dx;	

			diffzHx_re[first_idx] = (Hx_re[first_idx] - Hx_re[last_idx]) / dz;

			Ey_re[first_idx] = CEy1 * Ey_re[first_idx] + CEy2 * (diffzHx_re[first_idx] - diffxHz_re[first_idx]);

		}
	}

	return;
}

void mz_rankML( \
	int myNx, int Ny, int Nz,
	double dt, double dx, double dy, double dz, \
	double *eps_Ex, double *eps_Ey, \
	double *econ_Ex, double *econ_Ey, \
	double *recvHzfirst_re, 
	double *Ex_re, 
	double *Ey_re, 
	double *Hx_re, 
	double *Hy_re, 
	double *Hz_re, 
	double *diffxHz_re, 
	double *diffyHz_re, 
	double *diffzHx_re, 
	double *diffzHy_re
){

	int i,j,k;
	int first_idx, first_idx_i, first_idx_j, last_idx;
	int first_ik_idx, last_k_idx;

	double CEx1, CEx2;
	double CEy1, CEy2;

	// Update Ex at k=0
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dy, dz, Hy_re, Hz_re, Ex_re, eps_Ex, econ_Ex, diffyHz_re, diffzHy_re)	\
		private(i, j, first_idx, first_idx_j, last_idx, CEx1, CEx2)
	for(i=0; i < myNx; i++){
		for(j=1; j < Ny; j++){

			first_idx   = (0   ) + (j  ) * Nz + (i  ) * Nz * Ny;
			first_idx_j = (0   ) + (j-1) * Nz + (i  ) * Nz * Ny;
			last_idx    = (Nz-1) + (j  ) * Nz + (i  ) * Nz * Ny;
			
			CEx1 = (2.*eps_Ex[first_idx] - econ_Ex[first_idx]*dt) / (2.*eps_Ex[first_idx] + econ_Ex[first_idx]*dt);
			CEx2 = (2.*dt) / (2.*eps_Ex[first_idx] + econ_Ex[first_idx]*dt);

			// PEC condtion.
			if(eps_Ex[first_idx] > 1e3){
				CEx1 = 0.;
				CEx2 = 0.;
			}

			diffyHz_re[first_idx] = (Hz_re[first_idx] - Hz_re[first_idx_j]) / dy;	

			diffzHy_re[first_idx] = (Hy_re[first_idx] - Hy_re[last_idx]) / dz;

			Ex_re[first_idx] = CEx1 * Ex_re[first_idx] + CEx2 * (diffyHz_re[first_idx] - diffzHy_re[first_idx]);

		}
	}

	// Update Ey at k=0
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dz, Hx_re, Hz_re, Ey_re, eps_Ey, econ_Ey, diffxHz_re, diffzHx_re)	\
		private(i, j, first_idx, first_idx_i, last_idx, CEy1, CEy2)
	for(i=1; i < myNx; i++){
		for(j=0; j < Ny; j++){

			first_idx   = (0   ) + (j  ) * Nz + (i  ) * Nz * Ny;
			first_idx_i = (0   ) + (j  ) * Nz + (i-1) * Nz * Ny;
			last_idx    = (Nz-1) + (j  ) * Nz + (i  ) * Nz * Ny;
			
			CEy1 = (2.*eps_Ey[first_idx] - econ_Ey[first_idx]*dt) / (2.*eps_Ey[first_idx] + econ_Ey[first_idx]*dt);
			CEy2 = (2.*dt) / (2.*eps_Ey[first_idx] + econ_Ey[first_idx]*dt);

			if(eps_Ey[first_idx] > 1e3){
				CEy1 = 0.;
				CEy2 = 0.;
			}

			diffxHz_re[first_idx] = (Hz_re[first_idx] - Hz_re[first_idx_i]) / dx;	

			diffzHx_re[first_idx] = (Hx_re[first_idx] - Hx_re[last_idx]) / dz;

			Ey_re[first_idx] = CEy1 * Ey_re[first_idx] + CEy2 * (diffzHx_re[first_idx] - diffxHz_re[first_idx]);

		}
	}

	// Update Ey at i=0 and k=0 by using Hx at k=(Nz-1) and Hz from previous rank.
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dz, Hx_re, Hz_re, recvHzfirst_re, Ey_re, eps_Ey, econ_Ey, diffxHz_re, diffzHx_re)	\
		private(j, first_ik_idx, last_k_idx, CEy1, CEy2)
	for(j=0; j < Ny; j++){
		
		first_ik_idx = (0   ) + j * Nz + (0  ) * Nz * Ny;
		last_k_idx   = (Nz-1) + j * Nz + (0  ) * Nz * Ny;

		CEy1 = (2.*eps_Ey[first_ik_idx] - econ_Ey[first_ik_idx]*dt) / (2.*eps_Ey[first_ik_idx] + econ_Ey[first_ik_idx]*dt);
		CEy2 = (2.*dt) / (2.*eps_Ey[first_ik_idx] + econ_Ey[first_ik_idx]*dt);

		if(eps_Ey[first_ik_idx] > 1e3){
			CEy1 = 0.;
			CEy2 = 0.;
		}

		diffxHz_re[first_ik_idx] = (Hz_re[first_ik_idx] - recvHzfirst_re[first_ik_idx]) / dx;

		diffzHx_re[first_ik_idx] = (Hx_re[first_ik_idx] - Hx_re[last_k_idx]) / dz;

		Ey_re[first_ik_idx] = CEy1 * Ey_re[first_ik_idx] + CEy2 * (diffzHx_re[first_ik_idx] - diffxHz_re[first_ik_idx]);

	}

	return;
}


void pz_rankFM(\
	int myNx, int Ny, int Nz,
	double dt, double dx, double dy, double dz, \
	double *mu_Hx, double *mu_Hy, \
	double *mcon_Hx, double *mcon_Hy, \
	double *recvEzlast_re, 
	double *Hx_re, 
	double *Hy_re, 
	double *Ex_re, 
	double *Ey_re, 
	double *Ez_re, 
	double *diffxEz_re, 
	double *diffyEz_re, 
	double *diffzEx_re, 
	double *diffzEy_re
){

	// By the Yee grid, a plane of Hx or Hy at k=(Nz-1) cannot be updated.
	// We will update this plane with the plane of Ey or Ex at k=0, respectively.

	int i,j;
	int last_idx, first_idx, i_last_idx, j_last_idx;
	int last_ik_idx, first_i_idx, first_k_idx;

	double CHx1, CHx2;
	double CHy1, CHy2;

	// Update Hx at k=(Nz-1)
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dy, dz, Ey_re, Ez_re, Hx_re, mu_Hx, mcon_Hx, diffyEz_re, diffzEy_re)	\
		private(i, j, last_idx, j_last_idx, first_idx, CHx1, CHx2)
	for(i=0; i < myNx; i++){
		for(j=0; j < (Ny-1); j++){

			last_idx   = (Nz-1) + (j  ) * Nz + (i  ) * Nz * Ny;
			j_last_idx = (Nz-1) + (j+1) * Nz + (i  ) * Nz * Ny;
			first_idx  = (0   ) + (j  ) * Nz + (i  ) * Nz * Ny;

			CHx1 =	(2.*mu_Hx[last_idx] - mcon_Hx[last_idx]*dt) / (2.*mu_Hx[last_idx] + mcon_Hx[last_idx]*dt);
			CHx2 =	(-2*dt) / (2.*mu_Hx[last_idx] + mcon_Hx[last_idx]*dt);

			diffyEz_re[last_idx] = (Ez_re[j_last_idx] - Ez_re[last_idx]) / dy;	

			diffzEy_re[last_idx] = (Ey_re[first_idx] - Ey_re[last_idx]) / dz;

			Hx_re[last_idx] = CHx1 * Hx_re[last_idx] + CHx2 * (diffyEz_re[last_idx] - diffzEy_re[last_idx]);

		}
	}

	// Update Hy at k=(Nz-1)
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dz, Ex_re, Ez_re, Hy_re, mu_Hy, mcon_Hy, diffxEz_re, diffzEx_re)	\
		private(i, j, last_idx, i_last_idx, first_idx, CHy1, CHy2)
	for(i=0; i < (myNx-1); i++){
		for(j=0; j < Ny; j++){

			last_idx   = (Nz-1) + (j  ) * Nz + (i  ) * Nz * Ny;
			i_last_idx = (Nz-1) + (j  ) * Nz + (i+1) * Nz * Ny;
			first_idx  = (0   ) + (j  ) * Nz + (i  ) * Nz * Ny;
			
			CHy1 =	(2.*mu_Hy[last_idx] - mcon_Hy[last_idx]*dt) / (2.*mu_Hy[last_idx] + mcon_Hy[last_idx]*dt);
			CHy2 =	(-2*dt) / (2.*mu_Hy[last_idx] + mcon_Hy[last_idx]*dt);

			diffxEz_re[last_idx] = (Ez_re[i_last_idx] - Ez_re[last_idx]) / dx;	

			diffzEx_re[last_idx] = (Ex_re[first_idx] - Ex_re[last_idx]) / dz;

			Hy_re[last_idx] = CHy1 * Hy_re[last_idx] + CHy2 * (diffzEx_re[last_idx] - diffxEz_re[last_idx]);

		}
	}

	// Update Hy at k=(Nz-1) by using Ex at k=0 and Ez from next rank.
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dz, Ex_re, Ez_re, recvEzlast_re, Hy_re, mu_Hy, mcon_Hy, diffxEz_re, diffzEx_re)	\
		private(j, last_ik_idx, first_i_idx, first_k_idx, CHy1, CHy2)
	for(j=0; j < Ny; j++){
		
		last_ik_idx = (Nz-1) + j * Nz + (myNx-1) * Nz * Ny;
		first_i_idx = (Nz-1) + j * Nz + (0     ) * Nz * Ny;
		first_k_idx = (0   ) + j * Nz + (myNx-1) * Nz * Ny;

		// i = myNx-1, k=Nz-1, j = 0~Ny-1
		CHy1 =	(2.*mu_Hy[last_ik_idx] - mcon_Hy[last_ik_idx]*dt) / (2.*mu_Hy[last_ik_idx] + mcon_Hy[last_ik_idx]*dt);
		CHy2 =	(-2*dt) / (2.*mu_Hy[last_ik_idx] + mcon_Hy[last_ik_idx]*dt);

		diffxEz_re[last_ik_idx] = (recvEzlast_re[first_i_idx] - Ez_re[last_ik_idx]) / dx;

		diffzEx_re[last_ik_idx] = (Ex_re[first_k_idx] - Ex_re[last_ik_idx]) / dz;

		Hy_re[last_ik_idx] = CHy1 * Hy_re[last_ik_idx] + CHy2 * (diffzEx_re[last_ik_idx] - diffxEz_re[last_ik_idx]);

	}

	return;
}


void pz_rank_L(\
	int myNx, int Ny, int Nz,
	double dt, double dx, double dy, double dz, \
	double *mu_Hx, double *mu_Hy, \
	double *mcon_Hx, double *mcon_Hy, \
	double *Hx_re, 
	double *Hy_re, 
	double *Ex_re, 
	double *Ey_re, 
	double *Ez_re, 
	double *diffxEz_re, 
	double *diffyEz_re, 
	double *diffzEx_re, 
	double *diffzEy_re
){

	int i,j,k;
	int last_idx, first_idx, i_last_idx, j_last_idx;

	double CHx1, CHx2;
	double CHy1, CHy2;

	// Update Hx at k=(Nz-1)
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dy, dz, Ey_re, Ez_re, Hx_re, mu_Hx, mcon_Hx, diffyEz_re, diffzEy_re)	\
		private(i, j, last_idx, j_last_idx, first_idx, CHx1, CHx2)
	for(i=0; i < myNx; i++){
		for(j=0; j < (Ny-1); j++){

			last_idx   = (Nz-1) + (j  ) * Nz + (i  ) * Nz * Ny;
			j_last_idx = (Nz-1) + (j+1) * Nz + (i  ) * Nz * Ny;
			first_idx  = (0   ) + (j  ) * Nz + (i  ) * Nz * Ny;
			
			CHx1 =	(2.*mu_Hx[last_idx] - mcon_Hx[last_idx]*dt) / (2.*mu_Hx[last_idx] + mcon_Hx[last_idx]*dt);
			CHx2 =	(-2*dt) / (2.*mu_Hx[last_idx] + mcon_Hx[last_idx]*dt);

			diffyEz_re[last_idx] = (Ez_re[j_last_idx] - Ez_re[last_idx]) / dy;	

			diffzEy_re[last_idx] = (Ey_re[first_idx] - Ey_re[last_idx]) / dz;

			Hx_re[last_idx] = CHx1 * Hx_re[last_idx] + CHx2 * (diffyEz_re[last_idx] - diffzEy_re[last_idx]);

		}
	}

	// Update Hy at k=(Nz-1)
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dz, Ex_re, Ez_re, Hy_re, mu_Hy, mcon_Hy, diffxEz_re, diffzEx_re)	\
		private(i, j, last_idx, i_last_idx, first_idx, CHy1, CHy2)
	for(i=0; i < (myNx-1); i++){
		for(j=0; j < Ny; j++){

			last_idx   = (Nz-1) + (j  ) * Nz + (i  ) * Nz * Ny;
			i_last_idx = (Nz-1) + (j  ) * Nz + (i+1) * Nz * Ny;
			first_idx  = (0   ) + (j  ) * Nz + (i  ) * Nz * Ny;
			
			CHy1 =	(2.*mu_Hy[last_idx] - mcon_Hy[last_idx]*dt) / (2.*mu_Hy[last_idx] + mcon_Hy[last_idx]*dt);
			CHy2 =	(-2*dt) / (2.*mu_Hy[last_idx] + mcon_Hy[last_idx]*dt);

			diffxEz_re[last_idx] = (Ez_re[i_last_idx] - Ez_re[last_idx]) / dx;	

			diffzEx_re[last_idx] = (Ex_re[first_idx] - Ex_re[last_idx]) / dz;

			Hy_re[last_idx] = CHy1 * Hy_re[last_idx] + CHy2 * (diffzEx_re[last_idx] - diffxEz_re[last_idx]);

		}
	}

	return;
}


/***********************************************************************************/
/******************* DESCRIPT FUNCTIONS TO APPLY PBC ON PML REGION *****************/
/***********************************************************************************/


void mxPML_myPBC( \
	int myNx, int Ny, int Nz, int npml, \
	double dt, \
	double *PMLkappax, double *PMLbx, double *PMLax, \
	double *eps_Ez, double *econ_Ez, \
	double *Ez_re, 
	double *diffxHy_re, 
	double *psi_ezx_m_re
){

	// By the staggered nature of the Yee grid, a plane of Ex or Ez at j=0 cannot be updated.
	// We will update this plane with the plane of Hz or Hx at j=(Ny-1), respectively.
	// This procedure effectively makes the simulation space periodic along y-axis.

	int i,k;
	int odd;
	int psiidx;
	int first_idx;

	double CEz2;

	// Update Ez at j=0 by using Hx at j=Ny-1
	#pragma omp parallel for \
		shared(	npml, Ny, Nz, dt, \
				eps_Ez, econ_Ez, Ez_re, diffxHy_re, PMLkappax, PMLbx, PMLax, psi_ezx_m_re)\
		private(i, k, odd, psiidx, first_idx, CEz2)
	for(i=1; i < npml; i++){
		for(k=0; k < Nz; k++){
	
			odd        = (2*npml) - (2*i+1);
			psiidx     = (k  ) + (0   ) * Nz + (i  ) * Nz * Ny;
			first_idx  = (k  ) + (0   ) * Nz + (i  ) * Nz * Ny;

			CEz2 = (2.*dt) / (2.*eps_Ez[first_idx] + econ_Ez[first_idx]*dt);

			if(eps_Ez[first_idx] > 1e3){
				CEz2 = 0.;
			}

			psi_ezx_m_re[psiidx] = (PMLbx[odd] * psi_ezx_m_re[psiidx]) + (PMLax[odd] * diffxHy_re[first_idx]);

			Ez_re[first_idx] += CEz2 * (+(1./PMLkappax[odd] - 1.) * diffxHy_re[first_idx] + psi_ezx_m_re[psiidx]);

		}
	}

	return;
};


void mxPML_pyPBC( \
	int myNx, int Ny, int Nz, int npml,\
	double dt, \
	double *PMLkappax, double *PMLbx, double *PMLax, \
	double *mu_Hz, double *mcon_Hz, \
	double *Hz_re, 
	double *diffxEy_re, 
	double *psi_hzx_m_re
){

	int i,k;
	int even;
	int psiidx;
	int last_idx;

	double CHz2;

	// Update Hz at j=(Ny-1) by using Ex at j=0
	#pragma omp parallel for \
		shared(	npml, Ny, Nz, dt, \
				mu_Hz, mcon_Hz, Hz_re, diffxEy_re, PMLkappax, PMLbx, PMLax, psi_hzx_m_re)\
		private(i, k, even, psiidx, last_idx, CHz2)
	for(i=0; i < npml; i++){
		for(k=0; k < Nz; k++){
	
			even     = (2*npml) - (2*i + 2);
			psiidx   = (k  ) + (Ny-1) * Nz + (i  ) * Nz * Ny;
			last_idx = (k  ) + (Ny-1) * Nz + (i  ) * Nz * Ny;

			CHz2 = (-2*dt) / (2.*mu_Hz[last_idx] + mcon_Hz[last_idx]*dt);

			//printf("%f\n", diffxEy_re[last_idx]);
			//printf("%d\n", last_idx);

			psi_hzx_m_re[psiidx] = (PMLbx[even] * psi_hzx_m_re[psiidx]) + (PMLax[even] * diffxEy_re[last_idx]);

			Hz_re[last_idx] += CHz2 * (+((1./PMLkappax[even] - 1.) * diffxEy_re[last_idx]) + psi_hzx_m_re[psiidx]);

		}
	}

	return;
}


void mxPML_mzPBC( \
	int myNx, int Ny, int Nz, int npml,\
	double dt, \
    double *PMLkappax, double *PMLbx,  double *PMLax, \
	double *eps_Ey, double *econ_Ey, \
	double *Ey_re, 
	double *diffxHz_re, 
	double *psi_eyx_m_re
){

	// By the staggered nature of the Yee grid, a plane of Ex or Ey at j=0 cannot be updated.
	// We will update this plane with the plane of Hz or Hx at j=(Ny-1), respectively.
	// This procedure effectively makes the simulation space periodic along y-axis.

	int i,j,k;
	int odd;
	int psiidx;
	int first_idx;

	double CEy1, CEy2;

	// Update Ey at k=0
	#pragma omp parallel for \
		shared(	npml, Ny, Nz, dt, \
				eps_Ey, econ_Ey, Ey_re, diffxHz_re, PMLkappax, PMLbx, PMLax, psi_eyx_m_re)\
		private(i, j, odd, psiidx, first_idx, CEy2)
	for(i=1; i < npml; i++){
		for(j=0; j < Ny; j++){

			odd       = (2*npml) - (2*i+1);
			psiidx    = (0  ) + (j  ) * Nz + (i  ) * Nz * Ny;
			first_idx = (0  ) + (j  ) * Nz + (i  ) * Nz * Ny;
			
			CEy2 = (2.*dt) / (2.*eps_Ey[first_idx] + econ_Ey[first_idx]*dt);

			if(eps_Ey[first_idx] > 1e3){
				CEy2 = 0.;
			}

			psi_eyx_m_re[psiidx] = (PMLbx[odd] * psi_eyx_m_re[psiidx]) + (PMLax[odd] * diffxHz_re[first_idx]);

			Ey_re[first_idx] += CEy2 * (-(1./PMLkappax[odd] - 1.) * diffxHz_re[first_idx] - psi_eyx_m_re[psiidx]);

		}
	}

	return;

}


void mxPML_pzPBC(\
	int myNx, int Ny, int Nz, int npml,\
	double dt, \
    double *PMLkappax, double *PMLbx,  double *PMLax, \
	double *mu_Hy, double *mcon_Hy, \
	double *Hy_re, 
	double *diffxEz_re, 
	double *psi_hyx_m_re
){

	// By the Yee grid, a plane of Hx or Hy at k=(Nz-1) cannot be updated.
	// We will update this plane with the plane of Ey or Ex at k=0, respectively.

	int i,j;
	int even;
	int psiidx;
	int last_idx;

	double CHy1, CHy2;

	// Update Hy at k=(Nz-1)
	#pragma omp parallel for \
		shared(	npml, Ny, Nz, dt, \
				mu_Hy, mcon_Hy, Hy_re, diffxEz_re, PMLkappax, PMLbx, PMLax, psi_hyx_m_re)\
		private(i, j, even, psiidx, last_idx, CHy2)
	for(i=0; i < npml; i++){
		for(j=0; j < Ny; j++){

			even     = (2*npml) - (2*i + 2);
			psiidx   = (Nz-1) + (j  ) * Nz + (i  ) * Nz * Ny;
			last_idx = (Nz-1) + (j  ) * Nz + (i  ) * Nz * Ny;
			
			CHy2 =	(-2*dt) / (2.*mu_Hy[last_idx] + mcon_Hy[last_idx]*dt);

			psi_hyx_m_re[psiidx] = (PMLbx[even] * psi_hyx_m_re[psiidx]) + (PMLax[even] * diffxEz_re[last_idx]);

			Hy_re[last_idx] += CHy2 * (-((1./PMLkappax[even] - 1.) * diffxEz_re[last_idx]) - psi_hyx_m_re[psiidx]);

		}
	}

	return;

}


void pxPML_myPBC( \
	int myNx, int Ny, int Nz, int npml, \
	double dt, \
	double *PMLkappax, double *PMLbx, double *PMLax, \
	double *eps_Ez, double *econ_Ez, \
	double *Ez_re, 
	double *diffxHy_re, 
	double *psi_ezx_p_re
){

	int i,j,k;
	int even;
	int psiidx, myidx;
	
	double CEy2, CEz2;

	// Update Ez. Note that i starts from 0, not 1.
	#pragma omp parallel for \
		shared(	npml, myNx, Ny, Nz,	dt,	\
				eps_Ez, econ_Ez, Ez_re, diffxHy_re, PMLkappax,	PMLbx,	PMLax, psi_ezx_p_re)	\
		private(i, j, k, even, psiidx, myidx, CEy2, CEz2)
	for(i=0; i < npml; i++){
		for(k=0; k < Nz; k++){

			even   = 2*i;
			psiidx = (k  ) + (0  ) * Nz + (i		  ) * Nz * Ny;
			myidx  = (k  ) + (0  ) * Nz + (i+myNx-npml) * Nz * Ny;

			CEz2 =	(2.*dt) / (2.*eps_Ez[myidx] + econ_Ez[myidx]*dt);

			if(eps_Ez[myidx] > 1e3){
				CEz2 = 0.;
			}

			psi_ezx_p_re[psiidx] = (PMLbx[even] * psi_ezx_p_re[psiidx]) + (PMLax[even] * diffxHy_re[myidx]);

			Ez_re[myidx] += CEz2 * (+(1./PMLkappax[even] - 1.) * diffxHy_re[myidx] + psi_ezx_p_re[psiidx]);

		}
	}

	return;
};


void pxPML_pyPBC( \
	int myNx, int Ny, int Nz, int npml,\
	double dt, \
	double *PMLkappax, double *PMLbx, double *PMLax, \
	double *mu_Hz, double *mcon_Hz, \
	double *Hz_re, 
	double *diffxEy_re, 
	double *psi_hzx_p_re
){

	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CHy2, CHz2;

	// Update Hz
	#pragma omp parallel for \
		shared(	npml, myNx, Ny, Nz,	dt,	\
				mu_Hz, mcon_Hz,	Hz_re, diffxEy_re, PMLkappax, PMLbx, PMLax, psi_hzx_p_re) \
		private(i, j, k,odd, psiidx, myidx, CHz2)
	for(i=0; i < (npml-1); i++){
		for(k=0; k < Nz; k++){
			
			odd    = 2*i + 1;
			psiidx = (k  ) + (Ny-1) * Nz + (i		   ) * Nz * Ny;
			myidx  = (k  ) + (Ny-1) * Nz + (i+myNx-npml) * Nz * Ny;

			CHz2 =	(-2*dt) / (2.*mu_Hz[myidx] + mcon_Hz[myidx]*dt);

			psi_hzx_p_re[psiidx] = (PMLbx[odd] * psi_hzx_p_re[psiidx]) + (PMLax[odd] * diffxEy_re[myidx]);

			Hz_re[myidx] += CHz2 * (+((1./PMLkappax[odd] - 1.) * diffxEy_re[myidx]) + psi_hzx_p_re[psiidx]);
		}
	}

	return;
};


void pxPML_mzPBC( \
	int myNx, int Ny, int Nz, int npml,\
	double dt, \
    double *PMLkappax, double *PMLbx,  double *PMLax, \
	double *eps_Ey, double *econ_Ey, \
	double *Ey_re, 
	double *diffxHz_re, 
	double *psi_eyx_p_re
){

	int i,j,k;
	int even;
	int psiidx, myidx;
	
	double CEy2, CEz2;

	// Update Ey. Note that i starts from 0, not 1.
	#pragma omp parallel for \
		shared(	npml, myNx, Ny, Nz,	dt,	\
				eps_Ey,	econ_Ey, Ey_re, diffxHz_re, PMLkappax, PMLbx, PMLax, psi_eyx_p_re) \
		private(i, j, k, even, psiidx, myidx, CEy2, CEz2)
	for(i=0; i < npml; i++){
		for(j=0; j < Ny; j++){

			even   = 2*i;
			psiidx = (0  ) + (j  ) * Nz + (i		  ) * Nz * Ny;
			myidx  = (0  ) + (j  ) * Nz + (i+myNx-npml) * Nz * Ny;

			CEy2 =	(2.*dt) / (2.*eps_Ey[myidx] + econ_Ey[myidx]*dt);

			if(eps_Ey[myidx] > 1e3){
				CEy2 = 0.;
			}

			psi_eyx_p_re[psiidx] = (PMLbx[even] * psi_eyx_p_re[psiidx]) + (PMLax[even] * diffxHz_re[myidx]);

			Ey_re[myidx] += CEy2 * (-(1./PMLkappax[even] - 1.) * diffxHz_re[myidx] - psi_eyx_p_re[psiidx]);

		}
	}

	return;
};


void pxPML_pzPBC(\
	int myNx, int Ny, int Nz, int npml,\
	double dt, \
    double *PMLkappax, double *PMLbx,  double *PMLax, \
	double *mu_Hy, double *mcon_Hy, \
	double *Hy_re, 
	double *diffxEz_re, 
	double *psi_hyx_p_re
){

	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CHy2, CHz2;

	// Update Hy
	#pragma omp parallel for \
		shared(	npml, myNx, Ny, Nz,	dt,	\
				mu_Hy, mcon_Hy, Hy_re, diffxEz_re, PMLkappax, PMLbx, PMLax, psi_hyx_p_re) \
		private(i, j, k,odd, psiidx, myidx, CHy2)
	for(i=0; i < (npml-1); i++){
		for(j=0; j < Ny; j++){
				
			odd    = 2*i + 1;
			psiidx = (Nz-1) + (j  ) * Nz + (i		   ) * Nz * Ny;
			myidx  = (Nz-1) + (j  ) * Nz + (i+myNx-npml) * Nz * Ny;

			CHy2 =	(-2*dt) / (2.*mu_Hy[myidx] + mcon_Hy[myidx]*dt);

			psi_hyx_p_re[psiidx] = (PMLbx[odd] * psi_hyx_p_re[psiidx]) + (PMLax[odd] * diffxEz_re[myidx]);

			Hy_re[myidx] += CHy2 * (-((1./PMLkappax[odd] - 1.) * diffxEz_re[myidx]) - psi_hyx_p_re[psiidx]);
		}
	}

	return;
};
