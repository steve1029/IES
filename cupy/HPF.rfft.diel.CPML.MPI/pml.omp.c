#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

/*
	Auther: Donggun Lee
	Date  : 18.07.24

	Update equations of PML region is written in here.
	Simulation method is the hybrid PSTD-FDTD method.
	Applied PML theory is Convolution PML(CPML) introduced by Roden and Gedney.
	For more details, see 'Convolution PML (CPML): An efficient FDTD implementation
	of the CFS-PML for arbitrary media', 22 June 2000, Microwave and Optical technology letters.

	Function declaration
	------------------------------

	void PML_updateH_px();
	void PML_updateH_mx();
	void PML_updateH_py();
	void PML_updateH_my();
	void PML_updateH_pz();
	void PML_updateH_mz();

	void PML_updateE_px();
	void PML_updateE_mx();
	void PML_updateE_py();
	void PML_updateE_my();
	void PML_updateE_pz();
	void PML_updateE_mz();
*/

/***********************************************************************************/
/******************************** FUNCTION DECLARATION *****************************/
/***********************************************************************************/

// PML at x+.
void PML_updateH_px(												\
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hy_re,			
	double *Hz_re,			
	double *diffxEy_re,		
	double *diffxEz_re,		
	double *psi_hyx_p_re,	
	double *psi_hzx_p_re
);

void PML_updateE_px(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ey_re,			
	double *Ez_re,			
	double *diffxHy_re,		
	double *diffxHz_re,		
	double *psi_eyx_p_re,
	double *psi_ezx_p_re
);

// PML at x-.
void PML_updateH_mx(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hy_re,			
	double *Hz_re,			
	double *diffxEy_re,		
	double *diffxEz_re,		
	double *psi_hyx_m_re,	
	double *psi_hzx_m_re
);

void PML_updateE_mx(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ey_re,			
	double *Ez_re,			
	double *diffxHy_re,	
	double *diffxHz_re,
	double *psi_eyx_m_re,
	double *psi_ezx_m_re
);

// PML at y+.
void PML_updateH_py(
	int MPIsize,	int MPIrank,
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hx_re,	
	double *Hz_re,	
	double *diffyEx_re,
	double *diffyEz_re,
	double *psi_hxy_p_re,	
	double *psi_hzy_p_re
);

void PML_updateE_py(
	int MPIsize,	int MPIrank,
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ex_re,		
	double *Ez_re,		
	double *diffyHx_re,	
	double *diffyHz_re,	
	double *psi_exy_p_re,
	double *psi_ezy_p_re
);

// PML at y-.
void PML_updateH_my(
	int MPIsize,	int MPIrank,
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hx_re,	
	double *Hz_re,	
	double *diffyEx_re,
	double *diffyEz_re,
	double *psi_hxy_p_re,	
	double *psi_hzy_p_re
);
void PML_updateE_my(
	int MPIsize,	int MPIrank,
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ex_re,		
	double *Ez_re,		
	double *diffyHx_re,	
	double *diffyHz_re,	
	double *psi_exy_p_re,
	double *psi_ezy_p_re
);

// PML at z+.
void PML_updateH_pz(
	int MPIsize,	int MPIrank,
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hx_re,		
	double *Hy_re,		
	double *diffzEx_re,	
	double *diffzEy_re,	
	double *psi_hxz_p_re,	
	double *psi_hyz_p_re
);

void PML_updateE_pz(
	int MPIsize,	int MPIrank,
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ex_re,			
	double *Ey_re,			
	double *diffzHx_re,		
	double *diffzHy_re,		
	double *psi_exz_p_re,	
	double *psi_eyz_p_re
);

// PML at z-.
void PML_updateH_mz(
	int MPIsize,	int MPIrank,
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hx_re,			
	double *Hy_re,			
	double *diffzEx_re,	
	double *diffzEy_re,	
	double *psi_hxz_m_re,
	double *psi_hyz_m_re
);
void PML_updateE_mz(
	int MPIsize,	int MPIrank,
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ex_re,		
	double *Ey_re,		
	double *diffzHx_re,	
	double *diffzHy_re,	
	double *psi_exz_m_re,
	double *psi_eyz_m_re
);

/***********************************************************************************/
/******************************** FUNCTION DESCRIPTION *****************************/
/***********************************************************************************/

/*----------------------------------- PML at x+ -----------------------------------*/
void PML_updateH_px(											\
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hy_re,			
	double *Hz_re,			
	double *diffxEy_re,		
	double *diffxEz_re,		
	double *psi_hyx_p_re,	
	double *psi_hzx_p_re
){

	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	//printf("Here?\n");
	double CHy2, CHz2;
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappax, PMLbx, PMLax,\
				mu_HEE,		mu_EHH,		\
				mcon_HEE,	mcon_EHH,	\
				Hy_re, \
				Hz_re, \
				diffxEy_re, \
				diffxEz_re, \
				psi_hyx_p_re, \
				psi_hzx_p_re )\
		private(i, j, k,odd, psiidx, myidx, CHy2, CHz2)
	for(i=0; i < npml; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				odd    = 2*i + 1;
				psiidx = (k  ) + (j  ) * Nz + (i          ) * Nz * Ny;
				myidx  = (k  ) + (j  ) * Nz + (i+myNx-npml) * Nz * Ny;

				CHy2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);
				CHz2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);

				// Update Hy
				psi_hyx_p_re[psiidx] = (PMLbx[odd] * psi_hyx_p_re[psiidx]) + (PMLax[odd] * diffxEz_re[myidx]);

				Hy_re[myidx] += CHy2 * (-((1./PMLkappax[odd] - 1.) * diffxEz_re[myidx]) - psi_hyx_p_re[psiidx]);

				// Update Hz
				psi_hzx_p_re[psiidx] = (PMLbx[odd] * psi_hzx_p_re[psiidx]) + (PMLax[odd] * diffxEy_re[myidx]);

				Hz_re[myidx] += CHz2 * (+((1./PMLkappax[odd] - 1.) * diffxEy_re[myidx]) + psi_hzx_p_re[psiidx]);
			}
		}
	}
	return;
};


void PML_updateE_px(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ey_re,			
	double *Ez_re,			
	double *diffxHy_re,		
	double *diffxHz_re,		
	double *psi_eyx_p_re,	
	double *psi_ezx_p_re
){

	int i,j,k;
	int even;
	int psiidx, myidx;
	
	double CEy2, CEz2;

	#pragma omp parallel for				\
		shared(	npml, myNx, Ny, Nz,			\
				dt,							\
				PMLkappax,	PMLbx,	PMLax,	\
				eps_EHH,	eps_HEE,		\
				econ_EHH,	econ_HEE,		\
				Ey_re,		\
				Ez_re,		\
				diffxHy_re, \
				diffxHz_re, \
				psi_eyx_p_re, \
				psi_ezx_p_re )\
		private(i, j, k, even, psiidx, myidx, CEy2, CEz2)
	for(i=0; i < npml; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				even   = 2*i;
				psiidx = (k  ) + (j  ) * Nz + (i          ) * Nz * Ny;
				myidx  = (k  ) + (j  ) * Nz + (i+myNx-npml) * Nz * Ny;

				CEy2 =	(2.*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);
				CEz2 =	(2.*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);

				// Update Ey.
				psi_eyx_p_re[psiidx] = (PMLbx[even] * psi_eyx_p_re[psiidx]) + (PMLax[even] * diffxHz_re[myidx]);

				Ey_re[myidx] += CEy2 * (-(1./PMLkappax[even] - 1.) * diffxHz_re[myidx] - psi_eyx_p_re[psiidx]);

				// Update Ez.
				psi_ezx_p_re[psiidx] = (PMLbx[even] * psi_ezx_p_re[psiidx]) + (PMLax[even] * diffxHy_re[myidx]);

				Ez_re[myidx] += CEz2 * (+(1./PMLkappax[even] - 1.) * diffxHy_re[myidx] + psi_ezx_p_re[psiidx]);

			}		
		}
	}

	return;
}

/*----------------------------------- PML at x- -----------------------------------*/
void PML_updateH_mx(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hy_re,			
	double *Hz_re,			
	double *diffxEy_re,		
	double *diffxEz_re,		
	double *psi_hyx_m_re,	
	double *psi_hzx_m_re
){

	int i,j,k;
	int even;
	int psiidx, myidx;
	
	double CHy2, CHz2;
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappax, PMLbx, PMLax,\
				mu_HEE,		mu_EHH,		\
				mcon_HEE,	mcon_EHH,	\
				Hy_re, \
				Hz_re, \
				diffxEy_re, \
				diffxEz_re, \
				psi_hyx_m_re, \
				psi_hzx_m_re)	\
		private(i, j, k, even, psiidx, myidx, CHy2, CHz2)
	for(i=0; i < npml; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				even   = (2*npml) - (2*i + 2);
				psiidx = (k  ) + (j  ) * Nz + (i          ) * Nz * Ny;
				myidx  = (k  ) + (j  ) * Nz + (i		  ) * Nz * Ny;

				CHy2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);
				CHz2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);

				// Update Hy
				psi_hyx_m_re[psiidx] = (PMLbx[even] * psi_hyx_m_re[psiidx]) + (PMLax[even] * diffxEz_re[myidx]);

				Hy_re[myidx] += CHy2 * (-((1./PMLkappax[even] - 1.) * diffxEz_re[myidx]) - psi_hyx_m_re[psiidx]);

				// Update Hz
				psi_hzx_m_re[psiidx] = (PMLbx[even] * psi_hzx_m_re[psiidx]) + (PMLax[even] * diffxEy_re[myidx]);

				Hz_re[myidx] += CHz2 * (+((1./PMLkappax[even] - 1.) * diffxEy_re[myidx]) + psi_hzx_m_re[psiidx]);
			}
		}
	}
	return;
};

void PML_updateE_mx(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ey_re,			
	double *Ez_re,			
	double *diffxHy_re,		
	double *diffxHz_re,		
	double *psi_eyx_m_re,
	double *psi_ezx_m_re
){
	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CEy2, CEz2;

	#pragma omp parallel for				\
		shared(	npml, myNx, Ny, Nz,			\
				dt,							\
				PMLkappax,	PMLbx,	PMLax,	\
				eps_EHH,	eps_HEE,		\
				econ_EHH,	econ_HEE,		\
				Ey_re,		\
				Ez_re,		\
				diffxHy_re, \
				diffxHz_re, \
				psi_eyx_m_re, \
				psi_ezx_m_re)\
		private(i, j, k, odd, psiidx, myidx, CEy2, CEz2)
	for(i=0; i < npml; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				odd    = (2*npml) - (2*i+1);
				psiidx = (k  ) + (j  ) * Nz + (i          ) * Nz * Ny;
				myidx  = (k  ) + (j  ) * Nz + (i		  ) * Nz * Ny;

				CEy2 =	(2.*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);
				CEz2 =	(2.*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);

				// Update Ey.
				psi_eyx_m_re[psiidx] = (PMLbx[odd] * psi_eyx_m_re[psiidx]) + (PMLax[odd] * diffxHz_re[myidx]);

				Ey_re[myidx] += CEy2 * (-(1./PMLkappax[odd] - 1.) * diffxHz_re[myidx] - psi_eyx_m_re[psiidx]);

				// Update Ez.
				psi_ezx_m_re[psiidx] = (PMLbx[odd] * psi_ezx_m_re[psiidx]) + (PMLax[odd] * diffxHy_re[myidx]);

				Ez_re[myidx] += CEz2 * (+(1./PMLkappax[odd] - 1.) * diffxHy_re[myidx] + psi_ezx_m_re[psiidx]);

			}		
		}
	}
	return;
};

/*----------------------------------- PML at y+ -----------------------------------*/
void PML_updateH_py(
	int MPIsize,	int MPIrank,
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hx_re,			
	double *Hz_re,			
	double *diffyEx_re,		
	double *diffyEz_re,		
	double *psi_hxy_p_re,	
	double *psi_hzy_p_re
){
	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CHx2, CHz2;
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappay, PMLby, PMLay,\
				mu_HEE,		mu_EHH,		\
				mcon_HEE,	mcon_EHH,	\
				Hx_re, \
				Hz_re, \
				diffyEx_re, \
				diffyEz_re, \
				psi_hxy_p_re, \
				psi_hzy_p_re)	\
		private(i, j, k, odd, psiidx, myidx, CHx2, CHz2)
	for(i=0; i < myNx; i++){
		for(j=0; j < npml; j++){
			for(k=0; k < Nz; k++){
				
				odd    = 2*j+1;
				psiidx = (k  ) + (j		   ) * Nz + (i  ) * Nz * npml;
				myidx  = (k  ) + (j+Ny-npml) * Nz + (i  ) * Nz * Ny;
				//if(i==0){printf("%d has %d\n", omp_get_thread_num(), psiidx);};

				CHx2 =	(-2*dt) / (2.*mu_HEE[myidx] + mcon_HEE[myidx]*dt);
				CHz2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);

				// Update Hx
				psi_hxy_p_re[psiidx] = (PMLby[odd] * psi_hxy_p_re[psiidx]) + (PMLay[odd] * diffyEz_re[myidx]);

				Hx_re[myidx] += CHx2 * (+((1./PMLkappay[odd] - 1.) * diffyEz_re[myidx]) + psi_hxy_p_re[psiidx]);

				// Update Hz
				if (MPIrank < (MPIsize-1)) {
					psi_hzy_p_re[psiidx] = (PMLby[odd] * psi_hzy_p_re[psiidx]) + (PMLay[odd] * diffyEx_re[myidx]);

					Hz_re[myidx] += CHz2 * (-((1./PMLkappay[odd] - 1.) * diffyEx_re[myidx]) - psi_hzy_p_re[psiidx]);
				}
				else if (i < (myNx-1)){
					psi_hzy_p_re[psiidx] = (PMLby[odd] * psi_hzy_p_re[psiidx]) + (PMLay[odd] * diffyEx_re[myidx]);

					Hz_re[myidx] += CHz2 * (-((1./PMLkappay[odd] - 1.) * diffyEx_re[myidx]) - psi_hzy_p_re[psiidx]);
				}
			}
		}
	}

	return;
};

void PML_updateE_py(
	int MPIsize,	int MPIrank,
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ex_re,			
	double *Ez_re,			
	double *diffyHx_re,		
	double *diffyHz_re,	
	double *psi_exy_p_re,
	double *psi_ezy_p_re
){
	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CEx2, CEz2;
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappay, PMLby, PMLay,\
				eps_HEE,	eps_EHH,	\
				econ_HEE,	econ_EHH,	\
				Ex_re, \
				Ez_re, \
				diffyHx_re, \
				diffyHz_re, \
				psi_exy_p_re, \
				psi_ezy_p_re)	\
		private(i, j, k, odd, psiidx, myidx, CEx2, CEz2)
	for(i=0; i < myNx; i++){
		for(j=0; j < npml; j++){
			for(k=0; k < Nz; k++){
				
				odd    = 2*j+1;
				psiidx = (k  ) + (j		   ) * Nz + (i  ) * Nz * npml;
				myidx  = (k  ) + (j+Ny-npml) * Nz + (i  ) * Nz * Ny;

				CEx2 =	(2*dt) / (2.*eps_EHH[myidx] + econ_EHH[myidx]*dt);
				CEz2 =	(2*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);

				// Update Ex
				psi_exy_p_re[psiidx] = (PMLby[odd] * psi_exy_p_re[psiidx]) + (PMLay[odd] * diffyHz_re[myidx]);

				Ex_re[myidx] += CEx2 * (+((1./PMLkappay[odd] - 1.) * diffyHz_re[myidx]) + psi_exy_p_re[psiidx]);

				// Update Ez
				if (MPIrank > 0){
					psi_ezy_p_re[psiidx] = (PMLby[odd] * psi_ezy_p_re[psiidx]) + (PMLay[odd] * diffyHx_re[myidx]);

					Ez_re[myidx] += CEz2 * (-((1./PMLkappay[odd] - 1.) * diffyHx_re[myidx]) - psi_ezy_p_re[psiidx]);
				}
				else if (i > 0){
					psi_ezy_p_re[psiidx] = (PMLby[odd] * psi_ezy_p_re[psiidx]) + (PMLay[odd] * diffyHx_re[myidx]);

					Ez_re[myidx] += CEz2 * (-((1./PMLkappay[odd] - 1.) * diffyHx_re[myidx]) - psi_ezy_p_re[psiidx]);
				}
			}
		}
	}
	return;
};

/*----------------------------------- PML at y- -----------------------------------*/
void PML_updateH_my(
	int MPIsize,	int MPIrank,
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hx_re,			
	double *Hz_re,			
	double *diffyEx_re,		
	double *diffyEz_re,		
	double *psi_hxy_p_re,	
	double *psi_hzy_p_re
){
	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CHx2, CHz2;
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappay, PMLby, PMLay,\
				mu_HEE,		mu_EHH,		\
				mcon_HEE,	mcon_EHH,	\
				Hx_re, \
				Hz_re, \
				diffyEx_re, \
				diffyEz_re, \
				psi_hxy_p_re, \
				psi_hzy_p_re)	\
		private(i, j, k, odd, psiidx, myidx, CHx2, CHz2)
	for(i=0; i < myNx; i++){
		for(j=0; j < npml; j++){
			for(k=0; k < Nz; k++){
				
				odd    = (2*npml) - (2*j+1);
				psiidx = (k  ) + (j  ) * Nz + (i  ) * Nz * npml;
				myidx  = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				//if(i==0){printf("%d has %d\n", omp_get_thread_num(), psiidx);};

				CHx2 =	(-2*dt) / (2.*mu_HEE[myidx] + mcon_HEE[myidx]*dt);
				CHz2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);

				// Update Hx
				psi_hxy_p_re[psiidx] = (PMLby[odd] * psi_hxy_p_re[psiidx]) + (PMLay[odd] * diffyEz_re[myidx]);

				Hx_re[myidx] += CHx2 * (+((1./PMLkappay[odd] - 1.) * diffyEz_re[myidx]) + psi_hxy_p_re[psiidx]);

				// Update Hz
				if (MPIrank < (MPIsize-1)) {
					psi_hzy_p_re[psiidx] = (PMLby[odd] * psi_hzy_p_re[psiidx]) + (PMLay[odd] * diffyEx_re[myidx]);

					Hz_re[myidx] += CHz2 * (-((1./PMLkappay[odd] - 1.) * diffyEx_re[myidx]) - psi_hzy_p_re[psiidx]);
				}
				else if (i < (myNx-1)){
					psi_hzy_p_re[psiidx] = (PMLby[odd] * psi_hzy_p_re[psiidx]) + (PMLay[odd] * diffyEx_re[myidx]);

					Hz_re[myidx] += CHz2 * (-((1./PMLkappay[odd] - 1.) * diffyEx_re[myidx]) - psi_hzy_p_re[psiidx]);
				}
			}
		}
	}
	return;
};

void PML_updateE_my(
	int MPIsize,	int MPIrank,
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ex_re,			
	double *Ez_re,			
	double *diffyHx_re,		
	double *diffyHz_re,		
	double *psi_exy_p_re,	
	double *psi_ezy_p_re
){
	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CEx2, CEz2;
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappay, PMLby, PMLay,\
				eps_HEE,	eps_EHH,	\
				econ_HEE,	econ_EHH,	\
				Ex_re, \
				Ez_re, \
				diffyHx_re, \
				diffyHz_re, \
				psi_exy_p_re, \
				psi_ezy_p_re)	\
		private(i, j, k, odd, psiidx, myidx, CEx2, CEz2)
	for(i=0; i < myNx; i++){
		for(j=0; j < npml; j++){
			for(k=0; k < Nz; k++){
				
				odd    = (2*npml) - (2*j+1);
				psiidx = (k  ) + (j  ) * Nz + (i  ) * Nz * npml;
				myidx  = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CEx2 =	(2*dt) / (2.*eps_EHH[myidx] + econ_EHH[myidx]*dt);
				CEz2 =	(2*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);

				// Update Ex
				psi_exy_p_re[psiidx] = (PMLby[odd] * psi_exy_p_re[psiidx]) + (PMLay[odd] * diffyHz_re[myidx]);

				Ex_re[myidx] += CEx2 * (+((1./PMLkappay[odd] - 1.) * diffyHz_re[myidx]) + psi_exy_p_re[psiidx]);

				// Update Ez
				if (MPIrank > 0){
					psi_ezy_p_re[psiidx] = (PMLby[odd] * psi_ezy_p_re[psiidx]) + (PMLay[odd] * diffyHx_re[myidx]);

					Ez_re[myidx] += CEz2 * (-((1./PMLkappay[odd] - 1.) * diffyHx_re[myidx]) - psi_ezy_p_re[psiidx]);
				}
				else if (i > 0){
					psi_ezy_p_re[psiidx] = (PMLby[odd] * psi_ezy_p_re[psiidx]) + (PMLay[odd] * diffyHx_re[myidx]);

					Ez_re[myidx] += CEz2 * (-((1./PMLkappay[odd] - 1.) * diffyHx_re[myidx]) - psi_ezy_p_re[psiidx]);
				}
			}
		}
	}
	return;
};

/*----------------------------------- PML at z+ -----------------------------------*/
void PML_updateH_pz(
	int MPIsize,	int MPIrank,
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hx_re,			
	double *Hy_re,			
	double *diffzEx_re,		
	double *diffzEy_re,		
	double *psi_hxz_p_re,	
	double *psi_hyz_p_re
){

	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CHx2, CHy2;
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappaz, PMLbz, PMLaz,\
				mu_HEE,		mu_EHH,		\
				mcon_HEE,	mcon_EHH,	\
				Hx_re, \
				Hy_re, \
				diffzEx_re, \
				diffzEy_re, \
				psi_hxz_p_re, \
				psi_hyz_p_re)	\
		private(i, j, k, odd, psiidx, myidx, CHx2, CHy2)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < npml; k++){
				
				odd    = 2*k+1;
				psiidx = (k        ) + (j  ) * npml + (i  ) * npml * Ny;
				myidx  = (k+Nz-npml) + (j  ) * Nz   + (i  ) * Nz   * Ny;
				//if(i==0){printf("%d has %d\n", omp_get_thread_num(), myidx);};

				CHx2 =	(-2*dt) / (2.*mu_HEE[myidx] + mcon_HEE[myidx]*dt);
				CHy2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);

				// Update Hx
				psi_hxz_p_re[psiidx] = (PMLbz[odd] * psi_hxz_p_re[psiidx]) + (PMLaz[odd] * diffzEy_re[myidx]);

				Hx_re[myidx] += CHx2 * (-((1./PMLkappaz[odd] - 1.) * diffzEy_re[myidx]) - psi_hxz_p_re[psiidx]);

				// Update Hy
				if (MPIrank < (MPIsize-1)) {
					psi_hyz_p_re[psiidx] = (PMLbz[odd] * psi_hyz_p_re[psiidx]) + (PMLaz[odd] * diffzEx_re[myidx]);

					Hy_re[myidx] += CHy2 * (+((1./PMLkappaz[odd] - 1.) * diffzEx_re[myidx]) + psi_hyz_p_re[psiidx]);
				}
				else if (i < (myNx-1)){
					psi_hyz_p_re[psiidx] = (PMLbz[odd] * psi_hyz_p_re[psiidx]) + (PMLaz[odd] * diffzEx_re[myidx]);

					Hy_re[myidx] += CHy2 * (+((1./PMLkappaz[odd] - 1.) * diffzEx_re[myidx]) + psi_hyz_p_re[psiidx]);
				}

			}
		}
	}
	return;
};

void PML_updateE_pz(
	int MPIsize,	int MPIrank,
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ex_re,			
	double *Ey_re,		
	double *diffzHx_re,	
	double *diffzHy_re,	
	double *psi_exz_p_re,
	double *psi_eyz_p_re
){
	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CEx2, CEy2;
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappaz, PMLbz, PMLaz,\
				eps_HEE,	eps_EHH,	\
				econ_HEE,	econ_EHH,	\
				Ex_re, \
				Ey_re, \
				diffzHx_re, \
				diffzHy_re, \
				psi_exz_p_re, \
				psi_eyz_p_re)	\
		private(i, j, k, odd, psiidx, myidx, CEx2, CEy2)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < npml; k++){
				
				odd    = 2*k+1;
				psiidx = (k        ) + (j  ) * npml + (i  ) * npml * Ny;
				myidx  = (k+Nz-npml) + (j  ) * Nz   + (i  ) * Nz   * Ny;

				CEx2 =	(2*dt) / (2.*eps_EHH[myidx] + econ_EHH[myidx]*dt);
				CEy2 =	(2*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);

				// Update Ex
				psi_exz_p_re[psiidx] = (PMLbz[odd] * psi_exz_p_re[psiidx]) + (PMLaz[odd] * diffzHy_re[myidx]);

				Ex_re[myidx] += CEx2 * (-((1./PMLkappaz[odd] - 1.) * diffzHy_re[myidx]) - psi_exz_p_re[psiidx]);

				// Update Ey
				if (MPIrank > 0){
					psi_eyz_p_re[psiidx] = (PMLbz[odd] * psi_eyz_p_re[psiidx]) + (PMLaz[odd] * diffzHx_re[myidx]);

					Ey_re[myidx] += CEy2 * (+((1./PMLkappaz[odd] - 1.) * diffzHx_re[myidx]) + psi_eyz_p_re[psiidx]);
				}
				else if (i > 0){
					psi_eyz_p_re[psiidx] = (PMLbz[odd] * psi_eyz_p_re[psiidx]) + (PMLaz[odd] * diffzHx_re[myidx]);

					Ey_re[myidx] += CEy2 * (+((1./PMLkappaz[odd] - 1.) * diffzHx_re[myidx]) + psi_eyz_p_re[psiidx]);
				}

			}
		}
	}
	return;
};

/*----------------------------------- PML at z- -----------------------------------*/
void PML_updateH_mz(
	int MPIsize,	int MPIrank,
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hx_re,			
	double *Hy_re,			
	double *diffzEx_re,		
	double *diffzEy_re,		
	double *psi_hxz_m_re,	
	double *psi_hyz_m_re
){

	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CHx2, CHy2;
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappaz, PMLbz, PMLaz,\
				mu_HEE,		mu_EHH,		\
				mcon_HEE,	mcon_EHH,	\
				Hx_re, \
				Hy_re, \
				diffzEx_re, \
				diffzEy_re, \
				psi_hxz_m_re,\
				psi_hyz_m_re)	\
		private(i, j, k, odd, psiidx, myidx, CHx2, CHy2)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < npml; k++){
				
				odd    = (2*npml) - (2*k+1);
				psiidx = (k        ) + (j  ) * npml + (i  ) * npml * Ny;
				myidx  = (k		   ) + (j  ) * Nz   + (i  ) * Nz   * Ny;

				CHx2 =	(-2.*dt) / (2.*mu_HEE[myidx] + mcon_HEE[myidx]*dt);
				CHy2 =	(-2.*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);

				// Update Hx
				psi_hxz_m_re[psiidx] = (PMLbz[odd] * psi_hxz_m_re[psiidx]) + (PMLaz[odd] * diffzEy_re[myidx]);

				Hx_re[myidx] += CHx2 * (-((1./PMLkappaz[odd] - 1.) * diffzEy_re[myidx]) - psi_hxz_m_re[psiidx]);

				// Update Hy
				if (MPIrank < (MPIsize-1)) {
					psi_hyz_m_re[psiidx] = (PMLbz[odd] * psi_hyz_m_re[psiidx]) + (PMLaz[odd] * diffzEx_re[myidx]);

					Hy_re[myidx] += CHy2 * (+((1./PMLkappaz[odd] - 1.) * diffzEx_re[myidx]) + psi_hyz_m_re[psiidx]);
				}
				else if (i < (myNx-1)){
					psi_hyz_m_re[psiidx] = (PMLbz[odd] * psi_hyz_m_re[psiidx]) + (PMLaz[odd] * diffzEx_re[myidx]);

					Hy_re[myidx] += CHy2 * (+((1./PMLkappaz[odd] - 1.) * diffzEx_re[myidx]) + psi_hyz_m_re[psiidx]);
				}

			}
		}
	}
	return;
};

void PML_updateE_mz(
	int MPIsize,	int MPIrank,
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ex_re,			
	double *Ey_re,			
	double *diffzHx_re,		
	double *diffzHy_re,	
	double *psi_exz_m_re,
	double *psi_eyz_m_re
){
	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CEx2, CEy2;
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappaz, PMLbz, PMLaz,\
				eps_HEE,	eps_EHH,	\
				econ_HEE,	econ_EHH,	\
				Ex_re, \
				Ey_re, \
				diffzHx_re, \
				diffzHy_re, \
				psi_exz_m_re, \
				psi_eyz_m_re)	\
		private(i, j, k, odd, psiidx, myidx, CEx2, CEy2)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < npml; k++){
				
				odd    = (2*npml) - (2*k+1);
				psiidx = (k        ) + (j  ) * npml + (i  ) * npml * Ny;
				myidx  = (k		   ) + (j  ) * Nz   + (i  ) * Nz   * Ny;
				//if(i==0){printf("%d has %d\n", omp_get_thread_num(), psiidx);};

				CEx2 =	(2.*dt) / (2.*eps_EHH[myidx] + econ_EHH[myidx]*dt);
				CEy2 =	(2.*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);

				// Update Ex
				psi_exz_m_re[psiidx] = (PMLbz[odd] * psi_exz_m_re[psiidx]) + (PMLaz[odd] * diffzHy_re[myidx]);

				Ex_re[myidx] += CEx2 * (-((1./PMLkappaz[odd] - 1.) * diffzHy_re[myidx]) - psi_exz_m_re[psiidx]);

				// Update Ey
				if (MPIrank > 0){
					psi_eyz_m_re[psiidx] = (PMLbz[odd] * psi_eyz_m_re[psiidx]) + (PMLaz[odd] * diffzHx_re[myidx]);

					Ey_re[myidx] += CEy2 * (+((1./PMLkappaz[odd] - 1.) * diffzHx_re[myidx]) + psi_eyz_m_re[psiidx]);
				}
				else if (i > 0){
					psi_eyz_m_re[psiidx] = (PMLbz[odd] * psi_eyz_m_re[psiidx]) + (PMLaz[odd] * diffzHx_re[myidx]);

					Ey_re[myidx] += CEy2 * (+((1./PMLkappaz[odd] - 1.) * diffzHx_re[myidx]) + psi_eyz_m_re[psiidx]);
				}

			}
		}
	}

	return;
};
