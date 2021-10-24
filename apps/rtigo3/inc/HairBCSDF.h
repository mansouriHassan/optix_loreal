
/* ---------------------------------------------------------
 * Bidirectional curve scattering distribution function
 * ---------------------------------------------------------
 * Pre-computes the azimuthal scattering functions
 * Based on the hair rendering implementation from the
 * tungsten renderer for "energy-conserving hair reflectance
 * model" and importance sampling for physically-based hair
 * fiber models" from dEon et al.
 * --------------------------------------------------------- */

#pragma once

#ifndef HAIR_BCSDF_H
#define HAIR_BCSDF_H

/* ----------------------- Libraries ----------------------- */

#include "app_config.h"

#include <optix.h>
#include <optixu/optixpp_namespace.h>

using namespace optix;

// Number of points for the computation of lookup tables of hair CDFs
#define NUM_POINTS 70

/* ---------------------- Hair BCSDF ----------------------- */
class HairBCSDF {

public:

	/* Constructors ---------------------------------------- */
	HairBCSDF();
	~HairBCSDF();

	/* Standard normalized Gaussian ---------------------------- */
	float g(float beta, float theta);
	/* Wrapped Gaussina "deterctor", computed as an infinite sum of
	 * Gaussians (approximiated with a finite sum) --------- */
	float D(float beta, float phi);
	/* Computes the existant azimuthal angle of the p'th
	* perfect specular scattering event, derived using
	* Bravais theory ------------------------------------------ */
	float Phi(float gamma_i, float gamma_t, int p);
	float I0(float const& x);
	float logI0(float const& x);
	float M(float const& v, float const& sin_theta_i,
		float const& sin_theta_o, float const& cos_theta_i,
		float const& cos_theta_o);
	float FrDielectric(float const& cos_theta, float const& n1, float const& n2);
	float3 ApR(float const& h, float const& cos_theta_o, float const& eta);
	float Logistic(float const& x, float const& s);
	float LogisticCDF(float const& x, float const& s);
	float TrimmedLogistic(float const& x, float const& s, float const& a, float const& b);
	float SampleTrimmedLogistic(float const& u, float const& s, float const& a, float const& b);
	float Phi(int const& p, float const& gammat, float const& gamma0);
	float Np(float const& phi, int const& p, float const& s, float const& gammaO, float const& gammaT);

	/* Methods --------------------------------------------- */
	bool precomputeAzimuthalDistribution(float3 beta, float ior,float scale_angle_rad,
		float3 absorption);

	void calculateFunctions(Context context, float scale_angle_rad);

	/* Gets ------------------------------------------------ */
	Buffer getBufferValuesR() const;
	Buffer getBufferValuesTT() const;
	Buffer getBufferValuesTRT() const;
	Buffer getBufferValuessop() const;
	Buffer getBufferValuescop() const;
	
	Buffer getbuffer_values_N_R(){ return m_buffer_values_N_R; };
	Buffer getbuffer_values_N_TT(){ return m_buffer_values_N_TT; };
	Buffer getbuffer_values_N_TRT(){ return m_buffer_values_N_TRT; };
	Buffer getbuffer_values_gammaO(){ return m_buffer_values_gammaO; };
	Buffer getbuffer_values_gammaT(){ return m_buffer_values_gammaT; };
	Buffer getbuffer_values_fracT(){ return m_buffer_values_fracT; };
	Buffer getbuffer_values_Fr(){ return m_buffer_values_Fr; };

	int getResolution() const;

private:
	/* Constants ------------------------------------------- */
	const int resolution = 64;
	const int num_gaussian_samples = 2048;

	/* Variables ------------------------------------------- */
	float *m_values_R;
	float *m_values_TT;
	float *m_values_TRT;
	float3 *m_values_cop;
	float3 *m_values_sop;
	float3 *m_values_Fr;
	float *m_values_gammaO;
	float *m_values_gammaT;
	float *m_values_fracT;
	float *m_values_N_R;
	float *m_values_N_TT;
	float *m_values_N_TRT;


	Buffer m_buffer_values_R;
	Buffer m_buffer_values_TT;
	Buffer m_buffer_values_TRT;
	Buffer m_buffer_values_cop;
	Buffer m_buffer_values_sop;
	Buffer m_buffer_values_N_R;
	Buffer m_buffer_values_N_TT;
	Buffer m_buffer_values_N_TRT;
	Buffer m_buffer_values_gammaO;
	Buffer m_buffer_values_gammaT;
	Buffer m_buffer_values_fracT;
	Buffer m_buffer_values_Fr;



};

#endif