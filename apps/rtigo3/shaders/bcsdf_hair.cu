
/* ---------------------------------------------------------
* Bidirectional curve scattering distribution function
* for a hair material
* ---------------------------------------------------------
* Based on the hair rendering implementation from the
* tungsten renderer for "energy-conserving hair reflectance
* model" and importance sampling for physically-based hair
* fiber models" from dEon et al.
* ---------------------------------------------------------
*/

/* ----------------------- Libraries ----------------------- */

#include <stdio.h>
#include "config.h"

#include <optix.h>

#include "per_ray_data.h"
#include "material_definition.h"
#include "shader_common.h"
#include "random_number_generators.h"
#include "curve.h"

/*rtBuffer<float3> id_values_sop;
rtBuffer<float3> id_values_cop;*/

/* ----------------------- Functions ----------------------- */

__forceinline__ __device__ float logI0(float const& x) {
	return (x > 12.0f) ? x + 0.5f * (-logf((2.f * M_PIf * x)) + fdividef(1.0f, (8.0f * x))) : logf(cyl_bessel_i0f(x));
}

__forceinline__ __device__ float FrDielectric(float const& cos_theta, float const& n1,
	float const& n2) {

	float const R0 = fdividef(n1 - n2, n1 + n2) * fdividef(n1 - n2, n1 + n2);
	return R0 + (1.f - R0) * (1.f - cos_theta) * (1.f - cos_theta) * (1.f - cos_theta) * (1.f - cos_theta) * (1.f - cos_theta);
}


/* Rough longitudinal scattering function with variance
* v = beta^2 ---------------------------------------------- */
__forceinline__ __device__ float M(float const& v, float const& sin_theta_i,
	float const& sin_theta_o, float const& cos_theta_i,
	float const& cos_theta_o) {
	float a = cos_theta_i * fdividef(cos_theta_o, v);
	float b = sin_theta_i * fdividef(sin_theta_o, v);
	float mp =
		(v <= 0.1f)
		? (expf(logI0(a) - b - fdividef(1.f, v) + 0.6931f + logf(fdividef(1.f, (2.f * v)))))
		: fdividef((expf(-b) * cyl_bessel_i0f(a)), (sinhf(fdividef(1.f, v)) * 2.f * v));
	return mp;
}


/* Sampling methods ----------------------------------------- */
__forceinline__ __device__ float3 ApR(float const& h, float const& cos_theta_o,
	float const& eta) {

	float cosgamma0 = sqrtf(1.f - h * h);
	float cosTheta = cos_theta_o * cosgamma0;
	float f = FrDielectric(cosTheta, 1.f, eta);
	return make_float3(f);
}

__forceinline__ __device__ float Phi(int const& p, float const& gammat,
	float const& gamma0) {

	return 2.f * p * gammat - 2.f * gamma0 + p * M_PIf;
}

__forceinline__ __device__ float Logistic(float const& x,
	float const& s) {
	float ax = fabsf(x);
	float frac = fdividef(ax, s);
	return expf(-frac) / (s * (1.f + expf(-frac)) * (1.f + expf(-frac)));
}

__forceinline__ __device__ float LogisticCDF(float const& x, float const& s) {
	return fdividef(1.f, (1.f + expf(-fdividef(x, s))));
}

__forceinline__ __device__ float TrimmedLogistic(float const& x, float const& s, float const& a, float const& b) {
	return fdividef(Logistic(x, s), (LogisticCDF(b, s) - LogisticCDF(a, s)));
}

__forceinline__ __device__ float SampleTrimmedLogistic(float const& u, float const& s, float const& a, float const& b) {
	float k = LogisticCDF(b, s) - LogisticCDF(a, s);
	float x = -s * logf(fdividef(1.f, (u * k + LogisticCDF(a, s))) - 1.f);
	return clamp(x, a, b);
}

__forceinline__ __device__ float Np(float const& phi,
	int const& p, float const& s, float const& gammaO, float const& gammaT) {
	float dphi = phi - Phi(p, gammaT, gammaO);
	// Remap _dphi_ to $[-\pi,\pi]$
	while (dphi > M_PIf) { dphi -= 2.f * M_PIf; }
	while (dphi < -M_PIf) { dphi += 2.f * M_PIf; }
	return TrimmedLogistic(dphi, s, -M_PIf, M_PIf);
}

extern "C" __device__ void __direct_callable__sample_bcsdf_hair(MaterialDefinition const& material, State const& state, PerRayData * prd) {
	// Conventional wi correspond to -wo in the code
	// Conventional wo correspond to wi in the code

	
	float h = -1.f + 2.f * rng(prd->seed);//dot(state.normal, cross(state.tangent, state.texcoord)); //
	float gammaO = asinf(h);

	float2 xi_N = rng2(prd->seed);
	float2 xi_M = rng2(prd->seed);

	float ior = 1.55f;

	const float sin_theta_o = dot(prd->wo, state.tangent);
	const float cos_theta_o = trigInverse(sin_theta_o);
	const float ndwo = dot(prd->wo, state.texcoord);
	const float bdwo = dot(prd->wo, normalize(cross(state.texcoord,state.tangent)));
	float phio = atan2f(prd->wo.x, prd->wo.z);

	float sin2kAlpha0 = sinf(material.scale_angle_rad);
    float cos2kAlpha0 = trigInverse(sin2kAlpha0);
	float sin2kAlpha1 = 2.f * cos2kAlpha0 * sin2kAlpha0;
    float cos2kAlpha1 = cos2kAlpha0*cos2kAlpha0-sin2kAlpha0*sin2kAlpha0;
	float sin2kAlpha2 = 2.f * cos2kAlpha1 * sin2kAlpha1;
	float cos2kAlpha2 = cos2kAlpha1 * cos2kAlpha1 - sin2kAlpha1 * sin2kAlpha1;



	float sinThetaT = sin_theta_o / ior;
	float cosThetaT = trigInverse(sinThetaT);
	float etap = sqrtf(ior * ior - (sin_theta_o * sin_theta_o)) / cos_theta_o;
	//float etas = ior * ior*cos_theta_o/sqrtf(ior * ior - (sin_theta_o*sin_theta_o));
	float sinGammaT = h / etap;
	float cosGammaT = trigInverse(sinGammaT);
	float gammaT = asinf(sinGammaT);


	float3 absorption = material.absorption;
	const float3 absorption_eumelanin = make_float3(0.419f, 0.697f, 1.37f);
	const float3 absorption_pheomelanin = make_float3(0.187f, 0.4f, 1.05f);

	float melanin_ratio = clamp(material.melanin_ratio * (1.f + material.melanin_ratio_disparity * state.rand.y),0.f,1.f);
	float melanin_concentration = fmaxf(material.melanin_concentration * (1.f + state.rand.y * material.melanin_concentration_disparity),.0f);
	absorption += (state.rand.x > material.whitepercen) ? melanin_concentration * lerp(absorption_eumelanin, absorption_pheomelanin, melanin_ratio) : make_float3(0.f);

	float3 T = expf(-absorption * 2.f * cosGammaT / cosThetaT);

	float3 R = ApR(h, cos_theta_o, etap);
	//float3 Rs = ApR(h, cos_theta_o,etas);
	//float3 R = 0.5f*(Rp+Rs);
	float3 TT = (make_float3(1.f) - R) * (make_float3(1.f) - R) * T;
	float3 TRT = TT * R * T;
	//float3 TRRT = TRT*T*R/(1.f-T*R);

	float lum = luminance(R) + luminance(TT) + luminance(TRT);
	float3 apsample = make_float3(luminance(R), luminance(TT), luminance(TRT)) / lum;

	//Pour ajouter un 4e lobe

	//float lum =  luminance(R)  +luminance(TT) + luminance(TRT)+luminance(TRRT) ;
	//float4 apsample =  make_float4(luminance(R),luminance(TT),luminance(TRT),luminance(TRRT))/lum;

	const float s = 0.626657069f * (0.265f * material.betaN + 1.194f * (material.betaN * material.betaN) + 5.372f * (material.betaN * material.betaN) * (material.betaN * material.betaN) * (material.betaN * material.betaN) *
		(material.betaN * material.betaN) * (material.betaN * material.betaN) * (material.betaN * material.betaN) * (material.betaN * material.betaN) * (material.betaN * material.betaN) *
		(material.betaN * material.betaN) * (material.betaN * material.betaN) * (material.betaN * material.betaN));

	float dphi = SampleTrimmedLogistic(xi_N.y, s, -M_PIf, M_PIf);

	float cosPhi = cosf(2.f * M_PIf * xi_M.y);

	//Sample M
	const float sqrtv = 0.726f * material.betaM + 0.812f * (material.betaM * material.betaM) + 3.7f * (material.betaM * material.betaM) * (material.betaM * material.betaM) * (material.betaM * material.betaM) *
		(material.betaM * material.betaM) * (material.betaM * material.betaM) * (material.betaM * material.betaM) * (material.betaM * material.betaM) * (material.betaM * material.betaM) *
		(material.betaM * material.betaM) * (material.betaM * material.betaM);

	dphi += (xi_N.x < apsample.x) ? Phi(0, gammaO, gammaT) : (xi_N.x < apsample.x + apsample.y) ? Phi(1, gammaO, gammaT) : Phi(2, gammaO, gammaT);
	float phiI = phio + dphi;

	//modèle non séparable

	float theta_0 = sin_theta_o - 2.f * sin2kAlpha0 * (cosf(phio * 0.5f) * cos2kAlpha0 * cos_theta_o + sin_theta_o * sin2kAlpha0);
	float sinThetaOp0 = sinf(theta_0);// sin_theta_o*cos2kAlpha1-cos_theta_o*sin2kAlpha1;
	float cosThetaOp0 = cosf(theta_0);//  cos_theta_o*cos2kAlpha1+sin_theta_o*sin2kAlpha1;

	//Modèle séparable
	//float sinThetaOp0 = sin_theta_o*cos2kAlpha1-cos_theta_o*sin2kAlpha1;
	//float cosThetaOp0 = cos_theta_o*cos2kAlpha1+sin_theta_o*sin2kAlpha1;


	float sinThetaOp1 = sin_theta_o * cos2kAlpha0 + cos_theta_o * sin2kAlpha0;
	float cosThetaOp1 = cos_theta_o * cos2kAlpha0 - sin_theta_o * sin2kAlpha0;
	float sinThetaOp2 = sin_theta_o * cos2kAlpha2 + cos_theta_o * sin2kAlpha2;
	float cosThetaOp2 = cos_theta_o * cos2kAlpha2 - sin_theta_o * sin2kAlpha2;

	float coef = (xi_N.x < apsample.x) ? 1.0f : (xi_N.x < apsample.x + apsample.y) ? 0.25f : 4.f;

	float cosTheta = 1.f + coef * sqrtv * sqrtv * logf(xi_M.x + (1.f - xi_M.x) * expf(-2.f / (coef * sqrtv * sqrtv)));

	float sinTheta = trigInverse(cosTheta);



	float sinThetaI = (xi_N.x < apsample.x) ? -cosTheta * sinThetaOp0 + sinTheta * cosPhi * cosThetaOp0
		: (xi_N.x < apsample.x + apsample.y) ? -cosTheta * sinThetaOp1 + sinTheta * cosPhi * cosThetaOp1
		: -cosTheta * sinThetaOp2 + sinTheta * cosPhi * cosThetaOp2;
	float cosThetaI = trigInverse(sinThetaI);


	//prd->wi = normalize(make_float3( sinThetaI, cosThetaI * cosf(phiI), cosThetaI * sinf(phiI)));
	prd->wi = normalize(make_float3(cosThetaI * sinf(phiI), sinThetaI, cosThetaI * cosf(phiI)));
	// Evaluate longitudinal scattering functions
	cosThetaOp0 = fabsf(cosThetaOp0);
	cosThetaOp1 = fabsf(cosThetaOp1);
	cosThetaOp2 = fabsf(cosThetaOp2);

	const float M_R = M(sqrtv * sqrtv, sinThetaI, sinThetaOp0, cosThetaI, cosThetaOp0);
	const float M_TT = M(0.25f * sqrtv * sqrtv, sinThetaI, sinThetaOp1, cosThetaI, cosThetaOp1);
	const float M_TRT = M(sqrtv * sqrtv * 4.f, sinThetaI, sinThetaOp2, cosThetaI, cosThetaOp2);
	//const float M_TRRT = M(sqrtv*sqrtv*4.f, sinThetaI, fabsf(sin_theta_o),cosThetaI, fabsf(cos_theta_o));

	const float N_R = Np(dphi, 0, s, gammaO, gammaT);//0.25f*abs(cosf(0.5f*(dphi)));//
	const float N_TT = Np(dphi, 1, s, gammaO, gammaT);
	const float N_TRT = Np(dphi, 2, s, gammaO, gammaT);


	prd->pdf = M_R* N_R* apsample.x + M_TT * N_TT * apsample.y + M_TRT * N_TRT * apsample.z;//+M_TRRT*apsample.w*0.5f*M_1_PIf;

	prd->wi = prd->wi.x* state.texcoord + prd->wi.y * state.tangent + prd->wi.z * normalize(cross(state.texcoord, state.tangent));

	
	if (dot(prd->wi, state.normal) < 0.f)
	{
		prd->pos = prd->pos - 2.f * state.normal * state.radius;
	}
	
	prd->f_over_pdf = (M_R * N_R * R + M_TT * N_TT * TT + N_TRT * M_TRT * TRT/* +M_TRRT*TRRT*0.5*M_1_PIf*/) / prd->pdf;

	prd->flags |= FLAG_DIFFUSE;
	
}


/* Azimuthal and logitudinal evaluation for given theta and phi */
extern "C" __device__ float4 __direct_callable__eval_bcsdf_hair(MaterialDefinition const& material, State const& state, PerRayData* const prd, const float3 wiL) {

	float ior = 1.55f;

	float h = -1.f + 2.f * rng(prd->seed);
	float gammaO = asinf(h);


	const float sin_theta_o = dot(prd->wo, state.tangent);
	const float cos_theta_o = trigInverse(sin_theta_o);
	const float ndwo = dot(prd->wo, state.texcoord);
	const float bdwo = dot(prd->wo, cross(state.texcoord, state.tangent));
	float phio = atan2f(ndwo, bdwo);//
	//float phio = asinf(dot(prd->wo, state.normal));

	const float sin_theta_i = dot(wiL, state.tangent);
	const float cos_theta_i = trigInverse(sin_theta_i);
	const float ndwi = dot(wiL, state.texcoord);
	const float bdwi = dot(wiL, cross(state.texcoord, state.tangent));
	float phii = atan2f(ndwi, bdwi);//
	//float phii = asinf(dot(wiL, state.normal));

	// Compute $\cos \thetat$ for refracted ray
	float sinThetaT = sin_theta_o / ior;
	float cosThetaT = trigInverse(sinThetaT);

	// Compute $\gammat$ for refracted ray
	float etap = sqrtf(ior * ior - (sin_theta_o * sin_theta_o)) / cos_theta_o;
	float sinGammaT = h / etap;
	float cosGammaT = trigInverse(sinGammaT);
	float gammaT = asinf(sinGammaT);

	float phi = phii - phio;

	float3 absorption = material.absorption;
	const float3 absorption_eumelanin = make_float3(0.419f, 0.697f, 1.37f);
	const float3 absorption_pheomelanin = make_float3(0.187f, 0.4f, 1.05f);

	float melanin_ratio = clamp(material.melanin_ratio * (1.f + material.melanin_ratio_disparity * state.rand.y), 0.f, 1.f);
	float melanin_concentration = fmaxf(material.melanin_concentration * (1.f + state.rand.y * material.melanin_concentration_disparity), .0f);
	absorption += (state.rand.x > material.whitepercen) ? melanin_concentration * lerp(absorption_eumelanin, absorption_pheomelanin, melanin_ratio) : make_float3(0.f);


	float3 T = expf(-absorption * 2.f * cosGammaT / cosThetaT);
	float3 R = ApR(h, cos_theta_o, etap);
	//float3 Rs = ApR(h, cos_theta_o,etas);
	//float3 R = 0.5f*(Rp+Rs);
	float3 TT = (make_float3(1.f) - R) * (make_float3(1.f) - R) * T;
	float3 TRT = TT * R * T;
	//float3 TRRT = TRT*T*R/(1.f-T*R);

	//float lum =  luminance(R)  +luminance(TT) + luminance(TRT)+luminance(TRRT) ;
	//float4 apsample =  make_float4(luminance(R),luminance(TT),luminance(TRT),luminance(TRRT))/lum;

	float lum = luminance(R) + luminance(TT) + luminance(TRT);
	float3 apsample = make_float3(luminance(R), luminance(TT), luminance(TRT)) / lum;

	float sin2kAlpha0 = sinf(material.scale_angle_rad);
	float cos2kAlpha0 = trigInverse(sin2kAlpha0);
	float sin2kAlpha1 = 2.f * cos2kAlpha0 * sin2kAlpha0;
	float cos2kAlpha1 = cos2kAlpha0 * cos2kAlpha0 - sin2kAlpha0 * sin2kAlpha0;
	float sin2kAlpha2 = 2.f * cos2kAlpha1 * sin2kAlpha1;
	float cos2kAlpha2 = cos2kAlpha1 * cos2kAlpha1 - sin2kAlpha1 * sin2kAlpha1;

	float theta_0 = sin_theta_o - 2.f * sin2kAlpha0 * (cosf(phio * 0.5f) * cos2kAlpha0 * cos_theta_o + sin_theta_o * sin2kAlpha0);
	float sinThetaOp0 = sinf(theta_0);// sin_theta_o*cos2kAlpha1-cos_theta_o*sin2kAlpha1;
	float cosThetaOp0 = cosf(theta_0);//  cos_theta_o*cos2kAlpha1+sin_theta_o*sin2kAlpha1;

	//float sinThetaOp0 = sin_theta_o*cos2kAlpha1-cos_theta_o*sin2kAlpha1;
	//float cosThetaOp0 = cos_theta_o*cos2kAlpha1+sin_theta_o*sin2kAlpha1;

	float sinThetaOp1 = sin_theta_o * cos2kAlpha0 + cos_theta_o * sin2kAlpha0;
	float cosThetaOp1 = cos_theta_o * cos2kAlpha0 - sin_theta_o * sin2kAlpha0;

	float sinThetaOp2 = sin_theta_o * cos2kAlpha2 + cos_theta_o * sin2kAlpha2;
	float cosThetaOp2 = cos_theta_o * cos2kAlpha2 - sin_theta_o * sin2kAlpha2;


	const float sqrtv = 0.726f * material.betaM + 0.812f * (material.betaM * material.betaM) + 3.7f * (material.betaM * material.betaM) * (material.betaM * material.betaM) * (material.betaM * material.betaM) *
		(material.betaM * material.betaM) * (material.betaM * material.betaM) * (material.betaM * material.betaM) * (material.betaM * material.betaM) * (material.betaM * material.betaM) *
		(material.betaM * material.betaM) * (material.betaM * material.betaM);

	const float s = 0.626657069f * (0.265f * material.betaN + 1.194f * (material.betaN * material.betaN) + 5.372f * (material.betaN * material.betaN) * (material.betaN * material.betaN) * (material.betaN * material.betaN) *
		(material.betaN * material.betaN) * (material.betaN * material.betaN) * (material.betaN * material.betaN) * (material.betaN * material.betaN) * (material.betaN * material.betaN) *
		(material.betaN * material.betaN) * (material.betaN * material.betaN) * (material.betaN * material.betaN));

	// Evaluate longitudinal scattering functions
	cosThetaOp0 = fabsf(cosThetaOp0);
	cosThetaOp1 = fabsf(cosThetaOp1);
	cosThetaOp2 = fabsf(cosThetaOp2);

	const float M_R = M(sqrtv * sqrtv, sin_theta_i, sinThetaOp0, cos_theta_i, cosThetaOp0);
	const float M_TT = M(0.25f * sqrtv * sqrtv, sin_theta_i, sinThetaOp1, cos_theta_i, cosThetaOp1);
	const float M_TRT = M(sqrtv * sqrtv * 4.f, sin_theta_i, sinThetaOp2, cos_theta_i, cosThetaOp2);
	//const float M_TRRT = M(sqrtv*sqrtv*4.f, sin_theta_i, sin_theta_o,cos_theta_i, cos_theta_o);

	const float N_R = Np(phi, 0.f, s, gammaO, gammaT);
	const float N_TT = Np(phi, 1.f, s, gammaO, gammaT);
	const float N_TRT = Np(phi, 2.f, s, gammaO, gammaT);

	const float3 f = M_R * N_R * R + M_TT * N_TT * TT + N_TRT * M_TRT * TRT;//+M_TRRT*TRRT*0.5*M_1_PIf;
	float pdf = M_R * N_R * apsample.x + M_TT * N_TT * apsample.y + M_TRT * N_TRT * apsample.z;//+M_TRRT*apsample.w*0.5f*M_1_PIf;
	return make_float4(f, pdf);
}