
/* ---------------------------------------------------------
 * Bidirectional curve scattering distribution function
 * ---------------------------------------------------------
 * Pre-computes the azimuthal scattering functions
 * Based on the hair rendering implementation from the
 * tungsten renderer for "energy-conserving hair reflectance
 * model" and importance sampling for physically-based hair
 * fiber models" from dEon et al.
 * --------------------------------------------------------- */

/* ----------------------- Libraries ----------------------- */

#include "inc/HairBCSDF.h"

#include "inc/GaussLegendre.h"

/* ---------------------- Hair BCSDF ----------------------- */

/* Constructors -------------------------------------------- */
HairBCSDF::HairBCSDF() : /*m_values_R(new float[181*181*101]),
m_values_TT(new float[181 * 181 * 101]),
m_values_TRT(new float[181 * 181 * 101]),*/
m_values_cop(new float3[181]), m_values_sop(new float3[181])
 //m_values_Fr(new float3[181*101]),
 //m_values_gammaO(new float[181  * 101]),
 //m_values_gammaT(new float[181  * 101]),
 //m_values_fracT(new float[181 * 101]),
 //m_values_N_R(new float[181 * 361 * 101 * 101]),
 //m_values_N_TT(new float[181 * 361 * 101 * 101]),
 //m_values_N_TRT(new float[181*361 * 101 * 101])
	//m_buffer_cdf(nullptr), m_buffer_pdf(nullptr), m_buffer_sum(nullptr)
{}

HairBCSDF::~HairBCSDF() {}

/* Evaluate a Fresnel dielectric function ------------------ */
/*static inline float evaluateDielectricFresnel(float eta,
	float cos_theta_i, float &cos_theta_t) {
	if (cos_theta_i < 0.0f) {
		eta = 1.0f / eta;
		cos_theta_i = -cos_theta_i;
	}
	float sin_theta_t_sq = eta*eta*(1.0f - cos_theta_i*cos_theta_i);
	if (sin_theta_t_sq > 1.0f) {
		cos_theta_t = 0.0f;
		return 1.0f;
	}
	cos_theta_t = sqrtf(max(1.0f - sin_theta_t_sq, 0.0f));

	float Rs = (eta*cos_theta_i - cos_theta_t) / (eta*cos_theta_i + cos_theta_t);
	float Rp = (eta*cos_theta_t - cos_theta_i) / (eta*cos_theta_t + cos_theta_i);

	return (Rs*Rs + Rp*Rp)*0.5f;
}

static inline float evaluateDielectricFresnel(float eta,
	float cos_theta_i) {
	float cos_theta_t;
	return evaluateDielectricFresnel(eta, cos_theta_i, cos_theta_t);
}

/* Standard normalized Gaussian ---------------------------- */
/*float HairBCSDF::g(float beta, float theta) {
	return exp(-theta*theta / (2.0f*beta*beta)) / (sqrt(2.0f*M_PIf)*beta);
}

float HairBCSDF::Logistic(float x, float s) {
	x = std::abs(x);
	return std::exp(-x / s) / (s * (1 + std::exp(-x / s))*(1 + std::exp(-x / s)));
}

float HairBCSDF::LogisticCDF(float x, float s) {
	return 1 / (1 + std::exp(-x / s));
}

float HairBCSDF::TrimmedLogistic(float x, float s, float a, float b) {
	return Logistic(x, s) / (LogisticCDF(b, s) - LogisticCDF(a, s));
}

/* Wrapped Gaussina "deterctor", computed as an infinite sum of
 * Gaussians (approximiated with a finite sum) ------------- */
/*float HairBCSDF::D(float beta, float phi) {
	float result = 0.0f;
	float delta;
	float shift = 0.0f;
	do {
		delta = g(beta, phi + shift) + g(beta, phi - shift - M_2PIf);
		result += delta;
		shift += M_2PIf;
	} while (delta > 1e-4f);
	return result;
}

/* Computes the existant azimuthal angle of the p'th
 * perfect specular scattering event, derived using
 * Bravais theory ------------------------------------------ */
/*float HairBCSDF::Phi(float gamma_i, float gamma_t, int p) {
	return 2.0f*p*gamma_t - 2.0f*gamma_i + p*M_PIf;
}
*/

float HairBCSDF::I0(float const& x) {
	float result = 1.0f;
	const float x_sq = x*x;
	float xi = x_sq;
	float denom = 4.0f;
	for (int i = 1; i <= 10; ++i) {
		result += xi / denom;
		xi *= x_sq;
		denom *= 4.0f*float((i + 1)*(i + 1));
	}
	return result;
}

float HairBCSDF::logI0(float const& x) {
		return (x > 12.0f)? x + 0.5f*(-logf( (M_2PIf*x)) + 1.0f / (8.0f*x)) : logf(I0(x));
}


/* Rough longitudinal scattering function with variance
* v = beta^2 ---------------------------------------------- */
float HairBCSDF::M(float const& v, float const& sin_theta_i,
	float const& sin_theta_o, float const& cos_theta_i,
	float const& cos_theta_o) {
	float a = cos_theta_i * cos_theta_o / v;
	float b = sin_theta_i * sin_theta_o / v;
	float mp =
		(v <= 0.1f)
		? (expf(logI0(a) - b - 1.f / v + 0.6931f + logf(1.f / (2.f * v))))
		: (expf(-b) * I0(a)) / (sinhf(1.f / v) * 2.f * v);
	return mp;
}

float HairBCSDF::FrDielectric(float const& cos_theta, float const& n1,
	float const& n2){

	float const R0 = ((n1 - n2) / (n1 + n2))*((n1 - n2) / (n1 + n2));
	return R0 + (1.f - R0)*(1.f - cos_theta)*(1.f - cos_theta)*(1.f - cos_theta)*(1.f - cos_theta)*(1.f - cos_theta);
}



float3 HairBCSDF::ApR(float const& h, float const& cos_theta_o,
	float const& eta){

	float cosgamma0 = sqrtf(1.f - h*h);
	float cosTheta = cos_theta_o * cosgamma0;
	float f = FrDielectric(cosTheta, 1.f, eta);
	return make_float3(f);
}

float HairBCSDF::Logistic(float const& x, float const& s){
	float ax = fabsf(x);
	return expf(-ax / s) / (s * (1.f + expf(-ax / s))*(1 + expf(-ax / s)));
}

float HairBCSDF::LogisticCDF(float const& x, float const& s) {
	return 1.f / (1.f + expf(-x / s));
}

float HairBCSDF::TrimmedLogistic(float const& x, float const& s, float const& a, float const& b) {
	return Logistic(x, s) / (LogisticCDF(b, s) - LogisticCDF(a, s));
}

float HairBCSDF::SampleTrimmedLogistic(float const& u, float const& s, float const& a, float const& b) {
	float k = LogisticCDF(b, s) - LogisticCDF(a, s);
	float x = -s * logf(1.f / (u * k + LogisticCDF(a, s)) - 1.f);
	return clamp(x, a, b);
}

float HairBCSDF::Phi(int const& p, float const& gammat,
	float const& gamma0){

	return 2.f * p * gammat - 2.f * gamma0 + p * M_PIf;
}


float HairBCSDF::Np(float const& phi,
	int const& p, float const& s, float const& gammaO, float const& gammaT) {
	float dphi = phi - Phi(p, gammaT, gammaO);
	// Remap _dphi_ to $[-\pi,\pi]$
	while (dphi > M_PIf) { dphi -= 2.f * M_PIf; }
	while (dphi < -M_PIf) { dphi += 2.f * M_PIf; }
	return TrimmedLogistic(dphi, s, -M_PIf, M_PIf);
}

/* Azimuthal PDF and CDF precomputation-------------------*/
void HairBCSDF::calculateFunctions(Context context, float scale_angle_rad) {

	float sin2kAlpha0 = sinf(scale_angle_rad);
	float cos2kAlpha0 = trigInverse(sin2kAlpha0);
	float sin2kAlpha1 = 2.f * cos2kAlpha0 * sin2kAlpha0;
	float cos2kAlpha1 = cos2kAlpha0*cos2kAlpha0 - sin2kAlpha0*sin2kAlpha0;
	float sin2kAlpha2 = 2.f * cos2kAlpha1 * sin2kAlpha1;
	float cos2kAlpha2 = cos2kAlpha1*cos2kAlpha1 - sin2kAlpha1*sin2kAlpha1;

	for (int thetao = -90; thetao < 91; ++thetao){
		float sin_theta_o = sinf(thetao*M_PIf / 180.f);
		float cos_theta_o = trigInverse(sin_theta_o);

		float sinThetaOp0 = sin_theta_o*cos2kAlpha1 - cos_theta_o*sin2kAlpha1;
		float cosThetaOp0 = cos_theta_o*cos2kAlpha1 + sin_theta_o*sin2kAlpha1;
		float sinThetaOp1 = sin_theta_o*cos2kAlpha0 + cos_theta_o*sin2kAlpha0;
		float cosThetaOp1 = cos_theta_o*cos2kAlpha0 - sin_theta_o*sin2kAlpha0;
		float sinThetaOp2 = sin_theta_o*cos2kAlpha2 + cos_theta_o*sin2kAlpha2;
		float cosThetaOp2 = cos_theta_o*cos2kAlpha2 - sin_theta_o*sin2kAlpha2;

		m_values_cop[(thetao + 90)] = make_float3(cosThetaOp0, cosThetaOp1, cosThetaOp2);
		m_values_sop[(thetao + 90)] = make_float3(sinThetaOp0, sinThetaOp1, sinThetaOp2);


		/*cosThetaOp0 = fabsf(cosThetaOp0);
		cosThetaOp1 = fabsf(cosThetaOp1);
		cosThetaOp2 = fabsf(cosThetaOp2);

		for (int thetai = -90; thetai < 91; ++thetai){
			float sin_theta_i = sinf(thetai*M_PIf / 180.f);
			float cos_theta_i = trigInverse(sin_theta_i);

			for (int beta_m = 0; beta_m < 101; ++beta_m){
				float beta = beta_m / 100.f;
				float sqrtv = 0.726f*beta + 0.812f*(beta*beta) + 3.7f*(beta*beta)*(beta*beta)*(beta*beta)*
					(beta*beta)*(beta*beta)*(beta*beta)*(beta*beta)*(beta*beta)*
					(beta*beta)*(beta*beta);

				float M_R = M(sqrtv*sqrtv, sin_theta_i, sinThetaOp0,
				cos_theta_i, cosThetaOp0);
				float M_TT = M(0.25f*sqrtv*sqrtv, sin_theta_i, sinThetaOp1,
				cos_theta_i, cosThetaOp1);
				float M_TRT = M(sqrtv*sqrtv*4.f, sin_theta_i, sinThetaOp2,
				cos_theta_i, cosThetaOp2);

				m_values_R[101 * 181 * (thetao+90) + 101 * (thetai+90) + beta_m] = M_R;
				m_values_TT[101 * 181 * (thetao + 90) + 101 * (thetai + 90) + beta_m] = M_TT;
				m_values_TRT[101 * 181 * (thetao + 90) + 101 * (thetai + 90) + beta_m] = M_TRT;

			}
		}

		for (int ha = 0; ha < 101; ++ha){
			const float h = 2.f*ha*0.01f - 1.f;
			float sinThetaT = sin_theta_o / 1.55f;
			float cosThetaT = trigInverse(sinThetaT);
			float etap = sqrtf(1.55f * 1.55f - (sin_theta_o*sin_theta_o)) / cos_theta_o;
			float etas = 1.55f * 1.55f *cos_theta_o / sqrtf(1.55f * 1.55f - (sin_theta_o*sin_theta_o));
			float sinGammaT = h / etap;
			float cosGammaT = trigInverse(sinGammaT);
			float gammaT = asinf(sinGammaT);
			float gammaO = asinf(h);
			float3 Rp = ApR(h, cos_theta_o, etap);
			float3 Rs = ApR(h, cos_theta_o, etas);
			float3 R = 0.5f*(Rp + Rs);

			/*float betaN = 0.9f;
			float s = 0.626657069f*(0.265f*betaN + 1.194f*(betaN*betaN) + 5.372f*(betaN*betaN)*(betaN*betaN)*(betaN*betaN)*
				(betaN*betaN)*(betaN*betaN)*(betaN*betaN)*(betaN*betaN)*(betaN*betaN)*
				(betaN*betaN)*(betaN*betaN)*(betaN*betaN));

			cout << s << endl;*/

			/*m_values_Fr[101 * (thetao + 90) + ha] = R;
			m_values_gammaO[101 * (thetao + 90) + ha] = gammaO;
			m_values_gammaT[101 * (thetao + 90) + ha] = gammaT;

			m_values_fracT[101 * (thetao + 90) + ha] = cosGammaT / cosThetaT;*/
			

			/*for (int betan = 0; betan < 101; ++betan){
				float betaN = float(betan) / 100.f;
				float s = 0.626657069f*(0.265f*betaN + 1.194f*(betaN*betaN) + 5.372f*(betaN*betaN)*(betaN*betaN)*(betaN*betaN)*
					(betaN*betaN)*(betaN*betaN)*(betaN*betaN)*(betaN*betaN)*(betaN*betaN)*
					(betaN*betaN)*(betaN*betaN)*(betaN*betaN));

			}*/



			//float3 T = expf(-parameters.absorption*2.f*cosGammaT / cosThetaT);
			//float3 TT = (make_float3(1.f) - R)*(make_float3(1.f) - R)*T;
			//float3 TRT = TT*R*T;

			//float lum = Luminance(R) + Luminance(TT) + Luminance(TRT);
			//return make_float3(Luminance(R) / lum, Luminance(TT) / lum, Luminance(TRT) / lum);




		//
//	}
	



}





	void* buf;

	/*m_buffer_values_R = context->createBuffer(RT_BUFFER_INPUT,
		RT_FORMAT_FLOAT, 181 * 181 * 101);
	buf = m_buffer_values_R->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	memcpy(buf, m_values_R, sizeof(float) * 181 * 181 * 101);
	m_buffer_values_R->unmap();

	m_buffer_values_TT = context->createBuffer(RT_BUFFER_INPUT,
		RT_FORMAT_FLOAT, 181 * 181 * 101);
	buf = m_buffer_values_TT->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	memcpy(buf, m_values_TT, sizeof(float) * 181 * 181 * 101);
	m_buffer_values_TT->unmap();

	m_buffer_values_TRT = context->createBuffer(RT_BUFFER_INPUT,
		RT_FORMAT_FLOAT, 181 * 181 * 101);
	buf = m_buffer_values_TRT->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	memcpy(buf, m_values_TRT, sizeof(float) * 181 * 181 * 101);
	m_buffer_values_TRT->unmap();*/

	m_buffer_values_cop = context->createBuffer(RT_BUFFER_INPUT,
		RT_FORMAT_FLOAT3, 181);
	buf = m_buffer_values_cop->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	memcpy(buf, m_values_cop, sizeof(float3) * 181);
	m_buffer_values_cop->unmap();

	m_buffer_values_sop = context->createBuffer(RT_BUFFER_INPUT,
		RT_FORMAT_FLOAT3, 181);
	buf = m_buffer_values_sop->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	memcpy(buf, m_values_sop, sizeof(float3) * 181);
	m_buffer_values_sop->unmap();

	/*m_buffer_values_N_R = context->createBuffer(RT_BUFFER_INPUT,
		RT_FORMAT_FLOAT, 181*361*101*101);
	buf = m_buffer_values_N_R->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	memcpy(buf, m_values_N_R, sizeof(float) * 181 * 361 * 101 * 101);
	m_buffer_values_N_R->unmap();

	m_buffer_values_N_TT = context->createBuffer(RT_BUFFER_INPUT,
		RT_FORMAT_FLOAT, 181 * 361 * 101 * 101);
	buf = m_buffer_values_N_TT->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	memcpy(buf, m_values_N_TT, sizeof(float) * 181 * 361 * 101 * 101);
	m_buffer_values_N_TT->unmap();

	m_buffer_values_N_TRT = context->createBuffer(RT_BUFFER_INPUT,
		RT_FORMAT_FLOAT, 181 * 361 * 101 * 101);
	buf = m_buffer_values_N_TT->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	memcpy(buf, m_values_N_TRT, sizeof(float) * 181 * 361 * 101 * 101);
	m_buffer_values_N_TT->unmap();*/

//	Buffer m_buffer_values_gammaO;
//	Buffer m_buffer_values_gammaT;
//	Buffer m_buffer_values_fracT;
//	Buffer m_buffer_values_Fr;

	}

/*
	GaussLegendre<NUM_POINTS> integrator;
	const auto points = integrator.getPoints();
	const auto weights = integrator.getWeights();

	// Cache the gamma across all integration points
	array<float, NUM_POINTS> gamma_is;
	for (int i = 0; i < NUM_POINTS; ++i) {
		gamma_is[i] = asin(points[i]);
	}

	// Precompute the Gaussian detector and sample it into an OptiX buffer,
	// this is the only part of the precomputation that is actualy approximate
	// 2048 samples are enough to support the lowest roughness that the BCSDF
	// can reliably simulate
	float3 *Ds = new float3[num_gaussian_samples];
	for (int i = 0; i < num_gaussian_samples; ++i) {
		Ds[i].x = D(beta.x, i / (num_gaussian_samples - 1.0f)*M_2PIf);
		Ds[i].y = D(beta.y, i / (num_gaussian_samples - 1.0f)*M_2PIf);
		Ds[i].z = D(beta.z, i / (num_gaussian_samples - 1.0f)*M_2PIf);
	}

	// Simple wrapped linear interpolation of the precomputed table
	auto approxD = [&](int p, float phi) {
		float u = fabsf(phi*((0.5f*M_1_PI)*(num_gaussian_samples - 1)));
		int x_0 = int(u);
		int x_1 = x_0 + 1;
		u -= x_0;
		switch (p) {
		case 0:
		default:
			return Ds[x_0 % num_gaussian_samples].x*(1.0f - u) +
				Ds[x_1 % num_gaussian_samples].x*u;
			break;
		case 1:
			return Ds[x_0 % num_gaussian_samples].y*(1.0f - u) +
				Ds[x_1 % num_gaussian_samples].y*u;
			break;
		case 2:
			return Ds[x_0 % num_gaussian_samples].z*(1.0f - u) +
				Ds[x_1 % num_gaussian_samples].z*u;
			break;
		}
	};

	// Here follows the actual precomputation of the azimuthal scattering
	// functions, the scattering functions are parametrized with the 
	// azimuthal angle phi, and the cosine of the half angle, cos(theta_d)
	// This parametrization makes the azimuthal function relatively smooth
	// and allows using really low resolution for the buffer (64x64 in this
	// case) withouth any visual deviation from ground truth, even at the
	// lowest supported roughness setting
	array<float, 4096> pdf_R, pdf_TT, pdf_TRT;
	for (int y = 0; y < resolution; ++y) {
		float cos_half_angle = y / (resolution - 1.0f);

		// Precompute reflection Fresnel factor and reduced absorption coefficient
		float ior_prime = sqrt(ior*ior -
			(1.0f - cos_half_angle*cos_half_angle)) / cos_half_angle;
		float cos_theta_t = sqrt(1.0f - (1.0f - cos_half_angle*cos_half_angle)*sqr(1.0f / ior));
		float3 sigma_a_prime = absorption / cos_theta_t;

		// Precumpute gamma_t, f_t and internal absorption across all integration
		// points
		array<float, NUM_POINTS> fresnel_terms, gamma_ts;
		array<float3, NUM_POINTS> absorptions;
		for (int i = 0; i < NUM_POINTS; ++i) {
			gamma_ts[i] = asinf(clamp(points[i] / ior_prime, -1.0f, 1.0f));
			fresnel_terms[i] = evaluateDielectricFresnel(
				1.0f / ior, cos_half_angle*cosf(gamma_is[i]));
			absorptions[i] = expf(-sigma_a_prime*2.0f*(1 + cosf(2.0*gamma_ts[i])));
		}

		for (int phi_i = 0; phi_i < resolution; ++phi_i) {
			float phi = M_2PIf*phi_i / (resolution - 1.0f);

			float integral_R = 0.0f;
			float integral_TT = 0.0f;
			float integral_TRT = 0.0f;

			// Here follows the integration across the fiber width (h)
			for (int i = 0; i < integrator.numSamples(); ++i) {
				float f_R = fresnel_terms[i];
				float3 T = absorptions[i];

				float AR = f_R;
				float3 ATT = (1.0f - f_R)*(1.0f - f_R)*T;
				float3 ATRT = ATT*f_R*T;

				integral_R += weights[i] * approxD(0, phi - Phi(gamma_is[i], gamma_ts[i], 0));
				integral_TT += weights[i] * approxD(1, phi - Phi(gamma_is[i], gamma_ts[i], 1));
				integral_TRT += weights[i] * approxD(2, phi - Phi(gamma_is[i], gamma_ts[i], 2));
			}

			pdf_R[phi_i + y*resolution] = 0.5f*integral_R;
			pdf_TT[phi_i + y*resolution] = 0.5f*integral_TT;
			pdf_TRT[phi_i + y*resolution] = 0.5f*integral_TRT;
		}
	}

//----------------------------------------------------
	float3 *pdfs = new float3[resolution*resolution];
	float3 *cdfs = new float3[(resolution + 1)*resolution];
	float3 *sums = new float3[resolution];
	
	// Calculate pdfs values
	for (int i = 0; i < resolution*resolution; ++i) {
		pdfs[i].x = pdf_R[i];
		pdfs[i].y = pdf_TT[i];
		pdfs[i].z = pdf_TRT[i];
	}

	// Dilate weights slightly to stay conservative
	for (int y = 0; y < resolution; ++y) {
		for (int x = 0; x < resolution - 1; ++x) {
			pdfs[x + y*resolution] = fmaxf(pdfs[x + y*resolution],
				pdfs[x + 1 + y*resolution]);
		}
		for (int x = resolution - 1; x > 0; --x) {
			pdfs[x + y*resolution] = fmaxf(pdfs[x + y*resolution],
				pdfs[x - 1 + y*resolution]);
		}
	}

	for (int x = 0; x < resolution; ++x) {
		for (int y = 0; y < resolution - 1; ++y) {
			pdfs[x + y*resolution] = fmaxf(pdfs[x + y*resolution],
				pdfs[x + (y + 1)*resolution]);
		}
		for (int y = resolution - 1; y > 0; --y) {
			pdfs[x + y*resolution] = fmaxf(pdfs[x + y*resolution],
				pdfs[x + (y - 1)*resolution]);
		}
	}

	// Calculate cdfs and sums values
	float ratio = 1.0f / resolution;
	for (int dist = 0; dist < resolution; ++dist) {
		cdfs[0 + dist*(resolution + 1)] = make_float3(0.0f);
		for (int x = 0; x < resolution; ++x) {
			cdfs[(x + 1) + dist*(resolution + 1)] = pdfs[x + dist*resolution] +
				cdfs[x + dist*(resolution + 1)];
		}

		sums[dist] = cdfs[resolution + dist*(resolution + 1)];
		float3 scale = make_float3(1.0f) / sums[dist];
		// For R
		if (sums[dist].x < 1e-4f) {
			for (int x = 0; x < resolution; ++x) {
				pdfs[x + dist*resolution].x = ratio;
				cdfs[x + dist*(resolution + 1)].x = x*ratio;
			}
		}
		else {
			for (int x = 0; x < resolution; ++x) {
				pdfs[x + dist*resolution].x *= scale.x;
				cdfs[x + dist*(resolution + 1)].x *= scale.x;
			}
		}
		// For TT
		if (sums[dist].y < 1e-4f) {
			for (int x = 0; x < resolution; ++x) {
				pdfs[x + dist*resolution].y = ratio;
				cdfs[x + dist*(resolution + 1)].y = x*ratio;
			}
		}
		else {
			for (int x = 0; x < resolution; ++x) {
				pdfs[x + dist*resolution].y *= scale.y;
				cdfs[x + dist*(resolution + 1)].y *= scale.y;
			}
		}
		// For TRT
		if (sums[dist].z < 1e-4f) {
			for (int x = 0; x < resolution; ++x) {
				pdfs[x + dist*resolution].z = ratio;
				cdfs[x + dist*(resolution + 1)].z = x*ratio;
			}
		}
		else {
			for (int x = 0; x < resolution; ++x) {
				pdfs[x + dist*resolution].z *= scale.z;
				cdfs[x + dist*(resolution + 1)].z *= scale.z;
			}
		}
		cdfs[resolution + dist*(resolution + 1)] = make_float3(1.0f);
	}
	
	void* buf;

	m_buffer_values_R = context->createBuffer(RT_BUFFER_INPUT,
		RT_FORMAT_FLOAT3, resolution*resolution);
	buf = m_buffer_values_R->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	memcpy(buf, m_values_R, sizeof(float3)*resolution*resolution);
	m_buffer_values_R->unmap();
	
	m_buffer_values_TT = context->createBuffer(RT_BUFFER_INPUT,
		RT_FORMAT_FLOAT3, resolution*resolution);
	buf = m_buffer_values_TT->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	memcpy(buf, m_values_TT, sizeof(float3)*resolution*resolution);
	m_buffer_values_TT->unmap();

	m_buffer_values_TRT = context->createBuffer(RT_BUFFER_INPUT,
		RT_FORMAT_FLOAT3, resolution*resolution);
	buf = m_buffer_values_TRT->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	memcpy(buf, m_values_TRT, sizeof(float3)*resolution*resolution);
	m_buffer_values_TRT->unmap();

	m_buffer_pdf = context->createBuffer(RT_BUFFER_INPUT,
		RT_FORMAT_FLOAT3, resolution*resolution);
	buf = m_buffer_pdf->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	memcpy(buf, pdfs, sizeof(float3)*resolution*resolution);
	m_buffer_pdf->unmap();

	m_buffer_cdf = context->createBuffer(RT_BUFFER_INPUT,
		RT_FORMAT_FLOAT3, (resolution+1)*resolution);
	buf = m_buffer_cdf->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	memcpy(buf, cdfs, sizeof(float3)*(resolution+1)*resolution);
	m_buffer_cdf->unmap();

	m_buffer_sum = context->createBuffer(RT_BUFFER_INPUT,
		RT_FORMAT_FLOAT3, resolution);
	buf = m_buffer_sum->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	memcpy(buf, sums, sizeof(float3)*resolution);
	m_buffer_sum->unmap();
}

/* Gets ---------------------------------------------------- */
Buffer HairBCSDF::getBufferValuesR() const {
	return m_buffer_values_R;
}

Buffer HairBCSDF::getBufferValuesTT() const {
	return m_buffer_values_TT;
}
Buffer HairBCSDF::getBufferValuesTRT() const {
	return m_buffer_values_TRT;
}
Buffer HairBCSDF::getBufferValuessop() const {
	return m_buffer_values_sop;
}
Buffer HairBCSDF::getBufferValuescop() const {
	return m_buffer_values_cop;
}
/*
Buffer HairBCSDF::getBufferCDF() const {
	return m_buffer_cdf;
}
Buffer HairBCSDF::getBufferPDF() const {
	return m_buffer_pdf;
}
Buffer HairBCSDF::getBufferSum() const {
	return m_buffer_sum;
}

int HairBCSDF::getResolution() const {
	return resolution;
}*/
