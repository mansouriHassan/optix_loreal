/* 
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "config.h"

#include <optix.h>

#include "system_data.h"
#include "per_ray_data.h"
#include "vertex_attributes.h"
#include "function_indices.h"
#include "material_definition.h"
#include "light_definition.h"
#include "shader_common.h"
#include "random_number_generators.h"


extern "C" __constant__ SystemData sysData;


// Get the 3x4 object to world transform and its inverse from a two-level hierarchy.
__forceinline__ __device__ void getTransforms(float4* mW, float4* mO) 
{
  OptixTraversableHandle handle = optixGetTransformListHandle(0);
  
  const float4* tW = optixGetInstanceTransformFromHandle(handle);
  const float4* tO = optixGetInstanceInverseTransformFromHandle(handle);

  mW[0] = tW[0];
  mW[1] = tW[1];
  mW[2] = tW[2];

  mO[0] = tO[0];
  mO[1] = tO[1];
  mO[2] = tO[2];
}

// Functions to get the individual transforms in case only one of them is needed.

__forceinline__ __device__ void getTransformObjectToWorld(float4* mW) 
{
  OptixTraversableHandle handle = optixGetTransformListHandle(0);
  
  const float4* tW = optixGetInstanceTransformFromHandle(handle);

  mW[0] = tW[0];
  mW[1] = tW[1];
  mW[2] = tW[2];
}

__forceinline__ __device__ void getTransformWorldToObject(float4* mO) 
{
  OptixTraversableHandle handle = optixGetTransformListHandle(0);
  
  const float4* tO = optixGetInstanceInverseTransformFromHandle(handle);

  mO[0] = tO[0];
  mO[1] = tO[1];
  mO[2] = tO[2];
}


// Matrix3x4 * point. v.w == 1.0f
__forceinline__ __device__ float3 transformPoint(const float4* m, const float3& v)
{
  float3 r;

  r.x = m[0].x * v.x + m[0].y * v.y + m[0].z * v.z + m[0].w;
  r.y = m[1].x * v.x + m[1].y * v.y + m[1].z * v.z + m[1].w;
  r.z = m[2].x * v.x + m[2].y * v.y + m[2].z * v.z + m[2].w;

  return r;
}

// Matrix3x4 * vector. v.w == 0.0f
__forceinline__ __device__ float3 transformVector(const float4* m, const float3& v)
{
  float3 r;

  r.x = m[0].x * v.x + m[0].y * v.y + m[0].z * v.z;
  r.y = m[1].x * v.x + m[1].y * v.y + m[1].z * v.z;
  r.z = m[2].x * v.x + m[2].y * v.y + m[2].z * v.z;

  return r;
}

// InverseMatrix3x4^T * normal. v.w == 0.0f
// Get the inverse matrix as input and applies it as inverse transpose.
__forceinline__ __device__ float3 transformNormal(const float4* m, const float3& v)
{
  float3 r;

  r.x = m[0].x * v.x + m[1].x * v.y + m[2].x * v.z;
  r.y = m[0].y * v.x + m[1].y * v.y + m[2].y * v.z;
  r.z = m[0].z * v.x + m[1].z * v.y + m[2].z * v.z;

  return r;
}


extern "C" __global__ void __closesthit__radiance()
{
  GeometryInstanceData* theData = reinterpret_cast<GeometryInstanceData*>(optixGetSbtDataPointer());

  // Cast the CUdeviceptr to the actual format for Triangles geometry.
  const unsigned int thePrimitiveIndex = optixGetPrimitiveIndex();

  const uint3* indices = reinterpret_cast<uint3*>(theData->indices);
  const uint3  tri     = indices[thePrimitiveIndex];

  const TriangleAttributes* attributes = reinterpret_cast<TriangleAttributes*>(theData->attributes);

  const TriangleAttributes& attr0 = attributes[tri.x];
  const TriangleAttributes& attr1 = attributes[tri.y];
  const TriangleAttributes& attr2 = attributes[tri.z];

  const float2 theBarycentrics = optixGetTriangleBarycentrics(); // beta and gamma
  const float  alpha = 1.0f - theBarycentrics.x - theBarycentrics.y;

  const float3 ng = cross(attr1.vertex - attr0.vertex, attr2.vertex - attr0.vertex);
  const float3 tg = attr0.tangent * alpha + attr1.tangent * theBarycentrics.x + attr2.tangent * theBarycentrics.y;
  const float3 ns = attr0.normal  * alpha + attr1.normal  * theBarycentrics.x + attr2.normal  * theBarycentrics.y;
  
  // DAR PERF This State lies in memory. It's more efficient to hold the data in registers.
  //          Problem is that more advanced material systems need the State all the time.
  State state; // All in world space coordinates!

  state.texcoord = attr0.texcoord * alpha + attr1.texcoord * theBarycentrics.x + attr2.texcoord * theBarycentrics.y;

  float4 objectToWorld[3];
  float4 worldToObject[3];
  
  getTransforms(objectToWorld, worldToObject);
  
  state.normalGeo = normalize(transformNormal(worldToObject, ng));
  state.tangent   = normalize(transformVector(objectToWorld, tg));
  state.normal    = normalize(transformNormal(worldToObject, ns));

  // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  thePrd->distance = optixGetRayTmax(); // Return the current path segment distance, needed for absorption calculations in the integrator.
  
  //thePrd->pos = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
  thePrd->pos += thePrd->wi * thePrd->distance; // DEBUG Check which version is more efficient.

  // Explicitly include edge-on cases as frontface condition!
  // Keeps the material stack from overflowing at silhouettes.
  // Prevents that silhouettes of thin-walled materials use the backface material.
  // Using the true geometry normal attribute as originally defined on the frontface!
  thePrd->flags |= (0.0f <= dot(thePrd->wo, state.normalGeo)) ? FLAG_FRONTFACE : 0;

  if ((thePrd->flags & FLAG_FRONTFACE) == 0) // Looking at the backface?
  {
    // Means geometric normal and shading normal are always defined on the side currently looked at.
    // This gives the backfaces of opaque BSDFs a defined result.
    state.normalGeo = -state.normalGeo;
    //state.tangent   = -state.tangent;
    state.normal    = -state.normal;
    // Explicitly DO NOT recalculate the frontface condition!
  }
  
  thePrd->radiance = make_float3(0.0f);

  // When hitting a geometric light, evaluate the emission first, because this needs the previous diffuse hit's pdf.
  const int idLight = theData->idLight;

  if (0 <= idLight && (thePrd->flags & FLAG_FRONTFACE)) // This material is emissive and we're looking at the front face.
  {
    const float cosTheta = dot(thePrd->wo, state.normalGeo);
    if (DENOMINATOR_EPSILON < cosTheta)
    {
      const LightDefinition& light = sysData.lightDefinitions[idLight];

      float3 emission = light.emission;

#if USE_NEXT_EVENT_ESTIMATION
      const float lightPdf = (thePrd->distance * thePrd->distance) / (light.area * cosTheta); // This assumes the light.area is greater than zero.

      // If it's an implicit light hit from a diffuse scattering event and the light emission was not returning a zero pdf (e.g. backface or edge on).
      if ((thePrd->flags & FLAG_DIFFUSE) && DENOMINATOR_EPSILON < lightPdf)
      {
        // The light.areaId buffer and light.area are always present on mesh lights.
        //pdf *= light.areaId[thePrimitiveIndex] / light.area; // This is not needed for parallelogram lights, the primitive area is the light area.

        // Scale the emission with the power heuristic between the initial BSDF sample pdf and this implicit light sample pdf.
        emission *= powerHeuristic(thePrd->pdf, lightPdf);
      }
#endif // USE_NEXT_EVENT_ESTIMATION

      thePrd->radiance = emission;
      
      // PERF End the path when hitting a light. Emissive materials with a non-black BSDF would normally just continue.
      thePrd->flags |= FLAG_TERMINATE;
      return;
    }
  }

  // Start fresh with the next BSDF sample. (Either of these values remaining zero is an end-of-path condition.)
  // The pdf of the previous evene was needed for the emission calculation above.
  thePrd->f_over_pdf = make_float3(0.0f);
  thePrd->pdf        = 0.0f;

  const MaterialDefinition& material = sysData.materialDefinitions[theData->idMaterial];

  state.albedo = material.albedo;

  if (material.textureAlbedo != 0)
  {
    const float3 texColor = make_float3(tex2D<float4>(material.textureAlbedo, state.texcoord.x, state.texcoord.y));

    // Modulate the incoming color with the texture.
    state.albedo *= texColor;               // linear color, resp. if the texture has been uint8 and readmode set to use sRGB, then sRGB.
    //state.albedo *= powf(texColor, 2.2f); // sRGB gamma correction done manually.
  }
 
  // Only the last diffuse hit is tracked for multiple importance sampling of implicit light hits.
  thePrd->flags = (thePrd->flags & ~FLAG_DIFFUSE) | FLAG_HIT | material.flags; // FLAG_THINWALLED can be set directly from the material.

  // Sample a new path direction. 
  const int indexBSDF = NUM_LENS_SHADERS + NUM_LIGHT_TYPES + material.indexBSDF * 2;

  optixDirectCall<void, const MaterialDefinition&, const State&, PerRayData*>(indexBSDF, material, state, thePrd);

#if USE_NEXT_EVENT_ESTIMATION
  // Direct lighting if the sampled BSDF was diffuse and any light is in the scene.
  const int numLights = sysData.numLights;
  if ((thePrd->flags & FLAG_DIFFUSE) && 0 < numLights)
  {
    // Sample one of many lights. 
    const float2 sample = rng2(thePrd->seed); // Use lower dimension samples for the position. (Irrelevant for the LCG).
   
    // The caller picks the light to sample. Make sure the index stays in the bounds of the sysData.lightDefinitions array.
    const int indexLight = (1 < numLights) ? clamp(static_cast<int>(floorf(rng(thePrd->seed) * numLights)), 0, numLights - 1) : 0;
    
    const LightDefinition& light = sysData.lightDefinitions[indexLight];
    
    const int indexCallable = NUM_LENS_SHADERS + light.type;

    LightSample lightSample = optixDirectCall<LightSample, const LightDefinition&, const float3, const float2>(indexCallable, light, thePrd->pos, sample);

    if (0.0f < lightSample.pdf) // Useful light sample?
    {
      // Evaluate the BSDF in the light sample direction. Normally cheaper than shooting rays.
      // Returns BSDF f in .xyz and the BSDF pdf in .w
      // BSDF eval function is one index after the sample fucntion.
      const float4 bsdf_pdf = optixDirectCall<float4, const MaterialDefinition&, const State&, const PerRayData* const, const float3>(indexBSDF + 1, material, state, thePrd, lightSample.direction);

      if (0.0f < bsdf_pdf.w && isNotNull(make_float3(bsdf_pdf)))
      {
        // Pass the current payload registers through to the shadow ray.
        unsigned int p0 = optixGetPayload_0();
        unsigned int p1 = optixGetPayload_1();

        // Note that the sysData.sceneEpsilon is applied on both sides of the shadow ray [t_min, t_max] interval 
        // to prevent self-intersections with the actual light geometry in the scene.
        optixTrace(sysData.topObject,
                   thePrd->pos, lightSample.direction, // origin, direction
                   sysData.sceneEpsilon, lightSample.distance - sysData.sceneEpsilon, 0.0f, // tmin, tmax, time
                   OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, // The shadow ray type only uses anyhit programs.
                   RAYTYPE_SHADOW, NUM_RAYTYPES, RAYTYPE_SHADOW,
                   p0, p1); // Pass through thePrd to the shadow ray. It needs the seed and sets flags.

        if ((thePrd->flags & FLAG_SHADOW) == 0) // Shadow flag not set?
        {
          if (thePrd->flags & FLAG_VOLUME) // Supporting nested materials includes having lights inside a volume.
          {
            // Calculate the transmittance along the light sample's distance in case it's inside a volume.
            // The light must be in the same volume or it would have been shadowed.
            lightSample.emission *= expf(-lightSample.distance * thePrd->sigma_t);
          }

          const float weightMis = powerHeuristic(lightSample.pdf, bsdf_pdf.w);
            
          thePrd->radiance += make_float3(bsdf_pdf) * lightSample.emission * (weightMis * dot(lightSample.direction, state.normal) / lightSample.pdf);
        }
      }
    }
  }
#endif // USE_NEXT_EVENT_ESTIMATION
}
