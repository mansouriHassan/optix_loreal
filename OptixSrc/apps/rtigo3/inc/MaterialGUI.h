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

#pragma once

#ifndef MATERIAL_GUI_H
#define MATERIAL_GUI_H

#include "shaders/function_indices.h"

#include <string>

 // Host side GUI material parameters 
struct MaterialGUI
{
  std::string   name;             // The name used in the scene description to identify this material instance.
  FunctionIndex indexBSDF;        // BSDF index to use in the closest hit program.
  float3        albedo;           // Tint, throughput change for specular materials.
  float3        absorptionColor;  // absorptionColor and absorptionScale together build the absorption coefficient
  float         absorptionScale;  
  float2        roughness;        // Anisotropic roughness for microfacet distributions.
  float         ior;              // Index of Refraction.
  bool          thinwalled;
  bool          useEyeTexture= false; // FIXME Implement materials which can have different textures. 
  bool          useHeadTexture = false;
  bool			useAlbedoTexture;
  bool			useCutoutTexture;

  // Hair parameters
  float			whitepercen;
  float3		dye;
  float			dye_concentration;
  float			scale_angle_deg;		// Cuticle tilt angle
  float			roughnessM, roughnessN; // Roughness
  float			melanin_concentration;	// Melanin concentration
  float			melanin_ratio;			// Melanin ratio
  float			melanin_concentration_disparity; // Melanin ratio
  float			melanin_ratio_disparity; // Melanin ratio

  //Hair expert color
  float3 cendre = make_float3(136.0f/255.0f, 136.0f/255.0f, 255.0f/255.0f); 
  float3 irise = make_float3(95.0f/255.0f, 25.0f/255.0f, 120.0f/255.0f);
  float3 doree = make_float3(200.0f/255.0f, 150.0f/255.0f, 0.0f/255.0f);
  float3 cuivre = make_float3(229.0f/255.0f, 54.0f/255.0f, 5.0f/255.0f);
  float3 acajou = make_float3(64.0f/255.0f, 6.0f/255.0f, 31.0f/255.0f);
  float3 red = make_float3(255.0f/255.0f, 0.0f/255.0f, 0.0f/255.0f);
  float3 vert = make_float3(0.0f/255.0f, 255.0f/255.0f, 0.0f/255.0f);

   int HT; //PSAN
   float f_HT;
   float concentrationCendre;
   float concentrationIrise;
   float concentrationDore;
   float concentrationCuivre;
   float concentrationAcajou;
   float concentrationRouge;
   float concentrationVert;

   //PSAN Interface GUI color neutral
   float concentrationBleuOrange;
   float concentrationVertRouge;
   float concentrationVioletJaune;

   int int_VertRouge_Concentration;
   int int_CendreCuivre_Concentration;
   int int_IriseDore_Concentration;

   float dye_ConcentrationVertRouge;
   float dye_ConcentrationCendreCuivre;   
   float dye_ConcentrationIriseDore;

   float dyeNeutralHT_Concentration;  
   float3 dyeNeutralHT;

   bool shouldModify{ false };
};

#endif // MATERIAL_GUI_H
