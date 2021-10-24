#include "inc/Application.h"
#include "..\inc\userGUI.h"

#include "imgui.h"
#include <ctype.h>          // toupper, isprint
#include <math.h>           // sqrtf, powf, cosf, sinf, floorf, ceilf
#include <stdio.h>          // vsnprintf, sscanf, printf
#include <stdlib.h>         // NULL, malloc, free, atoi
#if defined(_MSC_VER) && _MSC_VER <= 1500 // MSVC 2008 or earlier
#include <stddef.h>         // intptr_t
#else
#include <stdint.h>         // intptr_t
#endif

#ifdef _MSC_VER
#pragma warning (disable: 4996) // 'This function or variable may be unsafe': strcpy, strdup, sprintf, vsnprintf, sscanf, fopen
#define snprintf _snprintf
#endif
#ifdef __clang__
#pragma clang diagnostic ignored "-Wold-style-cast"             // warning : use of old-style cast                              // yes, they are more terse.
#pragma clang diagnostic ignored "-Wdeprecated-declarations"    // warning : 'xx' is deprecated: The POSIX name for this item.. // for strdup used in demo code (so user can copy & paste the code)
#pragma clang diagnostic ignored "-Wint-to-void-pointer-cast"   // warning : cast to 'void *' from smaller integer type 'int'
#pragma clang diagnostic ignored "-Wformat-security"            // warning : warning: format string is not a string literal
#pragma clang diagnostic ignored "-Wexit-time-destructors"      // warning : declaration requires an exit-time destructor       // exit-time destruction order is undefined. if MemFree() leads to users code that has been disabled before exit it might cause problems. ImGui coding style welcomes static/globals.
#if __has_warning("-Wreserved-id-macro")
#pragma clang diagnostic ignored "-Wreserved-id-macro"          // warning : macro name is a reserved identifier                //
#endif
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"          // warning: cast to pointer from integer of different size
#pragma GCC diagnostic ignored "-Wformat-security"              // warning : format string is not a string literal (potentially insecure)
#pragma GCC diagnostic ignored "-Wdouble-promotion"             // warning: implicit conversion from 'float' to 'double' when passing argument to function
#pragma GCC diagnostic ignored "-Wconversion"                   // warning: conversion to 'xxxx' from 'xxxx' may alter its value
#if (__GNUC__ >= 6)
#pragma GCC diagnostic ignored "-Wmisleading-indentation"       // warning: this 'if' clause does not guard this statement      // GCC 6.0+ only. See #883 on GitHub.
#endif
#endif

// Play it nice with Windows users. Notepad in 2017 still doesn't display text data with Unix-style \n.
#ifdef _WIN32
#define IM_NEWLINE "\r\n"
#else
#define IM_NEWLINE "\n"
#endif

#define IM_MAX(_A,_B)       (((_A) >= (_B)) ? (_A) : (_B))


userGUI::userGUI() :m_isVisibleGUI(true)
{

}

void userGUI::userWindow(bool* p_open)
{
    bool refresh = false;
    static bool no_close = false;

    ImGuiWindowFlags window_flags = 0;

    if (no_close)     p_open = NULL; // Don't pass our bool* to Begin

    ImGui::SetNextWindowSize(ImVec2(550, 680), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("User", p_open, window_flags))
    {
        // Early out if the window is collapsed, as an optimization.
        ImGui::End();
        return;
    }

    //ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.65f);    // 2/3 of the space for widget and 1/3 for labels
    ImGui::PushItemWidth(-140);                                 // Right align, keep 140 pixels for labels

    ImGui::Text("dear imgui says hello. (%s)", IMGUI_VERSION);


    if (ImGui::CollapsingHeader("Colortheque"))
    {
        MaterialGUI& materialGUI1 = m_materialsGUI[1]; //Hair 1
        MaterialGUI& materialGUI2 = m_materialsGUI[2]; //Hair 2
        bool changed = false;
        static int swatch = 1;
        if (ImGui::Combo("Color Swatch", &swatch, "Swatch 1\0 Swatch 2\0 Swatch 3\0 Swatch 4\0\0")) {
            switch (swatch) {
            case 0:
                materialGUI1.dye = make_float3(236.0 / 255.0, 71.0 / 255.0, 71.0 / 255.0);
                materialGUI1.dye_concentration = 0.03;
                materialGUI1.melanin_concentration = 2.83;
                materialGUI1.melanin_ratio = 0.38;
                materialGUI1.melanin_concentration_disparity = 0.0;
                materialGUI1.melanin_ratio_disparity = 0.0;
                materialGUI2.dye = make_float3(236.0 / 255.0, 71.0 / 255.0, 71.0 / 255.0);
                materialGUI2.dye_concentration = 0.03;
                materialGUI2.melanin_concentration = 2.83;
                materialGUI2.melanin_ratio = 0.38;
                materialGUI2.melanin_concentration_disparity = 0.0;
                materialGUI2.melanin_ratio_disparity = 0.0;
                changed = true;
                break;
            case 1:
                materialGUI1.dye = make_float3(71.0 / 255.0, 144.0 / 255.0, 236.0 / 255.0);
                materialGUI1.dye_concentration = 0.25;
                materialGUI1.melanin_concentration = 1.62;
                materialGUI1.melanin_ratio = 0.05;
                materialGUI1.melanin_concentration_disparity = 0.0;
                materialGUI1.melanin_ratio_disparity = 0.0;
                materialGUI2.dye = make_float3(71.0 / 255.0, 144.0 / 255.0, 236.0 / 255.0);
                materialGUI2.dye_concentration = 0.25;
                materialGUI2.melanin_concentration = 1.62;
                materialGUI2.melanin_ratio = 0.05;
                materialGUI2.melanin_concentration_disparity = 0.0;
                materialGUI2.melanin_ratio_disparity = 0.0;
                changed = true;
                break;
            case 2:
                materialGUI1.dye = make_float3(85.0 / 255.0, 71.0 / 255.0, 236.0 / 255.0);
                materialGUI1.dye_concentration = 0.16;
                materialGUI1.melanin_concentration = 0.28;
                materialGUI1.melanin_ratio = 0.00;
                materialGUI1.melanin_concentration_disparity = 0.0;
                materialGUI1.melanin_ratio_disparity = 0.0;
                materialGUI2.dye = make_float3(85.0 / 255.0, 71.0 / 255.0, 236.0 / 255.0);
                materialGUI2.dye_concentration = 0.16;
                materialGUI2.melanin_concentration = 0.28;
                materialGUI2.melanin_ratio = 0.00;
                materialGUI2.melanin_concentration_disparity = 0.0;
                materialGUI2.melanin_ratio_disparity = 0.0;
                changed = true;
                break;
            case 3:
                materialGUI1.dye = make_float3(225.0 / 255.0, 43.0 / 255.0, 12.0 / 255.0);
                materialGUI1.dye_concentration = 0.79;
                materialGUI1.melanin_concentration = 1.72;
                materialGUI1.melanin_ratio = 0.54;
                materialGUI1.melanin_concentration_disparity = 0.0;
                materialGUI1.melanin_ratio_disparity = 0.0;
                materialGUI2.dye = make_float3(225.0 / 255.0, 43.0 / 255.0, 12.0 / 255.0);
                materialGUI2.dye_concentration = 0.79;
                materialGUI2.melanin_concentration = 1.72;
                materialGUI2.melanin_ratio = 0.54;
                materialGUI2.melanin_concentration_disparity = 0.0;
                materialGUI2.melanin_ratio_disparity = 0.0;
                changed = true;
                break;
            }
        }
        if (changed)
        {
            //m_raytracer->updateMaterial(1, materialGUI1);
            //m_raytracer->updateMaterial(2, materialGUI2);
            refresh = true;
        }
    }

    if (ImGui::CollapsingHeader("Materials"))
    {
        for (int i = 0; i < static_cast<int>(m_materialsGUI.size()); ++i)
        {
            bool changed = false;

            MaterialGUI& materialGUI = m_materialsGUI[i];

            if (ImGui::TreeNode((void*)(intptr_t)i, "%s", m_materialsGUI[i].name.c_str()))
            {

                if (ImGui::Combo("BxDF Type", (int*)&materialGUI.indexBSDF, "BRDF Diffuse\0BRDF Specular\0BSDF Specular\0BRDF GGX Smith\0BSDF GGX Smith\0BSDF Hair\0\0"))
                {
                    changed = true;
                }
                if (materialGUI.indexBSDF != INDEX_BCSDF_HAIR) {
                    if (ImGui::ColorEdit3("Albedo", (float*)&materialGUI.albedo))
                    {
                        changed = true;
                    }
                    if (ImGui::Checkbox("Use Albedo Texture", &materialGUI.useAlbedoTexture))
                    {
                        changed = true;
                    }
                    if (ImGui::Checkbox("Use Cutout Texture", &materialGUI.useCutoutTexture))
                    {
                        changed = true;
                    }
                    if (ImGui::Checkbox("Thin-Walled", &materialGUI.thinwalled)) // Set this to true when using cutout opacity!
                    {
                        changed = true;
                    }
                    // Only show material parameters for the BxDFs which are affected by IOR and volume absorption.
                    if (materialGUI.indexBSDF == INDEX_BSDF_SPECULAR ||
                        materialGUI.indexBSDF == INDEX_BSDF_GGX_SMITH)
                    {
                        if (ImGui::ColorEdit3("Absorption", (float*)&materialGUI.absorptionColor))
                        {
                            changed = true;
                        }
                        if (ImGui::DragFloat("Absorption Scale", &materialGUI.absorptionScale, 0.01f, 0.0f, 1000.0f, "%.2f"))
                        {
                            changed = true;
                        }
                        if (ImGui::DragFloat("IOR", &materialGUI.ior, 0.01f, 0.0f, 10.0f, "%.2f"))
                        {
                            changed = true;
                        }
                    }
                    // Only show material parameters for the BxDFs which are affected by roughness.
                    if (materialGUI.indexBSDF == INDEX_BRDF_GGX_SMITH ||
                        materialGUI.indexBSDF == INDEX_BSDF_GGX_SMITH)
                    {
                        if (ImGui::DragFloat2("Roughness", reinterpret_cast<float*>(&materialGUI.roughness), 0.001f, 0.0f, 1.0f, "%.3f"))
                        {
                            // Clamp the microfacet roughness to working values minimum values.
                            // FIXME When both roughness values fall below that threshold, use INDEX_BSDF_SPECULAR_REFLECTION instead.
                            if (materialGUI.roughness.x < MICROFACET_MIN_ROUGHNESS)
                            {
                                materialGUI.roughness.x = MICROFACET_MIN_ROUGHNESS;
                            }
                            if (materialGUI.roughness.y < MICROFACET_MIN_ROUGHNESS)
                            {
                                materialGUI.roughness.y = MICROFACET_MIN_ROUGHNESS;
                            }
                            changed = true;
                        }
                    }
                }
                else {
                    if (ImGui::ColorEdit3("Dye", (float*)&materialGUI.dye)) {
                        changed = true;
                    }
                    if (ImGui::SliderFloat("Dye Concentration", &materialGUI.dye_concentration,
                        0.0f, 5.0f, "%.2f")) {
                        changed = true;
                    }
                    if (ImGui::SliderFloat("White Hair percent", &materialGUI.whitepercen,
                        0.0f, 1.0f, "%.02f")) {
                        changed = true;
                    }
                    if (ImGui::SliderFloat("Cuticle Tilt Angle", &materialGUI.scale_angle_deg,
                        0.0f, 45.0f, "%.2f")) {
                        changed = true;
                    }
                    if (ImGui::SliderFloat("RoughnessAzimutal", &materialGUI.roughnessN,
                        0.0f, 1.0f, "%.2f")) {
                        changed = true;
                    }
                    if (ImGui::SliderFloat("RoughnessLonitudinal", &materialGUI.roughnessM,
                        0.0f, 1.0f, "%.2f")) {
                        changed = true;
                    }
                    if (ImGui::SliderFloat("Melanin Concentration", &materialGUI.melanin_concentration,
                        0.0f, 8.0f, "%.2f")) {
                        changed = true;
                    }
                    if (ImGui::SliderFloat("Melanin Ratio", &materialGUI.melanin_ratio,
                        0.0f, 1.0f, "%.2f")) {
                        changed = true;
                    }
                    if (ImGui::SliderFloat("Melanin Concentration Disparity", &materialGUI.melanin_concentration_disparity,
                        0.0f, 1.0f, "%.2f")) {
                        changed = true;
                    }
                    if (ImGui::SliderFloat("Melanin Ratio Disparity", &materialGUI.melanin_ratio_disparity,
                        0.0f, 1.0f, "%.2f")) {
                        changed = true;
                    }

                }

                if (changed)
                {
                   // m_raytracer->updateMaterial(i, materialGUI);
                    refresh = true;
                }
                ImGui::TreePop();
            }
        }
    }

    ImGui::End();
}