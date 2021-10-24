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

#ifndef APPLICATION_H
#define APPLICATION_H

// DAR This renderer only uses the CUDA Driver API!
// (CMake uses the CUDA_CUDA_LIBRARY which is nvcuda.lib. At runtime that loads nvcuda.dll from the driver.)
//#include <cuda.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
#endif

#include "imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS 1
#include "imgui_internal.h"
#include "imgui_impl_glfw_gl3.h"

#include <GL/glew.h>
#if defined( _WIN32 )
#include <GL/wglew.h>
#endif

#include "inc/Camera.h"
#include "inc/Options.h"
#include "inc/Rasterizer.h"
#include "inc/Raytracer.h"
#include "inc/SceneGraph.h"
#include "inc/Texture.h"
#include "inc/Timer.h"


#include <dp/math/Matmnt.h>

#include "shaders/material_definition.h"

// This include gl.h and needs to be done after glew.h
#include <GLFW/glfw3.h>

// assimp include files.
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/LogStream.hpp>

#include <map>
#include <memory>


#define APP_EXIT_SUCCESS          0

#define APP_ERROR_UNKNOWN        -1
#define APP_ERROR_CREATE_WINDOW  -2
#define APP_ERROR_GLFW_INIT      -3
#define APP_ERROR_GLEW_INIT      -4
#define APP_ERROR_APP_INIT       -5


enum GuiState
{
  GUI_STATE_NONE,
  GUI_STATE_ORBIT,
  GUI_STATE_PAN,
  GUI_STATE_DOLLY,
  GUI_STATE_FOCUS
};


enum KeywordScene
{
  KS_ALBEDO,
  KS_ROUGHNESS,
  KS_ABSORPTION,
  KS_ABSORPTION_SCALE,
  KS_IOR,
  KS_THINWALLED,
  KS_WHITEPERCEN,
  KS_DYE,
  KS_DYE_CONCENTRATION,
  KS_SCALE_ANGLE_DEG,
  KS_ROUGHNESS_M,
  KS_ROUGHNESS_N,
  KS_MELANIN_CONCENTRATION,
  KS_MELANIN_RATIO,
  KS_MELANIN_CONCENTRATION_DISPARITY,
  KS_MELANIN_RATIO_DISPARITY,
  KS_MATERIAL,
  KS_COLOR,
  KS_SETTING,
  KS_IDENTITY,
  KS_PUSH,
  KS_POP,
  KS_ROTATE,
  KS_SCALE,
  KS_TRANSLATE,
  KS_MODEL
};


struct ColorSwitch {
    std::string name;
    MaterialGUI Material1; 
    MaterialGUI Material2;
    std::string SettingFile;
};

struct ModelSwitch {
    std::string name;
    std::string file_name;
    std::string map_identifier;
    std::string material1Name;
    std::string material2Name;
};
struct HDRSwitch {
    std::string name;
    std::string file_name;
};

class Application
{
public:

  Application(GLFWwindow* window, Options const& options);
  ~Application();

  bool isValid() const;

  void reshape(const int w, const int h);
  bool render();
  void benchmark();

  void display();

  void guiNewFrame();
  void guiWindow();
  void guiEventHandler();
  void guiReferenceManual(); // The ImGui "programming manual" in form of a live window.
  void guiRender();

  void guiUserWindow(bool* p_open = NULL); //PSAN user ui
  void ShowOptionLayout(bool* p_open); // Here old version material interface and others settings
  void ShowAbsolueLayout(bool* p_open); // Test interface with 7 dye colors 

  bool is_fullscreen() { return m_is_fullscreen; }

private:
  bool loadSystemDescription(std::string const& filename);
  bool saveSystemDescription();
  bool loadSceneDescription(std::string const& filename);

  void restartRendering();

  void updateDYE(MaterialGUI& materialGUI); //PSAN 
  void updateDYEconcentration(MaterialGUI &materialGUI); //PSAN
  void updateHT(MaterialGUI& materialGUI); //PSAN TEST update HT
  void updateDYEinterface(MaterialGUI& materialGUI);

  bool screenshot(const bool tonemap);
  bool screenshot(const bool tonemap, std::string name);
  bool screenshot360();
  bool loading_bar(const float progress, const int bar_width = 70);

  float captureVariance();

  void createCameras();
  void createLights();
  void createPictures();

  void appendInstance(std::shared_ptr<sg::Group>& group,
                      std::shared_ptr<sg::Node> geometry, 
                      dp::math::Mat44f const& matrix, 
                      std::string const& reference, 
                      unsigned int& idInstance);

  std::shared_ptr<sg::Group> createASSIMP(std::string const& filename);
  std::shared_ptr<sg::Group> traverseScene(const struct aiScene *scene, const unsigned int indexSceneBase, const struct aiNode* node);

  void calculateTangents(std::vector<VertexAttributes>& attributes, std::vector<unsigned int> const& indices);

  void guiRenderingIndicator(const bool isRendering);

  bool loadString(std::string const& filename, std::string& text);
  bool saveString(std::string const& filename, std::string const& text);
  std::string getDateTime();
  void convertPath(std::string& path);
  void convertPath(char* path);
  void chargeSettingsFromFile(std::string Path);
  void saveSettingToFile(std::string path);


private:
  GLFWwindow* m_window;
  bool m_is_fullscreen{ false };
  bool        m_isValid;

  GuiState m_guiState;
  bool     m_isVisibleGUI;

  // Command line options:
  int         m_width;   // Client window size.
  int         m_height;
  int         m_mode;   // Application mode 0 = interactive, 1 = batched benchmark (single shot).

  // System options:
  int         m_strategy;    // "strategy"
  int         m_devicesMask; // "devicesMask" // Bitmask with enabled devices, default 0xFF for 8 devices. Only the visible ones will be used.
  std::vector<int>         m_area_light;       // "light"
  int         m_miss;        // "miss"
  std::string m_environment; // "envMap"
  int         m_interop;     // "interop"´// 0 = none all through host, 1 = register texture image, 2 = register pixel buffer
  bool        m_present;     // "present"
  bool        m_catchVariance;     // 0 = no variance calculation, 1 = variance calculation
  
  bool        m_presentNext;      // (derived)
  double      m_presentAtSecond;  // (derived)
  bool        m_previousComplete; // (derived) // Prevents spurious benchmark prints and image updates.

  // GUI Data representing raytracer settings.
  LensShader m_lensShader;          // "lensShader"
  int2       m_pathLengths;         // "pathLengths"   // min, max
  int2       m_resolution;          // "resolution"    // The actual size of the rendering, independent of the window's client size. (Preparation for final frame rendering.)
  int2       m_tileSize;            // "tileSize"      // Multi-GPU distribution tile size. Must be power-of-two values.
  int        m_samplesSqrt;         // "sampleSqrt"
  float      m_epsilonFactor;       // "epsilonFactor"
  float      m_environmentRotation; // "envRotation"
  float      m_clockFactor;         // "clockFactor"
  int        m_screenshotImageNum;

  std::string m_prefixScreenshot;   // "prefixScreenshot", allows to set a path and the prefix for the screenshot filename. spp, data, time and extension will be appended.
  
  std::string m_prefixColorSwitch;

  std::string m_prefixSettings;

  TonemapperGUI m_tonemapperGUI;    // "gamma", "whitePoint", "burnHighlights", "crushBlacks", "saturation", "brightness"
  
  Camera m_camera;                  // "center", "camera"

  float m_mouseSpeedRatio;
  
  Timer m_timer;


  float m_melanineConcentration[10] = { 8.f, 6.5f, 4.9f, 4.0f , 2.f, 1.16f, 0.78f, 0.41f, 0.30f, 0.25f }; // PSAN TEST add stock melanine before color modificiation

  float m_melanineRatio[10] = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f }; // ratio by HT fondamental

  float m_factorColorantHT[10] = { 1.f, 1.f, 1.f, 1.7f, 3.f, 2.5f, 4.5f, 10.f, 16.f, 17.f }; //colorant 

  float m_dyeNeutralHT_Concentration[10] = { 0.f, 0.15f, 0.13f, 0.13f, 0.33f, 0.f, 0.f, 0.f, 0.f, 0.f };

  float3 m_dyeNeutralHT[10] = {  make_float3 (255.f / 255.f, 255.f / 255.f, 255.f / 255.f),
									   make_float3(153.f / 255.f, 140.f / 255.f, 186.f / 255.f),
									   make_float3(153.f / 255.f, 140.f / 255.f, 186.f / 255.f),
									   make_float3(84.0f / 255.f, 182.f / 255.f, 157.f / 255.f),
									   make_float3(84.0f / 255.f, 182.f / 255.f, 157.f / 255.f),
									   make_float3(125.f / 255.f, 146.f / 255.f, 234.f / 255.f),
									   make_float3(255.f / 255.f, 255.f / 255.f, 255.f / 255.f),
									   make_float3(255.f / 255.f, 255.f / 255.f, 255.f / 255.f),
									   make_float3(255.f / 255.f, 255.f / 255.f, 255.f / 255.f),
									   make_float3(255.f / 255.f, 255.f / 255.f, 255.f / 255.f),
  };

  float m_lightened_x10[10] = {6.5f, 4.f, 3.12f, 1.73f, 1.34f, 0.65f, 0.39f, 0.28f, 0.25f, 0.15f };

  float m_lightened_x2[10] = { 6.5f, 4.f, 3.175f, 1.842f, 1.421f, 0.46f, 0.38f, 0.26f, 0.2f, 0.16f };

  float m_lightened_x1[10] = { 6.5f, 4.f, 3.12f, 1.73f, 1.34f, 0.65f, 0.39f, 0.28f, 0.25f, 0.15f };

  
  float lightened = 1.36f;

  float m_concentrationCendre[4] = {1.f, 2.f, 3.f, 4.f};
  float m_concentrationIrise[4] = { 1.f, 2.f, 3.f, 4.f };
  float m_concentrationDore[4] = { 1.f, 2.f, 3.f, 4.f };
  float m_concentrationCuivre[4] = { 1.f, 2.f, 3.f, 4.f };
  float m_concentrationRouge[4] = { 1.f, 2.f, 3.f, 4.f };
  float m_concentrationVert[4] = { 1.f, 2.f, 3.f, 4.f };

  float m_dye_ConcentrationVert[4] = { 1.f, 2.f, 3.5f, 4.f };
  float m_dye_ConcentrationRouge[4] = { 1.f, 2.f, 2.5f, 3.f };

  float m_dye_ConcentrationCender[4] = { 1.f, 2.f, 4.5f, 4.f };
  float m_dye_ConcentrationCover[4] = { 1.f, 2.f, 3.5f, 4.f };

  float m_dye_ConcentrationAsh[4] = { 1.f, 2.f, 2.5f, 2.f };
  float m_dye_ConcentrationGold[4] = { 1.f, 2.f, 3.5f, 4.f };

  std::pair<MaterialGUI, MaterialGUI>* QuickSaveValue[5];
  int nbQuickSaveValue;


  std::map<std::string, KeywordScene> m_mapKeywordScene;

  std::unique_ptr<Rasterizer> m_rasterizer;
  
  std::unique_ptr<Raytracer> m_raytracer; // This is a base class.
  DeviceState                m_state;
  DeviceState                m_default_state;

  // The scene description:
  // Unique identifiers per host scene node.
  unsigned int m_idGroup;
  unsigned int m_idInstance;
  unsigned int m_idGeometry;

  const char* current_item_model_value;
  ModelSwitch* current_item_model;

  const char* current_settings_value;
  bool hasChanged;

  std::shared_ptr<sg::Group> m_scene; // Root group node of the scene.
  
  std::vector< std::shared_ptr<sg::Node> > m_geometries; // All geometries in the scene.

  // For the runtime generated objects, this allows to find geometries with the same type and construction parameters.
  std::map<std::string, unsigned int> m_mapGeometries;

  // For all model file format loaders. Allows instancing of full models in the host side scene graph.
  std::map< std::string, std::shared_ptr<sg::Group> > m_mapGroups;

  std::vector<CameraDefinition> m_cameras;
  std::vector<LightDefinition>  m_lights;
  std::vector<MaterialGUI>      m_materialsGUI;
  std::vector<ColorSwitch>      m_materialsColor;
  std::vector<ModelSwitch>      m_models;
  std::vector<Camera> m_camera_pov;
  int m_current_camera;
  bool m_lock_camera;
  std::vector<std::pair<std::string, std::string>> m_settings;
  std::vector<HDRSwitch> m_HDR;
  // Map of local material names to indices in the m_materialsGUI vector.
  std::map<std::string, int> m_mapMaterialReferences; 

  std::map<std::string, Picture*> m_mapPictures;

  std::vector<unsigned int> m_remappedMeshIndices; 

  dp::math::Mat44f curMatrix;

  const Options* f_options;
  int m_scr_w;
  int m_scr_h;

  bool m_lightings_on[5] = { true,true,true,true,true };
  bool m_geo_group[8] = { true,true,true,true,true,true,true,true };
  int m_lighting_emission[5] = { 12,12,12,12,12};
};

#endif // APPLICATION_H
