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

#include "inc/Application.h"
#include "inc/Parser.h"

#include "inc/RaytracerSingleGPU.h"
#include "inc/RaytracerMultiGPUZeroCopy.h"
#include "inc/RaytracerMultiGPUPeerAccess.h"
#include "inc/RaytracerMultiGPULocalCopy.h"

#include <filesystem>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stack>
#include <memory>
#include <numeric>


#include <dp/math/Matmnt.h>

#include "inc/CheckMacros.h"

#include "inc/MyAssert.h"


namespace fs = std::filesystem;

Application::Application(GLFWwindow* window, Options const& options)
    : m_window(window)
    , m_isValid(false)
    , m_guiState(GUI_STATE_NONE)
    , m_isVisibleGUI(true)
    , m_width(512)
    , m_height(512)
    , m_mode(0)
    , m_strategy(RS_INTERACTIVE_SINGLE_GPU)
    , m_devicesMask(255)
    , m_area_light(0)
    , m_miss(1)
    , m_interop(0)
    , m_catchVariance(0)
    , m_present(false)
    , m_presentNext(true)
    , m_presentAtSecond(1.0)
    , m_previousComplete(false)
    , m_lensShader(LENS_SHADER_PINHOLE)
    , m_samplesSqrt(1)
    , m_epsilonFactor(500.0f)
    , m_environmentRotation(0.0f)
    , m_clockFactor(1000.0f)
    , m_mouseSpeedRatio(10.0f)
    , m_idGroup(0)
    , m_idInstance(0)
    , m_idGeometry(0)
    , m_screenshotImageNum(6)
    , m_current_camera(0)
    , m_lock_camera(0)
    , nbQuickSaveValue(0)
{
  try
  {
    m_timer.restart();
    f_options = &options;
    m_scr_w = f_options->getWidth();
    m_scr_h = f_options->getHeight();
    // Initialize the top-level keywords of the scene description for faster search.
    m_mapKeywordScene["albedo"]                 = KS_ALBEDO;
    m_mapKeywordScene["roughness"]              = KS_ROUGHNESS;
    m_mapKeywordScene["absorption"]             = KS_ABSORPTION;
    m_mapKeywordScene["absorptionScale"]        = KS_ABSORPTION_SCALE;
    m_mapKeywordScene["ior"]                    = KS_IOR;
    m_mapKeywordScene["thinwalled"]             = KS_THINWALLED;
    m_mapKeywordScene["whitepercen"]            = KS_WHITEPERCEN;
    m_mapKeywordScene["dye"]                    = KS_DYE;
    m_mapKeywordScene["dyeConcentration"]       = KS_DYE_CONCENTRATION;
    m_mapKeywordScene["cuticleTiltDeg"]         = KS_SCALE_ANGLE_DEG;
    m_mapKeywordScene["roughnessM"]             = KS_ROUGHNESS_M;
    m_mapKeywordScene["roughnessN"]             = KS_ROUGHNESS_N;
    m_mapKeywordScene["melaninConcentration"]   = KS_MELANIN_CONCENTRATION;
    m_mapKeywordScene["melaninRatio"]           = KS_MELANIN_RATIO;
    m_mapKeywordScene["melaninConcentrationDisparity"] = KS_MELANIN_CONCENTRATION_DISPARITY;
    m_mapKeywordScene["melaninRatioDisparity"] = KS_MELANIN_RATIO_DISPARITY;
    m_mapKeywordScene["material"]               = KS_MATERIAL;
    m_mapKeywordScene["color"]                  = KS_COLOR;
    m_mapKeywordScene["settings"]               = KS_SETTING;
    m_mapKeywordScene["identity"]               = KS_IDENTITY;
    m_mapKeywordScene["push"]                   = KS_PUSH;
    m_mapKeywordScene["pop"]                    = KS_POP;
    m_mapKeywordScene["rotate"]                 = KS_ROTATE;
    m_mapKeywordScene["scale"]                  = KS_SCALE;
    m_mapKeywordScene["translate"]              = KS_TRANSLATE;
    m_mapKeywordScene["model"]                  = KS_MODEL;
    
    const double timeConstructor = m_timer.getTime();

    m_width  = std::max(1, options.getWidth());
    m_height = std::max(1, options.getHeight());
    m_mode   = std::max(0, options.getMode());

    // Initialize the system options to minimum defaults to work, but require useful settings inside the system options file.
    // The minumum path length values will generate useful direct lighting results, but transmissions will be mostly black.
    m_resolution  = make_int2(1, 1);
    m_tileSize    = make_int2(8, 8);
    m_pathLengths = make_int2(0, 2);

    m_prefixScreenshot = std::string("./img"); // Default to current working directory and prefix "img".
    m_prefixColorSwitch = std::string("./ColorSwitch/");
    m_prefixSettings = std::string("./Settings");
    // Tonmapper neutral defaults. The system description overrides these.
    m_tonemapperGUI.gamma           = 1.0f;
    m_tonemapperGUI.whitePoint      = 1.0f;
    m_tonemapperGUI.colorBalance[0] = 1.0f;
    m_tonemapperGUI.colorBalance[1] = 1.0f;
    m_tonemapperGUI.colorBalance[2] = 1.0f;
    m_tonemapperGUI.burnHighlights  = 1.0f;
    m_tonemapperGUI.crushBlacks     = 0.0f;
    m_tonemapperGUI.saturation      = 1.0f; 
    m_tonemapperGUI.brightness      = 1.0f;

    // System wide parameters are loaded from this file to keep the number of command line options small.
    //const std::string filenameSystem = options.getSystem();
    const std::string filenameSystem("system_optix_hair_single_gpu.txt");
    if (!loadSystemDescription(filenameSystem))
    {
      std::cerr << "ERROR: Application() failed to load system description file " << filenameSystem << '\n';
      MY_ASSERT(!"Failed to load system description");
      return; // m_isValid == false.
    }
    if (!fs::exists(m_prefixColorSwitch))
    {
        fs::create_directory(m_prefixColorSwitch);
    }
    if (!fs::exists(m_prefixSettings))
    {
        fs::create_directory(m_prefixSettings);
    }
    // The user interface is part of the main application.
    // Setup ImGui binding.
    ImGui::CreateContext();
    ImGui_ImplGlfwGL3_Init(window, true);

    // This initializes the GLFW part including the font texture.
    ImGui_ImplGlfwGL3_NewFrame();
    ImGui::EndFrame();

#if 0
    // Style the GUI colors to a neutral greyscale with plenty of transparency to concentrate on the image.
    ImGuiStyle& style = ImGui::GetStyle();

    // Change these RGB values to get any other tint.
    const float r = 1.0f;
    const float g = 1.0f;
    const float b = 1.0f;
  
    style.Colors[ImGuiCol_Text]                  = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    style.Colors[ImGuiCol_TextDisabled]          = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
    style.Colors[ImGuiCol_WindowBg]              = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 0.6f);
    style.Colors[ImGuiCol_ChildWindowBg]         = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 1.0f);
    style.Colors[ImGuiCol_PopupBg]               = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 1.0f);
    style.Colors[ImGuiCol_Border]                = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_BorderShadow]          = ImVec4(r * 0.0f, g * 0.0f, b * 0.0f, 0.4f);
    style.Colors[ImGuiCol_FrameBg]               = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_FrameBgHovered]        = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_FrameBgActive]         = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_TitleBg]               = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_TitleBgCollapsed]      = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_TitleBgActive]         = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_MenuBarBg]             = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 1.0f);
    style.Colors[ImGuiCol_ScrollbarBg]           = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 0.2f);
    style.Colors[ImGuiCol_ScrollbarGrab]         = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_ScrollbarGrabHovered]  = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_ScrollbarGrabActive]   = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_CheckMark]             = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_SliderGrab]            = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_SliderGrabActive]      = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_Button]                = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_ButtonHovered]         = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_ButtonActive]          = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_Header]                = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_HeaderHovered]         = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_HeaderActive]          = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_Column]                = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_ColumnHovered]         = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_ColumnActive]          = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_ResizeGrip]            = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_ResizeGripHovered]     = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_ResizeGripActive]      = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
    style.Colors[ImGuiCol_CloseButton]           = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_CloseButtonHovered]    = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_CloseButtonActive]     = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_PlotLines]             = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 1.0f);
    style.Colors[ImGuiCol_PlotLinesHovered]      = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
    style.Colors[ImGuiCol_PlotHistogram]         = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 1.0f);
    style.Colors[ImGuiCol_PlotHistogramHovered]  = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
    style.Colors[ImGuiCol_TextSelectedBg]        = ImVec4(r * 0.5f, g * 0.5f, b * 0.5f, 1.0f);
    style.Colors[ImGuiCol_ModalWindowDarkening]  = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 0.2f);
    style.Colors[ImGuiCol_DragDropTarget]        = ImVec4(r * 1.0f, g * 1.0f, b * 0.0f, 1.0f); // Yellow
    style.Colors[ImGuiCol_NavHighlight]          = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
    style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
#endif
  
    const double timeGUI = m_timer.getTime();

    m_camera.setResolution(m_resolution.x, m_resolution.y);
    m_camera.setSpeedRatio(m_mouseSpeedRatio);

    // Initialize the OpenGL rasterizer.
    m_rasterizer = std::make_unique<Rasterizer>(m_width, m_height, m_interop);
    
    // Must set the resolution explicitly to be able to calculate 
    // the proper vertex attributes for display and the PBO size in case of interop.
    m_rasterizer->setResolution(m_resolution.x, m_resolution.y); 
    m_rasterizer->setTonemapper(m_tonemapperGUI);

    const unsigned int tex = m_rasterizer->getTextureObject();
    const unsigned int pbo = m_rasterizer->getPixelBufferObject();

    const double timeRasterizer = m_timer.getTime();

    // Initialize the OptiX raytracer.
    switch (m_strategy) // The strategy is limited to valid enums by the caller 
    {
      case RS_INTERACTIVE_SINGLE_GPU:
        m_raytracer = std::make_unique<RaytracerSingleGPU>(m_devicesMask, m_miss, m_interop, tex, pbo);
        m_state.distribution = 0; // Full frames.
        break;

      case RS_INTERACTIVE_MULTI_GPU_ZERO_COPY:
        m_raytracer = std::make_unique<RaytracerMultiGPUZeroCopy>(m_devicesMask, m_miss, m_interop, tex, pbo);
        m_state.distribution = 1; // Distributed rendering of one frame, when device count > 1, tiled rendering.
        break;

      case RS_INTERACTIVE_MULTI_GPU_PEER_ACCESS:
        m_raytracer = std::make_unique<RaytracerMultiGPUPeerAccess>(m_devicesMask, m_miss, m_interop, tex, pbo);
        m_state.distribution = 1; // Distributed rendering of one frame, when device count > 1, tiled rendering.
        break;

      case RS_INTERACTIVE_MULTI_GPU_LOCAL_COPY:
        m_raytracer = std::make_unique<RaytracerMultiGPULocalCopy>(m_devicesMask, m_miss, m_interop, tex, pbo);
        m_state.distribution = 1; // Distributed rendering of one frame, when device count > 1, tiled rendering.
        break;
    }

    // If the raytracer could not be initialized correctly, return and leave Application invalid.
    if (!m_raytracer->m_isValid)
    {
      std::cerr << "ERROR: Application() Could not initialize Raytracer with strategy = " << m_strategy << '\n';
      return; // Exit application.
    }

    // Determine which device is the one running the OpenGL implementation.
    // The first OpenGL-CUDA device match wins.
    int deviceMatch = -1;

#if 1
    // UUID works under Windows and Linux.
    const int numDevicesOGL = m_rasterizer->getNumDevices();

    for (int i = 0; i < numDevicesOGL && deviceMatch == -1; ++i)
    {
      deviceMatch = m_raytracer->matchUUID(reinterpret_cast<const char*>(m_rasterizer->getUUID(i)));
    }
#else
    // LUID only works under Windows because it requires the EXT_external_objects_win32 extension.
    // DEBUG With multicast enabled, both devices have the same LUID and the OpenGL node mask is the OR of the individual device node masks.
    // Means the result of the deviceMatch here is depending on the CUDA device order. 
    // Seems like multicast needs to handle CUDA - OpenGL interop differently.
    // With multicast enabled, uploading the PBO with glTexImage2D halves the framerate when presenting each image in both the single-GPU and multi-GPU P2P strategy.
    // Means there is an expensive PCI-E copy going on in that case.
    const unsigned char* luid = m_rasterizer->getLUID();
    const int nodeMask        = m_rasterizer->getNodeMask(); 

    // The cuDeviceGetLuid() takes char* and unsigned int though.
    deviceMatch = m_raytracer->matchLUID(reinterpret_cast<const char*>(luid), nodeMask);
#endif

    if (deviceMatch == -1)
    {
      if (m_interop == INTEROP_MODE_TEX)
      {
        std::cerr << "ERROR: Application() OpenGL texture image interop without OpenGL device in active devices will not display the image!\n";
        return; // Exit application.
      }
      if (m_interop == INTEROP_MODE_PBO)
      {
        std::cerr << "WARNING: Application() OpenGL pixel buffer interop without OpenGL device in active devices will result in reduced performance!\n";
      }
    }

    m_state.resolution    = m_resolution;
    m_state.tileSize      = m_tileSize;
    m_state.pathLengths   = m_pathLengths;
    m_state.samplesSqrt   = m_samplesSqrt;
    m_state.lensShader    = m_lensShader;
    m_state.epsilonFactor = m_epsilonFactor;
    m_state.envRotation   = m_environmentRotation;
    m_state.clockFactor   = m_clockFactor;
    m_state.catchVariance = m_catchVariance;
    m_state.screenshotImageNum = m_screenshotImageNum;

    // Sync the state with the default GUI data.
    m_raytracer->initState(m_state);
    //m_raytracer->initVarianceCatching(m_catchVariance);

    const double timeRaytracer = m_timer.getTime();

    // Host side scene graph information.
    m_scene = std::make_shared<sg::Group>(m_idGroup++); // Create the scene's root group first.

    createPictures();
    createCameras();
    createLights();
    
    // Load the scene description file and generate the host side scene.
    //const std::string filenameScene = options.getScene();
    const std::string filenameScene("scene_optix_hair_half_head.txt");
    if (!loadSceneDescription(filenameScene))
    {
      std::cerr << "ERROR: Application() failed to load scene description file " << filenameScene << '\n';
      MY_ASSERT(!"Failed to load scene description");
      return;
    }
    if (m_models.size() != 0)
    {
        current_item_model_value = m_models[0].name.c_str();
        current_item_model = &m_models[0];
    }
    for (int i = 0; i < 5; i++)
    {
        QuickSaveValue[i] = NULL;
    }
    current_settings_value = NULL;
    hasChanged = false;
    MY_ASSERT(m_idGeometry == m_geometries.size());

    const double timeScene = m_timer.getTime();

    // Device side scene information.
    m_raytracer->initTextures(m_mapPictures); // HACK Hardcoded textures. // FIXME Implement a full material system.
    m_raytracer->initCameras(m_cameras);
    m_raytracer->initLights(m_lights);
    m_raytracer->initMaterials(m_materialsGUI);
    m_raytracer->initScene(m_scene, m_idGeometry); // m_idGeometry is the number of geometries in the scene.
    
    const double timeRenderer = m_timer.getTime();

    // Print out hiow long the initialization of each module took.
    std::cout << "Application(): " << timeRenderer - timeConstructor   << " seconds overall\n";
    std::cout << "{\n";
    std::cout << "  GUI        = " << timeGUI        - timeConstructor << " seconds\n";
    std::cout << "  Rasterizer = " << timeRasterizer - timeGUI         << " seconds\n";
    std::cout << "  Raytracer  = " << timeRaytracer  - timeRasterizer  << " seconds\n";
    std::cout << "  Scene      = " << timeScene      - timeRaytracer   << " seconds\n";
    std::cout << "  Renderer   = " << timeRenderer   - timeScene       << " seconds\n";
    std::cout << "}\n";

    restartRendering(); // Trigger a new rendering.

    m_isValid = true;   
  }
  catch (std::exception const& e)
  {
    std::cerr << e.what() << '\n';
  }
}


Application::~Application()
{
  for (std::map<std::string, Picture*>::const_iterator it =  m_mapPictures.begin(); it != m_mapPictures.end(); ++it)
  {
    delete it->second;
  }
  
  ImGui_ImplGlfwGL3_Shutdown();
  ImGui::DestroyContext();
}

bool Application::isValid() const
{
  return m_isValid;
}

void Application::reshape(const int w, const int h)
{
   // Do not allow zero size client windows! All other reshape() routines depend on this and do not check this again.
  if ( (m_width != w || m_height != h) && w != 0 && h != 0)
  {
    m_width  = w;
    m_height = h;
    
    m_rasterizer->reshape(m_width, m_height);
  }
}

void Application::restartRendering()
{
  guiRenderingIndicator(true);

  m_presentNext      = true;
  m_presentAtSecond  = 1.0;
  
  m_previousComplete = false;
  
  m_timer.restart();
}

bool Application::render()
{
  bool finish = false;
  bool flush  = false;

  try
  {
    CameraDefinition camera;

    const bool cameraChanged = m_camera.getFrustum(camera.P, camera.U, camera.V, camera.W);
    if (cameraChanged)
    {
      m_cameras[0] = camera;
      m_raytracer->updateCamera(0, camera);

      restartRendering();
    }

    const unsigned int iterationIndex = m_raytracer->render();
    
    // When the renderer has completed all iterations, change the GUI title bar to green.
    const bool complete = ((unsigned int)(m_samplesSqrt * m_samplesSqrt) <= iterationIndex);

    if (complete)
    {
      guiRenderingIndicator(false); // Not rendering anymore.

      flush = !m_previousComplete && complete; // Completion status changed to true.
    }
    
    m_previousComplete = complete;
    
    // When benchmark is enabled, exit the application when the requested samples per pixel have ben rendered.
    // Actually this render() function is not called when m_mode == 1 but keep the finish here to exit on exceptions.
    finish = ((m_mode == 1) && complete);
    
    // Only update the texture when a restart happened, one second passed to reduce required bandwidth, or the rendering is newly complete.
    if (m_presentNext || flush)
    {
      m_raytracer->updateDisplayTexture(); // This directly updates the display HDR texture for all rendering strategies.

      m_presentNext = m_present;
    }

    double seconds = m_timer.getTime();
#if 1
    // When in interactive mode, show the all rendered frames during the first half second to get some initial refinement.
    if (m_mode == 0 && seconds < 0.5)
    {
      m_presentAtSecond = 1.0;
      m_presentNext     = true;
    }
#endif

    if (m_presentAtSecond < seconds || flush || finish) // Print performance every second or when the rendering is complete or the benchmark finished.
    {
      m_presentAtSecond = ceil(seconds);
      m_presentNext     = true; // Present at least every second.
      
      if (flush || finish) // Only print the performance when the samples per pixels are reached.
      {
        const double fps = double(iterationIndex) / seconds;

        std::ostringstream stream; 
        stream.precision(3); // Precision is # digits in fraction part.
        
        stream << std::fixed << "Samples number : " << iterationIndex << " / Time elapsed : " << seconds << " s / fps : " << fps;
        if (m_catchVariance)
            stream << " / Confidence Interval : " << captureVariance() * 100.f << " %";
        std::cout << stream.str() << '\n';

#if 0   // Automated benchmark in interactive mode. Change m_isVisibleGUI default to false!
        std::ostringstream filename;
        filename << "result_interactive_" << m_strategy << "_" << m_interop << "_" << m_tileSize.x << "_" << m_tileSize.y << ".log";
        const bool success = saveString(filename.str(), stream.str());
        if (success)
        {
          std::cout << filename.str()  << '\n'; // Print out the filename to indicate success.
        }
        finish = true; // Exit application after interactive rendering finished.
#endif
      }
    }
  }
  catch (std::exception const& e)
  {
    std::cerr << e.what() << '\n';
    finish = true;
  }
  return finish;
}

void Application::benchmark()
{
  try
      {         
      std::string standard_prefix = std::string("./benchmark/" + getDateTime() + "/");

      if (!fs::exists("./benchmark/")) {
          fs::create_directory("./benchmark/");
      }
      if (!fs::exists(standard_prefix)) {
          fs::create_directory(standard_prefix);
      }

    const unsigned int spp = (unsigned int)(m_samplesSqrt * m_samplesSqrt);
    unsigned int iterationIndex = 0; 

    std::ostringstream stream;
    stream.precision(3); // Precision is # digits in fraction part.
    const unsigned int samples_sqrt[6] = { 1,4,16,64,256,1024 };
    m_timer.restart();
    for (auto i : samples_sqrt) {
        
        while (iterationIndex < i)
        {
            iterationIndex = m_raytracer->render();
            
            if (i % 32 == 0) {
                float progress = iterationIndex / 1024.f;

                loading_bar(progress);
            }
        }
        m_raytracer->synchronize(); // Wait until any asynchronous operations have finished.

        const double seconds = m_timer.getTime();
        const double fps = double(iterationIndex) / seconds;
       
        stream << std::setprecision(4) << iterationIndex << " / " << seconds << " = " << fps << " fps";
        stream << " / Confidence Interval : " << captureVariance() * 100.f << " %\n";

        m_prefixScreenshot = std::to_string(i);
        screenshot(true, standard_prefix + std::to_string(i));
    }
    stream << "strategy : "<< m_strategy << ";\n" << "interoperability : " << m_interop << ";\n" << "tilesize : [" << m_tileSize.x << "," << m_tileSize.y << "]";
    

#if 1 // Automated benchmark in batch mode.
    std::ostringstream filename;
    filename << standard_prefix << "results.txt";
    const bool success = saveString(filename.str(), stream.str());
    if (success)
    {
      std::cout << '\n' << filename.str() << '\n'; // Print out the filename to indicate success.
    }
#endif


  }
  catch (std::exception const& e)
  {
    std::cerr << e.what() << '\n';
  }
}

void Application::display()
{
  m_rasterizer->display();
}

void Application::guiNewFrame()
{
  ImGui_ImplGlfwGL3_NewFrame();
}

void Application::guiReferenceManual()
{
  ImGui::ShowTestWindow();
}

void Application::guiRender()
{
  ImGui::Render();
  ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());
}

void Application::createPictures()
{
  // DAR HACK Load some hardcoded Pictures referenced by the materials.   
  unsigned int flags = IMAGE_FLAG_2D; // Load only the LOD into memory.

  Picture* picture = new Picture();
  auto ok = picture->load(std::string("./eye.tga"), flags);
  MY_ASSERT(ok);
  m_mapPictures[std::string("eye")] = picture;

  picture = new Picture();
  ok = picture->load(std::string("./head.tga"), flags);
  MY_ASSERT(ok);
  m_mapPictures[std::string("head")] = picture; // The map owns the pointers. //std::string("albedo")

  picture = new Picture();
  ok = picture->load(std::string("./NVIDIA_Logo.jpg"), flags);
  MY_ASSERT(ok);
  m_mapPictures[std::string("Albedo")] = picture; // The map owns the pointers. //std::string("albedo")

  picture = new Picture();
  ok = picture->load(std::string("./slots_alpha.png"), flags);
  MY_ASSERT(ok);
  m_mapPictures[std::string("Cutout")] = picture;

  if (m_miss == 2 && !m_environment.empty())
  {
    flags |= IMAGE_FLAG_ENV; // Special case for the spherical environment.
    picture = new Picture();
    picture->load(m_environment, flags);
    m_mapPictures[std::string("environment")] = picture;
  }
}

void Application::createCameras()
{
  CameraDefinition camera;

  m_camera.getFrustum(camera.P, camera.U, camera.V, camera.W, true);

  m_cameras.push_back(camera);
}
  
void Application::createLights()
{
  LightDefinition light;

  // Unused in environment lights. 
  light.position = make_float3(0.0f, 0.0f, 0.0f);
  light.vecU     = make_float3(1.0f, 0.0f, 0.0f);
  light.vecV     = make_float3(0.0f, 1.0f, 0.0f);
  light.normal   = make_float3(0.0f, 0.0f, 1.0f);
  light.area     = 1.0f;
  light.emission = make_float3(1.0f, 1.0f, 1.0f);
  light.lighting_activated = 1;
  
  // The environment light is expected in sysData.lightDefinitions[0], but since there is only one, 
  // the sysData struct contains the data for the spherical HDR environment light when enabled.
  // All other lights are indexed by their position inside the array.
  switch (m_miss)
  {
  case 0: // No environment light at all. Faster than a zero emission constant environment!
  default:
    break;

  case 1: // Constant environment light.
  case 2: // HDR Environment mapping with loaded texture. Texture handling happens in the Raytracer::initTextures().
    light.type = LIGHT_ENVIRONMENT;
    light.area = 4.0f * M_PIf; // Unused.

    m_lights.push_back(light); // The environment light is always in m_lights[0].
    break;
  }

  const int indexLight = static_cast<int>(m_lights.size());
  float3 normal;
  for (int i=0;i<m_area_light.size();++i)
  {
      auto area_light = m_area_light.at(i);
      switch (area_light)
      {
      case 0: // No area light.
      default:
          break;

      case 1: // Add a 1x1 square area light over the scene objects at y = 1.95 to fit into a 2x2x2 box.
          light.type = LIGHT_PARALLELOGRAM;              // A geometric area light with diffuse emission distribution function.
          light.location = float(area_light);//front
          light.position = make_float3(-6.f, 15.f, 25.f); // Corner position.
          light.vecU = make_float3(0.0f, 8.0f, -8.0f);    // To the right.
          light.vecV = make_float3(sqrtf(128), 0.0f, 0.0f);    // To the front. 
          normal = cross(light.vecU, light.vecV);   // Length of the cross product is the area.
          light.area = length(normal);                  // Calculate the world space area of that rectangle, unit is [m^2]
          light.normal = normal / light.area;             // Normalized normal
          light.emission = make_float3(12.0f);              // Radiant exitance in Watt/m^2.
          m_lights.push_back(light);
          break;

      case 2:
          light.type = LIGHT_PARALLELOGRAM;              // A geometric area light with diffuse emission distribution function.
          light.location = float(area_light);//back
          light.position = make_float3(3.f, 10.f, -15.f); // Corner position.
          light.vecU = make_float3(0.0f, 4.0f, 4.0f);    // To the right.
          light.vecV = make_float3(-5.657f, 0.0f, 0.0f);    // To the front. 
          normal = cross(light.vecU, light.vecV);   // Length of the cross product is the area.
          light.area = length(normal);                  // Calculate the world space area of that rectangle, unit is [m^2]
          light.normal = normal / light.area;             // Normalized normal
          light.emission = make_float3(12.0f);              // Radiant exitance in Watt/m^2.
          m_lights.push_back(light);
          break;

      case 3: // Add a 1x1 square area light over the scene objects at y = 1.95 to fit into a 2x2x2 box.
          light.type = LIGHT_PARALLELOGRAM;              // A geometric area light with diffuse emission distribution function.
          light.location = float(area_light);//left
          light.position = make_float3(18.f, 8.f, -5.f); // Corner position.
          light.vecU = make_float3(-4.0f, 4.0f, 0.0f);    // To the right.
          light.vecV = make_float3(0.0f, 0.0f, -5.657f);    // To the front. 
          normal = cross(light.vecU, light.vecV);   // Length of the cross product is the area.
          light.area = length(normal);                  // Calculate the world space area of that rectangle, unit is [m^2]
          light.normal = normal / light.area;             // Normalized normal
          light.emission = make_float3(12.0f);              // Radiant exitance in Watt/m^2.
          m_lights.push_back(light);
          break;

      case 4: // Add a 4x4 square area light over the scene objects at y = 4.0.
          light.type = LIGHT_PARALLELOGRAM;             // A geometric area light with diffuse emission distribution function.
          light.location = float(area_light);//right
          light.position = make_float3(-18.0f, 8.0f, -10.657f); // Corner position.
          light.vecU = make_float3(4.0f, 4.0f, 0.0f);   // To the right.
          light.vecV = make_float3(0.0f, 0.0f, 5.657f);   // To the front. 
          normal = cross(light.vecU, light.vecV);   // Length of the cross product is the area.
          light.area = length(normal);                  // Calculate the world space area of that rectangle, unit is [m^2]
          light.normal = normal / light.area;             // Normalized normal
          light.emission = make_float3(12.0f);              // Radiant exitance in Watt/m^2.
          m_lights.push_back(light);
          break;

      case 5: // Add a 4x4 square area light over the scene objects at y = 4.0.
          light.type = LIGHT_PARALLELOGRAM;             // A geometric area light with diffuse emission distribution function.
          light.location = float(area_light); //top
          light.position = make_float3(-2.0f, 15.0f, -2.0f); // Corner position.
          light.vecU = make_float3(4.0f, 0.0f, 0.0f);   // To the right.
          light.vecV = make_float3(0.0f, 0.0f, 4.0f);   // To the front. 
          normal = cross(light.vecU, light.vecV);   // Length of the cross product is the area.
          light.area = length(normal);                  // Calculate the world space area of that rectangle, unit is [m^2]
          light.normal = normal / light.area;             // Normalized normal
          light.emission = make_float3(12.0f);              // Radiant exitance in Watt/m^2.
          m_lights.push_back(light);
          break;
      }

      if (0 < area_light) // If there is an area light in the scene
      {

          // Create a material for this light.
          std::string reference = "optix_hair_area_light_" + std::to_string(i);

          const int indexMaterial = static_cast<int>(m_materialsGUI.size());

          MaterialGUI materialGUI;

          materialGUI.name = reference;
          materialGUI.indexBSDF = INDEX_BRDF_SPECULAR;
          materialGUI.albedo = make_float3(0.0f); // Black
          materialGUI.roughness = make_float2(0.1f);
          materialGUI.absorptionColor = make_float3(1.0f); // White means no absorption.
          materialGUI.absorptionScale = 0.0f;              // 0.0f means no absoption.
          materialGUI.ior = 1.5f;
          materialGUI.thinwalled = true;


          m_materialsGUI.push_back(materialGUI); // at indexMaterial.

          m_mapMaterialReferences[reference] = indexMaterial;

          // Create the Triangles for this parallelogram light.
          m_mapGeometries[reference] = m_idGeometry;

          std::shared_ptr<sg::Triangles> geometry(new sg::Triangles(m_idGeometry++));
          geometry->createParallelogram(light.position, light.vecU, light.vecV, light.normal);
          
          m_geometries.push_back(geometry);

          std::shared_ptr<sg::Instance> instance(new sg::Instance(m_idInstance++));
          // instance->setTransform(trafo); // Instance default matrix is identity.
          instance->setChild(geometry);
          instance->setMaterial(indexMaterial);
          instance->setLight(indexLight);
          instance->set_activation(true);
          m_scene->addChild(instance);
      }
  }
}

void Application::guiEventHandler()
{
  ImGuiIO const& io = ImGui::GetIO();

  if (ImGui::IsKeyPressed(' ', false)) // Key Space: Toggle the GUI window display.
  {
    m_isVisibleGUI = !m_isVisibleGUI;
  }
  if (ImGui::IsKeyPressed('S', false)) // Key S: Save the current system options to a file "system_optix_hair_<year><month><day>_<hour><minute><second>_<millisecond>.txt"
  {
    MY_VERIFY( saveSystemDescription() );
  }
  if (ImGui::IsKeyPressed('P', false)) // Key P: Save the current output buffer with tonemapping into a *.png file.
  {
    MY_VERIFY( screenshot(true) );
  }
  if (ImGui::IsKeyPressed('H', false)) // Key H: Save the current linear output buffer into a *.hdr file.
  {
    MY_VERIFY( screenshot(false) );
  }
  if (ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape), false))
  {
      if (m_is_fullscreen)
      {
          glfwSetWindowMonitor(m_window, nullptr, 100, 100, 1400, 900, 0);
          m_is_fullscreen = false;
      }
  }
  const ImVec2 mousePosition = ImGui::GetMousePos(); // Mouse coordinate window client rect.
  const int x = int(mousePosition.x);
  const int y = int(mousePosition.y);
  if (m_lock_camera)
      return;
  switch (m_guiState)
  {
    case GUI_STATE_NONE:
      if (!io.WantCaptureMouse) // Only allow camera interactions to begin when interacting with the GUI.
      {
        if (ImGui::IsMouseDown(0)) // LMB down event?
        {
          m_camera.setBaseCoordinates(x, y);
          m_guiState = GUI_STATE_ORBIT;
        }
        else if (ImGui::IsMouseDown(1)) // RMB down event?
        {
          m_camera.setBaseCoordinates(x, y);
          m_guiState = GUI_STATE_DOLLY;
        }
        else if (ImGui::IsMouseDown(2)) // MMB down event?
        {
          m_camera.setBaseCoordinates(x, y);
          m_guiState = GUI_STATE_PAN;
        }
        else if (io.MouseWheel != 0.0f) // Mouse wheel zoom.
        {
          m_camera.zoom(io.MouseWheel);
        }
      }
      break;

    case GUI_STATE_ORBIT:
      if (ImGui::IsMouseReleased(0)) // LMB released? End of orbit mode.
      {
        m_guiState = GUI_STATE_NONE;
      }
      else
      {
        m_camera.orbit(x, y);
      }
      break;

    case GUI_STATE_DOLLY:
      if (ImGui::IsMouseReleased(1)) // RMB released? End of dolly mode.
      {
        m_guiState = GUI_STATE_NONE;
      }
      else
      {
        m_camera.dolly(x, y);
      }
      break;

    case GUI_STATE_PAN:
      if (ImGui::IsMouseReleased(2)) // MMB released? End of pan mode.
      {
        m_guiState = GUI_STATE_NONE;
      }
      else
      {
        m_camera.pan(x, y);
      }
      break;
  }
}

std::string roundFloat(float var)
{
   std::string temp = std::to_string(var);
   temp.resize(temp.size() - 4);
   return temp;
}

// Demonstrate create a window with multiple child windows.
void Application::ShowOptionLayout(bool* p_open)
{   
    ImGui::SetNextWindowSize(ImVec2(500, 440), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Option", p_open, ImGuiWindowFlags_MenuBar))
    {        
        // to do option : system, camera, light, 
        ImGui::PushItemWidth(-140); // Right-aligned, keep pixels for the labels.
        if (!m_isVisibleGUI || m_mode == 1) // Use SPACE to toggle the display of the GUI window.
        {
            return;
        }

        bool refresh = false;
        bool clicked = true;
        if (ImGui::CollapsingHeader("System"))
        {
            if (ImGui::DragFloat("Mouse Ratio", &m_mouseSpeedRatio, 0.1f, 0.1f, 1000.0f, "%.1f"))
            {
                m_camera.setSpeedRatio(m_mouseSpeedRatio);
            }
            if (ImGui::Checkbox("Present", &m_present))
            {
                // No action needed, happens automatically.
            }
            if (ImGui::Combo("Camera", (int*)&m_lensShader, "Pinhole\0Fisheye\0Spherical\0\0"))
            {
                m_state.lensShader = m_lensShader;
                m_raytracer->updateState(m_state);
                refresh = true;
            }
            if (ImGui::Checkbox("Lock the camera", &m_lock_camera)){}
            ImGui::Text("Camera POV");
            ImGui::SameLine();
            ImVec2 sz(25, 25);
            if (ImGui::Button("<", sz))
            {
                if (m_current_camera == 0)
        m_current_camera = static_cast<int>(m_camera_pov.size() - 1);
                else
                    m_current_camera--;
                m_camera.m_phi = m_camera_pov[m_current_camera].m_phi;
                m_camera.m_theta = m_camera_pov[m_current_camera].m_theta;
                m_camera.m_fov = m_camera_pov[m_current_camera].m_fov;
                m_camera.m_distance = m_camera_pov[m_current_camera].m_distance;
                m_camera.markDirty();
            };
            ImGui::SameLine();
            if (ImGui::Button(">", sz))
            {
                if (m_current_camera == (m_camera_pov.size() - 1))
                    m_current_camera = 0;
                else
                    m_current_camera++;
                m_camera.m_phi = m_camera_pov[m_current_camera].m_phi;
                m_camera.m_theta = m_camera_pov[m_current_camera].m_theta;
                m_camera.m_fov = m_camera_pov[m_current_camera].m_fov;
                m_camera.m_distance = m_camera_pov[m_current_camera].m_distance;
                m_camera.markDirty();
            };
            if (ImGui::InputInt2("Resolution", &m_resolution.x, ImGuiInputTextFlags_EnterReturnsTrue)) // This requires RETURN to apply a new value.
            {
                m_resolution.x = std::max(1, m_resolution.x);
                m_resolution.y = std::max(1, m_resolution.y);

                m_camera.setResolution(m_resolution.x, m_resolution.y);
                m_rasterizer->setResolution(m_resolution.x, m_resolution.y);
                m_state.resolution = m_resolution;
                m_raytracer->updateState(m_state);
                refresh = true;
            }
            // bool ImGui::InputInt(const char* label, int* v, int step, int step_fast, ImGuiInputTextFlags extra_flags)
            if (ImGui::InputInt("SamplesSqrt", &m_samplesSqrt, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue))
            {
                m_samplesSqrt = clamp(m_samplesSqrt, 1, 256); // Samples per pixel are squares in the range [1, 65536].
                m_state.samplesSqrt = m_samplesSqrt;
                m_raytracer->updateState(m_state);
                refresh = true;
            }
            if (ImGui::DragInt2("Path Lengths", &m_pathLengths.x, 1.0f, 0, 100))
            {
                m_state.pathLengths = m_pathLengths;
                m_raytracer->updateState(m_state);
                refresh = true;
            }
            if (ImGui::DragFloat("Scene Epsilon", &m_epsilonFactor, 1.0f, 0.0f, 10000.0f))
            {
                m_state.epsilonFactor = m_epsilonFactor;
                m_raytracer->updateState(m_state);
                refresh = true;
            }
            if (ImGui::DragFloat("Env Rotation", &m_environmentRotation, 0.001f, 0.0f, 1.0f))
            {
                m_state.envRotation = m_environmentRotation;
                m_raytracer->updateState(m_state);
                refresh = true;
            }
            if (ImGui::InputInt("Number of 360 images", &m_screenshotImageNum, 1, 0, 100))
            {
                m_state.screenshotImageNum = m_screenshotImageNum;
                m_raytracer->updateState(m_state);
                refresh = true;
            }
            if (ImGui::Button("Acquire 360 rendering"))
            {
                screenshot360();
                refresh = true;
                //std::cout << "Confidence interval :" << 100.f * confidenceInterval << " %" << std::endl;
            }
            if (ImGui::Checkbox("Activate Variance Catching", &m_catchVariance))
            {
                m_state.catchVariance = m_catchVariance;
                m_raytracer->updateState(m_state);
                refresh = true;
                //std::cout << "Confidence interval :" << 100.f * confidenceInterval << " %" << std::endl;
            }
            if (m_catchVariance) {
                if (ImGui::Button("Catch Current Variance"))
                {
                    float confidenceInterval = captureVariance();
                    std::cout << "Confidence interval :" << 100.f * confidenceInterval << " %" << std::endl;
                }
            }

#if USE_TIME_VIEW
            if (ImGui::DragFloat("Clock Factor", &m_clockFactor, 1.0f, 0.0f, 1000000.0f, "%.0f"))
            {
                m_state.clockFactor = m_clockFactor;
                m_raytracer->updateState(m_state);
                refresh = true;
            }
#endif
        }

#if !USE_TIME_VIEW
        if (ImGui::CollapsingHeader("Tonemapper"))
        {
            bool changed = false;
            if (ImGui::ColorEdit3("Balance", (float*)&m_tonemapperGUI.colorBalance))
            {
                changed = true;
            }
            if (ImGui::DragFloat("Gamma", &m_tonemapperGUI.gamma, 0.01f, 0.01f, 10.0f)) // Must not get 0.0f
            {
                changed = true;
            }
            if (ImGui::DragFloat("White Point", &m_tonemapperGUI.whitePoint, 0.01f, 0.01f, 255.0f, "%.2f", 2.0f)) // Must not get 0.0f
            {
                changed = true;
            }
            if (ImGui::DragFloat("Burn Lights", &m_tonemapperGUI.burnHighlights, 0.01f, 0.0f, 10.0f, "%.2f"))
            {
                changed = true;
            }
            if (ImGui::DragFloat("Crush Blacks", &m_tonemapperGUI.crushBlacks, 0.01f, 0.0f, 1.0f, "%.2f"))
            {
                changed = true;
            }
            if (ImGui::DragFloat("Saturation", &m_tonemapperGUI.saturation, 0.01f, 0.0f, 10.0f, "%.2f"))
            {
                changed = true;
            }
            if (ImGui::DragFloat("Brightness", &m_tonemapperGUI.brightness, 0.01f, 0.0f, 100.0f, "%.2f", 2.0f))
            {
                changed = true;
            }
            if (changed)
            {
                m_rasterizer->setTonemapper(m_tonemapperGUI); // This doesn't need a refresh.
            }
        }
#endif // !USE_TIME_VIEW

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
                        m_raytracer->updateMaterial(i, materialGUI);
                        refresh = true;
                    }
                    ImGui::TreePop();
                }
            }
        }
        if (ImGui::CollapsingHeader("Lights"))
        {
            for (int i = 0; i < static_cast<int>(m_lights.size()); ++i)
            {
                LightDefinition& light = m_lights[i];

                // Allow to change the emission (radiant exitance in Watt/m^2 of the rectangle lights in the scene.
                if (light.type == LIGHT_PARALLELOGRAM)
                {
                    if (ImGui::TreeNode((void*)(intptr_t)i, "Light %d", i))
                    {
                        if (ImGui::DragFloat3("Emission", (float*)&light.emission, 0.1f, 0.0f, 10000.0f, "%.1f"))
                        {
                            m_raytracer->updateLight(i, light);
                            refresh = true;
                        }
                        ImGui::TreePop();
                    }
                }
                if (light.type == LIGHT_ENVIRONMENT) //PSAN Add 
                {
                    if (ImGui::TreeNode((void*)(intptr_t)i, "Light %d", i))
                    {
                        if (ImGui::SliderFloat("Emission Environment", (float*)&light.emission, 0.0f, 10000.0f, "%.1f")) // TO DO  CUdeviceptr OptixDenoiserParams::hdrIntensity
                        {
                            m_raytracer->updateLight(i, light);
                            refresh = true;
                        }
                        ImGui::TreePop();
                    }
                }
            }
        }

        ImGui::PopItemWidth();       
    
	    ImGui::End();

	    if (refresh)
	    {
	        restartRendering();
	    }
	}
}

void Application::chargeSettingsFromFile(std::string filename)
{
    Parser parser;

    if (!parser.load(filename))
    {
        std::cerr << "ERROR: loadSystemDescription() failed in loadString(" << filename << ")\n";
        return;
    }

    ParserTokenType tokenType;
    std::string token;

    while ((tokenType = parser.getNextToken(token)) != PTT_EOF)
    {
        if (token == "Melanine_Concentration")
        {
            for (int i = 0; i < 10; i++)
            {
                parser.getNextToken(token);
                m_melanineConcentration[i] = std::stof(token);
            }
        }
        if (token == "Melanine_Ratio")
        {
            for (int i = 0; i < 10; i++)
            {
                parser.getNextToken(token);
                m_melanineRatio[i] = std::stof(token);
            }
        }
        if (token == "Factor_Colorant_HT")
        {
            for (int i = 0; i < 10; i++)
            {
                parser.getNextToken(token);
                m_factorColorantHT[i] = std::stof(token);
            }
        }
        if (token == "Dye_Neutral_HT_Concentration")
        {
            for (int i = 0; i < 10; i++)
            {
                parser.getNextToken(token);
                m_dyeNeutralHT_Concentration[i] = std::stof(token);
            }
        }
        if (token == "Dye_Neutral_HT")
        {
            for (int i = 0; i < 10; i++)
            {
                parser.getNextToken(token);
                float x = 1.0f, y = 1.0f, z = 1.0f;
                std::string::size_type dotposition, previousdotposition;
                dotposition = token.find_first_of(";", 0);
                if (dotposition != std::string::npos)
                {
                    x = std::stof(token.substr(0, dotposition));
                    previousdotposition = dotposition + 1;
                    dotposition = token.find_first_of(";", previousdotposition);
                    if (dotposition != std::string::npos)
                    {
                        y = std::stof(token.substr(previousdotposition, dotposition));
                        previousdotposition = dotposition + 1;
                        z = std::stof(token.substr(previousdotposition, std::string::npos));
                    }
                }
                m_dyeNeutralHT[i] = make_float3(x, y, z);
            }
        }
        if (token == "Lightened_x10")
        {
            for (int i = 0; i < 10; i++)
            {
                parser.getNextToken(token);
                m_lightened_x10[i] = std::stof(token);
            }
        }
        if (token == "Lightened_x2")
        {
            for (int i = 0; i < 10; i++)
            {
                parser.getNextToken(token);
                m_lightened_x2[i] = std::stof(token);
            }
        }
        if (token == "Lightened_x1")
        {
            for (int i = 0; i < 10; i++)
            {
                parser.getNextToken(token);
                m_lightened_x1[i] = std::stof(token);
            }
        }
        if (token == "Lightened")
        {
            parser.getNextToken(token);
            lightened = std::stof(token);
        }
        if (token == "Concentration_cendre")
        {
            for (int i = 0; i < 4; i++)
            {
                parser.getNextToken(token);
                m_concentrationCendre[i] = std::stof(token);
            }
        }
        if (token == "Concentration_irise")
        {
            for (int i = 0; i < 4; i++)
            {
                parser.getNextToken(token);
                m_concentrationIrise[i] = std::stof(token);
            }
        }
        if (token == "Concentration_dore")
        {
            for (int i = 0; i < 4; i++)
            {
                parser.getNextToken(token);
                m_concentrationDore[i] = std::stof(token);
            }
        }
        if (token == "Concentration_cuivre")
        {
            for (int i = 0; i < 4; i++)
            {
                parser.getNextToken(token);
                m_concentrationCuivre[i] = std::stof(token);
            }
        }
        if (token == "Concentration_rouge")
        {
            for (int i = 0; i < 4; i++)
            {
                parser.getNextToken(token);
                m_concentrationRouge[i] = std::stof(token);
            }
        }
        if (token == "Concentration_vert")
        {
            for (int i = 0; i < 4; i++)
            {
                parser.getNextToken(token);
                m_concentrationVert[i] = std::stof(token);
            }
        }
        if (token == "dye_concentration_vert")
        {
            for (int i = 0; i < 4; i++)
            {
                parser.getNextToken(token);
                m_dye_ConcentrationVert[i] = std::stof(token);
            }
        }
        if (token == "dye_concentration_rouge")
        {
            for (int i = 0; i < 4; i++)
            {
                parser.getNextToken(token);
                m_dye_ConcentrationRouge[i] = std::stof(token);
            }
        }
        if (token == "dye_concentration_cender")
        {
            for (int i = 0; i < 4; i++)
            {
                parser.getNextToken(token);
                m_dye_ConcentrationCender[i] = std::stof(token);
            }
        }
        if (token == "dye_concentration_cover")
        {
            for (int i = 0; i < 4; i++)
            {
                parser.getNextToken(token);
                m_dye_ConcentrationCover[i] = std::stof(token);
            }
        }
        if (token == "dye_concentration_ash")
        {
            for (int i = 0; i < 4; i++)
            {
                parser.getNextToken(token);
                m_dye_ConcentrationAsh[i] = std::stof(token);
            }
        }
        if (token == "dye_concentration_gold")
        {
            for (int i = 0; i < 4; i++)
            {
                parser.getNextToken(token);
                m_dye_ConcentrationGold[i] = std::stof(token);
            }
        }
    }
    hasChanged = false;
}

void Application::ShowAbsolueLayout(bool* p_open)
{
    bool refresh = false;
    static bool closable_group = false;

    ImGui::SetNextWindowSize(ImVec2(500, 440), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("TEST INTERFACE", p_open, ImGuiWindowFlags_MenuBar))
    {
        ImGui::PushItemWidth(-120); // Right-aligned, keep pixels for the labels.

        ImGui::Checkbox("HT Fondamental setting", &closable_group);

        if (ImGui::CollapsingHeader("Material"))
        {
            for (int i = 0; i < static_cast<int>(m_materialsGUI.size()); ++i)
            {
                bool changed = false;

                MaterialGUI& materialGUI = m_materialsGUI[i];
                if (materialGUI.name.find("default") != std::string::npos
                    || materialGUI.name.find("Hair") != std::string::npos)
                {
                    if (ImGui::TreeNode((void*)(intptr_t)i, "%s", materialGUI.name.c_str()))
                {
                    if (ImGui::Combo("BxDF Type", (int*)&materialGUI.indexBSDF, "BRDF Diffuse\0BRDF Specular\0BSDF Specular\0BRDF GGX Smith\0BSDF GGX Smith\0BSDF Hair\0\0"))
                    {
                        changed = true;
                    }
                    if (materialGUI.indexBSDF == INDEX_BCSDF_HAIR)
                    {
                        //Interface with 7 colors
                        float pasAffinage(0.25f);
                        ImVec2 sz(20, 20); //button size + -

                        if (ImGui::SliderFloat("Melanine", &materialGUI.melanin_concentration, 0.f, 8.0f)) //Slider of "Hauteur de ton"
                        {
                            changed = true;
                        }
                        ImGui::Text("Dye Concentration - Dye color");
                        if (ImGui::SliderFloat("", &materialGUI.dye_concentration, 0.0f, 5.0f, "%.2f"))
                        {
                            changed = true;
                        }
                        ImGui::SameLine();
                        if (ImGui::ColorEdit3("", (float*)&materialGUI.dye, ImGuiColorEditFlags_NoInputs))
                        {
                            changed = true;
                        }

                        ImGui::PushID("Cendre");
                        ImGui::Text("Concentration Cendre");
                        ImGui::PushItemWidth(180);
                        if (ImGui::SliderFloat("1", &materialGUI.concentrationCendre, 0.0f, 5.0f))
                        {
                            changed = true;
                        }
                        ImGui::PopItemWidth();
                        ImGui::SameLine();

                        if (ImGui::Button("-", sz))
                        {
                            if (materialGUI.concentrationCendre == 0.0f)
                            {
                                materialGUI.concentrationCendre = 0.0f;
                            }
                            else
                            {
                                materialGUI.concentrationCendre -= pasAffinage;
                                changed = true;
                            }
                        };
                        ImGui::SameLine();
                        if (ImGui::Button("+", sz))
                        {
                            if (materialGUI.concentrationCendre == 5.0f)
                            {
                                materialGUI.concentrationCendre = 5.0f;
                            }
                            else
                            {
                                materialGUI.concentrationCendre += pasAffinage;
                                changed = true;
                            }
                        };
                        ImGui::SameLine();
                        if (ImGui::ColorEdit3("Cendre", (float*)&materialGUI.cendre, ImGuiColorEditFlags_NoInputs))
                        {
                            changed = true;
                        }
                        ImGui::PopID();

                        ImGui::Text("Concentration Irise");
                        ImGui::PushID("Irise");
                        ImGui::PushItemWidth(180);
                        if (ImGui::SliderFloat("2", &materialGUI.concentrationIrise, 0.0, 5.0))
                        {
                            changed = true;
                        }
                        ImGui::PopItemWidth();
                        ImGui::SameLine();
                        if (ImGui::Button("-", sz))
                        {
                            if (materialGUI.concentrationIrise == 0.0f)
                            {
                                materialGUI.concentrationIrise = 0.0f;
                            }
                            else
                            {
                                materialGUI.concentrationIrise -= pasAffinage;
                            }
                        };
                        ImGui::SameLine();
                        if (ImGui::Button("+", sz))
                        {
                            if (materialGUI.concentrationIrise == 5.0f)
                            {
                                materialGUI.concentrationIrise = 5.0f;
                            }
                            else
                            {
                                materialGUI.concentrationIrise += pasAffinage;
                            }
                        };
                        ImGui::SameLine();
                        if (ImGui::ColorEdit3("Irise", (float*)&materialGUI.irise, ImGuiColorEditFlags_NoInputs))
                        {
                            changed = true;
                        }
                        ImGui::PopID();

                        ImGui::PushID("Dore");
                        ImGui::Text("Concentration dore");
                        ImGui::PushItemWidth(180);
                        if (ImGui::SliderFloat("3", &materialGUI.concentrationDore, 0.0, 5.0))
                        {
                            changed = true;
                        }
                        ImGui::PopItemWidth();

                        ImGui::SameLine();
                        if (ImGui::Button("-", sz))
                        {
                            if (materialGUI.concentrationDore == 0.0f)
                            {
                                materialGUI.concentrationDore = 0.0f;
                            }
                            else
                            {
                                materialGUI.concentrationDore -= pasAffinage;
                            }
                        };
                        ImGui::SameLine();
                        if (ImGui::Button("+", sz))
                        {
                            if (materialGUI.concentrationDore == 5.0f)
                            {
                                materialGUI.concentrationDore = 5.0f;
                            }
                            else
                            {
                                materialGUI.concentrationDore += pasAffinage;
                            }
                        };
                        ImGui::SameLine();
                        if (ImGui::ColorEdit3("Dore", (float*)&materialGUI.doree, ImGuiColorEditFlags_NoInputs))
                        {
                            changed = true;
                        }
                        ImGui::PopID();

                        ImGui::PushID("Cuivre");
                        ImGui::Text("Concentration Cuivre");
                        ImGui::PushItemWidth(180);
                        if (ImGui::SliderFloat("4", &materialGUI.concentrationCuivre, 0.0f, 5.0f))
                        {
                            changed = true;
                        }
                        ImGui::PopItemWidth();

                        ImGui::SameLine();
                        if (ImGui::Button("-", sz))
                        {
                            if (materialGUI.concentrationCuivre == 0.0f)
                            {
                                materialGUI.concentrationCuivre = 0.0f;
                            }
                            else
                            {
                                materialGUI.concentrationCuivre -= pasAffinage;
                            }
                        };
                        ImGui::SameLine();

                        if (ImGui::Button("+", sz))
                        {
                            if (materialGUI.concentrationCuivre == 5.0f)
                            {
                                materialGUI.concentrationCuivre = 5.0f;
                            }
                            else
                            {
                                materialGUI.concentrationCuivre += pasAffinage;
                            }
                        };
                        ImGui::SameLine();

                        if (ImGui::ColorEdit3("Cuivre", (float*)&materialGUI.cuivre, ImGuiColorEditFlags_NoInputs))
                        {
                            changed = true;
                        }
                        ImGui::PopID(); // end off ID cuivre

                        ImGui::PushID("acajou");
                        ImGui::Text("Concentration acajou");
                        ImGui::PushItemWidth(180);
                        if (ImGui::SliderFloat("5", &materialGUI.concentrationAcajou, 0.0, 5.0))
                        {
                            changed = true;
                        }
                        ImGui::PopItemWidth();

                        ImGui::SameLine();
                        if (ImGui::Button("-", sz))
                        {
                            if (materialGUI.concentrationAcajou == 0.0f)
                            {
                                materialGUI.concentrationAcajou = 0.0f;
                            }
                            else
                            {
                                materialGUI.concentrationAcajou -= pasAffinage;
                            }
                        };
                        ImGui::SameLine();

                        if (ImGui::Button("+", sz))
                        {
                            if (materialGUI.concentrationAcajou == 5.0f)
                            {
                                materialGUI.concentrationAcajou = 5.0f;
                            }
                            else
                            {
                                materialGUI.concentrationAcajou += pasAffinage;
                            }
                        };
                        ImGui::SameLine();

                        if (ImGui::ColorEdit3("acajou", (float*)&materialGUI.acajou, ImGuiColorEditFlags_NoInputs))
                        {
                            changed = true;
                        }

                        ImGui::PopID();

                        ImGui::PushID("Rouge");
                        ImGui::Text("Concentration Rouge");
                        ImGui::PushItemWidth(180);
                        if (ImGui::SliderFloat("6", &materialGUI.concentrationRouge, 0.0, 5.0)) //tester la fonction clamp
                        {
                            changed = true;
                        }
                        ImGui::PopItemWidth();

                        ImGui::SameLine();
                        if (ImGui::Button("-", sz))
                        {
                            if (materialGUI.concentrationRouge == 0.0f)
                            {
                                materialGUI.concentrationRouge = 0.0f;
                            }
                            else
                            {
                                materialGUI.concentrationRouge -= pasAffinage;
                            }
                        };
                        ImGui::SameLine();

                        if (ImGui::Button("+", sz))
                        {
                            if (materialGUI.concentrationRouge == 5.0f)
                            {
                                materialGUI.concentrationRouge = 5.0f;
                            }
                            else
                            {
                                materialGUI.concentrationRouge += pasAffinage;
                            }
                        };
                        ImGui::SameLine();

                        if (ImGui::ColorEdit3("Rouge", (float*)&materialGUI.red, ImGuiColorEditFlags_NoInputs))
                        {
                            changed = true;
                        }

                        ImGui::PopID();

                        ImGui::PushID("Vert");
                        ImGui::Text("Concentration vert");
                        ImGui::PushItemWidth(180);
                        if (ImGui::SliderFloat("7", &materialGUI.concentrationVert, 0.0, 5.0))
                        {
                            changed = true;
                        }
                        ImGui::PopItemWidth();

                        ImGui::SameLine();
                        if (ImGui::Button("-", sz))
                        {
                            if (materialGUI.concentrationVert == 0.0f)
                            {
                                materialGUI.concentrationVert = 0.0f;
                            }
                            else
                            {
                                materialGUI.concentrationVert -= pasAffinage;
                            }
                        };
                        ImGui::SameLine();
                        if (ImGui::Button("+", sz))
                        {
                            if (materialGUI.concentrationVert == 5.0f)
                            {
                                materialGUI.concentrationVert = 5.0f;
                            }
                            else
                            {
                                materialGUI.concentrationVert += pasAffinage;
                            }
                        };
                        ImGui::SameLine();
                        if (ImGui::ColorEdit3("Vert", (float*)&materialGUI.vert, ImGuiColorEditFlags_NoInputs))
                        {
                            changed = true;
                        }
                        ImGui::PopID();
                    }

                    if (changed)
                    {
                        updateDYE(materialGUI);
                        m_raytracer->updateMaterial(i, materialGUI);
                        refresh = true;
                    }
                    ImGui::TreePop();
                }
                }
            }
        }
        if (ImGui::CollapsingHeader("HT Fondamental Setting", &closable_group))
        {
            ImGui::PushID("Melanine_fondamentale");
            ImGui::Text("Melanine fondamentale");
            ImGui::SliderFloat("Melanine HT 1", &m_melanineConcentration[0], 0, 8);
            ImGui::SliderFloat("Melanine HT 2", &m_melanineConcentration[1], 0, 8);
            ImGui::SliderFloat("Melanine HT 3", &m_melanineConcentration[2], 0, 8);
            ImGui::SliderFloat("Melanine HT 4", &m_melanineConcentration[3], 0, 8);
            ImGui::SliderFloat("Melanine HT 5", &m_melanineConcentration[4], 0, 8);
            ImGui::SliderFloat("Melanine HT 6", &m_melanineConcentration[5], 0, 8);
            ImGui::SliderFloat("Melanine HT 7", &m_melanineConcentration[6], 0, 8);
            ImGui::SliderFloat("Melanine HT 8", &m_melanineConcentration[7], 0, 8);
            ImGui::SliderFloat("Melanine HT 9", &m_melanineConcentration[8], 0, 8);
            ImGui::SliderFloat("Melanine HT 10", &m_melanineConcentration[9], 0, 8);
            ImGui::PopID();

            ImGui::PushID("Melanine_Ratio");
            ImGui::Text("Melanine ratio");
            ImGui::SliderFloat("Melanine HT 1", &m_melanineRatio[0], 0, 5);
            ImGui::SliderFloat("Melanine HT 2", &m_melanineRatio[1], 0, 5);
            ImGui::SliderFloat("Melanine HT 3", &m_melanineRatio[2], 0, 5);
            ImGui::SliderFloat("Melanine HT 4", &m_melanineRatio[3], 0, 5);
            ImGui::SliderFloat("Melanine HT 5", &m_melanineRatio[4], 0, 5);
            ImGui::SliderFloat("Melanine HT 6", &m_melanineRatio[5], 0, 5);
            ImGui::SliderFloat("Melanine HT 7", &m_melanineRatio[6], 0, 5);
            ImGui::SliderFloat("Melanine HT 8", &m_melanineRatio[7], 0, 5);
            ImGui::SliderFloat("Melanine HT 9", &m_melanineRatio[8], 0, 5);
            ImGui::SliderFloat("Melanine HT 10", &m_melanineRatio[9], 0, 5);
            ImGui::PopID();

            ImGui::PushID("DyeNeutral_HT");
            ImGui::Text("Dye Neutral HT");
            ImGui::SliderFloat("Melanine HT 1", &m_dyeNeutralHT_Concentration[0], 0, 8);
            ImGui::SliderFloat("Melanine HT 2", &m_dyeNeutralHT_Concentration[1], 0, 8);
            ImGui::SliderFloat("Melanine HT 3", &m_dyeNeutralHT_Concentration[2], 0, 8);
            ImGui::SliderFloat("Melanine HT 4", &m_dyeNeutralHT_Concentration[3], 0, 8);
            ImGui::SliderFloat("Melanine HT 5", &m_dyeNeutralHT_Concentration[4], 0, 8);
            ImGui::SliderFloat("Melanine HT 6", &m_dyeNeutralHT_Concentration[5], 0, 8);
            ImGui::SliderFloat("Melanine HT 7", &m_dyeNeutralHT_Concentration[6], 0, 8);
            ImGui::SliderFloat("Melanine HT 8", &m_dyeNeutralHT_Concentration[7], 0, 8);
            ImGui::SliderFloat("Melanine HT 9", &m_dyeNeutralHT_Concentration[8], 0, 8);
            ImGui::SliderFloat("Melanine HT 10", &m_dyeNeutralHT_Concentration[9], 0, 8);
            ImGui::PopID();

            ImGui::SameLine();

            //ImGui::ColorEdit3("", (float*)&dyeNeutralHT, ImGuiColorEditFlags_NoInputs);
           

        }
        if (ImGui::CollapsingHeader("Dev Setting"))
        {  
            bool changed = false;
            for (int i = 0; i < m_materialsGUI.size(); i++)
            {
                if(m_materialsGUI.at(i).indexBSDF == INDEX_BCSDF_HAIR)
                {
                    std::string tmp = "Should modify " + m_materialsGUI[i].name + " material";
                    ImGui::Checkbox(tmp.c_str(), &m_materialsGUI[i].shouldModify);
                }                
            }
            ImGui::PushID("Factor_Colorant");
            ImGui::Text("Factor Colorant");
            if(ImGui::SliderFloat("Melanine HT 1", &m_factorColorantHT[0], 0, 20))
                changed = true;
            if(ImGui::SliderFloat("Melanine HT 2", &m_factorColorantHT[1], 0, 20))
                changed = true;
            if (ImGui::SliderFloat("Melanine HT 3", &m_factorColorantHT[2], 0, 20))
                changed = true;
            if (ImGui::SliderFloat("Melanine HT 4", &m_factorColorantHT[3], 0, 20))
                changed = true;
            if (ImGui::SliderFloat("Melanine HT 5", &m_factorColorantHT[4], 0, 20))
                changed = true;
            if (ImGui::SliderFloat("Melanine HT 6", &m_factorColorantHT[5], 0, 20))
                changed = true;
            if (ImGui::SliderFloat("Melanine HT 7", &m_factorColorantHT[6], 0, 20))
                changed = true;
            if (ImGui::SliderFloat("Melanine HT 8", &m_factorColorantHT[7], 0, 20))
                changed = true;
            if (ImGui::SliderFloat("Melanine HT 9", &m_factorColorantHT[8], 0, 20))
                changed = true;
            if (ImGui::SliderFloat("Melanine HT 10", &m_factorColorantHT[9], 0, 20))
                changed = true;
            ImGui::PopID();

            ImGui::PushID("Normal");
            ImGui::Text("Lightened Normal");
            if (ImGui::SliderFloat("Lightened HT 1", &m_lightened_x1[0], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 2", &m_lightened_x1[1], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 3", &m_lightened_x1[2], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 4", &m_lightened_x1[3], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 5", &m_lightened_x1[4], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 6", &m_lightened_x1[5], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 7", &m_lightened_x1[6], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 8", &m_lightened_x1[7], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 9", &m_lightened_x1[8], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 10", &m_lightened_x1[9], 0, 8))
                changed = true;
            ImGui::PopID();

            ImGui::PushID("x2");
            ImGui::Text("Lightened x2");
            if (ImGui::SliderFloat("Lightened HT 1", &m_lightened_x2[0], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 2", &m_lightened_x2[1], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 3", &m_lightened_x2[2], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 4", &m_lightened_x2[3], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 5", &m_lightened_x2[4], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 6", &m_lightened_x2[5], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 7", &m_lightened_x2[6], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 8", &m_lightened_x2[7], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 9", &m_lightened_x2[8], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 10", &m_lightened_x2[9], 0, 8))
                changed = true;
            ImGui::PopID();

            ImGui::PushID("x10");
            ImGui::Text("Lightened x10");
            if (ImGui::SliderFloat("Lightened HT 1", &m_lightened_x10[0], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 2", &m_lightened_x10[1], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 3", &m_lightened_x10[2], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 4", &m_lightened_x10[3], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 5", &m_lightened_x10[4], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 6", &m_lightened_x10[5], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 7", &m_lightened_x10[6], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 8", &m_lightened_x10[7], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 9", &m_lightened_x10[8], 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Lightened HT 10", &m_lightened_x10[9], 0, 8))
                changed = true;
            ImGui::PopID();

            ImGui::PushID("Lightened");
            ImGui::Text("Lightened");
            if (ImGui::SliderFloat("Lightened", &lightened, 0, 8))
                changed = true;
            ImGui::PopID();

            ImGui::PushID("Dye VertRouge");
            ImGui::Text("Dye Vert Rouge");
            if (ImGui::SliderFloat4("Vert", m_dye_ConcentrationVert, 0, 4))
                changed = true;
            if (ImGui::SliderFloat4("Rouge", m_dye_ConcentrationRouge, 0, 4))
                changed = true;
            ImGui::PopID();

            ImGui::PushID("Dye CenderCover");
            ImGui::Text("Dye CenderCover");
            if (ImGui::SliderFloat4("Cender", m_dye_ConcentrationCender, 0, 4))
                changed = true;
            if (ImGui::SliderFloat4("Cover", m_dye_ConcentrationCover, 0, 4))
                changed = true;
            ImGui::PopID();

            ImGui::PushID("Dye AshGold");
            ImGui::Text("Dye Ash Gold");
            if (ImGui::SliderFloat4("Ash", m_dye_ConcentrationAsh, 0, 4))
                changed = true;
            if (ImGui::SliderFloat4("Gold", m_dye_ConcentrationGold, 0, 4))
                changed = true;
            ImGui::PopID();

            ImGui::PushID("Ponderation");
            ImGui::Text("Ponderation");
            if (ImGui::SliderFloat4("Cendre", m_concentrationCendre, 0, 4))
                changed = true;
            if (ImGui::SliderFloat4("Irise", m_concentrationIrise, 0, 4))
                changed = true;
            if (ImGui::SliderFloat4("Dore", m_concentrationDore, 0, 4))
                changed = true;
            if (ImGui::SliderFloat4("Cuivre", m_concentrationCuivre, 0, 4))
                changed = true;
            if (ImGui::SliderFloat4("Rouge", m_concentrationRouge, 0, 4))
                changed = true;
            if (ImGui::SliderFloat4("Vert", m_concentrationVert, 0, 4))
                changed = true;

            ImGui::PopID();
            if (changed)
            {
                for (int i = 0; i < static_cast<int>(m_materialsGUI.size()); ++i )
                {
                    MaterialGUI& materialGUI = m_materialsGUI[i];
                    if (materialGUI.shouldModify)
                    {
                        hasChanged = true;
                        updateDYE(materialGUI);
                        updateDYEconcentration(materialGUI);
                        //updateHT(materialGUI);
                        m_raytracer->updateMaterial(i, materialGUI);
                        refresh = true;
                    }
                }
            }
            bool clicked = false;
            static char SettingsName[30] = "Settings";
            ImGui::InputText("Settings file Name", SettingsName, IM_ARRAYSIZE(SettingsName));
            if (ImGui::Button("Save settings"))
            {
                clicked = true;
            }
            if (clicked == true)
            {
                std::string tmp = SettingsName;
                std::string Path = m_prefixSettings + tmp + ".settings";
                int i = 0;
                while (fs::exists(Path))
                {
                    tmp = "Settings" + std::to_string(i);
                    Path = m_prefixSettings + tmp + ".settings";
                    i++;
                }
                strcpy(SettingsName, tmp.c_str());
                saveSettingToFile(Path);
                hasChanged = false;
                m_settings.push_back(std::pair<std::string, std::string>(tmp, Path));
                current_settings_value = m_settings[m_settings.size() - 1].first.c_str();
                std::string newLine = "settings " + tmp + " \"" + Path + "\"";
                f_options->addCommand(newLine);
                refresh = true;
            }
        }
        ImGui::PopItemWidth();
    }   
    ImGui::End();
    if (refresh)
    {
        restartRendering();
    }
}

void Application::saveSettingToFile(std::string path)
{
    std::ofstream i_writeCmd(path, std::ios::app);
    i_writeCmd << "Melanine_Concentration\n";
    i_writeCmd << m_melanineConcentration[0] << " " << m_melanineConcentration[1] << " " << m_melanineConcentration[2] << " "
        << m_melanineConcentration[3] << " " << m_melanineConcentration[4] << " " << m_melanineConcentration[5] << " "
        << m_melanineConcentration[6] << " " << m_melanineConcentration[7] << " " << m_melanineConcentration[8] << " "
        << m_melanineConcentration[9] << "\n";
    i_writeCmd << "Melanine_Ratio\n";
    i_writeCmd << m_melanineRatio[0] << " " << m_melanineRatio[1] << " " << m_melanineRatio[2] << " "
        << m_melanineRatio[3] << " " << m_melanineRatio[4] << " " << m_melanineRatio[5] << " "
        << m_melanineRatio[6] << " " << m_melanineRatio[7] << " " << m_melanineRatio[8] << " "
        << m_melanineRatio[9] << "\n";
    i_writeCmd << "Factor_Colorant_HT\n";
    i_writeCmd << m_factorColorantHT[0] << " " << m_factorColorantHT[1] << " " << m_factorColorantHT[2] << " "
        << m_factorColorantHT[3] << " " << m_factorColorantHT[4] << " " << m_factorColorantHT[5] << " "
        << m_factorColorantHT[6] << " " << m_factorColorantHT[7] << " " << m_factorColorantHT[8] << " "
        << m_factorColorantHT[9] << "\n";
    i_writeCmd << "Dye_Neutral_HT_Concentration\n";
    i_writeCmd << m_dyeNeutralHT_Concentration[0] << " " << m_dyeNeutralHT_Concentration[1] << " " << m_dyeNeutralHT_Concentration[2] << " "
        << m_dyeNeutralHT_Concentration[3] << " " << m_dyeNeutralHT_Concentration[4] << " " << m_dyeNeutralHT_Concentration[5] << " "
        << m_dyeNeutralHT_Concentration[6] << " " << m_dyeNeutralHT_Concentration[7] << " " << m_dyeNeutralHT_Concentration[8] << " "
        << m_dyeNeutralHT_Concentration[9] << "\n";
    i_writeCmd << "Dye_Neutral_HT\n";
    i_writeCmd << m_dyeNeutralHT[0].x << ";" << m_dyeNeutralHT[0].y << ";" << m_dyeNeutralHT[0].z << "\n";
    i_writeCmd << m_dyeNeutralHT[1].x << ";" << m_dyeNeutralHT[1].y << ";" << m_dyeNeutralHT[1].z << "\n";
    i_writeCmd << m_dyeNeutralHT[2].x << ";" << m_dyeNeutralHT[2].y << ";" << m_dyeNeutralHT[2].z << "\n";
    i_writeCmd << m_dyeNeutralHT[3].x << ";" << m_dyeNeutralHT[3].y << ";" << m_dyeNeutralHT[3].z << "\n";
    i_writeCmd << m_dyeNeutralHT[4].x << ";" << m_dyeNeutralHT[4].y << ";" << m_dyeNeutralHT[4].z << "\n";
    i_writeCmd << m_dyeNeutralHT[5].x << ";" << m_dyeNeutralHT[5].y << ";" << m_dyeNeutralHT[5].z << "\n";
    i_writeCmd << m_dyeNeutralHT[6].x << ";" << m_dyeNeutralHT[6].y << ";" << m_dyeNeutralHT[6].z << "\n";
    i_writeCmd << m_dyeNeutralHT[7].x << ";" << m_dyeNeutralHT[7].y << ";" << m_dyeNeutralHT[7].z << "\n";
    i_writeCmd << m_dyeNeutralHT[8].x << ";" << m_dyeNeutralHT[8].y << ";" << m_dyeNeutralHT[8].z << "\n";
    i_writeCmd << m_dyeNeutralHT[9].x << ";" << m_dyeNeutralHT[9].y << ";" << m_dyeNeutralHT[9].z << "\n";
    i_writeCmd << "Lightened_x10\n";
    i_writeCmd << m_lightened_x10[0] << " " << m_lightened_x10[1] << " " << m_lightened_x10[2] << " "
        << m_lightened_x10[3] << " " << m_lightened_x10[4] << " " << m_lightened_x10[5] << " "
        << m_lightened_x10[6] << " " << m_lightened_x10[7] << " " << m_lightened_x10[8] << " "
        << m_lightened_x10[9] << "\n";
    i_writeCmd << "Lightened_x2\n";
    i_writeCmd << m_lightened_x2[0] << " " << m_lightened_x2[1] << " " << m_lightened_x2[2] << " "
        << m_lightened_x2[3] << " " << m_lightened_x2[4] << " " << m_lightened_x2[5] << " "
        << m_lightened_x2[6] << " " << m_lightened_x2[7] << " " << m_lightened_x2[8] << " "
        << m_lightened_x2[9] << "\n";
    i_writeCmd << "Lightened_x1\n";
    i_writeCmd << m_lightened_x1[0] << " " << m_lightened_x1[1] << " " << m_lightened_x1[2] << " "
        << m_lightened_x1[3] << " " << m_lightened_x1[4] << " " << m_lightened_x1[5] << " "
        << m_lightened_x1[6] << " " << m_lightened_x1[7] << " " << m_lightened_x1[8] << " "
        << m_lightened_x1[9] << "\n";
    i_writeCmd << "Lightened " << lightened << "\n";
    i_writeCmd << "Concentration_cendre " << m_concentrationCendre[0] << " " << m_concentrationCendre[1] << " " << m_concentrationCendre[2] << " " << m_concentrationCendre[3] << "\n";
    i_writeCmd << "Concentration_irise " << m_concentrationIrise[0] << " " << m_concentrationIrise[1] << " " << m_concentrationIrise[2] << " " << m_concentrationIrise[3] << "\n";
    i_writeCmd << "Concentration_dore " << m_concentrationDore[0] << " " << m_concentrationDore[1] << " " << m_concentrationDore[2] << " " << m_concentrationDore[3] << "\n";
    i_writeCmd << "Concentration_cuivre " << m_concentrationCuivre[0] << " " << m_concentrationCuivre[1] << " " << m_concentrationCuivre[2] << " " << m_concentrationCuivre[3] << "\n";
    i_writeCmd << "Concentration_rouge " << m_concentrationRouge[0] << " " << m_concentrationRouge[1] << " " << m_concentrationRouge[2] << " " << m_concentrationRouge[3] << "\n";
    i_writeCmd << "Concentration_vert " << m_concentrationVert[0] << " " << m_concentrationVert[1] << " " << m_concentrationVert[2] << " " << m_concentrationVert[3] << "\n";

    i_writeCmd << "dye_concentration_vert " << m_dye_ConcentrationVert[0] << " " << m_dye_ConcentrationVert[1] << " " << m_dye_ConcentrationVert[2] << " " << m_dye_ConcentrationVert[3] << "\n";
    i_writeCmd << "dye_concentration_rouge " << m_dye_ConcentrationRouge[0] << " " << m_dye_ConcentrationRouge[1] << " " << m_dye_ConcentrationRouge[2] << " " << m_dye_ConcentrationRouge[3] << "\n";
    i_writeCmd << "dye_concentration_cender " << m_dye_ConcentrationCender[0] << " " << m_dye_ConcentrationCender[1] << " " << m_dye_ConcentrationCender[2] << " " << m_dye_ConcentrationCender[3] << "\n";
    i_writeCmd << "dye_concentration_cover " << m_dye_ConcentrationCover[0] << " " << m_dye_ConcentrationCover[1] << " " << m_dye_ConcentrationCover[2] << " " << m_dye_ConcentrationCover[3] << "\n";
    i_writeCmd << "dye_concentration_ash " << m_dye_ConcentrationAsh[0] << " " << m_dye_ConcentrationAsh[1] << " " << m_dye_ConcentrationAsh[2] << " " << m_dye_ConcentrationAsh[3] << "\n";
    i_writeCmd << "dye_concentration_gold " << m_dye_ConcentrationGold[0] << " " << m_dye_ConcentrationGold[1] << " " << m_dye_ConcentrationGold[2] << " " << m_dye_ConcentrationGold[3] << "\n";
    i_writeCmd.close();

}
void Application::saveImageWindow(bool* p_open)
{
    //ImGui::ShowDemoWindow();

    static bool show_option_layout = false;
    static bool show_absolue_layout = false;

    if (show_option_layout)             ShowOptionLayout(&show_option_layout);
    if (show_absolue_layout)            ShowAbsolueLayout(&show_absolue_layout);

    bool refresh = false;

    static bool no_close = false;
    static bool no_move = true;
    static bool no_collapse = true;

    //static bool no_custom = false; //PSAN test

    ImGuiTreeNodeFlags custom_flag = 0; // PSAN test 
    ImGuiWindowFlags window_flags = 0;

    if (no_close)     p_open = NULL; // Don't pass our bool* to Begin
    if (no_move)     window_flags |= ImGuiWindowFlags_NoMove;
    if (no_collapse)  window_flags |= ImGuiWindowFlags_NoCollapse;

    //if (no_custom) custom_flag |= ImGuiTreeNodeFlags_OpenOnDoubleClick; // PSAN test

    ImGui::SetNextWindowSize(ImVec2(550, 680), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("User", p_open, window_flags))
    {
        // Early out if the window is collapsed, as an optimization.
        ImGui::End();
        return;
    }

    //ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.65f);    // 2/3 of the space for widget and 1/3 for labels
    ImGui::PushItemWidth(-140);                                 // Right align, keep 140 pixels for labels


    bool clicked = false;
    static const char* current_item_colorswatch_value = "";
    ColorSwitch* current_item_colorswatch = NULL;
    MaterialGUI* materialGUI1 = &m_materialsGUI.at(m_mapMaterialReferences.find(current_item_model->material1Name)->second);
    static int materialGUI1ID = m_mapMaterialReferences.find(current_item_model->material1Name)->second;
    static int materialGUI2ID = materialGUI1ID;

    MaterialGUI* materialGUI2 = materialGUI1;
    if (current_item_model->material1Name != current_item_model->material2Name)
        materialGUI2 = &m_materialsGUI.at(m_mapMaterialReferences.find(current_item_model->material2Name)->second);

    if (ImGui::Button("Save Images", ImVec2(100, 20)))
    {
        image_view = 0;
        looping = true;
        //std::thread th(save_image_handler, this);   // starts running
        //save_image(NULL);
    }
    
    if (ImGui::CollapsingHeader("Material", true))
    {
        int i = 0;
        for (; i < static_cast<int>(m_materialsGUI.size()); ++i)
        {
            bool changed = false;

            MaterialGUI& materialGUI = m_materialsGUI[i];

            if ((materialGUI.name.find("Hair") != std::string::npos)
                && ImGui::TreeNode((void*)(intptr_t)i, "%s", m_materialsGUI[i].name.c_str()))
            {
                if (ImGui::Combo("BxDF Type", (int*)&materialGUI.indexBSDF, "BRDF Diffuse\0BRDF Specular\0BSDF Specular\0BRDF GGX Smith\0BSDF GGX Smith\0BSDF Hair\0\0"))
                {
                    changed = true;
                }
                if (materialGUI.indexBSDF == INDEX_BCSDF_HAIR)
                {
                    float pasAffinage(0.25f);
                    ImVec2 sz(20, 20);// size button + -   

                    ImGui::PushID("HT");
                    if (ImGui::SliderInt("HT", &materialGUI.HT, 1, 10))
                    {
                        materialGUI.melanin_concentration = m_melanineConcentration[materialGUI.HT - 1];
                        materialGUI.dyeNeutralHT_Concentration = m_dyeNeutralHT_Concentration[materialGUI.HT - 1];
                        materialGUI.dyeNeutralHT = m_dyeNeutralHT[materialGUI.HT - 1];
                        materialGUI.melanin_ratio = m_melanineRatio[materialGUI.HT - 1];
                        changed = true;
                    }
                    ImGui::SameLine();

                    if (ImGui::ColorEdit3("", (float*)&materialGUI.dyeNeutralHT, ImGuiColorEditFlags_NoInputs))
                    {
                        changed = true;
                    }
                    ImGui::PopID();

                    //PSAN VERT ROUGE 
                    ImGui::Text("Vert - Rouge");
                    ImGui::PushID("Vert");
                    if (ImGui::ColorEdit3("", (float*)&materialGUI.vert, ImGuiColorEditFlags_NoInputs))
                    {
                        changed = true;
                    }
                    ImGui::PopID();
                    ImGui::SameLine();

                    ImGui::PushID("Vert-Rouge");
                    char* vertRougeIndex[] = { "70","77","7","07","","06","6","66","60" };
                    ImGui::PushItemWidth(250);
                    if (ImGui::SliderInt("", &materialGUI.int_VertRouge_Concentration, 0, 8, vertRougeIndex[materialGUI.int_VertRouge_Concentration]))
                    {
                        changed = true;
                    }
                    ImGui::PopItemWidth();
                    ImGui::PopID();

                    ImGui::PushID("Rouge");
                    ImGui::SameLine();
                    if (ImGui::ColorEdit3("", (float*)&materialGUI.red, ImGuiColorEditFlags_NoInputs))
                    {
                        changed = true;
                    }
                    ImGui::PopID();

                    // PSAN CENDRE CUIVRE
                    ImGui::Text("Cendre - Cuivre");
                    ImGui::PushID("Bleu");
                    if (ImGui::ColorEdit3("", (float*)&materialGUI.cendre, ImGuiColorEditFlags_NoInputs))
                    {
                        changed = true;
                    }
                    ImGui::PopID();
                    ImGui::SameLine();
                    ImGui::PushID("Cendre-Cuivre");
                    ImGui::PushItemWidth(250);

                    char* CendreCuivreIndex[] = { "10","11","1","01","","04","4","44","40" };
                    if (ImGui::SliderInt("", &materialGUI.int_CendreCuivre_Concentration, 0, 8, CendreCuivreIndex[materialGUI.int_CendreCuivre_Concentration]))
                    {

                        changed = true;

                    }
                    ImGui::PopItemWidth();
                    ImGui::PopID();


                    ImGui::PushID("Cuivre");
                    ImGui::SameLine();
                    if (ImGui::ColorEdit3("", (float*)&materialGUI.cuivre, ImGuiColorEditFlags_NoInputs))
                    {
                        changed = true;
                    }
                    ImGui::PopID();

                    // PSAN IRISE DORE
                    ImGui::Text("Irise - Dore");
                    ImGui::PushID("Irise");
                    if (ImGui::ColorEdit3("", (float*)&materialGUI.irise, ImGuiColorEditFlags_NoInputs))
                    {
                        changed = true;
                    }
                    ImGui::PopID();

                    ImGui::SameLine();
                    ImGui::PushID("Irise-Dore");
                    char* IriseDoreIndex[] = { "20","22","2","02","","03","3","33","30" };
                    ImGui::PushItemWidth(250);
                    if (ImGui::SliderInt("", &materialGUI.int_IriseDore_Concentration, 0, 8, IriseDoreIndex[materialGUI.int_IriseDore_Concentration]))
                    {

                        changed = true;

                    }
                    ImGui::PopItemWidth();
                    ImGui::PopID();

                    ImGui::PushID("Dore");
                    ImGui::SameLine();
                    if (ImGui::ColorEdit3("", (float*)&materialGUI.doree, ImGuiColorEditFlags_NoInputs))
                    {
                        changed = true;
                    }
                    ImGui::PopID();
                }

                if (changed)
                {
                    updateDYEinterface(materialGUI);
                    updateDYE(materialGUI);
                    updateDYEconcentration(materialGUI);
                    updateHT(materialGUI);

                    m_raytracer->updateMaterial(i, materialGUI);
                    refresh = true;
                }
                ImGui::TreePop();
            }
        }
    }
    if (ImGui::CollapsingHeader("Dynamic settings"))
    {
        bool changed = false;
        const char* current_HDR_value;
        HDRSwitch* current_item_HDR = NULL;
        for (int i = 0; i < m_HDR.size(); i++)
        {
            if (std::strcmp(m_HDR[i].file_name.c_str(), m_environment.c_str()) == 0)
            {
                current_item_HDR = &m_HDR[i];
                current_HDR_value = m_HDR[i].name.c_str();
                break;
            }
        }
        if (current_item_model->material1Name != current_item_model->material2Name)
            materialGUI2 = &m_materialsGUI.at(m_mapMaterialReferences.find(current_item_model->material2Name)->second);


        if (ImGui::BeginCombo("Hair type", current_item_model_value))
        {
            for (int n = 0; n < m_models.size(); n++)
            {
                bool is_selected = (current_item_model == &m_models[n]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(m_models[n].name.c_str(), is_selected))
                {
                    if (current_item_model != &m_models[n])
                    {
                        m_scene->removeCurvesChild();

                        std::string HairModel = current_item_model->file_name;
                        convertPath(HairModel);
                        std::ostringstream keyGeometry_delete1;
                        std::ostringstream keyGeometry_delete2;
                        if (current_item_model->material1Name == current_item_model->material2Name)
                            keyGeometry_delete1 << current_item_model->map_identifier;
                        else
                        {
                            keyGeometry_delete1 << current_item_model->map_identifier << "_half_1";
                            keyGeometry_delete2 << current_item_model->map_identifier << "_half_2";
                        }
                        std::map<std::string, unsigned int>::const_iterator itg_delete1 = m_mapGeometries.find(keyGeometry_delete1.str());
                        if (itg_delete1 != m_mapGeometries.end())
                        {
                            auto geometry_id = itg_delete1->second;
                            for (int i = 0; i < m_geometries.size(); i++)
                            {
                                if (m_geometries[i].get()->getId() == geometry_id)
                                {
                                    m_geometries.erase(m_geometries.begin() + i);
                                    m_mapGeometries.erase(keyGeometry_delete1.str());
                                    m_idGeometry--;
                                }
                            }
                        }
                        if (current_item_model->material1Name != current_item_model->material2Name)
                        {
                            std::map<std::string, unsigned int>::const_iterator itg_delete2 = m_mapGeometries.find(keyGeometry_delete2.str());
                            if (itg_delete2 != m_mapGeometries.end())
                            {
                                auto geometry_id = itg_delete2->second;
                                for (int i = 0; i < m_geometries.size(); i++)
                                {
                                    if (m_geometries[i].get()->getId() == geometry_id)
                                    {
                                        m_geometries.erase(m_geometries.begin() + i);
                                        m_mapGeometries.erase(keyGeometry_delete2.str());
                                        m_idGeometry--;
                                    }
                                }
                            }
                        }

                        current_item_model_value = m_models[n].name.c_str();
                        current_item_model = &m_models[n];

                        HairModel = current_item_model->file_name;
                        convertPath(HairModel);
                        std::ostringstream keyGeometry1;
                        std::ostringstream keyGeometry2;
                        if (current_item_model->material1Name == current_item_model->material2Name)
                            keyGeometry1 << current_item_model->map_identifier;
                        else
                        {
                            keyGeometry1 << current_item_model->map_identifier << "_half_1";
                            keyGeometry2 << current_item_model->map_identifier << "_half_2";
                        }

                        materialGUI1 = &(m_materialsGUI.at(m_mapMaterialReferences.find(current_item_model->material1Name)->second));
                        materialGUI2 = &(m_materialsGUI.at(m_mapMaterialReferences.find(current_item_model->material2Name)->second));

                        std::shared_ptr<sg::Curves> geometry_left;
                        std::map<std::string, unsigned int>::const_iterator itg1 = m_mapGeometries.find(keyGeometry1.str());
                        if (itg1 == m_mapGeometries.end())
                        {
                            m_mapGeometries[keyGeometry1.str()] = m_idGeometry;
                            geometry_left = std::make_shared<sg::Curves>(m_idGeometry++);
                            const char* file = current_item_model->file_name.c_str();
                            if (current_item_model->material1Name != current_item_model->material2Name)
                                geometry_left->createHairFromFile(file, true);
                            else
                                geometry_left->createHairFromFile(file);
                            m_geometries.push_back(geometry_left);
                        }
                        else
                            geometry_left = std::dynamic_pointer_cast<sg::Curves>(m_geometries[itg1->second]);
                        appendInstance(m_scene, geometry_left, curMatrix, current_item_model->material1Name, m_idInstance);

                        if (current_item_model->material1Name != current_item_model->material2Name)
                        {
                            std::shared_ptr<sg::Curves> geometry_right;
                            std::map<std::string, unsigned int>::const_iterator itg2 = m_mapGeometries.find(keyGeometry2.str());
                            if (itg2 == m_mapGeometries.end())
                            {
                                m_mapGeometries[keyGeometry2.str()] = m_idGeometry;
                                geometry_right = std::make_shared<sg::Curves>(m_idGeometry++);
                                const char* file = current_item_model->file_name.c_str();
                                geometry_right->createHairFromFile(file, false);
                                m_geometries.push_back(geometry_right);
                            }
                            else
                                geometry_right = std::dynamic_pointer_cast<sg::Curves>(m_geometries[itg2->second]);
                            appendInstance(m_scene, geometry_right, curMatrix, current_item_model->material2Name, m_idInstance);
                        }
                        m_raytracer->initMaterials(m_materialsGUI);
                        m_raytracer->initScene(m_scene, m_idGeometry); // m_idGeometry is the number of geometries in the scene
                        refresh = true;
                        m_isValid = true;
                        m_raytracer->updateCamera(0, m_cameras[0]);
                    }
                }
            }
            ImGui::EndCombo();
        }
        
    }
    
    ImGui::PopItemWidth();
    ImGui::End();

    if (refresh)
    {
        restartRendering();
        if (clicked)
        {
            current_item_colorswatch_value = m_materialsColor.back().name.c_str();
            current_item_colorswatch = &m_materialsColor.back();
        }
    }
}
void Application::guiUserWindow(bool* p_open)
{
    //ImGui::ShowDemoWindow();
    
    static bool show_option_layout = false;
    static bool show_absolue_layout = false;

    if (show_option_layout)             ShowOptionLayout(&show_option_layout);
    if (show_absolue_layout)            ShowAbsolueLayout(&show_absolue_layout);

    bool refresh = false;    

    static bool no_close = false;
    static bool no_move = true;
    static bool no_collapse = true; 

    //static bool no_custom = false; //PSAN test
    
    ImGuiTreeNodeFlags custom_flag = 0; // PSAN test 
    ImGuiWindowFlags window_flags = 0;

    if (no_close)     p_open = NULL; // Don't pass our bool* to Begin
    if (no_move)     window_flags |= ImGuiWindowFlags_NoMove;
    if (no_collapse)  window_flags |= ImGuiWindowFlags_NoCollapse;

    //if (no_custom) custom_flag |= ImGuiTreeNodeFlags_OpenOnDoubleClick; // PSAN test
        
    ImGui::SetNextWindowSize(ImVec2(550, 680), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("User", p_open, window_flags))
    {
        // Early out if the window is collapsed, as an optimization.
        ImGui::End();
        return;
    }

    //ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.65f);    // 2/3 of the space for widget and 1/3 for labels
    ImGui::PushItemWidth(-140);                                 // Right align, keep 140 pixels for labels


    bool clicked = false;
    static const char* current_item_colorswatch_value = "";
    ColorSwitch* current_item_colorswatch = NULL;
    MaterialGUI* materialGUI1 = &m_materialsGUI.at(m_mapMaterialReferences.find(current_item_model->material1Name)->second);
    static int materialGUI1ID = m_mapMaterialReferences.find(current_item_model->material1Name)->second;
    static int materialGUI2ID = materialGUI1ID;
   
    MaterialGUI* materialGUI2 = materialGUI1;
    if (current_item_model->material1Name != current_item_model->material2Name)
        materialGUI2 = &m_materialsGUI.at(m_mapMaterialReferences.find(current_item_model->material2Name)->second);

    if (ImGui::Button("Option", ImVec2 (60,20)))
    {
        show_option_layout = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Affinage", ImVec2(60, 20)))
    {
        show_absolue_layout = true;
    }
    if (ImGui::CollapsingHeader("Camera", true))
    {
        int tmp = m_camera.pov;
        ImGui::RadioButton("Center", &(m_camera.pov), 0);
        ImGui::SameLine();
        ImGui::RadioButton("Zoomed Center", &m_camera.pov, 1);
        ImGui::SameLine();
        ImGui::RadioButton("Top Front", &(m_camera.pov), 4);
        ImGui::RadioButton("Left", &(m_camera.pov), 2);
        ImGui::SameLine();
        ImGui::RadioButton("Right", &(m_camera.pov), 3);
        if (m_camera.pov != tmp)
        {
            switch (m_camera.pov)
            {
            case 0:
                m_camera.m_phi = 0.251406f;
                m_camera.m_theta = 0.570703f;
                m_camera.m_fov = 32.f;
                m_camera.m_distance = 10.f;
                break;
            case 1:
                m_camera.m_phi = 0.251406f;
                m_camera.m_theta = 0.570703f;
                m_camera.m_fov = 12.f;
                m_camera.m_distance = 10.f;
                break;
            case 2:
                m_camera.m_phi = 0.981875f;
                m_camera.m_theta = 0.535547;
                m_camera.m_fov = 32.f;
                m_camera.m_distance = 10.f;
                break;
            case 3:
                m_camera.m_phi = 0.5092198f;
                m_camera.m_theta = 0.521875f;
                m_camera.m_fov = 29.f;
                m_camera.m_distance = 10.f;
                break;
            case 4:
                m_camera.m_phi = 0.757265f;
                m_camera.m_theta = 0.719141f;
                m_camera.m_fov = 29.f;
                m_camera.m_distance = 10.f;
                break;
            default:
                m_camera.m_phi = 0.251406f;
                m_camera.m_theta = 0.570703f;
                m_camera.m_fov = 32.f;
                m_camera.m_distance = 10.f;
                break;
            }
            m_camera.markDirty(true);
        }

    }
    if (ImGui::CollapsingHeader("Material", true))
    {
        int i = 0;
        for (; i < static_cast<int>(m_materialsGUI.size()); ++i)
        {
            bool changed = false;
            
            MaterialGUI& materialGUI = m_materialsGUI[i];
                        
            if ((materialGUI.name.find("default") != std::string::npos
                || materialGUI.name.find("Hair") != std::string::npos)
                && ImGui::TreeNode((void*)(intptr_t)i, "%s", m_materialsGUI[i].name.c_str()))
            {
                if (ImGui::Combo("BxDF Type", (int*)&materialGUI.indexBSDF, "BRDF Diffuse\0BRDF Specular\0BSDF Specular\0BRDF GGX Smith\0BSDF GGX Smith\0BSDF Hair\0\0"))
                {
                    changed = true;
                }
                if (materialGUI.indexBSDF == INDEX_BCSDF_HAIR)
                {
                    float pasAffinage(0.25f);
                    ImVec2 sz(20, 20);// size button + -   

                    ImGui::PushID("HT");
                    if (ImGui::SliderInt("HT", &materialGUI.HT, 1, 10))
                    {
                        materialGUI.melanin_concentration = m_melanineConcentration[materialGUI.HT-1];
                        materialGUI.dyeNeutralHT_Concentration = m_dyeNeutralHT_Concentration[materialGUI.HT-1];
                        materialGUI.dyeNeutralHT = m_dyeNeutralHT[materialGUI.HT-1];
                        materialGUI.melanin_ratio = m_melanineRatio[materialGUI.HT - 1];
                        changed = true;
                    }
                    ImGui::SameLine();

                    if (ImGui::ColorEdit3("", (float*)&materialGUI.dyeNeutralHT, ImGuiColorEditFlags_NoInputs))
                    {
                        changed = true;
                    }     
                    ImGui::PopID();                    
                   
                    //PSAN VERT ROUGE 
                    ImGui::Text("Vert - Rouge");
                    ImGui::PushID("Vert");
                    if (ImGui::ColorEdit3("", (float*)&materialGUI.vert, ImGuiColorEditFlags_NoInputs))
                    {
                        changed = true;
                    }
                    ImGui::PopID();
                    ImGui::SameLine();

                    ImGui::PushID("Vert-Rouge");
                    char* vertRougeIndex[] = { "70","77","7","07","","06","6","66","60" };
                    ImGui::PushItemWidth(250);
                    if (ImGui::SliderInt("", &materialGUI.int_VertRouge_Concentration, 0, 8, vertRougeIndex[materialGUI.int_VertRouge_Concentration]))
                    {
                        changed = true;
                    }
                    ImGui::PopItemWidth();
                    ImGui::PopID();

                    ImGui::PushID("Rouge");
                    ImGui::SameLine();
                    if (ImGui::ColorEdit3("", (float*)&materialGUI.red, ImGuiColorEditFlags_NoInputs))
                    {
                        changed = true;
                    }
                    ImGui::PopID();

                    // PSAN CENDRE CUIVRE
                    ImGui::Text("Cendre - Cuivre");
                    ImGui::PushID("Bleu");
                    if (ImGui::ColorEdit3("", (float*)&materialGUI.cendre, ImGuiColorEditFlags_NoInputs))
                    {
                        changed = true;
                    }
                    ImGui::PopID();
                    ImGui::SameLine();
                    ImGui::PushID("Cendre-Cuivre");
                    ImGui::PushItemWidth(250);

                    char* CendreCuivreIndex[] = { "10","11","1","01","","04","4","44","40" };
                    if (ImGui::SliderInt("", &materialGUI.int_CendreCuivre_Concentration, 0, 8, CendreCuivreIndex[materialGUI.int_CendreCuivre_Concentration]))
                    {
                        
                        changed = true;

                    }
                    ImGui::PopItemWidth();
                    ImGui::PopID();


                    ImGui::PushID("Cuivre");
                    ImGui::SameLine();
                    if (ImGui::ColorEdit3("", (float*)&materialGUI.cuivre, ImGuiColorEditFlags_NoInputs))
                    {
                        changed = true;
                    }
                    ImGui::PopID();

                    // PSAN IRISE DORE
                    ImGui::Text("Irise - Dore");
                    ImGui::PushID("Irise");
                    if (ImGui::ColorEdit3("", (float*)&materialGUI.irise, ImGuiColorEditFlags_NoInputs))
                    {
                        changed = true;
                    }
                    ImGui::PopID();

                    ImGui::SameLine();
                    ImGui::PushID("Irise-Dore");
                    char* IriseDoreIndex[] = { "20","22","2","02","","03","3","33","30" };
                    ImGui::PushItemWidth(250);
                    if (ImGui::SliderInt("", &materialGUI.int_IriseDore_Concentration, 0, 8, IriseDoreIndex[materialGUI.int_IriseDore_Concentration]))
                    {
                        
                        changed = true;

                    }
                    ImGui::PopItemWidth();
                    ImGui::PopID();

                    ImGui::PushID("Dore");
                    ImGui::SameLine();
                    if (ImGui::ColorEdit3("", (float*)&materialGUI.doree, ImGuiColorEditFlags_NoInputs))
                    {
                        changed = true;
                    }
                    ImGui::PopID();                                
                }
               
                if (changed)
                {
                    updateDYEinterface(materialGUI);
                    updateDYE(materialGUI);
                    updateDYEconcentration(materialGUI);
                    updateHT(materialGUI);

                    m_raytracer->updateMaterial(i, materialGUI);
                    refresh = true;                   
                }
                ImGui::TreePop();                
            }
        }
        if (ImGui::TreeNode((void*)(intptr_t)i, "Save and Switch"))// (ImGui::CollapsingHeader("Save and Switch"))
        {
            static std::string current_Material1_value;
            static std::string current_Material2_value;
            static MaterialGUI* current_Material1;
            static MaterialGUI* current_Material2;
            for (auto& mat:m_materialsGUI)
            {
                if (mat.name.find("default") != std::string::npos)
                { 
                    current_Material1 = &mat;
                    current_Material2 = &mat;
                    break;
                }
            }
            MY_ASSERT(current_Material1 != nullptr);
            current_Material1_value = current_Material1->name.c_str();
            current_Material2_value = current_Material1_value;
            ImGui::Text("Switch the materials");
            if (ImGui::BeginCombo("Material1", current_Material1_value.c_str()))
            {
                for (int n = 0; n < m_materialsGUI.size(); n++)
                {
                    if (m_materialsGUI.at(n).name.find("default") != std::string::npos
                        || m_materialsGUI.at(n).name.find("Hair") != std::string::npos)
                    {
                        bool is_selected = (current_Material1 == &m_materialsGUI[n]); // You can store your selection however you want, outside or inside your objects
                        if (ImGui::Selectable(m_materialsGUI[n].name.c_str()))
                        {
                            if (&m_materialsGUI[n] != current_Material1)
                            {
                                current_Material1 = &m_materialsGUI[n];
                                current_Material1_value = m_materialsGUI[n].name.c_str();
                            }
                        }
                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            if (ImGui::Button("<"))
            {
                if (current_Material1 != NULL && current_Material2 != NULL && current_Material1 != current_Material2)
                {
                    std::string tmp = current_Material1->name;
                    *current_Material1 = *current_Material2;
                    current_Material1->name = tmp;
                    for (int i = 0; i < static_cast<int>(m_materialsGUI.size()); ++i)
                    {
                        if (current_Material1 == &m_materialsGUI[i])
                            m_raytracer->updateMaterial(i, *current_Material1);
                    }
                }



            }
            ImGui::SameLine();
            if (ImGui::Button("<>"))
            {
                if (current_Material1 != NULL && current_Material2 != NULL && current_Material1 != current_Material2)
                {
                    std::string tmp = current_Material1->name;
                    std::string tmp2 = current_Material2->name;
                    MaterialGUI tmpMaterial = *current_Material2;
                    *current_Material2 = *current_Material1;
                    current_Material2->name = tmp2;
                    *current_Material1 = tmpMaterial;
                    current_Material1->name = tmp;
                    for (int i = 0; i < static_cast<int>(m_materialsGUI.size()); ++i)
                    {
                        if (current_Material2 == &m_materialsGUI[i])
                            m_raytracer->updateMaterial(i, *current_Material2);
                        if (current_Material1 == &m_materialsGUI[i])
                            m_raytracer->updateMaterial(i, *current_Material1);
                    }
                }



            }
            ImGui::SameLine();

            if (ImGui::Button(">"))
            {
                if (current_Material1 != NULL && current_Material2 != NULL && current_Material1 != current_Material2)
                {
                    std::string tmp = current_Material2->name;
                    *current_Material2 = *current_Material1;
                    current_Material2->name = tmp;
                    for (int i = 0; i < static_cast<int>(m_materialsGUI.size()); ++i)
                    {
                        if (current_Material2 == &m_materialsGUI[i])
                            m_raytracer->updateMaterial(i, *current_Material2);
                    }
                }

            }
            if (ImGui::BeginCombo("Material2", current_Material2_value.c_str()))
            {
                for (int n = 0; n < m_materialsGUI.size()-3; n++)
                {
                    bool is_selected = (current_Material2 == &m_materialsGUI[n]); // You can store your selection however you want, outside or inside your objects
                    if (ImGui::Selectable(m_materialsGUI[n].name.c_str()))
                    {
                        if (&m_materialsGUI[n] != current_Material2)
                        {
                            current_Material2 = &m_materialsGUI[n];
                            current_Material2_value = m_materialsGUI[n].name.c_str();
                        }
                    }
                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }

            ImGui::NewLine();
            ImGui::Text("Quick Saves");
            const char* quickvalue_name[] = { "Quick Save 1", "Quick Save 2", "Quick Save 3", "Quick Save 4", "Quick Save 5"};
            static int listbox_item_current = -1;
            ImGui::ListBox("", &listbox_item_current, quickvalue_name, nbQuickSaveValue, 4);
            ImGui::SameLine();
            if(ImGui::Button("Quick Load"))
            {
                if (listbox_item_current != -1)
                {
                    *materialGUI1 = QuickSaveValue[listbox_item_current]->first;
                    *materialGUI2 = QuickSaveValue[listbox_item_current]->second;

                    m_raytracer->updateMaterial(materialGUI1ID, *materialGUI1);
                    if (materialGUI1 != materialGUI2)
                        m_raytracer->updateMaterial(materialGUI2ID, *materialGUI2);
                    refresh = true;
                }
            }
            ImGui::NewLine();
            if (ImGui::Button("Quick Save"))
            {
                if (nbQuickSaveValue == 5)
                {
                    delete QuickSaveValue[0];
                    QuickSaveValue[0] = NULL;
                    QuickSaveValue[0] = QuickSaveValue[1];
                    QuickSaveValue[1] = QuickSaveValue[2];
                    QuickSaveValue[2] = QuickSaveValue[3];
                    QuickSaveValue[3] = QuickSaveValue[4];
                    QuickSaveValue[4] = NULL;
                }
                if (nbQuickSaveValue < 5)
                    nbQuickSaveValue++;
                std::pair<MaterialGUI, MaterialGUI>* value = new std::pair<MaterialGUI, MaterialGUI>();
                value->first = *materialGUI1;
                value->second = *materialGUI2;
                QuickSaveValue[nbQuickSaveValue-1] = value;
            }
            ImGui::NewLine();
            ImGui::Text("Save the materials");
            static char SwitchName[30] = "Switch";
            ImGui::InputText("Switch Name", SwitchName, IM_ARRAYSIZE(SwitchName));
            if (ImGui::Button("Save Materials as Color Switch"))
                clicked = true;
            if (clicked == true)
            {
                for (int i = 0; i < 5; i++)
                {
                    delete QuickSaveValue[i];
                    QuickSaveValue[i] = NULL;
                }
                nbQuickSaveValue = 0;

                ColorSwitch new_element;
                std::string Path = m_prefixColorSwitch + SwitchName + ".color";
                new_element.name = SwitchName;
                int i = 0;
                while (fs::exists(Path))
                {
                    new_element.name = "Switch" + std::to_string(i);
                    Path = m_prefixColorSwitch + new_element.name + ".color";
                    i++;
                }
                new_element.Material1 = *materialGUI1;
                new_element.Material2 = *materialGUI2;
                strcpy(SwitchName, new_element.name.c_str());
                if (!hasChanged && current_settings_value != NULL)
                    new_element.SettingFile = current_settings_value;
                else
                {
                    std::string tmp = new_element.name + "_Setting";
                    std::string Path = m_prefixSettings + tmp + ".settings";
                    int i = 0;
                    while (fs::exists(Path))
                    {
                        tmp = new_element.name + "_Settings_" + std::to_string(i);
                        Path = m_prefixSettings + tmp + ".settings";
                        i++;
                    }

                    saveSettingToFile(Path);
                    m_settings.push_back(std::pair<std::string, std::string>(tmp, Path));
                    new_element.SettingFile = tmp.c_str();
                    std::string newLine = "settings " + tmp + " \"" + Path + "\"";
                    f_options->addCommand(newLine);
                    hasChanged = false;
                    refresh = true;
                }
                m_materialsColor.push_back(new_element);
                current_settings_value = m_materialsColor[m_materialsColor.size() - 1].SettingFile.c_str();
                //create the command to add to the file
                std::ofstream i_writeCmd(Path, std::ios::app);
                i_writeCmd << roundFloat(new_element.Material1.dye.x * 255.0f) + ";" + roundFloat(new_element.Material1.dye.y * 255.0f) + ";" + roundFloat(new_element.Material1.dye.z * 255.0f) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.dye_concentration) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.whitepercen) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.scale_angle_deg) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.roughnessM) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.roughnessN) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.melanin_concentration) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.melanin_ratio) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.melanin_concentration_disparity) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.melanin_ratio_disparity) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.dyeNeutralHT.x * 255.0f) + ";" + roundFloat(new_element.Material1.dyeNeutralHT.y * 255.0f) + ";" + roundFloat(new_element.Material1.dyeNeutralHT.z * 255.0f) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.dyeNeutralHT_Concentration) + "\n";
                i_writeCmd << roundFloat(static_cast<float>(new_element.Material1.HT)) + "\n";
                i_writeCmd << roundFloat(static_cast<float>(new_element.Material1.int_VertRouge_Concentration)) + "\n";
                i_writeCmd << roundFloat(static_cast<float>(new_element.Material1.int_CendreCuivre_Concentration)) + "\n";
                i_writeCmd << roundFloat(static_cast<float>(new_element.Material1.int_IriseDore_Concentration)) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.cendre.x * 255.0f) + ";" + roundFloat(new_element.Material1.cendre.y * 255.0f) + ";" + roundFloat(new_element.Material1.cendre.z * 255.0f) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.irise.x * 255.0f) + ";" + roundFloat(new_element.Material1.irise.y * 255.0f) + ";" + roundFloat(new_element.Material1.irise.z * 255.0f) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.doree.x * 255.0f) + ";" + roundFloat(new_element.Material1.doree.y * 255.0f) + ";" + roundFloat(new_element.Material1.doree.z * 255.0f) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.cuivre.x * 255.0f) + ";" + roundFloat(new_element.Material1.cuivre.y * 255.0f) + ";" + roundFloat(new_element.Material1.cuivre.z * 255.0f) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.acajou.x * 255.0f) + ";" + roundFloat(new_element.Material1.acajou.y * 255.0f) + ";" + roundFloat(new_element.Material1.acajou.z * 255.0f) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.red.x * 255.0f) + ";" + roundFloat(new_element.Material1.red.y * 255.0f) + ";" + roundFloat(new_element.Material1.red.z * 255.0f) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.vert.x * 255.0f) + ";" + roundFloat(new_element.Material1.vert.y * 255.0f) + ";" + roundFloat(new_element.Material1.vert.z * 255.0f) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.concentrationCendre) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.concentrationIrise) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.concentrationDore) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.concentrationCuivre) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.concentrationAcajou) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.concentrationRouge) + "\n";
                i_writeCmd << roundFloat(new_element.Material1.concentrationVert) + "\n";

                i_writeCmd << roundFloat(new_element.Material2.dye.x * 255.0f) + ";" + roundFloat(new_element.Material2.dye.y * 255.0f) + ";" + roundFloat(new_element.Material2.dye.z * 255.0f) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.dye_concentration) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.whitepercen) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.scale_angle_deg) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.roughnessM) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.roughnessN) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.melanin_concentration) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.melanin_ratio) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.melanin_concentration_disparity) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.melanin_ratio_disparity) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.dyeNeutralHT.x * 255.0f) + ";" + roundFloat(new_element.Material2.dyeNeutralHT.y * 255.0f) + ";" + roundFloat(new_element.Material2.dyeNeutralHT.z * 255.0f) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.dyeNeutralHT_Concentration) + "\n";
                i_writeCmd << roundFloat(static_cast<float>(new_element.Material2.HT)) + "\n";
                i_writeCmd << roundFloat(static_cast<float>(new_element.Material2.int_VertRouge_Concentration)) + "\n";
                i_writeCmd << roundFloat(static_cast<float>(new_element.Material2.int_CendreCuivre_Concentration)) + "\n";
                i_writeCmd << roundFloat(static_cast<float>(new_element.Material2.int_IriseDore_Concentration)) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.cendre.x * 255.0f) + ";" + roundFloat(new_element.Material2.cendre.y * 255.0f) + ";" + roundFloat(new_element.Material2.cendre.z * 255.0f) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.irise.x * 255.0f) + ";" + roundFloat(new_element.Material2.irise.y * 255.0f) + ";" + roundFloat(new_element.Material2.irise.z * 255.0f) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.doree.x * 255.0f) + ";" + roundFloat(new_element.Material2.doree.y * 255.0f) + ";" + roundFloat(new_element.Material2.doree.z * 255.0f) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.cuivre.x * 255.0f) + ";" + roundFloat(new_element.Material2.cuivre.y * 255.0f) + ";" + roundFloat(new_element.Material2.cuivre.z * 255.0f) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.acajou.x * 255.0f) + ";" + roundFloat(new_element.Material2.acajou.y * 255.0f) + ";" + roundFloat(new_element.Material2.acajou.z * 255.0f) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.red.x * 255.0f) + ";" + roundFloat(new_element.Material2.red.y * 255.0f) + ";" + roundFloat(new_element.Material2.red.z * 255.0f) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.vert.x * 255.0f) + ";" + roundFloat(new_element.Material2.vert.y * 255.0f) + ";" + roundFloat(new_element.Material2.vert.z * 255.0f) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.concentrationCendre) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.concentrationIrise) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.concentrationDore) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.concentrationCuivre) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.concentrationAcajou) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.concentrationRouge) + "\n";
                i_writeCmd << roundFloat(new_element.Material2.concentrationVert) + "\n";
                i_writeCmd << new_element.SettingFile << "\n";
                i_writeCmd.close();
                std::string newLine = "color " + new_element.name + " \"" + Path + "\"";
                f_options->addCommand(newLine);
                refresh = true;
            }
            ImGui::TreePop();
        }  
    }
    if (ImGui::CollapsingHeader("Dynamic settings"))
    {        
        bool changed = false;
        const char* current_HDR_value;
        HDRSwitch* current_item_HDR = NULL;
        for (int i = 0; i < m_HDR.size(); i++)
        {
            if (std::strcmp(m_HDR[i].file_name.c_str(), m_environment.c_str()) == 0)
            {
                current_item_HDR = &m_HDR[i];
                current_HDR_value = m_HDR[i].name.c_str();
                break;
            }
        }
        if (current_item_model->material1Name != current_item_model->material2Name)
            materialGUI2 = &m_materialsGUI.at(m_mapMaterialReferences.find(current_item_model->material2Name)->second);


        if (ImGui::BeginCombo("HDR", current_HDR_value))
        {
            for (int n = 0; n < m_HDR.size(); n++)
            {
                bool is_selected = (current_item_HDR == &m_HDR[n]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(m_HDR[n].name.c_str(), is_selected))
                {
                    if (&m_HDR[n] != current_item_HDR)
                    {
                        current_item_HDR = &m_HDR[n];
                        current_HDR_value = m_HDR[n].name.c_str();
                        m_environment = m_HDR[n].file_name;
                        convertPath(m_environment);

                        Picture* picture = new Picture();
                        picture->load(m_environment, IMAGE_FLAG_2D);
                        delete m_mapPictures[std::string("environment")];
                        m_mapPictures[std::string("environment")] = picture;
                        m_raytracer->initTextures(m_mapPictures);
                        m_raytracer->initLights(m_lights);
                        m_raytracer->updateCamera(0, m_cameras[0]);
                    }
                }
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        if (ImGui::BeginCombo("Hair type", current_item_model_value))
        {
            for (int n = 0; n < m_models.size(); n++)
            {
                bool is_selected = (current_item_model == &m_models[n]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(m_models[n].name.c_str(), is_selected))
                {
                    if (current_item_model != &m_models[n])
                    {
                        m_scene->removeCurvesChild();

                        std::string HairModel = current_item_model->file_name;
                        convertPath(HairModel);
                        std::ostringstream keyGeometry_delete1;
                        std::ostringstream keyGeometry_delete2;
                        if (current_item_model->material1Name == current_item_model->material2Name)
                            keyGeometry_delete1 << current_item_model->map_identifier;
                        else
                        {
                            keyGeometry_delete1 << current_item_model->map_identifier << "_half_1";
                            keyGeometry_delete2 << current_item_model->map_identifier << "_half_2";
                        }
                        std::map<std::string, unsigned int>::const_iterator itg_delete1 = m_mapGeometries.find(keyGeometry_delete1.str());
                        if (itg_delete1 != m_mapGeometries.end())
                        {
                            auto geometry_id = itg_delete1->second;
                            for (int i = 0; i < m_geometries.size(); i++)
                            {
                                if (m_geometries[i].get()->getId() == geometry_id)
                                {
                                    m_geometries.erase(m_geometries.begin() + i);
                                    m_mapGeometries.erase(keyGeometry_delete1.str());
                                    m_idGeometry--;
                                }
                            }
                        }
                        if (current_item_model->material1Name != current_item_model->material2Name)
                        {
                            std::map<std::string, unsigned int>::const_iterator itg_delete2 = m_mapGeometries.find(keyGeometry_delete2.str());
                            if (itg_delete2 != m_mapGeometries.end())
                            {
                                auto geometry_id = itg_delete2->second;
                                for (int i = 0; i < m_geometries.size(); i++)
                                {
                                    if (m_geometries[i].get()->getId() == geometry_id)
                                    {
                                        m_geometries.erase(m_geometries.begin() + i);
                                        m_mapGeometries.erase(keyGeometry_delete2.str());
                                        m_idGeometry--;
                                    }
                                }
                            }
                        }

                        current_item_model_value = m_models[n].name.c_str();
                        current_item_model = &m_models[n];

                        HairModel = current_item_model->file_name;
                        convertPath(HairModel);
                        std::ostringstream keyGeometry1;
                        std::ostringstream keyGeometry2;
                        if (current_item_model->material1Name == current_item_model->material2Name)
                            keyGeometry1 << current_item_model->map_identifier;
                        else
                        {
                            keyGeometry1 << current_item_model->map_identifier << "_half_1";
                            keyGeometry2 << current_item_model->map_identifier << "_half_2";
                        }

                        materialGUI1 = &(m_materialsGUI.at(m_mapMaterialReferences.find(current_item_model->material1Name)->second));
                        materialGUI2 = &(m_materialsGUI.at(m_mapMaterialReferences.find(current_item_model->material2Name)->second));

                        std::shared_ptr<sg::Curves> geometry_left;
                        std::map<std::string, unsigned int>::const_iterator itg1 = m_mapGeometries.find(keyGeometry1.str());
                        if (itg1 == m_mapGeometries.end())
                        {
                            m_mapGeometries[keyGeometry1.str()] = m_idGeometry;
                            geometry_left = std::make_shared<sg::Curves>(m_idGeometry++);
                            const char* file = current_item_model->file_name.c_str();
                            if (current_item_model->material1Name != current_item_model->material2Name)
                                geometry_left->createHairFromFile(file, true);
                            else
                                geometry_left->createHairFromFile(file);
                            m_geometries.push_back(geometry_left);
                        }
                        else
                            geometry_left = std::dynamic_pointer_cast<sg::Curves>(m_geometries[itg1->second]);
                        appendInstance(m_scene, geometry_left, curMatrix, current_item_model->material1Name, m_idInstance);

                        if (current_item_model->material1Name != current_item_model->material2Name)
                        {
                            std::shared_ptr<sg::Curves> geometry_right;
                            std::map<std::string, unsigned int>::const_iterator itg2 = m_mapGeometries.find(keyGeometry2.str());
                            if (itg2 == m_mapGeometries.end())
                            {
                                m_mapGeometries[keyGeometry2.str()] = m_idGeometry;
                                geometry_right = std::make_shared<sg::Curves>(m_idGeometry++);
                                const char* file = current_item_model->file_name.c_str();
                                geometry_right->createHairFromFile(file, false);
                                m_geometries.push_back(geometry_right);
                            }
                            else
                                geometry_right = std::dynamic_pointer_cast<sg::Curves>(m_geometries[itg2->second]);
                            appendInstance(m_scene, geometry_right, curMatrix, current_item_model->material2Name, m_idInstance);
                        }
                        m_raytracer->initMaterials(m_materialsGUI);
                        m_raytracer->initScene(m_scene, m_idGeometry); // m_idGeometry is the number of geometries in the scene
                        refresh = true;
                        m_isValid = true;
                        m_raytracer->updateCamera(0, m_cameras[0]);
                    }
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::BeginCombo("Color Switch", current_item_colorswatch_value))
        {
            for (int n = 0; n < m_materialsColor.size(); n++)
            {
                bool is_selected = (current_item_colorswatch == &m_materialsColor[n]);
                if (ImGui::Selectable(m_materialsColor[n].name.c_str(), is_selected))
                {
                    current_item_colorswatch_value = m_materialsColor[n].name.c_str();
                    materialGUI1->dye = m_materialsColor[n].Material1.dye;
                    materialGUI1->dye_concentration = m_materialsColor[n].Material1.dye_concentration;
                    materialGUI1->melanin_concentration = m_materialsColor[n].Material1.melanin_concentration;
                    materialGUI1->melanin_ratio = m_materialsColor[n].Material1.melanin_ratio;
                    materialGUI1->melanin_concentration_disparity = m_materialsColor[n].Material1.melanin_concentration_disparity;
                    materialGUI1->melanin_ratio_disparity = m_materialsColor[n].Material1.melanin_ratio_disparity;
                    materialGUI1->whitepercen = m_materialsColor[n].Material1.whitepercen;
                    materialGUI1->scale_angle_deg = m_materialsColor[n].Material1.scale_angle_deg;
                    materialGUI1->roughnessM = m_materialsColor[n].Material1.roughnessM;
                    materialGUI1->roughnessN = m_materialsColor[n].Material1.roughnessN;

                    materialGUI1->dyeNeutralHT = m_materialsColor[n].Material1.dyeNeutralHT;
                    materialGUI1->dyeNeutralHT_Concentration = m_materialsColor[n].Material1.dyeNeutralHT_Concentration;
                    materialGUI1->HT = m_materialsColor[n].Material1.HT;
                    materialGUI1->int_VertRouge_Concentration = m_materialsColor[n].Material1.int_VertRouge_Concentration;
                    materialGUI1->int_CendreCuivre_Concentration = m_materialsColor[n].Material1.int_CendreCuivre_Concentration;
                    materialGUI1->int_IriseDore_Concentration = m_materialsColor[n].Material1.int_IriseDore_Concentration;
                    materialGUI1->cendre = m_materialsColor[n].Material1.cendre;
                    materialGUI1->acajou = m_materialsColor[n].Material1.acajou;
                    materialGUI1->vert = m_materialsColor[n].Material1.vert;
                    materialGUI1->red = m_materialsColor[n].Material1.red;
                    materialGUI1->irise = m_materialsColor[n].Material1.irise;
                    materialGUI1->cuivre = m_materialsColor[n].Material1.cuivre;
                    materialGUI1->doree = m_materialsColor[n].Material1.doree;
                    materialGUI1->concentrationCendre = m_materialsColor[n].Material1.concentrationCendre;
                    materialGUI1->concentrationIrise = m_materialsColor[n].Material1.concentrationIrise;
                    materialGUI1->concentrationDore = m_materialsColor[n].Material1.concentrationDore;
                    materialGUI1->concentrationCuivre = m_materialsColor[n].Material1.concentrationCuivre;
                    materialGUI1->concentrationAcajou = m_materialsColor[n].Material1.concentrationAcajou;
                    materialGUI1->concentrationRouge = m_materialsColor[n].Material1.concentrationRouge;
                    materialGUI1->concentrationVert = m_materialsColor[n].Material1.concentrationVert;
                    if (materialGUI1 != materialGUI2)
                    {
                        materialGUI2->dye = m_materialsColor[n].Material2.dye;
                        materialGUI2->dye_concentration = m_materialsColor[n].Material2.dye_concentration;
                        materialGUI2->melanin_concentration = m_materialsColor[n].Material2.melanin_concentration;
                        materialGUI2->melanin_ratio = m_materialsColor[n].Material2.melanin_ratio;
                        materialGUI2->melanin_concentration_disparity = m_materialsColor[n].Material2.melanin_concentration_disparity;
                        materialGUI2->melanin_ratio_disparity = m_materialsColor[n].Material2.melanin_ratio_disparity;
                        materialGUI2->whitepercen = m_materialsColor[n].Material2.whitepercen;
                        materialGUI2->scale_angle_deg = m_materialsColor[n].Material2.scale_angle_deg;
                        materialGUI2->roughnessM = m_materialsColor[n].Material2.roughnessM;
                        materialGUI2->roughnessN = m_materialsColor[n].Material2.roughnessN;
                        materialGUI2->dyeNeutralHT = m_materialsColor[n].Material2.dyeNeutralHT;
                        materialGUI2->dyeNeutralHT_Concentration = m_materialsColor[n].Material2.dyeNeutralHT_Concentration;
                        materialGUI2->HT = m_materialsColor[n].Material2.HT;
                        materialGUI2->int_VertRouge_Concentration = m_materialsColor[n].Material2.int_VertRouge_Concentration;
                        materialGUI2->int_CendreCuivre_Concentration = m_materialsColor[n].Material2.int_CendreCuivre_Concentration;
                        materialGUI2->int_IriseDore_Concentration = m_materialsColor[n].Material2.int_IriseDore_Concentration;
                        materialGUI2->cendre = m_materialsColor[n].Material2.cendre;
                        materialGUI2->acajou = m_materialsColor[n].Material2.acajou;
                        materialGUI2->vert = m_materialsColor[n].Material2.vert;
                        materialGUI2->red = m_materialsColor[n].Material2.red;
                        materialGUI2->irise = m_materialsColor[n].Material2.irise;
                        materialGUI2->cuivre = m_materialsColor[n].Material2.cuivre;
                        materialGUI2->doree = m_materialsColor[n].Material2.doree;
                        materialGUI2->concentrationCendre = m_materialsColor[n].Material2.concentrationCendre;
                        materialGUI2->concentrationIrise = m_materialsColor[n].Material2.concentrationIrise;
                        materialGUI2->concentrationDore = m_materialsColor[n].Material2.concentrationDore;
                        materialGUI2->concentrationCuivre = m_materialsColor[n].Material2.concentrationCuivre;
                        materialGUI2->concentrationAcajou = m_materialsColor[n].Material2.concentrationAcajou;
                        materialGUI2->concentrationRouge = m_materialsColor[n].Material2.concentrationRouge;
                        materialGUI2->concentrationVert = m_materialsColor[n].Material2.concentrationVert;
                    }
                    if (current_settings_value == NULL || m_materialsColor[n].SettingFile != current_settings_value)
                    {
                        for (int i = 0; i < m_settings.size(); i++)
                        {
                            if (m_materialsColor[n].SettingFile == m_settings[i].first)
                            {
                                chargeSettingsFromFile(m_settings[i].second);
                                current_settings_value = m_settings[i].first.c_str();
                            }
                        }
                    }
                    changed = true;
                }
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
                if (changed)
                {
                    int index = -1;
                    for (int ij = 0; ij < m_materialsGUI.size(); ++ij)
                    {
                        if (m_materialsGUI.at(ij).indexBSDF == INDEX_BCSDF_HAIR)
                        {
                            index = ij;
                            break;
                        }
                    }
                    MY_ASSERT(index > 0);
                    m_raytracer->updateMaterial(index, *materialGUI1);
                    if (materialGUI1 != materialGUI2)
                        m_raytracer->updateMaterial(index + 1, *materialGUI2);
                    refresh = true;
                }
                // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
            }
            ImGui::EndCombo();
        }
        if (ImGui::BeginCombo("Settings", current_settings_value))
        {
            for (int n = 0; n < m_settings.size(); ++n)
            {
                bool is_selected;
                if (current_settings_value != NULL)
                    is_selected = strcmp(current_settings_value, m_settings[n].first.c_str());
                else
                    is_selected = false;
                if (ImGui::Selectable(m_settings[n].first.c_str(), is_selected))
                {
                    current_settings_value = m_settings[n].first.c_str();
                    chargeSettingsFromFile(m_settings[n].second);
                    for (int i = 0; i < static_cast<int>(m_materialsGUI.size()); ++i)
                    {
                        MaterialGUI& materialGUI = m_materialsGUI[i];
                        if (materialGUI.indexBSDF == INDEX_BCSDF_HAIR)
                        {
                            if (materialGUI.shouldModify)
                            {
                                updateDYEinterface(materialGUI);
                                updateDYE(materialGUI);
                                updateDYEconcentration(materialGUI);
                                updateHT(materialGUI);
                                m_raytracer->updateMaterial(i, materialGUI);
                                refresh = true;
                            }
                        }
                    }
                }
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        
    }
    if (ImGui::CollapsingHeader("Lighting",true))//if (ImGui::CollapsingHeader("Lighting", true))
    {
        static bool lightings[5] = { m_lightings_on[0], m_lightings_on[1], m_lightings_on[2], m_lightings_on[3], m_lightings_on [4]};
        static int emissions[5] = { m_lighting_emission[0], m_lighting_emission[1], m_lighting_emission[2], m_lighting_emission[3], m_lighting_emission[4]};
        bool geo_changed = false, emi_changed = false;
        const std::string checkboxname[5] = { "Front Lighting","Back Lighting", "Left Side Lighting","Right Side Lighting", "Top Lighting" };
        const std::string checkboxshortname[5] = { "FL","BL", "LSL","RSL", "TL" };
        auto num_lights = m_lights.size();
        size_t area_light_start = m_lights.at(0).type == LIGHT_PARALLELOGRAM ? 0 : 1;
        for (size_t i = area_light_start; i < num_lights; ++i)
        {
            auto loc = static_cast<int>(m_lights.at(i).location);
            ImGui::Checkbox(checkboxname[loc - 1].c_str(), &lightings[i - area_light_start]);
            std::string slidername = "Emission " + checkboxshortname[loc - 1];
            ImGui::SliderInt(slidername.c_str(), &emissions[i - area_light_start], 0, 100);
            if (lightings[i - area_light_start] != m_lightings_on[i - area_light_start])
            {
                m_lightings_on[i - area_light_start] = lightings[i - area_light_start];
                m_lights.at(i).lighting_activated = m_lightings_on[i - area_light_start];
                m_scene->getChild(i - area_light_start)->set_activation(m_lightings_on[i - area_light_start]);
                geo_changed = true;
            }
            if (emissions[i - area_light_start] != m_lighting_emission[i - area_light_start]) 
            {
                m_lighting_emission[i - area_light_start] = emissions[i - area_light_start];
                m_lights.at(i).emission = make_float3(static_cast<float>(m_lighting_emission[i - area_light_start]));
                emi_changed = true;
            }
            ImGui::Separator();
            //ImGui::Spacing();
        }
        if (geo_changed||emi_changed) {
            //m_raytracer->initMaterials(m_materialsGUI);
            //m_raytracer->initScene(m_scene, m_idGeometry);
            m_raytracer->initLights(m_lights);
            m_raytracer->updateCamera(0, m_cameras[0]);
            geo_changed = false;
            emi_changed = false;
        }
    }
    if (ImGui::CollapsingHeader("Object geometries", true))
    {
        const std::vector<std::string> checkboxname = {
                                                       "Front Light",
                                                       "Back Light",
                                                       "Left Side Light",
                                                       "Right Side Light",
                                                       "Top Light",
                                                       "Head"/*,
                                                       "Left Hair",
                                                       "Right Hair"*/};
        static bool geo_group[8] = { true,true,true,true,true,true,true,true };
        bool geo_changed = false;
        int area_light_start = -1;
        for (int i=0;i<m_lights.size();++i)
        {
            if (m_lights.at(i).type == LIGHT_PARALLELOGRAM)
            {
                area_light_start = i;
                break;
            }
        }
        int num_area_lights = 0;
        if (area_light_start > -1)
        {
            num_area_lights = static_cast<int>(m_lights.size() - area_light_start);
        }
        
        int loc = 5;
        for (int i = 0; i < m_scene->getNumChildren() - 2; ++i)
        {
            if (num_area_lights > 0 && i < num_area_lights)
            {
                loc = static_cast<int>(m_lights.at(i + area_light_start).location) - 1;
            }
            else loc = 5;
            ImGui::Checkbox(checkboxname[loc].c_str(), &geo_group[loc]);
            if (geo_group[loc] != m_geo_group[loc])
            {
                m_geo_group[loc] = geo_group[loc];
                m_scene->getChild(i)->set_activation(m_geo_group[loc]);
                geo_changed = true;
            }
            //if (loc >= 4)++loc;//in case there are other objects besides of head and hair
        }

        if (geo_changed)
        {
            m_raytracer->initScene(m_scene, m_idGeometry);
            m_raytracer->initLights(m_lights);
            m_raytracer->updateCamera(0, m_cameras[0]);
            geo_changed = false;
        }
    }
    if (ImGui::CollapsingHeader("Window screen", true))
    {
        static int screen = 1;
        int tmp = screen;
        if (m_is_fullscreen)
        {
            screen = 0;
        }
        else 
        {
            screen = 1;
        }
        ImGui::RadioButton("Fullscreen", &screen, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Windowed", &screen, 1);

        if (screen != tmp)
        {
            switch (screen)
            {
            case 0:
                m_is_fullscreen = true;
                glfwSetWindowMonitor(m_window, glfwGetPrimaryMonitor(), 0, 0, m_scr_w, m_scr_h, 0);
                break;
            case 1:
                m_is_fullscreen = false;
                glfwSetWindowMonitor(m_window, nullptr, 100, 100, m_scr_w, m_scr_h, 0);
                break;
            default:
                break;
            }
            m_camera.markDirty();
        }
    }
    ImGui::PopItemWidth();
    ImGui::End();

    if (refresh)
    {
        restartRendering();
        if (clicked)
        {
            current_item_colorswatch_value = m_materialsColor.back().name.c_str();
            current_item_colorswatch = &m_materialsColor.back();
        }
    }   
}

void Application::updateHT(MaterialGUI& materialGUI)
{
    float result = 0;
    
    // hot color
    materialGUI.melanin_concentration = m_melanineConcentration[materialGUI.HT-1];   

    if ( materialGUI.int_IriseDore_Concentration == 7 || materialGUI.int_IriseDore_Concentration == 8)
    {
        result -= (m_melanineConcentration[materialGUI.HT - 1] - m_melanineConcentration[materialGUI.HT])/2;
    }
    if (materialGUI.int_CendreCuivre_Concentration == 7 || materialGUI.int_CendreCuivre_Concentration == 8)
    {
        result -= (m_melanineConcentration[materialGUI.HT - 1] - m_melanineConcentration[materialGUI.HT])/4;
    }
    if (  materialGUI.int_VertRouge_Concentration == 7 || materialGUI.int_VertRouge_Concentration == 8)
    { 
         result -= (m_melanineConcentration[materialGUI.HT-1] - m_melanineConcentration[materialGUI.HT])/4 ;
    } 

    // ROUGE 
    if (materialGUI.int_VertRouge_Concentration == 0 || materialGUI.int_VertRouge_Concentration == 1 || materialGUI.int_VertRouge_Concentration == 2)
    {
        result -= (m_melanineConcentration[materialGUI.HT - 1] - m_melanineConcentration[materialGUI.HT]) / 2;
    }

   

    // cold color
// CENDER "10" IRISE "20"
    if (materialGUI.int_CendreCuivre_Concentration == 0 || materialGUI.int_IriseDore_Concentration == 0 )
    {
       result -= m_lightened_x10[materialGUI.HT-1];    
    }

 // CENDER "11" IRISER "22"  
    else if (materialGUI.int_CendreCuivre_Concentration == 1 || materialGUI.int_IriseDore_Concentration == 1 )
    {
        result -= m_lightened_x2[materialGUI.HT - 1];       
    }

//CENDER "1" IRISER "2"
    else if (materialGUI.int_CendreCuivre_Concentration == 2 || materialGUI.int_IriseDore_Concentration == 2)
    {
        result -= m_lightened_x1[materialGUI.HT - 1];
    }     

// CENDER "01" IRISE "02" 
    else if (materialGUI.int_CendreCuivre_Concentration == 3 || materialGUI.int_IriseDore_Concentration == 3)
    {
        result -= (m_melanineConcentration[materialGUI.HT - 1] - m_melanineConcentration[materialGUI.HT]) / 2;
    }


//VERT "7" 
    if (materialGUI.int_VertRouge_Concentration == 0 || materialGUI.int_VertRouge_Concentration == 1 || materialGUI.int_VertRouge_Concentration == 2)
    {
        result -= (m_melanineConcentration[materialGUI.HT - 2] - m_melanineConcentration[materialGUI.HT - 1])/2;
    }


    materialGUI.melanin_concentration += result;

    return ;
}

void Application::updateDYEconcentration(MaterialGUI  &materialGUI)
{
    float dyemoyenne;   

    if (materialGUI.dye_ConcentrationCendreCuivre >= materialGUI.dye_ConcentrationIriseDore && materialGUI.dye_ConcentrationCendreCuivre >= materialGUI.dye_ConcentrationVertRouge)
    {                
        dyemoyenne = materialGUI.dye_ConcentrationCendreCuivre;
    }    

     if (materialGUI.dye_ConcentrationIriseDore >= materialGUI.dye_ConcentrationCendreCuivre && materialGUI.dye_ConcentrationIriseDore >=  materialGUI.dye_ConcentrationVertRouge)
    {
        dyemoyenne = materialGUI.dye_ConcentrationIriseDore;
    }   

     if  (materialGUI.dye_ConcentrationVertRouge >= materialGUI.dye_ConcentrationCendreCuivre && materialGUI.dye_ConcentrationVertRouge >= materialGUI.dye_ConcentrationIriseDore)
    {
        dyemoyenne = materialGUI.dye_ConcentrationVertRouge;
    }   
    
     materialGUI.dye_concentration = dyemoyenne / m_factorColorantHT[materialGUI.HT - 1];
    
}

void Application::updateDYEinterface(MaterialGUI& materialGUI)
{
    if (materialGUI.int_VertRouge_Concentration == 0)
    {
        materialGUI.concentrationVert = m_concentrationVert[3];
        materialGUI.concentrationRouge = 0.f;
        materialGUI.dye_ConcentrationVertRouge = m_dye_ConcentrationVert[3];        
    }

    if (materialGUI.int_VertRouge_Concentration == 1)
    {
        materialGUI.concentrationVert = m_concentrationVert[2];
        materialGUI.concentrationRouge = 0.f;
        materialGUI.dye_ConcentrationVertRouge = m_dye_ConcentrationVert[2];      
    }

    if (materialGUI.int_VertRouge_Concentration == 2)
    {
        materialGUI.concentrationVert = m_concentrationVert[1];
        materialGUI.concentrationRouge = 0.f;
        materialGUI.dye_ConcentrationVertRouge = m_dye_ConcentrationVert[1];       
    }

    if (materialGUI.int_VertRouge_Concentration == 3)
    {
        materialGUI.concentrationVert = m_concentrationVert[0];
        materialGUI.concentrationRouge = 0.f;
        materialGUI.dye_ConcentrationVertRouge = m_dye_ConcentrationVert[0];       
    }
    if (materialGUI.int_VertRouge_Concentration == 4)
    {
        materialGUI.concentrationVert = 0.f;
        materialGUI.concentrationRouge = 0.f;
        materialGUI.dye_ConcentrationVertRouge = 0.f;        
    }

    if (materialGUI.int_VertRouge_Concentration == 5)
    {
        materialGUI.concentrationRouge = m_concentrationRouge[0];
        materialGUI.concentrationVert = 0.f;
        materialGUI.dye_ConcentrationVertRouge = m_dye_ConcentrationRouge[0];        
    }

    if (materialGUI.int_VertRouge_Concentration == 6)
    {
        materialGUI.concentrationRouge = m_concentrationRouge[1];
        materialGUI.concentrationVert = 0.f;
        materialGUI.dye_ConcentrationVertRouge = m_dye_ConcentrationRouge[1];        
    }

    if (materialGUI.int_VertRouge_Concentration == 7)
    {
        materialGUI.concentrationRouge = m_concentrationRouge[2];
        materialGUI.concentrationVert = 0.f;
        materialGUI.dye_ConcentrationVertRouge = m_dye_ConcentrationRouge[2];        
    }

    if (materialGUI.int_VertRouge_Concentration == 8)
    {
        materialGUI.concentrationRouge = m_concentrationRouge[3];
        materialGUI.concentrationVert = 0.f;
        materialGUI.dye_ConcentrationVertRouge = m_dye_ConcentrationRouge[3];        
    }

    if (materialGUI.int_CendreCuivre_Concentration == 0)
    {
        materialGUI.concentrationCendre = m_concentrationCendre[3];
        materialGUI.concentrationCuivre = 0.f;
        materialGUI.dye_ConcentrationCendreCuivre = m_dye_ConcentrationCender[3];
    }

    if (materialGUI.int_CendreCuivre_Concentration == 1)
    {
        materialGUI.concentrationCendre = m_concentrationCendre[2];
        materialGUI.concentrationCuivre = 0.f;

        materialGUI.dye_ConcentrationCendreCuivre = m_dye_ConcentrationCender[2];
    }

    if (materialGUI.int_CendreCuivre_Concentration == 2)
    {
        materialGUI.concentrationCendre = m_concentrationCendre[1];
        materialGUI.concentrationCuivre = 0.f;

        materialGUI.dye_ConcentrationCendreCuivre = m_dye_ConcentrationCender[1];
    }

    if (materialGUI.int_CendreCuivre_Concentration == 3)
    {
        materialGUI.concentrationCendre = m_concentrationCendre[0];
        materialGUI.concentrationCuivre = 0.f;

        materialGUI.dye_ConcentrationCendreCuivre = m_dye_ConcentrationCender[0];
    }

    // NEUTRE
    if (materialGUI.int_CendreCuivre_Concentration == 4)
    {
        materialGUI.concentrationCendre = 0.f;
        materialGUI.concentrationCuivre = 0.f;
        materialGUI.dye_ConcentrationCendreCuivre = 0.0f;
    }
    // CUIVRE
    if (materialGUI.int_CendreCuivre_Concentration == 5)
    {
        materialGUI.concentrationCuivre = m_concentrationCuivre[0];
        materialGUI.concentrationCendre = 0.f;
        materialGUI.dye_ConcentrationCendreCuivre = m_dye_ConcentrationCover[0];
    }

    if (materialGUI.int_CendreCuivre_Concentration == 6)
    {
        materialGUI.concentrationCuivre = m_concentrationCuivre[1];
        materialGUI.concentrationCendre = 0.f;
        materialGUI.dye_ConcentrationCendreCuivre = m_dye_ConcentrationCover[1];
    }

    if (materialGUI.int_CendreCuivre_Concentration == 7)
    {
        materialGUI.concentrationCuivre = m_concentrationCuivre[2];
        materialGUI.concentrationCendre = 0.f;
        materialGUI.dye_ConcentrationCendreCuivre = m_dye_ConcentrationCover[2];
    }

    if (materialGUI.int_CendreCuivre_Concentration == 8)
    {
        materialGUI.concentrationCuivre = m_concentrationCuivre[3];
        materialGUI.concentrationCendre = 0.f;
        materialGUI.dye_ConcentrationCendreCuivre = m_dye_ConcentrationCover[3];
    }

    // Irise
    if (materialGUI.int_IriseDore_Concentration == 0)
    {
        materialGUI.concentrationIrise = m_concentrationIrise[3];
        materialGUI.concentrationDore = 0.f;
        materialGUI.dye_ConcentrationIriseDore = m_dye_ConcentrationAsh[3];       
    }
    if (materialGUI.int_IriseDore_Concentration == 1)
    {
        materialGUI.concentrationIrise = m_concentrationIrise[2];
        materialGUI.concentrationDore = 0.f;
        materialGUI.dye_ConcentrationIriseDore = m_dye_ConcentrationAsh[2];        
    }

    if (materialGUI.int_IriseDore_Concentration == 2)
    {
        materialGUI.concentrationIrise = m_concentrationIrise[1];
        materialGUI.concentrationDore = 0.f;
        materialGUI.dye_ConcentrationIriseDore = m_dye_ConcentrationAsh[1];       
    }

    if (materialGUI.int_IriseDore_Concentration == 3)
    {
        materialGUI.concentrationIrise = m_concentrationIrise[0];
        materialGUI.concentrationDore = 0.f;
        materialGUI.dye_ConcentrationIriseDore = m_dye_ConcentrationAsh[0];        
    }
    // neutre
    if (materialGUI.int_IriseDore_Concentration == 4)
    {
        materialGUI.concentrationIrise = 0.f;
        materialGUI.concentrationDore = 0.f;
        materialGUI.dye_ConcentrationIriseDore = 0.f;       
    }
    //Dore
    if (materialGUI.int_IriseDore_Concentration == 5)
    {
        materialGUI.concentrationDore = m_concentrationDore[0];
        materialGUI.concentrationIrise = 0.f;
        materialGUI.dye_ConcentrationIriseDore = m_dye_ConcentrationGold[0];        
    }
    if (materialGUI.int_IriseDore_Concentration == 6)
    {
        materialGUI.concentrationDore = m_concentrationDore[1];
        materialGUI.concentrationIrise = 0.f;
        materialGUI.dye_ConcentrationIriseDore = m_dye_ConcentrationGold[1];       
    }
    if (materialGUI.int_IriseDore_Concentration == 7)
    {
        materialGUI.concentrationDore = m_concentrationDore[2];
        materialGUI.concentrationIrise = 0.f;
        materialGUI.dye_ConcentrationIriseDore = m_dye_ConcentrationGold[2];        
    }
    if (materialGUI.int_IriseDore_Concentration == 8)
    {
        materialGUI.concentrationDore = m_concentrationDore[3];
        materialGUI.concentrationIrise = 0.f;
        materialGUI.dye_ConcentrationIriseDore = m_dye_ConcentrationGold[3];        
    }
}

void Application::updateDYE( MaterialGUI& materialGUI)
{   
        float3 cendre = make_float3(materialGUI.cendre.x * materialGUI.concentrationCendre,
            materialGUI.cendre.y * materialGUI.concentrationCendre,
            materialGUI.cendre.z * materialGUI.concentrationCendre);

        float3 irise = make_float3(materialGUI.irise.x * materialGUI.concentrationIrise,
            materialGUI.irise.y * materialGUI.concentrationIrise,
            materialGUI.irise.z * materialGUI.concentrationIrise);

        float3 doree = make_float3(materialGUI.doree.x * materialGUI.concentrationDore,
            materialGUI.doree.y * materialGUI.concentrationDore,
            materialGUI.doree.z * materialGUI.concentrationDore);

        float3 cuivre = make_float3(materialGUI.cuivre.x * materialGUI.concentrationCuivre,
            materialGUI.cuivre.y * materialGUI.concentrationCuivre,
            materialGUI.cuivre.z * materialGUI.concentrationCuivre);

        float3 acajou = make_float3(materialGUI.acajou.x * materialGUI.concentrationAcajou,
            materialGUI.acajou.y * materialGUI.concentrationAcajou,
            materialGUI.acajou.z * materialGUI.concentrationAcajou);

        float3 red = make_float3(materialGUI.red.x * materialGUI.concentrationRouge,
            materialGUI.red.y * materialGUI.concentrationRouge,
            materialGUI.red.z * materialGUI.concentrationRouge);

        float3 vert = make_float3(materialGUI.vert.x * materialGUI.concentrationVert,
            materialGUI.vert.y * materialGUI.concentrationVert,
            materialGUI.vert.z * materialGUI.concentrationVert);

        float3 moyenRGB = make_float3(cendre.x + irise.x + doree.x + cuivre.x + acajou.x + red.x + vert.x,
            cendre.y + irise.y + doree.y + cuivre.y + acajou.y+ red.y + vert.y, 
            cendre.z + irise.z+ doree.z + cuivre.z + acajou.z + red.z + vert.z);

        float coeffisient = (materialGUI.concentrationCendre +
                materialGUI.concentrationIrise +
                materialGUI.concentrationDore +
                materialGUI.concentrationCuivre +
                materialGUI.concentrationAcajou +
                materialGUI.concentrationRouge +
                materialGUI.concentrationVert);

        float3 rgb;
        if(coeffisient != 0)
            rgb = make_float3(moyenRGB.x / coeffisient, moyenRGB.y / coeffisient, moyenRGB.z / coeffisient);
        else
            rgb = make_float3(255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0);            

       materialGUI.dye = rgb;
}

void Application::guiRenderingIndicator(const bool isRendering)
{
  // NVIDIA Green when rendering is complete.
  float r = 0.462745f;
  float g = 0.72549f;
  float b = 0.0f;
  
  if (isRendering)
  {
    // Neutral grey while rendering.
    r = 1.0f;
    g = 1.0f;
    b = 1.0f;
  }

  ImGuiStyle& style = ImGui::GetStyle();

  // Use the GUI window title bar color as rendering indicator. Green when rendering is completed.
  style.Colors[ImGuiCol_TitleBg]          = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
  style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
  style.Colors[ImGuiCol_TitleBgActive]    = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
}

bool Application::loadSystemDescription(std::string const& filename)
{
  Parser parser;

  if (!parser.load(filename))
  {
    std::cerr << "ERROR: loadSystemDescription() failed in loadString(" << filename << ")\n";
    return false;
  }

  ParserTokenType tokenType;
  std::string token;

  while ((tokenType = parser.getNextToken(token)) != PTT_EOF)
  {
    if (tokenType == PTT_UNKNOWN)
    {
      std::cerr << "ERROR: loadSystemDescription() " << filename << " (" << parser.getLine() << "): Unknown token type.\n";
      MY_ASSERT(!"Unknown token type.");
      return false;
    }

    if (tokenType == PTT_ID) 
    {
      if (token == "strategy")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        const int strategy = atoi(token.c_str());
        if (0 <= strategy && strategy < NUM_RENDERER_STRATEGIES)
        {
          m_strategy = static_cast<RendererStrategy>(strategy);
        }
        else
        {
          std::cerr << "WARNING: loadSystemDescription() Invalid renderer strategy " << strategy << ", using Interactive Single GPU.\n";
        }
      }
      else if (token == "devicesMask")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_devicesMask = atoi(token.c_str());
      }
      else if (token == "interop")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_interop = atoi(token.c_str());
        if (m_interop < 0 || 2 < m_interop)
        {
          std::cerr << "WARNING: loadSystemDescription() Invalid interop value " << m_interop << ", using interop 0 (host).\n";
          m_interop = 0;
        }
      }
      else if (token == "present")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_present = (atoi(token.c_str()) != 0);
      }
      else if (token == "resolution")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_resolution.x = std::max(1, atoi(token.c_str()));
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_resolution.y = std::max(1, atoi(token.c_str()));
      }
      else if (token == "tileSize")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tileSize.x = std::max(1, atoi(token.c_str()));
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tileSize.y = std::max(1, atoi(token.c_str()));
       
        // Make sure the values are power-of-two.
        if (m_tileSize.x & (m_tileSize.x - 1))
        {
          std::cerr << "ERROR: loadSystemDescription() tileSize.x = " << m_tileSize.x << " is not power-of-two, using 8.\n";
          m_tileSize.x = 8;
        }
        if (m_tileSize.y & (m_tileSize.y - 1))
        {
          std::cerr << "ERROR: loadSystemDescription() tileSize.y = " << m_tileSize.y << " is not power-of-two, using 8.\n";
          m_tileSize.y = 8;
        }
      }
      else if (token == "samplesSqrt")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_samplesSqrt = std::max(1, atoi(token.c_str()));  // spp = m_samplesSqrt * m_samplesSqrt
      }
      else if (token == "miss")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_miss = atoi(token.c_str());
      }
      else if (token == "envMap")
      {
        HDRSwitch HDR;
        tokenType = parser.getNextToken(HDR.name); // Needs to be a filename in quotation marks.
        MY_ASSERT(tokenType == PTT_STRING);

        tokenType = parser.getNextToken(HDR.file_name); // Needs to be a filename in quotation marks.
        MY_ASSERT(tokenType == PTT_STRING);
        convertPath(HDR.file_name);
        if (fs::exists(HDR.file_name)) {
            if (m_HDR.size() == 0)
                m_environment = HDR.file_name;
            m_HDR.push_back(HDR);
        }
      }
      else  if (token == "envRotation")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_environmentRotation = (float) atof(token.c_str());
      }
      else  if (token == "clockFactor")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_clockFactor = (float) atof(token.c_str());
      }
      else if (token == "light")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        int light = atoi(token.c_str());
        if (light < 0||light>5)
        {
          light = 0;
        }
        m_area_light.push_back(light);
      }
      else if (token == "pathLengths")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_pathLengths.x = atoi(token.c_str()); // min path length before Russian Roulette kicks in
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_pathLengths.y = atoi(token.c_str()); // max path length
      }
      else if (token == "epsilonFactor")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_epsilonFactor = (float) atof(token.c_str());
      }
      else if (token == "lensShader")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_lensShader = static_cast<LensShader>(atoi(token.c_str()));
        if (m_lensShader < LENS_SHADER_PINHOLE || LENS_SHADER_SPHERE < m_lensShader)
        {
          m_lensShader = LENS_SHADER_PINHOLE;
        }
      }
      else if (token == "center")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        const float x = (float) atof(token.c_str());
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        const float y = (float) atof(token.c_str());
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        const float z = (float) atof(token.c_str());
        m_camera.m_center = make_float3(x, y, z);
        m_camera.markDirty();
      }
      else if (token == "lock_camera")
      {
      m_lock_camera = true;
      }
      else if (token == "camera")
      {
        Camera camera;
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        camera.m_phi = (float) atof(token.c_str());
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        camera.m_theta = (float)atof(token.c_str());
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        camera.m_fov = (float)atof(token.c_str());
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        camera.m_distance = (float)atof(token.c_str());
        camera.markDirty();
        if (m_camera_pov.size() == 0)
        {
            m_camera.m_phi = camera.m_phi;
            m_camera.m_theta = camera.m_theta;
            m_camera.m_fov = camera.m_fov;
            m_camera.m_distance = camera.m_distance;
            m_camera.markDirty();
        }
        m_camera_pov.push_back(camera);
      }
      else if (token == "prefixScreenshot")
      {
        tokenType = parser.getNextToken(token); // Needs to be a path in quotation marks.
        MY_ASSERT(tokenType == PTT_STRING);
        convertPath(token);
        m_prefixScreenshot = token;
      }
      else if (token == "prefixColorSwitch")
      {
          tokenType = parser.getNextToken(token); // Needs to be a path in quotation marks.
          MY_ASSERT(tokenType == PTT_STRING);
          convertPath(token);
          m_prefixColorSwitch= token;
      }
      else if (token == "prefixSettings")
      {
          tokenType = parser.getNextToken(token); // Needs to be a path in quotation marks.
          MY_ASSERT(tokenType == PTT_STRING);
          convertPath(token);
          m_prefixSettings = token;
      }
      else if (token == "gamma")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tonemapperGUI.gamma = (float) atof(token.c_str());
      }
      else if (token == "colorBalance")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tonemapperGUI.colorBalance[0] = (float) atof(token.c_str());
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tonemapperGUI.colorBalance[1] = (float) atof(token.c_str());
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tonemapperGUI.colorBalance[2] = (float) atof(token.c_str());
      }
      else if (token == "whitePoint")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tonemapperGUI.whitePoint = (float) atof(token.c_str());
      }
      else if (token == "burnHighlights")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tonemapperGUI.burnHighlights = (float) atof(token.c_str());
      }
      else if (token == "crushBlacks")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tonemapperGUI.crushBlacks = (float) atof(token.c_str());
      }
      else if (token == "saturation")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tonemapperGUI.saturation = (float) atof(token.c_str());
      }
      else if (token == "brightness")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tonemapperGUI.brightness = (float) atof(token.c_str());
      }
      else if (token == "screenshotImageNum")
      {
      tokenType = parser.getNextToken(token);
      MY_ASSERT(tokenType == PTT_VAL);
      m_screenshotImageNum = atoi(token.c_str());
      }
      else if (token == "catchVariance")
      {
      tokenType = parser.getNextToken(token);
      MY_ASSERT(tokenType == PTT_VAL);
      m_catchVariance = (atoi(token.c_str()) != 0);
      }
      else
      {
        std::cerr << "WARNING: loadSystemDescription(): Unknown system option name: " << token << '\n';
      }
    }
  }

  return true;
}

bool Application::saveSystemDescription()
{
  std::ostringstream description;

  description << "strategy " << m_strategy << '\n';
  description << "devicesMask " << m_devicesMask << '\n';
  description << "interop " << m_interop << '\n';
  description << "present " << ((m_catchVariance) ? "1" : "0") << '\n';
  description << "catchVariance " << ((m_catchVariance) ? "1" : "0") << '\n';
  description << "resolution " << m_resolution.x << " " << m_resolution.y << '\n';
  description << "tileSize " << m_tileSize.x << " " << m_tileSize.y << '\n';
  description << "samplesSqrt " << m_samplesSqrt << '\n';
  description << "miss " << m_miss << '\n';
  if (!m_environment.empty())
  {
    description << "envMap \"" << m_environment << "\"\n";
  }
  description << "envRotation " << m_environmentRotation << '\n';
  description << "clockFactor " << m_clockFactor << '\n';
  for (auto i : m_area_light)
  {
      description << "light " << i << '\n';
  }
  description << "pathLengths " << m_pathLengths.x << " " << m_pathLengths.y << '\n';
  description << "epsilonFactor " << m_epsilonFactor << '\n';
  description << "lensShader " << m_lensShader << '\n';
  description << "center " << m_camera.m_center.x << " " << m_camera.m_center.y << " " << m_camera.m_center.z << '\n';
  description << "camera " << m_camera.m_phi << " " << m_camera.m_theta << " " << m_camera.m_fov << " " << m_camera.m_distance << '\n';
  if (!m_prefixScreenshot.empty())
  {
    description << "prefixScreenshot \"" << m_prefixScreenshot << "\"\n";
  }
  description << "gamma " << m_tonemapperGUI.gamma << '\n';
  description << "colorBalance " << m_tonemapperGUI.colorBalance[0] << " " << m_tonemapperGUI.colorBalance[1] << " " << m_tonemapperGUI.colorBalance[2] << '\n';
  description << "whitePoint " << m_tonemapperGUI.whitePoint << '\n';
  description << "burnHighlights " << m_tonemapperGUI.burnHighlights << '\n';
  description << "crushBlacks " << m_tonemapperGUI.crushBlacks << '\n';
  description << "saturation " << m_tonemapperGUI.saturation << '\n';
  description << "brightness " << m_tonemapperGUI.brightness << '\n';

  const std::string filename = std::string("system_optix_hair_") + getDateTime() + std::string(".txt");
  const bool success = saveString(filename, description.str());
  if (success)
  {
    std::cout << filename << '\n'; // Print out the filename to indicate success.
  }
  return success;
}

void Application::appendInstance(std::shared_ptr<sg::Group>& group,
                                 std::shared_ptr<sg::Node> geometry, 
                                 dp::math::Mat44f const& matrix, 
                                 std::string const& reference, 
                                 unsigned int& idInstance)
{
  // nvpro-pipeline matrices are row-major multiplied from the right, means the translation is in the last row. Transpose!
  const float trafo[12] =
  {
    matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0], 
    matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1], 
    matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2]
  };
 MY_ASSERT(matrix[0][3] == 0.0f && 
            matrix[1][3] == 0.0f && 
            matrix[2][3] == 0.0f && 
            matrix[3][3] == 1.0f);

  std::shared_ptr<sg::Instance> instance(new sg::Instance(idInstance++));
  instance->setTransform(trafo);
  instance->setChild(geometry);

  int indexMaterial = -1;
  std::map<std::string, int>::const_iterator itm = m_mapMaterialReferences.find(reference);
  if (itm != m_mapMaterialReferences.end())
  {
    indexMaterial = itm->second;
  }
  else
  {
    std::cerr << "WARNING: appendInstance() No material found for " << reference << ". Trying default.\n";

    std::map<std::string, int>::const_iterator itmd = m_mapMaterialReferences.find(std::string("default"));
    if (itmd != m_mapMaterialReferences.end())
    {
      indexMaterial = itmd->second;
    }
    else 
    {
      std::cerr << "ERROR: appendInstance() No default material found\n";
    }
  }

  instance->setMaterial(indexMaterial);

  group->addChild(instance);
}


bool Application::loadSceneDescription(std::string const& filename)
{
  Parser parser;

  if (!parser.load(filename))
  {
    std::cerr << "ERROR: loadSceneDescription() failed in loadString(" << filename << ")\n";
    return false;
  }

  ParserTokenType tokenType;
  std::string token;

  // Reusing some math routines from the NVIDIA nvpro-pipeline https://github.com/nvpro-pipeline/pipeline
  // Note that matrices in the nvpro-pipeline are defined row-major but are multiplied from the right,
  // which means the order of transformations is simply from left to right matrix, means first matrix is applied first,
  // but puts the translation into the last row elements (12 to 14).

  std::stack<dp::math::Mat44f> stackMatrix;
  std::stack<dp::math::Mat44f> stackInverse;      // DAR FIXME This will become important for arbitrary mesh lights.
  std::stack<dp::math::Quatf>  stackOrientation;

  // Initialize all current transformations with identity.
  curMatrix = dp::math::Mat44f(dp::math::cIdentity44f);      // object to world
  dp::math::Mat44f curInverse(dp::math::cIdentity44f);     // world to object
  dp::math::Quatf  curOrientation(0.0f, 0.0f, 0.0f, 1.0f); // object to world

  // Material parameters.
  // Original
  float3 curAlbedo          = make_float3(1.0f);
  float2 curRoughness       = make_float2(0.1f);
  float3 curAbsorptionColor = make_float3(1.0f);
  float  curAbsorptionScale = 0.0f; // 0.0f means off.
  float  curIOR             = 1.55f;
  bool   curThinwalled      = false;

  // Hair
  float  curWhitePercen     = 0.f;
  float3 curDye             = make_float3(1.f);
  float  curDyeConcentration = 0.f;
  float  curScaleAngleDeg   = 2.5f;
  float  curRoughnessN      = 0.9f;
  float  curRoughnessM      = 0.3f;
  float  curMelaninConcentration = 1.5f;
  float  curMelaninRatio   = .5f;
  float  curMelaninConcentrationDisparity = 0.1f;
  float  curMelaninRatioDisparity = .1f;

  //PSAN INIT PARAMETER NEW INTERFACE 
  int curHT = 5;
  float curf_HT = 5.0f;
  //int curQuart = 0;

  float curConcentrationCendre = 0.0f;
  float curConcentrationIrise = 0.0f;
  float curConcentrationDore = 0.0f;
  float curConcentrationCuivre = 0.0f;
  float curConcentrationAcajou = 0.0f;
  float curconcentrationRouge = 0.0f;
  float curConcentrationVert = 0.0f;

  float curConcentrationBleuOrange = 0.0f;
  float curConcentrationVertRouge = 0.0f;
  float curConcentrationVioletJaune = 0.0f;

  int curIntConcentrationVertRouge = 4;
  int curIntConcentrationCendreCuivre = 4;
  int curIntConcentrationIriseDore = 4;

  float curDYE_ConcentrationVertRouge = 0.f;
  float curDYE_ConcentrationCendreCuivre = 0.f;
  float curDYE_ConcentrationIriseDore = 0.f;

  float curdyeNeutralHT_Concentration = 0.f;

  
  // FIXME Add a mechanism to specify albedo textures per material and make that resetable or add a push/pop mechanism for materials.
  // E.g. special case filename "none" which translates to empty filename, which switches off albedo textures.
  // Get rid of the single hardcoded texture and the toggle.

  while ((tokenType = parser.getNextToken(token)) != PTT_EOF)
  {
    if (tokenType == PTT_UNKNOWN)
    {
      std::cerr << "ERROR: loadSceneDescription() " << filename << " (" << parser.getLine() << "): Unknown token type.\n";
      MY_ASSERT(!"Unknown token type.");
      return false;
    }

    if (tokenType == PTT_ID) 
    {
      std::map<std::string, KeywordScene>::const_iterator it = m_mapKeywordScene.find(token);
      if (it == m_mapKeywordScene.end())
      {
        std::cerr << "loadSceneDescription(): Unknown token " << token << " ignored.\n";
        // MY_ASSERT(!"loadSceneDescription(): Unknown token ignored.");
        continue; // Just keep getting the next token until a known keyword is found.
      }

      const KeywordScene keyword = it->second;

      switch (keyword)
      {
        case KS_ALBEDO:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          curAlbedo.x = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          curAlbedo.y = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          curAlbedo.z = (float) atof(token.c_str());
          break;

        case KS_ROUGHNESS:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          curRoughness.x = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          curRoughness.y = (float) atof(token.c_str());
          break;

        case KS_ABSORPTION: // For convenience this is an absoption color used to calculate the absorption coefficient.
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          curAbsorptionColor.x = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          curAbsorptionColor.y = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          curAbsorptionColor.z = (float) atof(token.c_str());
          break;

        case KS_ABSORPTION_SCALE:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          curAbsorptionScale = (float) atof(token.c_str());
          break;

        case KS_IOR:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          curIOR = (float) atof(token.c_str());
          break;

        case KS_THINWALLED:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          curThinwalled = (atoi(token.c_str()) != 0);
          break;

        case KS_WHITEPERCEN:
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            curWhitePercen = (atoi(token.c_str()) != 0);
            break;

        case KS_DYE:
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            curDye.x = (float)atof(token.c_str());
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            curDye.y = (float)atof(token.c_str());
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            curDye.z = (float)atof(token.c_str());
            break;

        case KS_DYE_CONCENTRATION:
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            curDyeConcentration = (atoi(token.c_str()) != 0);
            break;

        case KS_SCALE_ANGLE_DEG:
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            curScaleAngleDeg = (atoi(token.c_str()) != 0);
            break;

        case KS_ROUGHNESS_M:
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            curRoughnessM = (atoi(token.c_str()) != 0);
            break;

        case KS_ROUGHNESS_N:
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            curRoughnessN = (atoi(token.c_str()) != 0);
            break;

        case KS_MELANIN_CONCENTRATION:
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            curMelaninConcentration = (atoi(token.c_str()) != 0);
            break;

        case KS_MELANIN_RATIO:
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            curMelaninRatio = (atoi(token.c_str()) != 0);
            break;

        case KS_MELANIN_CONCENTRATION_DISPARITY:
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            curMelaninConcentrationDisparity = (atoi(token.c_str()) != 0);
            break;

        case KS_MELANIN_RATIO_DISPARITY:
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            curMelaninRatioDisparity = (atoi(token.c_str()) != 0);
            break;

        case KS_MATERIAL:
          {
            std::string nameMaterialReference;
            tokenType = parser.getNextToken(nameMaterialReference); // Internal material name. If there are duplicates the last name wins.
            //MY_ASSERT(tokenType == PTT_ID); // Allow any type of identifier, including strings and numbers.

            std::string nameMaterial;
            tokenType = parser.getNextToken(nameMaterial); // The actual material name.
            //MY_ASSERT(tokenType == PTT_ID); // Allow any type of identifier, including strings and numbers.

            // Create this material in the GUI.
            const int indexMaterial = static_cast<int>(m_materialsGUI.size());

            MaterialGUI materialGUI;

            materialGUI.name = nameMaterialReference;

            if (std::string::npos != nameMaterialReference.find("02_-_Default"))
            {
                materialGUI.name = "Eye";
            }
            if (std::string::npos != nameMaterialReference.find("Material__11"))
            {
                materialGUI.name = "Head";
            }
            if (std::string::npos != nameMaterialReference.find("Material__12"))
            {
                materialGUI.name = "Eyelash";
            }

            materialGUI.indexBSDF = INDEX_BRDF_DIFFUSE; // Set a default BSDF. // Base direct callable index for the BXDFs.
            // Handle all cases to get the correct error.
            // DAR FIXME Put these into a std::map and do a fined here.
            if (nameMaterial == std::string("brdf_diffuse"))
            {
              materialGUI.indexBSDF = INDEX_BRDF_DIFFUSE;
            }
            else if (nameMaterial == std::string("brdf_specular"))
            {
              materialGUI.indexBSDF = INDEX_BRDF_SPECULAR;
            }
            else if (nameMaterial == std::string("bsdf_specular"))
            {
              materialGUI.indexBSDF = INDEX_BSDF_SPECULAR;
            }
            else if (nameMaterial == std::string("brdf_ggx_smith"))
            {
              materialGUI.indexBSDF = INDEX_BRDF_GGX_SMITH;
            }
            else if (nameMaterial == std::string("bsdf_ggx_smith"))
            {
              materialGUI.indexBSDF = INDEX_BSDF_GGX_SMITH;
            }
            else if (nameMaterial == std::string("bsdf_hair"))
            {
              materialGUI.indexBSDF = INDEX_BCSDF_HAIR;
            }
            else
            {
              std::cerr << "WARNING: loadSceneDescription() unknown material " << nameMaterial << '\n';
            }

            materialGUI.albedo                  = curAlbedo;
            materialGUI.roughness               = curRoughness;
            materialGUI.absorptionColor         = curAbsorptionColor;
            materialGUI.absorptionScale         = curAbsorptionScale;
            materialGUI.ior                     = curIOR;
            materialGUI.thinwalled              = curThinwalled;
            materialGUI.whitepercen             = curWhitePercen;
            materialGUI.dye                     = curDye;
            materialGUI.dye_concentration       = curDyeConcentration;
            materialGUI.scale_angle_deg         = curScaleAngleDeg;
            materialGUI.roughnessM              = curRoughnessM;
            materialGUI.roughnessN              = curRoughnessN;
            materialGUI.melanin_concentration   = curMelaninConcentration;
            materialGUI.melanin_ratio           = curMelaninRatio;
            materialGUI.melanin_concentration_disparity = curMelaninConcentrationDisparity;
            materialGUI.melanin_ratio_disparity = curMelaninRatioDisparity;
            
            materialGUI.dyeNeutralHT = make_float3(1.0f, 1.0f, 1.0f);
            materialGUI.dyeNeutralHT_Concentration = 1.0f;
materialGUI.dyeNeutralHT = make_float3(84.0f / 255.f, 182.f / 255.f, 157.f / 255.f);
            //PSAN ADD PARAMETER NEW INTEFACE
            materialGUI.HT = curHT ;
            materialGUI.f_HT = curf_HT;
            //materialGUI.quart = curQuart;
            materialGUI.concentrationCendre = curConcentrationCendre ;
            materialGUI.concentrationIrise = curConcentrationIrise ;
            materialGUI.concentrationDore = curConcentrationDore ;
            materialGUI.concentrationCuivre = curConcentrationCuivre ;
            materialGUI.concentrationAcajou = curConcentrationAcajou ;
            materialGUI.concentrationRouge = curconcentrationRouge ;
            materialGUI.concentrationVert = curConcentrationVert ;

            materialGUI.concentrationBleuOrange = curConcentrationBleuOrange;
            materialGUI.concentrationVertRouge = curConcentrationVertRouge;
            materialGUI.concentrationVioletJaune = curConcentrationVioletJaune;

            materialGUI.int_VertRouge_Concentration = curIntConcentrationVertRouge;
            materialGUI.int_CendreCuivre_Concentration = curIntConcentrationCendreCuivre;
            materialGUI.int_IriseDore_Concentration = curIntConcentrationIriseDore;

            materialGUI.dye_ConcentrationVertRouge = curDYE_ConcentrationVertRouge;
            materialGUI.dye_ConcentrationCendreCuivre = curDYE_ConcentrationCendreCuivre;
            materialGUI.dye_ConcentrationIriseDore = curDYE_ConcentrationIriseDore;

            //materialGUI.dyeNeutralHT_Concentration = curdyeNeutralHT_Concentration;
            if(materialGUI.indexBSDF == INDEX_BCSDF_HAIR)            materialGUI.shouldModify = true;
            m_materialsGUI.push_back(materialGUI); // at indexMaterial.

            m_mapMaterialReferences[nameMaterialReference] = indexMaterial; // FIXME Change this to a full blown material system later.
          }
          break;
        case KS_COLOR:
          {
            std::string nameMaterial;
            tokenType = parser.getNextToken(nameMaterial);
           
            ColorSwitch materials;
            materials.name = nameMaterial;

            std::string filePath;
            tokenType = parser.getNextToken(filePath);
            convertPath(filePath);
            if (fs::exists(filePath))
            {
                //we take all the default value (we'll need to modify them dynamically when used)
                materials.Material1.indexBSDF = INDEX_BRDF_DIFFUSE;
                materials.Material1.albedo = curAlbedo;
                materials.Material1.roughness = curRoughness;
                materials.Material1.absorptionColor = curAbsorptionColor;
                materials.Material1.absorptionScale = curAbsorptionScale;
                materials.Material1.ior = curIOR;
                materials.Material1.thinwalled = curThinwalled;
                materials.Material2.indexBSDF = INDEX_BRDF_DIFFUSE;
                materials.Material2.albedo = curAlbedo;
                materials.Material2.roughness = curRoughness;
                materials.Material2.absorptionColor = curAbsorptionColor;
                materials.Material2.absorptionScale = curAbsorptionScale;
                materials.Material2.ior = curIOR;
                materials.Material2.thinwalled = curThinwalled;

                float x = 1.0f, y = 1.0f, z = 1.0f;
                std::ifstream file(filePath, std::ios::in);
                if (file)
                {
                    std::string line;
                    std::getline(file, line);
                    std::string::size_type dotposition, previousdotposition;
                    dotposition = line.find_first_of(";", 0);
                    if (dotposition != std::string::npos)
                    {
                        x = std::stof(line.substr(0, dotposition)) / 255.0f;
                        previousdotposition = dotposition + 1;
                        dotposition = line.find_first_of(";", previousdotposition);
                        if (dotposition != std::string::npos)
                        {
                            y = std::stof(line.substr(previousdotposition, dotposition)) / 255.0f;
                            previousdotposition = dotposition + 1;
                            z = std::stof(line.substr(previousdotposition, std::string::npos)) / 255.0f;
                        }
                    }
                    materials.Material1.dye = make_float3(x, y, z);

                    std::getline(file, line);
                    materials.Material1.dye_concentration = std::stof(line);

                    std::getline(file, line);
                    materials.Material1.whitepercen = std::stof(line);

                    std::getline(file, line);
                    materials.Material1.scale_angle_deg = std::stof(line);

                    std::getline(file, line);
                    materials.Material1.roughnessM = std::stof(line);

                    std::getline(file, line);
                    materials.Material1.roughnessN = std::stof(line);

                    std::getline(file, line);
                    materials.Material1.melanin_concentration = std::stof(line);

                    std::getline(file, line);
                    materials.Material1.melanin_ratio = std::stof(line);

                    std::getline(file, line);
                    materials.Material1.melanin_concentration_disparity = std::stof(line);

                    std::getline(file, line);
                    materials.Material1.melanin_ratio_disparity = std::stof(line);

                    std::getline(file, line);

                    x = y = z = 1.0f;
                    dotposition = 0;
                    previousdotposition = 0;
                    dotposition = line.find_first_of(";", 0);
                    if (dotposition != std::string::npos)
                    {
                        x = std::stof(line.substr(0, dotposition)) / 255.0f;
                        previousdotposition = dotposition + 1;
                        dotposition = line.find_first_of(";", previousdotposition);
                        if (dotposition != std::string::npos)
                        {
                            y = std::stof(line.substr(previousdotposition, dotposition)) / 255.0f;
                            previousdotposition = dotposition + 1;
                            z = std::stof(line.substr(previousdotposition, std::string::npos)) / 255.0f;
                        }
                    }
                    materials.Material1.dyeNeutralHT = make_float3(x, y, z);

                    std::getline(file, line);
                    materials.Material1.dyeNeutralHT_Concentration = std::stof(line);

                    std::getline(file, line);
                    materials.Material1.HT = std::stoi(line);

                    std::getline(file, line);
                    materials.Material1.int_VertRouge_Concentration = static_cast<int>(std::stof(line));

                    std::getline(file, line);
                    materials.Material1.int_CendreCuivre_Concentration = std::stoi(line);

                    std::getline(file, line);
                    materials.Material1.int_IriseDore_Concentration = std::stoi(line);

                    std::getline(file, line);
                    dotposition = 0;
                    previousdotposition = 0;
                    dotposition = line.find_first_of(";", 0);
                    if (dotposition != std::string::npos)
                    {
                        x = std::stof(line.substr(0, dotposition)) / 255.0f;
                        previousdotposition = dotposition + 1;
                        dotposition = line.find_first_of(";", previousdotposition);
                        if (dotposition != std::string::npos)
                        {
                            y = std::stof(line.substr(previousdotposition, dotposition)) / 255.0f;
                            previousdotposition = dotposition + 1;
                            z = std::stof(line.substr(previousdotposition, std::string::npos)) / 255.0f;
                        }
                    }
                    materials.Material1.cendre = make_float3(x, y, z);

                    std::getline(file, line);
                    dotposition = 0;
                    previousdotposition = 0;
                    dotposition = line.find_first_of(";", 0);
                    if (dotposition != std::string::npos)
                    {
                        x = std::stof(line.substr(0, dotposition)) / 255.0f;
                        previousdotposition = dotposition + 1;
                        dotposition = line.find_first_of(";", previousdotposition);
                        if (dotposition != std::string::npos)
                        {
                            y = std::stof(line.substr(previousdotposition, dotposition)) / 255.0f;
                            previousdotposition = dotposition + 1;
                            z = std::stof(line.substr(previousdotposition, std::string::npos)) / 255.0f;
                        }
                    }
                    materials.Material1.irise = make_float3(x, y, z);

                    std::getline(file, line);
                    dotposition = 0;
                    previousdotposition = 0;
                    dotposition = line.find_first_of(";", 0);
                    if (dotposition != std::string::npos)
                    {
                        x = std::stof(line.substr(0, dotposition)) / 255.0f;
                        previousdotposition = dotposition + 1;
                        dotposition = line.find_first_of(";", previousdotposition);
                        if (dotposition != std::string::npos)
                        {
                            y = std::stof(line.substr(previousdotposition, dotposition)) / 255.0f;
                            previousdotposition = dotposition + 1;
                            z = std::stof(line.substr(previousdotposition, std::string::npos)) / 255.0f;
                        }
                    }
                    materials.Material1.doree = make_float3(x, y, z);

                    std::getline(file, line);
                    dotposition = 0;
                    previousdotposition = 0;
                    dotposition = line.find_first_of(";", 0);
                    if (dotposition != std::string::npos)
                    {
                        x = std::stof(line.substr(0, dotposition)) / 255.0f;
                        previousdotposition = dotposition + 1;
                        dotposition = line.find_first_of(";", previousdotposition);
                        if (dotposition != std::string::npos)
                        {
                            y = std::stof(line.substr(previousdotposition, dotposition)) / 255.0f;
                            previousdotposition = dotposition + 1;
                            z = std::stof(line.substr(previousdotposition, std::string::npos)) / 255.0f;
                        }
                    }
                    materials.Material1.cuivre = make_float3(x, y, z);

                    std::getline(file, line);
                    dotposition = 0;
                    previousdotposition = 0;
                    dotposition = line.find_first_of(";", 0);
                    if (dotposition != std::string::npos)
                    {
                        x = std::stof(line.substr(0, dotposition)) / 255.0f;
                        previousdotposition = dotposition + 1;
                        dotposition = line.find_first_of(";", previousdotposition);
                        if (dotposition != std::string::npos)
                        {
                            y = std::stof(line.substr(previousdotposition, dotposition)) / 255.0f;
                            previousdotposition = dotposition + 1;
                            z = std::stof(line.substr(previousdotposition, std::string::npos)) / 255.0f;
                        }
                    }
                    materials.Material1.acajou = make_float3(x, y, z);

                    std::getline(file, line);
                    dotposition = 0;
                    previousdotposition = 0;
                    dotposition = line.find_first_of(";", 0);
                    if (dotposition != std::string::npos)
                    {
                        x = std::stof(line.substr(0, dotposition)) / 255.0f;
                        previousdotposition = dotposition + 1;
                        dotposition = line.find_first_of(";", previousdotposition);
                        if (dotposition != std::string::npos)
                        {
                            y = std::stof(line.substr(previousdotposition, dotposition)) / 255.0f;
                            previousdotposition = dotposition + 1;
                            z = std::stof(line.substr(previousdotposition, std::string::npos)) / 255.0f;
                        }
                    }
                    materials.Material1.red = make_float3(x, y, z);

                    std::getline(file, line);
                    dotposition = 0;
                    previousdotposition = 0;
                    dotposition = line.find_first_of(";", 0);
                    if (dotposition != std::string::npos)
                    {
                        x = std::stof(line.substr(0, dotposition)) / 255.0f;
                        previousdotposition = dotposition + 1;
                        dotposition = line.find_first_of(";", previousdotposition);
                        if (dotposition != std::string::npos)
                        {
                            y = std::stof(line.substr(previousdotposition, dotposition)) / 255.0f;
                            previousdotposition = dotposition + 1;
                            z = std::stof(line.substr(previousdotposition, std::string::npos)) / 255.0f;
                        }
                    }
                    materials.Material1.vert = make_float3(x, y, z);

                    std::getline(file, line);
                    materials.Material1.concentrationCendre = std::stof(line);

                    std::getline(file, line);
                    materials.Material1.concentrationIrise = std::stof(line);

                    std::getline(file, line);
                    materials.Material1.concentrationDore = std::stof(line);

                    std::getline(file, line);
                    materials.Material1.concentrationCuivre = std::stof(line);

                    std::getline(file, line);
                    materials.Material1.concentrationAcajou = std::stof(line);

                    std::getline(file, line);
                    materials.Material1.concentrationRouge = std::stof(line);

                    std::getline(file, line);
                    materials.Material1.concentrationVert = std::stof(line);

                    std::getline(file, line);

                    x = y = z = 1.0f;
                    dotposition = 0;
                    previousdotposition = 0;
                    dotposition = line.find_first_of(";", 0);
                    if (dotposition != std::string::npos)
                    {
                        x = std::stof(line.substr(0, dotposition)) / 255.0f;
                        previousdotposition = dotposition + 1;
                        dotposition = line.find_first_of(";", previousdotposition);
                        if (dotposition != std::string::npos)
                        {
                            y = std::stof(line.substr(previousdotposition, dotposition)) / 255.0f;
                            previousdotposition = dotposition + 1;
                            z = std::stof(line.substr(previousdotposition, std::string::npos)) / 255.0f;
                        }
                    }
                    materials.Material2.dye = make_float3(x, y, z);

                    std::getline(file, line);
                    materials.Material2.dye_concentration = std::stof(line);

                    std::getline(file, line);
                    materials.Material2.whitepercen = std::stof(line);

                    std::getline(file, line);
                    materials.Material2.scale_angle_deg = std::stof(line);

                    std::getline(file, line);
                    materials.Material2.roughnessM = std::stof(line);

                    std::getline(file, line);
                    materials.Material2.roughnessN = std::stof(line);

                    std::getline(file, line);
                    materials.Material2.melanin_concentration = std::stof(line);

                    std::getline(file, line);
                    materials.Material2.melanin_ratio = std::stof(line);

                    std::getline(file, line);
                    materials.Material2.melanin_concentration_disparity = std::stof(line);

                    std::getline(file, line);
                    materials.Material2.melanin_ratio_disparity = std::stof(line);

                    std::getline(file, line);

                    x = y = z = 1.0f;
                    dotposition = 0;
                    previousdotposition = 0;
                    dotposition = line.find_first_of(";", 0);
                    if (dotposition != std::string::npos)
                    {
                        x = std::stof(line.substr(0, dotposition)) / 255.0f;
                        previousdotposition = dotposition + 1;
                        dotposition = line.find_first_of(";", previousdotposition);
                        if (dotposition != std::string::npos)
                        {
                            y = std::stof(line.substr(previousdotposition, dotposition)) / 255.0f ;
                            previousdotposition = dotposition + 1;
                            z = std::stof(line.substr(previousdotposition, std::string::npos)) / 255.0f;
                        }
                    }
                    materials.Material2.dyeNeutralHT = make_float3(x, y, z);
                    std::getline(file, line);
                    materials.Material2.dyeNeutralHT_Concentration = std::stof(line);

                    std::getline(file, line);
                    materials.Material2.HT = std::stoi(line);

                    std::getline(file, line);
                    materials.Material2.int_VertRouge_Concentration = static_cast<int>(std::stof(line));

                    std::getline(file, line);
                    materials.Material2.int_CendreCuivre_Concentration = std::stoi(line);

                    std::getline(file, line);
                    materials.Material2.int_IriseDore_Concentration = std::stoi(line);
                    std::getline(file, line);
                    dotposition = 0;
                    previousdotposition = 0;
                    dotposition = line.find_first_of(";", 0);
                    if (dotposition != std::string::npos)
                    {
                        x = std::stof(line.substr(0, dotposition)) / 255.0f;
                        previousdotposition = dotposition + 1;
                        dotposition = line.find_first_of(";", previousdotposition);
                        if (dotposition != std::string::npos)
                        {
                            y = std::stof(line.substr(previousdotposition, dotposition)) / 255.0f;
                            previousdotposition = dotposition + 1;
                            z = std::stof(line.substr(previousdotposition, std::string::npos)) / 255.0f;
                        }
                    }
                    materials.Material2.cendre = make_float3(x, y, z);

                    std::getline(file, line);
                    dotposition = 0;
                    previousdotposition = 0;
                    dotposition = line.find_first_of(";", 0);
                    if (dotposition != std::string::npos)
                    {
                        x = std::stof(line.substr(0, dotposition)) / 255.0f;
                        previousdotposition = dotposition + 1;
                        dotposition = line.find_first_of(";", previousdotposition);
                        if (dotposition != std::string::npos)
                        {
                            y = std::stof(line.substr(previousdotposition, dotposition)) / 255.0f;
                            previousdotposition = dotposition + 1;
                            z = std::stof(line.substr(previousdotposition, std::string::npos)) / 255.0f;
                        }
                    }
                    materials.Material2.irise = make_float3(x, y, z);

                    std::getline(file, line);
                    dotposition = 0;
                    previousdotposition = 0;
                    dotposition = line.find_first_of(";", 0);
                    if (dotposition != std::string::npos)
                    {
                        x = std::stof(line.substr(0, dotposition)) / 255.0f;
                        previousdotposition = dotposition + 1;
                        dotposition = line.find_first_of(";", previousdotposition);
                        if (dotposition != std::string::npos)
                        {
                            y = std::stof(line.substr(previousdotposition, dotposition)) / 255.0f;
                            previousdotposition = dotposition + 1;
                            z = std::stof(line.substr(previousdotposition, std::string::npos)) / 255.0f;
                        }
                    }
                    materials.Material2.doree = make_float3(x, y, z);

                    std::getline(file, line);
                    dotposition = 0;
                    previousdotposition = 0;
                    dotposition = line.find_first_of(";", 0);
                    if (dotposition != std::string::npos)
                    {
                        x = std::stof(line.substr(0, dotposition)) / 255.0f;
                        previousdotposition = dotposition + 1;
                        dotposition = line.find_first_of(";", previousdotposition);
                        if (dotposition != std::string::npos)
                        {
                            y = std::stof(line.substr(previousdotposition, dotposition)) / 255.0f;
                            previousdotposition = dotposition + 1;
                            z = std::stof(line.substr(previousdotposition, std::string::npos)) / 255.0f;
                        }
                    }
                    materials.Material2.cuivre = make_float3(x, y, z);

                    std::getline(file, line);
                    dotposition = 0;
                    previousdotposition = 0;
                    dotposition = line.find_first_of(";", 0);
                    if (dotposition != std::string::npos)
                    {
                        x = std::stof(line.substr(0, dotposition)) / 255.0f;
                        previousdotposition = dotposition + 1;
                        dotposition = line.find_first_of(";", previousdotposition);
                        if (dotposition != std::string::npos)
                        {
                            y = std::stof(line.substr(previousdotposition, dotposition)) / 255.0f;
                            previousdotposition = dotposition + 1;
                            z = std::stof(line.substr(previousdotposition, std::string::npos)) / 255.0f;
                        }
                    }
                    materials.Material2.acajou = make_float3(x, y, z);

                    std::getline(file, line);
                    dotposition = 0;
                    previousdotposition = 0;
                    dotposition = line.find_first_of(";", 0);
                    if (dotposition != std::string::npos)
                    {
                        x = std::stof(line.substr(0, dotposition)) / 255.0f;
                        previousdotposition = dotposition + 1;
                        dotposition = line.find_first_of(";", previousdotposition);
                        if (dotposition != std::string::npos)
                        {
                            y = std::stof(line.substr(previousdotposition, dotposition)) / 255.0f;
                            previousdotposition = dotposition + 1;
                            z = std::stof(line.substr(previousdotposition, std::string::npos)) / 255.0f;
                        }
                    }
                    materials.Material2.red = make_float3(x, y, z);

                    std::getline(file, line);
                    dotposition = 0;
                    previousdotposition = 0;
                    dotposition = line.find_first_of(";", 0);
                    if (dotposition != std::string::npos)
                    {
                        x = std::stof(line.substr(0, dotposition)) / 255.0f;
                        previousdotposition = dotposition + 1;
                        dotposition = line.find_first_of(";", previousdotposition);
                        if (dotposition != std::string::npos)
                        {
                            y = std::stof(line.substr(previousdotposition, dotposition)) / 255.0f;
                            previousdotposition = dotposition + 1;
                            z = std::stof(line.substr(previousdotposition, std::string::npos)) / 255.0f;
                        }
                    }
                    materials.Material2.vert = make_float3(x, y, z);

                    std::getline(file, line);
                    materials.Material2.concentrationCendre = std::stof(line);

                    std::getline(file, line);
                    materials.Material2.concentrationIrise = std::stof(line);

                    std::getline(file, line);
                    materials.Material2.concentrationDore = std::stof(line);

                    std::getline(file, line);
                    materials.Material2.concentrationCuivre = std::stof(line);

                    std::getline(file, line);
                    materials.Material2.concentrationAcajou = std::stof(line);

                    std::getline(file, line);
                    materials.Material2.concentrationRouge = std::stof(line);

                    std::getline(file, line);
                    materials.Material2.concentrationVert = std::stof(line);

                    std::getline(file, line);
                    materials.SettingFile = line;
                    m_materialsColor.push_back(materials);
                }
            }
          }
          break;
        case KS_SETTING:
            {
                
                std::pair<std::string, std::string> Setting;

                tokenType = parser.getNextToken(Setting.first);
                tokenType = parser.getNextToken(Setting.second);
                convertPath(Setting.second);
                if (fs::exists(Setting.second))
                    m_settings.push_back(Setting);
            }
            break;
        case KS_IDENTITY:
          curMatrix      = dp::math::cIdentity44f;
          curInverse     = dp::math::cIdentity44f;
          curOrientation = dp::math::Quatf(0.0f, 0.0f, 0.0f, 1.0f); // identity orientation
          break;

        case KS_PUSH:
          stackMatrix.push(curMatrix);
          stackInverse.push(curInverse);
          stackOrientation.push(curOrientation);
          break;

        case KS_POP:
          if (!stackMatrix.empty())
          {
            MY_ASSERT(!stackInverse.empty());
            MY_ASSERT(!stackOrientation.empty());
            curMatrix = stackMatrix.top();
            stackMatrix.pop();
            curInverse = stackInverse.top();
            stackInverse.pop();
            curOrientation = stackOrientation.top();
            stackOrientation.pop();
          }
          else
          {
            std::cerr << "ERROR: loadSceneDescription() pop on empty stack. Resetting to identity.\n";
            curMatrix      = dp::math::cIdentity44f;
            curInverse     = dp::math::cIdentity44f;
            curOrientation = dp::math::Quatf(0.0f, 0.0f, 0.0f, 1.0f); // identity orientation
          }
          break;

        case KS_ROTATE:
          {
            dp::math::Vec3f axis;

            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            axis[0] = (float) atof(token.c_str());
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            axis[1] = (float) atof(token.c_str());
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            axis[2] = (float) atof(token.c_str());
            axis.normalize();

            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const float angle = dp::math::degToRad((float) atof(token.c_str()));

            dp::math::Quatf rotation(axis, angle);
            curOrientation *= rotation;
        
            dp::math::Mat44f matrix(rotation, dp::math::Vec3f(0.0f, 0.0f, 0.0f)); // Zero translation to get a Mat44f back. 
            curMatrix *= matrix; // DEBUG No need for the local matrix variable.
        
            // Inverse. Opposite order of matrix multiplications to make M * M^-1 = I.
            dp::math::Quatf rotationInv(axis, -angle);
            dp::math::Mat44f matrixInv(rotationInv, dp::math::Vec3f(0.0f, 0.0f, 0.0f)); // Zero translation to get a Mat44f back. 
            curInverse = matrixInv * curInverse; // DEBUG No need for the local matrixInv variable.
          }
          break;

        case KS_SCALE:
          {
            dp::math::Mat44f scaling(dp::math::cIdentity44f);

            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            scaling[0][0] = (float) atof(token.c_str());
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            scaling[1][1] = (float) atof(token.c_str());
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            scaling[2][2] = (float) atof(token.c_str());

            curMatrix *= scaling;

            // Inverse. // DEBUG Requires scalings to not contain zeros.
            scaling[0][0] = 1.0f / scaling[0][0];
            scaling[1][1] = 1.0f / scaling[1][1];
            scaling[2][2] = 1.0f / scaling[2][2];

            curInverse = scaling * curInverse;
          }
          break;

        case KS_TRANSLATE:
          {
            dp::math::Mat44f translation(dp::math::cIdentity44f);
        
            // Translation is in the third row in dp::math::Mat44f.
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            translation[3][0] = (float) atof(token.c_str());
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            translation[3][1] = (float) atof(token.c_str());
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            translation[3][2] = (float) atof(token.c_str());

            curMatrix *= translation;

            translation[3][0] = -translation[3][0];
            translation[3][1] = -translation[3][1];
            translation[3][2] = -translation[3][2];

            curInverse = translation * curInverse;
          }
          break;

        case KS_MODEL:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_ID);
       
          if (token == "plane")
          {
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const unsigned int tessU = atoi(token.c_str());

            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const unsigned int tessV = atoi(token.c_str());

            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const unsigned int upAxis = atoi(token.c_str());

            std::string nameMaterialReference;
            tokenType = parser.getNextToken(nameMaterialReference);
                    
            std::ostringstream keyGeometry;
            keyGeometry << "plane_" << tessU << "_" << tessV << "_" << upAxis;

            std::shared_ptr<sg::Triangles> geometry;

            std::map<std::string, unsigned int>::const_iterator itg = m_mapGeometries.find(keyGeometry.str());
            if (itg == m_mapGeometries.end())
            {
              m_mapGeometries[keyGeometry.str()] = m_idGeometry; // PERF Equal to static_cast<unsigned int>(m_geometries.size());

              geometry = std::make_shared<sg::Triangles>(m_idGeometry++);
              geometry->createPlane(tessU, tessV, upAxis);

              m_geometries.push_back(geometry);
            }
            else
            {
              geometry = std::dynamic_pointer_cast<sg::Triangles>(m_geometries[itg->second]);
            }

            appendInstance(m_scene, geometry, curMatrix, nameMaterialReference, m_idInstance);
          }
          else if (token == "box")
          {
            std::string nameMaterialReference;
            tokenType = parser.getNextToken(nameMaterialReference);
          
            // FIXME Implement tessellation. Must be a single value to get even distributions across edges.
            std::string keyGeometry("box_1_1");

            std::shared_ptr<sg::Triangles> geometry;

            std::map<std::string, unsigned int>::const_iterator itg = m_mapGeometries.find(keyGeometry);
            if (itg == m_mapGeometries.end())
            {
              m_mapGeometries[keyGeometry] = m_idGeometry;

              geometry = std::make_shared<sg::Triangles>(m_idGeometry++);
              geometry->createBox();

              m_geometries.push_back(geometry);
            }
            else
            {
              geometry = std::dynamic_pointer_cast<sg::Triangles>(m_geometries[itg->second]);
            }

            appendInstance(m_scene, geometry, curMatrix, nameMaterialReference, m_idInstance);
          }
          else if (token == "sphere")
          {
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const unsigned int tessU = atoi(token.c_str());

            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const unsigned int tessV = atoi(token.c_str());

            // Theta is in the range [0.0f, 1.0f] and 1.0f means closed sphere, smaller values open the noth pole.
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const float theta = float(atof(token.c_str()));

            std::string nameMaterialReference;
            tokenType = parser.getNextToken(nameMaterialReference);
                    
            std::ostringstream keyGeometry;
            keyGeometry << "sphere_" << tessU << "_" << tessV << "_" << theta;

            std::shared_ptr<sg::Triangles> geometry;

            std::map<std::string, unsigned int>::const_iterator itg = m_mapGeometries.find(keyGeometry.str());
            if (itg == m_mapGeometries.end())
            {
              m_mapGeometries[keyGeometry.str()] = m_idGeometry;

              geometry = std::make_shared<sg::Triangles>(m_idGeometry++);
              geometry->createSphere(tessU, tessV, 1.0f, theta * M_PIf);

              m_geometries.push_back(geometry);
            }
            else
            {
              geometry = std::dynamic_pointer_cast<sg::Triangles>(m_geometries[itg->second]);
            }

            appendInstance(m_scene, geometry, curMatrix, nameMaterialReference, m_idInstance);
          }
          else if (token == "torus")
          {
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const unsigned int tessU = atoi(token.c_str());

            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const unsigned int tessV = atoi(token.c_str());

            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const float innerRadius = float(atof(token.c_str()));

            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const float outerRadius = float(atof(token.c_str()));

            std::string nameMaterialReference;
            tokenType = parser.getNextToken(nameMaterialReference);
                    
            std::ostringstream keyGeometry;
            keyGeometry << "torus_" << tessU << "_" << tessV << "_" << innerRadius << "_" << outerRadius;

            std::shared_ptr<sg::Triangles> geometry;

            std::map<std::string, unsigned int>::const_iterator itg = m_mapGeometries.find(keyGeometry.str());
            if (itg == m_mapGeometries.end())
            {
              m_mapGeometries[keyGeometry.str()] = m_idGeometry;

              geometry = std::make_shared<sg::Triangles>(m_idGeometry++);
              geometry->createTorus(tessU, tessV, innerRadius, outerRadius);

              m_geometries.push_back(geometry);
            }
            else
            {
              geometry = std::dynamic_pointer_cast<sg::Triangles>(m_geometries[itg->second]);
            }

            appendInstance(m_scene, geometry, curMatrix, nameMaterialReference, m_idInstance);
          }
          else if (token == "hair")
          {

              std::ostringstream keyGeometry;
              keyGeometry << "hair_" << (m_mapGeometries.size() + 1);

              ModelSwitch model;
              tokenType = parser.getNextToken(model.name);
              MY_ASSERT(tokenType == PTT_STRING);
              tokenType = parser.getNextToken(model.file_name); // Needs to be a path in quotation marks.
              MY_ASSERT(tokenType == PTT_STRING);
              convertPath(model.file_name);
              tokenType = parser.getNextToken(model.material1Name);
              model.material2Name = model.material1Name;
              model.map_identifier = keyGeometry.str();
              if (fs::exists(model.file_name))
              {
                  m_models.push_back(model);

                  if (m_models.size() == 1)
                  {
                      std::shared_ptr<sg::Curves> geometry;

                      std::map<std::string, unsigned int>::const_iterator itg = m_mapGeometries.find(keyGeometry.str());
                      if (itg == m_mapGeometries.end())
                      {
                          m_mapGeometries[keyGeometry.str()] = m_idGeometry;

                          geometry = std::make_shared<sg::Curves>(m_idGeometry++);
                          const char* file = model.file_name.c_str();
                          geometry->createHairFromFile(file);

                          m_geometries.push_back(geometry);
                      }
                      else
                      {
                          geometry = std::dynamic_pointer_cast<sg::Curves>(m_geometries[itg->second]);
                      }

                      appendInstance(m_scene, geometry, curMatrix, model.material1Name, m_idInstance);
                  }
              }
          }
          else if (token == "hair_modified")
          {
          std::string filenameModel;
          tokenType = parser.getNextToken(filenameModel); // Needs to be a path in quotation marks.
          MY_ASSERT(tokenType == PTT_STRING);
          convertPath(filenameModel);

          float density;
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          density = (float) atof(token.c_str());

          float disparity;
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          disparity = (float) atof(token.c_str());

          std::string nameMaterialReference;
          tokenType = parser.getNextToken(nameMaterialReference);

          std::ostringstream keyGeometry;
          keyGeometry << "hair_modified_";

          std::shared_ptr<sg::Curves> geometry;

          std::map<std::string, unsigned int>::const_iterator itg = m_mapGeometries.find(keyGeometry.str());
          if (itg == m_mapGeometries.end())
          {
              m_mapGeometries[keyGeometry.str()] = m_idGeometry;

              geometry = std::make_shared<sg::Curves>(m_idGeometry++, density, disparity );
              const char* file = filenameModel.c_str();
              geometry->createHairFromFile(file);

              m_geometries.push_back(geometry);
          }
          else
          {
              geometry = std::dynamic_pointer_cast<sg::Curves>(m_geometries[itg->second]);
          }

          appendInstance(m_scene, geometry, curMatrix, nameMaterialReference, m_idInstance);
          }
          else if (token == "hair_half")
          {

              ModelSwitch model;
              tokenType = parser.getNextToken(model.name);
              MY_ASSERT(tokenType == PTT_STRING);
              tokenType = parser.getNextToken(model.file_name); // Needs to be a path in quotation marks.
              MY_ASSERT(tokenType == PTT_STRING);
              convertPath(model.file_name);
              tokenType = parser.getNextToken(model.material1Name);
              tokenType = parser.getNextToken(model.material2Name);

              std::string HairName = model.file_name.substr(0, model.file_name.find("."));
              model.map_identifier = HairName + "_" + std::to_string(m_mapGeometries.size() + 1);
              if (fs::exists(model.file_name))
              {
                  m_models.push_back(model);

                  if (m_models.size() == 1)
                  {
                      std::ostringstream keyGeometry1;
                      keyGeometry1 << model.map_identifier << "_half_1";

                      std::ostringstream keyGeometry2;
                      keyGeometry2 << model.map_identifier << "_half_2";

                      std::shared_ptr<sg::Curves> geometry_left;
                      std::map<std::string, unsigned int>::const_iterator itg1 = m_mapGeometries.find(keyGeometry1.str());
                      if (itg1 == m_mapGeometries.end())
                      {
                          m_mapGeometries[keyGeometry1.str()] = m_idGeometry;

                          geometry_left = std::make_shared<sg::Curves>(m_idGeometry++);
                          const char* file = model.file_name.c_str();
                          geometry_left->createHairFromFile(file, true);

                          m_geometries.push_back(geometry_left);
                      }
                      else
                          geometry_left = std::dynamic_pointer_cast<sg::Curves>(m_geometries[itg1->second]);
                      appendInstance(m_scene, geometry_left, curMatrix, model.material1Name, m_idInstance);

                      std::shared_ptr<sg::Curves> geometry_right;

                      std::map<std::string, unsigned int>::const_iterator itg2 = m_mapGeometries.find(keyGeometry2.str());
                      if (itg2 == m_mapGeometries.end())
                      {
                          m_mapGeometries[keyGeometry2.str()] = m_idGeometry;

                          geometry_right = std::make_shared<sg::Curves>(m_idGeometry++);
                          const char* file = model.file_name.c_str();
                          geometry_right->createHairFromFile(file, false);

                          m_geometries.push_back(geometry_right);
                      }
                      else
                          geometry_right = std::dynamic_pointer_cast<sg::Curves>(m_geometries[itg2->second]);

                      appendInstance(m_scene, geometry_right, curMatrix, model.material2Name, m_idInstance);
                  }
              }
          }
          else if (token == "assimp")
          {
            std::string filenameModel;
            tokenType = parser.getNextToken(filenameModel); // Needs to be a path in quotation marks.
            MY_ASSERT(tokenType == PTT_STRING);
            convertPath(filenameModel);

            std::shared_ptr<sg::Group> model = createASSIMP(filenameModel);

            // nvpro-pipeline matrices are row-major multiplied from the right, means the translation is in the last row. Transpose!
            const float trafo[12] =
            {
              curMatrix[0][0], curMatrix[1][0], curMatrix[2][0], curMatrix[3][0], 
              curMatrix[0][1], curMatrix[1][1], curMatrix[2][1], curMatrix[3][1], 
              curMatrix[0][2], curMatrix[1][2], curMatrix[2][2], curMatrix[3][2]
            };

            MY_ASSERT(curMatrix[0][3] == 0.0f && 
                      curMatrix[1][3] == 0.0f && 
                      curMatrix[2][3] == 0.0f && 
                      curMatrix[3][3] == 1.0f);

            std::shared_ptr<sg::Instance> instance(new sg::Instance(m_idInstance++));
            
            instance->setTransform(trafo);
            instance->setChild(model);

            m_scene->addChild(instance);
          }
          break;

        default:
          std::cerr << "ERROR: loadSceneDescription() Unexpected KeywordScene value " << keyword << " ignored.\n";
          MY_ASSERT(!"ERROR: loadSceneDescription() Unexpected KeywordScene value");
          break;
      }
    }
  }

  std::cout << "loadSceneDescription(): m_idGroup = " << m_idGroup << ", m_idInstance = " << m_idInstance << ", m_idGeometry = " << m_idGeometry << '\n';

  return true;
}


bool Application::loadString(std::string const& filename, std::string& text)
{
  std::ifstream inputStream(filename);

  if (!inputStream)
  {
    std::cerr << "ERROR: loadString() Failed to open file " << filename << '\n';
    return false;
  }

  std::stringstream data;

  data << inputStream.rdbuf();

  if (inputStream.fail())
  {
    std::cerr << "ERROR: loadString() Failed to read file " << filename << '\n';
    return false;
  }

  text = data.str();
  return true;
}

bool Application::saveString(std::string const& filename, std::string const& text)
{
  std::ofstream outputStream(filename);

  if (!outputStream)
  {
    std::cerr << "ERROR: saveString() Failed to open file " << filename << '\n';
    return false;
  }

  outputStream << text;

  if (outputStream.fail())
  {
    std::cerr << "ERROR: saveString() Failed to write file " << filename << '\n';
    return false;
  }

  return true;
}


std::string Application::getDateTime()
{
#if defined(_WIN32)
  SYSTEMTIME time;
  GetLocalTime(&time);
#elif defined(__linux__)
  time_t rawtime;
  struct tm* ts;
  time(&rawtime);
  ts = localtime(&rawtime);
#else
  #error "OS not supported."
#endif

  std::ostringstream oss;

#if defined( _WIN32 )
  oss << time.wYear;
  if (time.wMonth < 10)
  {
    oss << '0';
  }
  oss << time.wMonth;
  if (time.wDay < 10)
  {
    oss << '0';
  }
  oss << time.wDay << '_';
  if (time.wHour < 10)
  {
    oss << '0';
  }
  oss << time.wHour;
  if (time.wMinute < 10)
  {
    oss << '0';
  }
  oss << time.wMinute;
  if (time.wSecond < 10)
  {
    oss << '0';
  }
  oss << time.wSecond << '_';
  if (time.wMilliseconds < 100)
  {
    oss << '0';
  }
  if (time.wMilliseconds <  10)
  {
    oss << '0';
  }
  oss << time.wMilliseconds; 
#elif defined(__linux__)
  oss << ts->tm_year;
  if (ts->tm_mon < 10)
  {
    oss << '0';
  }
  oss << ts->tm_mon;
  if (ts->tm_mday < 10)
  {
    oss << '0';
  }
  oss << ts->tm_mday << '_';
  if (ts->tm_hour < 10)
  {
    oss << '0';
  }
  oss << ts->tm_hour;
  if (ts->tm_min < 10)
  {
    oss << '0';
  }
  oss << ts->tm_min;
  if (ts->tm_sec < 10)
  {
    oss << '0';
  }
  oss << ts->tm_sec << '_';
  oss << "000"; // No milliseconds available.
#else
  #error "OS not supported."
#endif

  return oss.str();
}


static void updateAABB(float3& minimum, float3& maximum, float3 const& v)
{
  if (v.x < minimum.x)
  {
    minimum.x = v.x;
  }
  else if (maximum.x < v.x)
  {
    maximum.x = v.x;
  }

  if (v.y < minimum.y)
  {
    minimum.y = v.y;
  }
  else if (maximum.y < v.y)
  {
    maximum.y = v.y;
  }

  if (v.z < minimum.z)
  {
    minimum.z = v.z;
  }
  else if (maximum.z < v.z)
  {
    maximum.z = v.z;
  }
}

//static void calculateTexcoordsSpherical(std::vector<InterleavedHost>& attributes, std::vector<unsigned int> const& indices)
//{
//  dp::math::Vec3f center(0.0f, 0.0f, 0.0f);
//  for (size_t i = 0; i < attributes.size(); ++i)
//  {
//    center += attributes[i].vertex;
//  }
//  center /= (float) attributes.size();
//
//  float u;
//  float v;
//
//  for (size_t i = 0; i < attributes.size(); ++i)
//  {
//    dp::math::Vec3f p = attributes[i].vertex - center;
//    if (FLT_EPSILON < fabsf(p[1]))
//    {
//      u = 0.5f * atan2f(p[0], -p[1]) / dp::math::PI + 0.5f;
//    }
//    else
//    {
//      u = (0.0f <= p[0]) ? 0.75f : 0.25f;
//    }
//    float d = sqrtf(dp::math::square(p[0]) + dp::math::square(p[1]));
//    if (FLT_EPSILON < d)
//    {
//      v = atan2f(p[2], d) / dp::math::PI + 0.5f;
//    }
//    else
//    {
//      v = (0.0f <= p[2]) ? 1.0f : 0.0f;
//    }
//    attributes[i].texcoord0 = dp::math::Vec3f(u, v, 0.0f);
//  }
//
//  //// The code from the environment texture lookup.
//  //for (size_t i = 0; i < attributes.size(); ++i)
//  //{
//  //  dp::math::Vec3f R = attributes[i].vertex - center;
//  //  dp::math::normalize(R);
//
//  //  // The seam u == 0.0 == 1.0 is in positive z-axis direction.
//  //  // Compensate for the environment rotation done inside the direct lighting.
//  //  const float u = (atan2f(R[0], -R[2]) + dp::math::PI) * 0.5f / dp::math::PI;
//  //  const float theta = acosf(-R[1]); // theta == 0.0f is south pole, theta == M_PIf is north pole.
//  //  const float v = theta / dp::math::PI; // Texture is with origin at lower left, v == 0.0f is south pole.
//
//  //  attributes[i].texcoord0 = dp::math::Vecf(u, v, 0.0f);
//  //}
//}


// Calculate texture tangents based on the texture coordinate gradients.
// Doesn't work when all texture coordinates are identical! Thats the reason for the other routine below.
//static void calculateTangents(std::vector<InterleavedHost>& attributes, std::vector<unsigned int> const& indices)
//{
//  for (size_t i = 0; i < indices.size(); i += 4)
//  {
//    unsigned int i0 = indices[i    ];
//    unsigned int i1 = indices[i + 1];
//    unsigned int i2 = indices[i + 2];
//
//    dp::math::Vec3f e0 = attributes[i1].vertex - attributes[i0].vertex;
//    dp::math::Vec3f e1 = attributes[i2].vertex - attributes[i0].vertex;
//    dp::math::Vec2f d0 = dp::math::Vec2f(attributes[i1].texcoord0) - dp::math::Vec2f(attributes[i0].texcoord0);
//    dp::math::Vec2f d1 = dp::math::Vec2f(attributes[i2].texcoord0) - dp::math::Vec2f(attributes[i0].texcoord0);
//    attributes[i0].tangent += d1[1] * e0 - d0[1] * e1;
//
//    e0 = attributes[i2].vertex - attributes[i1].vertex;
//    e1 = attributes[i0].vertex - attributes[i1].vertex;
//    d0 = dp::math::Vec2f(attributes[i2].texcoord0) - dp::math::Vec2f(attributes[i1].texcoord0);
//    d1 = dp::math::Vec2f(attributes[i0].texcoord0) - dp::math::Vec2f(attributes[i1].texcoord0);
//    attributes[i1].tangent += d1[1] * e0 - d0[1] * e1;
//
//    e0 = attributes[i0].vertex - attributes[i2].vertex;
//    e1 = attributes[i1].vertex - attributes[i2].vertex;
//    d0 = dp::math::Vec2f(attributes[i0].texcoord0) - dp::math::Vec2f(attributes[i2].texcoord0);
//    d1 = dp::math::Vec2f(attributes[i1].texcoord0) - dp::math::Vec2f(attributes[i2].texcoord0);
//    attributes[i2].tangent += d1[1] * e0 - d0[1] * e1;
//  }
//
//  for (size_t i = 0; i < attributes.size(); ++i)
//  {
//    dp::math::Vec3f tangent(attributes[i].tangent);
//    dp::math::normalize(tangent); // This normalizes the sums from above!
//
//    dp::math::Vec3f normal(attributes[i].normal);
//
//    dp::math::Vec3f bitangent = normal ^ tangent;
//    dp::math::normalize(bitangent);
//
//    tangent = bitangent ^ normal;
//    dp::math::normalize(tangent);
//    
//    attributes[i].tangent = tangent;
//
//#if USE_BITANGENT
//    attributes[i].bitangent = bitantent;
//#endif
//  }
//}


// Calculate (geometry) tangents with the global tangent direction aligned to the biggest AABB extend of this part.
void Application::calculateTangents(std::vector<VertexAttributes>& attributes, std::vector<unsigned int> const& indices)
{
  MY_ASSERT(3 <= indices.size());

  // Initialize with the first vertex to be able to use else-if comparisons in updateAABB().
  float3 aabbLo = attributes[indices[0]].vertex;
  float3 aabbHi = attributes[indices[0]].vertex;

  // Build an axis aligned bounding box.
  for (size_t i = 0; i < indices.size(); i += 3)
  {
    unsigned int i0 = indices[i    ];
    unsigned int i1 = indices[i + 1];
    unsigned int i2 = indices[i + 2];

    updateAABB(aabbLo, aabbHi, attributes[i0].vertex);
    updateAABB(aabbLo, aabbHi, attributes[i1].vertex);
    updateAABB(aabbLo, aabbHi, attributes[i2].vertex);
  }

  // Get the longest extend and use that as general tangent direction.
  const float3 extents = aabbHi - aabbLo;
  
  float f = extents.x;
  int maxComponent = 0;

  if (f < extents.y)
  {
    f = extents.y;
    maxComponent = 1;
  }
  if (f < extents.z)
  {
    maxComponent = 2;
  }

  float3 direction;
  float3 bidirection;

  switch (maxComponent)
  {
  case 0: // x-axis
  default:
    direction   = make_float3(1.0f, 0.0f, 0.0f);
    bidirection = make_float3(0.0f, 1.0f, 0.0f);
    break;
  case 1: // y-axis // DEBUG It might make sense to keep these directions aligned to the global coordinate system. Use the same coordinates as for z-axis then.
    direction   = make_float3(0.0f, 1.0f, 0.0f); 
    bidirection = make_float3(0.0f, 0.0f, -1.0f);
    break;
  case 2: // z-axis
    direction   = make_float3(0.0f, 0.0f, -1.0f);
    bidirection = make_float3(0.0f, 1.0f,  0.0f);
    break;
  }

  // Build an ortho-normal basis with the existing normal.
  for (size_t i = 0; i < attributes.size(); ++i)
  {
    float3 tangent   = direction;
    float3 bitangent = bidirection;
    // float3 normal    = attributes[i].normal;
    float3 normal;
    normal.x = attributes[i].normal.x;
    normal.y = attributes[i].normal.y;
    normal.z = attributes[i].normal.z;

    if (0.001f < 1.0f - fabsf(dot(normal, tangent)))
    {
      bitangent = normalize(cross(normal, tangent));
      tangent   = normalize(cross(bitangent, normal));
    }
    else // Normal and tangent direction too collinear.
    {
      //MY_ASSERT(0.001f < 1.0f - fabsf(dot(bitangent, normal)));
      tangent   = normalize(cross(bitangent, normal));
      //bitangent = normalize(cross(normal, tangent));
    }
    attributes[i].tangent = tangent;
  }
}

bool Application::screenshot(const bool tonemap)
{
  ILboolean hasImage = false;
  
  const int spp = m_samplesSqrt * m_samplesSqrt; // Add the samples per pixel to the filename for quality comparisons.

  std::ostringstream path;
  std::cerr << "the camera values are\n m_phi : " << m_camera.m_phi << "\nm_theta : " << m_camera.m_theta << "\nm_fov : " << m_camera.m_fov << "\nm_distance : " << m_camera.m_distance << "\n";
  path << m_prefixScreenshot << "_" << spp << "spp_" << getDateTime();
  
  std::string tmpValue = path.str();
  auto path_value = tmpValue.find_last_of("\\");
  tmpValue = tmpValue.substr(0, path_value);
  std::cerr << tmpValue << std::endl;
  if (!fs::exists(tmpValue)) {
      fs::create_directory(tmpValue);
  }

  unsigned int imageID;

  ilGenImages(1, (ILuint *) &imageID);

  ilBindImage(imageID);
  ilActiveImage(0);
  ilActiveFace(0);

  ilDisable(IL_ORIGIN_SET);

  const float4* bufferHost = reinterpret_cast<const float4*>(m_raytracer->getOutputBufferHost());
  
  if (tonemap)
  {
    // Store a tonemapped RGB8 *.png image
    path << ".png";

    if (ilTexImage(m_resolution.x, m_resolution.y, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, nullptr))
    {
      uchar3* dst = reinterpret_cast<uchar3*>(ilGetData());

      const float  invGamma       = 1.0f / m_tonemapperGUI.gamma;
      const float3 colorBalance   = make_float3(m_tonemapperGUI.colorBalance[0], m_tonemapperGUI.colorBalance[1], m_tonemapperGUI.colorBalance[2]);
      const float  invWhitePoint  = m_tonemapperGUI.brightness / m_tonemapperGUI.whitePoint;
      const float  burnHighlights = m_tonemapperGUI.burnHighlights;
      const float  crushBlacks    = m_tonemapperGUI.crushBlacks + m_tonemapperGUI.crushBlacks + 1.0f;
      const float  saturation     = m_tonemapperGUI.saturation;

      for (int y = 0; y < m_resolution.y; ++y)
      {
        for (int x = 0; x < m_resolution.x; ++x)
        {
          const int idx = y * m_resolution.x + x;

          // Tonemapper. // PERF Add a native CUDA kernel doing this.
          float3 hdrColor = make_float3(bufferHost[idx]);
          float3 ldrColor = invWhitePoint * colorBalance * hdrColor;
          ldrColor       *= ((ldrColor * burnHighlights) + 1.0f) / (ldrColor + 1.0f);
          
          float luminance = dot(ldrColor, make_float3(0.3f, 0.59f, 0.11f));
          ldrColor = lerp(make_float3(luminance), ldrColor, saturation); // This can generate negative values for saturation > 1.0f!
          ldrColor = fmaxf(make_float3(0.0f), ldrColor); // Prevent negative values.

          luminance = dot(ldrColor, make_float3(0.3f, 0.59f, 0.11f));
          if (luminance < 1.0f)
          {
            const float3 crushed = powf(ldrColor, crushBlacks);
            ldrColor = lerp(crushed, ldrColor, sqrtf(luminance));
            ldrColor = fmaxf(make_float3(0.0f), ldrColor); // Prevent negative values.
          }
          ldrColor = clamp(powf(ldrColor, invGamma), 0.0f, 1.0f); // Saturate, clamp to range [0.0f, 1.0f].

          dst[idx] = make_uchar3((unsigned char) (ldrColor.x * 255.0f),
                                 (unsigned char) (ldrColor.y * 255.0f),
                                 (unsigned char) (ldrColor.z * 255.0f));
        }
      }
      hasImage = true;
    }
  }
  else
  {
    // Store the float4 linear output buffer as *.hdr image.
    // FIXME Add a half float conversion and store as *.exr. (Pre-built DevIL 1.7.8 supports EXR, DevIL 1.8.0 doesn't!)
    path << ".hdr";

    hasImage = ilTexImage(m_resolution.x, m_resolution.y, 1, 4, IL_RGBA, IL_FLOAT, (void*) bufferHost);
  }

  if (hasImage)
  {
    ilEnable(IL_FILE_OVERWRITE); // By default, always overwrite
    
    std::string filename = path.str();
    convertPath(filename);
	
    if (ilSaveImage((const ILstring) filename.c_str()))
    {
      ilDeleteImages(1, &imageID);

      std::cout << filename << '\n'; // Print out filename to indicate that a screenshot has been taken.
      return true;
    }
  }

  // There was an error when reaching this code.
  ILenum error = ilGetError(); // DEBUG 
  std::cerr << "ERROR: screenshot() failed with IL error " << error << '\n';

  while (ilGetError() != IL_NO_ERROR) // Clean up errors.
  {
  }

  // Free all resources associated with the DevIL image
  ilDeleteImages(1, &imageID);

  return false;
}

bool Application::screenshot360()
{
    std::string standard_prefix = std::string("./360_screenshots/" + getDateTime() + "/");

    if (!fs::exists("./360_screenshots/")) {
        fs::create_directory("./360_screenshots/");
    }
    if (!fs::exists(standard_prefix)) {
        fs::create_directory(standard_prefix);
    }
    int index = 0;
    float phi_screenshot;
    for (int i = 0; i < m_screenshotImageNum; i++) {

        phi_screenshot = i * 1.f / (1.f * m_screenshotImageNum);
        m_camera.setTheta(0.65f);
        m_camera.setPhi(phi_screenshot);
        restartRendering();

        for (int j = 0; j < 1024; j++) {
            // Iteration index is zero based
            render();
            index++;
            if (j % 32 == 0) {
                float progress = index / (1024.f * m_screenshotImageNum);

                loading_bar(progress);
            }
        }
        display();

        screenshot(true, standard_prefix + std::to_string(i));
    }
    return true;
}


bool Application::screenshot(const bool tonemap, std::string name)
{
    ILboolean hasImage = false;

    const int spp = m_samplesSqrt * m_samplesSqrt; // Add the samples per pixel to the filename for quality comparisons.

    std::ostringstream path;

    path << name;

    unsigned int imageID;

    ilGenImages(1, (ILuint*)&imageID);

    ilBindImage(imageID);
    ilActiveImage(0);
    ilActiveFace(0);

    ilDisable(IL_ORIGIN_SET);

    const float4* bufferHost = reinterpret_cast<const float4*>(m_raytracer->getOutputBufferHost());

    if (tonemap)
    {
        // Store a tonemapped RGB8 *.png image
        path << ".png";

        if (ilTexImage(m_resolution.x, m_resolution.y, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, nullptr))
        {
            uchar3* dst = reinterpret_cast<uchar3*>(ilGetData());

            const float  invGamma = 1.0f / m_tonemapperGUI.gamma;
            const float3 colorBalance = make_float3(m_tonemapperGUI.colorBalance[0], m_tonemapperGUI.colorBalance[1], m_tonemapperGUI.colorBalance[2]);
            const float  invWhitePoint = m_tonemapperGUI.brightness / m_tonemapperGUI.whitePoint;
            const float  burnHighlights = m_tonemapperGUI.burnHighlights;
            const float  crushBlacks = m_tonemapperGUI.crushBlacks + m_tonemapperGUI.crushBlacks + 1.0f;
            const float  saturation = m_tonemapperGUI.saturation;

            for (int y = 0; y < m_resolution.y; ++y)
            {
                for (int x = 0; x < m_resolution.x; ++x)
                {
                    const int idx = y * m_resolution.x + x;

                    // Tonemapper. // PERF Add a native CUDA kernel doing this.
                    float3 hdrColor = make_float3(bufferHost[idx]);
                    float3 ldrColor = invWhitePoint * colorBalance * hdrColor;
                    ldrColor *= ((ldrColor * burnHighlights) + 1.0f) / (ldrColor + 1.0f);

                    float luminance = dot(ldrColor, make_float3(0.3f, 0.59f, 0.11f));
                    ldrColor = lerp(make_float3(luminance), ldrColor, saturation); // This can generate negative values for saturation > 1.0f!
                    ldrColor = fmaxf(make_float3(0.0f), ldrColor); // Prevent negative values.

                    luminance = dot(ldrColor, make_float3(0.3f, 0.59f, 0.11f));
                    if (luminance < 1.0f)
                    {
                        const float3 crushed = powf(ldrColor, crushBlacks);
                        ldrColor = lerp(crushed, ldrColor, sqrtf(luminance));
                        ldrColor = fmaxf(make_float3(0.0f), ldrColor); // Prevent negative values.
                    }
                    ldrColor = clamp(powf(ldrColor, invGamma), 0.0f, 1.0f); // Saturate, clamp to range [0.0f, 1.0f].

                    dst[idx] = make_uchar3((unsigned char)(ldrColor.x * 255.0f),
                        (unsigned char)(ldrColor.y * 255.0f),
                        (unsigned char)(ldrColor.z * 255.0f));
                }
            }
            hasImage = true;
        }
    }
    else
    {
        // Store the float4 linear output buffer as *.hdr image.
        // FIXME Add a half float conversion and store as *.exr. (Pre-built DevIL 1.7.8 supports EXR, DevIL 1.8.0 doesn't!)
        path << ".hdr";

        hasImage = ilTexImage(m_resolution.x, m_resolution.y, 1, 4, IL_RGBA, IL_FLOAT, (void*)bufferHost);
    }

    if (hasImage)
    {
        ilEnable(IL_FILE_OVERWRITE); // By default, always overwrite

        std::string filename = path.str();
        convertPath(filename);

        if (ilSaveImage((const ILstring)filename.c_str()))
        {
            ilDeleteImages(1, &imageID);

            //std::cout << filename << '\n'; // Print out filename to indicate that a screenshot has been taken.
            return true;
        }
    }

    // There was an error when reaching this code.
    ILenum error = ilGetError(); // DEBUG 
    std::cerr << "ERROR: screenshot() failed with IL error " << error << '\n';

    while (ilGetError() != IL_NO_ERROR) // Clean up errors.
    {
    }

    // Free all resources associated with the DevIL image
    ilDeleteImages(1, &imageID);

    return false;
}


// Convert between slashes and backslashes in paths depending on the operating system
void Application::convertPath(std::string& path)
{
#if defined(_WIN32)
  std::string::size_type pos = path.find("/", 0);
  while (pos != std::string::npos)
  {
    path[pos] = '\\';
    pos = path.find("/", pos);
  }
#elif defined(__linux__)
  std::string::size_type pos = path.find("\\", 0);
  while (pos != std::string::npos)
  {
    path[pos] = '/';
    pos = path.find("\\", pos);
  }
#endif
}

void Application::convertPath(char* path)
{
#if defined(_WIN32)
  for (size_t i = 0; i < strlen(path); ++i)
  {
    if (path[i] == '/')
    {
      path[i] = '\\';
    }
  }
#elif defined(__linux__)
  for (size_t i = 0; i < strlen(path); ++i)
  {
    if (path[i] == '\\')
    {
      path[i] = '/';
    }
  }
#endif
}

float Application::captureVariance()
{
    if (m_raytracer->m_iterationIndex == 0)
        return 1.f;

    const float* varbufferHost = reinterpret_cast<const float*>(m_raytracer->getOutputVarBufferHost());
    float variance = 0.f;
    int idx = 0;
    for (int y = 0; y < m_resolution.y; ++y)
    {
        for (int x = 0; x < m_resolution.x; ++x)
        {
            idx = y * m_resolution.x + x;
            variance += varbufferHost[idx];
        }
    }
    variance /= (m_resolution.x * m_resolution.y);

    float confidence_interval = 2.f * 1.96f * sqrtf(variance / m_raytracer->m_iterationIndex);
    //std::cout << "IC :" << confidence_interval << std::endl;
    return confidence_interval;
}

bool Application::loading_bar(float progress, int bar_width ) {
    std::string loading_bar(bar_width, ' ');
    int pos = static_cast<int>(bar_width * progress);
    for (int k = 0; k < bar_width; k++) {
        if (k <= pos) loading_bar[k] = '#';
    }
    std::cout << "progress: [" << loading_bar << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
    return true;
}