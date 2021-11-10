# OptiX Applications

Advanced Samples for the [NVIDIA OptiX 7 Ray Tracing SDK](https://developer.nvidia.com/rtx/raytracing)

The goal of the three initial introduction examples is to show how to port an existing OptiX application based on the previous OptiX 5 or 6 API to OptiX 7.

For that, two of the existing [OptiX Introduction Samples](https://github.com/nvpro-samples/optix_advanced_samples/tree/master/src/optixIntroduction) have been ported to the OptiX 7.0.0 SDK.

*intro_runtime* and *intro_driver* are ports from *optixIntro_07*, and *intro_denoiser* is a port of the *optixIntro_10* example showing the built-in AI denoiser.
Those are already demonstrating some advanced methods to architect renderers using OptiX 7 on the way.

If you need a basic introduction into OptiX 7 programming, please refer to the [OptiX 7 SIGGRAPH course material](https://gitlab.com/ingowald/optix7course) first and maybe read though the [OptiX developer forum](https://devtalk.nvidia.com/default/board/254/optix/) as well for many topics about OptiX 7.

The landing page for online NVIDIA ray tracing programming guides and API reference documentation can be found here: [NVIDIA ray tracing documentation](https://raytracing-docs.nvidia.com/). This generally contains more up-to-date information compared to documents shipping with the SDKs and is easy to search including cross-reference links.

Please always read the OptiX SDK release notes before setting up a development environment.

# Overview

OptiX 7 applications are written using the CUDA programming APIs. There are two to choose from: The CUDA Runtime API and the CUDA Driver API.

The CUDA Runtime API is a little more high-level and requires a library to be shipped with the application, while the CUDA Driver API is more explicit and ships with the driver. The documentation inside the CUDA API headers cross-reference the respective function names of each other API.

To demonstrate the differences, *intro_runtime* and *intro_driver* are both a port of [OptiX Introduction sample #7](https://github.com/nvpro-samples/optix_advanced_samples/tree/master/src/optixIntroduction/optixIntro_07) just using the CUDA Runtime API resp. CUDA Driver API for easy comparison.

*intro_denoiser* is a port from [OptiX Introduction sample #10](https://github.com/nvpro-samples/optix_advanced_samples/tree/master/src/optixIntroduction/optixIntro_10) to OptiX 7.0.0.
That example is the same as intro_driver with additional code demonstrating the built-in denoiser functionality with HDR denoising on beauty and optional albedo and normal buffers, all in float4 and half4 format (compile time options in config.h).

All three intro examples implement the exact same rendering and runtime generated scene data and make use of a single device (ordinal 0) only.
(If you have multiple NVIDIA devices installed you can switch between them, by using the CUDA_VISIBLE_DEVICES environment variable.)

*rtigo3* is meant as a testbed for multi-GPU rendering ditribution and OpenGL interoperability.
There are different multi-GPU strategies implemented (single GPU, dual GPU peer-to-peer, multi-GPU pinned memory, multi-GPU local distribution and compositing). 
Then there are three different OpenGL interop modes (none, pixel buffer object, direct texture mapping).

The implementation is using the CUDA Driver API on purpose because that allows more fine grained control over CUDA contexts and devices and alleviates the need to ship a CUDA runtime library.

This example contains the same runtime generated geometry as the introduction examples, but also implements a simple file loader using [ASSIMP](https://github.com/assimp/assimp) for triangle mesh data.
The application operation and scene setup is controlled by two simple text files which also allows generating any scene setup complexity for tests.
It's not rendering infinitely as the introduction examples but uses a selectable number of camera samples, as well as render resolutions independent of the windows client area.

*nvlink_shared* demonstrates peer-to-peer sharing of texture data and/or geometry acceleration structures among GPU devices in an NVLINK island.
Note that peer-to-peer access under Windows requires Windows 10 64-bit and SLI enabled inside the NVIDIA Display Control Panel. Under Linux it should work out of the box.
Peer-to-peer device resource sharing can effectively double the scene size loaded onto a dual-GPU NVLINK setup.
Texture sharing comes at a moderate performance cost while geometry acceleration structure sharing can be up to two times slower, which is reasonably fast given the bandwidth difference between NVLINK and VRAM transfers. Still a lot better than not being able to load a scene at all on a single board.
The Raytracer class got more smarts over the Device class because the resource distribution decisions need to happen above the devices.

This example is derived from rtigo3, but uses only one rendering strategy ("local-copy") and is solely meant to run multi-GPU NVLINK systems, because the local-copy rendering distribution of sub-frames is always doing the compositing step.
The scene description format has been slightly changed to allow different albedo and/or cutout opacity textures per material reference.

**User Interaction inside the examples**:
* Left Mouse Button + Drag = Orbit (around center of interest)
* Middle Mouse Button + Drag = Pan (The mouse ratio field in the GUI defines how many pixels is one unit.)
* Right Mouse Button + Drag = Dolly (nearest distance limited to center of interest)
* Mouse Wheel = Zoom (1 - 179 degrees field of view possible)
* SPACE  = Toggle GUI display on/off

Additionally in rtigo3:
* S = Saves the current system description settings into a new file (e.g. to save camera positions)
* P = Saves the current tonemapped output buffer to a new PNG file. (Destination folder must exist! Check the `prefixScreenshot` option inside the *system* text files.)
* H = Saves the current linear output buffer to a new HDR file.

# Building

The application framework for all these examples uses GLFW for the window management, GLEW 2.1.0 for the OpenGL functions, DevIL 1.8.0 (optionally 1.7.8) for all image loading and saving, local ImGUI code for the simple GUI, and for *rtigo3*, ASSIMP to load triangle mesh geometry. 
GLEW 2.1.0 is required for *rtigo3* for the UUID matching of devices between OpenGL and CUDA which requires a specific OpenGL extension not supported by GLEW 2.0.0. The intro examples compile with GLEW 2.0.0 though.

The individual applications' CMakeLists.txt files are currently setup for OptiX 7.0.0, but they compile and work with OptiX 7.1.0 as well, if the `find_package(OptiX7 REQUIRED)` is replaced with `find_package(OptiX71 REQUIRED)`

**Windows**

Pre-requisites:
* NVIDIA GPU supported by OptiX 7 (Maxwell GPU or newer, RTX boards highly recommended.)
* Display drivers supporting OptiX 7.0.0 (442.50 and newer recommended) or OptiX 7.1.0 (451.48 and newer).
* Visual Studio 2017 or Visual Studio 2019
* CUDA 10.x Toolkit or with OptiX 7.1.0 optionally CUDA 11.0 Toolkit. (There will be SM 5.0 deprecation warnings with CUDA 11.) 
* OptiX SDK 7.0.0 or 7.1.0. When using 7.1.0 replace the `find_package(OptiX7 REQUIRED)` against `find_package(OptiX71 REQUIRED)`
* CMake 3.10 or newer.

(This looks more complicated than it is. With the pre-requisites installed this is a matter of minutes.)

3rdparty library setup:
* From the Start Menu open the *Windows Command Prompt*
* Create an empty directory to contain your working files. Give it whatever name you want:
	- mkdir `<WORKING_DIRECTORY>`
* Change directory to the working folder:
	- cd `<WORKING_DIRECTORY>`
* Download the sources from the gitlab:
	- git clone https://github.com/mansouriHassan/optix_loreal.git
* Change directory to the folder containing the `sources`:
	- cd optix-hair-7.1
* Execute these commands to setup the 3rdparty library:
	- call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x64
	- 3rdparty.cmd
* Copy the assimp and devil libraries to the 3rdparty directory:
	- Xcopy /E /I /Y .\libs\assimp 3rdparty\assimp
	- Xcopy /E /I /Y .\libs\devil_1_8_0 3rdparty\devil_1_8_0

Generate the solution:
* Create an empty directory for building the project. Give it whatever name you want:
	- mkdir `<BUILDING_DIRECTORY>`
* Execute this command to generate th solution for Visual Studio 2019:
	- cmake -G "Visual Studio 16 2019" -A x64 -S OptixSrc -B `<BUILDING_DIRECTORY>`

Building the examples:
* Change directory to the building folder:
	- cd `<BUILDING_DIRECTORY>`
* Build the solution with the appropriate mode `Debug or Release`:
	- cmake --build . --config `<build_mod>`

* Or use the Visual Studio to build the solution:
	- Open Visual Studio 2017 resp. Visual Studio 2019 and load the solution from your build folder.
	- Select the *Debug* or *Release* *x64* target and pick *Menu* -> *Build* -> *Rebuild Solution*. That builds all projects in the solution in parallel.

Adding the libraries and data (Yes, this could be done automatically but this is required only once.):
* Copy the x64 library DLLs: `cudart64_<toolkit_version>.dll, glew32.dll, DevIL.dll, ILU.dll, ILUT.dll assimp-vc<compiler_version>-mt.dll` into the build folder with the executables (*bin/Release* or *bin/Debug*). (E.g. `cudart64_101.dll` from CUDA Toolkit 10.1 and `assimp-vc141-mt.dll` from the 3rdparty/assimp folder when building with MSVS 2017.)
	- Xcopy /E /I /Y ..\optix-hair-7.1\dll .\bin\\`<build_mod>`
	
* Copy all files from the *data* folder into the build folder with the executables (*bin/Release* or *bin/Debug*).
	- Xcopy /E /I /Y ..\optix-hair-7.1\data .\bin\\`<build_mod>`
# Running

When running the examples from inside the debugger, make sure the working directory points to the folder with the executable because files are searched relative to that. In Visual Studio that is the same as `$(TargetDir)`.

Open a command prompt and change directory to the folder with the executables (same under Linux, just without the .exe suffix.)


The following scene description uses the [Buggy.gltf](https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0/Buggy/glTF) model from Khronos which is not contained inside this source code repository.
The link is also listed inside the `scene_rtigo3_hair_half_head.txt` file.

* Change directory to the bin folder in the building directory:
	- cd /d bin\\`<build_mod>`
* `rtigo3.exe -s system_rtigo3_single_gpu.txt -d scene_rtigo3_hair_half_head.txt`

If you run a multi-GPU system, read the `system_rtigo3_dual_gpu_local.txt` for the modes of operation and interop settings.

* `rtigo3.exe -s system_rtigo3_dual_gpu_local.txt -d scene_rtigo3_hair_half_head.txt`

# Example: building the sources in Debug mode

* git clone https://github.com/mansouriHassan/optix_loreal.git

* cd /d optix_loreal
* call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x64
* 3rdparty.cmd

* Xcopy /E /I /Y .\libs\assimp 3rdparty\assimp
* Xcopy /E /I /Y .\libs\devil_1_8_0 3rdparty\devil_1_8_0

* mkdir ..\OptixBuild
* cmake -G "Visual Studio 16 2019" -A x64 -S . -B ..\OptixBuild
* cd /d ..\OptixBuild
* cmake --build . --config Debug

* Xcopy /E /I /Y ..\optix_loreal\dll .\bin\Debug
* Xcopy /E /I /Y ..\optix_loreal\data .\bin\Debug

* cd /d bin\Debug
* rtigo3.exe -s system_rtigo3_single_gpu.txt -d scene_rtigo3_hair_half_head.txt

# Pull Requests

NVIDIA is happy to review and consider pull requests for merging into the main tree of the optix_apps for bug fixes and features. Before providing a pull request to NVIDIA, please note the following:

* A pull request provided to this repo by a developer constitutes permission from the developer for NVIDIA to merge the provided changes or any NVIDIA modified version of these changes to the repo. NVIDIA may remove or change the code at any time and in any way deemed appropriate.
* Not all pull requests can be or will be accepted. NVIDIA will close pull requests that it does not intend to merge.
* The modified files and any new files must include the unmodified NVIDIA copyright header seen at the top of all shipping files.

# Support

Technical support is available on [NVIDIA's Developer Forum](https://forums.developer.nvidia.com/c/professional-graphics-and-rendering/advanced-graphics/optix/167), or you can create a git issue.
