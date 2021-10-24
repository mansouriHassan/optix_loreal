#pragma once


#ifndef USERGUI_H
#define USERGUI_H

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



class userGUI
{
	
	
public :	
	userGUI();
	void userWindow(bool* p_open = NULL);

private :
	bool     m_isVisibleGUI;
	std::vector<MaterialGUI>      m_materialsGUI;


};



#endif
