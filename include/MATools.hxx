#pragma once

// features
#include<MATimers/MATimers.hxx>
#include<MATrace/MATrace.hxx>
#include<Common/MAMemory.hxx>

// options
#include<MATrace/MATraceOptional.hxx>
#include<MATimers/MATimerOptional.hxx>

namespace MATools
{
	void initialize();
	void initialize(int*,  char***);
	void finalize();
}

// API
#include <MAToolsAPI/MATimersAPI.hxx>
#include <MAToolsAPI/MAMemoryAPI.hxx>
