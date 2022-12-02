#pragma once

// features
#include<MATimers.hxx>
#include<MATrace.hxx>
#include<MAMemory.hxx>

// options
#include<MATraceOptional.hxx>
#include<MATimerOptional.hxx>

namespace MATools
{
	void initialize();
	void initialize(int*,  char***, bool = true);
	void finalize(bool = true);
}
