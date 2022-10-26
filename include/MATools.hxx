#pragma once

#include<MATimers.hxx>
#include<MATrace.hxx>

namespace MATools
{
	void initialize();
	void initialize(int*,  char***, bool = true);
	void finalize(bool = true, bool = true, bool = true);
}
