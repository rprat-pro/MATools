#pragma once

#include <MATrace/MATraceTypes.hxx>

namespace MATools
{
	namespace MATrace
	{
		extern MATrace_point& get_ref_MATrace_point();
		extern MATrace_point& get_MATrace_point();
		extern Trace& get_local_MATrace();
		extern std::vector<MATrace_point>& get_MATrace_omp_point();
		extern std::vector<Trace>& get_omp_MATrace();
	};
}
