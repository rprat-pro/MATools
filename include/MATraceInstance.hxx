#pragma once

#include <MATraceTypes.hxx>

namespace MATools
{
	namespace MATrace
	{
		extern MATrace_point& get_ref_MATrace_point();
		extern MATrace_point& get_MATrace_point();
		extern Trace& get_local_MATrace();
	};
}
