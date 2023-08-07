#pragma once

namespace MATools
{
	namespace MATrace
	{
		struct MATraceRGB{double r; double g; double b;	};
		constexpr int default_color_size();
		MATraceRGB get_idle_color();
		MATraceRGB get_default_color(int color_id);
	}
}
