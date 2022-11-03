#pragma once

namespace MATools
{
	namespace MATrace
	{
		namespace Optional
		{
			extern bool& get_MATrace_mode();
			extern bool& get_omp_mode();
			
			void active_MATrace_mode();
			void active_omp_mode();
			
			bool is_MATrace_mode();
			bool is_omp_mode();
		};
	};
};
