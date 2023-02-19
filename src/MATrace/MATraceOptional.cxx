#include <MATrace/MATraceOptional.hxx>
#include <MAOutput/MAOutput.hxx>
#include <MATrace/MATrace.hxx>

namespace MATools
{
	namespace MATrace
	{
		namespace Optional
		{
			// define default mode values
			constexpr bool MATrace_default_mode = false;
			constexpr bool omp_default_mode = false;

			extern bool& get_MATrace_mode()
			{
				static bool _ftm = MATrace_default_mode;
				return _ftm;
			}

			void active_MATrace_mode()
			{
				bool& mode = get_MATrace_mode();
				mode = true;
				MATools::MAOutput::printMessage("MATrace_LOG: MATrace is activated");
			}

			bool is_MATrace_mode()
			{
				bool ret = get_MATrace_mode();
				return ret;
			}

			extern bool& get_omp_mode()
			{
				static bool _ftm = omp_default_mode;
				return _ftm;
			}

			void active_omp_mode()
			{
				bool& mode = get_omp_mode();
				bool use_MATrace = get_MATrace_mode();
				mode = true && use_MATrace;
				MATools::MAOutput::printMessage("MATrace_LOG: omp mode is activated");
				init_omp_trace();
			}

			bool is_omp_mode()
			{
				bool ret = get_omp_mode();
				return ret;
			}
		};
	};
};
