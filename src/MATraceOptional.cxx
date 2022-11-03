#include <MATraceOptional.hxx>
#include <MAOutput.hxx>
#include <MATrace.hxx>

namespace MATools
{
	namespace MATrace
	{
		namespace Optional
		{
			extern bool& get_MATrace_mode()
			{
				static bool _ftm = false;
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
				static bool _ftm = false;
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
