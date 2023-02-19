#include <MATrace/MATraceInstance.hxx>

namespace MATools
{
	namespace MATrace
	{
		extern MATrace_point& get_ref_MATrace_point()
		{
			static MATrace_point instance;
			return instance;
		}

		extern MATrace_point& get_MATrace_point()
		{
			static MATrace_point instance;
			return instance;
		}
		
		extern Trace& get_local_MATrace()
		{
			static Trace instance;
			return instance;
		}

		extern std::vector<MATrace_point>& get_MATrace_omp_point()
		{
			static std::vector<MATrace_point> instance;
			return instance;
		}
		
		extern std::vector<Trace>& get_omp_MATrace()
		{
			static std::vector<Trace> instance;
			return instance;
		}
	};
};
