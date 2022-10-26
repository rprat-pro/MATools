#include<MATraceInstance.hxx>

namespace MATimer
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
	};
};
