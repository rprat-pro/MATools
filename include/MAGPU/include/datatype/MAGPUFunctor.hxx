#pragma once

namespace MATools
{
	namespace MAGPU
	{
		template<typename Func, GPU_TYPE GT>
			struct MAGPUFunctor
			{
				template<typename... Args>
					void operator()(Args&&... a_args);
			};
	};
}

#include <serial/MAGPUFunctorSerial.hxx>
#include <cuda/MAGPUFunctorCuda.hxx>
