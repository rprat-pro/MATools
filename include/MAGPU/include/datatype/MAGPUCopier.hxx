#pragma once
#include <datatype/MAGPUTypes.hxx>

// MAGPUCopier declaration
namespace MATools
{
	namespace MAGPU
	{
		/**
		 * @brief non-specialized declaration of MAGPUCopier, this class has to be specialized for every gpu parallelization type.
		 */
		template<GPU_TYPE GT>
			class MAGPUCopier
			{
				MAGPUCopier() {}
				//empty class
			};
	}
}

#include <serial/MAGPUCopierSerial.hxx>
#ifdef __CUDA__
#include <cuda/MAGPUCopierCuda.hxx>
#endif
