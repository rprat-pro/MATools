#pragma once

#include <cstddef>
#include <cstdlib>
#include <datatype/MAGPUTypes.hxx>

// MAGPUAllocator declaration
namespace MATools
{
	namespace MAGPU
	{
		/**
		 * @brief non-specialized declaration of MAGPUAllocator, this class has to be specialized for every gpu parallelization type.
		 */
		template<typename T, GPU_TYPE GT>
			class MAGPUAllocator
			{
				T* allocate(std::size_t a_size)
				{
					// should not be used
					std::abort();
				}
			};
	}
}

#include <serial/MAGPUAllocatorSerial.hxx>
//#ifdef __CUDA__
#include <cuda/MAGPUAllocatorCuda.hxx>
//#endif
