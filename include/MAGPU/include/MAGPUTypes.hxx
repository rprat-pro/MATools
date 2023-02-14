#pragma once

namespace MATools
{
	namespace MAGPU
	{

		/** @brief bunch of macros */
#define CPU_WORLD(X) if constexpr (X != MEM_MODE::GPU)
#define GPU_WORLD(X) if constexpr (X != MEM_MODE::CPU)
#define BOTH_WORLD(X) if constexpr (X == MEM_MODE::BOTH)

		/**
		 * @brief define GPU type
		 */
		enum GPU_TYPE
		{
			SERIAL, //< no gpu
			CUDA,
			SYCL,
			KOKKOS
		};

		/**
		 * @brief define GPU mode
		 */
		enum MEM_MODE
		{
			CPU,
			GPU,
			BOTH
		};
	}
}
