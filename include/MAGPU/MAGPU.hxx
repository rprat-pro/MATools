#pragma once

namespace MATools
{
	namespace MAGPU
	{
		/**
		 * @brief define GPU type
		 */
		enum GPU_TYPE
		{
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
		}
	}
}
