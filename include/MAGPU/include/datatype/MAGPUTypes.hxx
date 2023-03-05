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

		/* some renames */
		constexpr auto mem_cpu = MATools::MAGPU::MEM_MODE::CPU;
		constexpr auto mem_gpu = MATools::MAGPU::MEM_MODE::GPU;
		constexpr auto mem_cpu_and_gpu = MATools::MAGPU::MEM_MODE::BOTH;
	}
}
