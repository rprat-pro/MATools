#pragma once

#include <datatype/MAGPUVector.hxx>
#include <datatype/MAGPUDefineMacros.hxx>

namespace MATools
{
	namespace MAGPU
	{

		/**
		 * @brief Gets data pointer if it's a MAGPUVector, otherwise it returns the object
		 * @param [in] a_in is a parameter used in MAGPURunner::operator()
		 * @return the object -> default case
		 */
		template<MEM_MODE RUNNER_MODE, typename T>
			T&& get_data(T&& a_in)
			{
				return a_in;
			}

		/**
		 * @brief Gets data pointer if it's a MAGPUVector, otherwise it returns the object
		 * @param [in] a_in is a parameter used in MAGPURunner::operator()
		 * @return a pointer -> specific case case
		 */
		template<MEM_MODE RUNNER_MODE, typename T, MEM_MODE MODE, GPU_TYPE GT>
			T* get_data(MAGPUVector<T, MODE, GT>& a_in)
			{
				if(MODE != MEM_MODE::BOTH)
				{
					auto ret = a_in.get_data();
					return ret;
				}
				else
				{
					T* ret;
					if constexpr (RUNNER_MODE == MEM_MODE::CPU)
					{
						ret = a_in.get_data(mem_cpu);
						return ret;
					}
					if constexpr (RUNNER_MODE == MEM_MODE::GPU)
					{
						ret = a_in.get_data(mem_gpu);
						return ret;
					}
				}
			}

		/**
		 * @brief Gets object if the object is not a pointer
		 * @param [in] a_idx not used if the object is not a pointer 
		 * @param [in] a_data is the object returned 
		 * @return the object a_data declared as parameter
		 */
		template<typename T>
			MAGPU_DECORATION
			T&& eval_data(size_t a_idx, T&& a_data)
			{
				return a_data;
			}

		/**
		 * @brief Gets the reference on the idx-ieme object
		 * @param [in] a_idx is the shift on a_data
		 * @param [in] a_data is a pointer
		 * @return the object a_data[a_idx]
		 */
		template<typename T>
			MAGPU_DECORATION
			T& eval_data(size_t a_idx, T* const a_data)
			{
				return a_data[a_idx];
			}

		/**
		 * @brief Function that iterates from 0 to a_size over elements. The functor doesn't know the value of idx (the shift).
		 * @param [in] a_functor is the functor applied for all elements
		 * @param [in] a_size is the number of elements
		 * @param [inout] a_args are the parameters of the functor
		 */
		template<typename Functor, typename... Args>
			void host_launcher_test(const Functor& a_functor, size_t a_size, Args&&... a_args)
			{
				for(size_t idx = 0 ; idx < a_size ; idx++)
					a_functor.launch_test(eval_data(idx, a_args)...);
			}


		/**
		 * @brief Non-specialized version of the MAGPURunner class, this class is used to launch a functor on the CPU or GPU depending on the gpu parallelization type.
		 *	It's possible to not used MAGPUFunctor and MAGPUVector even if it's designed to use it.	 
		 */
		template<MEM_MODE MODE, GPU_TYPE GT>
			struct MAGPURunner
			{
			};
	}
}

#include <host/MAGPURunnerHost.hxx>
#include <serial/MAGPURunnerSerial.hxx>
#include <cuda/MAGPURunnerCuda.hxx>
