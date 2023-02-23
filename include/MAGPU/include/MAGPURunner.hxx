#pragma once

#include <MAGPUVector.hxx>

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
						ret = a_in.get_data(0);
						return ret;
					}
					if constexpr (RUNNER_MODE == MEM_MODE::GPU)
					{
						ret = a_in.get_data(1);
						return ret;
					}
				}
			}

		/**
		 * @brief Non-specialized version of the MAGPURunner class, this class is used to launch a functor on the CPU or GPU depending on the gpu parallelization type.
		 *	It's possible to not used MAGPUFunctor and MAGPUVector even if it's designed to use it.	 
		 */
		template<MEM_MODE MODE, GPU_TYPE GT>
			struct MAGPURunner
			{
			};

		/**
		 * @brief partial-specialized version of the MAGPURunner class, this class is used to launch a functor with the memeory mode CPU ONLY.
		 */
		template<GPU_TYPE GT>
			struct MAGPURunner<MEM_MODE::CPU, GT>
			{
				template<typename Functor, typename... Args>
					void host(Functor& a_functor, size_t a_size, Args&&... a_args) const
					{
#ifdef __VERBOSE_LAUNCHER
						std::cout << " HOST: " << a_functor.m_name << std::endl;
#endif

						for(size_t it = 0 ; it < a_size ; it++)
							a_functor(it, a_args...);
					}

				template<typename Functor, typename... Args>
					void operator()(Functor& a_functor, unsigned int a_size, Args&&... a_args)
					{
						host(a_functor, a_size,  get_data<MEM_MODE::CPU>(a_args)...);
					}
			};

		template<>
			struct MAGPURunner<MEM_MODE::GPU, GPU_TYPE::SERIAL>
			{
				template<typename Functor, typename... Args>
					void serial(Functor& a_functor, size_t a_size, Args&&... a_args) const
					{
#ifdef __VERBOSE_LAUNCHER
						std::cout << " SERIAL: " << a_functor.m_name << std::endl;
#endif

						for(size_t it = 0 ; it < a_size ; it++)
							a_functor(it, a_args...);
					}


				template<typename Functor, typename... Args>
					void operator()(Functor& a_functor, unsigned int a_size, Args&&... a_args)
					{
						serial(a_functor, a_size,  get_data<MEM_MODE::GPU>(a_args)...);
					}
			};

#ifdef __CUDA__

		// TODO : manage cudastream
		/*
			 template<typename DATA>
			 cudaStream_t cuda_stream(DATA& a_in)
			 {
			 return cudaStream_t(0);
			 }

			 template<typename DATA, MEM_MODE DATA_MODE>
			 cudaStream_t cuda_stream(MAGPUVector<DATA, DATA_MODE, GPU_TYPE::CUDA>& a_vec)
			 {
			 cudaStream_t steam = a_vec.get_stream();
			 return stream;
			 }

			 template<typename... Args>
			 cudaStream_t cuda_stream()

		 */

		template<typename... Args>
			cudaStream_t get_cuda_stream(Args&& ... a_args)
			{
				return cudaStream_t(0);
			}

		template<>
			struct MAGPURunner<MEM_MODE::GPU, GPU_TYPE::CUDA>
			{

				// CUDA STUFF
				constexpr size_t cudaThreads = 256;

				template<typename Functor, typename... Args>
					__global__
					void cuda(Functor a_functor, size_t a_size, Args... a_args) const
					{
						unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
						if(idx < a_size)
							a_functor(it, a_args...);
					}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
				inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
				{
					if (code != cudaSuccess) 
					{
						fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
						if (abort) exit(code);
					}
				}


				template<typename Functor, typename... Args>
					void operator()(Functor& a_functor, unsigned int a_size, Args&&... a_args)
					{
						auto stream = get_cuda_stream(std::forward<Args>(a_args)...);  // TODO : currently it returns the default stream
						const size_t cudaBlocks = (a_size + cudaThreads - 1) / cudaThreads;
#ifdef __VERBOSE_LAUNCHER
						std::cout << " CUDA: " << a_functor.get_name() << std::endl;
						std::cout << " stats - blocks:" << cudaBlocks << " - threads: " << cudaThreads << " - number of elements: " << a_size << std::endl;
#endif

						cuda<<<<cudaBlocks, cudaThreads, 0, stream>>> (a_functor, a_size,  get_data<MEM_MODE::GPU>(a_args)...);
					}
			};
#endif //__CUDA__
	}
}
