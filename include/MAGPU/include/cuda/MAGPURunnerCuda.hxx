#pragma once

#ifdef __CUDA__

namespace MATools
{
	namespace MAGPU
	{
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

		// CUDA STUFF HERE

		template<typename... Args>
			cudaStream_t get_cuda_stream(Args&& ... a_args)
			{
				return cudaStream_t(0);
			}

		template<typename Functor, typename... Args>
			__global__
			void cuda_launcher(Functor a_functor, size_t a_size, Args... a_args)
			{
				unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
				if(idx < a_size)
					a_functor(idx, a_args...);
			}


		template<typename Functor, typename... Args>
			__global__
			void cuda_launcher_test(Functor a_functor, size_t a_size, Args... a_args)
			{
				unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
				if(idx < a_size)
					a_functor.launch_test(eval_data(idx, a_args)...);
			}

		template<>
			struct MAGPURunner<MEM_MODE::GPU, GPU_TYPE::CUDA>
			{
				// CUDA STUFF
				const size_t cudaThreads = 256;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
				inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
				{
					if (code != cudaSuccess) 
					{
						fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
						if (abort) exit(code);
					}
				}

				/**
				 * @brief This operator is a FORALL function that iterates over the elements of vector(s) (a_args). 
				 * @param [in] a_functor is the functor applied for all elements
				 * @param [in] a_size is the number of elements
				 * @param [inout] a_args are the parameters of the functor
				 */
				template<typename Functor, typename... Args>
					void operator()(Functor& a_functor, unsigned int a_size, Args&&... a_args)
					{
						// cuda stuff
						auto stream = get_cuda_stream(std::forward<Args>(a_args)...);  // TODO : currently it returns the default stream
						const size_t cudaBlocks = (a_size + cudaThreads - 1) / cudaThreads;
#ifdef __VERBOSE_MAGPU
						std::cout << " CUDA: " << a_functor.get_name() << std::endl;
						std::cout << " stats - blocks:" << cudaBlocks << " - threads: " << cudaThreads << " - number of elements: " << a_size << std::endl;
#endif
						//launcher
						cuda_launcher<<<cudaBlocks, cudaThreads, 0, stream>>> (a_functor, a_size, get_data<MEM_MODE::GPU>(a_args)...);
					}

				/**
				 * @brief test launcher is a FORALL function that iterates over the elements of vectors. The functor doesn't know the value of idx (the shift).
				 * @param [in] a_functor is the functor applied for all elements
				 * @param [in] a_size is the number of elements
				 * @param [inout] a_args are the parameters of the functor
				 */
				template<typename Functor, typename... Args>
					void launcher_test(Functor& a_functor, unsigned int a_size, Args&&... a_args)
					{
						// cuda stuff
						auto stream = get_cuda_stream(std::forward<Args>(a_args)...);  // TODO : currently it returns the default stream
						const size_t cudaBlocks = (a_size + cudaThreads - 1) / cudaThreads;
#ifdef __VERBOSE_MAGPU
						std::cout << " CUDA: " << a_functor.get_name() << std::endl;
						std::cout << " stats - blocks:" << cudaBlocks << " - threads: " << cudaThreads << " - number of elements: " << a_size << std::endl;
#endif

#ifdef __VERBOSE_MAGPU
						cudaEvent_t cstart;
						cudaEvent_t cstop;
						cudaEventCreate(&cstart);
						cudaEventCreate(&cstop);
						cudaEventRecord(cstart);
#endif
						//launcher
						cuda_launcher_test<<<cudaBlocks, cudaThreads, 0, stream>>> (a_functor, a_size, get_data<MEM_MODE::GPU>(a_args)...);

#ifdef __VERBOSE_MAGPU
						cudaEventRecord(cstop);
						cudaDeviceSynchronize();
						cudaEventSynchronize(cstop);
						float milliseconds = 0;
						cudaEventElapsedTime(&milliseconds, cstart, cstop);
						auto seconds = milliseconds*0.001;
						gpuErrchk( cudaPeekAtLastError());
						std::cout << " " << a_functor.get_name() << "("<< a_size << ") " << ": " << a_size / seconds << " elements/s "<< std::endl;
#endif
					}
			};
	}
}
#endif //__CUDA__
