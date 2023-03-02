#pragma once

#include <MAGPUVector.hxx>
#include <MAGPUDefineMacros.hxx>

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

    /**
     * @brief partial-specialized version of the MAGPURunner class, this class is used to launch a functor with the memeory mode CPU ONLY.
     */
    template<GPU_TYPE GT>
      struct MAGPURunner<MEM_MODE::CPU, GT>
      {
	template<typename Functor, typename... Args>
	  void host(Functor& a_functor, size_t a_size, Args&&... a_args) const
	  {
#ifdef __VERBOSE_MAGPU
	    std::cout << " HOST: " << a_functor.m_name << std::endl;
#endif

	    for(size_t it = 0 ; it < a_size ; it++)
	      a_functor(it, a_args...);
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
	    host(a_functor, a_size,  get_data<MEM_MODE::CPU>(a_args)...);
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
	    host_launcher_test(a_functor, a_size,  get_data<MEM_MODE::CPU>(a_args)...);
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


	/**
	 * @brief This operator is a FORALL function that iterates over the elements of vector(s) (a_args). 
	 * @param [in] a_functor is the functor applied for all elements
	 * @param [in] a_size is the number of elements
	 * @param [inout] a_args are the parameters of the functor
	 */
	template<typename Functor, typename... Args>
	  void operator()(Functor& a_functor, unsigned int a_size, Args&&... a_args)
	  {
	    serial(a_functor, a_size,  get_data<MEM_MODE::GPU>(a_args)...);
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
	    host_launcher_test(a_functor, a_size,  get_data<GPU>(a_args)...); 
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
#endif //__CUDA__
  }
}
