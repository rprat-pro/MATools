#pragma once 

namespace MATools
{
  namespace MAGPU
  {
    /**
     * @brief partial template specialization of the MAGPUAlloctor class with cuda
     */
    template<typename T>
      class MAGPUAllocator<T, GPU_TYPE::CUDA>
      {
	public :
	  /**
	   * @brief Allocates the memory on the gpu and returns the pointer
	   * @param [in] a_size is the number of T element
	   * @return pointer on the allocated memory
	   */
	  T* allocate(std::size_t a_size)
	  {
	    T* ret;
	    cudaMalloc((void**)&ret, a_size * sizeof(a_size));
	    return ret;
	  }

	  /**
	   * @brief Frees the memory
	   * @param [in] ptr is the pointer on the freed memory
	   */
	  void destroy(T* const a_ptr)
	  {
	    cudaFree(a_ptr);
	  }
      };
  }
}
