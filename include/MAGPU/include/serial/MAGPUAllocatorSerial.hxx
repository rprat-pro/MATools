#pragma once 
#include<cassert>

namespace MATools
{
  namespace MAGPU
  {
    /**
     * @brief partial template specialization of the MAGPUAlloctor class with cuda
     */
    template<typename T>
      class MAGPUAllocator<T, GPU_TYPE::SERIAL>
      {
	public :
	  /**
	   * @brief Allocates the memory on the gpu and returns the pointer
	   * @param [in] a_size is the number of T element
	   * @return pointer on the allocated memory
	   */
	  T* allocate(std::size_t a_size)
	  {
	    T* ret = new T[a_size];
	    return ret;
	  }

	  /**
	   * @brief Frees the memory
	   * @param [in] ptr is the pointer on the freed memory
	   */
	  void destroy(T* const a_ptr)
	  {
	    assert(a_ptr != nullptr);
	    delete a_ptr;
	  }
      };
  }
}
