#pragma once

#include <cassert>
#include <algorithm>

namespace MATools
{
  namespace MAGPU
  {
    /** 
     * @brief template specialization of the copier for serial
     */
    template<>
      class MAGPUCopier<GPU_TYPE::SERIAL>
      {
	public :
	  /**
	   * @brief Copies the data from the gpu to the cpu
	   * @param [inout] a_dst is the cpu memory
	   * @param [in] a_src is the gpu memory
	   * @param [in] a_size is the number of elements (>0)
	   */
	  template<typename T>
	    void copy_to_host(T* const a_dst, const T* const a_src, std::size_t a_size)
	    {
	      assert(a_dst != nullptr);
	      assert(a_src != nullptr);
	      assert(a_size != 0);
	      std::copy(a_src, a_src + a_size, a_dst);
	    }

	  /**
	   * @brief Copies the data from the cpu to the gpu
	   * @param [inout] a_dst is the gpu memory
	   * @param [in] a_src is the cpu memory
	   * @param [in] a_size is the number of elements (>0)
	   */
	  template<typename T>
	    void copy_to_device(T* const a_dst, const T* const a_src, std::size_t a_size)
	    {
	      assert(a_dst != nullptr);
	      assert(a_src != nullptr);
	      assert(a_size != 0);
	      std::copy(a_src, a_src + a_size, a_dst);
	    }
      };
  }
}
