#pragma once

namespace MATools
{
  namespace MAGPU
  {
    /** 
     * @brief template specialization of the copier for cuda
     */
    template<>
      class MAGPUCopier<GPU_TYPE::CUDA>
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
	      unsigned int nbytes = a_size * sizeof(T);
	      auto type = cudaMemcpyDeviceToHost;
	      cudaMemcpy(a_dst, a_src, nbytes, type);
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
	      unsigned int nbytes = a_size * sizeof(T);
	      auto type = cudaMemcpyHostToDevice;
	      cudaMemcpy(a_dst, a_src, nbytes, type);
	    }
      };
  }
}
