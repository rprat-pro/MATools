#ifdef __CUDA__

#pragma once
#include <cuda.h>

namespace MATools
{
  namespace MAGPU
  {

    namespace CUDA_KERNEL
    {
      template<typename T>
	__global__
	void init(T* const a_ptr, std::size_t a_size, const T a_val)
	{
	  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x; 
	  if(idx < a_size)
	    a_ptr[idx] = a_val;
	}

      template<typename T>
	__global__
	void init(T* const a_out, std::size_t a_size, T* a_in)
	{
	  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	  if(idx < a_size)
	    a_out[idx] = a_in[idx];
	}
    }

    template<typename T>
      class MADeviceMemory<T, GPU_TYPE::CUDA>
      {
	public:

	  void gpu_sync()
	  {
	    cudaDeviceSynchronize();
	  }

	  std::vector<T> copy_to_vector_from_device()
	  {
	    std::vector<T> ret;
	    unsigned int size = get_device_size();
	    std::cout << " size : "<< size << std::endl;
	    ret.resize(size);
	    device_to_host(ret.data());
	    return ret;
	  }

	protected:

	  /**
	   * @brief Gets cuda stream
	   * @return the cuda stream associated to this vector, by default, it's the default stream.
	   */
	  cudaStream_t get_cuda_stream()
	  {
	    cudaStream_t ret = m_stream;
	    return ret;
	  }

	  /**
	   * @brief GPU allocator if MEM_MODE is set to GPU or BOTH
	   * @param[in] a_size size of the storage
	   */
	  void gpu_allocator(const std::size_t a_size)
	  {
	    T* ptr = nullptr;
	    std::cout << " cuda malloc, size :  " <<  a_size*sizeof(T) << std::endl;
	    cudaMalloc(&ptr, a_size*sizeof(T));
	    m_device = std::shared_ptr<T>(ptr, [](T* a_ptr){cudaFree(a_ptr); std::cout << " destructor " << std::endl;});
	    std::cout << " ptr :  " <<  ptr << std::endl;
	    set_device_size(a_size);
	  }

	  /**
	   * @brief Initializes the gpu memory if MEM_MODE is set to GPU or BOTH
	   * @param[in] a_val is the filling value
	   * @param[in] a_size is the size of the storage
	   */
	  void gpu_init(const T& a_val, const std::size_t a_size)
	  {
	    T* raw_ptr = get_device_data();
	    std::cout << " raw_ptr :  " <<  raw_ptr << std::endl;
	    auto stream = get_cuda_stream();
	    const int block_size = 256;
	    const int number_of_blocks = (int)ceil((float)a_size/block_size);
	    CUDA_KERNEL::init<<<number_of_blocks, block_size>>>(raw_ptr, a_size, a_val);
	  }

	  /**
	   * @brief Initializes the gpu memory by copying data if MEM_MODE is set to GPU or BOTH
	   * @param[in] a_ptr contains the filling values
	   * @param[in] a_size is the size of the storage
	   */
	  void gpu_init(T* a_ptr, const std::size_t a_size)
	  {
	    T* raw_ptr = get_device_data();
	    auto stream = get_cuda_stream();
	    const int block_size = 256;
	    const int number_of_blocks = (int)ceil((float)a_size/block_size);
	    CUDA_KERNEL::init<<<number_of_blocks, block_size, 0, stream>>>(raw_ptr, a_size, a_ptr);
	  }

	  /**
	   * @brief Fills the gpu memory if MEM_MODE is set to GPU or BOTH
	   * @param[in] a_val is the filling value
	   */
	  void gpu_fill(const T& a_val)
	  {
	    T* raw_ptr = get_device_data();
	    auto stream = get_cuda_stream();
	    const int size = get_device_size();
	    const int block_size = 256;
	    const int number_of_blocks = (int)ceil((float)size/block_size);
	    CUDA_KERNEL::init<<<number_of_blocks, block_size, 0, stream>>>(raw_ptr, size, a_val);
	  }

	  /**
	   * @brief initialize MAGPUVector with a device pointer.
	   * @param[in] a_ptr device pointer on the data storage
	   * @param[in] a_size is the data size
	   */
	  void gpu_aliasing(T* a_ptr, unsigned int a_size)
	  {
	    m_device = std::shared_ptr<T>(a_ptr, [](T* a_in) {cudaFree(a_in);});
	  }

	  /**
	   * @brief Gets device memory pointer
	   * @return device pointer, this pointer is defined for each specialization
	   */
	  T* get_device_data()
	  {
	    T* ret = m_device.get();
	    return ret;
	  }

	  void host_to_device(T* a_host)
	  {
	    T* device = get_device_data();
	    auto stream =  get_cuda_stream();
	    auto size = get_device_size();

	    if(stream != cudaStream_t(0))
	    { 
	      cudaMemcpyAsync(device, a_host, size*sizeof(T), cudaMemcpyHostToDevice, stream);
	    }
	    else
	    { 
	      cudaMemcpy(device, a_host, size*sizeof(T), cudaMemcpyHostToDevice);
	    }
	  }

	  void device_to_host(T* a_host)
	  {
	    T* device = get_device_data();
	    auto stream =  get_cuda_stream();
	    auto size = get_device_size();

	    if(stream != cudaStream_t(0))
	    {
	      cudaMemcpyAsync(a_host, device, size*sizeof(T), cudaMemcpyDeviceToHost, stream);
	    }
	    else
	    {
	      cudaMemcpy(a_host, device, size*sizeof(T), cudaMemcpyDeviceToHost);
	    }
	  }

	  /**
	   * @brief Gets size
	   * @return m_size member
	   */
	  unsigned int get_device_size()
	  {
	    unsigned int ret = m_size;
	    return ret;
	  }

	  /**
	   * @brief Sets size
	   * @param new value of m_size
	   */
	  void set_device_size(unsigned int a_size)
	  {
	    m_size = a_size;
	  }

	private:
	  /** @brief device data */
	  std::shared_ptr<T> m_device;

	  /** @brief device data size */
	  unsigned int m_size = 0;

	  /** @brief possibility to use a cuda stream */
	  cudaStream_t m_stream = cudaStream_t(0); // default stream
      };
  } // MAGPU
} // MATools


#endif // __CUDA__
