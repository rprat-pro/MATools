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
			class MADeviceMemory<T, GPU_TYPE::CUDA> : public MAGPUAllocator<T, GPU_TYPE::CUDA>, public MAGPUCopier<GPU_TYPE::CUDA>
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
#ifdef __VERBOSE_MAGPU
						std::cout << " copy_to_vector_from_device -> size : "<< size << std::endl;
#endif
						ret.resize(size);
						device_to_host(ret.data());
						return ret;
					}

				protected:
					/** 
					 * @brief default constructor
					 */
					MADeviceMemory() : m_device_size(0), m_device(nullptr), m_stream(cudaStream_t(0)) {}

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
#ifdef __VERBOSE_MAGPU
						std::cout << " cuda malloc, size :  " <<  a_size*sizeof(T) << std::endl;
#endif
						ptr = this->allocate(a_size);
						auto deleter = MAGPUAllocator<T, GPU_TYPE::CUDA>();
						m_device = std::shared_ptr<T>(ptr, [&deleter](T* const a_ptr){ deleter.destroy(a_ptr); 
#ifdef __VERBOSE_MAGPU
								std::cout << " destructor " << std::endl;
#endif
								});
#ifdef __VERBOSE_MAGPU
						std::cout << " ptr :  " <<  ptr << std::endl;
#endif
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
#ifdef __VERBOSE_MAGPU
						std::cout << " raw_ptr :  " <<  raw_ptr << std::endl;
#endif
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

#ifdef __VERBOSE_MAGPU
						std::cout << " gpu_aliasing of size " << a_size << std::endl; 
#endif
						m_device = std::shared_ptr<T>(a_ptr, [](T* a_in) {});
						set_device_size(a_size);
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

					void gpu_resize(unsigned int a_size)
					{
						if(a_size == 0) 
						{
							set_device_size(0);
							m_device = nullptr;
						}
						else if(m_device_size > a_size) /* */
						{
							/* Warning, this vector has been reduced */
							set_device_size(a_size);
						}
						else if(a_size > m_device_size)
						{ /* */
							if(m_device_size == 0) // scenario : only host memory has been defined with an extern storage
							{
								gpu_allocator(a_size);
							}
							else
							{
								const unsigned int old_size = get_device_size();
								std::shared_ptr<T> data = m_device;
								gpu_allocator(a_size);
								cudaMemcpy(m_device.get(), data.get(), old_size * sizeof(double),cudaMemcpyDeviceToDevice);	
							}
						}
					}

					void host_to_device(T* a_host, unsigned int a_host_size)
					{
#ifdef __VERBOSE_MAGPU
						std::cout << "host_to_device  -> host size = " << a_host_size << std::endl;
						std::cout << "resizing if needed -- " << a_host_size << " " << get_device_size() << std::endl;
						
#endif
						gpu_resize(a_host_size);
						auto size = get_device_size(); /* equal to a_host_size */
						T* device = get_device_data();
						auto stream =  get_cuda_stream();
						assert(a_host_size !=0);
						std::cout << device << " " << a_host << std::endl;
						if(stream != cudaStream_t(0))
						{ 
							cudaMemcpyAsync(device, a_host, size*sizeof(T), cudaMemcpyHostToDevice, stream);
						}
						else
						{	
							// MAGPUCopier
							// T, Cuda
						//	copy_to_device(device, a_host, size);		
						}
					}

					void device_to_host(T* a_host)
					{
#ifdef __VERBOSE_MAGPU
						std::cout << "device to host  -> device size = " << get_device_size() << std::endl;
#endif
						T* device = get_device_data();
						auto stream =  get_cuda_stream();
						auto size = get_device_size();
						assert(size != 0);
						if(stream != cudaStream_t(0))
						{
							cudaMemcpyAsync(a_host, device, size*sizeof(T), cudaMemcpyDeviceToHost, stream);
						}
						else
						{
							// MAGPUCopier
							// T, Cuda
							this->copy_to_host(a_host, device, size);
						}
					}

					/**
					 * @brief Gets size
					 * @return m_device_size member
					 */
					unsigned int get_device_size()
					{
						unsigned int ret = m_device_size;
						return ret;
					}

					/**
					 * @brief Sets size
					 * @param new value of m_device_size
					 */
					void set_device_size(unsigned int a_size)
					{
#ifdef __VERBOSE_MAGPU
						std::cout << " set_device_size = " << a_size << std ::endl;
#endif
						m_device_size = a_size;
					}

				private:
					/** @brief device data */
					std::shared_ptr<T> m_device;

					/** @brief device data size */
					unsigned int m_device_size = 0;

					/** @brief possibility to use a cuda stream */
					cudaStream_t m_stream = cudaStream_t(0); // default stream
			};
	} // MAGPU
} // MATools


#endif // __CUDA__
