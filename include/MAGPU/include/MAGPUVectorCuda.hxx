#ifdef __CUDA__

#pragma once
#include <cuda.h>

namespace MATools
{
	namespace MAGPU
	{

		namespace CUDA
		{
			__global__
				template<typename T>
				void init(T* a_ptr, std::size_t a_size, const T& a_val)
				{
					unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x; ;
					if(idx < a_size)
						a_ptr[idx] = a_val;
				}

			__global__
				template<typename T>
				void init(T* a_out, std::size_t a_size, T* a_in)
				{
					unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x; ;
					if(idx < a_size)
						a_out[idx] = a_in[idx];
				}
		}

		template<typename T, MEM_MODE MODE>
			class MADeviceMemory<T, MODE, GPU_TYPE::SERIAL>
			{
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
					 * @brief Gets device memory pointer
					 * @return device pointer, this pointer is defined for each specialization
					 */
					T* get_device_data()
					{
						T* ret = m_device.get();
						return ret;
					}


					/**
					 * @brief GPU allocator if MEM_MODE is set to GPU or BOTH
					 * @param[in] a_size size of the storage
					 */
					void gpu_allocator(const std::size_t a_size)
					{
						GPU_WORLD(MODE)
						{
							T* ptr;
							cudaMalloc((void**)&ptr, a_size*sizeof(T));
							m_device = std::shared_ptr<T>(ptr, [](T* a_ptr)->void {cudaFree(ptr)});
						}
					}

					/**
					 * @brief Initializes the gpu memory if MEM_MODE is set to GPU or BOTH
					 * @param[in] a_val is the filling value
					 * @param[in] a_size is the size of the storage
					 */
					void gpu_init(const T& a_val, const std::size_t a_size)
					{
						T* raw_ptr = get_gpu_data();
						auto stream = get_cuda_stream();
						const int block_size = 256;
						const int number_of_blocks = (int)ceil((float)a_size/block_size);
						init<<<number_of_block, block_size, cuda_stream>>>(raw_ptr, a_size, a_val);
					}

					/**
					 * @brief Initializes the gpu memory by copying data if MEM_MODE is set to GPU or BOTH
					 * @param[in] a_ptr contains the filling values
					 * @param[in] a_size is the size of the storage
					 */
					void gpu_init(T* a_ptr, const std::size_t a_size)
					{
						T* raw_ptr = get_gpu_data();
						auto stream = get_cuda_stream();
						const int block_size = 256;
						const int number_of_blocks = (int)ceil((float)a_size/block_size);
						init<<<number_of_block, block_size, cuda_stream>>>(raw_ptr, a_size, a_ptr);
					}
					/**
					 * @brief initialize MAGPUVector with a device pointer.
					 * @param[in] a_ptr device pointer on the data storage
					 * @param[in] a_size is the data size
					 */
					void gpu_aliasing(T* a_ptr, unsigned int a_size)
					{
						m_device = std::shared_ptr<T>(a_ptr, [](T* a_in) {cudaFree(a_ptr);});
					}

					/**
					 * @brief Gets device memory pointer
					 * @return device pointer, this pointer is defined for each specialization
					 */
					T* get_device_data()
					{
						GPU_WORLD(MODE)
						{
							T* ret = m_device.get();
						}
						else
						{
							assert(0 != 1 && "MATools_LOG: Wrong use of get_device_data(), MEM_MODE = CPU, Release mode returns a nullptr");
							return nullptr;
						}
					}

					void sync()
					{
						GPU_WORLD(MODE)
						{
							cudaDeviceSynchronize();
						}
					}

					void host_sync()
					{
						BOTH_WROLD(MODE)
						{
							sync();
							device_to_host();
						}

					}

					void host_to_device()
					{
						BOTH_WORLD()
						{
							T* host = get_host_data();
							T* device = get_device_data();
							auto stream =  get_cuda_stream();
							auto size = get_size();

							if(stream != cudaStream_t(0));
							{ 
								cudaMemcpyAsync(device, host, size*sizeof(T), cudaMemcpyHostToDevice, stream);
							}
							else
							{ 
								cudaMemcpy(device, host, size*sizeof(T), cudaMemcpyHostToDevice);
							}
						}
					}

					void gpu_sync()
					{
						BOTH_WROLD(MODE)
						{
							host_to_device();
						}
					}

					void host_to_device()
					{
						BOTH_WORLD()
						{
							T* host = get_host_data();
							T* device = get_device_data();
							auto stream =  get_cuda_stream();
							auto size = get_size();

							if(stream != cudaStream_t(0));
							{ 
								cudaMemcpyAsync(device, host, size*sizeof(T), cudaMemcpyHostToDevice, stream);
							}
							else
							{ 
								cudaMemcpy(device, host, size*sizeof(T), cudaMemcpyHostToDevice);
							}
						}

					}

					void device_to_host()
					{
						BOTH_WORLD()
						{
							T* host = get_host_data();
							T* device = get_device_data();
							auto stream =  get_cuda_stream();
							auto size = get_size();

							if(stream != cudaStream_t(0));
							{
								cudaMemcpyAsync(host, device, size*sizeof(T), cudaMemcpyDeviceToHost, stream);
							}
							else
							{
								cudaMemcpy(host, device, size*sizeof(T), cudaMemcpyDeviceToHost);
							}
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
						GPU_WORLD(MODE)
						{
							m_size = a_size;
						}
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
