#pragma once

#include <vector>
#include <memory>
#include <cassert>
#include <iostream>

namespace MATools
{
	namespace MAGPU
	{

		/** @brief bunch of macros */
#define CPU_WORLD(X) if constexpr (X != MEM_MODE::GPU)
#define GPU_WORLD(X) if constexpr (X != MEM_MODE::CPU)
#define BOTH_WORLD(X) if constexpr (X == MEM_MODE::BOTH)

		/**
		 * @brief define GPU type
		 */
		enum GPU_TYPE
		{
			SERIAL, //< no gpu
			CUDA,
			SYCL,
			KOKKOS
		};

		/**
		 * @brief define GPU mode
		 */
		enum MEM_MODE
		{
			CPU,
			GPU,
			BOTH
		};

		/**
		 * @brief MAVector is a class with a GPU storage and a CPU Storage depending on the memory mode.  
		 */		
		template<typename T, MEM_MODE MODE, GPU_TYPE GTYPE>
			class MAGPUVector
			{
				public:

					/**
					 * @brief Host allocator if MEM_MODE is set to CPU or BOTH
					 * @param[in] a_size size of the storage
					 */
					void host_allocator(const std::size_t a_size)
					{
						CPU_WORLD(MODE)
						{
							m_host = std::shared_ptr<T>(new T[a_size], std::default_delete<T>());
							set_size(a_size);
						}
					}

					/**
					 * @brief Initializes the host memory if MEM_MODE is set to CPU or BOTH
					 * @param[in] a_val is the filling value
					 * @param[in] a_size is the size of the storage
					 */
					void host_init(const T& a_val, const std::size_t a_size)
					{
						CPU_WORLD(MODE)
						{
							T* host_ptr = get_host_data();
							for(std::size_t id = 0 ; id < a_size ; id++)
							{
								host_ptr[id] = a_val;
							}
						}
					}

					/**
					 * @brief Initializes the host memory by copying data if MEM_MODE is set to CPU or BOTH
					 * @param[in] a_ptr contains the filling values
					 * @param[in] a_size is the size of the storage
					 */
					void host_init(T* a_ptr, const std::size_t a_size)
					{
						CPU_WORLD(MODE)
						{
							for(int id = 0 ; id < a_size ; id++)
							{
								m_host[id] = a_ptr[id];
							}
						}
					}

					/**
					 * @brief GPU allocator if MEM_MODE is set to GPU or BOTH
					 * @param[in] a_size size of the storage
					 */
					virtual void gpu_allocator(const std::size_t a_size) = 0;

					/**
					 * @brief Initializes the gpu memory if MEM_MODE is set to GPU or BOTH
					 * @param[in] a_val is the filling value
					 * @param[in] a_size is the size of the storage
					 */
					virtual void gpu_init(const T& a_val, const std::size_t a_size) = 0;

					/**
					 * @brief Initializes the gpu memory by copying data if MEM_MODE is set to GPU or BOTH
					 * @param[in] a_ptr contains the filling values
					 * @param[in] a_size is the size of the storage
					 */
					virtual void gpu_init(T* a_ptr, const std::size_t a_size) = 0;

					/**
					 * @brief Initializes the MAGPUVector memory and fill it with the a_val value
					 * @param[in] a_val is the filling value
					 * @param[in] a_size is the size of the storage
					 */
					void init(const T& a_val, const std::size_t a_size)
					{
						host_allocator(a_size);
						host_init(a_val, a_size);
						gpu_allocator(a_size);
						gpu_init(a_val, a_size);
					}

					/**
					 * @brief Initializes the MAGPUVector memory and fill it with the values in a_ptr
					 * @param[in] a_ptr is the pointer of the filling values
					 * @param[in] a_size is the size of the storage
					 */
					void init(T* a_ptr, const std::size_t a_size)
					{
						// We assume that the default way to fill BOTH MEM_MODE is to use an host pointer and to copy the data to the device
						CPU_WORLD(MODE)
						{
							host_allocator(a_size);
							host_init(a_ptr, a_size);
							gpu_sync(); // BOTH MODE, do nothing else
							return;
						}

						GPU_WORLD(MODE)
						{
							gpu_allocator(a_size);
							gpu_init(a_ptr, a_size);
						}
					}

					/**
					 * @brief Initializes the MAGPUVector memory for BOTH mode with two vectors. Both pointers should have the same values.
					 * @param[in] a_host_ptr is the host pointer
					 * @param[in] a_gpu_ptr is the host pointer
					 * @param[in] a_size is the size of the storage
					 */
					void init(T* a_host_ptr, T* a_device_ptr, const std::size_t a_size)
					{
						assert( MODE==MEM_MODE::BOTH && "Only possible if MEM_MODE == BOTH");
						BOTH_WORLD(MODE)				
						{		
							host_allocator(a_size);
							host_init(a_host_ptr, a_size);
							gpu_allocator(a_size);
							gpu_init(a_device_ptr, a_size);
						}
						else
						{
							std::cout << "MATools_Error: wrong usage of MAGPUVector::init(...)" << std::endl;;
							std::abort();
						}
					}

					/**
					 * @brief Gets device memory pointer
					 * @return device pointer, this pointer is defined for each specialization
					 */
					virtual T* get_device_data() = 0;

					/**
					 * @brief Gets host memory pointer
					 * @return device pointer, this pointer is defined for each specialization
					 */
					T* get_host_data()
					{
						T* ret = m_host.get();
						return ret;
					}
					virtual void host_sync() = 0;
					virtual void gpu_sync() = 0;
					virtual void host_to_device() = 0;
					virtual void device_to_host() = 0;


					/**
					 * @brief Gets size
					 * @return m_size member
					 */
					unsigned int get_size()
					{
						unsigned int ret = m_size;
						return ret;
					}

					/**
					 * @brief Sets size
					 * @param new value of m_size
					 */
					void set_size(unsigned int a_size)
					{
						m_size = a_size;
					}

					/**
					 * @brief initialize MAGPUVector with a host pointer.
					 * @param[in] a_ptr host pointer on the data storage
					 * @param[in] a_size is the data size
					 */
					void host_aliasing(T* a_ptr, unsigned int a_size)
					{
						CPU_WORLD(MODE)
						{
							m_host = std::shared_ptr<T>(a_ptr, std::default_delete<T>());
							set_size(a_size);
						}					
					}

					/**
					 * @brief initialize MAGPUVector with a device pointer.
					 * @param[in] a_ptr device pointer on the data storage
					 * @param[in] a_size is the data size
					 */
					virtual void gpu_aliasing(T* a_ptr, unsigned int a_size) = 0;

					/**
					 * @brief initialize MAGPUVector with a pointer depending on the MEM_MODE. BOTH mode is forbidden.
					 * @param[in] a_ptr pointer on the data storage
					 */
					void aliasing(T* a_ptr, unsigned int a_size)
					{
						assert( (MODE == MEM_MODE::BOTH) && " error, this function needs two arguments with MEM_MODE == BOTH."); 
						assert(a_ptr != nullptr && "pointer is null !");

						host_aliasing(a_ptr, a_size);
						gpu_aliasing(a_ptr, a_size);
					}

					/**
					 * @brief initialize MAGPUVector with MEM_MODE. CPU mode and GPU mode are forbidden.
					 * @param[in] a_host_ptr pointer on the host data storage
					 * @param[in] a_gpu_ptr pointer on the device data storage
					 */
					void aliasing(T* a_host_ptr, T* a_gpu_ptr, unsigned int a_size)
					{
						assert( (MODE != MEM_MODE::BOTH) && " error, this function has to be used with MEM_MODE == BOTH.");
						assert(a_host_ptr != nullptr && "host pointer is null !");
						assert(a_gpu_ptr != nullptr && "gpu pointer is null !");

						host_aliasing(a_host_ptr, a_size);
						gpu_aliasing(a_gpu_ptr, a_size);
					}

					/**
					 * @brief access host data
					 * @return Return the pointer on the host data.
					 */ 
					T* get_data()
					{
						BOTH_WORLD(MODE)
						{
							std::cout << "MATools_LOG: Error in get_data(), with MEM_MODE=BOTH, you need to specify which memory you need with get_data(int type), 0 = cpu and 1 = device" << std::endl;
							std::abort();
						}

						CPU_WORLD(MODE)
						{
							return get_host_data();
						}

						GPU_WORLD(MODE)
						{
							return get_device_data();
						}
					}

					T* get_data(int a_type)
					{
						if(a_type == 0)
						{
							return get_host_data();
						}
						else if (a_type == 1)
						{
							return get_device_data();
						}
						else
						{
							std::abort();
						}
					}

				private:
					/** @brief host data */
					std::shared_ptr<T> m_host;
					/** @brief data size */
					unsigned int m_size;
			};

			template<typename T, MEM_MODE MODE>
			class MASERIALVector : public MAGPUVector<T, MODE, GPU_TYPE::SERIAL>
			{
				public:

					// already defined

					/**
					 * @brief access host data
					 * @return Return the pointer on the host data.
					 */ 
					//T* get_data();

					// 
					void device_error()
					{
						GPU_WORLD(MODE)
						{
							std::cout << "MATools_LOG: This vector is set with -SERIAL- gpu type, you can't use the gpu memory mode " << std::endl;
							std::abort();
						}
					}

					/**
					 * @brief Gets device memory pointer
					 * @return device pointer, this pointer is defined for each specialization
					 */
					T* get_device_data() final
					{
						device_error();
						return nullptr;
					}

					/**
					 * @brief GPU allocator if MEM_MODE is set to GPU or BOTH
					 * @param[in] a_size size of the storage
					 */
					virtual void gpu_allocator(const std::size_t a_size) final
					{
						device_error();
					}

					/**
					 * @brief Initializes the gpu memory if MEM_MODE is set to GPU or BOTH
					 * @param[in] a_val is the filling value
					 * @param[in] a_size is the size of the storage
					 */
					void gpu_init(const T& a_val, const std::size_t a_size)
					{
						device_error();
					}

					/**
					 * @brief Initializes the gpu memory by copying data if MEM_MODE is set to GPU or BOTH
					 * @param[in] a_ptr contains the filling values
					 * @param[in] a_size is the size of the storage
					 */
					void gpu_init(T* a_ptr, const std::size_t a_size)
					{
						device_error();
					}

					/**
					 * @brief initialize MAGPUVector with a device pointer.
					 * @param[in] a_ptr device pointer on the data storage
					 * @param[in] a_size is the data size
					 */
					void gpu_aliasing(T* a_ptr, unsigned int a_size)
					{
						device_error();
					}

					void sync()
					{
					}

					void host_to_device()
					{
						device_error();
					}

					void device_to_host()
					{
						device_error();
					}

					void host_sync()
					{
						device_error();
					}

					void gpu_sync()
					{
						device_error();
					}
			};

#ifdef __CUDA__
#include<cuda.h>

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

		template<typename T, MEM_MODE MODE>
			class MAGPUVector<T, MODE, GPU_TYPE::CUDA>
			{
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
				void gpu_init(T* a_ptr, const std::size_t a_size);
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

				void sync()
				{
					GPU_WORLD(MODE)
					{
						cudaDeviceSynchronize();
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

				void host_sync()
				{
					BOTH_WROLD(MODE)
					{
						sync();
						device_to_host();
					}
				}

				void gpu_sync();
				{
					BOTH_WROLD(MODE)
					{
						sync();
						host_to_device();
					}
				}



				private:
				/** @brief device data */
				std::shared_ptr<T> m_device;

				/** @brief possibility to use a cuda stream */
				cudaStream_t m_stream = cudaStream_t(0); // default stream
			};
#endif

		/*
			 template<typename T, MEM_MODE MM, GPU_TYPE GT>
			 void MAGPUVector<T,MM,GT>::init(const T& a_val, const std::size_t a_size)
			 {
			 host_allocator(a_size);
			 host_init(a_val, a_size);
			 gpu_allocator(a_size);
			 gpu_init(a_val, a_size);
			 }
		 */
	}
}
