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


		template<typename T, MEM_MODE MODE>
			class MAHostMemory
			{
				protected:
					/**
					 * @brief Host allocator if MEM_MODE is set to CPU or BOTH
					 * @param[in] a_size size of the storage
					 */
					void host_allocator(const std::size_t a_size)
					{
						CPU_WORLD(MODE)
						{
							m_host = std::shared_ptr<T>(new T[a_size], std::default_delete<T>());
							set_host_size(a_size);
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
					 * @brief initialize MAGPUVector with a host pointer.
					 * @param[in] a_ptr host pointer on the data storage
					 * @param[in] a_size is the data size
					 */
					void host_aliasing(T* a_ptr, unsigned int a_size)
					{
						CPU_WORLD(MODE)
						{
							m_host = std::shared_ptr<T>(a_ptr, std::default_delete<T>());
							set_host_size(a_size);
						}					
						else
						{
							assert(0==1 && "MATools_LOG: -host_aliasing- should not be called with the GPU memory type. Release mode do nothing.");
						}
					}

					/**
					 * @brief Gets host memory pointer
					 * @return device pointer, this pointer is defined for each specialization
					 */
					T* get_host_data()
					{
						CPU_WORLD(MODE)
						{
							T* ret = m_host.get();
							return ret;
						}
						else
						{
							assert(0==1 && "MATools_LOG: -get_host_data- should not be called with the GPU memory type. Release mode returns nullptr.");
							return nullptr;
						}
					}

					/**
					 * @brief Gets size
					 * @return m_size member
					 */
					unsigned int get_host_size()
					{
						CPU_WORLD(MODE)
						{
							unsigned int ret = m_size;
							return ret;
						}
						else
						{
							assert(0==1 && "MATools_LOG: -get_host_size- should not be called with the GPU memory type. Release mode returns 0.");
							return 0;
						}
					}

					/**
					 * @brief Sets size
					 * @param new value of m_size
					 */
					void set_host_size(unsigned int a_size)
					{
						CPU_WORLD(MODE)
						{
							m_size = a_size;
						}
						else
						{
							assert(0==1 && "MATools_LOG: -get_host_size- should not be called with the GPU memory type. Release mode do nothing.");
						}
					}

				private:
					/** @brief host data */
					std::shared_ptr<T> m_host;
					/** @brief data size */
					unsigned int m_size;
			};

		template<typename T, MEM_MODE MODE, GPU_TYPE GTYPE>
			class MADeviceMemory
			{
				/**
				 * @brief GPU allocator if MEM_MODE is set to GPU or BOTH
				 * @param[in] a_size size of the storage
				 */
				void gpu_allocator(const std::size_t a_size);

				/**
				 * @brief Initializes the gpu memory if MEM_MODE is set to GPU or BOTH
				 * @param[in] a_val is the filling value
				 * @param[in] a_size is the size of the storage
				 */
				void gpu_init(const T& a_val, const std::size_t a_size);

				/**
				 * @brief Initializes the gpu memory by copying data if MEM_MODE is set to GPU or BOTH
				 * @param[in] a_ptr contains the filling values
				 * @param[in] a_size is the size of the storage
				 */
				void gpu_init(T* a_ptr, const std::size_t a_size);

				/**
				 * @brief initialize MAGPUVector with a device pointer.
				 * @param[in] a_ptr device pointer on the data storage
				 * @param[in] a_size is the data size
				 */
				void gpu_aliasing(T* a_ptr, unsigned int a_size);

				/**
				 * @brief Gets device memory pointer
				 * @return device pointer, this pointer is defined for each specialization
				 */
				T* get_device_data();
				void host_sync();
				void gpu_sync();
				void host_to_device();
				void device_to_host();

				/**
				 * @brief Gets size
				 * @return m_size member
				 */
				unsigned int get_device_size();

				/**
				 * @brief Sets size
				 * @param new value of m_size
				 */
				void set_device_size(unsigned int a_size);
			};

		template<typename T, MEM_MODE MODE>
			class MADeviceMemory<T, MODE, GPU_TYPE::SERIAL>
			{
				protected:
					void device_error()
					{
						GPU_WORLD(MODE)
						{
							std::cout << "MATools_LOG: This vector is set with -SERIAL- gpu type, you can't use the gpu memory mode " << std::endl;
							std::abort();
						}
					}

					/**
					 * @brief GPU allocator if MEM_MODE is set to GPU or BOTH
					 * @param[in] a_size size of the storage
					 */
					void gpu_allocator(const std::size_t a_size)
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

					/**
					 * @brief Gets device memory pointer
					 * @return device pointer, this pointer is defined for each specialization
					 */
					T* get_device_data()
					{
						return nullptr;
					}

					void host_sync()
					{
						device_error();
					}

					void gpu_sync()
					{
						device_error();
					}

					void host_to_device()
					{
						device_error();
					}

					void device_to_host()
					{
						device_error();
					}

					/**
					 * @brief Gets size
					 * @return m_size member
					 */
					unsigned int get_device_size()
					{
						device_error();
						return 0;
					}

					/**
					 * @brief Sets size
					 * @param new value of m_size
					 */
					void set_device_size(unsigned int a_size)
					{
						device_error();
					}
			};

		/**
		 * @brief MAVector is a class with a GPU storage and a CPU Storage depending on the memory mode.  
		 */		
		template<typename T, MEM_MODE MODE, GPU_TYPE GTYPE>
			class MAGPUVector : public MAHostMemory<T,MODE> , MADeviceMemory<T,MODE,GTYPE>
		{
			public:

				/**
				 * @brief Initializes the MAGPUVector memory and fill it with the a_val value
				 * @param[in] a_val is the filling value
				 * @param[in] a_size is the size of the storage
				 */
				void init(const T& a_val, const std::size_t a_size)
				{
					this->host_allocator(a_size);
					this->host_init(a_val, a_size);
					this->gpu_allocator(a_size);
					this->gpu_init(a_val, a_size);
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
						this->host_allocator(a_size);
						this->host_init(a_ptr, a_size);
						this->gpu_sync(); // BOTH MODE, do nothing else
						return;
					}

					GPU_WORLD(MODE)
					{
						this->gpu_allocator(a_size);
						this->gpu_init(a_ptr, a_size);
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
						this->host_allocator(a_size);
						this->host_init(a_host_ptr, a_size);
						this->gpu_allocator(a_size);
						this->gpu_init(a_device_ptr, a_size);
					}
					else
					{
						std::cout << "MATools_Error: wrong usage of MAGPUVector::init(...)" << std::endl;;
						std::abort();
					}
				}

				/**
				 * @brief Gets size
				 * @return m_size member
				 */
				unsigned int get_size()
				{
					BOTH_WORLD(MODE)
					{
						unsigned int host_size 		= this->get_host_size();
						unsigned int device_size 	= this->get_device_size();
						assert(host_size == device_size && "host and device size are not the same");
						return host_size;
					}

					CPU_WORLD(MODE)
					{
						unsigned int ret = this->get_host_size();
						return ret;
					}

					GPU_WORLD(MODE)
					{
						unsigned int ret = this->get_device_size();
						return ret;
					}
				}

				/**
				 * @brief Sets size
				 * @param new value of m_size
				 */
				void set_size(unsigned int a_size)
				{
					this->set_host_size(a_size);
					this->set_device_size(a_size);
				}


				/**
				 * @brief initialize MAGPUVector with a pointer depending on the MEM_MODE. BOTH mode is forbidden.
				 * @param[in] a_ptr pointer on the data storage
				 */
				void aliasing(T* a_ptr, unsigned int a_size)
				{
					assert( (MODE == MEM_MODE::BOTH) && " error, this function needs two arguments with MEM_MODE == BOTH."); 
					assert(a_ptr != nullptr && "pointer is null !");

					this->host_aliasing(a_ptr, a_size);
					this->gpu_aliasing(a_ptr, a_size);
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

					this->host_aliasing(a_host_ptr, a_size);
					this->gpu_aliasing(a_gpu_ptr, a_size);
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
						return this->get_host_data();
					}

					GPU_WORLD(MODE)
					{
						return this->get_device_data();
					}
				}

				T* get_data(int a_type)
				{
					if(a_type == 0)
					{
						return this->get_host_data();
					}
					else if (a_type == 1)
					{
						return this->get_device_data();
					}
					else
					{
						std::abort();
					}
				}
		};
	} // namespace MAGPU
} //namespace MATools

