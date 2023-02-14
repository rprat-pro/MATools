#pragma once

#include <vector>
#include <memory>
#include <cassert>
#include <iostream>

#include <MAGPUTypes.hxx>
#include <MAGPUVectorHost.hxx>
#include <MAGPUVectorDevice.hxx>

namespace MATools
{
	namespace MAGPU
	{
		/**
		 * @brief MAVector is a class with a GPU storage and a CPU Storage depending on the memory mode.  
		 */		
		template<typename T, MEM_MODE MODE, GPU_TYPE GTYPE>
			class MAGPUVector : public MAHostMemory<T,MODE> , MADeviceMemory<T,MODE,GTYPE>
		{
			public:

				/** resize vector */ 
				void resize(const std::size_t a_size)
				{
					this->host_allocator(a_size);
					this->gpu_allocator(a_size);
				}

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
					assert( (MODE != MEM_MODE::BOTH) && " error, this function needs two arguments with MEM_MODE == BOTH."); 
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

