#pragma once

#include <vector>
#include <memory>
#include <cassert>
#include <iostream>

#include <datatype/MAGPUTypes.hxx>
#include <host/MAGPUVectorHost.hxx>
#include <datatype/MAGPUVectorDevice.hxx>

namespace MATools
{
	namespace MAGPU
	{
		/**
		 * @brief MAVector is a class with a GPU storage and a CPU Storage depending on the memory mode.  
		 */		
		template<typename T, MEM_MODE MODE, GPU_TYPE GTYPE>
			class MAGPUVector : public MAHostMemory<T> , public MADeviceMemory<T, GTYPE>
		{
			public:

				//MAGPUVector() : this->MAHostMemory(), this->MADeviceMemory() {}
				MAGPUVector() = default;

				/** resize vector */ 
				void resize(const std::size_t a_size)
				{
					CPU_WORLD(MODE) this->host_allocator(a_size);
					GPU_WORLD(MODE) this->gpu_allocator(a_size);
				}

				/**
				 * @brief Initializes the MAGPUVector memory and fill it with the a_val value
				 * @param[in] a_val is the filling value
				 * @param[in] a_size is the size of the storage
				 */
				void init(const T& a_val, const std::size_t a_size)
				{
					resize(a_size);
					fill(a_val);
				}

				/**
				 * @brief Fills the MAGPUVector memory and fill it with the a_val value
				 * @param[in] a_val is the filling value
				 */
				void fill(const T& a_val)
				{
					CPU_WORLD(MODE) this->host_fill(a_val);
					GPU_WORLD(MODE) this->gpu_fill(a_val);
				}

				/**
				 * @brief Equal operator that fills an MAGPUVector with a same value.
				 * @param[in] a_val is the filling value
				 */
				MAGPUVector& operator=(const T& a_val)
				{
					this->fill(a_val);
					return *this;
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
						BOTH_WORLD(MODE)
						{
							this->gpu_allocator(a_size);
							this->gpu_init(a_ptr, a_size);
						}
						this->host_allocator(a_size);
						this->host_init(a_ptr, a_size);
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
						unsigned int host_size = this->get_host_size();
						unsigned int device_size = this->get_device_size();
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
					CPU_WORLD(MODE) this->set_host_size(a_size);
					GPU_WORLD(MODE) this->set_device_size(a_size);
				}


				/**
				 * @brief initialize a MAGPUVector with a pointer depending on the MEM_MODE. BOTH mode is forbidden.
				 * @param[in] a_ptr pointer on the data storage
				 */
				void aliasing(T* a_ptr, unsigned int a_size)
				{
					assert( (MODE != MEM_MODE::BOTH) && " error, this function needs two arguments with MEM_MODE == BOTH."); 
					assert(a_ptr != nullptr && "pointer is null !");

					CPU_WORLD(MODE) this->host_aliasing(a_ptr, a_size);
					GPU_WORLD(MODE) this->gpu_aliasing(a_ptr, a_size);
				}

				/**
				 * @brief initialize a MAGPUVector with a pointer depending on the MEM_MODE defined in template. Used with Both mode
				 * @param[in] other is defined the memory type 
				 * @param[in] a_ptr is the pointer on the data storage
				 * @param[in] a_size is the number of elements
				 */
				void define (const MEM_MODE other_mode, T* a_ptr, unsigned int a_size)
				{
					assert(  (MODE == MEM_MODE::BOTH) && (other_mode != MEM_MODE::BOTH) || (other_mode == MODE) );
					if(other_mode == MEM_MODE::CPU) this->host_aliasing(a_ptr, a_size);
					if(other_mode == MEM_MODE::GPU) this->gpu_aliasing(a_ptr, a_size);
				}

				/**
				 * @brief Initializes a MAGPUVector with a pointer depending on the MEM_MODE defined in template and updates the other memory. Used with Both mode
				 * @param[in] other is defined the memory type 
				 * @param[in] a_ptr is the pointer on the data storage
				 * @param[in] a_size is the number of elements
				 */
				void define_and_update(const MEM_MODE other_mode, T* a_ptr, unsigned int a_size)
				{

				  assert(  (MODE == MEM_MODE::BOTH) && (other_mode != MEM_MODE::BOTH) || (other_mode == MODE) );
				  define(other_mode, a_ptr, a_size);
				  if(other_mode == mem_cpu) update_device();
				  if(other_mode == mem_gpu) update_host();
				}

				/**
				 * @brief initialize MAGPUVector with MEM_MODE. CPU mode and GPU mode are forbidden.
				 * @param[in] a_host_ptr pointer on the host data storage
				 * @param[in] a_gpu_ptr pointer on the device data storage
				 */
				void aliasing(T* const a_host_ptr, T* const a_gpu_ptr, unsigned int a_size)
				{
				  assert( (MODE == MEM_MODE::BOTH) && " error, this function has to be used with MEM_MODE == BOTH.");
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

				  T* ret;
				  CPU_WORLD(MODE) ret = this->get_host_data();
				  GPU_WORLD(MODE) ret = this->get_device_data();

				  return ret;
				}

				T* get_data(const MEM_MODE other_mode)
				{
				  assert(MODE == MEM_MODE::BOTH && "BOTH MODE is needed to use get_data(int)");
				  if(other_mode == MEM_MODE::CPU)
				  {
				    return this->get_host_data();
				  }
				  else if (other_mode == MEM_MODE::GPU)
				  {
				    return this->get_device_data();
				  }
				  else
				  {
				    std::abort();
				  }
				}

				void copy_host_to_device(T* const a_host, unsigned int a_host_size)
				{
				  GPU_WORLD(MODE)
				  {
				    this->host_to_device(a_host, a_host_size);
				  }
				}

				/**
				 * @brief Calls the device synchronization function and return true if the MEM_MODE is set to GPU or BOTH
				 * @return Returns true if the MEM_MODE is set to GPU or BOTH
				 */
				bool sync()
				{
				  bool ret = false;
				  GPU_WORLD(MODE)
				  {
				    this->gpu_sync();
				    ret = true;
				  }
				  return ret;
				}

				void copy_device_to_host(T* const a_host)
				{
				  GPU_WORLD(MODE)
				  {
				    this->device_to_host(a_host);
				    this->gpu_sync();
				  }
				}

				bool update_device()
				{
				  BOTH_WORLD(MODE)
				  {
				    T* const host = this->get_host_data();
				    const unsigned int host_size = this->get_host_size();
				    this->copy_host_to_device(host, host_size);
				    return true;
				  }
				  return false;
				}

				bool update_host()
				{
				  BOTH_WORLD(MODE)
				  {
				    T* const host = this->get_host_data();
				    this->copy_device_to_host(host);
				    return true;
				  }
				  return false;
				}

				/**
				 * @brief Copies the data in a new vector depending on the memory mode. Note that if you are using BOTH mode, this function return the data stored in the device memory. If you want to copy the data stored in the host memory, you need to use the function copy_to_vector_from_host directly.
				 * @return a std::vector with a copy of the data
				 */
				std::vector<T> copy_to_vector()
				{
				  GPU_WORLD(MODE)
				  {
				    this->gpu_sync();
				    return this->copy_to_vector_from_device();
				  }
				  else
				  {
				    return this->copy_to_vector_from_host();
				  }
				}



				/** 
				 * @brief Gets memory mode
				 * @return MEM_MODE value
				 */
				MEM_MODE get_memory_mode()
				{
				  MEM_MODE ret = MODE;
				  return ret;
				}
		};
	} // namespace MAGPU
} //namespace MATools

