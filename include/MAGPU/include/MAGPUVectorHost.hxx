#pragma once

#include <MAGPUTypes.hxx>

namespace MATools
{
	namespace MAGPU
	{
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
							m_host = std::shared_ptr<T>(a_ptr, [](T*){}); // no destructor
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
	}
}
