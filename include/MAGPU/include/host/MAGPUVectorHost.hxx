#pragma once

#include <datatype/MAGPUTypes.hxx>

namespace MATools
{
  namespace MAGPU
  {
    template<typename T>
      class MAHostMemory
      {
				protected:
					/** 
					 * @brief default constructor
					 */
					MAHostMemory() : m_host_size(0), m_host(nullptr) {}

					/**
					 * @brief Host allocator if MEM_MODE is set to CPU or BOTH
					 * @param[in] a_size size of the storage
					 */
					void host_allocator(const std::size_t a_size)
					{
						m_host = std::shared_ptr<T>(new T[a_size], std::default_delete<T>());
						set_host_size(a_size);
					}

					/**
					 * @brief Initializes the host memory if MEM_MODE is set to CPU or BOTH
					 * @param[in] a_val is the filling value
					 * @param[in] a_size is the size of the storage
					 */
					void host_init(const T& a_val, const std::size_t a_size)
					{
						T* host_ptr = get_host_data();
						for(std::size_t id = 0 ; id < a_size ; id++)
						{
							host_ptr[id] = a_val;
						}
					}

					/**
					 * @brief Fills the host memory if MEM_MODE is set to CPU or BOTH
					 * @param[in] a_val is the filling value
					 */
					void host_fill(const T& a_val)
					{
						T* host_ptr = get_host_data();
						const int size = get_host_size();
						for(std::size_t id = 0 ; id < size ; id++)
						{
							host_ptr[id] = a_val;
						}
					}
					/**
					 * @brief Initializes the host memory by copying data if MEM_MODE is set to CPU or BOTH
					 * @param[in] a_ptr contains the filling values
					 * @param[in] a_size is the size of the storage
					 */
					void host_init(T* a_ptr, const std::size_t a_size)
					{
						assert(a_ptr != nullptr);
						assert(a_size > 0);
						for(int id = 0 ; id < a_size ; id++)
						{
							m_host[id] = a_ptr[id];
						}
					}

					void host_resize(unsigned int a_size)
					{
						if(a_size == 0) 
						{
							set_host_size(0);
						}
						else if(m_host_size > a_size) /* */
						{
							/* Warning, this vector has been reduced */
							set_host_size(a_size);
						}
						else if(m_host_size != a_size)
						{
							if(m_host.get() == nullptr) // scenario : only host memory has been defined with an extern storage
							{
								host_allocator(a_size);
							}
							else
							{
								std::cout << " HOST RESIZE : It's not possible to enlarge the memory of a MAGPUVector that has already been defined" << std::endl;
								std::abort();
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
#ifdef __VERBOSE_MAGPU
						std::cout << " host_aliasing -> size " << a_size << std::endl;
#endif
						assert(a_ptr != nullptr);
						m_host = std::shared_ptr<T>(a_ptr, [](T*){}); // no destructor
						set_host_size(a_size);
					}
					/**
					 * @brief Gets host memory pointer
					 * @return device pointer, this pointer is defined for each specialization
					 */
					T* get_host_data()
					{
						T* ret = m_host.get();
						return ret;
					}

					/**
					 * @brief Gets size
					 * @return m_host_size member
					 */
					unsigned int get_host_size()
					{
						unsigned int ret = m_host_size;
						return ret;
					}

					/**
					 * @brief Sets size
					 * @param new value of m_host_size
					 */
					void set_host_size(unsigned int a_size)
					{
						m_host_size = a_size;
					}


				public:
					std::vector<T> copy_to_vector_from_host()
					{
						std::vector<T> ret;
						T* raw_ptr = get_host_data();
						unsigned int size = get_host_size();
						ret.resize(size);
						std::copy(raw_ptr, raw_ptr + m_host_size, ret.data());
						return ret;
					}

				private:
					/** @brief host data */
					std::shared_ptr<T> m_host;
					/** @brief data size */
					unsigned int m_host_size;
			};
	}
}
