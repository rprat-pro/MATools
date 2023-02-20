#pragma once

namespace MATools
{
  namespace MAGPU
  {
    // equivalent to a vector
    template<typename T>
      class MADeviceMemory<T, GPU_TYPE::SERIAL>
      {
				protected:
					/** 
					 * @brief default constructor
					 */
					MADeviceMemory() : m_size(0), m_data(nullptr) {}

					/**
					 * @brief GPU allocator if MEM_MODE is set to GPU or BOTH
					 * @param[in] a_size size of the storage
					 */
					void gpu_allocator(const std::size_t a_size)
					{
						// just a resize in this case
						m_data = new T[a_size];
						m_size = a_size;
					}

					/**
					 * @brief Initializes the gpu memory if MEM_MODE is set to GPU or BOTH
					 * @param[in] a_val is the filling value
					 * @param[in] a_size is the size of the storage
					 */
					void gpu_init(const T& a_val, const std::size_t a_size)
					{
						gpu_allocator(a_size);
						gpu_fill(a_val);
					}

					/**
					 * @brief Fills the gpu memory if MEM_MODE is set to GPU or BOTH
					 * @param[in] a_val is the filling value
					 */
					void gpu_fill(const T& a_val)
					{
						for(int id = 0 ; id < m_size ; id++)
						{
							m_data[id] = a_val;  
						}
					}

					/**
					 * @brief Initializes the gpu memory by copying data if MEM_MODE is set to GPU or BOTH
					 * @param[in] a_ptr contains the filling values
					 * @param[in] a_size is the size of the storage
					 */
					void gpu_init(T* const a_ptr, const std::size_t a_size)
					{
						assert(a_size == m_size);
						for(int id = 0 ; id < m_size ; id++)
						{
							m_data[id] = a_ptr[id];
						}
					}

					/**
					 * @brief Initializes an MAGPUVector with a device pointer.
					 * @param[in] a_ptr device pointer on the data storage
					 * @param[in] a_size is the data size
					 */
					void gpu_aliasing(T* a_ptr, unsigned int a_size)
					{
						assert(a_ptr != nullptr);
						m_size = a_size;
						m_data = a_ptr;
					}

					/**
					 * @brief Gets device memory pointer
					 * @return device pointer, this pointer is defined for each specialization
					 */
					T* get_device_data()
					{
						T* ret = m_data;
						return ret;
					}

					void gpu_sync()
					{
					}

					void gpu_resize(unsigned int a_size)
					{
						if(m_size > a_size)
						{
							m_size = a_size;
						}
						else{
							T * new_ptr = new double [a_size];
							std::copy (m_data, m_data + m_size, new_ptr);
							delete m_data;
							m_data = new_ptr;
							m_size = a_size;
						}
					}

					void host_to_device(T* a_host, unsigned int a_size)
					{

						gpu_resize(a_size);
						std::copy(a_host, a_host + a_size, m_data);
					}

					void device_to_host(T* a_host)
					{
						std::copy(m_data, m_data + m_size, a_host);
					}

					/**
					 * @brief Gets size
					 * @return m_size member
					 */
					unsigned int get_device_size()
					{
						unsigned int size = m_size;
						return size;
					}

					/**
					 * @brief Sets size -> in this case it's a resize
					 * @param new value of m_size
					 */
					void set_device_size(unsigned int a_size)
					{
						m_size = a_size;
					}

					std::vector<T> copy_to_vector_from_device()
					{
						std::vector<T> ret;
						unsigned int size = get_device_size();
						ret.resize(size);
						device_to_host(ret.data());
						return ret;
					}

				private :
					// I can't use a std::vector because it's not possible to use alias it with a pointer
					T* m_data;
					int m_size;
			};
	};
};
