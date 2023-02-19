#pragma once

#include<MAGPUTypes.hxx>

namespace MATools
{
	namespace MAGPU
	{

		template<typename T, GPU_TYPE GTYPE>
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
	}
}

#include<MAGPUVectorSerial.hxx>
#include<MAGPUVectorCuda.hxx>
