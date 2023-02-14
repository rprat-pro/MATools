#pragma once

namespace MATools
{
  namespace MAGPU
  {
    template<typename T, MEM_MODE MODE>
      class MADeviceMemory<T, MODE, GPU_TYPE::SERIAL>
      {
        protected:
          void device_error()
          {
            GPU_WORLD(MODE)
            {
              std::cout << "MATools_LOG: WARNING This vector is set with -SERIAL- gpu type, you can't use the gpu memory mode " << std::endl;
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
	};
};
