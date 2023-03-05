#pragma once

namespace MATools
{
	namespace MAGPU
	{
		template<typename Func>
			struct MAGPUFunctor<Func, GPU_TYPE::SERIAL>
			{
				/**
				 * @brief full construtor 
				 * @param [in] a_ker is the kernel used in operator()
				 * @param [in] a_name is the name given to the kernel
				 */
				MAGPUFunctor(Func& a_ker, std::string a_name="default_name") : m_kernel(a_ker), m_name(a_name) {} 	

				/**
				 * @brief apply m_kernel on the idx ieme element
				 */
				template<typename... Args>
					void operator()(unsigned int a_idx, Args&&... a_args)
					{
						m_kernel(a_idx, std::forward<Args>(a_args)...);
					}

				/**
				 * @brief apply m_kernel
				 */
				template<typename... Args>
					void launch_test(Args&&... a_args)
					{
						m_kernel(std::forward<Args>(a_args)...);
					}

				/**
				 * @brief apply m_kernel 
				 */
				template<typename... Args>
					void launch_test(Args&&... a_args) const
					{
						m_kernel(std::forward<Args>(a_args)...);
					}

				/**
				 * @brief Gets name
				 * @return current name
				 */
				std::string get_name()
				{
					std::string ret = m_name;
					return ret;
				}

				/**
				 * @brief Gets gpu parallelization type
				 * @return The template GPU_TYPE value
				 */
				GPU_TYPE get_gpu_type()
				{
					GPU_TYPE ret = GPU_TYPE::SERIAL;
					return ret;
				}

				Func m_kernel;
				std::string m_name;
			};
	};
}
