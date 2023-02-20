
#pragma once

namespace MATools
{
	namespace MAGPU
	{
		template<typename Func, GPU_TYPE GT>
		struct MAGPUFunctor
		{
			template<typename... Args>
			void operator()(Args... a_args);
		};

		template<typename Func>
		struct MAGPUFunctor<Func, GPU_TYPE::SERIAL>
		{
		  MAGPUFunctor(Func& a_ker, std::string a_name) : m_kernel(a_ker), m_name(a_name) {} 	

			template<typename... Args>
			void operator()(unsigned int a_idx, Args&&... a_args)
			{
				m_kernel(a_idx, std::forward<Args>(a_args)...);
			}

			Func m_kernel;
			std::string m_name;
		};


#ifdef __CUDA__
		template<typename Func>
		struct MAGPUFunctor<Func, GPU_TYPE::CUDA>
		{
		  MAGPUFunctor(Func& a_ker, std::string a_name) : m_kernel(a_ker), m_name(a_name) {} 	

			template<typename... Args>
			__host__ __device__
			void operator()(unsigned int a_idx, Args...&& a_args)
			{
				m_kernel(a_idx, std::forward<Args>(a_args)...);
			}

			Func m_kernel;
			std::string m_name;
		};
#endif

	};
}
