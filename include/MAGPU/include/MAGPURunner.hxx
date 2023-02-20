#pragma once

#include <MAGPUVector.hxx>

namespace MATools
{
	namespace MAGPU
	{
			template<typename T>
				T&& get_data(T&& a_in)
				{
					return a_in;
				}

			template<typename T, MEM_MODE MODE, GPU_TYPE GT>
				T* get_data(MAGPUVector<T, MODE, GT>& a_in)
				{
					auto ret = a_in.get_data();
					return ret;
				}

		template<MEM_MODE MODE, GPU_TYPE GT>
		struct MAGPURunner
		{
		};

		template<GPU_TYPE GT>
			struct MAGPURunner<MEM_MODE::CPU, GT>
		{
			template<typename Functor, typename... Args>
				void host(Functor& a_functor, size_t a_size, Args&&... a_args) const
				{
#ifdef __VERBOSE_LAUNCHER
					std::cout << " HOST: " << a_functor.m_name << std::endl;
#endif

					for(size_t it = 0 ; it < a_size ; it++)
						a_functor(it, a_args...);
				}

			template<typename Functor, typename... Args>
				void operator()(Functor& a_functor, unsigned int a_size, Args&&... a_args)
				{
					host(a_functor, a_size,  get_data(a_args)...);
				}
		};

		template<>
			struct MAGPURunner<MEM_MODE::GPU, GPU_TYPE::SERIAL>
		{
			template<typename Functor, typename... Args>
				void serial(Functor& a_functor, size_t a_size, Args&&... a_args) const
				{
#ifdef __VERBOSE_LAUNCHER
					std::cout << " SERIAL: " << a_functor.m_name << std::endl;
#endif

					for(size_t it = 0 ; it < a_size ; it++)
						a_functor(it, a_args...);
				}


			template<typename Functor, typename... Args>
				void operator()(Functor& a_functor, unsigned int a_size, Args&&... a_args)
				{
					serial(a_functor, a_size,  get_data(a_args)...);
				}
		};
	}
}
