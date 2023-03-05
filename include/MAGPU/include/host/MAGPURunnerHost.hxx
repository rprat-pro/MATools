#pragma once

namespace MATools
{
	namespace MAGPU
	{
		/**
		 * @brief partial-specialized version of the MAGPURunner class, this class is used to launch a functor with the memeory mode CPU ONLY.
		 */
		template<GPU_TYPE GT>
			struct MAGPURunner<MEM_MODE::CPU, GT>
			{
				template<typename Functor, typename... Args>
					void host(Functor& a_functor, size_t a_size, Args&&... a_args) const
					{
#ifdef __VERBOSE_MAGPU
						std::cout << " HOST: " << a_functor.m_name << std::endl;
#endif

						for(size_t it = 0 ; it < a_size ; it++)
							a_functor(it, a_args...);
					}

				/**
				 * @brief This operator is a FORALL function that iterates over the elements of vector(s) (a_args). 
				 * @param [in] a_functor is the functor applied for all elements
				 * @param [in] a_size is the number of elements
				 * @param [inout] a_args are the parameters of the functor
				 */
				template<typename Functor, typename... Args>
					void operator()(Functor& a_functor, unsigned int a_size, Args&&... a_args)
					{
						host(a_functor, a_size,  get_data<MEM_MODE::CPU>(a_args)...);
					}

				/**
				 * @brief test launcher is a FORALL function that iterates over the elements of vectors. The functor doesn't know the value of idx (the shift).
				 * @param [in] a_functor is the functor applied for all elements
				 * @param [in] a_size is the number of elements
				 * @param [inout] a_args are the parameters of the functor
				 */
				template<typename Functor, typename... Args>
					void launcher_test(Functor& a_functor, unsigned int a_size, Args&&... a_args)
					{
						host_launcher_test(a_functor, a_size,  get_data<MEM_MODE::CPU>(a_args)...);
					}
			};
	}
}
