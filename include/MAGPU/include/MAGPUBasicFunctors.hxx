#pragma once

#include<MAGPUFunctor.hxx>

namespace MATools
{
	namespace MAGPU
	{
		namespace Ker
		{
			template<typename T>
			void reset(unsigned int idx, T* const out)
			{
				out[idx] = 0;
			}

			template<typename T>
			void add(unsigned int idx, T* const a, const T* const b)
			{
				a[idx] += b[idx];
			}

			template<typename T>
			void sub(unsigned int idx, T* const a, const T* const b)
			{
				a[idx] -= b[idx];
			}

			template<typename T>
			void mult(unsigned int idx, T* const a, const T* const b)
			{
				a[idx] *= b[idx];
			}

			template<typename T>
			void div(unsigned int idx, T* const a, const T* const b)
			{
				a[idx] /= b[idx];
			}

			template<typename T>
			void fill(unsigned int idx, T* const a, const T& b)
			{
				a[idx] = b;
			}

			
			template<typename T>
			void add_sub_mult_div(unsigned int idx, T* const a, const T* const b)
			{
				add(idx, a, b);
				sub(idx, a, b);
				mult(idx, a, b);
				div(idx, a, b);
			}

			template<GPU_TYPE GT, typename Functor> 
				MAGPUFunctor<Functor, GT> create_functor(Functor& a_functor, std::string a_name = "default_name")
				{
					MAGPUFunctor<Functor, GT> ret(a_functor, a_name);
					return ret;
				}

		}
	}
}
