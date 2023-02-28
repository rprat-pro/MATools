#pragma once

#include <MAGPUFunctor.hxx>
#include <MAGPUDefineMacros.hxx>

namespace MATools
{
  namespace MAGPU
  {
    namespace Ker
    {
      template<typename T>
        DEFINE_KERNEL(resetF)(unsigned int idx, T* const out)
        { 
          out[idx] = 0; 
        }
      END_KERNEL(resetF)


      template<typename T>
	DEFINE_KERNEL(addF)(unsigned int idx, T* const a, const T* const b)
	{
	  a[idx] += b[idx];
	}
      END_KERNEL(addF)

      template<typename T>
	DEFINE_KERNEL(subF)(unsigned int idx, T* const a, const T* const b)
	{
	  a[idx] -= b[idx];
	}
      END_KERNEL(subF)

      template<typename T>
	DEFINE_KERNEL(multF)(unsigned int idx, T* const a, const T* const b)
	{
	  a[idx] *= b[idx];
	}
      END_KERNEL(multF)

      template<typename T>
	DEFINE_KERNEL(divF)(unsigned int idx, T* const a, const T* const b)
	{
	  a[idx] /= b[idx];
	}
      END_KERNEL(divF)


      template<typename T>
	DEFINE_KERNEL(fillF)(unsigned int idx, T* const a, const T& b)
	{
	  a[idx] = b;
	}
      END_KERNEL(fillF)


      template<typename T>
	DEFINE_KERNEL(add_sub_mult_divF)(unsigned int idx, T* const a, const T* const b)
	{
	  addF(idx, a, b);
	  subF(idx, a, b);
	  multF(idx, a, b);
	  divF(idx, a, b);
	}
      END_KERNEL(add_sub_mult_divF)

      template<GPU_TYPE GT, typename Functor> 
	MAGPUFunctor<Functor, GT> create_functor(Functor& a_functor, std::string a_name = "default_name")
	{
	  MAGPUFunctor<Functor, GT> ret(a_functor, a_name);
	  return ret;
	}

    }
  }
}
