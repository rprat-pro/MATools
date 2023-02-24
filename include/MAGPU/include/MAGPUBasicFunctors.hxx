#pragma once

#include<MAGPUFunctor.hxx>

#ifdef __CUDA__
#define MAGPU_DECORATION __host__ __device__
#else
#define MAGPU_DECORATION
#endif

namespace MATools
{
  namespace MAGPU
  {
    namespace Ker
    {
      template<typename T>
	MAGPU_DECORATION
	void reset(unsigned int idx, T* const out)
	{
	  out[idx] = 0;
	}

      template<typename T>
	MAGPU_DECORATION
	void add(unsigned int idx, T* const a, const T* const b)
	{
	  a[idx] += b[idx];
	}	

      template<typename T>
	MAGPU_DECORATION
	void sub(unsigned int idx, T* const a, const T* const b)
	{
	  a[idx] -= b[idx];
	}

      template<typename T>
	MAGPU_DECORATION
	void mult(unsigned int idx, T* const a, const T* const b)
	{
	  a[idx] *= b[idx];
	}

      template<typename T>
	MAGPU_DECORATION
	void div(unsigned int idx, T* const a, const T* const b)
	{
	  a[idx] /= b[idx];
	}

      template<typename T>
	MAGPU_DECORATION
	void fill(unsigned int idx, T* const a, const T& b)
	{
	  a[idx] = b;
	}


      template<typename T>
	MAGPU_DECORATION
	void add_sub_mult_div(unsigned int idx, T* const a, const T* const b)
	{
	  add<T>(idx, a, b);
	  sub<T>(idx, a, b);
	  mult<T>(idx, a, b);
	  div<T>(idx, a, b);
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
