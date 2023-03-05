#pragma once

#include<datatype/MAGPUFunctor.hxx>

/* ***************** */
/* MAGPU decorations */
/* ***************** */

#ifdef __CUDA__
#define MAGPU_DECORATION __host__ __device__
#else
#define MAGPU_DECORATION
#endif


/* *********************** */
/* Define functors/kernels */
/* *********************** */


#define DEFINE_KERNEL(NAME) MAGPU_DECORATION\
  void kernel_##NAME 

#define END_KERNEL(NAME) struct functor_##NAME {\
  template<typename... Args>\
  MAGPU_DECORATION\
  void operator() (Args&&... a_args) const\
  {\
    kernel_##NAME(std::forward<Args>(a_args)...);\
  };\
};\
const functor_##NAME NAME;

/* ****************************** */
/* Help to create an MAGPUFunctor */
/* ****************************** */

namespace MATools 
{ 
  namespace MAGPU 
  {
    template<GPU_TYPE GT, typename Functor> 
      MAGPUFunctor<Functor, GT> create_functor(Functor& a_functor, std::string a_name = "default_name")
      {
	MAGPUFunctor<Functor, GT> ret(a_functor, a_name);
	return ret;
      }
  }
}
