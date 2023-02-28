#pragma once

#include<MAGPUFunctor.hxx>


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



