
#pragma once
#include <MAGPUBasicFunctors.hxx>
#include <MAGPUFunctor.hxx>
#include <test_helper.hpp>

  template<typename T, MATools::MAGPU::GPU_TYPE GT>
bool run_test_create_functor()
{
  using namespace MATools::MAGPU;
  constexpr auto my_func = Ker::resetF;
  auto functor = Ker::create_functor<GT> (my_func, "reset"); 
  if(functor.get_name() != "reset") return EXIT_FAILURE;
  return EXIT_SUCCESS;
}

  template<typename T, MATools::MAGPU::GPU_TYPE GT>
bool run_test_functor_empty()
{
  using namespace MATools::MAGPU;
  constexpr auto empty_function = [](unsigned int idx) {} ;
  MAGPUFunctor<decltype(empty_function), GT> my_functor(empty_function);
  my_functor(123456789);
  if(my_functor.get_name() != "default_name") return EXIT_FAILURE;
  return EXIT_SUCCESS;
}

  template<typename T, MATools::MAGPU::MEM_MODE MM, MATools::MAGPU::GPU_TYPE GT>
bool run_test_functor_add_sub_mul_div()
{
  using namespace MATools::MAGPU;
  std::cout << MM << " -- " << GT << std::endl;
  auto my_add = Ker::create_functor<GT> (Ker::addF, "add"); 
  auto my_sub = Ker::create_functor<GT> (Ker::subF, "sub"); 
  auto my_mul = Ker::create_functor<GT> (Ker::multF, "mul"); 
  auto my_div = Ker::create_functor<GT> (Ker::divF, "div"); 

  if(MM == MATools::MAGPU::MEM_MODE::BOTH)
  {
    std::cout << " ERROR " <<  std::endl;
    constexpr MEM_MODE MG = MEM_MODE::GPU;	
    constexpr MEM_MODE MC = MEM_MODE::CPU;	
    constexpr T value = 4;
    constexpr int idx = 0;

    // host
    test_helper::create<T,MC,GT> h_allocator;
    test_helper::destroy<T,MC,GT> h_destructor;
    test_helper::copier<T,MC,GT> h_copy ;
    test_helper::mini_runner<MC,GT> h_launcher;

    T* h_two = h_allocator(2,1);
    T* h_res = h_allocator(value,1);

    h_launcher(my_add, idx, h_res, h_two);
    h_launcher(my_sub, idx, h_res, h_two);
    h_launcher(my_mul, idx, h_res, h_two);
    h_launcher(my_div, idx, h_res, h_two);

    std::vector<T> host_res(1,0);
    h_copy(host_res.data(), h_res, 1);

    // device
    test_helper::create<T,MG,GT> d_allocator;
    test_helper::destroy<T,MG,GT> d_destructor;
    test_helper::copier<T,MG,GT> d_copy ;
    test_helper::mini_runner<MG,GT> d_launcher;

    T* d_two = d_allocator(2,1);
    T* d_res = d_allocator(value,1);

    d_launcher(my_add, idx, d_res, d_two);
    d_launcher(my_sub, idx, d_res, d_two);
    d_launcher(my_mul, idx, d_res, d_two);
    d_launcher(my_div, idx, d_res, d_two);

    std::vector<T> devi_res(1,0);
    d_copy(devi_res.data(), d_res, 1);

    // check
    if(host_res[0] != value) return EXIT_FAILURE;
    if(devi_res[0] != value) return EXIT_FAILURE;
    h_destructor(h_two);
    h_destructor(h_res);
    d_destructor(d_two);
    d_destructor(d_res);
  }
  else
  {
    test_helper::create<T,MM,GT> allocator;
    test_helper::destroy<T,MM,GT> destructor;
    test_helper::copier<T,MM,GT> _copy ;

    // init
    T* two = allocator(2,1);
    assert(two != nullptr);
    T value = 4;
    T* res = allocator(value,1);
    assert(res != nullptr);

    // define runner
    test_helper::mini_runner<MM,GT> launcher;

    // run
    int idx = 0;
    launcher(my_add, idx, res, two);
    launcher(my_sub, idx, res, two);
    launcher(my_mul, idx, res, two);
    launcher(my_div, idx, res, two);	

    // check results : res = ( ( value + 2 - 2 ) * 2 ) / 2 = value
    std::vector<T> host(1,666);
    T* rawPtr = host.data();

    _copy(rawPtr, res, 1);

    if(host[0] != value)
    {
      std::cout << " error, host should be equal to 4, host = " << host[0] << std::endl;
      destructor(two);
      destructor(res);
      return EXIT_FAILURE;
    }

    destructor(two);
    destructor(res);
  }
  return EXIT_SUCCESS;
}

TYPE_GPU_TEST_CASE(create_functor);
TYPE_GPU_TEST_CASE(functor_empty);
SUPER_TEST_CASE(functor_add_sub_mul_div);
