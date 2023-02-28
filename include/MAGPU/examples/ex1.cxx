#include <iostream>
#include <MAGPUVector.hxx>
#include <MAGPUFunctor.hxx>
#include <MAGPUBasicFunctors.hxx>
#include <MAGPURunner.hxx>

#include <cstdlib>
#include "TFEL/Math/Array/View.hxx"
#include "TFEL/Math/qt.hxx"
#include "TFEL/Math/stensor.hxx"
#include "TFEL/Material/Lame.hxx"

template<typename Stensor, typename T>
DEFINE_KERNEL(behavior_law)(
    Stensor& a_sig, const Stensor& a_eto, const Stensor& a_deto, 
    const T& a_lambda, const T& a_mu, const unsigned short a_s, const Stensor& a_id, bool& a_error)
{
  const auto& e = a_eto + a_deto;
  a_sig = a_lambda * (e(0)+e(1)+e(2)) * a_id + 2 * a_mu * e;
  if constexpr (a_sig < 1E-16) a_error = false;
  else a_error = true
}
END_KERNEL(elasticity_version3)

using namespace tfel::math;
using namespace MATools::MAGPU;

  template<typename T, MATools::MAGPU::GPU_TYPE GT>
bool run_example_1()
{
  constexpr auto both_memory = MEM_MODE::BOTH;
  constexpr auto cpu_memory = MEM_MODE::GPU;
  constexpr auto gpu_memory = MEM_MODE::GPU;
  using _vector  = MAGPUVector<stensor<N, T>, both, GT>;
  using _checker = MAGPUVector<bool, both, GT>;



  using template_value = double; // replace typename T
  constexpr bool use_qt = false;
  using real = std::conditional_t<use_qt, qt<NoUnit, template_value>, template_value>;
  using stress = std::conditional_t<use_qt, qt<Stress, template_value>, template_value>;

  // mfront stuff
  constexpr unsigned short N = 3;
  constexpr auto s = StensorDimeToSize<N>::value;
  constexpr auto id = stensor<N, T>::Id();
  constexpr auto young = stress{150e9};
  constexpr auto nu = real{0.3};
  constexpr auto lambda = tfel::material::computeLambda(young, nu);
  constexpr auto mu = tfel::material::computeMu(young, nu);

  // MAGPU stuff
  auto functor = Ker::create_functor<GT> (elasticity_version3, "elasticity_v3");
  MAGPURunner<MODE, GT> runner;

  // init problem
  constexpr int size = 100000;

  // stuff from other applications, device version
  using mfront_vector = T*;
  mfront_vector mfront_sig  = new T[size];
  mfront_vector mfront_eto  = new T[size];
  mfront_vector mfront_deto = new T[size];

  for(int i = 0 ;  i < size ; i++)
  {
    mfront_sig[i]  = 666;
    mfront_eto[i]  = 1;
    mfront_deto[i] = 2;
  }


  // magpu 
  _vector sig;
  _vector eto;
  _vector deto;
  _checker error;

  sig.aliasing<cpu_memory>(mfront_sig, size);
  eto.aliasing<cpu_memory>(mfront_eto, size);
  deto.aliasing<cpu_memory>(mfront_deto, size);

  // run kernel
  runner.launcher_test(functor, size, sig, eto, deto, lambda, mu, s, id, error);	

  // check
  std::vector<stensor<N,T>> host = sig.copy_to_vector();

  return EXIT_SUCCESS;
}

int main()
{
  bool success = EXIT_SUCCESS;
  using namespace MATools::MAGPU;
  success &= run_test_elasticity_version3<double, MEM_MODE::GPU, GPU_TYPE::SERIAL>();
  if(success == EXIT_SUCCESS) std::cout << "Ok!" << std::endl;
  else std::cout << "Not ok!" << std::endl;
  return success;
}
