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



//template<unsigned short N, typename T>
template<typename Stensor, typename T>
DEFINE_KERNEL(elasticity_version3)(
    Stensor& a_sig, const Stensor& a_eto, const Stensor& a_deto, 
    const T& a_lambda, const T& a_mu, const unsigned short a_s, const Stensor& a_id)
{
  const auto& e = a_eto + a_deto;
  a_sig = a_lambda * (e(0)+e(1)+e(2)) * a_id + 2 * a_mu * e;
}
END_KERNEL(elasticity_version3)

using namespace tfel::math;

  template<typename T, MATools::MAGPU::MEM_MODE MODE, MATools::MAGPU::GPU_TYPE GT>
bool run_test_elasticity_version3()
{
  using namespace MATools::MAGPU;
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
  constexpr int size = 1000;
  MAGPUVector<stensor<N, T>, MODE, GT> sig;
  MAGPUVector<stensor<N, T>, MODE, GT> eto;
  MAGPUVector<stensor<N, T>, MODE, GT> deto;

  sig. init(stensor<N, T>(666), size);
  eto. init(stensor<N, T>(1.0), size);
  deto.init(stensor<N, T>(2.0), size);

  // run kernel
  runner.launcher_test(functor, size, sig, eto, deto, lambda, mu, s, id);	

  // check
  std::vector<stensor<N,T>> host = sig.copy_to_vector();

  auto check = [] (int i, double val) -> bool 
  { 
    if( ((i<3) && (val == 1.125e+12)) ||  ((i>=3 && i < 6) && ( std::abs( val - 3.46154e+11) <= 1e+6)) )
    {
      return true;
    }
    else
    {
      return false;
    }
  };

  auto check_stensor = [s, &check] (stensor<N, T>& it)->bool
  {
    bool res = true;
    for(int shift = 0 ; shift < s ; shift++)
    {
      res &= check(shift, it(shift));
    }
    return res;
  };

  for(auto& my_it : host)
  {
    bool tmp = check_stensor(my_it);
    if(tmp == false)
    {
      std::cout << " error, wrong values " << std::endl;
      return EXIT_FAILURE;
    }
  }

  std::cout << " This test Works for Mode : " << MODE << " and GPU parallelization type : " << GT << std::endl; 
  return EXIT_SUCCESS;
}

int main()
{
  bool success = EXIT_SUCCESS;
  using namespace MATools::MAGPU;
  success &= run_test_elasticity_version3<double, MEM_MODE::GPU, GPU_TYPE::SERIAL>();
//  success &= run_test_elasticity_version3<double, MEM_MODE::GPU, GPU_TYPE::CUDA>();
//  success &= run_test_elasticity_version3<double, MEM_MODE::CPU, GPU_TYPE::CUDA>();
  if(success == EXIT_SUCCESS) std::cout << "Ok!" << std::endl;
  else std::cout << "Not ok!" << std::endl;
  return success;
}
