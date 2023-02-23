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


template<unsigned short N, bool use_qt, typename T>
struct elasticity
{
  elasticity(const double a_lambda,
                const double a_mu,
                const unsigned short a_s,
                const tfel::math::stensor<N, std::conditional_t<false, tfel::math::qt<tfel::math::NoUnit, T>, T>> a_id)
    : m_lambda(a_lambda), m_mu(a_mu), m_s(a_s), m_id(a_id) {}

  void operator()(unsigned int a_idx,
    T * const a_sig,
                const T *const a_eto,
                const T *const a_deto) const
  {
      using namespace tfel::math;
      using NumericType = T;
      using real = std::conditional_t<use_qt, qt<NoUnit, NumericType>, NumericType>; // double
      using stress = std::conditional_t<use_qt, qt<Stress, NumericType>, NumericType>; // double
    auto sig = map<stensor<N, real>>(a_sig + a_idx * m_s);
          const auto eto = map<const stensor<N, real>>(a_eto + a_idx * m_s);
          const auto deto = map<const stensor<N, real>>(a_deto + a_idx * m_s);
          const auto e = eto + deto;
          sig = m_lambda * (e(0)+e(1)+e(2)) * m_id + 2 * m_mu * e;
          //sig = m_lambda * (trace(e)) * m_id + 2 * m_mu * e;
  }

  // variables
  const T m_lambda;
  const T m_mu;
  const unsigned short m_s;
  const tfel::math::stensor<N, std::conditional_t<false, tfel::math::qt<tfel::math::NoUnit, T>, T>> m_id;
};

template<unsigned short N, bool use_qt, typename T>
const elasticity<N,use_qt,T> initElasticity()
{
  // mfront stuff
  using namespace tfel::math;
  using real = std::conditional_t<use_qt, qt<NoUnit, T>, T>;
  using stress =
  std::conditional_t<use_qt, qt<Stress, T>, T>;
  constexpr auto s = StensorDimeToSize<N>::value;
  constexpr auto id = stensor<N, real>::Id();
  constexpr auto young = stress{150e9};
  constexpr auto nu = real{0.3};
  constexpr auto lambda = tfel::material::computeLambda(young, nu);
  constexpr auto mu = tfel::material::computeMu(young, nu);
  const elasticity<N,use_qt, T> ker(lambda,mu,s,id);
  return ker;
} 


	template<typename T, MATools::MAGPU::MEM_MODE MODE, MATools::MAGPU::GPU_TYPE GT>
bool run_test_elasticity()
{
	using namespace MATools::MAGPU;
	constexpr int size = 1000;

	MAGPUVector<T, MODE, GT> sig;
	MAGPUVector<T, MODE, GT> eto;
	MAGPUVector<T, MODE, GT> deto;
	T val = 13;
	constexpr int N = 3;
  constexpr auto s = tfel::math::StensorDimeToSize<N>::value;

	sig.init(val, s*size);
	eto.init(1.0, s*size);
	deto.init(2.0, s*size);

	auto kernel = initElasticity<3, false, T>() ;
	auto functor = Ker::create_functor<GT> (kernel, "elasticity");

	MAGPURunner<MODE, GT> runner;

	runner(functor, size, sig, eto, deto);	

	// check
	std::vector<T> host = sig.copy_to_vector();

	for(int id = 0 ; id < size ; id++)
	{
		if(host[id] == val)
		{
			std::cout << " error id: " << id << " host: " << host[id] << std::endl;  
			return EXIT_FAILURE;
		}
	}

	return EXIT_SUCCESS;
}

int main()
{
	bool success = EXIT_SUCCESS;
	using namespace MATools::MAGPU;
	success &= run_test_elasticity<double, MEM_MODE::CPU, GPU_TYPE::SERIAL>();
	success &= run_test_elasticity<double, MEM_MODE::GPU, GPU_TYPE::SERIAL>();
	if(success == EXIT_SUCCESS) std::cout << "Ok!" << std::endl;

	else std::cout << "Not ok!" << std::endl;
	return success;
}
