#include <iostream>
#include <MAGPUTools.hxx>

#include <cstdlib>
#include "TFEL/Math/Array/View.hxx"
#include "TFEL/Math/qt.hxx"
#include "TFEL/Math/stensor.hxx"
#include "TFEL/Material/Lame.hxx"

/* special check */ 
template<typename Stensor>
	MAGPU_DECORATION
bool ex1_check(Stensor& a_stensor)
{
	if(a_stensor(0) != 1.125e+12) return true;
	if(a_stensor(1) != 1.125e+12) return true;
	if(a_stensor(2) != 1.125e+12) return true;
	if(std::abs( a_stensor(3) - 3.46154e+11) > 1e+6) return true;
	if(std::abs( a_stensor(4) - 3.46154e+11) > 1e+6) return true;
	if(std::abs( a_stensor(5) - 3.46154e+11) > 1e+6) return true;
	return false;	
}

template<typename Stensor>
void print_stensor(Stensor& a_stensor)
{
	for(int it = 0; it < 6 ; it++)
		std::cout << a_stensor(it) << " ";
	std::cout << std::endl;
}

template<typename Stensor, typename T>
DEFINE_KERNEL(behavior_law)(
		Stensor& a_sig, const Stensor& a_eto, const Stensor& a_deto, 
		const T& a_lambda, const T& a_mu, const unsigned short a_s, const Stensor& a_id, bool& a_error)
{
	const auto& e = a_eto + a_deto;
	a_sig = a_lambda * (e(0)+e(1)+e(2)) * a_id + 2 * a_mu * e;
	a_error = ex1_check(a_sig);
}
END_KERNEL(behavior_law)

	using namespace tfel::math;
	using namespace MATools::MAGPU;

	template<typename T, MATools::MAGPU::GPU_TYPE gpu_type>
bool run_example_1()
{

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
	using _vector  = MAGPUVector<stensor<N, T>, mem_cpu_and_gpu, gpu_type>;
	using _checker = MAGPUVector<bool, mem_cpu_and_gpu, gpu_type>;
	using mfront_vector = stensor<N, T>*;
	auto functor = create_functor<gpu_type> (behavior_law, "behavior_law");
	MAGPURunner<mem_gpu, gpu_type> runner;

	// init problem
	constexpr int size = 1000000;

	// stuff from other applications, /* host version */
	mfront_vector mfront_sig  = new stensor<N, T>[size];
	mfront_vector mfront_eto  = new stensor<N, T>[size];
	mfront_vector mfront_deto = new stensor<N, T>[size];

	for(int i = 0 ;  i < size ; i++)
	{
		mfront_sig[i]  = stensor<N, T>(666);
		mfront_eto[i]  = stensor<N, T>(1);
		mfront_deto[i] = stensor<N, T>(2);
	}

	// magpu 
	_vector sig;
	_vector eto;
	_vector deto;
	_checker error;

	// init problem
	error.init(false, size);

	// it creates an alias on the host ptr and copies data on the devicon the device
	sig.define_and_update(mem_cpu, mfront_sig, size);
	eto.define_and_update(mem_cpu, mfront_eto, size);
	deto.define_and_update(mem_cpu, mfront_deto, size);

	// run kernel
	runner.launcher_test(functor, size, sig, eto, deto, lambda, mu, s, id, error);	

	sig.update_host();
	error.update_host();

	// check1
	std::vector<stensor<N,T>> host = sig.copy_to_vector();
	for(int it = 0 ; it < host.size() ; it++)
		if(ex1_check(host[it]) == true) 
		{
			std::cout << "error check 1: device value  (" << it << ")=";
			print_stensor(host[it]);
			return false;
		}

	// check2
	bool* host_error_ptr = error.get_data(mem_cpu);
	std::cout << " host.size() = " << host.size() << std::endl;

	for(int it = 0 ; it < error.get_size() ; it++)
		if(host_error_ptr[it] == true)
		{
			std::cout << "error check 2: host value" << std::endl;
			return false;
		}


	return true;
}

int main()
{
	bool success = true;
	using namespace MATools::MAGPU;
	success &= run_example_1<double, GPU_TYPE::SERIAL>();
#ifdef __CUDA__
	success &= run_example_1<double, GPU_TYPE::CUDA>();
#endif
	if(success == true) std::cout << "Ok!" << std::endl;
	else std::cout << "Not ok!" << std::endl;
	return success;
}
