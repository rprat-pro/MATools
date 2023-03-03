#include <iostream>
#include <MAGPUVector.hxx>
#include <MAGPUFunctor.hxx>
#include <MAGPURunner.hxx>

	template<typename T, MATools::MAGPU::MEM_MODE MODE, MATools::MAGPU::GPU_TYPE GT>
bool run_test_runner_with_MAGPUVector_MAGPUFunctor()
{
	using namespace MATools::MAGPU;
	constexpr int n = 1000;

	MAGPUVector<T, MODE, GT> res;
	MAGPUVector<T, MODE, GT> vec1;
	MAGPUVector<T, MODE, GT> vec2;
	T val  = 13;

	res.init(0.0, n);
	vec1.init(1.0, n);
	vec2.init(2.0, n);

	auto add_kernel = [] __host__ __device__ (unsigned int idx, T* const out, T* const in1, T* const in2, T val ) -> void
	{
		out[idx] = in1[idx] + in2[idx] + val;
	};

	if constexpr (MODE != MEM_MODE::BOTH)
	{
		MAGPUFunctor<decltype(add_kernel), GT> fun(add_kernel, "add");
		MAGPURunner<MODE, GT> runner;
		runner(fun, n, res, vec1, vec2, val);	

		// check
		std::vector<T> host = res.copy_to_vector();

		for(int id = 0 ; id < n ; id++)
		{
			if(host[id] != 3.0 + val)
				return EXIT_FAILURE;
		}
	}
	else
	{
		constexpr auto mc = MEM_MODE::CPU;
		constexpr auto mg = MEM_MODE::GPU;
		MAGPUFunctor<decltype(add_kernel), GT> fun(add_kernel, "add");
		MAGPURunner<mc, GT> host_runner;
		MAGPURunner<mg, GT> devi_runner;

		devi_runner(fun, n, res, vec1, vec2, val);	
		host_runner(fun, n, res, vec1, vec2, val);	

		auto devi_res = res.copy_to_vector(); // copy data from the device memory		
		auto host_res = res.copy_to_vector_from_host();

		for(int id = 0 ; id < n ; id++)
		{
			if(host_res[id] != 3.0 + val)
				return EXIT_FAILURE;
			if(devi_res[id] != 3.0 + val)
				return EXIT_FAILURE;
		}
		
	}
	return EXIT_SUCCESS;
}


	template<typename T, MATools::MAGPU::MEM_MODE MODE, MATools::MAGPU::GPU_TYPE GT>
bool run_test_runner_full_test()
{
	using namespace MATools::MAGPU;
	constexpr int n = 1000;

	MAGPUVector<T, MODE, GT> res;
	MAGPUVector<T, MODE, GT> vec1;
	MAGPUVector<T, MODE, GT> vec2;
	T val  = 13;

	res.init(0.0, n);
	vec2.init(2.0, n);

	auto my_kernel = create_functor<GT>(Ker::add_sub_mult_divF, "full"); 


	if constexpr (MODE != MEM_MODE::BOTH)
	{
		MAGPURunner<MODE, GT> runner;
		runner(my_kernel, n, res, vec2);	

		// check
		std::vector<T> host = res.copy_to_vector();

		for(int id = 0 ; id < n ; id++)
		{
			if(host[id] != 3.0 + val)
				return EXIT_FAILURE;
		}
	}
	else
	{
		constexpr auto mc = MEM_MODE::CPU;
		constexpr auto mg = MEM_MODE::GPU;
		MAGPUFunctor<decltype(my_kernel), GT> fun(my_kernel, "add");
		MAGPURunner<mc, GT> host_runner;
		MAGPURunner<mg, GT> devi_runner;

		devi_runner(fun, n, res, vec1, vec2, val);	
		host_runner(fun, n, res, vec1, vec2, val);	

		auto devi_res = res.copy_to_vector(); // copy data from the device memory		
		auto host_res = res.copy_to_vector_from_host();

		for(int id = 0 ; id < n ; id++)
		{
			if(host_res[id] != 3.0 + val)
				return EXIT_FAILURE;
			if(devi_res[id] != 3.0 + val)
				return EXIT_FAILURE;
		}
		
	}
	return EXIT_SUCCESS;
}


SUPER_TEST_CASE(runner_with_MAGPUVector_MAGPUFunctor);

