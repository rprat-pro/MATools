#include <iostream>
#include <MAGPUVector.hxx>
#include <MAGPUFunctor.hxx>
#include <MAGPURunner.hxx>

	template<typename T, MATools::MAGPU::MEM_MODE MODE, MATools::MAGPU::GPU_TYPE GT>
bool run_test_runner()
{
	using namespace MATools::MAGPU;
	constexpr int N = 1000;

	MAGPUVector<T, MODE, GT> res;
	MAGPUVector<T, MODE, GT> vec1;
	MAGPUVector<T, MODE, GT> vec2;
	T val = 13;

	res.init(0.0, N);
	vec1.init(1.0, N);
	vec2.init(2.0, N);

	auto add_kernel = [](unsigned int idx, T* const out, T* const in1, T* const in2, T val)
	{
		out[idx] = in1[idx] + in2[idx] + val;
	};

	MAGPUFunctor<decltype(add_kernel), GPU_TYPE::SERIAL> fun(add_kernel, "add");
	MAGPURunner<MODE, GT> runner;
	runner(fun, N, res, vec1, vec2, val);	

	// check
	std::vector<T> host = res.copy_to_vector();

	for(int id = 0 ; id < N ; id++)
	{
		if(host[id] != 3.0 + val)
		{
			std::cout << "MATools_LOG: id = "<<id << " and host[id] = " << host[id] << std::endl;
			std::cout << "MATools_LOG: Error in test runner " << std::endl;
			return EXIT_FAILURE;
		}
	}

	std::cout << "MATools_LOG: test runner works correctly " << std::endl;
	return EXIT_SUCCESS;
}

int main()
{
	bool success = EXIT_SUCCESS;
	using namespace MATools::MAGPU;
	success &= run_test_runner<double, MEM_MODE::CPU, GPU_TYPE::SERIAL>();
	success &= run_test_runner<double, MEM_MODE::GPU, GPU_TYPE::SERIAL>();
	return success;
}
