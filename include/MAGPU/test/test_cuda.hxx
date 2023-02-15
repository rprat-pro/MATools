#include <iostream>
#include <MAGPUVector.hxx>
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp fil
//#include "catch2/catch_all.hpp"
#include "catch2/catch.hpp"
 
bool run_test_init_cuda_device_version()
{
	using namespace MATools::MAGPU;
	constexpr int N = 1000;
	MAGPUVector<double,MEM_MODE::GPU, GPU_TYPE::CUDA> vector_of_one;

	double value = 1;
	vector_of_one.init(value,N);

	// check
	auto ptr = vector_of_one.get_data();

	std::vector<double> host(N, 0);
	cudaMemcpy(host.data(), ptr, N*sizeof(double), cudaMemcpyDeviceToHost);

	for(int id = 0 ; id < N ; id++)
	{
		if(host[id] != value)
		{
			std::cout << "MATools_LOG: Error in init for MEM_MODE = GPU and GPU_TYPE = CUDA " << std::endl;
			return EXIT_FAILURE;
		}
	}

	std::cout << "MATools_LOG: Init works correctly for MEM_MODE = GPU and GPU_TYPE = CUDA " << std::endl;
	return EXIT_SUCCESS;
}


bool run_test_resize_cuda_device_version()
{
	using namespace MATools::MAGPU;
	constexpr int N = 1000;
	MAGPUVector<double,MEM_MODE::GPU, GPU_TYPE::CUDA> vector_of_one;
	vector_of_one.resize(N);

	// check
	auto ptr = vector_of_one.get_data();
	if(ptr == nullptr)
	{
		std::cout << "MATools_LOG: Error in resize for MEM_MODE = GPU and GPU_TYPE = CUDA " << std::endl;
		return EXIT_FAILURE;
	}

	std::cout << "MATools_LOG: resize works correctly for MEM_MODE = GPU and GPU_TYPE = CUDA " << std::endl;
	return EXIT_SUCCESS;
}

bool run_test_aliasing_cuda_device_version()
{
	using namespace MATools::MAGPU;
	constexpr int N = 1000;
	std::vector<double> reg(N,0);
	MAGPUVector<double,MEM_MODE::GPU, GPU_TYPE::CUDA> vec;

	vec.aliasing(reg.data(), reg.size());

	// check
	auto p = vec.get_data();
	if(p == nullptr)
	{
		std::cout << "MATools_LOG: Error in aliasing for MEM_MODE = GPU and GPU_TYPE = CUDA " << std::endl;
		return EXIT_FAILURE;
	}


	std::cout << "MATools_LOG: Aliasing works correctly for MEM_MODE = GPU and GPU_TYPE = CUDA " << std::endl;
	return EXIT_SUCCESS;
}

bool run_test_get_size_cuda_device_version()
{
	using namespace MATools::MAGPU;
	constexpr int N = 1000;
	MAGPUVector<double,MEM_MODE::GPU, GPU_TYPE::CUDA> vec;
	vec.resize(N);

	// check
	auto size = vec.get_size();
	if(size == N)
	{
		std::cout << "MATools_LOG: get_size works correctly for MEM_MODE = GPU and GPU_TYPE = CUDA " << std::endl;
		return EXIT_SUCCESS;
	}

	std::cout << "MATools_LOG: Error in get_size for MEM_MODE = GPU and GPU_TYPE = CUDA " << std::endl;
	return EXIT_FAILURE;
}


TEST_CASE("run_test_init_cuda_device_version", "[MAGPU]")
{
	auto success = run_test_init_cuda_device_version();
	REQUIRE(success == EXIT_SUCCESS);
}

TEST_CASE("run_test_resize_cuda_device_version", "[MAGPU]")
{
	auto success = run_test_resize_cuda_device_version();
	REQUIRE(success == EXIT_SUCCESS);
}

TEST_CASE("run_test_aliasing_cuda_device_version", "[MAGPU]")
{
	auto success = run_test_aliasing_cuda_device_version();
	REQUIRE(success == EXIT_SUCCESS);
}

TEST_CASE("run_test_get_size_cuda_device_version", "[MAGPU]")
{
	auto success = run_test_get_size_cuda_device_version();
	REQUIRE(success == EXIT_SUCCESS);
}
/*
int main()
{
	return EXIT_SUCCESS;
}*/
