#include <iostream>
#include <MAGPUVector.hxx>


bool run_test_init_serial_host_version()
{
	using namespace MATools::MAGPU;
	constexpr int N = 1000;
	MAGPUVector<double,MEM_MODE::CPU, GPU_TYPE::SERIAL> vector_of_one;

	double value = 1;
	vector_of_one.init(value,N);

	// check
	auto ptr = vector_of_one.get_data();
	for(int id = 0 ; id < N ; id++)
	{
		if(ptr[id] != value)
		{
			std::cout << "MATools_LOG: Error in init for MEM_MODE = CPU and GPU_TYPE = SERIAL " << std::endl;
			return EXIT_FAILURE;
		}
	}

	std::cout << "MATools_LOG: Init works correctly for MEM_MODE = CPU and GPU_TYPE = SERIAL " << std::endl;
	return EXIT_SUCCESS;
}

bool run_test_resize_serial_host_version()
{
	using namespace MATools::MAGPU;
	constexpr int N = 1000;
	MAGPUVector<double,MEM_MODE::CPU, GPU_TYPE::SERIAL> vector_of_one;
	vector_of_one.resize(N);

	// check
	auto ptr = vector_of_one.get_data();
	if(ptr == nullptr)
	{
		std::cout << "MATools_LOG: Error in resize for MEM_MODE = CPU and GPU_TYPE = SERIAL " << std::endl;
		return EXIT_FAILURE;
	}

	std::cout << "MATools_LOG: resize works correctly for MEM_MODE = CPU and GPU_TYPE = SERIAL " << std::endl;
	return EXIT_SUCCESS;
}

bool run_test_aliasing_serial_host_version()
{
	using namespace MATools::MAGPU;
	constexpr int N = 1000;
	std::vector<double> reg(N,0);
	MAGPUVector<double,MEM_MODE::CPU, GPU_TYPE::SERIAL> vec;

	vec.aliasing(reg.data(), reg.size());

	// check
	auto p = vec.get_data();
	if(p == nullptr)
	{
		std::cout << "MATools_LOG: Error in aliasing for MEM_MODE = CPU and GPU_TYPE = SERIAL " << std::endl;
		return EXIT_FAILURE;
	}


	std::cout << "MATools_LOG: Aliasing works correctly for MEM_MODE = CPU and GPU_TYPE = SERIAL " << std::endl;
	return EXIT_SUCCESS;
}

bool run_test_get_size_serial_host_version()
{
	using namespace MATools::MAGPU;
	constexpr int N = 1000;
	MAGPUVector<double,MEM_MODE::CPU, GPU_TYPE::SERIAL> vec;
	vec.resize(N);

	// check
	auto size = vec.get_size();
	if(size == N)
	{
		std::cout << "MATools_LOG: get_size works correctly for MEM_MODE = CPU and GPU_TYPE = SERIAL " << std::endl;
		return EXIT_SUCCESS;
	}

	std::cout << "MATools_LOG: Error in get_size for MEM_MODE = CPU and GPU_TYPE = SERIAL " << std::endl;
	return EXIT_FAILURE;
}

int main()
{
	bool success = EXIT_SUCCESS;

	success &= run_test_init_serial_host_version();
	success &= run_test_resize_serial_host_version();
	success &= run_test_aliasing_serial_host_version();
	success &= run_test_get_size_serial_host_version();
	return success;
}
