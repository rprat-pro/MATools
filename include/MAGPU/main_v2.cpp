#include <iostream>
#include <MAGPUVector_V2.hxx>

int main()
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
			std::cout << " Error in init for MEM_MODE = CPU and GPU_TYPE = SERIAL " << std::endl;
			return EXIT_FAILURE;
		}
	}

	std::cout << " Init function works correctly for MEM_MODE = CPU and GPU_TYPE = SERIAL " << std::endl;
	return EXIT_SUCCESS;
}
