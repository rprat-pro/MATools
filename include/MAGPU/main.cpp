#include <iostream>
#include <MAGPUVector.hxx>

bool run_test_init_cuda_device_version()
{
  using namespace MATools::MAGPU;
  constexpr int N = 1000;
  MAGPUVector<double, MEM_MODE::GPU, GPU_TYPE::CUDA> vector_of_one;

  double value = 1;
  vector_of_one.init(value, N);

  // check
  std::vector<double> host = vector_of_one.copy_to_vector();

  for(int id = 0 ; id < N ; id++)
  {
    if(host[id] != value)
    {
      std::cout << "MATools_LOG: id = "<<id << " and host[id] = " << host[id] << std::endl;
      std::cout << "MATools_LOG: Error in init for MEM_MODE = GPU and GPU_TYPE = CUDA " << std::endl;
      return EXIT_FAILURE;
    }
  }

  std::cout << "MATools_LOG: Init works correctly for MEM_MODE = GPU and GPU_TYPE = CUDA " << std::endl;
  return EXIT_SUCCESS;
}

int main()
{
  bool success = EXIT_SUCCESS;
  success &= run_test_init_cuda_device_version();
  return success;
}
