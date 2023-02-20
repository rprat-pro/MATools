#include <iostream>
#include <MAGPUVector.hxx>
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp fil
//#include "catch2/catch_all.hpp"
#include "catch2/catch.hpp"

/*
	template<typename T, MATools::MAGPU::MEM_MODE MM, MATools::MAGPU::GPU_TYPE GT>
bool run_test_default_constructor()
{
	using namespace MATools::MAGPU;
	MAGPUVector<T, MM, GT> vec();

	if(MM == MATools::MAGPU::MEM_MODE::BOTH)
	{
		auto size = vec.get_size();
		auto host = vec.get_data(0);
		auto devi = vec.get_data(1);

		if(size == 0) return EXIT_FAILURE;
		if(host == nullptr) return EXIT_FAILURE;
		if(devi == nullptr) return EXIT_FAILURE;
	}
	else
	{
		auto size = vec.get_size();
		auto ptr = vec.get_data();
		if(size == 0) return EXIT_FAILURE;
		if(ptr == nullptr) return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
*/

	template<typename T, MATools::MAGPU::MEM_MODE MM, MATools::MAGPU::GPU_TYPE GT>
bool run_test_init()
{
	using namespace MATools::MAGPU;
	constexpr int N = 1000;
	MAGPUVector<T, MM, GT> vector_of_one;

	T value = 1;
	vector_of_one.init(value,N);


	// check
	auto vec = vector_of_one.copy_to_vector();
	for(int id = 0 ; id < N ; id++)
	{
		if(vec[id] != value)
		{
			return EXIT_FAILURE;
		}
	}

	if(MM == MATools::MAGPU::MEM_MODE::BOTH)
	{
		auto ptr = vector_of_one.get_data(0);
		for(int id = 0 ; id < N ; id++)
		{
			if(vec[id] != ptr[id])
			{
				return EXIT_FAILURE;
			}
		}
	}

	return EXIT_SUCCESS;
}



	template<typename T, MATools::MAGPU::MEM_MODE MM, MATools::MAGPU::GPU_TYPE GT>
bool run_test_resize()
{
	using namespace MATools::MAGPU;
	constexpr int N = 1000;
	MAGPUVector<T, MM, GT> vector_of_one;
	vector_of_one.resize(N);

	// check
	if(MM == MATools::MAGPU::MEM_MODE::BOTH)
	{
		auto host_ptr = vector_of_one.get_data(0);
		auto devi_ptr = vector_of_one.get_data(1);
		if(host_ptr == nullptr || devi_ptr == nullptr) EXIT_FAILURE;
		if(host_ptr==devi_ptr) return EXIT_FAILURE;
	}
	else
	{
		auto ptr = vector_of_one.get_data();
		if(ptr == nullptr)
		{
			return EXIT_FAILURE;
		}
	}

	return EXIT_SUCCESS;
}

	template<typename T, MATools::MAGPU::MEM_MODE MM, MATools::MAGPU::GPU_TYPE GT>
bool run_test_fill()
{
	using namespace MATools::MAGPU;
	constexpr int N = 1000;
	MAGPUVector<T,MM,GT> vector_of_one;
	vector_of_one.resize(N);

	const T val = 1.5;
	vector_of_one.fill(val);

	// check
	auto vec = vector_of_one.copy_to_vector();
	for(int id = 0 ; id < N ; id++)
	{
		if(vec[id] != val)
		{
			return EXIT_FAILURE;
		}
	}

	if(MM == MATools::MAGPU::MEM_MODE::BOTH)
	{
		auto ptr = vector_of_one.get_data(0);
		for(int id = 0 ; id < N ; id++)
		{
			if(vec[id] != ptr[id])
			{
				return EXIT_FAILURE;
			}
		}
	}
	return EXIT_SUCCESS;
}

	template<typename T, MATools::MAGPU::MEM_MODE MM, MATools::MAGPU::GPU_TYPE GT>
bool run_test_operator_equal_T()
{
	using namespace MATools::MAGPU;
	constexpr int N = 1000;
	MAGPUVector<T, MM, GT> vector_of_one;
	vector_of_one.resize(N);

	const double val = 33;
	vector_of_one = val;

	// check
	auto vec = vector_of_one.copy_to_vector();
	for(int id = 0 ; id < N ; id++)
	{
		if(vec[id] != val)
		{
			return EXIT_FAILURE;
		}
	}

	if(MM == MATools::MAGPU::MEM_MODE::BOTH)
	{
		auto ptr = vector_of_one.get_data(0);
		for(int id = 0 ; id < N ; id++)
		{
			if(vec[id] != ptr[id])
			{
				return EXIT_FAILURE;
			}
		}
	}
	return EXIT_SUCCESS;
}


namespace test_helper
{
	template<typename T, MATools::MAGPU::MEM_MODE MM, MATools::MAGPU::GPU_TYPE GT>
		struct create
		{
			T* operator()(unsigned int N);
		};

	template<typename T, MATools::MAGPU::GPU_TYPE GT>
		struct create<T, MATools::MAGPU::MEM_MODE::CPU, GT>
		{
			T* operator()(unsigned int N)
			{
				T* ret;
				ret = new T[N];
				return ret;
			}
		};

	template<typename T>
		struct create<T, MATools::MAGPU::MEM_MODE::GPU, MATools::MAGPU::GPU_TYPE::SERIAL>
		{
			T* operator()(unsigned int N)
			{
				T* ret;
				ret = new T[N];
				return ret;
			}
		};

	template<typename T>
		struct create<T, MATools::MAGPU::MEM_MODE::GPU, MATools::MAGPU::GPU_TYPE::CUDA>
		{
			T* operator()(unsigned int N)
			{
				T* ret;
				cudaMalloc(&ret,N*sizeof(T));
				return ret;
			}
		};


	template<typename T, MATools::MAGPU::MEM_MODE MM, MATools::MAGPU::GPU_TYPE GT>
		struct destroy
		{
			void operator()(T* ret)
			{
				delete ret;
			}
		};

	template<typename T>
		struct destroy<T, MATools::MAGPU::MEM_MODE::GPU, MATools::MAGPU::GPU_TYPE::CUDA>
		{
			void operator()(T* ret)
			{
				cudaFree(ret);
			}
		};
}

	template<typename T, MATools::MAGPU::MEM_MODE MM, MATools::MAGPU::GPU_TYPE GT>
bool run_test_aliasing()
{
	using namespace MATools::MAGPU;
	constexpr int N = 1000;
	MAGPUVector<T, MM, GT> vec;

	if(MM == MATools::MAGPU::MEM_MODE::BOTH)
	{
		test_helper::create<T,MATools::MAGPU::MEM_MODE::CPU,GT> allocator_h;
		test_helper::create<T,MATools::MAGPU::MEM_MODE::GPU,GT> allocator_d;
		test_helper::destroy<T,MATools::MAGPU::MEM_MODE::CPU,GT> destructor_h;
		test_helper::destroy<T,MATools::MAGPU::MEM_MODE::GPU,GT> destructor_d;

		T* host = allocator_h(N);
		T* devi = allocator_d(N);

		vec.aliasing(host,devi,N);

		// check
		auto host_ptr = vec.get_data(0);
		auto devi_ptr = vec.get_data(1);

		assert(host_ptr == host);
		assert(devi_ptr == devi);
		assert(host_ptr != devi_ptr);

		if(host_ptr != host) return EXIT_FAILURE;
		if(devi_ptr != devi) return EXIT_FAILURE;
		if(host_ptr==devi_ptr) return EXIT_FAILURE;
		destructor_h(host);
		destructor_d(devi);
	}
	else
	{
		test_helper::create<T,MM,GT> allocator;
		test_helper::destroy<T,MM,GT> destructor;
		T* reg = allocator(N);
		vec.aliasing(reg, N);

		// check
		auto p = vec.get_data();
		if(p != reg)
		{
			destructor(reg);
			return EXIT_FAILURE;
		}
		destructor(reg);
	}
	return EXIT_SUCCESS;
}

	template<typename T, MATools::MAGPU::MEM_MODE MM, MATools::MAGPU::GPU_TYPE GT>
bool run_test_get_size()
{
	using namespace MATools::MAGPU;
	constexpr int N = 1000;
	MAGPUVector<T,MM,GT> vec;
	vec.resize(N);

	// check
	auto size = vec.get_size();
	if(size == N)
	{
		return EXIT_SUCCESS;
	}

	return EXIT_FAILURE;
}





#define PRINT(X) #X

#define MY_TEST_CASE(NAME,TYPE,MEMORY,GPUTYPE) TEST_CASE( PRINT(test_[NAME]_[TYPE]_[MEMORY]_[GPUTYPE]),"[MAGPU]") \
{\
	using namespace MATools::MAGPU;\
	bool success = run_test_##NAME <TYPE, MEMORY, GPUTYPE>();\
	REQUIRE(success ==  EXIT_SUCCESS);\
};\


#define serial_TESTS(NAME,TYPE,MEMORY) MY_TEST_CASE(NAME, TYPE, MEMORY, MATools::MAGPU::GPU_TYPE::SERIAL)

#ifdef __CUDA__
#define cuda_TESTS(NAME,TYPE,MEMORY) MY_TEST_CASE(NAME, TYPE, MEMORY, MATools::MAGPU::GPU_TYPE::CUDA)
#else
#define cuda_TESTS(NAME,TYPE,MEMORY)
#endif

#ifdef __KOKKOS__
#define kokkos_TESTS(NAME,TYPE,MEMORY) MY_TEST_CASE(NAME, TYPE, MEMORY, MATools::MAGPU::GPU_TYPE::KOKKOS)
#else
#define kokkos_TESTS(NAME,TYPE,MEMORY)
#endif

#define GPU_TEST_CASE(NAME,TYPE,MEMORY) serial_TESTS(NAME,TYPE,MEMORY)\
		cuda_TESTS(NAME,TYPE,MEMORY)\
	kokkos_TESTS(NAME,TYPE,MEMORY)

#define MEM_TEST_CASE(NAME,TYPE) \
		GPU_TEST_CASE(NAME, TYPE, MATools::MAGPU::MEM_MODE::CPU)\
	GPU_TEST_CASE(NAME, TYPE, MATools::MAGPU::MEM_MODE::GPU)\
	GPU_TEST_CASE(NAME, TYPE, MATools::MAGPU::MEM_MODE::BOTH)


#define TYPE_TEST_CASE(NAME) \
		MEM_TEST_CASE(NAME,int)\
	MEM_TEST_CASE(NAME,float)\
	MEM_TEST_CASE(NAME,double)

#define SUPER_TEST_CASE(X) TYPE_TEST_CASE(X)


	// Test case is a single test that you run
	// You give it a name/description and also you give it some tags.
	TEST_CASE("Testing framework is working fine", "[Catch2]")
{

	// Tests have to meet some requirements to be considered valid
	REQUIRE(true);
}

SUPER_TEST_CASE(init);
SUPER_TEST_CASE(resize);
SUPER_TEST_CASE(fill);
SUPER_TEST_CASE(operator_equal_T);
SUPER_TEST_CASE(aliasing);
SUPER_TEST_CASE(get_size);

