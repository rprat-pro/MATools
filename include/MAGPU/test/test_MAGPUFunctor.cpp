
#pragma once
#include <MAGPUBasicFunctors.hxx>
#include <MAGPUFunctor.hxx>
#include <test_helper.hpp>

	template<typename T, MATools::MAGPU::GPU_TYPE GT>
bool run_test_create_functor()
{
	using namespace MATools::MAGPU;
	constexpr auto my_func = Ker::reset<T>;
	auto functor = Ker::create_functor<GT> (my_func, "reset"); 


	if(functor.get_name() != "reset") return EXIT_FAILURE;
	return EXIT_SUCCESS;
}

	template<typename T, MATools::MAGPU::GPU_TYPE GT>
bool run_test_functor_empty()
{
	using namespace MATools::MAGPU;
	constexpr auto empty_function = [](unsigned int idx) {} ;
	MAGPUFunctor<decltype(empty_function), MATools::MAGPU::GPU_TYPE::SERIAL> my_functor(empty_function);
	my_functor(123456789);
	if(my_functor.get_name() != "default_name") return EXIT_FAILURE;
	return EXIT_SUCCESS;
}

	template<typename T, MATools::MAGPU::MEM_MODE MM, MATools::MAGPU::GPU_TYPE GT>
bool run_test_functor_add()
{
	using namespace MATools::MAGPU;
	constexpr auto my_func = Ker::add<T>;
	auto functor = Ker::create_functor<GT> (my_func, "add"); 

	if(MM == MATools::MAGPU::MEM_MODE::BOTH)
	{
	}
	else
	{
		test_helper::create<T,MM,GT> allocator;
		test_helper::destroy<T,MM,GT> destructor;
		test_helper::copier<T,MM,GT> _copy ;
		T* two = allocator(2,1);
		T* res = allocator(1,1);

		assert(two != nullptr);
		assert(res != nullptr);

		test_helper::mini_runner<MM,GT> launcher;

		int idx = 0;
	  launcher(functor, idx, res, two);

		std::vector<T> host(1,0);

		_copy(host.data(), res, 1);

		if(host[0] != 3)
		{
			std::cout << " error, host should be equal to 3, host = " << host[0] << std::endl;
			destructor(two);
			destructor(res);
			return EXIT_FAILURE;
		}

		destructor(two);
		destructor(res);
	}
	return EXIT_SUCCESS;
}

TYPE_GPU_TEST_CASE(create_functor);
TYPE_GPU_TEST_CASE(functor_empty);
SUPER_TEST_CASE(functor_add);
