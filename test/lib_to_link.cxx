#pragma once
#include <MATimers.hxx>
#include <lib_to_link.hxx>

	template<int N>
void func_in_other_lib1()
{
	START_TIMER("func_in_other_lib1");
}

	template<int N>
void func_in_other_lib2()
{
	START_TIMER("func_in_other_lib2");
	for(int i = 0 ; i< 2*N; i++)
		func_in_other_lib1<N>();
}

	template<int N>
void func_in_other_lib3()
{
	START_TIMER("func_in_other_lib3");
	for(int i = 0 ; i< 3*N; i++)
		func_in_other_lib2<N>();
}

	template<int N>
void func_in_other_lib4()
{
	START_TIMER("func_in_other_lib4");
	for(int i = 0 ; i< 4*N; i++)
		func_in_other_lib3<N>();
}


	template<int N>
void func_in_other_lib5()
{
	START_TIMER("func_in_other_lib5");
	for(int i = 0 ; i< 5*N; i++)
		func_in_other_lib4<N>();
}
template<int N>
void launch_func_in_other_lib_other_lib(int _case)
{
	switch(_case)
	{
		case 0:
			break;
		case 1:
			func_in_other_lib1<N>();
			break;
		case 2:
			func_in_other_lib2<N>();
			break;
		case 3:
			func_in_other_lib3<N>();
			break;
		case 4:
			func_in_other_lib4<N>();
			break;
		case 5:
			func_in_other_lib5<N>();
			break;	
		default:
			break;
	}
}

