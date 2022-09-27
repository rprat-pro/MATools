#pragma once

	template<int N>
void func1()
{
	START_TIMER("func1");
}

	template<int N>
void func2()
{
	START_TIMER("func2");
	for(int i = 0 ; i< 2*N; i++)
		func1<N>();
}

	template<int N>
void func3()
{
	START_TIMER("func3");
	for(int i = 0 ; i< 3*N; i++)
		func2<N>();
}

	template<int N>
void func4()
{
	START_TIMER("func4");
	for(int i = 0 ; i< 4*N; i++)
		func3<N>();
}


	template<int N>
void func5()
{
	START_TIMER("func5");
	for(int i = 0 ; i< 5*N; i++)
		func4<N>();
}

template<int N>
void launch(int _case)
{
	switch(_case)
	{
		case 0:
			break;
		case 1:
			func1<N>();
			break;
		case 2:
			func2<N>();
			break;
		case 3:
			func3<N>();
			break;
		case 4:
			func4<N>();
			break;
		case 5:
			func5<N>();
			break;	
		default:
			break;
	}
}

