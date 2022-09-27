#include <timer.hxx>
#include <tclap/CmdLine.h>

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


int main(int argc, char const* argv[]) 
{
	int robust = -1; 
	int tcase = -1;

	try {
		TCLAP::CmdLine cmd("timers tests", ' ', "1.0");
		TCLAP::ValueArg<int> _tcase("c", "case", "test case", false, -2, "int");
		TCLAP::ValueArg<int> _robust("r", "robustness", "test of robustess", false, -2, "int");


		cmd.add(_tcase);
		cmd.add(_robust);

		cmd.parse(argc, argv);

		tcase 	= _tcase.getValue();
		robust 	= _robust.getValue();
	} 
	catch (TCLAP::ArgException& e) 
	{
		std::cerr << "error: " << e.error() << " for argument " << e.argId() << std::endl;
	}

	assert(	
		robust >= 0 
		&& robust < 2 
		&& "bad choice for robust parameter"
	);

	assert(
		tcase >= 0 
		&& tcase < 6 
		&& "bad choice for tcase parameter"
	);

	MATimer::timers::init_timers();

	if(robust)
	{
		constexpr int size = 100;
		launch<size>(tcase);
	}
	else
	{
		constexpr int size = 1;
		launch<size>(tcase);
	}

	MATimer::timers::print_and_write_timers();

	return 0;
}
