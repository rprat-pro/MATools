#include <MATools.hxx>
#include <tclap/CmdLine.h>
#include "common.hxx"

int main(int argc, char * argv[]) 
{
	using namespace MATools::MATimer;
	MATools::initialize(&argc,&argv);
	constexpr int robust = 1; 
	constexpr int tcase = 4;
	constexpr bool test_get_filtered_timers = false;
	constexpr bool test_print_filtered_timers = true;

	if(robust)
	{
		auto mpi_rank = MATools::MPI::get_rank();
		constexpr int size = 10;
		for(int i = 0 ; i < 10*(mpi_rank+1); i++)
		{
			int _case = tcase;
			while(_case >= 1)
			{
				launch<size>(tcase);
				_case--;
			}
		}
	}
	else
	{
		constexpr int size = 1;
		int _case = tcase;
		while(_case >= 1)
		{
			launch<size>(_case);
			_case--;
		}
	}

	// test
	if(test_get_filtered_timers)
	{
		using namespace MATools::MAOutputManager;
		auto vec_timers = get_filtered_timers("func1");
		if(((int)vec_timers.size()) != tcase)
		{
			using namespace MATools::MAOutput;
			printMessage("The corresct number of filterd timers should be :", tcase, "instead of", vec_timers.size());
			std::abort();
		}
	}


	if(test_print_filtered_timers)
	{
		using namespace MATools::MAOutputManager;
		print_filtered_timers("func1");
	}

	MATools::finalize();
	MATools::MPI::mpi_finalize();

	return EXIT_SUCCESS;
}
