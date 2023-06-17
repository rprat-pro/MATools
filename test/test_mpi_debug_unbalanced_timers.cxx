#include <MATools.hxx>
#include <tclap/CmdLine.h>
#include "common.hxx"

int main(int argc, char * argv[]) 
{
	MATools::initialize(&argc,&argv);
	MATools::MATimer::Optional::disable_print_timetable();
	MATools::MATimer::Optional::disable_write_file();

	constexpr int ncase = 5;
	auto rank = MATools::MPI::get_rank();
	int tcase = rank % ncase; 
	constexpr int size = 5;
	launch<size>(tcase);

	MATools::MAOutputManager::write_debug_file();
	MATools::finalize();
	MATools::MPI::mpi_finalize();
	return 0;
}
