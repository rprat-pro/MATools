#include <MATools.hxx>
#include <tclap/CmdLine.h>
#include "common.hxx"

int main(int argc, char * argv[]) 
{
	MATools::initialize(&argc,&argv);


	constexpr int tcase = 4; 
	constexpr int size = 5;
	launch<size>(tcase);

	MATools::MAOutputManager::write_debug_file();
	MATools::finalize();
	MATools::MPI::mpi_finalize();
	return 0;
}
