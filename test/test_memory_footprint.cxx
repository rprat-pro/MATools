#include <vector>
#include <MATools.hxx>
#include <MAMemory.hxx>
#ifdef __MPI
#include <mpi.h>
#endif

int main(int argc, char * argv[]) 
{
#ifdef __MPI
	MPI_Init(&argc,&argv);
#endif
	using namespace MATools::MAMemory;
	std::vector<char> vec(1E6);
	print_memory_footprint();
#ifdef __MPI
	MPI_Finalize();
#endif
	return 0;
}
