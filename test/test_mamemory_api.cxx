#include <vector>
#include <MATools.hxx>
#include <MAToolsAPI/MAMemoryAPI.hxx>
#ifdef __MPI
#include <mpi.h>
#endif

int main(int argc, char * argv[]) 
{
#ifdef __MPI
	MPI_Init(&argc,&argv);
#endif
	using namespace MATools::MAMemory;

	MAMemoryManager mem_manager;
	
	for(int i = 0; i < 100 ; i++)
	{
		std::vector<char> vec(10000*i);
		Add_Mem_Point();
	}
	mem_manager.write_trace_memory_footprint();
	return 0;
}
