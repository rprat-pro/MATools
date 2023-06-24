#include <vector>
#include <MATools.hxx>
#include <Common/MAMemory.hxx>
#ifdef __MPI
#include <mpi.h>
#endif

int main(int argc, char * argv[]) 
{
#ifdef __MPI
	MPI_Init(&argc,&argv);
#endif
	using namespace MATools::MAMemory;

	MAFootprint mem;
	
	for(int i = 0; i < 10 ; i++)
	{
		std::vector<char> vec(10000*i);
		mem.add_memory_checkpoint();
	}
	write_memory_checkpoints(mem);
#ifdef __MPI
	MPI_Finalize();
#endif
	return 0;
}
