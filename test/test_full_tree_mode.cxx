#include <MATools.hxx>
#include "common.hxx"
#include <cassert>

int main(int argc, char * argv[]) 
{
#ifdef __MPI
	MATools::initialize(&argc,&argv, true);
	MATools::MATimer::Optional::active_full_tree_mode();

	constexpr int size = 1;
	int rank = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	assert(rank >=0);
	int tcase = rank % 5 + 1;
	launch<size>(tcase);
#else
	MATools::initialize();
	unsigned int tcase = 1;
	constexpr int size = 1;
	launch<size>(tcase);
#endif

	MATools::finalize();

	return 0;
}
