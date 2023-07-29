#include <vector>
#include <MATools.hxx>
#ifdef __MPI
#include <mpi.h>
#endif

using namespace MATools::MATimer;

int main(int argc, char * argv[]) 
{
	MATools::initialize(&argc,&argv);
	HybridTimer first_timer("hybrid_timer");
	first_timer.start_time_section();
	{
		Catch_Nested_Time_Section("inside_hybrid_timer");
	}
	first_timer.end_time_section();
	MATools::finalize();
	MATools::MPI::mpi_finalize();
	return 0;
}
