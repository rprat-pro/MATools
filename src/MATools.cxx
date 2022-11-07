#include <MATimers.hxx>
#include <MATrace.hxx>

namespace MATools
{
	void initialize()
	{
		MATimer::initialize();
		MATrace::initialize();
	}

	void initialize([[maybe_unused]] int *argc, [[maybe_unused]]char ***argv, [[maybe_unused]] bool do_mpi_init)
	{
#ifdef __MPI
		if(do_mpi_init) MPI_Init(argc,argv);
#endif
		initialize();
	}

	void finalize([[maybe_unused]]bool do_mpi_final)
	{
		MATimer::finalize();
		MATrace::finalize();
#ifdef __MPI
		if(do_mpi_final) MPI_Finalize();
#endif
	}
};
