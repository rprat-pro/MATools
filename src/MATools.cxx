#include <MATimers.hxx>

namespace MATools
{
	void initialize()
	{
		MATimer::initialize();
		MATrace::initialize();
	}

	void initialize(int *argc, char ***argv, bool do_mpi_init)
	{
#ifdef __MPI
		if(do_mpi_init) MPI_Init(argc,argv);
#endif
		initialize();
	}

	void finalize(bool do_mpi_final)
	{
		MATrace::finalize();
		MATimer::finalize();
#ifdef __MPI
		if(do_mpi_final) MPI_Finalize();
#endif
	}
};
