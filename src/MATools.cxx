#include <MATimers/MATimers.hxx>
#include <MATrace/MATrace.hxx>

namespace MATools
{
	void initialize()
	{
		MATimer::initialize();
		MATrace::initialize();
	}

	void initialize([[maybe_unused]] int *argc, [[maybe_unused]]char ***argv)
	{
		MATools::MPI::mpi_initialize(argc, argv);
		initialize();
	}

	void finalize()
	{
		MATimer::finalize();
		MATrace::finalize();
//		MATools::MPI::mpi_finalize();
	}
};
