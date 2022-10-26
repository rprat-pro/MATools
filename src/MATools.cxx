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

	void finalize(bool a_print_timetable, bool a_write_file, bool do_mpi_final)
	{
		MATrace::finalize();
		MATimer::finalize(a_print_timetable, a_write_file);
#ifdef __MPI
		if(do_mpi_final) MPI_Finalize();
#endif
	}
};
