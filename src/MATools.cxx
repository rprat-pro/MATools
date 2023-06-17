#include <MATimers/MATimers.hxx>
#include <MATrace/MATrace.hxx>

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
		using namespace MPI;
		if(do_mpi_init)
		{
			const auto is_init = check_mpi_initialied(); 			
			if(!is_init)
			{
				MPI_Init(argc,argv);
			}
			else
			{
				printMessage("MPI_Init has already be called, this step is skipped");
			}
		}
#endif /* __MPI */
		initialize();
	}

	void finalize([[maybe_unused]]bool do_mpi_final)
	{
		MATimer::finalize();
		MATrace::finalize();
#ifdef __MPI
		using namespace MPI;
		if(do_mpi_final) 
		{
			const bool is_finalized = check_mpi_finalized();
			if(!is_finalized)
			{
				using namespace MAOutput;
				MPI_Finalize();
			}
			else
			{
				printMessage("MPI_Finalize has already be called, this step is skipped");
			}
		}
#endif /* __MPI */
	}
};
