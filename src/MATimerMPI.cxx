#include<MATimerMPI.hxx>


namespace MATimer
{
	namespace mpi
	{
		constexpr int master=0;

		bool is_master()
		{
#ifdef __MPI
			int rank;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			return (rank == master);
#else
			return true;
#endif
		}

		double reduce_max(double a_duration)
		{
#ifdef __MPI
			int size = -1;

			double global = 0.0;
                        MPI_Reduce(&a_duration, &global, 1, MPI_DOUBLE, MPI_MAX, master, MPI_COMM_WORLD); // master rank is 0
			return global;
#else
			return a_duration;
#endif
		}

	};
};
