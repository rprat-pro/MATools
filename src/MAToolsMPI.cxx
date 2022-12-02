#include <MAToolsMPI.hxx>
#include <iostream>

namespace MATools
{
	namespace MPI
	{
		constexpr int master=0;

		int get_rank()
		{
#ifdef __MPI
			int rank;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			return rank;
#else
			return master;
#endif
		}

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

		template<typename T>
		inline
		T reduce(T a_in, MPI_Op a_op)
		{
			std::cout << "error" << std::endl;
			std::abort();
			return -666;
		}
		
		template<>
		double reduce(double a_in, MPI_Op a_op)
		{
#ifdef __MPI
			double global(0.0);
                        MPI_Reduce(&a_in, &global, 1, MPI_DOUBLE, a_op, master, MPI_COMM_WORLD); // master rank is 0
			return global;
#else
			return a_in;
#endif
		}

		template<>
		int reduce(int a_in, MPI_Op a_op)
		{
#ifdef __MPI
			int global(0.0);
                        MPI_Reduce(&a_in, &global, 1, MPI_INT, a_op, master, MPI_COMM_WORLD); // master rank is 0
			return global;
#else
			return a_in;
#endif
		}

		double reduce_max(double a_duration)
		{
#ifdef __MPI
			double ret = reduce(a_duration, MPI_MAX);
			return ret;
#else
			return a_duration;
#endif
		}

	};
};
