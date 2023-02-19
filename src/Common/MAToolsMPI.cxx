#include <Common/MAToolsMPI.hxx>
#include <iostream>
#include <cassert>


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

		int get_mpi_size()
		{
#ifdef __MPI
			int ret;
			MPI_Comm_size(MPI_COMM_WORLD, &ret);
			return ret;
#else
			constexpr int ret = 1;
			return ret;
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

#ifdef __MPI
		template<typename T>
			T reduce(T a_in, MPI_Op a_op)
			{
				std::cout << "error" << std::endl;
				std::abort();
				return -666;
			}

		template<>
			double reduce(double a_in, MPI_Op a_op)
			{
				double global(0.0);
				MPI_Reduce(&a_in, &global, 1, MPI_DOUBLE, a_op, master, MPI_COMM_WORLD); // master rank is 0
				return global;
			}

		template<>
			int reduce(int a_in, MPI_Op a_op)
			{
				int global(0.0);
				MPI_Reduce(&a_in, &global, 1, MPI_INT, a_op, master, MPI_COMM_WORLD); // master rank is 0
				return global;
			}
#endif

		double reduce_max(double a_duration)
		{
#ifdef __MPI
			double ret = reduce(a_duration, MPI_MAX);
			return ret;
#else
			return a_duration;
#endif
		}

		double reduce_mean(double a_duration)
		{
#ifdef __MPI
			double ret = reduce(a_duration, MPI_SUM);
			auto mpi_size = get_mpi_size(); 
			assert(mpi_size>0);
			return ret / mpi_size;
#else
			return a_duration;
#endif
		}
	};
};
