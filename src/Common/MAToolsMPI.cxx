#include <Common/MAToolsMPI.hxx>
#include <iostream>
#include <cassert>


/**
 * @namespace MATools
 * @brief Namespace containing utility tools for various purposes.
 */
namespace MATools
{
	/**
	 * @namespace MPI
	 * @brief Namespace containing MPI-related utilities.
	 */
	namespace MPI
	{
		constexpr int master=0;

		void mpi_initialize([[maybe_unused]] int *argc, [[maybe_unused]]char ***argv)
		{
#ifdef __MPI
			const auto is_init = check_mpi_initialized(); 			
			if(!is_init)
			{
				MPI_Init(argc,argv);
			}
#endif /* __MPI */
		}

		void mpi_finalize()
		{
#ifdef __MPI
			const auto is_final = check_mpi_finalized(); 			
			if(!is_final)
			{
				MPI_Finalize();
			}
#endif /* __MPI */
		}

		bool check_mpi_initialized()
		{
#ifdef __MPI
			int val = -1;
			MPI_Initialized(&val);
			assert(val != -1 && "error in check mpi init");
			bool ret = val == 1 ? true : false;
#else
			bool ret = false;
#endif
			return ret;
		}

		bool check_mpi_finalized()
		{
#ifdef __MPI
			int val = -1;
			MPI_Finalized(&val);
			assert(val != -1 && "error in check mpi finalize");
			bool ret = val == 1 ? true : false;
#else
			bool ret = false;
#endif
			return ret;
		}

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
