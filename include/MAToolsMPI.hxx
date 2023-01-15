#pragma once

#ifdef __MPI
	#include <mpi.h>
#endif


namespace MATools
{
	namespace MPI
	{
		bool is_master();
		int get_rank();
		int get_mpi_size();
		double reduce_max(double);
		double reduce_mean(double);
	};
}
