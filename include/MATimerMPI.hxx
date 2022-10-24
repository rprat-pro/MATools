#pragma once
#ifdef __MPI
	#include "mpi.h"
#endif


namespace MATimer
{
	namespace mpi
	{
		bool is_master();
		int get_rank();
		double reduce_max(double a_duration);
	};
}
