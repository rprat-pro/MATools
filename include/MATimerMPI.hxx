#pragma once


namespace MATimer
{
	namespace mpi
	{
		bool is_master();
		double reduce_max(double a_duration);
	};
}
