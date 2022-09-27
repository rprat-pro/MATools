#pragma once

#include <MATimerMPI.hxx>

namespace MATimer
{
	namespace output
	{
		template<typename Arg>
		void printMessage(Arg a_msg)
		{
			if(mpi::is_master())
			{
				std::cout << a_msg << std::endl;
			}
		}

		template<typename Arg, typename... Args>
		void printMessage(Arg a_msg, Args... a_msgs)
		{
			if(mpi::is_master())
			{
				std::cout << a_msg << " ";
				printMessage(a_msgs...);
			}
		}
	};
};
