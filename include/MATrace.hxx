#pragma once

#include <chrono>
#include <fstream>
#include <vector>
#include <cassert>
#include <string.h>
#include <thread>
#include <map>

#include <MATimerMPI.hxx>

#include <MATraceTypes.hxx>
#include <MATraceInstance.hxx>


namespace MATimer
{
	namespace MATrace
	{
		void initialize();
		void finalize();
		void start();
		void stop(std::string a_name);

		void header(std::ofstream& out, vite_event& event);
		void ending(std::ofstream& out, double last);

		template<typename Fun>
			void MATrace_kernel (std::string a_name,Fun& a_fun)
			{
				start();
				a_fun();
				stop(a_name);
			}

		template<typename Fun, typename... Args>
			void MATrace_functor (std::string a_name, Fun& a_fun, Args&&... a_args)
			{
				start();
				a_fun(std::forward<Args>(a_args)...);
				stop(a_name);
			}
	}
}
