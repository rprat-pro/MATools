#pragma once

#include <iostream>
#include <cassert>

#include <EnumTimer.hxx>
#include <Column.hxx>
#include <MATimerNode.hxx>
#include <Timer.hxx>
#include <OutputManager.hxx>
#include <MATrace.hxx>


namespace MATimer
{
	namespace timers
	{
		void initialize(int*,  char***, bool = true);
		void initialize();
		void print_and_write_timers();
		void finalize(bool = true, bool = true, bool = true);

		template<typename Lambda>
		double chrono_section(Lambda&& lambda)
		{
			using steady_clock = std::chrono::steady_clock;
			using time_point = std::chrono::time_point<steady_clock>;
			time_point tic, toc;
			tic = steady_clock::now();
			lambda();
			toc = steady_clock::now();
			auto measure = toc - tic;
			return measure.count();	
		}
	};
};



#ifdef NO_TIMER
// do nothing
#define START_TIMER(XNAME) 

#else

#define START_TIMER(XNAME) auto& current = MATimer::timers::get_MATimer_node<CURRENT>();\
	assert(current != nullptr && "do not use an undefined MATimerNode");\
	current = current->find(XNAME); \
        MATimer::timer::Timer tim(current->get_ptr_duration());
#endif
