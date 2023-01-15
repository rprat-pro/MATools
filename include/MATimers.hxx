#pragma once

#include <iostream>
#include <cassert>

#include <EnumTimer.hxx>
#include <Column.hxx>
#include <MATimerNode.hxx>
#include <Timer.hxx>
#include <MAOutputManager.hxx>
#include <MATimerOptional.hxx>
#include <MATimerInfo.hxx>

namespace MATools
{
	namespace MATimer
	{
		void initialize();
		void print_and_write_timers();
		void finalize();

		template<typename Lambda>
		double chrono_section(Lambda&& lambda)
		{
			BasicTimer time;
			time.start();
			lambda();
			time.end();
			return time.get_duration();	
		}
	};
};



#ifdef NO_TIMER
// do nothing
#define START_TIMER(XNAME) 

#else

#define START_TIMER(XNAME) auto& current = MATools::MATimer::get_MATimer_node<CURRENT>();\
					   assert(current != nullptr && "do not use an undefined MATimerNode");\
					   current = current->find(XNAME); \
					   MATools::MATimer::Timer tim(current->get_ptr_duration());
#endif
