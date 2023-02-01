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
		/**
		 * @brief initialize a root timer node. This function has to be followed by finalize function. Do not call this function twice.  
		 */
		void initialize();
		
		/**
		 * @brief This function displays each timer node and writes a file with the same information.
		 */
		void print_and_write_timers();

		/**
		 * @brief finalize the root timer node. This function has to be called after the intialize function. Do not call this function twice.  
		 */
		void finalize();

		/**
		 * @brief This function captures the runtime of a given section.
		 * @param [in] lambda section that the user wants to measure.
		 * @return The runtime of lambda 
		 */
		template<typename Lambda>
			double chrono_section(Lambda&& lambda)
			{
				double ret;
				BasicTimer time;
				time.start();
				lambda();
				time.end();
				ret = time.get_duration();
				return ret;	
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
