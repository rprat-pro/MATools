#pragma once

#ifndef NO_TIMER
#include <iostream>
#include <cassert>

#include <MATimers/EnumTimer.hxx>
#include <MATimers/MATimerNode.hxx>
#include <MATimers/Timer.hxx>
#include <MATimers/MATimerInfo.hxx>
#include <MAOutput/MAOutputManager.hxx>

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
	}
}
#endif /* NO_TIMER */

