#include <MATimers.hxx>

namespace MATimer
{
	namespace timers
	{
		void init_timers()
		{
#ifdef NO_TIMER
			MATimer::output::printMessage(" No timers initialization - timers are disabled by the ROCKABLE_USE_NO_TIMER option");
			return; // end here
#endif
			MATimer::output::printMessage(" Timers initialization ");
			MATimerNode*& root_timer_ptr 	= MATimer::timers::get_MATimer_node<ROOT>() ;
		        assert(root_timer_ptr == nullptr);	
			root_timer_ptr 			= new MATimerNode(); 
			MATimerNode*& current 	= MATimer::timers::get_MATimer_node<CURRENT>(); 
			current 			= root_timer_ptr;
		        assert(current != nullptr);	
			MATimer::timer::start_global_timer<ROOT>();
		}

		void print_and_write_timers()
		{
#ifdef NO_TIMER
			MATimer::output::printMessage(" No timetable - timers are disabled by the ROCKABLE_USE_NO_TIMER option");
			return; // end here
#endif
			MATimer::timer::end_global_timer<ROOT>(); 
			MATimer::outputManager::write_file(); 
			MATimer::outputManager::print_timetable();
		}
	};
};
