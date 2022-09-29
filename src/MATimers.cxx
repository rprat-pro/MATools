#include <MATimers.hxx>

namespace MATimer
{
	namespace timers
	{
		void init_timers()
		{
#ifdef NO_TIMER
			MATimer::output::printMessage("MATimers_LOG: No timers initialization - timers are disabled by the ROCKABLE_USE_NO_TIMER option");
			return; // end here
#endif
			MATimer::output::printMessage("MATimers_LOG: MATimers initialization ");
			MATimerNode*& root_timer_ptr 	= MATimer::timers::get_MATimer_node<ROOT>() ;
		        assert(root_timer_ptr == nullptr);	
			root_timer_ptr 			= new MATimerNode(); 
			MATimerNode*& current 	        = MATimer::timers::get_MATimer_node<CURRENT>(); 
			current 			= root_timer_ptr;
		        assert(current != nullptr);	
			MATimer::timer::start_global_timer<ROOT>();
		}

		void print_and_write_timers()
		{
#ifdef NO_TIMER
			MATimer::output::printMessage(" No timetable - timers are disabled by the MATimers_USE_NO_TIMER option");
			return; // end here
#endif
			MATimer::timer::end_global_timer<ROOT>(); 
			MATimer::outputManager::write_file(); 
			MATimer::outputManager::print_timetable();
		}

		void finalise(bool a_print_timetable, bool a_write_file)
		{
			MATimerNode* root_ptr 	 = MATimer::timers::get_MATimer_node<ROOT>() ;
			MATimerNode* current_ptr = MATimer::timers::get_MATimer_node<CURRENT>() ;
			assert(root_ptr != nullptr);
			assert(current_ptr != nullptr);
			if(root_ptr != current_ptr) 
				MATimer::output::printMessage("MATimers_DEBUG_LOG: MATimers are not corretly used, root node is ", root_ptr, " and current node is " , current_ptr);
			else 
				MATimer::output::printMessage("MATimers_LOG: MATimers finalisation");
				
			MATimer::timer::end_global_timer<ROOT>(); 

			if(a_print_timetable)
				MATimer::outputManager::print_timetable();


			if(a_write_file)
			{
				MATimer::output::printMessage("MATimers_LOG: Writing timetable ... ");
				MATimer::outputManager::write_file(); 
			}

			delete root_ptr;
		}
	};
};
