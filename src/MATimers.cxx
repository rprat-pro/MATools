#include <MATimerOptional.hxx>
#include <MATimersFullTreeMode.hxx>
#include <MATimers.hxx>

namespace MATools
{
	namespace MATimer
	{
		/**
		 * @brief initialize a root timer node. This function has to be followed by finalize function. Do not call this function twice.  
		 */
		void initialize()
		{
#ifdef NO_TIMER
			MATools::MAOutput::printMessage("MATimers_LOG: No timers initialization - timers are disabled by the ROCKABLE_USE_NO_TIMER option");
			return; // end here
#endif
			MATools::MAOutput::printMessage("MATimers_LOG: MATimers initialization ");
			MATimerNode*& root_timer_ptr 	= MATools::MATimer::get_MATimer_node<ROOT>() ;
			assert(root_timer_ptr == nullptr && "MATimer::initialize has already be called");	
			root_timer_ptr 			= new MATimerNode(); 
			MATimerNode*& current 	        = MATools::MATimer::get_MATimer_node<CURRENT>(); 
			current 			= root_timer_ptr;
			assert(current != nullptr);	
			MATools::MATimer::start_global_timer<ROOT>();
		}


		/**
		 * @brief This function displays each timer node and writes a file with the same information.
		 */
		void print_and_write_timers()
		{
#ifdef NO_TIMER
			MATools::MAOutput::printMessage(" No timetable - timers are disabled by the MATimers_USE_NO_TIMER option");
			return; // end here
#endif
			MATools::MATimer::end_global_timer<ROOT>(); 
			MATools::MAOutputManager::write_file(); 
			MATools::MAOutputManager::print_timetable<ROOT>();
		}

		/**
		 * @brief finalize the root timer node. This function has to be called after the intialize function. Do not call this function twice.  
		 */
		void finalize()
		{
			using namespace MATools::MATimer::Optional;
			using namespace MATools::MAOutput;
			MATimerNode* root_ptr 	 = MATools::MATimer::get_MATimer_node<ROOT>() ;
			MATimerNode* current_ptr = MATools::MATimer::get_MATimer_node<CURRENT>() ;
			assert(root_ptr != nullptr);
			assert(current_ptr != nullptr);

			if(root_ptr != current_ptr) 
				printMessage("MATimers_DEBUG_LOG: MATimers are not corretly used, root node is ", root_ptr, " and current node is " , current_ptr);
			else 
				printMessage("MATimers_LOG: MATimers finalization");

			MATools::MATimer::end_global_timer<ROOT>(); 
#ifdef __MPI
			if(is_full_tree_mode())
				MATools::MATimer::FullTreeMode::build_full_tree();
#endif
			if(is_print_timetable())
				MATools::MAOutputManager::print_timetable<enumTimer::ROOT>();

			if(is_write_file())
			{
				MATools::MAOutput::printMessage("MATimers_LOG: Writing timetable ... ");
				MATools::MAOutputManager::write_file(); 
			}
			delete root_ptr;
			root_ptr = nullptr;
		}
	};
};
