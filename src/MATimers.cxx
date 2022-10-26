#include <MATimers.hxx>

namespace MATimer
{
	namespace timers
	{
		void initialize(int *argc, char ***argv, bool do_mpi_init)
		{
#ifdef __MPI
			if(do_mpi_init) MPI_Init(argc,argv);
#endif
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
			MATimer::MATrace::initialize();
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

		void finalize(bool a_print_timetable, bool a_write_file, bool do_mpi_final)
		{
			MATimer::MATrace::finalize();
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
#ifdef __MPI
			if(do_mpi_final) MPI_Finalize();
#endif
			delete root_ptr;
		}
	};
};
