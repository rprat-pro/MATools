#include <MATimers.hxx>

namespace MATools
{
	namespace MATimer
	{
		void initialize(int *argc, char ***argv, bool do_mpi_init)
		{
#ifdef __MPI
			if(do_mpi_init) MPI_Init(argc,argv);
#endif
			initialize();
		}

		void initialize()
		{
#ifdef NO_TIMER
			MATools::MAOutput::printMessage("MATimers_LOG: No timers initialization - timers are disabled by the ROCKABLE_USE_NO_TIMER option");
			return; // end here
#endif
			MATools::MAOutput::printMessage("MATimers_LOG: MATimers initialization ");
			MATimerNode*& root_timer_ptr 	= MATools::MATimer::get_MATimer_node<ROOT>() ;
			assert(root_timer_ptr == nullptr);	
			root_timer_ptr 			= new MATimerNode(); 
			MATimerNode*& current 	        = MATools::MATimer::get_MATimer_node<CURRENT>(); 
			current 			= root_timer_ptr;
			assert(current != nullptr);	
			MATools::MATimer::start_global_timer<ROOT>();
			MATools::MATrace::initialize();
		}


		void print_and_write_timers()
		{
#ifdef NO_TIMER
			MATools::MAOutput::printMessage(" No timetable - timers are disabled by the MATimers_USE_NO_TIMER option");
			return; // end here
#endif
			MATools::MATimer::end_global_timer<ROOT>(); 
			MATools::MAOutputManager::write_file(); 
			MATools::MAOutputManager::print_timetable();
		}

		void finalize(bool a_print_timetable, bool a_write_file, bool do_mpi_final)
		{
			MATools::MATrace::finalize();
			MATimerNode* root_ptr 	 = MATools::MATimer::get_MATimer_node<ROOT>() ;
			MATimerNode* current_ptr = MATools::MATimer::get_MATimer_node<CURRENT>() ;
			assert(root_ptr != nullptr);
			assert(current_ptr != nullptr);
			if(root_ptr != current_ptr) 
				MATools::MAOutput::printMessage("MATimers_DEBUG_LOG: MATimers are not corretly used, root node is ", root_ptr, " and current node is " , current_ptr);
			else 
				MATools::MAOutput::printMessage("MATimers_LOG: MATimers finalisation");

			MATools::MATimer::end_global_timer<ROOT>(); 

			if(a_print_timetable)
				MATools::MAOutputManager::print_timetable();


			if(a_write_file)
			{
				MATools::MAOutput::printMessage("MATimers_LOG: Writing timetable ... ");
				MATools::MAOutputManager::write_file(); 
			}
#ifdef __MPI
			if(do_mpi_final) MPI_Finalize();
#endif
			delete root_ptr;
		}
	};
};
