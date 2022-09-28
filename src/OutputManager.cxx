#include <OutputManager.hxx>

namespace MATimer
{
	namespace outputManager
	{
		std::string build_name()
		{
			std::size_t nthreads=0;
#pragma omp parallel
			{
				nthreads = omp_get_num_threads();
			}
			std::string base_name = "MATimers";
#ifdef __MPI
			int mpiSize;
			MPI_Comm_size(MPI_COMM_WORLD,&mpiSize);
			std::string file_name = base_name + "." + std::to_string(mpiSize) + ".perf";
#else
			std::string file_name = base_name + "." + std::to_string(nthreads) + ".perf";
#endif
			return file_name;
		}

		void print_timetable()
		{
			MATimerNode* root_timer = MATimer::timers::get_MATimer_node<ROOT>();
			assert(root_timer != nullptr);
			double runtime = root_timer->get_duration();
			runtime = MATimer::mpi::reduce_max(runtime); // if MPI, else return runtime

			auto my_print = [](MATimerNode* a_ptr, size_t a_shift, double a_runtime)
			{
				a_ptr->print(a_shift, a_runtime);
			};

			auto sort_comp = [] (MATimerNode* a_ptr, MATimerNode* b_ptr)
			{
				return a_ptr->get_duration() > b_ptr->get_duration() ;
			};


			auto max_length = [](MATimerNode* a_ptr, size_t& a_count, size_t& a_nbElem)
			{
				size_t length = a_ptr->get_level()*3 + a_ptr->get_name().size();
				a_count = std::max(a_count, length);
				a_nbElem++;
			};
			size_t count(0), nbElem(0);

			recursive_call(max_length, root_timer, count, nbElem);
			count += 6;
			root_timer->print_banner(count);
			recursive_sorted_call(my_print, sort_comp, root_timer, count, runtime);
			root_timer->print_ending(count);
		}

		void write_file()
		{
			auto name = build_name();
			write_file(name);
		}

		void write_file(std::string a_name)
		{
			using namespace MATimer::output;
			using namespace MATimer::mpi;
			//using MATimer::mpi::reduce_max;

			std::ofstream myFile (a_name, std::ofstream::out);	
			MATimerNode* root_timer = MATimer::timers::get_MATimer_node<ROOT>();
			auto rootTime = root_timer->get_duration();
			rootTime = MATimer::mpi::reduce_max(rootTime);
			auto my_write = [rootTime](MATimerNode* a_ptr, std::ofstream& a_file)
			{
				std::string space;
				std::string motif = "   ";

				for(std::size_t i = 0 ; i < a_ptr->get_level() ; i++) space +=motif;

				const auto max_time = reduce_max(a_ptr->get_duration());

				if(is_master())
				{
					a_file << space << a_ptr->get_name() 
						<< " " << a_ptr->get_iteration()
						<< " " << max_time
						<< " " <<(max_time/rootTime)*100
						<< std::endl;
				}
			};

			recursive_call(my_write,root_timer,myFile);
		}
	}
};
