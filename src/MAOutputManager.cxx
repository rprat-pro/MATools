#include <MAOutputManager.hxx>

namespace MATools
{
	namespace MAOutputManager
	{
		std::string build_name()
		{
			std::string base_name = "MATimers";
#ifdef __MPI
			int mpiSize;
			//MPI_Comm_rank(MPI_COMM_WORLD,&mpiSize);
			MPI_Comm_size(MPI_COMM_WORLD,&mpiSize);
			std::string file_name = base_name + "." + std::to_string(mpiSize) + ".perf";
#else
			std::size_t nthreads=0;
#pragma omp parallel
			{
				nthreads = omp_get_num_threads();
			}
			std::string file_name = base_name + "." + std::to_string(nthreads) + ".perf";
#endif
			return file_name;
		}

		void write_file()
		{
			auto name = build_name();
			write_file(name);
		}

		void write_file(std::string a_name)
		{
			using namespace MATools::MAOutput;
			using namespace MATools::MPI;
			//using MATools::MPI::reduce_max;

			std::ofstream myFile (a_name, std::ofstream::out);	
			MATimerNode* root_timer = MATools::MATimer::get_MATimer_node<ROOT>();
			assert(root_timer != nullptr);
			auto rootTime = root_timer->get_duration();
			rootTime = MATools::MPI::reduce_max(rootTime);
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
