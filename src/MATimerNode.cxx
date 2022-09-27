#include<MATimerNode.hxx>

namespace MATimer
{
	namespace timers
	{
		using duration = std::chrono::duration<double>;

		// constructor
		MATimerNode::MATimerNode() : m_daughter() // only used for root
		{
			m_name			= "root";
			m_iteration = 1;
			m_level  		= 0;
			m_mother 		= nullptr;

		}

		MATimerNode::MATimerNode(std::string name, MATimerNode* mother): m_daughter(), m_duration(0)
		{
			m_name      = name;
			m_iteration = 1;
			m_level 	  = mother->m_level + 1;
			m_mother 	  = mother;
		}

		MATimerNode* 
			MATimerNode::find(std::string name)
			{
				assert(this != nullptr);
				for(auto it = m_daughter.begin() ; it < m_daughter.end() ; it++)
				{
					if((*it)->m_name == name)
					{
						(*it)->m_iteration++;
						return (*it);
					}
				}
				MATimerNode* myTmp = new MATimerNode(name, this);
				m_daughter.push_back(myTmp);
				return myTmp;
			}

		void 
			MATimerNode::print_replicate(size_t begin, size_t end, std::string motif)
			{
				for(size_t i = begin ; i < end ; i++) std::cout << motif;
			}


		void 
			MATimerNode::space()
			{
				std::cout << " "; 
			}

		void 
			MATimerNode::column()
			{
				std::cout << "|"; 
			}

		void 
			MATimerNode::end_line()
			{
				std::cout << std::endl; 
			}

		void 
			MATimerNode::print_banner(size_t shift)
			{
				if(m_name == "root")
				{
#ifndef __MPI
					MATimer::output::printMessage(" MPI feature is disable for timers, if you use MPI please add -D__MPI ");
#else
					if(MATimer::mpi::is_master()) {
						MATimer::output::printMessage(" MPI feature activated, rank 0:");
#endif
						std::string start_name = " |-- start timetable "; 
						std::cout << start_name;
						size_t end = shift+ nColumns*(cWidth+1) + 1;	
						print_replicate(start_name.size(), end,"-");
						column(); end_line();
						std::string name = " |    name";
						std::cout << name;
						print_replicate(name.size(),shift + 1," ");
						for(size_t i =  0 ; i < nColumns ; i++)
						{
							column();
							int size = cName[i].size();
							print_replicate(0,(int(cWidth)-size - 1), " ");
							std::cout << cName[i];
							space();
						}
						column(); end_line();
						space(); column();
						print_replicate(2, end,"-");
						column(); end_line();
#ifdef __MPI
					}
#endif
				}
			}

		void 
			MATimerNode::print_ending(size_t shift)
			{
				if(m_name == "root")
				{
					if(MATimer::mpi::is_master()) 
					{
						shift+= nColumns*(cWidth+1) + 1; // +1 for "|";
						std::string end_name = " |-- end timetable " ;
						std::cout << end_name;
						print_replicate(end_name.size(),shift,"-");
						column(); end_line();
					}
				}
			}

		duration* 
			MATimerNode::get_ptr_duration()
			{
				return &m_duration;
			}

		void 
			MATimerNode::print(size_t shift, double total_time)
			{
				assert(total_time >= 0);
				std::string cValue[nColumns];
				if(MATimer::mpi::is_master()) 
				{
					size_t realShift = shift;
					space(); column(); space();
					size_t currentShift = 3;
					for(int i = 0 ; i < int(m_level) - 1; i++) 
					{
						int spaceSize = 3;
						for(int j = 0 ; j < spaceSize ; j++) space();
						currentShift += spaceSize;
					}
					if(m_level>0) {
						std::cout << "|--";
						currentShift += 3;
					}
					std::cout << "> "<< m_name;
					currentShift += m_name.size() + 1;
					print_replicate(currentShift, realShift, " ");

					cValue[0] = std::to_string(m_iteration);	
				}
#ifdef __MPI
				double local  = m_duration.count();	
				int size = -1;
				MPI_Comm_size(MPI_COMM_WORLD, &size);

				assert(size > 0);
				std::vector<double> list;

				if(MATimer::mpi::is_master()) list.resize(size);

				MPI_Gather(&local,1,MPI_DOUBLE, list.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); // master rank is 0


				if(MATimer::mpi::is_master())
				{
					const auto [min,max]	= std::minmax_element(list.begin(), list.end());
					auto global_max 	= *max;
					auto global_min 	= *min; 
					auto sum 		= std::accumulate(list.begin(), list.end(), double(0.));
					auto global_mean 	= sum / double(size);
					auto part_time	= (global_max / total_time ) * 100;

					assert(global_mean >= 0);
					assert(global_min >= 0);
					assert(global_max >= 0);

					assert(global_max >= global_mean);
					assert(global_mean >= global_min);

					cValue[1] = std::to_string( global_min);	
					cValue[2] = std::to_string( global_mean);	
					cValue[3] = std::to_string( global_max);	
					cValue[4] = std::to_string( part_time) + "%";	
					cValue[5] = std::to_string( (global_max/global_mean)-1) + "%";
				}
#else
				cValue[1] = std::to_string( m_duration.count());	
				cValue[2] = std::to_string( (m_duration.count()/total_time)*100 );	
#endif
				if(MATimer::mpi::is_master())
				{
					for(size_t i =  0 ; i < nColumns ; i++)
					{
						column();
						int size = cValue[i].size();
						print_replicate(0,(int(cWidth)-size - 1), " ");
						std::cout << cValue[i];
						space();
					}
					column();end_line();
				}
			}

		std::string 
			MATimerNode::get_name()
			{
				return m_name;
			}

		double 
			MATimerNode::get_duration()
			{
				return m_duration.count();
			}

		std::size_t 
			MATimerNode::get_iteration()
			{
				return m_iteration;
			}

		std::size_t 
			MATimerNode::get_level()
			{
				return m_level;
			}

		std::vector<MATimerNode*>& 
			MATimerNode::get_daughter()
			{
				return m_daughter;
			}


		MATimerNode* 
			MATimerNode::get_mother()
			{
				return m_mother;
			}
	}
};


