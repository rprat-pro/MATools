#include<Common/MAToolsMPI.hxx>
#include<MATimers/MATimerNode.hxx>
#include<iomanip>

namespace MATools
{
	/**
	 * MATimerNode is the storage class corresponding to a node of the MATimer tree.
	 */
	namespace MATimer
	{
		using duration = std::chrono::duration<double>;

		/**
		 * @brief default constructor.
		 */
		MATimerNode::MATimerNode() : m_daughter() // only used for root
		{
			m_name = "root";
			m_iteration = 1;
			m_level = 0;
			m_mother = nullptr;
#ifdef __MPI
			m_nb_mpi = 1;
#endif

		}

		/**
		 * @brief MATimerNode constructor used to initialize a node with a node name and a mother node.
		 */
		MATimerNode::MATimerNode(std::string name, MATimerNode* mother): m_daughter(), m_duration(0)
		{
			m_name = name;
			m_iteration = 1;
			m_level = mother->m_level + 1;
			m_mother = mother;
#ifdef __MPI
			m_nb_mpi = 1;
#endif
		}

		/**
		 * @brief This function is used to find if a daughter node is already defined with this node name. If this node does not exist, a new daughter MATimerNode is added
		 * @param[in] name name of the desired node
		 * @return the MATimerNode desired 
		 */
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

		/**
		 * @brief Displays a motif several times.
		 * @param[in] begin column number where the motif starts.
		 * @param[in] end column number where the motif finishs.
		 * @param[in] motif the replicated motif, this motif should have a length equal to 1.
		 */
		void 
			MATimerNode::print_replicate(int a_begin, int a_end, std::string a_motif)
			{
#ifndef NDEBUG
				if(a_end < a_begin) 
				{
					//MATools::MAOutput::printMessage("MATools_LOG: column width is probalby too small and it's not possible to replicated this motif");
				}
#endif

				for(int i = a_begin ; i < a_end ; i++) 
				{
					std::cout << a_motif;
				}
			}

		/**
		 * @brief Displays a blank character.
		 */
		void 
			MATimerNode::space()
			{
				std::cout << " "; 
			}

		/**
		 * @brief Displays a "|".
		 */
		void 
			MATimerNode::column()
			{
				std::cout << "|"; 
			}

		/**
		 * @brief Displays a return line.
		 */
		void 
			MATimerNode::end_line()
			{
				std::cout << std::endl; 
			}

		/**
		 * @brief Displays the banner/header.
		 * @param[in] shift number of blank character displayed
		 */
		void 
			MATimerNode::print_banner(size_t shift)
			{
				if(m_name == "root")
				{
#ifndef __MPI
					MATools::MAOutput::printMessage(" MPI feature is disable for timers, if you use MPI please add -D__MPI ");
#else
					if(MATools::MPI::is_master()) 
					{
						MATools::MAOutput::printMessage(" MPI feature activated, rank 0:");
#endif
						std::string start_name = " |-- start timetable "; 
						std::cout << start_name;
						size_t end = shift+ nColumns*(cWidth+1) + 1;	
						print_replicate(start_name.size(), end,"-");
						column(); end_line();
						std::string name = " |    name";
						std::cout << name;
						print_replicate(name.size(),shift + 1," ");

						// columns name are displayed, columns change if the MPI mode is ued
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

		/**
		 * @brief Displays the header.
		 * @param[in] shift number of blank character displayed
		 */
		void 
			MATimerNode::print_ending(size_t shift)
			{
				if(m_name == "root")
				{
					if(MATools::MPI::is_master()) 
					{
						shift+= nColumns*(cWidth+1) + 1; // +1 for "|";
						std::string end_name = " |-- end timetable " ;
						std::cout << end_name;
						print_replicate(end_name.size(),shift,"-");
						column(); end_line();
					}
				}
			}

		/**
		 * @brief Gets of the duration member
		 * @return pointer of the duration member of a MATimerNode
		 */
		duration* 
			MATimerNode::get_ptr_duration()
			{
				return &m_duration;
			}

		/**
		 * @brief Displays the local runtime.
		 * @param[in] shift number of blank character displayed
		 * @param[in] runtime local duration value
		 */
		void 
			MATimerNode::print_local(size_t shift, double total_time)
			{
				assert(total_time >= 0);
				int nC = 3;
				std::string cValue[3];
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
				cValue[1] = std::to_string( m_duration.count());	
				cValue[2] = std::to_string( (m_duration.count()/total_time)*100 );	

				for(int i =  0 ; i < nC ; i++)
				{
					column();
					int size = cValue[i].size();
					print_replicate(0,(int(cWidth)-size - 1), " ");
					std::cout << cValue[i];
					space();
				}
				column();end_line();
			}

		/**
		 * @brief Displays the runtime.
		 * @param[in] shift number of blank character displayed
		 * @param[in] runtime duration value
		 */
		void 
			MATimerNode::print(size_t shift, double total_time)
			{
				assert(total_time >= 0);
				std::string cValue[nColumns];
				if(MATools::MPI::is_master()) 
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
				int size = MATools::MPI::get_mpi_size();

				assert(size > 0);
				std::vector<double> list;

				if(MATools::MPI::is_master()) list.resize(size);

				MPI_Gather(&local,1,MPI_DOUBLE, list.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); // master rank is 0


				if(MATools::MPI::is_master())
				{
#ifdef UNDEFINED // __cplusplus > 201103L
					const auto [min,max]	= std::minmax_element(list.begin(), list.end());
					auto global_max 	= *max;
					auto global_min 	= *min; 
					auto sum 		= std::accumulate(list.begin(), list.end(), double(0.));
					auto global_mean 	= sum / double(size);
					auto part_time	= (global_max / total_time ) * 100;
#else
					double min= list[0];
					double max= list[0];
					double sum= list[0];
					int size = list.size();
					for(int id = 1 ; id < size ; id++)
					{
						min = std::min(min,list[id]);
						max = std::max(max,list[id]);
						sum += list[id];
					}
					auto global_max     = max;
					auto global_min     = min;
					auto global_mean    = sum / double(size);
					auto part_time  = (global_max / total_time ) * 100;
#endif
					assert(global_mean >= 0);
					assert(global_min >= 0);
					assert(global_max >= 0);

					assert(global_max >= global_mean);
					assert(global_mean >= global_min);

					std::cout << std::setprecision(25);
					const int precisionVal = 25;

					cValue[1] = std::to_string( global_min).substr(0, std::to_string(global_min).find(".") + precisionVal + 1);	
					cValue[2] = std::to_string( global_mean).substr(0, std::to_string(global_mean).find(".") + precisionVal + 1);	
					cValue[3] = std::to_string( global_max).substr(0, std::to_string(global_max).find(".") + precisionVal + 1);	
					cValue[4] = std::to_string( part_time).substr(0, std::to_string(part_time).find(".") + precisionVal + 1) + "%";	
					cValue[5] = std::to_string( (global_max/global_mean)-1).substr(0, std::to_string((global_max/global_mean)-1).find(".") + precisionVal + 1) + "%";
				}
#else
					std::cout << std::setprecision(25);
					const int precisionVal = 25;
				cValue[1] = std::to_string( m_duration.count()).substr(0, std::to_string(m_duration.count()).find(".") + precisionVal + 1);	
				cValue[2] = std::to_string( (m_duration.count()/total_time)*100 ).substr(0, std::to_string((m_duration.count()/total_time)*100).find(".") + precisionVal + 1);
#endif
				if(MATools::MPI::is_master())
				{
					for(size_t i =  0 ; i < nColumns ; i++)
					{
						column();
						int size = cValue[i].size();
						print_replicate(0,(int(cWidth)- size - 1), " ");
						std::cout << cValue[i];
						space();
					}
					column();end_line();
				}
			}

		/**
		 * @brief Retruns the MATimerNode name
		 * @return name
		 */
		std::string 
			MATimerNode::get_name()
			{
				std::string ret = m_name;
				return ret;
			}

		/**
		 * @brief Retruns the duration
		 * @return duration value
		 */
		double 
			MATimerNode::get_duration()
			{
				double ret = m_duration.count();
				return ret;
			}

		/**
		 * @brief Retruns the MATimerNode iteration number
		 * @return the iteration number
		 */
		std::size_t 
			MATimerNode::get_iteration()
			{
				std::size_t ret = m_iteration;
				return ret;
			}

		/**
		 * @brief Retruns the MATimerNode level
		 * @return level
		 */
		std::size_t 
			MATimerNode::get_level()
			{
				std::size_t ret = m_level;
				return ret;
			}

		/**
		 * @brief Retruns a vector of daughter MATimerNode pointers
		 * @return daughter nodes
		 */
		std::vector<MATimerNode*>& 
			MATimerNode::get_daughter()
			{
				return m_daughter;
			}

		/**
		 * @brief Retruns the mother MATimerNode pointer
		 * @return mother pointer
		 */
		MATimerNode* 
			MATimerNode::get_mother()
			{
				MATimerNode* ret = m_mother;
				return ret;
			}

		/** @brief This function displays information about one timer node */
		void MATimerNode::debug_info()
		{
			std::cout << " node name : " << get_name() << std::endl;
			std::cout << " mother node address : " << get_mother() << std::endl;
			std::cout << " number of daughter nodes : " << m_daughter.size() << std::endl;
			std::cout << " level : " << get_level() << std::endl;
			std::cout << " number of iterations : " << get_iteration() << std::endl;
			std::cout << " current duration : " << get_duration() << std::endl;
		}
	}
};


