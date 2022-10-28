
#ifdef __MPI

#include <cstring>
#include <vector>
#include <iostream>
#include <MATimersFullTreeMode.hxx>
#include <MAToolsMPI.hxx>
#include <MATimerNode.hxx>
#include <MAOutputManager.hxx>
#include <string.h>
namespace MATools 
{
	namespace MATimer
	{
		namespace FullTreeMode 
		{
			using namespace MATools::MPI;

			/**
			 * return the number of bytes in a minimal_info_size. This size doesn't change.
			 * @see class minimal_info
			 * @return int : number of bytes.
			 */
			int minimal_info_size()
			{
				int ret = sizeof(minimal_info);
				return ret;
			}

			/**
			 * minimal_info constructor*
			 * @see class minimal_info
			 */
			minimal_info::minimal_info(std::string a_name, double a_duration, std::size_t a_s)
			{
				strncpy(m_name, a_name.c_str(), minimal_info_name_size);
				m_duration =  a_duration;
				m_nb_daughter = a_s;
			}

			/**
			 * print information of a minimal_info structure on the master node 
			 * @see class minimal_info
			 * @return void
			 */
			void minimal_info::print()
			{
				if(is_master())
				{
					std::string name = m_name;
					std::cout << " (" << name << "," << m_duration << "," << m_nb_daughter << ")";
				}
			}

			/**
			 * build a vector of minimal info per MPI process to send them (after this routine) on the master process. 
			 * @see class minimal_info
			 * @return a vector of minimal_info
			 */
			std::vector<minimal_info> build_my_tree()
			{
				typedef std::vector<minimal_info> VMI;
				VMI ret;
				MATimerNode* root_timer = MATools::MATimer::get_MATimer_node<ROOT>();

				// this function is called on every MATimerNode.
				auto build_tree = [](MATimerNode* a_ptr, VMI& a_vec)
				{
					double duration = a_ptr->get_duration();
					auto& daughter = a_ptr->get_daughter();
					std::size_t dsize = daughter.size();
					a_vec.push_back(minimal_info(a_ptr->get_name(), duration, dsize));
				};   

				MATools::MAOutputManager::recursive_call(build_tree, root_timer, ret);

				return ret;
			}

			/**
			 * build a vector of minimal info on the master process. Every trees are packed in a vector of minimal_info and they are gathered on the master node. 
			 * @see class minimal_info
			 * @return void
			 */
			void build_full_tree()
			{
				auto my_info = build_my_tree();
				int info_size = my_info.size() * minimal_info_size();
				int mpi_size;

				MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
				std::vector<int> all_info_size(mpi_size);

				MPI_Allgather(&info_size, 1, MPI_INT, all_info_size.data(), 1, MPI_INT, MPI_COMM_WORLD);

				int size=0;
				for(auto it : all_info_size) {
					size += it;
				}

				std::vector<char> full_tree (size) ;  
				std::vector<int> displ(mpi_size, 0);

				MPI_Gatherv(my_info.data(), my_info.size(), MPI_CHAR, full_tree.data() , all_info_size.data() , displ.data(), MPI_CHAR, 0, MPI_COMM_WORLD);

				const int structure_size = minimal_info_size(); 
				char* ptr = full_tree.data();
				minimal_info* cast_ptr = (minimal_info*) (ptr);
				for(int mpi = 0 ; mpi < mpi_size ; mpi++)
				{
					if(is_master())
					{
						std::cout << std::endl << " MPI : " << mpi << std::endl; 
						for(int it = 0 ; it < all_info_size[mpi] ; it += structure_size)
						{
							cast_ptr->print();
							cast_ptr++;
						}
					}

				}
			}
		};
	};
};
#endif
