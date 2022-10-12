#pragma once

#ifdef __MPI
#include <cstring>

namespace MATimer {

	namespace modefulltree {

		constexpr size_t minimal_info_name_size = 64;

		void copy_to_string(const std::string& a_src, char a_dst[minimal_info_name_size])
		{
			strncpy(a_dst, a_src.c_str(), a_src.size());
			dst[a_src.size() - 1] = '\0';
		}

		// return the number of bytes
		int minimal_info_size() const
		{
			int ret = 0;
			ret += 8*sizeof(m_duration);
			ret += 8*sizeof(m_nb_daughter);
			ret += minimal_info_name_size;
		}

		class minimal_info
		{
			double m_duration;
			std::size_t m_nb_daughter;
			char m_name[minimal_info_name_size];
		
			// debug function
			void print()
			{
				if(isMaster())
				{
					std::cout << " (" << m_name << "," << m_duration << "," << m_nb_daughter << ")";
				}
			}

		}


		std::vector<minimal_info> build_my_tree()
		{
			std::vector<minimal_info> ret;
			MATimerNode* root_timer = MATimer::timers::get_MATimer_node<ROOT>();

			auto build_tree = [](MATimerNode* a_ptr, std::vector<minimal_info>&& a_vec)
			{
				auto& daughter = a_ptr->get_daughter();
				char name[minimal_info_name_size];
				copy_to_string(a_ptr->get_name(), name);
				auto duration = a_ptr->get_duration();
				minimal_info elem = {name,duration,daughter.size()};
				a_vec.push_back(elem)
			};   

			outputManager::recursive_call(build_tree, root_timer, ret);

			return ret;
		}


		void build_full_tree()
		{
			auto my_info = build_my_tree();
			int info_size = my_info.size() * minimal_info_size();
			int mpi_size;

			MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
			std::vector<int> all_info_size(mpi_size);

			MPI_Allgather(&info_size, 1, MPI_INT, all_info_size.data(), 1, MPI_INT, MPI_COMM_WORLD);
			size=0;

			for(auto it : all_info_size) {
				size += it;
			}

			std::vector<char> full_tree (size) ;  
			std::vector<const int> displ(mpi_size, 0);

			MPI_Allgatherv(my_info.data(), my_info.size(), MY_CHAR, full_tree.data() , all_info_size.data() , displ, MPI_CHAR , MPI_COMM_WORLD);

			const int structure_size = minimal_info_size(); 
			minimal_info* ptr = full_tree.data();
			for(int mpi = 0 ; mpi < mpi_size ; mpi++)
			{
				if(isMaster())
				{
					std::cout << std::endl << " MPI : " << mpi << std::endl; 
				}
				for(int it = 0 ; it < all_info_size[mpi] ; it += structure_size)
				{
					ptr->print();
					ptr++;
				}

			}
		}
	};
};
#endif
