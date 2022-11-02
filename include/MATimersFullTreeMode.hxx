#pragma once

#include <cstring>
#include <string>
#include <vector>

namespace MATools 
{
	namespace MATimer
	{
		namespace FullTreeMode
		{

			constexpr size_t minimal_info_name_size = 64;
			class minimal_info
			{
				public:
					minimal_info();
					minimal_info(std::string, std::size_t);
					void print();
					
					//members
					char m_name[minimal_info_name_size];
					std::uint64_t m_nb_daughter;
			};

			void copy_to_string(const std::string& a_src, char a_dst[minimal_info_name_size]);
			int minimal_info_size() ;
			std::vector<minimal_info> build_my_tree();
			void build_full_tree();
		};
	};
};
