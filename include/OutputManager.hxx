#pragma once

#include <algorithm>
#include <omp.h>
#include <fstream>
#include <Column.hxx>
#include <MATimerNode.hxx>


namespace MATimer
{
	namespace outputManager
	{
		using MATimer::timers::MATimerNode;

		std::string build_name();
		void print_timetable();
		void write_file();
		void write_file(std::string a_name);

		template<typename Func, typename... Args>
		void recursive_call(Func& func, MATimerNode* ptr, Args&... arg)
		{
			func(ptr, arg...);
			auto& daughters = ptr->get_daughter();
			for(auto& it: daughters)
			{
				assert(it != nullptr);
				recursive_call(func,it,arg...);
			}
		}

		template<typename Func, typename Sort, typename... Args>
		void recursive_sorted_call(Func& func, Sort mySort, MATimerNode* ptr, Args&... arg)
		{
			func(ptr, arg...);
			auto& daughters = ptr->get_daughter();
			std::sort(daughters.begin(),daughters.end(),mySort);
			for(auto& it:daughters)
			{
				assert(it != nullptr);
				recursive_sorted_call(func, mySort, it,arg...);
			}
		}


	};
};
