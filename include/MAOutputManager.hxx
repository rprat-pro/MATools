#pragma once

#include <algorithm>
#include <omp.h>
#include <fstream>
#include <Column.hxx>
#include <MATimerNode.hxx>
#include <MATimerOptional.hxx>
#include <MATimerInfo.hxx>

namespace MATools
{
	namespace MAOutputManager
	{
		using MATools::MATimer::MATimerNode;

		std::string build_name();
		std::string build_current_mpi_name();
		void print_timetable();
		void write_file();
		void write_file(std::string a_name);
		void write_debug_file();
		void write_debug_file(std::string);
		std::vector<MATools::MATimer::MATimerInfo> get_filtered_timers(std::string);
		void print_filtered_timers(std::string);

		template<enumTimer T>
			void print_timetable()
			{
				/*************************************************
				// DEBUG
				using namespace MATools::MAOutput;
				if(T == enumTimer::MPI) printMessage("enum::MPI");
				else printMessage("enum::ROOT");
				*************************************************/
				
				MATimerNode* root_timer = MATools::MATimer::get_MATimer_node<T>();
				assert(root_timer != nullptr);
				double runtime = root_timer->get_duration();
				runtime = MATools::MPI::reduce_max(runtime); // if MPI, else return runtime

				auto my_print = [](MATimerNode* a_ptr, size_t a_shift, double a_runtime)
				{
					a_ptr->print(a_shift, a_runtime);
				};


				using namespace MATools::MATimer::Optional;
				const bool sorted_by_name= is_full_tree_mode();
				
				// the sorting is finally disabled
				[[maybe_unused]] auto sort_comp = [sorted_by_name] (MATimerNode* a_ptr, MATimerNode* b_ptr)
				{
					if(sorted_by_name) return a_ptr->get_name() > b_ptr->get_name();
					else return a_ptr->get_duration() > b_ptr->get_duration() ;
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
				recursive_call(my_print, root_timer, count, runtime);
				//recursive_sorted_call(my_print, sort_comp, root_timer, count, runtime);
				root_timer->print_ending(count);
			}

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
