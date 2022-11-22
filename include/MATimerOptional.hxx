#pragma once

namespace MATools
{
	namespace MATimer
	{
		namespace Optional
		{
			extern bool& get_full_tree_mode();
			extern bool& get_print_timetable();
			extern bool& get_write_file();
			
			void active_full_tree_mode();
			void enable_print_timetable();
			void enable_write_file();
			void disable_print_timetable();
			void disable_write_file();
			
			bool is_full_tree_mode();
			bool is_print_timetable();
			bool is_write_file();
		};
	};
};
