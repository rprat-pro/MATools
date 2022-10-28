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
			void disable_print_timetable();
			void disable_write_file();
			
			bool is_full_tree_mode();
			bool is_print_timetable();
			bool is_write_file();
		};
	};
};
