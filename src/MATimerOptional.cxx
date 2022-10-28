#include <MATimerOptional.hxx>
#include <MAOutput.hxx>

namespace MATools
{
	namespace MATimer
	{
		namespace Optional
		{
			extern bool& get_full_tree_mode()
			{
				static bool _ftm = false;
				return _ftm;
			}

			void active_full_tree_mode()
			{
				bool& mode = get_full_tree_mode();
				mode = true;
				MATools::MAOutput::printMessage("INFO: full tree mode is activated");
			}

			bool is_full_tree_mode()
			{
				bool ret = get_full_tree_mode();
				return ret;
			}
			
			extern bool& get_print_timetable()
			{
				static bool _pt = true;
				return _pt;
			}

			void disable_print_timetable()
			{
				bool& mode = get_print_timetable();
				mode = false;
				MATools::MAOutput::printMessage("INFO: print timetable mode is disabled");
			}

			bool is_print_timetable()
			{
				bool ret = get_print_timetable();
				return ret;
			}
			
			extern bool& get_write_file()
			{
				static bool _wf = true;
				return _wf;
			}

			void disable_write_file()
			{
				bool& mode = get_write_file();
				mode = false;
				MATools::MAOutput::printMessage("INFO: write file mode is disabled");
			}

			bool is_write_file()
			{
				bool ret = get_write_file();
				return ret;
			}
		}
	}
}
