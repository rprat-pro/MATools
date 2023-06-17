#include <vector>
#include <MATools.hxx>
#ifdef __MPI
#include <mpi.h>
#endif

void func_short_name()
{
	START_TIMER("short_name");
}

void func_long_name()
{
	START_TIMER("This_is_a_very_long_name_used_for_debugging_and_bazinga_bazinga_bazinga_This_is_a_very_long_name_used_for_debugging_and_bazinga_bazinga_bazinga_This_is_a_very_long_name_used_for_debugging_and_bazinga_bazinga_bazinga");
	func_short_name();
}


int main(int argc, char * argv[]) 
{
	MATools::initialize(&argc, &argv);
	MATools::MATimer::Optional::disable_write_file();

	func_long_name();

	MATools::finalize();
	MATools::MPI::mpi_finalize();
	return 0;
}
