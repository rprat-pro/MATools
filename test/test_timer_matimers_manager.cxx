#include <vector>
#include <MATools.hxx>
#ifdef __MPI
#include <mpi.h>
#endif

void func_short_name()
{
	Catch_Time_Section("short_name");
}

void func_name()
{
	Catch_Time_Section("name")
	func_short_name();
}


int main(int argc, char * argv[]) 
{
	MATimersManager timers;
	func_name();
	return 0;
}
