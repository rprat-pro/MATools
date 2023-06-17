#include <vector>
#include <MATools.hxx>
#ifdef __MPI
#include <mpi.h>
#endif

using namespace MATools::MATimer;

void nested_function_level_v3()
{
	Catch_Time_Section("nested_function_level_v3");
	for(int i = 0; i < 100 ; i++)
	{
		Catch_Nested_Time_Section("hybrid_nested_function");
	}
}
void nested_function_level_v2()
{
	Catch_Time_Section("nested_function_level_v2");
	for(int i = 0; i < 100 ; i++)
	{
		HybridTimer g("hybrid_nested_function");
		g.start_time_section();
		g.end_time_section();
	}
}

void nested_function_level_v1()
{
	Catch_Time_Section("nested_function_level_v1");
	HybridTimer h("hybrid_nested_function");
	for(int i = 0; i < 100 ; i++)
	{
		h.start_time_section();
		h.end_time_section();
	}
}

void nested_function()
{
	Catch_Time_Section("nested_function");
	nested_function_level_v1();
	nested_function_level_v2();
	nested_function_level_v3();
}

int main(int argc, char * argv[]) 
{
	MATools::initialize(&argc,&argv);
	HybridTimer first_timer("first_timer");
	first_timer.start_time_section();
	nested_function();
	first_timer.end_time_section();
	MATools::finalize();
	MATools::MPI::mpi_finalize();
	return 0;
}
