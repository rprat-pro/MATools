#include <vector>
#include <MATools.hxx>
#ifdef __MPI
#include <mpi.h>
#endif

int main(int argc, char * argv[]) 
{
	MAToolsManager manager(&argc,&argv);
	
	for(int i = 0; i < 100 ; i++)
	{
    std::string name = "kernel_" + std::to_string(i);
    Catch_Section(name);
		std::vector<char> vec(10000*i);
	}
	manager.Display();
	return 0;
}
