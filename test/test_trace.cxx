#include<MATimers.hxx>
#include<MATrace.hxx>
#include<MAToolsMPI.hxx>
#include<chrono>
#include <unistd.h>
void f1()
{
	usleep(1);
}

void f2()
{
	usleep(2);
}


void f3()
{
	usleep(2);
}

int main(int argc, char * argv[]) 
{
	MATools::MATimer::initialize(&argc,&argv);
#ifdef __MPI
	using namespace MATools::MPI;
	for(int i = 0; i < 20 ; i++)
	{
		f3();
	}

	if(!is_master())
	{
		for(int i = 0; i < 20 ; i++)
		{
			MATools::MATrace::start();
			f1();
			MATools::MATrace::stop("f1");
			MATools::MATrace::start();
			f2();
			MATools::MATrace::stop("f2");
		}
	}
	else
	{
		for(int i = 0; i < 20 ; i++)
		{
			MATools::MATrace::start();
			f3();
			MATools::MATrace::stop("f3");
		}
	}
#else
	for(int i = 0; i < 20 ; i++)
	{
		MATools::MATrace::start();
		f3();
		MATools::MATrace::stop("f3");
	}
#endif
	MATools::MATimer::finalize();	
	return 0;
}
