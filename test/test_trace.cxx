#include<MATimers.hxx>
#include<MATrace.hxx>
#include<MATimerMPI.hxx>
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
	MATimer::timers::initialize(&argc,&argv);
	MATimer::MATrace::initialize();
#ifdef __MPI
	using namespace MATimer::mpi;
	for(int i = 0; i < 20 ; i++)
	{
		f3();
	}

	if(!is_master())
	{
		for(int i = 0; i < 20 ; i++)
		{
			MATimer::MATrace::start();
			f1();
			MATimer::MATrace::stop("f1");
			MATimer::MATrace::start();
			f2();
			MATimer::MATrace::stop("f2");
		}
	}
	else
	{
		for(int i = 0; i < 20 ; i++)
		{
			MATimer::MATrace::start();
			f3();
			MATimer::MATrace::stop("f3");
		}
	}
#else
	for(int i = 0; i < 20 ; i++)
	{
		MATimer::MATrace::start();
		f3();
		MATimer::MATrace::stop("f3");
	}
#endif
	MATimer::MATrace::finalize();	
	MATimer::timers::finalize();	

	return 0;
}
