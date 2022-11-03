#include <MATools.hxx>
#include <MATraceOptional.hxx>
#include <unistd.h>
#include <vector>

void task(double* out, double* in, int n)
{
	int thread = omp_get_thread_num();
	MATools::MATrace::omp_start();
	for(int i = 0; i < n ; i++)
	{
		out[thread] += in[thread];
	}
	MATools::MATrace::omp_stop("task");
}

int main(int argc, char * argv[]) 
{
	MATools::initialize(&argc, &argv);
	MATools::MATrace::Optional::active_MATrace_mode();
	MATools::MATrace::Optional::active_omp_mode();

	int n_threads;
#pragma omp parallel
	n_threads = omp_get_num_threads();

	std::vector<double> in(n_threads, n_threads);
	std::vector<double> out(n_threads, n_threads);
	constexpr int N = 1000;
	constexpr int L = 1000;

#pragma omp parallel
#pragma omp master
	for(int i = 0; i < N ; i++)
	{
#pragma omp task
		{

			task(out.data(), in.data(), L);
		}
	}
	MATools::finalize();	
	return 0;
}
