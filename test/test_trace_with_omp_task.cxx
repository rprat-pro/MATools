#include <MATools.hxx>
#include <unistd.h>
#include <vector>

void task(double* out, double* in, int n)
{
	using namespace MATools::MATrace;
	const int thread = omp_get_thread_num();
	assert(thread>=0);
	omp_start();
	for(int i = 0; i < n ; i++)
	{
		out[thread] += in[thread];
	}
	omp_stop("task");
}

int main(int argc, char * argv[]) 
{
	using namespace MATools::MATrace;
	MATools::initialize(&argc, &argv);
	Optional::active_MATrace_mode();
	Optional::active_omp_mode();

	int n_threads = -1;
#pragma omp parallel
	{
		n_threads = omp_get_num_threads();
	}
	assert(n_threads >= 1);

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
	MATools::MPI::mpi_finalize();
	return 0;
}
