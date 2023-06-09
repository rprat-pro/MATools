#include <random>
#include <omp.h>
#include <iostream>

#include <MATrace/MATrace.hxx>
#include <Common/MAToolsMPI.hxx>
#include <MATrace/MATraceColor.hxx>
#include <MATrace/MATraceOptional.hxx>
#include <MATrace/MATraceInstance.hxx>

namespace MATools
{
	namespace MATrace
	{
		/**
		 * @brief This function returns the current thread id according to the omp thread number if OpenMP is activated, 0 otherwise.
		 */
		const int get_thread_id()
		{
#ifdef _OPENMP
			const int ret = omp_get_thread_num();
#else
			const int ret = 0;
#endif /* OpenMP */
			return ret;
		}

		/**
		 * @brief This function returns the number of threads if OpenMP is activated, 1 otherwise.
		 */
		const int get_number_of_threads()
		{
#ifdef _OPENMP
			const int ret = omp_get_num_threads();
#else
			const int ret = 1;
#endif /* OpenMP */
			return ret;
		}


		void start()
		{
			auto& start = get_MATrace_point();
			start = MATrace_point();
		}

		void stop(std::string a_name)
		{
			auto end = MATrace_point();
			auto start = get_MATrace_point();
			auto ref = get_ref_MATrace_point();
			char name[64];
			strncpy(name, a_name.c_str(), 64);
			name[63]='\0'; // warning issue without this line
			auto section = MATrace_section(name, ref, start, end);
			auto& local_MATrace = get_local_MATrace();
			local_MATrace.push_back(section);
		}

		void omp_start()
		{
			auto& start = get_MATrace_omp_point();
			const auto id = get_thread_id();
			start[id] = MATrace_point();
		}

		void omp_stop(std::string a_name)
		{
			auto id = get_thread_id();
			auto end = MATrace_point();
			auto start = get_MATrace_omp_point();
			auto ref = get_ref_MATrace_point();
			char name[64];
			strncpy(name, a_name.c_str(), 64);
			name[63]='\0'; // warning issue without this line
			auto section = MATrace_section(name, ref, start[id], end);
			section.set_proc(id);
			auto& omp_MATrace = get_omp_MATrace();
			omp_MATrace[id].push_back(section);
		}

		void init_omp_trace()
		{
			auto& omp_trace = get_omp_MATrace();
			size_t num_threads;
#pragma omp parallel
			{	
				num_threads = get_number_of_threads();
				omp_trace.resize(num_threads);
#pragma omp for
				for(unsigned int id = 0 ; id < omp_trace.size() ; id++)
					omp_trace[id] = Trace();
			}
			auto& omp_points = get_MATrace_omp_point();
			omp_points.resize(num_threads);
		}

		void initialize()
		{
			auto start = get_ref_MATrace_point();
			start.set_proc();
			auto& local_MATrace = get_local_MATrace();
			local_MATrace.reserve(120000); 
		}

		void header(std::ofstream& out, vite_event& event)
		{
			out << "%EventDef PajeDefineContainerType 1" << std::endl;
			out << "% Alias string " << std::endl;
			out << "% ContainerType string " << std::endl;
			out << "% Name string " << std::endl;
			out << "%EndEventDef " << std::endl;
			out << "%EventDef PajeDefineStateType 3" << std::endl;
			out << "% Alias string " << std::endl;
			out << "% ContainerType string " << std::endl;
			out << "% Name string " << std::endl;
			out << "%EndEventDef " << std::endl;
			out << "%EventDef PajeDefineEntityValue 6" << std::endl;
			out << "% Alias string  " << std::endl;
			out << "% EntityType string  " << std::endl;
			out << "% Name string  " << std::endl;
			out << "% Color color " << std::endl;
			out << "%EndEventDef  " << std::endl;
			out << "%EventDef PajeCreateContainer 7" << std::endl;
			out << "% Time date  " << std::endl;
			out << "% Alias string  " << std::endl;
			out << "% Type string  " << std::endl;
			out << "% Container string  " << std::endl;
			out << "% Name string  " << std::endl;
			out << "%EndEventDef  " << std::endl;
			out << "%EventDef PajeDestroyContainer 8" << std::endl;
			out << "% Time date  " << std::endl;
			out << "% Name string  " << std::endl;
			out << "% Type string  " << std::endl;
			out << "%EndEventDef  " << std::endl;
			out << "%EventDef PajeSetState 10 " << std::endl;
			out << "% Time date  " << std::endl;
			out << "% Type string  " << std::endl;
			out << "% Container string  " << std::endl;
			out << "% Value string  " << std::endl;
			out << "%EndEventDef" << std::endl;
			out << "%EventDef PajeDefineVariableType 50 " << std::endl;
			out << "% Alias string" << std::endl;
			out << "% Name  string" << std::endl;
			out << "% ContainerType string " << std::endl;
			out << "%EndEventDef " << std::endl;
			out << "%EventDef PajeSetVariable 51" << std::endl;
			out << "% Time date " << std::endl;
			out << "% Type string " << std::endl;
			out << "% Container string " << std::endl;
			out << "% Value double " << std::endl;
			out << "%EndEventDef  " << std::endl;
			out << "%EventDef PajeAddVariable 52" << std::endl;
			out << "% Time date " << std::endl;
			out << "% Type string " << std::endl;
			out << "% Container string " << std::endl;
			out << "% Value double " << std::endl;
			out << "%EndEventDef  " << std::endl;
			out << "%EventDef PajeSubVariable 53" << std::endl;
			out << "% Time date " << std::endl;
			out << "% Type string " << std::endl;
			out << "% Container string " << std::endl;
			out << "% Value double " << std::endl;
			out << "%EndEventDef" << std::endl;
			event.write_items(out);
			out << "1 CT_Prog   0       'Program'" << std::endl;
			out << "1 CT_Thread CT_Prog 'Thread'" << std::endl;
			out << "3 ST_ThreadState CT_Thread 'Thread State'" << std::endl;
			event.write_colors(out);
			out << "50 V_Sem Semaphore CT_Thread" << std::endl;
			out << "7 0.000000 C_Prog CT_Prog 0 'Programme'" << std::endl;
			int mpi_size = -1;
			if(Optional::is_omp_mode())
			{
#pragma omp parallel
				mpi_size = get_number_of_threads();
			}
			else
			{
#ifdef __MPI
				MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
#else
				mpi_size=1;
#endif
			}
			for(int i = 0 ; i < mpi_size ; i++)
			{
				out << "7  0.000000 C_Thread" << i <<" CT_Thread C_Prog 'Thread " << i <<"'" << std::endl;
				out << "51 0.000000 V_Sem C_Thread" << i << " 0.0" << std::endl;
			}
		}

		void ending(std::ofstream& out, double last)
		{
			int mpi_size = -1;
			if(Optional::is_omp_mode())
			{
#pragma omp parallel
				mpi_size = get_number_of_threads();
			}
			else
			{
#ifdef __MPI
				MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
#else
				mpi_size=1;
#endif
			}
			for(int i = 0 ; i < mpi_size ; i++)
				out << "8 " << last << " C_Thread" << i << " CT_Thread" << std::endl;
			out <<"8 " << last << " C_Prog CT_Prog" << std::endl;
		}

		void finalize()
		{
			using namespace MATools::MPI;

			// no MATrace
			if(!Optional::is_MATrace_mode())
			{
				return;
			}

			// We copy data from omp traces to the master trace
			if(Optional::is_omp_mode())
			{
				auto& master_trace = get_local_MATrace();
				auto& omp_traces = get_omp_MATrace();
				for(auto& it : omp_traces)
					master_trace.insert(master_trace.end(),it.begin(), it.end());
			}

			auto& local_MATrace = get_local_MATrace();
#ifdef __MPI
			const int local_size = local_MATrace.size();
			int local_byte_size = sizeof(MATrace_section) * local_size;

			// update proc id
			const auto my_rank = get_rank();
			bool omp_mode = Optional::is_omp_mode();
			if(!omp_mode)
			{	
				for(auto& it : local_MATrace)
				{
					it.set_proc(my_rank);
				}
			}
			// all data are sent to the process 0
			// we gather the sizes for each process
			int mpi_size = -1;
			MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);

			assert(mpi_size >= 0);	
			std::vector<int> sizes (mpi_size,0);
			std::vector<int> dists (mpi_size,0);
			MPI_Gather(
					&local_byte_size, 1, MPI_INT, 
					sizes.data(), 1, MPI_INT, 
					0, MPI_COMM_WORLD
					);

			int total_byte_size = 0;
			int acc = 0;
			for(size_t i = 0 ; i < sizes.size() ; i++)
			{
				total_byte_size += sizes[i];
				dists[i]=acc;
				acc += sizes[i];
			}
			int total_size = total_byte_size / sizeof(MATrace_section);
			std::vector<char> recv (total_byte_size, 0);
			MPI_Gatherv(
					local_MATrace.data(), local_byte_size, MPI_CHAR, 
					recv.data(), sizes.data(), dists.data(), MPI_CHAR, 
					0, MPI_COMM_WORLD
					);


			MATrace_section * ptr = (MATrace_section*)recv.data() ;
#else
			MATrace_section * ptr = local_MATrace.data();
			int total_size = local_MATrace.size();
#endif

			// write trace header -> trace core -> end
			if(is_master())
			{
				std::ofstream out ("MATrace.txt", std::ofstream::out);
				vite_event event;
				for(int it = 0 ; it < total_size ; it++)
					event.add(ptr[it].m_name);

				header(out,event);


				for(int it = 0 ; it < total_size ; it++)
					ptr[it].write(out, event);

				// last time
				double last = 0;
				for(int it = 0 ; it < total_size ; it++)
				{
					last = std::max(last,ptr[it].m_end);
				}

				ending(out, last);
			}
		}
	}
};
