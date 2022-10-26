#include <MATrace.hxx>
#include <MATraceColor.hxx>
#include <random>

namespace MATimer
{
	namespace MATrace
	{
		using high_resolution_clock = std::chrono::high_resolution_clock;
		using time_point = std::chrono::time_point<high_resolution_clock>;

		MATrace_point::MATrace_point()
		{   
			m_time = std::chrono::system_clock::now();
		}

		time_point& MATrace_point::data()
		{
			return m_time;
		}

		const time_point MATrace_point::data() const
		{
			return m_time;
		}

		void MATrace_point::set_proc()
		{
#ifdef __MPI
			int rank;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			m_proc = rank;
#else 
			m_proc = 0;
#endif
		}

		void MATrace_point::set_proc(int a_rank)
		{
			m_proc = a_rank;
		}

		int MATrace_point::get_proc() const
		{
			return m_proc;
		}

		void vite_event::add(char a_name[64])
		{
			std::string name = a_name;
			auto it = m_data.find(name);
			if(it == m_data.end())
				m_data[name] = m_acc++;
		}

		int vite_event::operator[] (char a_name[64])
		{
			std::string name = a_name;
			return m_data[name];
		}

		void vite_event::write_items(std::ofstream& a_out)
		{
			for(auto& it : m_data)
			{
				a_out 	<< "%EventDef PajeSetState " << it.second << std::endl;
				a_out	<< "% Time date"<< std::endl;  
				a_out	<< "% Type string"<< std::endl;
				a_out	<< "% Container string" << std::endl;
				a_out	<< "% Value string"  << std::endl;
				a_out	<< "%EndEventDef"<< std::endl;
			}
		}

		void vite_event::write_colors(std::ofstream& a_out)
		{
			MATraceRGB idle_color = get_idle_color();
			a_out << "6 idle ST_ThreadState 'idle'  '" << idle_color.r << " " << idle_color.g << " " << idle_color.b << "'" << std::endl;
			for(auto& it : m_data)
			{
				auto idx = it.second;
				auto name = it.first;
				MATraceRGB color = get_default_color(idx - _sstart); // _start is defined in MATrace.hxx and idx >= _start
				a_out << "6 " <<name<<" ST_ThreadState '"<< name <<"'  '" << color.r << " " << color.g << " " << color.b << "'" << std::endl;
			}
		}

		MATrace_section::MATrace_section(char a_name[64], const MATrace_point& a_ref, const MATrace_point& a_start, const MATrace_point& a_end)
		{
			std::chrono::duration<double> duration ;

			// compute start
			duration = a_start.data() - a_ref.data();
			m_start = duration.count();
			assert(m_start >= 0);

			// compute end
			duration = a_end.data() - a_ref.data();
			m_end = duration.count();
			assert(m_end > m_start);

			// set proc id and name
			m_proc_id = a_ref.get_proc();
			strncpy(m_name, a_name, 64);
		}

		void MATrace_section::write(std::ofstream& a_out)
		{
			a_out   << m_name       << " "
				<< m_proc_id    << " "
				<< m_start      << " "
				<< m_end        << " "
				<< std::endl;
		}

		void MATrace_section::write(std::ofstream& a_out, vite_event& a_ve)
		{
			a_out   << a_ve[m_name] << " "
				<< m_start    << " "
				<< "ST_ThreadState "
				<< "C_Thread" << m_proc_id << " "<< m_name
				<< std::endl;
			a_out   << "10 "
				<< m_end    << " "
				<< "ST_ThreadState "
				<< "C_Thread" << m_proc_id << " idle"
				<< std::endl;
		}

		void MATrace_section::set_proc(int a_rank)
		{
			m_proc_id = a_rank;
		}

		extern MATrace_point& get_ref_MATrace_point()
		{
			static MATrace_point instance;
			return instance;
		}

		extern MATrace_point& get_MATrace_point()
		{
			static MATrace_point instance;
			return instance;
		}

		extern Trace& get_local_MATrace()
		{
			static Trace instance;
			return instance;
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
			auto section = MATrace_section(name, ref, start, end);
			auto& local_MATrace = get_local_MATrace();
			local_MATrace.push_back(section);
		}
		/*
		   template<typename Fun>
		   void MATrace_kernel (Fun& a_fun, std::string a_name)
		   {
		   start();
		   a_fun();
		   stop(a_name);
		   }

		   template<typename Fun, typename... Args>
		   void MATrace_functor (Fun& a_fun, Args&&... a_args)
		   {
		   auto start = MATrace_point();
		   a_fun(std::forward<Args>(a_args)...);
		   auto end = MATrace_point();
		   }
		   */
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
			MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
			for(int i = 0 ; i < mpi_size ; i++)
			{
				out << "7  0.000000 C_Thread" << i <<" CT_Thread C_Prog 'Thread " << i <<"'" << std::endl;
				out << "51 0.000000 V_Sem C_Thread" << i << " 0.0" << std::endl;
			}
		}

		void ending(std::ofstream& out, double last)
		{
			int mpi_size = -1;
			MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
			for(int i = 0 ; i < mpi_size ; i++)
				out << "8 " << last << " C_Thread" << i << " CT_Thread" << std::endl;
			out <<"8 " << last << " C_Prog CT_Prog" << std::endl;
		}

		void finalize()
		{
			using namespace MATimer::mpi;
			auto& local_MATrace = get_local_MATrace();
			const int local_size = local_MATrace.size();
			int local_byte_size = sizeof(MATrace_section) * local_size;

#ifdef __MPI
			// update proc id
			const auto my_rank = get_rank();
			for(auto& it : local_MATrace)
			{
				it.set_proc(my_rank);
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
