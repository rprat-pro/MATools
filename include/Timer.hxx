#pragma once

#include <chrono>
#include <MATimerNode.hxx>

namespace MATools
{
	namespace MATimer
	{
		using duration = std::chrono::duration<double>;
		using high_resolution_clock = std::chrono::high_resolution_clock;
		using MATime_point = std::chrono::time_point<high_resolution_clock>;
		class Timer
		{
			public:
			Timer(duration * acc);
			void start();
			void end();
			~Timer(); 

			private:
			MATime_point m_start;
			MATime_point m_stop;
			duration * m_duration; 
		};

		template<enumTimer T>
		Timer*& get_timer()
		{
			static Timer* __timer;
			return __timer;
		}

		template<enumTimer T>
		void start_global_timer()
		{
			assert(T == enumTimer::ROOT);
			auto& timer = get_timer<T>(); 
			auto& root_ptr = MATools::MATimer::get_MATimer_node<T>(); 
			assert(root_ptr != nullptr);
			timer = new Timer(root_ptr->get_ptr_duration());
			timer->start(); // reset start
			assert(timer != nullptr);
		}

		template<enumTimer T>
		void end_global_timer()
		{
			assert(T == enumTimer::ROOT);
			auto timer = get_timer<T>();
			assert(timer != nullptr);
			timer->end();
		}
	}
};
