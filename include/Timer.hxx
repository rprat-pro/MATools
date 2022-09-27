#pragma once

#include <chrono>
#include<MATimerNode.hxx>

namespace MATimer
{
	namespace timer
	{
		using duration = std::chrono::duration<double>;
		using steady_clock = std::chrono::steady_clock;
		using time_point = std::chrono::time_point<steady_clock>;
		class Timer
		{
			public:
			Timer(duration * acc);
			void start();
			void end();
			~Timer(); 

			private:
			time_point m_start;
			time_point m_stop;
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
			timer = new Timer(MATimer::timers::get_MATimer_node<T>()->get_ptr_duration());
			timer->start(); // reset start
		}

		template<enumTimer T>
		void end_global_timer()
		{
			assert(T == enumTimer::ROOT);
			auto timer = get_timer<T>();
			timer->end();
		}
	}
};
