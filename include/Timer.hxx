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
	
		template<typename T>	
		class BaseTimer
		{
			public:
			virtual void start() = 0;
			virtual void end() = 0;

			protected:
			T m_start;
			T m_stop;
		};
		
		class Timer : public BaseTimer<MATime_point>
		{
			public:
			Timer(duration * acc);
			void start();
			void end();
			~Timer(); 

			private:
			duration * m_duration; 
		};

		//Regular timer
		class BasicTimer : public BaseTimer<MATime_point>
		{
			public:
			void start();
			void end();
			double get_duration();
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
