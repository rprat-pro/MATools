#include<Timer.hxx>

namespace MATimer
{
	namespace timer
	{
		Timer::Timer(duration * acc) 
		{
			m_duration = acc;
			start();
		}

		void Timer::start()
		{
			m_start = steady_clock::now();
		}

		void Timer::end()
		{
			assert(m_duration != nullptr && "duration has to be initialised");
			m_stop = steady_clock::now();
			*m_duration += m_stop - m_start;
			assert(m_duration->count() >= 0);
		}

		Timer::~Timer() 
		{
			end();
			auto& current_timer = MATimer::timers::get_MATimer_node<CURRENT>();
			current_timer = current_timer->get_mother();
		}
	};
};
