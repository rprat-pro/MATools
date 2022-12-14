#include<Timer.hxx>

namespace MATools
{
	namespace MATimer
	{
		Timer::Timer(duration * acc) 
		{
			m_duration = acc;
			start();
		}

		void Timer::start()
		{
			m_start = high_resolution_clock::now();
		}

		void Timer::end()
		{
			assert(m_duration != nullptr && "duration has to be initialised");
			m_stop = high_resolution_clock::now();
			*m_duration += m_stop - m_start;
			assert(m_duration->count() >= 0);
		}

		Timer::~Timer() 
		{
			end();
			auto& current_timer = MATools::MATimer::get_MATimer_node<CURRENT>();
			current_timer = current_timer->get_mother();
		}

		void BasicTimer::start()
		{
			m_start = high_resolution_clock::now();
		}

		void BasicTimer::end()
		{
			m_stop = high_resolution_clock::now();
		}

		double BasicTimer::get_duration()
		{
			duration measure = m_stop - m_start;
			double ret = measure.count();
			return ret;
		}
	};
};
