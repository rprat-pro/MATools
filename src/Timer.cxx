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
			std::cout << " tic " << std::endl;
			std::cout << this << std::endl;
			m_start = high_resolution_clock::now();
		}

		void Timer::end()
		{
			assert(m_duration != nullptr && "duration has to be initialised");
			std::cout << " toc " << std::endl;
			std::cout << this << std::endl;
			m_stop = high_resolution_clock::now();
			*m_duration += m_stop - m_start;
			assert(m_duration->count() >= 0);
			std::cout << m_duration->count() << std::endl;

		}

		Timer::~Timer() 
		{
			end();
			auto& current_timer = MATimer::timers::get_MATimer_node<CURRENT>();
			current_timer = current_timer->get_mother();
		}
	};
};
