#pragma once

#include <MAOutput/MAOutput.hxx>
#include <Common/MAToolsDecoration.hxx>

namespace MATools
{
	namespace MATimer
	{
		class MATimerInfo
		{
			// all public
			public:

			// members
			std::string m_full_name;
			std::size_t m_n_iterations;
			double m_duration; // this value is the maximum duration with MPI
#ifdef __MPI
			double m_mean_duration;
#endif

			// default constructor -- not used
			MATools_DECORATION
			MATimerInfo () {};

			// used constructor
			MATools_DECORATION
			MATimerInfo (std::string a_full_path, std::size_t a_n_iterations,  double a_duration, [[maybe_unused]] double a_mean_duration = 0)
			{
				m_full_name = a_full_path;
				m_n_iterations = a_n_iterations;
				m_duration = a_duration;
#ifdef __MPI
				m_mean_duration = a_mean_duration;
#endif
			}

			// setters and getters
			MATools_DECORATION
			std::string get_name()
			{
				std::string ret = m_full_name;
				return ret;
			}
			
			MATools_DECORATION
			std::size_t get_number_of_iterations()
			{
				std::size_t ret = m_n_iterations;
				return ret;
			}

			MATools_DECORATION
			double get_duration()
			{
				double ret = m_duration;
				return ret;
			}

			MATools_DECORATION
			double get_max_duration()
			{
				double ret = get_duration();
				return ret;
			}

			MATools_DECORATION
			double get_mean_duration()
			{
#ifdef __MPI
				double ret = m_mean_duration;
#else
				double ret = get_duration();
#endif
				return ret;
			}

			// outputs
			void header()
			{
				using namespace MATools::MAOutput;
#ifdef __MPI
				printMessage("full name", "-", "number of iterations", "-", "mean time(s)", "-", "max time(s)");
#else
				printMessage("full name", "-", "number of iterations", "-", "time(s)");
#endif
			}

			void print()
			{
				using namespace MATools::MAOutput;
#ifdef __MPI
				printMessage(get_name(), get_number_of_iterations(), get_mean_duration(), get_max_duration());	
#else
				printMessage(get_name(), get_number_of_iterations(), get_duration());	
#endif
			}
		};
	}
}
