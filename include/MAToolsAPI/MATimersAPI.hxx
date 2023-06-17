#pragma once
#include <MATimers/MATimers.hxx>
#include <MATimers/MATimersVerbosity.hxx>

#ifdef NO_TIMER
// do nothing
#define START_TIMER(XNAME) 
#define Catch_Time_Section(X) 
#define Catch_Nested_Time_Section(X) 

/**
 * @brief This function captures the runtime of a given section.
 * @param [in] lambda section that the user wants to measure.
 * @return The runtime of lambda 
 */
	template<typename Lambda>
double chrono_section(Lambda&& lambda_function) {}

/**
 * @brief This function captures the runtime of a given section and add it in the current MATimerNode named a_name.
 * @param [in] a_name of the chrono section measured.
 * @param [in] a_lambda is the section captured.
 */
	template<typename Lambda>
void add_capture_chrono_section(std::string a_name, Lambda&& a_lambda_function) {}

#else /* NO_TIMER */

#define Catch_Time_Section(XNAME)\
	auto& current = MATools::MATimer::get_MATimer_node<CURRENT>();\
	assert(current != nullptr && "do not use an undefined MATimerNode");\
	current = current->find(XNAME);\
	MATools::MATimer::print_verbosity_level_1(XNAME, current->get_level());\
	MATools::MATimer::Timer non_generic_name(current->get_ptr_duration());

#define Catch_Nested_Time_Section(XNAME)\
	auto& nested_current = MATools::MATimer::get_MATimer_node<CURRENT>();\
	assert(nested_current != nullptr && "do not use an undefined nested MATimerNode");\
	nested_current = nested_current->find(XNAME);\
	MATools::MATimer::print_verbosity_level_1(XNAME, nested_current->get_level());\
	MATools::MATimer::Timer non_nested_generic_name(nested_current->get_ptr_duration());

#define START_TIMER(XNAME) Catch_Time_Section(XNAME)

/**
 * @brief This function captures the runtime of a given section.
 * @param [in] lambda section that the user wants to measure.
 * @return The runtime of lambda 
 */
	template<typename Lambda>
double chrono_section(Lambda&& lambda_function)
{
	using namespace MATools::MATimer;
	double ret;
	BasicTimer timer;
	timer.start();
	lambda_function();
	timer.end();
	ret = timer.get_duration();
	return ret;	
}

/**
 * @brief This function captures the runtime of a given section and add it in the current MATimerNode named a_name.
 * @param [in] a_name of the chrono section measured.
 * @param [in] a_lambda is the section captured.
 */
	template<typename Lambda>
void add_capture_chrono_section(std::string a_name, Lambda&& a_lambda_function) 
{
	Catch_Time_Section(a_name);
	a_lambda_function();
}
#endif /* NO_TIMER */

/** Create an object MATimersManager:
 * To enhance your application's functionality, you need to create an object called "MATimersManager" and place it at the beginning of your application. 
 * The MATimersManager serves as a tool for monitoring and analyzing the performance of your application. 
 * By initializing it at the outset, you can ensure that all relevant metrics and data are captured from the start and throughout the execution of your application. 
 * This allows for comprehensive profiling and optimization of your application's performance.
 */
class MATimersManager
{
	public:
		/**
		 * @brief Constructor for MATimersManager.
		 * Initializes the MATimersManager by initializing the timer.
		 */
		MATimersManager() 
		{
			MATools::MATimer::initialize();
		}

		/**
		 * @brief Disables printing of the timetable.
		 * This function disables the printing of the timetable during profiling.
		 */
		void disable_timetable()
		{
			MATools::MATimer::Optional::disable_print_timetable();
		}

		/**
		 * @brief Disables writing data to a file.
		 * This function disables writing profiling data to a file.
		 */
		void disable_write_file()
		{
			MATools::MATimer::Optional::disable_write_file();
		}

		/**
		 * @brief Destructor for MATimersManager.
		 * Finalizes the MATimersManager by cleaning up the timer resources.
		 */
		~MATimersManager() 
		{
			MATools::MATimer::finalize();
		}
};

