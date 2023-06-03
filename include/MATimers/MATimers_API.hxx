#pragma once
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
