#pragma once
#include <MATimers/MATimers.hxx>
#include <MATimers/MATimersVerbosity.hxx>
#include <MAToolsAPI/MATimersAPI.hxx>
#include <MAToolsAPI/MAMemoryAPI.hxx>

#define MEMVARNAME() CONCAT_COUNTER(MEM)

#define Catch_Section(XNAME)\
	Catch_Time_Section(XNAME);\
  MemorySection MEMVARNAME()(XNAME); 

struct MemorySection
{
  std::string m_name;
  MemorySection(std::string name) : m_name(name)
  {
    std::string start_name = "Start_" + name;
    Add_Mem_Point(start_name); 
  }
  ~MemorySection()
  {
    std::string end_name = "End_" + m_name;
    Add_Mem_Point(end_name);
  }
};


class MAToolsManager : public MATimersManager, public MAMemoryManager
{
	public:
		/**
		 * @brief Constructor for MATimersManager.
		 * Initializes the MATimersManager by initializing the timer.
		 */
		MAToolsManager() : MATimersManager(), MAMemoryManager() {}

		/**
		 * @brief Constructor for MATimersManager.
		 * Initializes the MATimersManager by initializing the timer.
		 */
		MAToolsManager([[maybe_unused]] int *argc, [[maybe_unused]]char ***argv) : MATimersManager(argc, argv), MAMemoryManager() {} 

    void Display()
    {
      this->print_trace_memory_footprint();
      this->write_trace_memory_footprint();
			//MATools::MATimer::finalize();
      //this->disable_timetable(); // avoid to call print_timetable in the MATimersManager destructor
      //this->disable_write_file(); // same
    }

		/**
		 * @brief Destructor for MATimersManager.
		 * Finalizes the MATimersManager by cleaning up the timer resources.
		 */
		~MAToolsManager() 
		{
      // call finalize (MATimersAPI)
		}
};

