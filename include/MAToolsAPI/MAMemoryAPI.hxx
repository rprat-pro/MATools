#pragma once


class MAMemoryManager
{
	public:
	/**
	* @brief default constructor
	*/
	MAMemoryManager()
	{
		using namespace MATools::MAMemory;
		using namespace MATools::MAOutput;
		// create a static MAFootprint variable
		get_MAFootprint();
		printMessage("MATools_LOG:","The memory profiler is activated");
	}

	/**
	 * @brief This function prints the memory usage points in a file collected via the use of Add_Mem_Points
	 */
	void print_trace_memory_footprint()
	{
		using namespace MATools::MAMemory;
		auto& mem = get_MAFootprint();
		print_checkpoints(mem);		
	}

	/**
	 * @brief This function writes the memory usage points in a file collected via the use of Add_Mem_Points
	 */
	void write_trace_memory_footprint()
	{
		using namespace MATools::MAMemory;
		auto& mem = get_MAFootprint();
		write_memory_checkpoints(mem);		
	}

};

/**
 * @brief Add a memory point (memory usage)
 */
inline
void Add_Mem_Point()
{
	using namespace MATools::MAMemory;
	get_MAFootprint().add_memory_checkpoint();	
}

// Macro-style version
#define ADD_MEMORY_POINT() MATools::MAMemory::get_MAFootprint().add_memory_checkpoint();	
#define WRITE_MEMORY_POINTS() write_memory_checkpoints(MATools::MAMemory::get_MAFootprint());	
#define TRACE_MEMORY_POINTS() print_checkpoints(MATools::MAMemory::get_MAFootprint());	
