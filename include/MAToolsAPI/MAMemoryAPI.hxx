#pragma once

#include <vector>
#include <string>
#include <Common/MAMemory.hxx>
#include <Common/MAToolsMPI.hxx>
#include <MAOutput/MAOutput.hxx>
#include <iostream>

std::vector<std::string>& get_mem_labels();

inline
void Add_Mem_Label(std::string a_name)
{
	auto& labels = get_mem_labels();
	labels.push_back(a_name);
}

inline 
void print_mem_labels()
{
	if(MATools::MPI::is_master())
	{
		auto& labels = get_mem_labels();
		for(auto& it : labels)
			std::cout << it << " ";
		std::cout << std::endl; 
	}
}

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
			print_mem_labels();
		}

		/**
		 * @brief This function writes the memory usage points in a file collected via the use of Add_Mem_Points
		 */
		void write_trace_memory_footprint()
		{
			using namespace MATools::MAMemory;
			auto& mem = get_MAFootprint();
			auto& labels = get_mem_labels();  
			if(labels.size() == mem.size())
			{
				write_memory_checkpoints(mem,labels);		
			}
			else
			{
				write_memory_checkpoints(mem);
			}
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

/**
 * @brief Add a memory point (memory usage)
 */
	inline
void Add_Mem_Point(std::string a_name)
{
	using namespace MATools::MAMemory;
	get_MAFootprint().add_memory_checkpoint();	
	Add_Mem_Label(a_name);
}
// Macro-style version
#define ADD_MEMORY_POINT() MATools::MAMemory::get_MAFootprint().add_memory_checkpoint();	
#define WRITE_MEMORY_POINTS() write_memory_checkpoints(MATools::MAMemory::get_MAFootprint());	
#define TRACE_MEMORY_POINTS() print_checkpoints(MATools::MAMemory::get_MAFootprint());	
