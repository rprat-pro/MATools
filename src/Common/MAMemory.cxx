
#include <vector>
#include <cassert>
#include <MAOutput/MAOutput.hxx>
#include <Common/MAMemory.hxx>
#include <Common/MAToolsMPI.hxx>

namespace MATools
{
	namespace MAMemory
	{

		/**
		 * This function creates a rusage variable
		 * @return rusage data type that get memory informations
		 */
		rusage make_memory_checkpoint()
		{
			rusage obj;
			int who = 0;
			[[maybe_unused]] auto res = getrusage(who, &obj);
			assert((res = -1) && "error: getrusage has failed");
			return obj;
		};

		/**
		 * Default constructor that calls default std::vector constructor 
		 */
		MAFootprint::MAFootprint() {}

		/**
		 * This function create a memory checkpoint, i.e. a rusage, and insert it in the data storage 
		 */
		void MAFootprint::add_memory_checkpoint()
		{
			auto obj = make_memory_checkpoint();
			this->push_back(obj);
		}


		/**
		 * Reduce is called to get the total memory footprint among MPI processes for every memory checkpoints
		 * Data are gathered/reduced on the master mpi process 0.
		 * @return a vector with the total memory footprint by memory checkpoints  
		 */
		std::vector<long> MAFootprint::reduce()
		{
			// one value per memory checkpoint
			std::vector<long> ret;
			int nb_points = this->size();
			auto m_data = this->data();
			ret.resize(nb_points);

			// get total memory footprint for every memory checkpoints
			for(int id = 0 ; id < nb_points ; id++)
			{
#ifdef __MPI
				ret[id] = 0;
				MPI_Reduce(&(m_data[id].ru_maxrss), &(ret[id]), 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
#else
				ret[id] = m_data[id].ru_maxrss;
#endif
			}

			return ret;
		}

		/*
		 * The memory footprint is printed for every memory checkpoints
		 * @param f is a mafootprint object that contains memory checkpoints
		 * @see mafootprint
		 */
		void print_checkpoints(MAFootprint& a_f)
		{
			using namespace MATools::MPI;
			// This function extracts the maxmimal data size (footprint) and do a reduction if the MPI feature is activated
			// For every memory checkpoints
			auto obj = a_f.reduce();
			if(is_master())
			{
				std::cout << " List (maximum resident size): ";
				for(auto it : obj)
				{
					std::cout << " " << it;
				}
				std::cout << std::endl;
			}
		};

		/*
		 * The memory footprint is printed where this function is called.
		 * @see MAFootprint
		 */
		void print_memory_footprint()
		{
			// Create an MAFootprint object that will be destroyed at the end
			MAFootprint f;
			// We fill this function with only one checkpoint
			f.add_memory_checkpoint();
			// This function extracts the maxmimal data size (footprint) and do a reduction if the MPI feature is activated
			auto obj = f.reduce();
			// As we have only one memory checkpoint, we get this element
			auto last = obj.back() * 1e-6; // conversion kb to Gb
			MATools::MAOutput::printMessage(" memory footprint: ", last, " GB");
		};

	};
};
