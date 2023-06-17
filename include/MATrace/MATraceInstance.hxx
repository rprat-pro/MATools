#pragma once

#include <MATrace/MATraceTypes.hxx>

/**
 * @namespace MATools
 * @brief Namespace containing utility tools for various purposes.
 */
namespace MATools
{
	/**
	 * @namespace MATrace
	 * @brief Namespace containing MATrace-related utilities.
	 */
	namespace MATrace
	{
		    /**
     * @brief Retrieves a reference to the MATrace_point object.
     * This function returns a reference to the MATrace_point object.
     * @return A reference to the MATrace_point object.
     */
    MATrace_point& get_ref_MATrace_point();

    /**
     * @brief Retrieves the current MATrace_point object for current thread ID.
     * This function returns the MATrace_point object.
     * @return The MATrace_point object.
     */
    MATrace_point& get_MATrace_point();

    /**
     * @brief Retrieves the local MATrace object depending on the OpenMP thread ID.
     * This function returns the local MATrace object.
     * @return The local MATrace object.
     */
    Trace& get_local_MATrace();

    /**
     * @brief Retrieves the vector of MATrace_point objects for OpenMP.
     * This function returns the vector of MATrace_point objects for OpenMP.
     * @return The vector of MATrace_point objects for OpenMP.
     */
    std::vector<MATrace_point>& get_MATrace_omp_point();

    /**
     * @brief Retrieves the vector of Trace objects for OpenMP.
     * This function returns the vector of Trace objects for OpenMP.
     * @return The vector of Trace objects for OpenMP.
     */
    std::vector<Trace>& get_omp_MATrace();
	};
}
