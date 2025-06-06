if(MATOOLS_MPI)
	set(LIBS MATools mpi)
	function(add_test_mpi name)
		add_test(NAME ${name}_4mpi COMMAND mpirun -n 4 ./check_${name})
	endfunction()
else()
	set(LIBS "MATools")
	function(add_test_mpi name)
	endfunction()
endif()


######## UNIT TESTS


if(NOT MATOOLS_MPI)
	find_package(Catch2 REQUIRED)
	add_executable(libUnitTests unit_tests.cxx)
	target_include_directories(libUnitTests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/)
	target_link_libraries(libUnitTests Catch2::Catch2 pthread MATools)
	include(CTest)
	include(Catch)
	catch_discover_tests(libUnitTests)
	add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
      DEPENDS libUnitTests)
endif()

#########

function(build_test name)
	add_executable(check_${name} ${name}.cxx)
	target_include_directories(check_${name} PUBLIC ${HEADERS} ${TCLAP_INCLUDE_DIR})
	target_link_libraries(check_${name} PUBLIC ${LIBS})
	#target_link_libraries(check_${name} MATools mpi)
	add_test(NAME ${name} COMMAND ./check_${name})
endfunction()


add_executable(checkTimers test_timer.cpp)
target_include_directories(checkTimers PUBLIC ${TCLAP_INCLUDE_DIR})
target_link_libraries(checkTimers MATools)

add_test(NAME test_timers_no_timer COMMAND checkTimers -c 0 -r 0)
add_test(NAME test_timers_case_1 COMMAND checkTimers -c 1 -r 0)
add_test(NAME test_timers_case_2 COMMAND checkTimers -c 2 -r 0)
add_test(NAME test_timers_case_3 COMMAND checkTimers -c 3 -r 0)
add_test(NAME test_timers_case_4 COMMAND checkTimers -c 4 -r 0)
add_test(NAME test_timers_case_5 COMMAND checkTimers -c 5 -r 0)
add_test(NAME test_timers_case_1_robustness COMMAND checkTimers -c 1 -r 1)
add_test(NAME test_timers_case_2_robustness COMMAND checkTimers -c 2 -r 1)
add_test(NAME test_timers_case_3_robustness COMMAND checkTimers -c 3 -r 1)
if(MATOOLS_MPI)
	add_test(NAME test_timers_case_1_robustness_4mpi COMMAND mpirun -n 4 ./checkTimers -c 1 -r 1)
	add_test(NAME test_timers_case_2_robustness_4mpi COMMAND mpirun -n 4 ./checkTimers -c 2 -r 1)
	add_test(NAME test_timers_case_3_robustness_4mpi COMMAND mpirun -n 4 ./checkTimers -c 3 -r 1)
endif()

if(OPENMP_FOUND)
	build_test(test_trace_with_omp)
	build_test(test_trace_with_omp_task)
endif()
set (testlist 
	test_minimal_long_name 
	test_timer_matimers_manager
	test_hybrid_timer
	test_hybrid_timer_short
	test_trace 
	test_full_tree_mode 
	test_memory_footprint 
	test_memory_footprint_write 
	test_mamemory_api
	test_matoolsmanager_api
	test_get_filtered_timers 
	test_print_filtered_timers 
	test_print_filtered_timers_unbalance 
	test_mpi_debug_unbalanced_timers
	test_mpi_debug_balanced_timers
  test_memory_footprint_write_per_mpi
)

foreach(elem IN LISTS testlist)
	message("---> add test ${elem}")
	build_test(${elem})
	add_test_mpi(${elem})
endforeach()
