add_executable(checkTimers test_timer.cpp)
target_include_directories(checkTimers PUBLIC ${TCLAP_INCLUDE_DIR})
target_link_libraries(checkTimers MATools )

add_test(NAME test_timers_no_timer COMMAND checkTimers -c 0 -r 0)
add_test(NAME test_timers_case_1 COMMAND checkTimers -c 1 -r 0)
add_test(NAME test_timers_case_2 COMMAND checkTimers -c 2 -r 0)
add_test(NAME test_timers_case_3 COMMAND checkTimers -c 3 -r 0)
add_test(NAME test_timers_case_4 COMMAND checkTimers -c 4 -r 0)
add_test(NAME test_timers_case_5 COMMAND checkTimers -c 5 -r 0)
add_test(NAME test_timers_case_1_robustness COMMAND checkTimers -c 1 -r 1)
add_test(NAME test_timers_case_2_robustness COMMAND checkTimers -c 2 -r 1)
add_test(NAME test_timers_case_3_robustness COMMAND checkTimers -c 3 -r 1)
add_test(NAME test_timers_case_1_robustness_4mpi COMMAND mpirun -n 4 ./checkTimers -c 1 -r 1)
add_test(NAME test_timers_case_2_robustness_4mpi COMMAND mpirun -n 4 ./checkTimers -c 2 -r 1)
add_test(NAME test_timers_case_3_robustness_4mpi COMMAND mpirun -n 4 ./checkTimers -c 3 -r 1)


add_executable(checkTrace test_trace.cxx)
target_link_libraries(checkTrace MATools)
add_test(NAME test_trace COMMAND checkTrace)
add_test(NAME test_trace_mpi_4 COMMAND mpirun -n 4 ./checkTrace)


add_executable(checkFullTreeMode test_full_tree_mode.cxx)
target_link_libraries(checkFullTreeMode MATools)
add_test(NAME test_full_tree_mode_mpi_4 COMMAND mpirun -n 4 ./checkFullTreeMode)
