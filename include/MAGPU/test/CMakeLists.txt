
find_package(Catch2 REQUIRED)
enable_testing()

add_executable(MyLibTests test.cpp)
target_link_libraries(MyLibTests PRIVATE MAGPU Catch2::Catch2)
#target_link_libraries(MyLibTests PUBLIC MAGPU Catch2::Catch2WithMain)
#catch_discover_tests(MyLibTests)

include(CTest)
include(Catch)
catch_discover_tests(MyLibTests)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS MyLibTests)

#include(ParseAndAddCatchTests)
#  cmake_policy(SET CMP0110 NEW)
#ParseAndAddCatchTests(MyLibTests)
