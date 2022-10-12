cmake_minimum_required(VERSION 3.3)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set the project name and version
project(MATimers VERSION 1.0)

option(MATIMERS_TESTING "add tests" OFF)
option(MATIMERS_MPI "add tests" ON)


if(MATIMERS_MPI)
	find_package(MPI REQUIRED)
	add_definitions(-DOMPI_SKIP_MPICXX)
	add_compile_options(-D__MPI)	
endif()

# openmp
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
	set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wall")
	set(CMAKE_CXX_FLAGS_DEBUG   "-g -traceback")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3 -xHOST")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wall")
	set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g3")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
endif()

set(HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx")
file(GLOB_RECURSE HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hxx")

set(MATIMERS  ${SRCS})
message(${MATIMERS})

add_library(MATimers STATIC ${MATIMERS})
target_include_directories(MATimers PUBLIC ${HEADERS})
if(MATIMERS_MPI)
	target_link_libraries(MATimers PRIVATE MPI_C)
	target_include_directories(MATimers PRIVATE ${MPI_C_INCLUDE_PATH}) 

endif()

target_compile_options(MATimers PRIVATE -fPIC)
foreach(header ${HDRS})
	set_property(TARGET MATimers APPEND PROPERTY PUBLIC_HEADER ${header})
endforeach(header)


install(TARGETS MATimers
	DESTINATION lib
	        PUBLIC_HEADER
	DESTINATION include/
)


if(MATIMERS_TESTING)
	enable_testing()
	set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
	find_package(TCLAP REQUIRED)
	add_subdirectory(test)
endif()
