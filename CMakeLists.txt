cmake_minimum_required(VERSION 3.3)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set the project name and version
project(MATools VERSION 1.0)

option(MATOOLS_TESTING "add tests" OFF)
option(MATOOLS_MPI "add tests" ON)
option(MATOOLS_STATIC_LIB "add tests" OFF)
option(MATOOLS_VERBOSITY_LEVEL_1 "add verbosity level 1" OFF)

if(MATOOLS_VERBOSITY_LEVEL_1)
	add_definitions(-DMATIMERS_VEROBSITY_LEVEL_1)
endif()

if(MATOOLS_MPI)
	find_package(MPI REQUIRED)
	if(MPI_FOUND)
		message("-- MPI feature is activated")
		add_definitions(-DOMPI_SKIP_MPICXX)
		add_compile_options(-D__MPI)
		include_directories(SYSTEM ${MPI_INCLUDE_PATH})
	endif()
endif()

# Openmp
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

#add_subdirectory(src)
#add_subdirectory(include)

set(MATOOLS  ${SRCS})


if(MATOOLS_STATIC_LIB)
add_library(MATools STATIC ${MATOOLS})
else()
add_library(MATools SHARED ${MATOOLS})
endif()

target_include_directories(MATools PUBLIC ${HEADERS})

if(MATOOLS_MPI)
	target_link_libraries(MATools PRIVATE mpi)
	target_include_directories(MATools PRIVATE ${MPI_C_INCLUDE_PATH}) 
	target_link_directories(MATools PRIVATE $ENV{MPI_ROOT}/lib) 
endif()


target_compile_options(MATools PRIVATE -fPIC)


install(TARGETS MATools
	DESTINATION lib)

install(DIRECTORY ${HEADERS}/
	DESTINATION include
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/share/cmake/
	DESTINATION .
)

if(MATOOLS_TESTING)
	enable_testing()
	set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
	find_package(TCLAP REQUIRED)
	add_subdirectory(test)
endif()
