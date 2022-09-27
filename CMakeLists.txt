cmake_minimum_required(VERSION 3.3)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set the project name and version
project(MATimers VERSION 1.0)


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
#file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)
#file(GLOB HDRS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hxx)

#foreach(source ${SRCS})
set(MATIMERS  ${SRCS})#${MATIMERS}) ${source})
	#endforeach(source)
message(${MATIMERS})

add_library(MATimers SHARED ${MATIMERS})
target_include_directories(MATimers PUBLIC ${HEADERS})
#message(${HDRS})
foreach(header ${HDRS})
	message(${header})
	set_target_properties(MATimers PROPERTIES PUBLIC_HEADER ${header})
endforeach(header)


install(TARGETS MATimers
        PUBLIC_HEADER
	DESTINATION include/MATimers
)

enable_testing()
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(TCLAP REQUIRED)
add_subdirectory(test)

