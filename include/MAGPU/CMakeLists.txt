cmake_minimum_required(VERSION 3.3)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


option(CUDA_OPTION "active MAGPUVector with cuda" ON)

if (CUDA_OPTION)
	# set the project name and version
	project(MAGPU VERSION 1.0 LANGUAGES CUDA CXX)
	find_package(CUDA)
	enable_language(CUDA)
	add_compile_options(-D__CUDA__)
else()
	# set the project name and version
	project(MAGPU VERSION 1.0)
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

set(MAGPU_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB_RECURSE MAGPU_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx")
file(GLOB_RECURSE MAGPU_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hxx")

cuda_add_library(MAGPU SHARED ${MAGPU_SRCS})
target_include_directories(MAGPU PUBLIC ${MAGPU_HEADERS})

add_executable(magpu main.cpp)

if (CUDA_OPTION)
target_link_libraries(MAGPU cuda)
set_source_files_properties(${MAGPU_SRCS} PROPERTIES LANGUAGE CUDA)
set_target_properties(MAGPU PROPERTIES CUDA_ARCHITECTURES "80" CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CXX)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(magpu PROPERTIES CUDA_ARCHITECTURES "80" CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CXX)
target_link_libraries(magpu cuda MAGPU)
else()
target_link_libraries(magpu PUBLIC MAGPU)
endif()




add_subdirectory(test)
