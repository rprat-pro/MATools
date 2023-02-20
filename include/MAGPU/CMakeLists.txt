cmake_minimum_required(VERSION 3.3)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


option(CUDA_OPTION "active MAGPUVector with cuda" ON)
option(KOKKOS_OPTION "active MAGPUVector with kokkos" OFF)

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
    add_compile_options(-fopenmp)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp")
    message(" cmake linker ${CMAKE_EXE_LINKER_FLAGS}")
endif()


set(MAGPU_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB_RECURSE MAGPU_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx")
file(GLOB_RECURSE MAGPU_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hxx")

if (CUDA_OPTION)
cuda_add_library(MAGPU SHARED ${MAGPU_SRCS})
else()
add_library(MAGPU SHARED ${MAGPU_SRCS})
endif()
target_include_directories(MAGPU PUBLIC ${MAGPU_HEADERS})

add_executable(magpu main.cpp)

if (CUDA_OPTION)
set(CMAKE_CUDA_FLAGS "--extended-lambda --expt-relaxed-constexpr -fPIC -rdc=true -dlink -std=c++17 -gencode arch=compute_80,code=sm_80 --relocatable-device-code=true")
target_link_libraries(MAGPU cuda)
set_source_files_properties(${MAGPU_SRCS} PROPERTIES LANGUAGE CUDA)
set_target_properties(MAGPU PROPERTIES CUDA_ARCHITECTURES "80" CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CXX)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(magpu PROPERTIES CUDA_ARCHITECTURES "80" CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(magpu PUBLIC ${MAGPU_HEADERS})
target_link_libraries(magpu cuda MAGPU)
else()
target_link_libraries(magpu PUBLIC MAGPU)
endif()




add_subdirectory(test)
