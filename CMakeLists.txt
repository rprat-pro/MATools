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

add_custom_target(doc)
add_subdirectory(doc)

option(enable-website "enable generation of the website" OFF)
if(enable-website)
  set(MATOOLS_GENERATE_WEBSITE ON)
  # Looking for pandoc (http://johnmacfarlane.net/pandoc)
  include(cmake/modules/pandoc.cmake)
  if(MATOOLS_HAVE_PANDOC)
    set(MATOOLS_GENERATE_WEBSITE ON)
  endif(MATOOLS_HAVE_PANDOC)
  if(MATOOLS_GENERATE_WEBSITE)
    list(APPEND CPACK_COMPONENTS_ALL website)
    set(CPACK_COMPONENT_WEBSITE_DESCRIPTION
      "Contains the MFEM/MGIS web site")
  endif(MATOOLS_GENERATE_WEBSITE)
else(enable-website)
  set(MATOOLS_GENERATE_WEBSITE OFF)
  message(STATUS "Generation of the MATools website disabled")
endif(enable-website)

# add a target to generate API documentation with Doxygen
option(enable-doxygen-doc "enable generation of the Doxygen documentation" OFF)
if(enable-doxygen-doc)
  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
      ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    add_custom_target(doc-html
      ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen" VERBATIM)
    add_dependencies(doc doc-html)
    list(APPEND CPACK_COMPONENTS_ALL doxygen)
    set(CPACK_COMPONENT_DOXYGEN_DESCRIPTION
      "Contains the TFEL doxygen documentation")
    set(GENERATE_DOXYGEN ON)
  else(DOXYGEN_FOUND)
    message(FATAL_ERROR "doxygen is required")
  endif(DOXYGEN_FOUND)
else(enable-doxygen-doc)
  message(STATUS "doxygen documentation is not enabled")
endif(enable-doxygen-doc)



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
