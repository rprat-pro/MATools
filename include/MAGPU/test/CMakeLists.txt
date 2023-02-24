
  find_package(Catch2 REQUIRED)
enable_testing()

  add_executable(MyLibTests test.cpp)
  if (CUDA_OPTION)
set_source_files_properties(test.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(MyLibTests PROPERTIES CUDA_ARCHITECTURES "80" CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CXX)
  target_link_libraries(MyLibTests PRIVATE cuda MAGPU Catch2::Catch2)
  else()
  target_link_libraries(MyLibTests PRIVATE MAGPU Catch2::Catch2)
endif()

target_include_directories(MyLibTests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  include(CTest)
  include(Catch)
catch_discover_tests(MyLibTests)

  add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
      DEPENDS MyLibTests)



  include(../cmake/modules/tfel.cmake)
  function(add_tfel exe)
  if(CUDA_OPTION)
set_source_files_properties(${exe}.cxx PROPERTIES LANGUAGE CUDA)
  set_target_properties(${exe} PROPERTIES CUDA_ARCHITECTURES "80" CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CXX)
target_include_directories(${exe} PUBLIC ${TFEL_INCLUDE_PATH} ${MAGPU_HEADERS})
target_link_directories(${exe} PUBLIC ${TFEL_LIBRARY_PATH})
target_link_libraries(${exe} cuda TFELException MAGPU)
  else()
target_include_directories(${exe} PUBLIC ${TFEL_INCLUDE_PATH} ${MAGPU_HEADERS})
target_link_directories(${exe} PUBLIC ${TFEL_LIBRARY_PATH})
target_link_libraries(${exe} TFELException MAGPU)
  endif()
  endfunction()
add_executable(elasticity elasticity.cxx)

add_tfel(elasticity)

add_test(NAME mfront_elasticity COMMAND elasticity)
