



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

  add_executable(ex1 ex1.cxx)
  add_tfel(ex1)
add_test(NAME example1 COMMAND ex1)

