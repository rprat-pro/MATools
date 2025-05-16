install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html
  DESTINATION share/doc/MATools
  COMPONENT doxygen)
install(FILES
  ${PROJECT_SOURCE_DIR}/doc/doxygen/img/background.svg
  DESTINATION share/doc/MATools/html/img/
  COMPONENT doxygen)
