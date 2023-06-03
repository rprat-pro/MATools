set(MATOOLS_INCLUDE_DIRS $ENV{MATools_DIR}/include)
set(MATOOLS_LIBRARY_DIRS $ENV{MATools_DIR}/lib)

if(DEFINED USE_MATOOLS_STATIC)
	message("-- use static library" )
  set(MATOOLS_LIBRARIES MATools.a)
else()
	message("-- use shared library" )
  set(MATOOLS_LIBRARIES MATools.so)
endif()


# Vérifier si les variables requises sont définies
if (NOT DEFINED MATOOLS_INCLUDE_DIRS OR NOT DEFINED MATOOLS_LIBRARY_DIRS OR NOT DEFINED MATOOLS_LIBRARIES)
    message(FATAL_ERROR "Le fichier MAToolsConfig.cmake doit définir les variables MATOOLS_INCLUDE_DIRS, MATOOLS_LIBRARY_DIRS et MATOOLS_LIBRARIES.")
endif()

# Créer une cible IMPORTED pour MATools
add_library(MATools INTERFACE)

# Définir les chemins d'inclusion et de bibliothèques
target_include_directories(MATools INTERFACE ${MATOOLS_INCLUDE_DIRS})
target_link_directories(MATools INTERFACE ${MATOOLS_LIBRARY_DIRS})

# Lier les bibliothèques de MATools
target_link_libraries(MATools INTERFACE ${MATOOLS_LIBRARIES})

